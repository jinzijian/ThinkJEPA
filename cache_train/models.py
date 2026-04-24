# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _run_attention_in_module_dtype(attn_module, query, key, value):
    target_dtype = getattr(attn_module.in_proj_weight, "dtype", query.dtype)
    output_dtype = query.dtype
    if query.dtype != target_dtype:
        query = query.to(dtype=target_dtype)
    if key.dtype != target_dtype:
        key = key.to(dtype=target_dtype)
    if value.dtype != target_dtype:
        value = value.to(dtype=target_dtype)
    attn_out, attn_weights = attn_module(query, key, value)
    if attn_out.dtype != output_dtype:
        attn_out = attn_out.to(dtype=output_dtype)
    return attn_out, attn_weights


class TrajectoryReadoutMLP(nn.Module):
    def __init__(
        self,
        d=1024,
        n_tokens=128,
        out_dims=52 * 3,
        mlp_hidden=1024,
        p=0.1,
        use_attn_pool=True,
        nhead=8,
        downsample=False,
    ):
        super().__init__()
        self.d = d
        self.n_tokens = n_tokens
        self.out_dims = out_dims
        self.use_attn_pool = use_attn_pool

        if use_attn_pool:
            self.token_query = nn.Parameter(torch.randn(1, 1, d))
            self.token_attn = nn.MultiheadAttention(
                d, nhead, batch_first=True, dropout=p
            )
        else:
            self.token_query = None
            self.token_attn = None

        self.ln_in = nn.LayerNorm(d)
        self.mlp_in = nn.Sequential(
            nn.Linear(d, mlp_hidden),
            nn.GELU(),
            nn.Dropout(p),
        )

        self.temporal_mlp = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(p),
        )

        self.downsample = downsample
        if downsample:
            # Average-pool over time with kernel=2, stride=2 => T -> floor(T/2)
            self.time_pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.ln_out = nn.LayerNorm(mlp_hidden)
        self.fc_out = nn.Linear(mlp_hidden, out_dims)

        self._initialize_readout_parameters()

    def _initialize_readout_parameters(self):
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _pool_temporal_tokens(self, x):
        if not self.use_attn_pool:
            return x.mean(dim=2)

        B, T, N, d = x.shape
        x = x.view(B * T, N, d)
        q = self.token_query.expand(B * T, -1, -1)
        y, _ = _run_attention_in_module_dtype(self.token_attn, q, x, x)
        y = y.squeeze(1)
        y = y.view(B, T, d)
        return y

    def forward(self, x):
        if x.ndim == 3:
            B, T, d = x.shape
            x = x.view(B, T, -1, d)
        else:
            B, T, _, _ = x.shape

        x = self._pool_temporal_tokens(x)
        x = self.ln_in(x)
        x = self.mlp_in(x)
        H = x.size(-1)

        x_flat = x.reshape(B * T, H)
        y = self.temporal_mlp(x_flat).view(B, T, H)
        x = x + y

        if self.downsample:
            # Downsample over time: convert to [B, H, T], pool, then return to [B, T', H]
            x = x.transpose(1, 2)  # [B, H, T]
            x = self.time_pool(x)  # [B, H, T']
            x = x.transpose(1, 2).contiguous()  # [B, T', H]

        x = self.ln_out(x)
        out = self.fc_out(x)
        return out


class VlmLatentPredictionTower(nn.Module):
    """Trainable VLM-side tower that predicts JEPA latent tokens from VLM cache features."""

    def __init__(
        self,
        vlm_old_dim=3584,
        vlm_new_dim=3584,
        latent_dim=1024,
        hidden_dim=384,
        max_frames=128,
        max_patches=4096,
        dropout=0.1,
    ):
        super().__init__()
        self.vlm_old_dim = int(vlm_old_dim)
        self.vlm_new_dim = int(vlm_new_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_frames = int(max_frames)
        self.max_patches = int(max_patches)

        self.old_adapter = nn.Linear(self.vlm_old_dim, hidden_dim, bias=False)
        self.new_adapter = nn.Linear(self.vlm_new_dim, hidden_dim, bias=False)
        self.context_adapter = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.time_embed = nn.Parameter(torch.zeros(1, self.max_frames, 1, hidden_dim))
        self.patch_embed = nn.Parameter(torch.zeros(1, 1, self.max_patches, hidden_dim))
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, latent_dim),
        )
        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _as_batched_stream(x):
        if x is None:
            return None
        if x.dim() == 3:
            return x.unsqueeze(0)
        if x.dim() == 4:
            return x
        raise ValueError(f"expected VLM stream [L,S,D] or [B,L,S,D], got {tuple(x.shape)}")

    @staticmethod
    def _as_batched_mask(mask, batch_size, layer_count, seq_len, device):
        if mask is None:
            return None
        if mask.dim() == 4:
            mask = mask.any(dim=-1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            raise ValueError(f"expected VLM mask [L,S] or [B,L,S], got {tuple(mask.shape)}")
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.size(0) == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        if (
            mask.size(0) != batch_size
            or mask.size(1) != layer_count
            or mask.size(2) != seq_len
        ):
            aligned = torch.zeros(
                (batch_size, layer_count, seq_len),
                dtype=torch.bool,
                device=device,
            )
            b = min(batch_size, int(mask.size(0)))
            l = min(layer_count, int(mask.size(1)))
            s = min(seq_len, int(mask.size(2)))
            aligned[:b, :l, :s] = mask[:b, :l, :s]
            return aligned
        return mask

    @staticmethod
    def _masked_mean(x, mask=None, eps=1e-6):
        if mask is None:
            return x.mean(dim=(1, 2))
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        weight = mask.to(dtype=x.dtype)
        numer = (x * weight).sum(dim=(1, 2))
        denom = weight.sum(dim=(1, 2)).clamp_min(eps)
        return numer / denom

    def _project_stream(self, stream, mask, adapter, batch_size, device):
        if stream is None:
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        stream = self._as_batched_stream(stream).to(device=device, dtype=adapter.weight.dtype)
        if stream.size(0) == 1 and batch_size > 1:
            stream = stream.expand(batch_size, -1, -1, -1)
        if stream.size(0) != batch_size:
            raise ValueError(
                f"VLM stream batch={stream.size(0)} does not match target batch={batch_size}"
            )
        if stream.size(-1) != adapter.in_features:
            raise ValueError(
                f"VLM feature dim {stream.size(-1)} does not match tower input dim {adapter.in_features}"
            )
        token_mask = self._as_batched_mask(
            mask,
            batch_size=batch_size,
            layer_count=stream.size(1),
            seq_len=stream.size(2),
            device=device,
        )
        projected = adapter(stream)
        return self._masked_mean(projected, token_mask)

    def forward(self, guidance_payload, context_feats, target_shape):
        """
        Args:
            guidance_payload: dict with batched vlm_old/vlm_new streams and masks.
            context_feats: [B,Tctx,P,D] JEPA context tokens.
            target_shape: tuple/list (B,Tfuture,P,D).
        """
        if guidance_payload is None:
            return None
        B, T, P, D = [int(v) for v in target_shape]
        if D != self.latent_dim:
            raise ValueError(f"target latent dim {D} does not match tower latent dim {self.latent_dim}")
        if T > self.max_frames:
            raise ValueError(f"target frames {T} exceed max_frames={self.max_frames}")
        if P > self.max_patches:
            raise ValueError(f"target patches {P} exceed max_patches={self.max_patches}")

        device = context_feats.device
        dtype = self.context_adapter.weight.dtype
        old_stream = guidance_payload.get("vlm_old")
        new_stream = guidance_payload.get("vlm_new")
        if old_stream is None and new_stream is None:
            return None

        old_summary = self._project_stream(
            old_stream,
            guidance_payload.get("vlm_old_mask"),
            self.old_adapter,
            B,
            device,
        )
        new_summary = self._project_stream(
            new_stream,
            guidance_payload.get("vlm_new_mask"),
            self.new_adapter,
            B,
            device,
        )
        context_summary = self.context_adapter(
            context_feats.to(device=device, dtype=dtype).mean(dim=(1, 2))
        )
        fused = self.fusion(
            torch.cat(
                [
                    old_summary,
                    new_summary,
                    (old_summary - new_summary).abs(),
                    old_summary * new_summary,
                    context_summary,
                ],
                dim=-1,
            )
        )
        hidden = (
            fused.view(B, 1, 1, self.hidden_dim)
            + self.time_embed[:, :T, :, :].to(device=device, dtype=fused.dtype)
            + self.patch_embed[:, :, :P, :].to(device=device, dtype=fused.dtype)
        )
        return self.readout(hidden).to(dtype=context_feats.dtype)
