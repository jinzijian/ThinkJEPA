# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

"""Reusable OmniJEPA building blocks.

This module intentionally stays backbone-agnostic. A concrete Qwen/InternVL/VLA
integration can call these pieces from the point where mid-layer VLM hidden
states are available, then route text to the normal LM head and actions to the
flow expert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


MODE_QA = "qa"
MODE_PLAN = "plan"
MODE_ACT = "act"
MODE_CHAT = "chat"
MODE_TOKENS = {
    "<MODE=QA>": MODE_QA,
    "<MODE=PLAN>": MODE_PLAN,
    "<MODE=ACT>": MODE_ACT,
    "<MODE=CHAT>": MODE_CHAT,
}
TEXT_MODES = {MODE_QA, MODE_PLAN, MODE_CHAT}
ACTION_MODES = {MODE_ACT}
REQUIRED_CACHE_FIELDS = {
    "episode_id",
    "timestep",
    "mode",
    "instruction",
    "obs_frames",
    "current_jepa_latent",
    "predicted_future_jepa_latent",
    "oracle_future_jepa_latent",
}


def normalize_omnijepa_mode(mode: str) -> str:
    """Normalize explicit OmniJEPA mode tokens to compact mode names."""

    if mode is None:
        raise ValueError("OmniJEPA mode cannot be None")
    text = str(mode).strip()
    if text in MODE_TOKENS:
        return MODE_TOKENS[text]
    text = text.lower().strip("<> ")
    if text.startswith("mode="):
        text = text[len("mode=") :]
    if text in TEXT_MODES or text in ACTION_MODES:
        return text
    valid = ", ".join(sorted(set(MODE_TOKENS) | TEXT_MODES | ACTION_MODES))
    raise ValueError(f"unknown OmniJEPA mode {mode!r}; expected one of {valid}")


def route_omnijepa_output(mode: str) -> str:
    """Return the output branch selected by an explicit mode token."""

    mode = normalize_omnijepa_mode(mode)
    if mode in ACTION_MODES:
        return "action"
    if mode in TEXT_MODES:
        return "text"
    raise AssertionError(f"unhandled OmniJEPA mode {mode!r}")


@dataclass(frozen=True)
class OmniJepaConfig:
    vlm_dim: int
    jepa_dim: int
    action_dim: int = 7
    action_horizon: int = 16
    future_token_count: int = 16
    guidance_layers: int = 12
    guidance_tokens: int = 16
    guidance_dim: int = 3584
    hidden_dim: int = 1024
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class OmniJepaCacheRecord:
    """Schema for offline OmniJEPA training caches.

    Tensor payloads can be local paths, memory-mapped handles, or already-loaded
    tensors; this dataclass only validates the logical keys that downstream
    loaders should provide.
    """

    episode_id: str
    timestep: int
    mode: str
    instruction: str
    obs_frames: Any
    current_jepa_latent: Any
    predicted_future_jepa_latent: Any
    oracle_future_jepa_latent: Any
    future_frames: Any | None = None
    target_text: str | None = None
    target_action_chunk: Any | None = None
    target_traj_tokens: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.mode = normalize_omnijepa_mode(self.mode)
        if self.mode in TEXT_MODES and self.target_text is None:
            raise ValueError(f"mode={self.mode!r} requires target_text")
        if self.mode in ACTION_MODES and self.target_action_chunk is None:
            raise ValueError("mode='act' requires target_action_chunk")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OmniJepaCacheRecord":
        missing = REQUIRED_CACHE_FIELDS.difference(payload.keys())
        if missing:
            raise KeyError(f"OmniJEPA cache record missing fields: {sorted(missing)}")
        return cls(**dict(payload))


def masked_mean_tokens(x: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Mean-pool sequence tokens with an optional boolean mask."""

    if mask is None:
        return x.mean(dim=1)
    mask = mask.to(device=x.device, dtype=torch.bool)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    weight = mask.to(dtype=x.dtype)
    return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(eps)


class VlmToJepaTaskAdapter(nn.Module):
    """Convert mid-layer VLM states into JEPA predictor guidance streams."""

    def __init__(
        self,
        vlm_dim: int,
        guidance_dim: int = 3584,
        num_layers: int = 12,
        num_tokens: int = 16,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vlm_dim = int(vlm_dim)
        self.guidance_dim = int(guidance_dim)
        self.num_layers = int(num_layers)
        self.num_tokens = int(num_tokens)
        if self.num_layers <= 0 or self.num_tokens <= 0:
            raise ValueError("num_layers and num_tokens must be positive")

        self.summary = nn.Sequential(
            nn.LayerNorm(self.vlm_dim),
            nn.Linear(self.vlm_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_queries = nn.Parameter(torch.empty(self.num_layers, self.num_tokens, hidden_dim))
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, guidance_dim),
        )
        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.token_queries, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, vlm_hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if vlm_hidden.dim() != 3:
            raise ValueError(f"vlm_hidden must be [B,N,D], got {tuple(vlm_hidden.shape)}")
        pooled = masked_mean_tokens(vlm_hidden, attention_mask)
        summary = self.summary(pooled)
        tokens = summary[:, None, None, :] + self.token_queries[None, :, :, :].to(
            device=summary.device, dtype=summary.dtype
        )
        stream = self.readout(tokens)
        mask = torch.ones(
            stream.shape[:3],
            dtype=torch.bool,
            device=stream.device,
        )
        # Reuse ThinkJEPA's old/new guidance contract. Downstream policies can
        # ablate either stream without changing the predictor interface.
        return {
            "vlm_old": stream,
            "vlm_new": stream,
            "vlm_old_mask": mask,
            "vlm_new_mask": mask,
        }


class JepaFutureTokenResampler(nn.Module):
    """Compress [B,T,P,D] future JEPA latents into K VLM-space tokens."""

    def __init__(
        self,
        jepa_dim: int,
        vlm_dim: int,
        num_tokens: int = 16,
        hidden_dim: int | None = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.jepa_dim = int(jepa_dim)
        self.vlm_dim = int(vlm_dim)
        self.num_tokens = int(num_tokens)
        hidden_dim = int(hidden_dim or vlm_dim)
        self.input_norm = nn.LayerNorm(self.jepa_dim)
        self.input_proj = nn.Linear(self.jepa_dim, hidden_dim)
        self.queries = nn.Parameter(torch.empty(1, self.num_tokens, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.vlm_dim),
        )
        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.queries, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, future_latent: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if future_latent.dim() != 4:
            raise ValueError(f"future_latent must be [B,T,P,D], got {tuple(future_latent.shape)}")
        B, T, P, D = future_latent.shape
        if D != self.jepa_dim:
            raise ValueError(f"future_latent dim {D} != jepa_dim {self.jepa_dim}")
        memory = self.input_proj(self.input_norm(future_latent).reshape(B, T * P, D))
        key_padding_mask = None
        if token_mask is not None:
            if token_mask.shape[:3] != (B, T, P):
                raise ValueError(
                    f"token_mask must start with {(B, T, P)}, got {tuple(token_mask.shape)}"
                )
            key_padding_mask = ~token_mask.reshape(B, T * P).to(dtype=torch.bool, device=future_latent.device)
        queries = self.queries.to(device=future_latent.device, dtype=memory.dtype).expand(B, -1, -1)
        tokens, _ = self.cross_attn(
            queries,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.out(tokens).to(dtype=future_latent.dtype)


class JepaLateFusionAdapter(nn.Module):
    """Late-layer VLM adapter that cross-attends to future JEPA tokens."""

    def __init__(self, vlm_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(vlm_dim)
        self.norm_kv = nn.LayerNorm(vlm_dim)
        self.cross_attn = nn.MultiheadAttention(
            vlm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        vlm_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
        future_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if vlm_hidden.dim() != 3 or future_tokens.dim() != 3:
            raise ValueError("vlm_hidden and future_tokens must both be [B,N,D]")
        key_padding_mask = None
        if future_mask is not None:
            key_padding_mask = ~future_mask.to(device=future_tokens.device, dtype=torch.bool)
        attn_out, _ = self.cross_attn(
            self.norm_q(vlm_hidden),
            self.norm_kv(future_tokens),
            self.norm_kv(future_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return vlm_hidden + torch.tanh(self.gate).to(dtype=attn_out.dtype) * attn_out


class FlowActionExpert(nn.Module):
    """Flow-matching action expert conditioned on VLM and JEPA future tokens."""

    def __init__(
        self,
        vlm_dim: int,
        action_dim: int = 7,
        action_horizon: int = 16,
        future_dim: int | None = None,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vlm_dim = int(vlm_dim)
        self.future_dim = int(future_dim or vlm_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_dim = int(hidden_dim)
        self.action_in = nn.Linear(self.action_dim, hidden_dim)
        self.time_in = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vlm_proj = nn.Linear(self.vlm_dim, hidden_dim)
        self.future_proj = nn.Linear(self.future_dim, hidden_dim)
        self.action_pos = nn.Parameter(torch.empty(1, self.action_horizon, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(depth))
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, self.action_dim)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.action_pos, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_memory(
        self,
        vlm_hidden: torch.Tensor,
        future_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if vlm_hidden.dim() != 3:
            raise ValueError(f"vlm_hidden must be [B,N,D], got {tuple(vlm_hidden.shape)}")
        memory_parts = [self.vlm_proj(vlm_hidden)]
        if future_tokens is not None:
            if future_tokens.dim() != 3:
                raise ValueError(f"future_tokens must be [B,K,D], got {tuple(future_tokens.shape)}")
            memory_parts.append(self.future_proj(future_tokens))
        return torch.cat(memory_parts, dim=1)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
        vlm_hidden: torch.Tensor,
        future_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_actions.dim() != 3:
            raise ValueError(f"noisy_actions must be [B,H,A], got {tuple(noisy_actions.shape)}")
        B, H, A = noisy_actions.shape
        if H > self.action_horizon:
            raise ValueError(f"action horizon {H} exceeds configured {self.action_horizon}")
        if A != self.action_dim:
            raise ValueError(f"action dim {A} != configured {self.action_dim}")
        if flow_time.dim() == 1:
            flow_time = flow_time[:, None]
        if flow_time.shape != (B, 1):
            raise ValueError(f"flow_time must be [B] or [B,1], got {tuple(flow_time.shape)}")
        memory = self._build_memory(vlm_hidden, future_tokens)
        x = self.action_in(noisy_actions)
        x = x + self.action_pos[:, :H, :].to(device=x.device, dtype=x.dtype)
        x = x + self.time_in(flow_time.to(device=x.device, dtype=x.dtype)).unsqueeze(1)
        x = self.decoder(tgt=x, memory=memory)
        return self.out(self.out_norm(x))

    def flow_matching_loss(
        self,
        target_actions: torch.Tensor,
        vlm_hidden: torch.Tensor,
        future_tokens: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B = target_actions.shape[0]
        if noise is None:
            noise = torch.randn_like(target_actions)
        if flow_time is None:
            flow_time = torch.rand(B, device=target_actions.device, dtype=target_actions.dtype)
        t = flow_time.view(B, 1, 1)
        noisy = (1.0 - t) * noise + t * target_actions
        target_velocity = target_actions - noise
        pred_velocity = self(noisy, flow_time, vlm_hidden, future_tokens)
        loss = F.mse_loss(pred_velocity, target_velocity)
        metrics = {
            "flow_loss": loss.detach(),
            "action_l1": F.l1_loss(pred_velocity, target_velocity).detach(),
        }
        return loss, metrics


class OmniJepaBridge(nn.Module):
    """Backbone-agnostic single-pass OmniJEPA bridge.

    The caller owns the concrete VLM blocks and JEPA predictor. This module
    provides the trainable interfaces around them.
    """

    def __init__(self, config: OmniJepaConfig):
        super().__init__()
        self.config = config
        self.vlm_to_jepa = VlmToJepaTaskAdapter(
            vlm_dim=config.vlm_dim,
            guidance_dim=config.guidance_dim,
            num_layers=config.guidance_layers,
            num_tokens=config.guidance_tokens,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.future_resampler = JepaFutureTokenResampler(
            jepa_dim=config.jepa_dim,
            vlm_dim=config.vlm_dim,
            num_tokens=config.future_token_count,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.late_fusion = JepaLateFusionAdapter(
            vlm_dim=config.vlm_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.action_expert = FlowActionExpert(
            vlm_dim=config.vlm_dim,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            future_dim=config.vlm_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

    def build_jepa_guidance(
        self,
        mid_vlm_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.vlm_to_jepa(mid_vlm_hidden, attention_mask)

    def build_future_tokens(self, predicted_future_latent: torch.Tensor) -> torch.Tensor:
        return self.future_resampler(predicted_future_latent)

    def fuse_text_hidden(
        self,
        late_vlm_hidden: torch.Tensor,
        predicted_future_latent: torch.Tensor,
    ) -> torch.Tensor:
        future_tokens = self.build_future_tokens(predicted_future_latent)
        return self.late_fusion(late_vlm_hidden, future_tokens)

    def action_loss(
        self,
        target_actions: torch.Tensor,
        vlm_hidden: torch.Tensor,
        predicted_future_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        future_tokens = self.build_future_tokens(predicted_future_latent)
        return self.action_expert.flow_matching_loss(
            target_actions,
            vlm_hidden,
            future_tokens,
        )
