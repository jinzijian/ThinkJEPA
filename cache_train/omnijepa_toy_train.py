# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

"""Tiny end-to-end OmniJEPA train/eval smoke run.

This is deliberately synthetic. It validates the full data path:

cache -> explicit mode -> VLM-to-JEPA guidance -> future latent prediction
-> JEPA-to-action conditioning -> flow action loss -> eval metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from cache_train.omnijepa import OmniJepaBridge, OmniJepaConfig
from cache_train.omnijepa_data import OmniJepaNpzDataset, write_omnijepa_cache_npz


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ToyGuidedJepaPredictor(nn.Module):
    """Small stand-in for a VLM-guided JEPA predictor."""

    def __init__(
        self,
        jepa_dim: int,
        guidance_dim: int,
        future_frames: int,
        patches: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.jepa_dim = int(jepa_dim)
        self.guidance_dim = int(guidance_dim)
        self.future_frames = int(future_frames)
        self.patches = int(patches)
        self.current_proj = nn.Linear(self.jepa_dim, hidden_dim)
        self.guidance_proj = nn.Linear(self.guidance_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.time_embed = nn.Parameter(torch.empty(1, self.future_frames, 1, hidden_dim))
        self.patch_embed = nn.Parameter(torch.empty(1, 1, self.patches, hidden_dim))
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.jepa_dim),
        )
        self._init_parameters()

    def _init_parameters(self):
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
    def _guidance_summary(guidance: dict[str, torch.Tensor]) -> torch.Tensor:
        old = guidance["vlm_old"].mean(dim=(1, 2))
        new = guidance["vlm_new"].mean(dim=(1, 2))
        return 0.5 * (old + new)

    def forward(self, current_latent: torch.Tensor, guidance: dict[str, torch.Tensor]) -> torch.Tensor:
        if current_latent.dim() != 4:
            raise ValueError(f"current_latent must be [B,T,P,D], got {tuple(current_latent.shape)}")
        B, _, P, D = current_latent.shape
        if P != self.patches or D != self.jepa_dim:
            raise ValueError(
                f"current_latent shape mismatch: patches={P}/{self.patches}, dim={D}/{self.jepa_dim}"
            )
        current = self.current_proj(current_latent.mean(dim=(1, 2)))
        guide = self.guidance_proj(self._guidance_summary(guidance))
        fused = self.fusion(torch.cat([current, guide, (current - guide).abs(), current * guide], dim=-1))
        hidden = (
            fused[:, None, None, :]
            + self.time_embed.to(device=fused.device, dtype=fused.dtype)
            + self.patch_embed.to(device=fused.device, dtype=fused.dtype)
        )
        return self.out(hidden)


def synthesize_vlm_hidden(current_latent: torch.Tensor, token_count: int, vlm_dim: int) -> torch.Tensor:
    """Deterministic toy VLM hidden states from current JEPA latents."""

    summary = current_latent.mean(dim=(1, 2))
    repeats = int(math.ceil(vlm_dim / summary.size(-1)))
    base = summary.repeat(1, repeats)[:, :vlm_dim]
    positions = torch.linspace(
        -1.0,
        1.0,
        token_count,
        device=current_latent.device,
        dtype=current_latent.dtype,
    )
    return base[:, None, :] + 0.05 * positions[None, :, None]


def make_action_from_future(future: np.ndarray, action_weights: np.ndarray, action_bias: np.ndarray) -> np.ndarray:
    summary = future.mean(axis=(0, 1))
    actions = np.tanh(np.einsum("d,had->ha", summary, action_weights) + action_bias)
    return actions.astype(np.float32)


def generate_toy_cache(args) -> tuple[Path, Path]:
    cache_root = Path(args.cache_root)
    train_dir = cache_root / "train"
    test_dir = cache_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    action_weights = rng.normal(
        scale=0.45,
        size=(args.action_horizon, args.action_dim, args.jepa_dim),
    ).astype(np.float32)
    action_bias = rng.normal(
        scale=0.05,
        size=(args.action_horizon, args.action_dim),
    ).astype(np.float32)

    def _write_split(split_dir: Path, count: int, offset: int):
        manifest_lines = []
        for i in range(count):
            idx = offset + i
            base = rng.normal(size=(args.patches, args.jepa_dim)).astype(np.float32)
            velocity = rng.normal(scale=0.15, size=(args.patches, args.jepa_dim)).astype(np.float32)
            current = []
            for t in range(args.context_frames):
                current.append(base - (args.context_frames - t) * velocity + rng.normal(scale=0.02, size=base.shape))
            current = np.stack(current, axis=0).astype(np.float32)
            future = []
            for t in range(args.future_frames):
                future.append(base + (t + 1) * velocity + rng.normal(scale=0.02, size=base.shape))
            oracle_future = np.stack(future, axis=0).astype(np.float32)
            predicted_future = (oracle_future + rng.normal(scale=0.12, size=oracle_future.shape)).astype(np.float32)
            action = make_action_from_future(oracle_future, action_weights, action_bias)
            path = split_dir / f"sample_{idx:05d}.npz"
            write_omnijepa_cache_npz(
                path,
                episode_id=f"toy_ep_{idx:05d}",
                timestep=idx,
                mode="<MODE=ACT>",
                instruction=f"toy instruction {idx}",
                obs_frames=np.zeros((args.context_frames, 16, 16, 3), dtype=np.uint8),
                current_jepa_latent=current,
                predicted_future_jepa_latent=predicted_future,
                oracle_future_jepa_latent=oracle_future,
                target_action_chunk=action,
            )
            manifest_lines.append(str(path))
        return manifest_lines

    train_manifest = cache_root / "train_manifest.txt"
    test_manifest = cache_root / "test_manifest.txt"
    train_manifest.write_text("\n".join(_write_split(train_dir, args.num_train, 0)) + "\n", encoding="utf-8")
    test_manifest.write_text(
        "\n".join(_write_split(test_dir, args.num_test, args.num_train)) + "\n",
        encoding="utf-8",
    )
    return train_manifest, test_manifest


def collate_toy_batch(samples):
    tensor_keys = [
        "current_jepa_latent",
        "predicted_future_jepa_latent",
        "oracle_future_jepa_latent",
        "target_action_chunk",
    ]
    batch = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in tensor_keys}
    batch["mode"] = [sample["mode"] for sample in samples]
    batch["instruction"] = [sample["instruction"] for sample in samples]
    batch["path"] = [sample["path"] for sample in samples]
    return batch


def build_models(args, device):
    config = OmniJepaConfig(
        vlm_dim=args.vlm_dim,
        jepa_dim=args.jepa_dim,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        future_token_count=args.future_token_count,
        guidance_layers=args.guidance_layers,
        guidance_tokens=args.guidance_tokens,
        guidance_dim=args.guidance_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    bridge = OmniJepaBridge(config).to(device)
    predictor = ToyGuidedJepaPredictor(
        jepa_dim=args.jepa_dim,
        guidance_dim=args.guidance_dim,
        future_frames=args.future_frames,
        patches=args.patches,
        hidden_dim=args.hidden_dim,
    ).to(device)
    return bridge, predictor


def run_epoch(args, bridge, predictor, loader, optimizer, device, train: bool):
    bridge.train(mode=train)
    predictor.train(mode=train)
    totals = {
        "loss": 0.0,
        "latent_loss": 0.0,
        "flow_loss": 0.0,
        "action_l1": 0.0,
    }
    count = 0
    for batch in loader:
        current = batch["current_jepa_latent"].to(device)
        oracle = batch["oracle_future_jepa_latent"].to(device)
        actions = batch["target_action_chunk"].to(device)
        vlm_hidden = synthesize_vlm_hidden(current, args.vlm_tokens, args.vlm_dim)
        with torch.set_grad_enabled(train):
            guidance = bridge.build_jepa_guidance(vlm_hidden)
            future_pred = predictor(current, guidance)
            future_tokens = bridge.build_future_tokens(future_pred)
            noise = torch.zeros_like(actions)
            flow_time = torch.full(
                (actions.size(0),),
                0.5,
                device=device,
                dtype=actions.dtype,
            )
            flow_loss, action_metrics = bridge.action_expert.flow_matching_loss(
                actions,
                vlm_hidden,
                future_tokens,
                noise=noise,
                flow_time=flow_time,
            )
            latent_loss = F.mse_loss(future_pred, oracle)
            loss = latent_loss + args.lambda_action * flow_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(bridge.parameters()) + list(predictor.parameters()), args.grad_clip)
                optimizer.step()
        B = int(current.size(0))
        totals["loss"] += float(loss.detach().cpu()) * B
        totals["latent_loss"] += float(latent_loss.detach().cpu()) * B
        totals["flow_loss"] += float(flow_loss.detach().cpu()) * B
        totals["action_l1"] += float(action_metrics["action_l1"].detach().cpu()) * B
        count += B
    return {key: value / max(count, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser("OmniJEPA toy train/eval")
    parser.add_argument("--cache_root", type=str, default="outputs/omnijepa_toy_cache")
    parser.add_argument("--output_dir", type=str, default="outputs/omnijepa_toy_run")
    parser.add_argument("--generate_toy_data", action="store_true")
    parser.add_argument("--num_train", type=int, default=24)
    parser.add_argument("--num_test", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda_action", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--context_frames", type=int, default=2)
    parser.add_argument("--future_frames", type=int, default=3)
    parser.add_argument("--patches", type=int, default=6)
    parser.add_argument("--jepa_dim", type=int, default=16)
    parser.add_argument("--vlm_dim", type=int, default=32)
    parser.add_argument("--vlm_tokens", type=int, default=8)
    parser.add_argument("--guidance_dim", type=int, default=24)
    parser.add_argument("--guidance_layers", type=int, default=2)
    parser.add_argument("--guidance_tokens", type=int, default=4)
    parser.add_argument("--future_token_count", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--action_horizon", type=int, default=4)
    parser.add_argument("--action_dim", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.generate_toy_data:
        train_manifest, test_manifest = generate_toy_cache(args)
    else:
        train_manifest = Path(args.cache_root) / "train_manifest.txt"
        test_manifest = Path(args.cache_root) / "test_manifest.txt"
    train_ds = OmniJepaNpzDataset.from_manifest(train_manifest)
    test_ds = OmniJepaNpzDataset.from_manifest(test_manifest)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_toy_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_toy_batch,
    )

    bridge, predictor = build_models(args, device)
    optimizer = torch.optim.AdamW(
        list(bridge.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    logs = {
        "config": vars(args),
        "device": str(device),
        "epochs": [],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] device={device} train={len(train_ds)} test={len(test_ds)}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(args, bridge, predictor, train_loader, optimizer, device, train=True)
        with torch.no_grad():
            test_metrics = run_epoch(args, bridge, predictor, test_loader, optimizer, device, train=False)
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "test": test_metrics,
        }
        logs["epochs"].append(row)
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} latent={train_metrics['latent_loss']:.4f} "
            f"flow={train_metrics['flow_loss']:.4f} action_l1={train_metrics['action_l1']:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} latent={test_metrics['latent_loss']:.4f} "
            f"flow={test_metrics['flow_loss']:.4f} action_l1={test_metrics['action_l1']:.4f}",
            flush=True,
        )

    metrics_path = output_dir / "metrics.json"
    ckpt_path = output_dir / "omnijepa_toy.pt"
    metrics_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
    torch.save(
        {
            "bridge": bridge.state_dict(),
            "predictor": predictor.state_dict(),
            "config": vars(args),
        },
        ckpt_path,
    )
    print(f"[INFO] wrote {metrics_path}")
    print(f"[INFO] wrote {ckpt_path}")


if __name__ == "__main__":
    main()
