#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import pickle
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import ebnerd_event_set_predictor as ep
import ebnerd_qwen_uih_text_jepa as q


class SampleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "Sample":
            return q.Sample
        return super().find_class(module, name)


def load_samples(path: Path) -> tuple[list[q.Sample], dict[str, Any]]:
    with path.open("rb") as f:
        packed = SampleUnpickler(f).load()
    return packed["samples"], packed.get("prepare_summary", {})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def official_hstu_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    try:
        import generative_recommenders  # type: ignore

        out["generative_recommenders"] = getattr(generative_recommenders, "__file__", "")
        for module in [
            "generative_recommenders.modules.hstu",
            "generative_recommenders.modules.stu",
            "fbgemm_gpu",
            "torchrec",
        ]:
            try:
                __import__(module)
                out[module] = "ok"
            except Exception as exc:  # pragma: no cover - depends on remote env
                out[module] = f"{type(exc).__name__}: {str(exc)[:240]}"
        out["available"] = any(out.get(m) == "ok" for m in ["generative_recommenders.modules.hstu", "generative_recommenders.modules.stu"])
    except Exception as exc:  # pragma: no cover - depends on remote env
        out["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    return out


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1).float()).sum(dim=1) / denom


def last_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    idx = mask.long().sum(dim=1).sub(1).clamp_min(0)
    return x[torch.arange(x.shape[0], device=x.device), idx]


RICH_REACTION_KINDS = (
    "future_clicked",
    "future_not_clicked_exposure",
    "current_clicked",
    "current_not_clicked_exposure",
)
RICH_SCALAR_DIM = 5


def horizon_thresholds(args: argparse.Namespace) -> list[float]:
    return [float(x) for x in str(args.horizon_bucket_hours).split(",") if x.strip()]


def rich_state_dim(args: argparse.Namespace) -> int:
    return len(RICH_REACTION_KINDS) + RICH_SCALAR_DIM + len(horizon_thresholds(args)) + 1


def encode_rich_event_state(event: Any, args: argparse.Namespace) -> np.ndarray:
    thresholds = horizon_thresholds(args)
    vec = np.zeros((rich_state_dim(args),), dtype=np.float32)
    kind = str(event.event_kind)
    if kind in RICH_REACTION_KINDS:
        vec[RICH_REACTION_KINDS.index(kind)] = 1.0
    offset = len(RICH_REACTION_KINDS)
    vec[offset + 0] = min(1.0, math.log1p(max(0.0, float(event.read_time))) / 5.0)
    vec[offset + 1] = min(1.0, max(0.0, float(event.scroll)) / 100.0)
    vec[offset + 2] = min(1.0, math.log1p(max(0.0, float(event.next_read_time))) / 5.0)
    vec[offset + 3] = min(1.0, max(0.0, float(event.next_scroll)) / 100.0)
    vec[offset + 4] = min(1.0, max(0.0, float(event.weight)) / 4.0)
    h_offset = offset + RICH_SCALAR_DIM
    bucket = len(thresholds)
    dt = max(0.0, float(event.dt_hours))
    for i, threshold in enumerate(thresholds):
        if dt <= threshold:
            bucket = i
            break
    vec[h_offset + bucket] = 1.0
    return vec


def add_rich_event_state_index(
    samples: list[q.Sample],
    article_to_idx: dict[int, int],
    full_index: q.FullSignalIndex,
    event_index: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    state_dim = rich_state_dim(args)
    n = len(samples)
    pos_state = np.zeros((n, args.max_pos_events, state_dim), dtype=np.float32)
    neg_state = np.zeros((n, args.max_neg_events, state_dim), dtype=np.float32)
    stats = Counter()
    for sample in samples:
        pos_events, neg_events, _ = ep.collect_event_records(sample, article_to_idx, full_index, args)
        sid = sample.sample_id
        for j, event in enumerate(pos_events[: args.max_pos_events]):
            pos_state[sid, j] = encode_rich_event_state(event, args)
            stats[f"pos_{event.event_kind}"] += 1
        for j, event in enumerate(neg_events[: args.max_neg_events]):
            neg_state[sid, j] = encode_rich_event_state(event, args)
            stats[f"neg_{event.event_kind}"] += 1
    event_index["pos_state"] = pos_state
    event_index["neg_state"] = neg_state
    event_index["rich_state_dim"] = np.asarray([state_dim], dtype=np.int32)
    return {
        "enabled": True,
        "state_dim": state_dim,
        "reaction_kinds": list(RICH_REACTION_KINDS),
        "scalar_fields": ["read_log", "scroll", "next_read_log", "next_scroll", "engagement_weight"],
        "horizon_bucket_hours": horizon_thresholds(args),
        "counts": dict(stats),
    }


def patch_event_batch_builder_for_rich_state() -> None:
    if getattr(ep.EventBatchBuilder, "_rich_state_patch", False):
        return
    original_call = ep.EventBatchBuilder.__call__

    def patched_call(self: Any, samples: list[q.Sample]) -> dict[str, torch.Tensor | np.ndarray]:
        out = original_call(self, samples)
        sample_ids = out["sample_ids"]
        for key in ("pos", "neg"):
            state_key = f"{key}_state"
            if state_key in self.event_index:
                out[state_key] = torch.from_numpy(self.event_index[state_key][sample_ids].astype(np.float32))
        return out

    ep.EventBatchBuilder.__call__ = patched_call
    ep.EventBatchBuilder._rich_state_patch = True


def build_next_round_index(
    samples: list[q.Sample],
    article_to_idx: dict[int, int],
    full_index: q.FullSignalIndex,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n = len(samples)
    c = int(args.max_next_rank_candidates)
    k_steps = int(args.rollout_steps)
    next_idx = np.zeros((n, c), dtype=np.int32)
    next_gain = np.zeros((n, c), dtype=np.float32)
    next_label = np.zeros((n, c), dtype=np.int32)
    next_mask = np.zeros((n, c), dtype=bool)
    next_article_id = np.zeros((n, c), dtype=np.int64)
    next_group_id = np.zeros((n,), dtype=np.int64)
    next_valid = np.zeros((n,), dtype=bool)
    next_gap_hours = np.zeros((n,), dtype=np.float32)
    rollout_idx = np.zeros((n, k_steps, c), dtype=np.int32)
    rollout_gain = np.zeros((n, k_steps, c), dtype=np.float32)
    rollout_label = np.zeros((n, k_steps, c), dtype=np.int32)
    rollout_mask = np.zeros((n, k_steps, c), dtype=bool)
    rollout_article_id = np.zeros((n, k_steps, c), dtype=np.int64)
    rollout_group_id = np.zeros((n, k_steps), dtype=np.int64)
    rollout_valid = np.zeros((n, k_steps), dtype=bool)
    rollout_gap_hours = np.zeros((n, k_steps), dtype=np.float32)
    stats = Counter()
    gap_values: list[float] = []

    for sample in samples:
        sid = int(sample.sample_id)
        rows, current_idx, current_row = full_index.current(sample)
        stats["samples"] += 1
        if current_idx is None or current_idx + 1 >= len(rows):
            stats["missing_next"] += 1
            continue
        current_time = pd.Timestamp(sample.time if current_row is None else getattr(current_row, "impression_time"))
        any_rollout = False
        for step in range(k_steps):
            row_idx = current_idx + step + 1
            if row_idx >= len(rows):
                stats[f"missing_step_{step + 1}"] += 1
                continue
            next_row = rows[row_idx]
            next_time = pd.Timestamp(getattr(next_row, "impression_time"))
            gap_h = max(0.0, (next_time - current_time).total_seconds() / 3600.0)
            if args.next_request_max_gap_hours > 0 and gap_h > args.next_request_max_gap_hours:
                stats[f"too_far_step_{step + 1}"] += 1
                if step == 0:
                    stats["too_far_next"] += 1
                continue
            clicked = set(q.ids(getattr(next_row, "article_ids_clicked")))
            read = q.clean_float(getattr(next_row, "read_time", 0.0))
            scroll = q.clean_float(getattr(next_row, "scroll_percentage", 0.0))
            rows_out: list[tuple[int, int, float, int]] = []
            for aid in q.ids(getattr(next_row, "article_ids_inview")):
                idx = article_to_idx.get(int(aid))
                if idx is None:
                    continue
                label = int(int(aid) in clicked)
                gain = ep.event_weight(read, scroll, gap_h, positive=True) if label else 0.0
                rows_out.append((int(idx), int(aid), float(gain), label))
            if not rows_out:
                stats[f"empty_step_{step + 1}"] += 1
                if step == 0:
                    stats["empty_next_candidates"] += 1
                continue
            rng = np.random.default_rng(int(args.seed) + sid * 104729 + 41 + step * 1009)
            if len(rows_out) > c:
                positives = [row for row in rows_out if row[3] > 0]
                negatives = [row for row in rows_out if row[3] <= 0]
                rng.shuffle(positives)
                rng.shuffle(negatives)
                rows_out = (positives + negatives)[:c]
                stats[f"truncated_step_{step + 1}"] += 1
                if step == 0:
                    stats["next_truncated"] += 1
            else:
                rng.shuffle(rows_out)
            positive_count = sum(row[3] for row in rows_out)
            if positive_count <= 0:
                stats[f"empty_positive_step_{step + 1}"] += 1
            stats[f"valid_step_{step + 1}"] += 1
            stats[f"candidates_step_{step + 1}"] += len(rows_out)
            stats[f"positive_step_{step + 1}"] += positive_count
            rollout_valid[sid, step] = True
            rollout_gap_hours[sid, step] = float(gap_h)
            rollout_group_id[sid, step] = int(getattr(next_row, "impression_id"))
            for j, (idx, aid, gain, label) in enumerate(rows_out):
                rollout_idx[sid, step, j] = idx
                rollout_article_id[sid, step, j] = aid
                rollout_gain[sid, step, j] = gain
                rollout_label[sid, step, j] = label
                rollout_mask[sid, step, j] = True
            any_rollout = True
            if step == 0:
                stats["valid_next"] += 1
                stats["next_candidates"] += len(rows_out)
                stats["next_positive"] += positive_count
                stats["next_empty_positive"] += int(positive_count == 0)
                gap_values.append(float(gap_h))
                next_valid[sid] = True
                next_gap_hours[sid] = float(gap_h)
                next_group_id[sid] = int(getattr(next_row, "impression_id"))
                for j, (idx, aid, gain, label) in enumerate(rows_out):
                    next_idx[sid, j] = idx
                    next_article_id[sid, j] = aid
                    next_gain[sid, j] = gain
                    next_label[sid, j] = label
                    next_mask[sid, j] = True
        stats["rollout_any"] += int(any_rollout)

    meta = {
        "samples": int(stats["samples"]),
        "valid_next": int(stats["valid_next"]),
        "missing_next": int(stats["missing_next"]),
        "too_far_next": int(stats["too_far_next"]),
        "empty_next_candidates": int(stats["empty_next_candidates"]),
        "next_candidates": int(stats["next_candidates"]),
        "next_positive": int(stats["next_positive"]),
        "next_empty_positive": int(stats["next_empty_positive"]),
        "next_truncated": int(stats["next_truncated"]),
        "avg_next_candidates": float(stats["next_candidates"] / max(1, stats["valid_next"])),
        "avg_next_positive": float(stats["next_positive"] / max(1, stats["valid_next"])),
        "avg_next_gap_hours": float(np.mean(gap_values)) if gap_values else 0.0,
        "rollout_steps": k_steps,
        "rollout_any": int(stats["rollout_any"]),
        "rollout_by_step": {
            str(step + 1): {
                "valid": int(stats[f"valid_step_{step + 1}"]),
                "candidates": int(stats[f"candidates_step_{step + 1}"]),
                "positive": int(stats[f"positive_step_{step + 1}"]),
                "avg_candidates": float(stats[f"candidates_step_{step + 1}"] / max(1, stats[f"valid_step_{step + 1}"])),
                "avg_positive": float(stats[f"positive_step_{step + 1}"] / max(1, stats[f"valid_step_{step + 1}"])),
                "empty_positive": int(stats[f"empty_positive_step_{step + 1}"]),
                "too_far": int(stats[f"too_far_step_{step + 1}"]),
                "missing": int(stats[f"missing_step_{step + 1}"]),
            }
            for step in range(k_steps)
        },
    }
    return {
        "next_idx": next_idx,
        "next_article_id": next_article_id,
        "next_gain": next_gain,
        "next_label": next_label,
        "next_mask": next_mask,
        "next_group_id": next_group_id,
        "next_valid": next_valid,
        "next_gap_hours": next_gap_hours,
        "rollout_idx": rollout_idx,
        "rollout_article_id": rollout_article_id,
        "rollout_gain": rollout_gain,
        "rollout_label": rollout_label,
        "rollout_mask": rollout_mask,
        "rollout_group_id": rollout_group_id,
        "rollout_valid": rollout_valid,
        "rollout_gap_hours": rollout_gap_hours,
    }, meta


def patch_event_batch_builder_for_next_round() -> None:
    if getattr(ep.EventBatchBuilder, "_next_round_patch", False):
        return
    original_call = ep.EventBatchBuilder.__call__

    def patched_call(self: Any, samples: list[q.Sample]) -> dict[str, torch.Tensor | np.ndarray]:
        out = original_call(self, samples)
        sample_ids = out["sample_ids"]
        if "next_idx" not in self.event_index:
            return out
        bsz = len(samples)
        max_c = self.event_index["next_idx"].shape[1]
        d = self.item_emb.shape[1]
        item_emb = np.zeros((bsz, max_c, d), dtype=np.float32)
        scalar = np.zeros((bsz, max_c, self.scalar_dim), dtype=np.float32)
        item_idx = np.zeros((bsz, max_c), dtype=np.int64)
        labels = self.event_index["next_gain"][sample_ids].astype(np.float32)
        eval_labels = self.event_index["next_label"][sample_ids].astype(np.int32)
        mask = self.event_index["next_mask"][sample_ids].astype(bool)
        article_ids = self.event_index["next_article_id"][sample_ids].astype(np.int64)
        group_ids = np.zeros((bsz, max_c), dtype=np.int64)
        supervised = self.event_index["next_valid"][sample_ids].astype(bool) & (eval_labels.sum(axis=1) > 0)
        gap_hours = self.event_index["next_gap_hours"][sample_ids].astype(np.float32)
        for bi, sample in enumerate(samples):
            sid = int(sample.sample_id)
            z_source = self.source_emb[sid].astype(np.float32)
            candidate_count = int(mask[bi].sum())
            for ci in range(max_c):
                if not mask[bi, ci]:
                    continue
                idx = int(self.event_index["next_idx"][sid, ci])
                aid = int(article_ids[bi, ci])
                item = self.item_emb[idx].astype(np.float32)
                item_emb[bi, ci] = item
                item_idx[bi, ci] = self.item_to_idx.get(aid, 0)
                position_feature = float(ci) / max(1.0, float(candidate_count - 1))
                base = [q.cosine(item, z_source), 0.0, float(gap_hours[bi]) / 24.0, position_feature]
                base += q.article_dense_meta(self.articles_by_id, aid, sample, candidate_count)
                scalar[bi, ci] = np.asarray(base + [0.0, 0.0, 0.0], dtype=np.float32)
                group_ids[bi, ci] = int(self.event_index["next_group_id"][sid])
        out["next_item_emb"] = torch.from_numpy(item_emb)
        out["next_scalar"] = torch.from_numpy(scalar)
        out["next_labels"] = torch.from_numpy(labels)
        out["next_eval_labels"] = torch.from_numpy(eval_labels)
        out["next_mask"] = torch.from_numpy(mask)
        out["next_item_idx"] = torch.from_numpy(item_idx)
        out["next_rank_supervised"] = torch.from_numpy(supervised)
        out["next_gap_hours"] = torch.from_numpy(gap_hours)
        out["next_group_ids"] = group_ids
        if "rollout_idx" in self.event_index:
            steps = self.event_index["rollout_idx"].shape[1]
            rollout_item_emb = np.zeros((bsz, steps, max_c, d), dtype=np.float32)
            rollout_scalar = np.zeros((bsz, steps, max_c, self.scalar_dim), dtype=np.float32)
            rollout_item_idx = np.zeros((bsz, steps, max_c), dtype=np.int64)
            rollout_labels = self.event_index["rollout_gain"][sample_ids].astype(np.float32)
            rollout_eval_labels = self.event_index["rollout_label"][sample_ids].astype(np.int32)
            rollout_mask = self.event_index["rollout_mask"][sample_ids].astype(bool)
            rollout_article_ids = self.event_index["rollout_article_id"][sample_ids].astype(np.int64)
            rollout_group_ids = np.zeros((bsz, steps, max_c), dtype=np.int64)
            rollout_valid = self.event_index["rollout_valid"][sample_ids].astype(bool) & (rollout_eval_labels.sum(axis=2) > 0)
            rollout_gap_hours = self.event_index["rollout_gap_hours"][sample_ids].astype(np.float32)
            for bi, sample in enumerate(samples):
                sid = int(sample.sample_id)
                z_source = self.source_emb[sid].astype(np.float32)
                for step in range(steps):
                    candidate_count = int(rollout_mask[bi, step].sum())
                    for ci in range(max_c):
                        if not rollout_mask[bi, step, ci]:
                            continue
                        idx = int(self.event_index["rollout_idx"][sid, step, ci])
                        aid = int(rollout_article_ids[bi, step, ci])
                        item = self.item_emb[idx].astype(np.float32)
                        rollout_item_emb[bi, step, ci] = item
                        rollout_item_idx[bi, step, ci] = self.item_to_idx.get(aid, 0)
                        position_feature = float(ci) / max(1.0, float(candidate_count - 1))
                        base = [q.cosine(item, z_source), 0.0, float(rollout_gap_hours[bi, step]) / 24.0, position_feature]
                        base += q.article_dense_meta(self.articles_by_id, aid, sample, candidate_count)
                        rollout_scalar[bi, step, ci] = np.asarray(base + [0.0, 0.0, 0.0], dtype=np.float32)
                        rollout_group_ids[bi, step, ci] = int(self.event_index["rollout_group_id"][sid, step])
            out["rollout_item_emb"] = torch.from_numpy(rollout_item_emb)
            out["rollout_scalar"] = torch.from_numpy(rollout_scalar)
            out["rollout_labels"] = torch.from_numpy(rollout_labels)
            out["rollout_eval_labels"] = torch.from_numpy(rollout_eval_labels)
            out["rollout_mask"] = torch.from_numpy(rollout_mask)
            out["rollout_item_idx"] = torch.from_numpy(rollout_item_idx)
            out["rollout_rank_supervised"] = torch.from_numpy(rollout_valid)
            out["rollout_gap_hours"] = torch.from_numpy(rollout_gap_hours)
            out["rollout_group_ids"] = rollout_group_ids
        return out

    ep.EventBatchBuilder.__call__ = patched_call
    ep.EventBatchBuilder._next_round_patch = True


class MiniHSTUBlock(nn.Module):
    """Pure PyTorch HSTU-style block using gated pointwise aggregation.

    This is a fallback for the remote environment where the official Meta HSTU
    package and FBGEMM jagged kernels are not installed. It deliberately keeps
    the HSTU ingredients we need for the EB-NeRD smoke: pointwise projected
    q/k/v/u, non-softmax aggregation, gating, and residual feed-forward.
    """

    def __init__(self, d_model: int, heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm = nn.LayerNorm(d_model)
        self.qkvu = nn.Linear(d_model, d_model * 4)
        self.out = nn.Linear(d_model, d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        valid = mask.bool()
        h = self.norm(x)
        q_proj, k_proj, v_proj, u_proj = self.qkvu(h).chunk(4, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)

        qh = split_heads(q_proj)
        kh = split_heads(k_proj)
        vh = split_heads(v_proj)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        attn_mask = valid[:, None, None, :] & valid[:, None, :, None]
        if causal:
            lower = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device).tril()
            attn_mask = attn_mask & lower[None, None, :, :]
        scores = scores.masked_fill(~attn_mask, -1e4)
        weights = F.relu(F.silu(scores)) * attn_mask.float()
        scale = valid.float().sum(dim=1).clamp_min(1.0).sqrt().view(bsz, 1, 1, 1)
        ctx = torch.matmul(weights / scale, vh).transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        gated = ctx * torch.sigmoid(u_proj)
        x = x + self.drop(self.out(gated)) * valid.unsqueeze(-1).float()
        x = x + self.drop(self.ff(self.norm_ff(x))) * valid.unsqueeze(-1).float()
        return x


class MiniHSTUStack(nn.Module):
    def __init__(self, layers: int, d_model: int, heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MiniHSTUBlock(d_model, heads, ff_dim, dropout) for _ in range(layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask, causal=causal)
        return x


class FutureReactionPredictor(nn.Module):
    """Predict future item reactions from an HSTU context state.

    The slots remain the JEPA-style latent future UIH set, but the predictor is
    now explicitly supervised to score each future exposure candidate as a
    positive/negative reaction. This gives us a direct probe of whether the
    future latent is learning useful item-level behavior before the ranker uses
    it.
    """

    def __init__(
        self,
        emb_dim: int,
        scalar_dim: int,
        d_model: int,
        heads: int,
        ff_dim: int,
        dropout: float,
        num_slots: int,
        state_dim: int,
        slot_query_scale: float,
        disable_slot_self_attn: bool,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.state_dim = state_dim
        self.slot_query_scale = float(slot_query_scale)
        self.disable_slot_self_attn = bool(disable_slot_self_attn)
        self.slot_query = nn.Parameter(torch.randn(num_slots, d_model) * 0.20)
        self.slot_stack = MiniHSTUStack(1, d_model, heads, ff_dim, dropout)
        self.slot_to_emb = nn.Linear(d_model, emb_dim)
        self.slot_proj = nn.Linear(emb_dim, d_model)
        self.slot_state_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim // 2),
            nn.GELU(),
            nn.Linear(ff_dim // 2, state_dim),
        )
        feature_dim = d_model * 6 + scalar_dim + 1 + state_dim
        self.reaction_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 2, 1),
        )

    def pointwise_context(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bcd,btd->bct", query, key) / math.sqrt(float(query.shape[-1]))
        logits = logits.masked_fill(~mask.unsqueeze(1), -1e4)
        weights = F.relu(F.silu(logits)) * mask.unsqueeze(1).float()
        weights = weights / mask.float().sum(dim=1).clamp_min(1.0).sqrt().view(-1, 1, 1)
        return torch.einsum("bct,btd->bcd", weights, value)

    def predict_slots(
        self,
        user_state: torch.Tensor,
        candidate_state: torch.Tensor,
        item_hidden: torch.Tensor,
        item_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz = user_state.shape[0]
        slot_mask = torch.ones((bsz, self.num_slots), dtype=torch.bool, device=user_state.device)
        slot = (
            self.slot_query_scale * self.slot_query.unsqueeze(0).expand(bsz, -1, -1)
            + user_state.unsqueeze(1)
            + candidate_state.unsqueeze(1)
        )
        if not self.disable_slot_self_attn:
            slot = self.slot_stack(slot, slot_mask, causal=False)
        slot = slot + self.pointwise_context(slot, item_hidden, item_hidden, item_mask)
        return F.normalize(self.slot_to_emb(slot), dim=-1, eps=1e-8)

    def forward(
        self,
        user_state: torch.Tensor,
        candidate_state: torch.Tensor,
        item_hidden: torch.Tensor,
        item_emb: torch.Tensor,
        scalar: torch.Tensor,
        item_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = self.predict_slots(user_state, candidate_state, item_hidden, item_mask)
        slot_hidden = self.slot_proj(slots)
        slot_state = torch.sigmoid(self.slot_state_head(slot_hidden))
        slot_mask = torch.ones((slots.shape[0], slots.shape[1]), dtype=torch.bool, device=slots.device)
        slot_ctx = self.pointwise_context(item_hidden, slot_hidden, slot_hidden, slot_mask)
        slot_logits = torch.einsum(
            "bcd,bkd->bck",
            F.normalize(item_emb, dim=-1, eps=1e-8),
            F.normalize(slots, dim=-1, eps=1e-8),
        )
        slot_dot = slot_logits.max(dim=-1).values
        slot_assign = F.softmax(slot_logits / 0.20, dim=-1)
        candidate_rich_state = torch.einsum("bck,bks->bcs", slot_assign, slot_state)
        user_expand = user_state.unsqueeze(1).expand_as(item_hidden)
        candidate_expand = candidate_state.unsqueeze(1).expand_as(item_hidden)
        features = torch.cat(
            [
                item_hidden,
                slot_ctx,
                item_hidden * slot_ctx,
                torch.abs(item_hidden - slot_ctx),
                user_expand,
                candidate_expand,
                scalar,
                slot_dot.unsqueeze(-1),
                candidate_rich_state,
            ],
            dim=-1,
        )
        reaction_logit = self.reaction_head(features).squeeze(-1)
        return slots, slot_state, slot_ctx, slot_dot, candidate_rich_state, reaction_logit


class HstuJepaRanker(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        scalar_dim: int,
        num_users: int,
        num_items: int,
        d_model: int,
        heads: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        num_slots: int,
        state_dim: int,
        use_jepa_slots: bool,
        use_item_id_emb: bool,
        slot_rank_scale: float,
        slot_query_scale: float,
        disable_slot_self_attn: bool,
        transition_residual_scale: float,
        transition_score_residual: bool,
        transition_score_scale: float,
    ) -> None:
        super().__init__()
        self.use_jepa_slots = use_jepa_slots
        self.use_item_id_emb = use_item_id_emb
        self.num_slots = num_slots
        self.slot_rank_scale = float(slot_rank_scale)
        self.slot_query_scale = float(slot_query_scale)
        self.disable_slot_self_attn = bool(disable_slot_self_attn)
        self.transition_residual_scale = float(transition_residual_scale)
        self.transition_score_residual = bool(transition_score_residual)
        self.transition_score_scale = float(transition_score_scale)
        self.vec_proj = nn.Linear(emb_dim, d_model)
        self.scalar_proj = nn.Linear(scalar_dim, d_model)
        self.hist_age_proj = nn.Linear(1, d_model)
        self.user_emb = nn.Embedding(num_users + 1, d_model, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.hist_type_emb = nn.Embedding(4, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(8, d_model)
        self.history_stack = MiniHSTUStack(max(1, layers), d_model, heads, ff_dim, dropout)
        self.item_stack = MiniHSTUStack(max(1, layers), d_model, heads, ff_dim, dropout)
        self.future_predictor = FutureReactionPredictor(
            emb_dim=emb_dim,
            scalar_dim=scalar_dim,
            d_model=d_model,
            heads=heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_slots=num_slots,
            state_dim=state_dim,
            slot_query_scale=slot_query_scale,
            disable_slot_self_attn=disable_slot_self_attn,
        )
        self.hist_gate = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.transition_head = nn.Sequential(
            nn.LayerNorm(d_model * 4 + 1),
            nn.Linear(d_model * 4 + 1, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.transition_score_head = nn.Sequential(
            nn.LayerNorm(d_model * 4 + 1),
            nn.Linear(d_model * 4 + 1, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 2, 1),
        )
        nn.init.zeros_(self.transition_score_head[-1].weight)
        nn.init.zeros_(self.transition_score_head[-1].bias)
        feature_dim = d_model * 9 + scalar_dim + 2 + state_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def pointwise_context(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bcd,btd->bct", query, key) / math.sqrt(float(query.shape[-1]))
        logits = logits.masked_fill(~mask.unsqueeze(1), -1e4)
        weights = F.relu(F.silu(logits)) * mask.unsqueeze(1).float()
        weights = weights / mask.float().sum(dim=1).clamp_min(1.0).sqrt().view(-1, 1, 1)
        return torch.einsum("bct,btd->bcd", weights, value)

    def encode_history_tensors(
        self,
        hist_emb: torch.Tensor,
        hist_type: torch.Tensor,
        hist_age: torch.Tensor,
        hist_mask: torch.Tensor,
        user_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist = (
            self.vec_proj(hist_emb)
            + self.hist_type_emb(hist_type)
            + self.hist_age_proj(hist_age)
            + self.type_emb.weight[0]
        )
        hist = self.history_stack(hist, hist_mask, causal=True)
        mean_state = masked_mean(hist, hist_mask)
        last_state = last_valid(hist, hist_mask)
        user_state = 0.5 * (mean_state + last_state) + self.user_emb(user_idx)
        return hist, hist_mask, user_state

    def encode_history(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode_history_tensors(
            batch["hist_emb"],
            batch["hist_type"],
            batch["hist_age"],
            batch["hist_mask"],
            batch["user_idx"],
        )

    def encode_history_with_events(
        self,
        batch: dict[str, torch.Tensor],
        event_emb: torch.Tensor,
        event_type: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, max_hist, dim = batch["hist_emb"].shape
        all_emb = torch.cat([batch["hist_emb"], event_emb], dim=1)
        all_type = torch.cat([batch["hist_type"], event_type], dim=1)
        all_mask = torch.cat([batch["hist_mask"], event_mask], dim=1)
        hist_emb = torch.zeros((bsz, max_hist, dim), dtype=all_emb.dtype, device=all_emb.device)
        hist_type = torch.zeros((bsz, max_hist), dtype=all_type.dtype, device=all_type.device)
        hist_age = torch.zeros((bsz, max_hist, 1), dtype=batch["hist_age"].dtype, device=all_emb.device)
        hist_mask = torch.zeros((bsz, max_hist), dtype=torch.bool, device=all_emb.device)
        for bi in range(bsz):
            valid = torch.nonzero(all_mask[bi], as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            keep = valid[-max_hist:]
            n = int(keep.numel())
            hist_emb[bi, :n] = all_emb[bi, keep]
            hist_type[bi, :n] = all_type[bi, keep]
            hist_mask[bi, :n] = True
            if n > 1:
                hist_age[bi, :n, 0] = torch.linspace(0.0, 1.0, n, dtype=hist_age.dtype, device=hist_age.device)
        return self.encode_history_tensors(hist_emb, hist_type, hist_age, hist_mask, batch["user_idx"])

    def jepa_parameters(self) -> list[nn.Parameter]:
        return [p for name, p in self.named_parameters() if not name.startswith("head.") and p.requires_grad]

    def encode_items(
        self,
        item_emb: torch.Tensor,
        scalar: torch.Tensor,
        mask: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        item = self.vec_proj(item_emb) + self.scalar_proj(scalar) + self.type_emb.weight[1]
        if self.use_item_id_emb:
            item = item + self.item_id_emb(item_idx)
        return self.item_stack(item, mask, causal=False)

    def score_items_from_state(
        self,
        user_state: torch.Tensor,
        item: torch.Tensor,
        raw_item_emb: torch.Tensor,
        scalar: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        candidate_state = masked_mean(item, mask)
        candidate_expand = candidate_state.unsqueeze(1).expand_as(item)
        state_expand = user_state.unsqueeze(1).expand_as(item)
        slot_ctx = torch.zeros_like(item)
        slot_dot = torch.zeros(mask.shape, dtype=item.dtype, device=item.device)
        reaction_feature = torch.zeros_like(slot_dot)
        rich_state_feature = torch.zeros((*mask.shape, self.future_predictor.state_dim), dtype=item.dtype, device=item.device)
        features = torch.cat(
            [
                item,
                state_expand,
                item * state_expand,
                torch.abs(item - state_expand),
                candidate_expand,
                item * candidate_expand,
                state_expand,
                slot_ctx,
                item * slot_ctx,
                scalar,
                slot_dot.unsqueeze(-1),
                reaction_feature.unsqueeze(-1),
                rich_state_feature,
            ],
            dim=-1,
        )
        return self.head(features).squeeze(-1)

    def score_raw_slate_from_state(
        self,
        user_state: torch.Tensor,
        item_emb: torch.Tensor,
        scalar: torch.Tensor,
        mask: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.encode_items(item_emb, scalar, mask, item_idx)
        score = self.score_items_from_state(user_state, item, item_emb, scalar, mask)
        return score, item

    def transition_next_state(
        self,
        user_state: torch.Tensor,
        current_item: torch.Tensor,
        reactions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        reactions = reactions.float().clamp(0.0, 1.0)
        slate_state = masked_mean(current_item, mask)
        pos_w = reactions * mask.float()
        neg_w = (1.0 - reactions) * mask.float()
        pos_state = (current_item * pos_w.unsqueeze(-1)).sum(dim=1) / pos_w.sum(dim=1, keepdim=True).clamp_min(1.0)
        neg_state = (current_item * neg_w.unsqueeze(-1)).sum(dim=1) / neg_w.sum(dim=1, keepdim=True).clamp_min(1.0)
        click_rate = pos_w.sum(dim=1, keepdim=True) / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        delta = self.transition_head(torch.cat([user_state, slate_state, pos_state, neg_state, click_rate], dim=-1))
        return user_state + self.transition_residual_scale * delta

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        hist, hist_mask, user_state = self.encode_history(batch)
        item = self.encode_items(batch["item_emb"], batch["scalar"], batch["mask"], batch["item_idx"])
        hist_ctx = self.pointwise_context(item, hist, hist, hist_mask)
        hist_mean = masked_mean(hist, hist_mask).unsqueeze(1).expand_as(item)
        gate = self.hist_gate(torch.cat([item, hist_ctx], dim=-1))
        hist_out = gate * hist_ctx + (1.0 - gate) * hist_mean
        candidate_state = masked_mean(item, batch["mask"])
        candidate_expand = candidate_state.unsqueeze(1).expand_as(item)
        user_expand = user_state.unsqueeze(1).expand_as(item)
        slot_ctx = torch.zeros_like(item)
        slot_dot = torch.zeros_like(batch["labels"])
        reaction_feature = torch.zeros_like(batch["labels"])
        candidate_rich_state = torch.zeros((*batch["labels"].shape, self.future_predictor.state_dim), dtype=item.dtype, device=item.device)
        rich_state_feature = torch.zeros_like(candidate_rich_state)
        reaction_logit: torch.Tensor | None = None
        slot_state: torch.Tensor | None = None
        slots: torch.Tensor | None = None
        if self.use_jepa_slots:
            slots, slot_state, raw_slot_ctx, raw_slot_dot, raw_rich_state, reaction_logit = self.future_predictor(
                user_state,
                candidate_state,
                item,
                batch["item_emb"],
                batch["scalar"],
                batch["mask"],
            )
            slot_ctx = self.slot_rank_scale * raw_slot_ctx
            slot_dot = self.slot_rank_scale * raw_slot_dot
            reaction_feature = self.slot_rank_scale * reaction_logit
            candidate_rich_state = raw_rich_state
            rich_state_feature = self.slot_rank_scale * raw_rich_state
        features = torch.cat(
            [
                item,
                hist_out,
                item * hist_out,
                torch.abs(item - hist_out),
                candidate_expand,
                item * candidate_expand,
                user_expand,
                slot_ctx,
                item * slot_ctx,
                batch["scalar"],
                slot_dot.unsqueeze(-1),
                reaction_feature.unsqueeze(-1),
                rich_state_feature,
            ],
            dim=-1,
        )
        return self.head(features).squeeze(-1), slots, reaction_logit, slot_state, candidate_rich_state

    def forward_next_round(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_transition: bool | None = None,
        state_mode: str | None = None,
    ) -> torch.Tensor:
        if state_mode is None:
            state_mode = "transition_true" if use_transition else "stale"
        _, _, user_state = self.encode_history(batch)
        current_item = self.encode_items(batch["item_emb"], batch["scalar"], batch["mask"], batch["item_idx"])
        if state_mode == "reencode_true":
            event_type = torch.where(batch["eval_labels"] > 0, 1, 2).long() * batch["mask"].long()
            _, _, user_state = self.encode_history_with_events(batch, batch["item_emb"], event_type, batch["mask"])
        elif state_mode == "transition_true":
            user_state = self.transition_next_state(user_state, current_item, batch["eval_labels"].float(), batch["mask"])
        elif state_mode == "transition_pred":
            current_score, _, _, _, _ = self(batch)
            user_state = self.transition_next_state(user_state, current_item, torch.sigmoid(current_score), batch["mask"])
        elif state_mode != "stale":
            raise ValueError(f"unknown next-round state_mode={state_mode}")
        score, next_item = self.score_raw_slate_from_state(
            user_state,
            batch["next_item_emb"],
            batch["next_scalar"],
            batch["next_mask"],
            batch["next_item_idx"],
        )
        if state_mode.startswith("transition") and self.transition_score_residual:
            base_state = self.encode_history(batch)[2]
            base_score = self.score_items_from_state(
                base_state,
                next_item,
                batch["next_item_emb"],
                batch["next_scalar"],
                batch["next_mask"],
            )
            state_expand = user_state.unsqueeze(1).expand_as(next_item)
            gap = batch["next_gap_hours"].view(-1, 1, 1).expand(next_item.shape[0], next_item.shape[1], 1) / 24.0
            residual_features = torch.cat(
                [next_item, state_expand, next_item * state_expand, torch.abs(next_item - state_expand), gap],
                dim=-1,
            )
            residual = self.transition_score_head(residual_features).squeeze(-1)
            score = base_score + self.transition_score_scale * residual
        return score

    def rollout_scores(
        self,
        batch: dict[str, torch.Tensor],
        *,
        state_mode: str,
    ) -> list[torch.Tensor]:
        _, _, base_state = self.encode_history(batch)
        user_state = base_state
        current_item = self.encode_items(batch["item_emb"], batch["scalar"], batch["mask"], batch["item_idx"])
        extra_embs: list[torch.Tensor] = []
        extra_types: list[torch.Tensor] = []
        extra_masks: list[torch.Tensor] = []
        if state_mode in {"transition_true", "transition_pred"}:
            if state_mode == "transition_true":
                current_reaction = batch["eval_labels"].float()
            else:
                current_score, _, _, _, _ = self(batch)
                current_reaction = torch.sigmoid(current_score)
            user_state = self.transition_next_state(user_state, current_item, current_reaction, batch["mask"])
        elif state_mode == "reencode_true":
            event_type = torch.where(batch["eval_labels"] > 0, 1, 2).long() * batch["mask"].long()
            extra_embs.append(batch["item_emb"])
            extra_types.append(event_type)
            extra_masks.append(batch["mask"])
            _, _, user_state = self.encode_history_with_events(batch, batch["item_emb"], event_type, batch["mask"])
        elif state_mode != "direct_all":
            raise ValueError(f"unknown rollout state_mode={state_mode}")

        scores: list[torch.Tensor] = []
        steps = int(batch["rollout_item_emb"].shape[1])
        for step in range(steps):
            item_emb = batch["rollout_item_emb"][:, step]
            scalar = batch["rollout_scalar"][:, step]
            mask = batch["rollout_mask"][:, step]
            item_idx = batch["rollout_item_idx"][:, step]
            score, item = self.score_raw_slate_from_state(user_state, item_emb, scalar, mask, item_idx)
            if state_mode.startswith("transition") and self.transition_score_residual:
                base_score = self.score_items_from_state(base_state, item, item_emb, scalar, mask)
                state_expand = user_state.unsqueeze(1).expand_as(item)
                gap = batch["rollout_gap_hours"][:, step].view(-1, 1, 1).expand(item.shape[0], item.shape[1], 1) / 24.0
                residual_features = torch.cat(
                    [item, state_expand, item * state_expand, torch.abs(item - state_expand), gap],
                    dim=-1,
                )
                residual = self.transition_score_head(residual_features).squeeze(-1)
                score = base_score + self.transition_score_scale * residual
            scores.append(score)
            if state_mode == "transition_true":
                user_state = self.transition_next_state(user_state, item, batch["rollout_eval_labels"][:, step].float(), mask)
            elif state_mode == "transition_pred":
                user_state = self.transition_next_state(user_state, item, torch.sigmoid(score), mask)
            elif state_mode == "reencode_true":
                event_type = torch.where(batch["rollout_eval_labels"][:, step] > 0, 1, 2).long() * mask.long()
                extra_embs.append(item_emb)
                extra_types.append(event_type)
                extra_masks.append(mask)
                all_emb = torch.cat(extra_embs, dim=1)
                all_type = torch.cat(extra_types, dim=1)
                all_mask = torch.cat(extra_masks, dim=1)
                _, _, user_state = self.encode_history_with_events(batch, all_emb, all_type, all_mask)
        return scores


class FusionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, residual_scale: float) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_dim // 2), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, base_score: torch.Tensor) -> torch.Tensor:
        return base_score + self.residual_scale * self.net(features).squeeze(-1)


@torch.no_grad()
def predict_rank_scores(
    model: HstuJepaRanker,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, gains, scores, groups = [], [], [], []
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out, _, _, _, _ = model(moved)
        pred = out.detach().cpu().numpy()
        lab = batch["eval_labels"].numpy()
        gain = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        group = batch["group_ids"]
        labels.append(lab[mask])
        gains.append(gain[mask])
        scores.append(pred[mask])
        groups.append(group[mask])
    return np.concatenate(labels), np.concatenate(gains), np.concatenate(scores), np.concatenate(groups)


def hstu_topk_metrics(labels: np.ndarray, gains: np.ndarray, scores: np.ndarray, groups: np.ndarray, ks: tuple[int, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    labels = labels.astype(np.int32, copy=False)
    gains = gains.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)
    groups = groups.astype(np.int64, copy=False)
    buckets: dict[str, list[float]] = {f"hr@{k}": [] for k in ks}
    buckets.update({f"ndcg@{k}": [] for k in ks})
    buckets.update({f"engagement_ndcg@{k}": [] for k in ks})
    for gid in np.unique(groups):
        idx = np.where(groups == gid)[0]
        y = labels[idx]
        g = gains[idx]
        if y.max(initial=0) <= 0:
            continue
        order = np.argsort(-scores[idx])
        y_rank = y[order]
        g_rank = g[order]
        y_ideal = np.sort(y)[::-1]
        g_ideal = np.sort(g)[::-1]
        for k in ks:
            top_y = y_rank[:k]
            buckets[f"hr@{k}"].append(float(top_y.max(initial=0) > 0))
            disc = 1.0 / np.log2(np.arange(2, len(top_y) + 2))
            dcg = float(np.sum(top_y * disc))
            ideal = y_ideal[:k]
            idcg = float(np.sum(ideal * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))))
            buckets[f"ndcg@{k}"].append(dcg / idcg if idcg > 0 else 0.0)
            top_g = g_rank[:k]
            g_disc = 1.0 / np.log2(np.arange(2, len(top_g) + 2))
            g_dcg = float(np.sum(top_g * g_disc))
            ideal_g = g_ideal[:k]
            g_idcg = float(np.sum(ideal_g * (1.0 / np.log2(np.arange(2, len(ideal_g) + 2)))))
            buckets[f"engagement_ndcg@{k}"].append(g_dcg / g_idcg if g_idcg > 0 else 0.0)
    for key, values in buckets.items():
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def evaluate_model(
    model: HstuJepaRanker,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    y, gain, score, group = predict_rank_scores(model, loader, device)
    metrics = ep.ranking_metrics_for_task(y, score, group, args, gain_labels=gain)
    metrics.update(hstu_topk_metrics(y, gain, score, group, tuple(args.hstu_eval_ks)))
    return metrics


@torch.no_grad()
def predict_next_round_scores(
    model: HstuJepaRanker,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    use_transition: bool,
    state_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, gains, scores, groups = [], [], [], []
    for batch in loader:
        if "next_mask" not in batch:
            continue
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        pred = model.forward_next_round(moved, use_transition=use_transition, state_mode=state_mode).detach().cpu().numpy()
        lab = batch["next_eval_labels"].numpy()
        gain = batch["next_labels"].numpy()
        mask = (batch["next_mask"] & batch["next_rank_supervised"].unsqueeze(1)).numpy()
        group = batch["next_group_ids"]
        if mask.any():
            labels.append(lab[mask])
            gains.append(gain[mask])
            scores.append(pred[mask])
            groups.append(group[mask])
    if not labels:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty, empty, empty
    return np.concatenate(labels), np.concatenate(gains), np.concatenate(scores), np.concatenate(groups)


def evaluate_next_round_model(
    model: HstuJepaRanker,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    *,
    use_transition: bool,
    state_mode: str | None = None,
) -> dict[str, Any]:
    if state_mode is None:
        state_mode = "transition_true" if use_transition else "stale"
    y, gain, score, group = predict_next_round_scores(model, loader, device, use_transition=use_transition, state_mode=state_mode)
    if len(y) == 0:
        return {}
    metrics = ep.ranking_metrics_for_task(y, score, group, args, gain_labels=gain)
    metrics.update(hstu_topk_metrics(y, gain, score, group, tuple(args.hstu_eval_ks)))
    metrics["state_source"] = state_mode
    return metrics


@torch.no_grad()
def evaluate_rollout_model(
    model: HstuJepaRanker,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    *,
    state_mode: str,
) -> dict[str, Any]:
    model.eval()
    labels_by_step: list[list[np.ndarray]] = []
    gains_by_step: list[list[np.ndarray]] = []
    scores_by_step: list[list[np.ndarray]] = []
    groups_by_step: list[list[np.ndarray]] = []
    for batch in loader:
        if "rollout_mask" not in batch:
            continue
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        scores = model.rollout_scores(moved, state_mode=state_mode)
        steps = len(scores)
        if not labels_by_step:
            labels_by_step = [[] for _ in range(steps)]
            gains_by_step = [[] for _ in range(steps)]
            scores_by_step = [[] for _ in range(steps)]
            groups_by_step = [[] for _ in range(steps)]
        for step, pred_t in enumerate(scores):
            pred = pred_t.detach().cpu().numpy()
            lab = batch["rollout_eval_labels"][:, step].numpy()
            gain = batch["rollout_labels"][:, step].numpy()
            mask = (batch["rollout_mask"][:, step] & batch["rollout_rank_supervised"][:, step].unsqueeze(1)).numpy()
            group = batch["rollout_group_ids"][:, step]
            if mask.any():
                labels_by_step[step].append(lab[mask])
                gains_by_step[step].append(gain[mask])
                scores_by_step[step].append(pred[mask])
                groups_by_step[step].append(group[mask])
    out: dict[str, Any] = {"state_source": state_mode, "steps": {}}
    avg_values: dict[str, list[float]] = {}
    for step, labels_parts in enumerate(labels_by_step):
        if not labels_parts:
            continue
        y = np.concatenate(labels_parts)
        gain = np.concatenate(gains_by_step[step])
        score = np.concatenate(scores_by_step[step])
        group = np.concatenate(groups_by_step[step])
        metrics = ep.ranking_metrics_for_task(y, score, group, args, gain_labels=gain)
        metrics.update(hstu_topk_metrics(y, gain, score, group, tuple(args.hstu_eval_ks)))
        out["steps"][f"step+{step + 1}"] = metrics
        for key in ["ndcg@10", "mrr", "hr@10", "ndcg@50", "hit1"]:
            if key in metrics and metrics[key] is not None and not (isinstance(metrics[key], float) and math.isnan(metrics[key])):
                avg_values.setdefault(key, []).append(float(metrics[key]))
    out["avg"] = {key: float(np.mean(values)) for key, values in avg_values.items() if values}
    return out


def normalize_scores_torch(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    masked = scores.masked_fill(~mask, 0.0)
    mean = masked.sum(dim=1, keepdim=True) / denom
    var = (((scores - mean) * mask_f).pow(2).sum(dim=1, keepdim=True) / denom).clamp_min(1e-6)
    return ((scores - mean) / var.sqrt()).masked_fill(~mask, 0.0)


def future_reaction_loss(
    reaction_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    teacher_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    rank_mask = ep.supervised_rank_mask(batch)
    if not rank_mask.any():
        zero = reaction_logits.sum() * 0.0
        return zero, {"reaction_bce": 0.0, "reaction_listwise": 0.0, "reaction_pos_rate": 0.0, "reaction_residual_weight": 0.0}
    target = batch["eval_labels"].float()
    y = target[rank_mask]
    pos = y.sum()
    neg = y.numel() - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, args.reaction_pos_weight_cap)
    bce_each = F.binary_cross_entropy_with_logits(reaction_logits[rank_mask], y, pos_weight=pos_weight, reduction="none")
    residual_weight_mean = 1.0
    if teacher_scores is not None and args.residual_teacher_weight > 0:
        with torch.no_grad():
            masked_teacher = teacher_scores.masked_fill(~rank_mask, 0.0)
            denom = rank_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = masked_teacher.sum(dim=1, keepdim=True) / denom
            var = (((masked_teacher - mean) * rank_mask.float()).pow(2).sum(dim=1, keepdim=True) / denom).clamp_min(1e-6)
            z = ((teacher_scores - mean) / var.sqrt())[rank_mask]
            hard = torch.where(y > 0.5, torch.sigmoid(-z), torch.sigmoid(z))
            weights = 1.0 + args.residual_teacher_weight * hard
            residual_weight_mean = float(weights.mean().detach().cpu())
        bce = (bce_each * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        bce = bce_each.mean()
    listwise = q.listwise_loss(reaction_logits, batch["labels"], rank_mask)
    total = args.reaction_bce_weight * bce + args.reaction_listwise_weight * listwise
    return total, {
        "reaction_bce": float(bce.detach().cpu()),
        "reaction_listwise": float(listwise.detach().cpu()),
        "reaction_pos_rate": float(y.mean().detach().cpu()) if y.numel() else 0.0,
        "reaction_residual_weight": residual_weight_mean,
    }


@torch.no_grad()
def predict_reaction_scores(
    model: HstuJepaRanker,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, gains, scores, groups = [], [], [], []
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        _, _, reaction_logits, _, _ = model(moved)
        if reaction_logits is None:
            continue
        pred = reaction_logits.detach().cpu().numpy()
        lab = batch["eval_labels"].numpy()
        gain = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        group = batch["group_ids"]
        labels.append(lab[mask])
        gains.append(gain[mask])
        scores.append(pred[mask])
        groups.append(group[mask])
    if not labels:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty, empty, empty
    return np.concatenate(labels), np.concatenate(gains), np.concatenate(scores), np.concatenate(groups)


def evaluate_reaction_predictor(
    model: HstuJepaRanker,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    y, gain, score, group = predict_reaction_scores(model, loader, device)
    if len(y) == 0:
        return {}
    metrics = ep.ranking_metrics_for_task(y, score, group, args, gain_labels=gain)
    metrics.update({f"reaction_{k}": v for k, v in hstu_topk_metrics(y, gain, score, group, tuple(args.hstu_eval_ks)).items()})
    return metrics


def normalize_scores_per_group(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32, copy=True)
    out = np.zeros_like(scores)
    for gid in np.unique(groups):
        idx = groups == gid
        vals = scores[idx]
        out[idx] = (vals - vals.mean()) / max(1e-6, float(vals.std()))
    return out


def metrics_from_flat_scores(
    labels: np.ndarray,
    gains: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = ep.ranking_metrics_for_task(labels, scores, groups, args, gain_labels=gains)
    metrics.update(hstu_topk_metrics(labels, gains, scores, groups, tuple(args.hstu_eval_ks)))
    return metrics


@torch.no_grad()
def evaluate_reaction_blend(
    rank_model: HstuJepaRanker,
    reaction_model: HstuJepaRanker,
    loaders: dict[str, torch.utils.data.DataLoader],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    y_val, gain_val, rank_val, group_val = predict_rank_scores(rank_model, loaders["ranker_val"], device)
    ry_val, rgain_val, reaction_val, rgroup_val = predict_reaction_scores(reaction_model, loaders["ranker_val"], device)
    y_test, gain_test, rank_test, group_test = predict_rank_scores(rank_model, loaders["ranker_test"], device)
    ry_test, rgain_test, reaction_test, rgroup_test = predict_reaction_scores(reaction_model, loaders["ranker_test"], device)
    if not (
        np.array_equal(y_val, ry_val)
        and np.allclose(gain_val, rgain_val)
        and np.array_equal(group_val, rgroup_val)
        and np.array_equal(y_test, ry_test)
        and np.allclose(gain_test, rgain_test)
        and np.array_equal(group_test, rgroup_test)
    ):
        raise RuntimeError("rank and reaction prediction streams are not aligned")

    rank_val_n = normalize_scores_per_group(rank_val, group_val)
    reaction_val_n = normalize_scores_per_group(reaction_val, group_val)
    rank_test_n = normalize_scores_per_group(rank_test, group_test)
    reaction_test_n = normalize_scores_per_group(reaction_test, group_test)

    grid = [float(x) for x in args.reaction_blend_lambda_grid.split(",") if x.strip()]
    best_lam = 0.0
    best_metric = -float("inf")
    best_val: dict[str, Any] = {}
    for lam in grid:
        val_metrics = metrics_from_flat_scores(y_val, gain_val, rank_val_n + lam * reaction_val_n, group_val, args)
        score = ep.primary_metric(val_metrics, args)
        if score > best_metric:
            best_metric = score
            best_lam = lam
            best_val = val_metrics
    test_metrics = metrics_from_flat_scores(y_test, gain_test, rank_test_n + best_lam * reaction_test_n, group_test, args)
    return {
        "lambda": best_lam,
        "rank_val": metrics_from_flat_scores(y_val, gain_val, rank_val_n, group_val, args),
        "reaction_val": metrics_from_flat_scores(y_val, gain_val, reaction_val_n, group_val, args),
        "blend_val": best_val,
        "rank_test": metrics_from_flat_scores(y_test, gain_test, rank_test_n, group_test, args),
        "reaction_test": metrics_from_flat_scores(y_test, gain_test, reaction_test_n, group_test, args),
        "blend_test": test_metrics,
    }


@torch.no_grad()
def fusion_feature_tensors(
    rank_model: HstuJepaRanker,
    reaction_model: HstuJepaRanker,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    rank_score, _, _, _, _ = rank_model(batch)
    _, _, reaction_logits, _, rich_state = reaction_model(batch)
    if reaction_logits is None:
        raise RuntimeError("fusion head requires a JEPA predictor with reaction logits")
    mask = batch["mask"]
    rank_z = normalize_scores_torch(rank_score, mask)
    reaction_z = normalize_scores_torch(reaction_logits, mask)
    interaction = rank_z * reaction_z
    features = torch.cat(
        [
            rank_z.unsqueeze(-1),
            reaction_z.unsqueeze(-1),
            interaction.unsqueeze(-1),
            rich_state,
        ],
        dim=-1,
    )
    return features, rank_z


@torch.no_grad()
def predict_fusion_scores(
    fusion_head: FusionHead,
    rank_model: HstuJepaRanker,
    reaction_model: HstuJepaRanker,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fusion_head.eval()
    rank_model.eval()
    reaction_model.eval()
    labels, gains, scores, groups = [], [], [], []
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        features, base_score = fusion_feature_tensors(rank_model, reaction_model, moved)
        out = fusion_head(features, base_score)
        pred = out.detach().cpu().numpy()
        lab = batch["eval_labels"].numpy()
        gain = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        group = batch["group_ids"]
        labels.append(lab[mask])
        gains.append(gain[mask])
        scores.append(pred[mask])
        groups.append(group[mask])
    return np.concatenate(labels), np.concatenate(gains), np.concatenate(scores), np.concatenate(groups)


def evaluate_fusion_head(
    fusion_head: FusionHead,
    rank_model: HstuJepaRanker,
    reaction_model: HstuJepaRanker,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    y, gain, score, group = predict_fusion_scores(fusion_head, rank_model, reaction_model, loader, device)
    return metrics_from_flat_scores(y, gain, score, group, args)


def train_fusion_head(
    name: str,
    rank_model: HstuJepaRanker,
    reaction_model: HstuJepaRanker,
    loaders: dict[str, torch.utils.data.DataLoader],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    rank_model.eval()
    reaction_model.eval()
    for param in rank_model.parameters():
        param.requires_grad = False
    for param in reaction_model.parameters():
        param.requires_grad = False
    input_dim = args.rich_state_dim + 3
    fusion_head = FusionHead(
        input_dim=input_dim,
        hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        residual_scale=args.fusion_residual_scale,
    ).to(device)
    opt = torch.optim.AdamW(fusion_head.parameters(), lr=args.fusion_lr, weight_decay=args.weight_decay)
    initial = evaluate_fusion_head(fusion_head, rank_model, reaction_model, loaders["ranker_val"], args, device)
    best = ep.primary_metric(initial, args)
    best_state = {k: v.detach().cpu().clone() for k, v in fusion_head.state_dict().items()}
    history: list[dict[str, Any]] = [{"epoch": 0, "loss": None, "fusion_val": initial}]
    patience = args.fusion_patience
    print(json.dumps({name: {"fusion_epoch": 0, "fusion_val": initial}}, ensure_ascii=False), flush=True)
    for epoch in range(1, args.fusion_epochs + 1):
        fusion_head.train()
        losses = []
        for batch in loaders["ranker_train"]:
            moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            with torch.no_grad():
                features, base_score = fusion_feature_tensors(rank_model, reaction_model, moved)
            score = fusion_head(features, base_score)
            loss = q.listwise_loss(score, moved["labels"], moved["mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(fusion_head.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate_fusion_head(fusion_head, rank_model, reaction_model, loaders["ranker_val"], args, device)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "fusion_val": val,
        }
        history.append(row)
        print(json.dumps({name: {"fusion": row}}, ensure_ascii=False), flush=True)
        score_value = ep.primary_metric(val, args)
        if score_value > best:
            best = score_value
            best_state = {k: v.detach().cpu().clone() for k, v in fusion_head.state_dict().items()}
            patience = args.fusion_patience
        else:
            patience -= 1
            if patience <= 0:
                break
    fusion_head.load_state_dict(best_state)
    fusion_val = evaluate_fusion_head(fusion_head, rank_model, reaction_model, loaders["ranker_val"], args, device)
    fusion_test = evaluate_fusion_head(fusion_head, rank_model, reaction_model, loaders["ranker_test"], args, device)
    return {
        "input_features": ["rank_z", "reaction_z", "rank_z_x_reaction_z", "predicted_rich_state"],
        "input_dim": input_dim,
        "history": history,
        "best_val": best,
        "blend_val": fusion_val,
        "blend_test": fusion_test,
    }


@torch.no_grad()
def lgbm_rerank_features(
    rank_model: HstuJepaRanker,
    reaction_model: HstuJepaRanker,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], list[str]]:
    rank_model.eval()
    reaction_model.eval()
    xs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    gains: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    group_sizes: list[int] = []
    feature_names: list[str] = []
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        rank_score, _, _, _, _ = rank_model(moved)
        _, _, reaction_logits, _, rich_state = reaction_model(moved)
        if reaction_logits is None:
            raise RuntimeError("LGBM reranker requires a JEPA predictor with reaction logits")
        mask_t = moved["mask"]
        rank_z = normalize_scores_torch(rank_score, mask_t)
        reaction_z = normalize_scores_torch(reaction_logits, mask_t)
        blocks = [
            rank_score.unsqueeze(-1),
            rank_z.unsqueeze(-1),
            reaction_logits.unsqueeze(-1),
            reaction_z.unsqueeze(-1),
            (rank_z * reaction_z).unsqueeze(-1),
            rich_state,
        ]
        if args.lgbm_include_scalar:
            blocks.append(moved["scalar"])
        feature = torch.cat(blocks, dim=-1).detach().cpu().numpy().astype(np.float32, copy=False)
        if not feature_names:
            feature_names = ["rank_raw", "rank_z", "reaction_raw", "reaction_z", "rank_x_reaction"]
            feature_names += [f"rich_state_{i}" for i in range(rich_state.shape[-1])]
            if args.lgbm_include_scalar:
                feature_names += [f"scalar_{i}" for i in range(moved["scalar"].shape[-1])]
        mask = batch["mask"].numpy()
        lab = batch["eval_labels"].numpy().astype(np.int32, copy=False)
        gain = batch["labels"].numpy().astype(np.float32, copy=False)
        group = batch["group_ids"]
        for bi in range(feature.shape[0]):
            m = mask[bi]
            size = int(m.sum())
            if size <= 0:
                continue
            xs.append(feature[bi, m])
            labels.append(lab[bi, m])
            gains.append(gain[bi, m])
            groups.append(group[bi, m])
            group_sizes.append(size)
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(labels),
        np.concatenate(gains),
        np.concatenate(groups),
        group_sizes,
        feature_names,
    )


def train_lgbm_rich_jepa_reranker(
    name: str,
    rank_model: HstuJepaRanker,
    reaction_model: HstuJepaRanker,
    loaders: dict[str, torch.utils.data.DataLoader],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("LightGBM is required for --enable-lgbm-reranker") from exc

    print(json.dumps({name: {"stage": "extract_lgbm_features"}}, ensure_ascii=False), flush=True)
    x_train, y_train_eval, y_train_gain, g_train, train_group, feature_names = lgbm_rerank_features(rank_model, reaction_model, loaders["ranker_train"], args, device)
    x_val, y_val_eval, y_val_gain, g_val, val_group, _ = lgbm_rerank_features(rank_model, reaction_model, loaders["ranker_val"], args, device)
    x_test, y_test_eval, y_test_gain, g_test, _, _ = lgbm_rerank_features(rank_model, reaction_model, loaders["ranker_test"], args, device)
    y_train = y_train_eval if args.lgbm_binary_labels else np.clip(
        np.rint(y_train_gain * args.lgbm_gain_scale),
        0,
        args.lgbm_max_gain_label,
    ).astype(np.int32)
    y_val_fit = y_val_eval if args.lgbm_binary_labels else np.clip(
        np.rint(y_val_gain * args.lgbm_gain_scale),
        0,
        args.lgbm_max_gain_label,
    ).astype(np.int32)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=list(range(args.lgbm_max_gain_label + 1)),
        n_estimators=args.lgbm_estimators,
        learning_rate=args.lgbm_learning_rate,
        num_leaves=args.lgbm_num_leaves,
        min_child_samples=args.lgbm_min_child_samples,
        subsample=args.lgbm_subsample,
        colsample_bytree=args.lgbm_colsample_bytree,
        reg_lambda=args.lgbm_reg_lambda,
        random_state=args.seed,
        n_jobs=args.lgbm_n_jobs,
        verbose=-1,
    )
    print(
        json.dumps(
            {
                name: {
                    "stage": "fit_lgbm",
                    "train_rows": int(x_train.shape[0]),
                    "val_rows": int(x_val.shape[0]),
                    "test_rows": int(x_test.shape[0]),
                    "features": int(x_train.shape[1]),
                    "binary_labels": bool(args.lgbm_binary_labels),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    callbacks = []
    if args.lgbm_early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(args.lgbm_early_stopping_rounds, verbose=False))
    model.fit(
        x_train,
        y_train,
        group=train_group,
        eval_set=[(x_val, y_val_fit)],
        eval_group=[val_group],
        eval_at=[10],
        callbacks=callbacks,
    )
    best_iteration = int(model.best_iteration_ or args.lgbm_estimators)
    val_score = model.predict(x_val, num_iteration=best_iteration)
    test_score = model.predict(x_test, num_iteration=best_iteration)
    val = metrics_from_flat_scores(y_val_eval, y_val_gain, val_score, g_val, args)
    test = metrics_from_flat_scores(y_test_eval, y_test_gain, test_score, g_test, args)
    importance = model.booster_.feature_importance(importance_type="gain")
    top_features = sorted(
        [
            {"feature": feature_names[i], "gain": float(importance[i])}
            for i in range(min(len(feature_names), len(importance)))
        ],
        key=lambda x: x["gain"],
        reverse=True,
    )[:20]
    print(
        json.dumps(
            {
                name: {
                    "stage": "done_lgbm",
                    "best_iteration": best_iteration,
                    "val_primary": ep.primary_metric(val, args),
                    "test_primary": ep.primary_metric(test, args),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return {
        "kind": "lgbm_reranker",
        "feature_names": feature_names,
        "top_features": top_features,
        "best_iteration": best_iteration,
        "blend_val": val,
        "blend_test": test,
    }


def slot_specialization_loss(
    slots: torch.Tensor,
    pos_emb: torch.Tensor,
    pos_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    norm_slots = F.normalize(slots, dim=-1, eps=1e-8)
    sim = torch.einsum("bkd,bld->bkl", norm_slots, norm_slots)
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool).unsqueeze(0)
    orth = sim.masked_select(~eye).pow(2).mean()
    if not pos_mask.any():
        zero = slots.sum() * 0.0
        return args.slot_orth_weight * orth + zero, {"slot_orth": float(orth.detach().cpu()), "slot_balance": 0.0, "slot_sharp": 0.0}

    pos = F.normalize(pos_emb, dim=-1, eps=1e-8)
    logits = torch.einsum("bkd,bpd->bpk", norm_slots, pos) / args.slot_assignment_temperature
    logits = logits.masked_fill(~pos_mask.unsqueeze(-1), 0.0)
    probs = F.softmax(logits, dim=-1)
    probs = probs * pos_mask.unsqueeze(-1).float()
    pos_count = pos_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    mass = probs.sum(dim=1) / pos_count
    mass = mass.clamp_min(1e-8)
    mass_entropy = -(mass * mass.log()).sum(dim=-1)
    effective_slots = torch.minimum(
        torch.full_like(pos_count.squeeze(-1), float(slots.shape[1])),
        pos_mask.float().sum(dim=1).clamp_min(1.0),
    )
    target_entropy = effective_slots.log()
    balance = (target_entropy - mass_entropy).clamp_min(0.0).mean()
    event_entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    sharp = (event_entropy * pos_mask.float()).sum() / pos_mask.float().sum().clamp_min(1.0)
    total = args.slot_orth_weight * orth + args.slot_balance_weight * balance + args.slot_sharp_weight * sharp
    return total, {
        "slot_orth": float(orth.detach().cpu()),
        "slot_balance": float(balance.detach().cpu()),
        "slot_sharp": float(sharp.detach().cpu()),
    }


def rich_future_state_loss(
    slots: torch.Tensor,
    slot_state: torch.Tensor,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not args.enable_rich_state or "pos_state" not in batch or "neg_state" not in batch:
        zero = slots.sum() * 0.0
        return zero, {"rich_state": 0.0, "rich_item": 0.0}
    event_emb = torch.cat([batch["pos_emb"], batch["neg_emb"]], dim=1)
    event_state = torch.cat([batch["pos_state"], batch["neg_state"]], dim=1)
    event_mask = torch.cat([batch["pos_mask"], batch["neg_mask"]], dim=1)
    if not event_mask.any():
        zero = slots.sum() * 0.0
        return zero, {"rich_state": 0.0, "rich_item": 0.0}
    sim = torch.einsum("bkd,bmd->bkm", F.normalize(slots, dim=-1, eps=1e-8), F.normalize(event_emb, dim=-1, eps=1e-8))
    max_sim, assign = sim.max(dim=1)
    gather_idx = assign.unsqueeze(-1).expand(-1, -1, slot_state.shape[-1])
    pred_state = torch.gather(slot_state, 1, gather_idx)
    state_err = (pred_state - event_state).pow(2).mean(dim=-1)
    item_err = (1.0 - max_sim).clamp_min(0.0).pow(2)
    weights = event_mask.float()
    if args.rich_positive_weight != 0:
        clicked_target = event_state[..., 0]
        weights = weights * (1.0 + args.rich_positive_weight * clicked_target)
    denom = weights.sum().clamp_min(1.0)
    state_loss = (state_err * weights).sum() / denom
    item_loss = (item_err * weights).sum() / denom
    total = args.rich_state_weight * state_loss + args.rich_item_weight * item_loss
    return total, {
        "rich_state": float(state_loss.detach().cpu()),
        "rich_item": float(item_loss.detach().cpu()),
    }


def weighted_event_mean(emb: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = mask.float() * weight.float().clamp_min(0.0)
    weights = torch.where(weights > 0, weights, mask.float())
    denom = weights.sum(dim=1, keepdim=True)
    mean = (emb * weights.unsqueeze(-1)).sum(dim=1) / denom.clamp_min(1.0)
    valid = denom.squeeze(1) > 0
    return mean, valid


def consequence_target_from_batch(batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    pos_mean, pos_valid = weighted_event_mean(batch["pos_emb"], batch["pos_mask"], batch["pos_weight"])
    neg_mean, neg_valid = weighted_event_mean(batch["neg_emb"], batch["neg_mask"], batch["neg_weight"])
    target = pos_mean - args.consequence_neg_weight * neg_mean
    valid = pos_valid | neg_valid
    target = F.normalize(target, dim=-1, eps=1e-8).detach()
    return target, valid


def consequence_latent_loss(
    slots: torch.Tensor,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    target, valid = consequence_target_from_batch(batch, args)
    if not valid.any():
        zero = slots.sum() * 0.0
        return zero, {"consequence_cosine": 0.0, "consequence_valid": 0.0}
    pred = F.normalize(slots.mean(dim=1), dim=-1, eps=1e-8)
    cosine = (pred * target).sum(dim=-1)
    loss = (1.0 - cosine[valid]).mean()
    return loss, {
        "consequence_cosine": float(cosine[valid].mean().detach().cpu()),
        "consequence_valid": float(valid.float().mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate_consequence_predictor(
    model: HstuJepaRanker,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    cosines: list[np.ndarray] = []
    mses: list[np.ndarray] = []
    valid_count = 0
    total_count = 0
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        _, slots, _, _, _ = model(moved)
        if slots is None:
            continue
        target, valid = consequence_target_from_batch(moved, args)
        pred = F.normalize(slots.mean(dim=1), dim=-1, eps=1e-8)
        cosine = (pred * target).sum(dim=-1)
        mse = (pred - target).pow(2).mean(dim=-1)
        cosines.append(cosine[valid].detach().cpu().numpy())
        mses.append(mse[valid].detach().cpu().numpy())
        valid_count += int(valid.sum().detach().cpu())
        total_count += int(valid.numel())
    if not cosines or valid_count <= 0:
        return {"num_eval": 0, "valid_rate": 0.0}
    cos = np.concatenate(cosines)
    mse = np.concatenate(mses)
    return {
        "num_eval": int(valid_count),
        "valid_rate": float(valid_count / max(1, total_count)),
        "cosine_to_consequence": float(np.mean(cos)),
        "mse_to_consequence": float(np.mean(mse)),
    }


@torch.no_grad()
def predict_model_slots(
    model: HstuJepaRanker,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int,
    num_slots: int,
    emb_dim: int,
) -> np.ndarray:
    model.eval()
    slots_np = np.zeros((n_samples, num_slots, emb_dim), dtype=np.float16)
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        _, slots, _, _, _ = model(moved)
        if slots is None:
            continue
        slots_np[batch["sample_ids"]] = slots.detach().cpu().numpy().astype(np.float16)
    return slots_np


def pretrain_jepa_predictor(
    name: str,
    model: HstuJepaRanker,
    loaders: dict[str, torch.utils.data.DataLoader],
    args: argparse.Namespace,
    device: torch.device,
    teacher_model: HstuJepaRanker | None = None,
) -> dict[str, Any]:
    if teacher_model is not None:
        teacher_model.eval()
    opt = torch.optim.AdamW(model.jepa_parameters(), lr=args.jepa_pretrain_lr, weight_decay=args.weight_decay)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    initial = evaluate_reaction_predictor(model, loaders["ranker_val"], args, device)
    best = ep.primary_metric(initial, args) if initial else -float("inf")
    history: list[dict[str, Any]] = [{"epoch": 0, "loss": None, "reaction_val": initial}]
    patience = args.jepa_pretrain_patience
    print(json.dumps({name: {"jepa_pretrain_epoch": 0, "reaction_val": initial}}, ensure_ascii=False), flush=True)
    for epoch in range(1, args.jepa_pretrain_epochs + 1):
        model.train()
        losses = []
        event_losses = []
        reaction_losses = []
        consequence_losses = []
        consequence_cosines = []
        for batch in loaders["ranker_train"]:
            moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            _, slots, reaction_logits, slot_state, _ = model(moved)
            assert slots is not None and reaction_logits is not None and slot_state is not None
            teacher_scores = None
            if teacher_model is not None and args.residual_teacher_weight > 0:
                with torch.no_grad():
                    teacher_scores, _, _, _, _ = teacher_model(moved)
            event_part, event_metrics = ep.event_loss(slots, moved, "E3_event_hybrid", args, rank_scores=None)
            spec_loss, spec_parts = slot_specialization_loss(slots, moved["pos_emb"], moved["pos_mask"], args)
            rich_loss, rich_parts = rich_future_state_loss(slots, slot_state, moved, args)
            reaction_part, reaction_metrics = future_reaction_loss(reaction_logits, moved, args, teacher_scores=teacher_scores)
            consequence_part, consequence_metrics = consequence_latent_loss(slots, moved, args)
            loss = (
                args.pretrain_event_weight * (event_part + spec_loss + rich_loss)
                + args.pretrain_reaction_weight * reaction_part
                + args.pretrain_consequence_weight * consequence_part
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.jepa_parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            event_losses.append(float((event_part + spec_loss + rich_loss).detach().cpu()))
            reaction_losses.append(float(reaction_part.detach().cpu()))
            consequence_losses.append(float(consequence_part.detach().cpu()))
            consequence_cosines.append(float(consequence_metrics.get("consequence_cosine", 0.0)))
        val = evaluate_reaction_predictor(model, loaders["ranker_val"], args, device)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "event_loss": float(np.mean(event_losses)) if event_losses else float("nan"),
            "reaction_loss": float(np.mean(reaction_losses)) if reaction_losses else float("nan"),
            "consequence_loss": float(np.mean(consequence_losses)) if consequence_losses else float("nan"),
            "consequence_cosine": float(np.mean(consequence_cosines)) if consequence_cosines else 0.0,
            "reaction_val": val,
        }
        history.append(row)
        print(json.dumps({name: {"jepa_pretrain": row}}, ensure_ascii=False), flush=True)
        score_value = ep.primary_metric(val, args)
        if score_value > best:
            best = score_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.jepa_pretrain_patience
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    return {
        "history": history,
        "best_val": best,
        "split_eval": {
            split: evaluate_reaction_predictor(model, loader, args, device)
            for split, loader in loaders.items()
            if split in {"ranker_train", "ranker_val", "ranker_test"}
        },
        "consequence_split_eval": {
            split: evaluate_consequence_predictor(model, loader, args, device)
            for split, loader in loaders.items()
            if split in {"ranker_train", "ranker_val", "ranker_test"}
        },
    }


def freeze_jepa_backbone_for_rank_probe(model: HstuJepaRanker) -> list[str]:
    frozen_prefixes = (
        "vec_proj.",
        "scalar_proj.",
        "hist_age_proj.",
        "user_emb.",
        "item_id_emb.",
        "hist_type_emb.",
        "type_emb.",
        "history_stack.",
        "item_stack.",
        "future_predictor.",
    )
    frozen = []
    for name, param in model.named_parameters():
        if name.startswith(frozen_prefixes):
            param.requires_grad = False
            frozen.append(name)
    return frozen


def freeze_base_for_transition(model: HstuJepaRanker) -> list[str]:
    frozen = []
    for name, param in model.named_parameters():
        if not (name.startswith("transition_head.") or name.startswith("transition_score_head.")):
            param.requires_grad = False
            frozen.append(name)
    return frozen


def train_model(
    name: str,
    use_jepa_slots: bool,
    use_transition: bool,
    samples: list[q.Sample],
    loaders: dict[str, torch.utils.data.DataLoader],
    event_index: dict[str, np.ndarray],
    item_emb: np.ndarray,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    args: argparse.Namespace,
    teacher_model: HstuJepaRanker | None = None,
    init_model: HstuJepaRanker | None = None,
) -> tuple[HstuJepaRanker, dict[str, Any], HstuJepaRanker | None]:
    device = torch.device(args.device)
    model = HstuJepaRanker(
        emb_dim=item_emb.shape[1],
        scalar_dim=ep.EventBatchBuilder.scalar_dim,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        num_slots=args.num_slots,
        state_dim=args.rich_state_dim,
        use_jepa_slots=use_jepa_slots,
        use_item_id_emb=args.use_item_id_emb,
        slot_rank_scale=args.slot_rank_scale,
        slot_query_scale=args.slot_query_scale,
        disable_slot_self_attn=args.disable_slot_self_attn,
        transition_residual_scale=args.transition_residual_scale,
        transition_score_residual=args.transition_score_residual,
        transition_score_scale=args.transition_score_scale,
    ).to(device)
    if init_model is not None:
        model.load_state_dict(init_model.state_dict())
        print(json.dumps({name: {"initialized_from": "R0_HSTU"}}, ensure_ascii=False), flush=True)
    pretrain_payload: dict[str, Any] | None = None
    pretrained_probe_model: HstuJepaRanker | None = None
    frozen_jepa_names: list[str] = []
    frozen_transition_base_names: list[str] = []
    if use_jepa_slots and args.jepa_pretrain_epochs > 0:
        pretrain_payload = pretrain_jepa_predictor(name, model, loaders, args, device, teacher_model=teacher_model)
        pretrained_probe_model = copy.deepcopy(model).to(device)
        pretrained_probe_model.eval()
        for param in pretrained_probe_model.parameters():
            param.requires_grad = False
        if args.freeze_jepa_after_pretrain:
            frozen_jepa_names = freeze_jepa_backbone_for_rank_probe(model)
            print(json.dumps({name: {"frozen_jepa_parameters": len(frozen_jepa_names)}}, ensure_ascii=False), flush=True)
    if use_transition and args.freeze_base_for_transition:
        frozen_transition_base_names = freeze_base_for_transition(model)
        print(json.dumps({name: {"frozen_transition_base_parameters": len(frozen_transition_base_names)}}, ensure_ascii=False), flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    initial = evaluate_model(model, loaders["ranker_val"], args, device)
    initial_next = evaluate_next_round_model(model, loaders["ranker_val"], args, device, use_transition=use_transition)
    best = ep.primary_metric(initial_next if use_transition and initial_next else initial, args)
    history: list[dict[str, Any]] = [{"epoch": 0, "loss": None, "ranker_val": initial, "next_round_val": initial_next}]
    patience = args.patience
    rank_epochs = args.jepa_ranker_epochs if use_jepa_slots and args.jepa_ranker_epochs >= 0 else args.epochs
    print(json.dumps({name: {"epoch": 0, "val": initial, "next_round_val": initial_next}}, ensure_ascii=False), flush=True)
    for epoch in range(1, rank_epochs + 1):
        model.train()
        losses = []
        rank_losses = []
        next_round_losses = []
        jepa_losses = []
        reaction_losses = []
        consequence_losses = []
        consequence_cosines = []
        for batch in loaders["ranker_train"]:
            moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            score, slots, reaction_logits, slot_state, _ = model(moved)
            rank_loss = q.listwise_loss(score, moved["labels"], moved["mask"])
            current_rank_weight = args.current_rank_weight if use_transition else 1.0
            loss = current_rank_weight * rank_loss
            parts: dict[str, float] = {}
            if use_transition and args.transition_weight > 0 and "next_mask" in moved:
                next_supervised = moved["next_mask"] & moved["next_rank_supervised"].unsqueeze(1)
                if next_supervised.any():
                    next_score = model.forward_next_round(moved, use_transition=True)
                    next_loss = q.listwise_loss(next_score, moved["next_labels"], next_supervised)
                    loss = loss + args.transition_weight * next_loss
                    next_round_losses.append(float(next_loss.detach().cpu()))
            if use_jepa_slots:
                assert slots is not None and reaction_logits is not None and slot_state is not None
                jepa_loss, parts = ep.event_loss(slots, moved, "E3_event_hybrid", args, rank_scores=score)
                spec_loss, spec_parts = slot_specialization_loss(slots, moved["pos_emb"], moved["pos_mask"], args)
                rich_loss, rich_parts = rich_future_state_loss(slots, slot_state, moved, args)
                reaction_loss, reaction_parts = future_reaction_loss(reaction_logits, moved, args)
                consequence_loss, consequence_parts = consequence_latent_loss(slots, moved, args)
                jepa_loss = (
                    jepa_loss
                    + spec_loss
                    + rich_loss
                    + args.reaction_weight * reaction_loss
                    + args.consequence_weight * consequence_loss
                )
                parts.update(spec_parts)
                parts.update(rich_parts)
                parts.update(reaction_parts)
                parts.update(consequence_parts)
                loss = loss + args.jepa_weight * jepa_loss
                jepa_losses.append(float(jepa_loss.detach().cpu()))
                reaction_losses.append(float(reaction_loss.detach().cpu()))
                consequence_losses.append(float(consequence_loss.detach().cpu()))
                consequence_cosines.append(float(consequence_parts.get("consequence_cosine", 0.0)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            rank_losses.append(float(rank_loss.detach().cpu()))
        val = evaluate_model(model, loaders["ranker_val"], args, device)
        next_val = evaluate_next_round_model(model, loaders["ranker_val"], args, device, use_transition=use_transition)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "rank_loss": float(np.mean(rank_losses)) if rank_losses else float("nan"),
            "next_round_loss": float(np.mean(next_round_losses)) if next_round_losses else 0.0,
            "jepa_loss": float(np.mean(jepa_losses)) if jepa_losses else 0.0,
            "reaction_loss": float(np.mean(reaction_losses)) if reaction_losses else 0.0,
            "consequence_loss": float(np.mean(consequence_losses)) if consequence_losses else 0.0,
            "consequence_cosine": float(np.mean(consequence_cosines)) if consequence_cosines else 0.0,
            "ranker_val": val,
            "next_round_val": next_val,
        }
        history.append(row)
        print(json.dumps({name: row}, ensure_ascii=False), flush=True)
        score_value = ep.primary_metric(next_val if use_transition and next_val else val, args)
        if score_value > best:
            best = score_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    split_eval = {
        split: evaluate_model(model, loader, args, device)
        for split, loader in loaders.items()
        if split in {"ranker_train", "ranker_val", "ranker_test"}
    }
    next_round_split_eval = {
        split: evaluate_next_round_model(model, loader, args, device, use_transition=use_transition)
        for split, loader in loaders.items()
        if split in {"ranker_train", "ranker_val", "ranker_test"}
    }
    fair_next_modes = ["stale", "reencode_true"]
    if use_transition:
        fair_next_modes.extend(["transition_true", "transition_pred"])
    fair_next_round_eval = {
        mode: evaluate_next_round_model(
            model,
            loaders["ranker_test"],
            args,
            device,
            use_transition=mode.startswith("transition"),
            state_mode=mode,
        )
        for mode in fair_next_modes
    }
    rollout_modes = ["direct_all", "reencode_true"]
    if use_transition:
        rollout_modes.extend(["transition_true", "transition_pred"])
    rollout_eval = {
        mode: evaluate_rollout_model(model, loaders["ranker_test"], args, device, state_mode=mode)
        for mode in rollout_modes
    }
    out: dict[str, Any] = {
        "history": history,
        "split_eval": split_eval,
        "next_round_split_eval": next_round_split_eval,
        "fair_next_round_eval": fair_next_round_eval,
        "rollout_eval": rollout_eval,
        "best_val": best,
        "use_jepa_slots": use_jepa_slots,
        "use_transition": use_transition,
        "rank_epochs": rank_epochs,
    }
    if use_jepa_slots and rank_epochs <= 0 and pretrain_payload is not None:
        out["skip_ranker_table"] = True
    if pretrain_payload is not None:
        out["jepa_pretrain"] = pretrain_payload
    if frozen_jepa_names:
        out["frozen_jepa_parameters"] = frozen_jepa_names
    if frozen_transition_base_names:
        out["frozen_transition_base_parameters"] = frozen_transition_base_names
    if use_jepa_slots:
        slots_np = predict_model_slots(model, loaders["ranker_test"], device, len(samples), args.num_slots, item_emb.shape[1])
        out["ranker_test_event_metrics"] = ep.sample_event_metrics(samples, "ranker_test", item_emb, event_index, slots_np)
        out["reaction_split_eval"] = {
            split: evaluate_reaction_predictor(model, loader, args, device)
            for split, loader in loaders.items()
            if split in {"ranker_train", "ranker_val", "ranker_test"}
        }
        out["consequence_split_eval"] = {
            split: evaluate_consequence_predictor(model, loader, args, device)
            for split, loader in loaders.items()
            if split in {"ranker_train", "ranker_val", "ranker_test"}
        }
    return model, out, pretrained_probe_model


def event_summary_from_values(values: np.ndarray) -> dict[str, Any]:
    return {
        "samples": int(values[0]),
        "pos_events": int(values[1]),
        "neg_events": int(values[2]),
        "empty_pos": int(values[3]),
        "empty_neg": int(values[4]),
        "avg_pos_per_sample": float(values[1] / max(1, values[0])),
        "avg_neg_per_sample": float(values[2] / max(1, values[0])),
        "rank_candidates": int(values[5]) if len(values) > 5 else 0,
        "rank_positive": int(values[6]) if len(values) > 6 else 0,
        "rank_empty": int(values[7]) if len(values) > 7 else 0,
        "rank_truncated": int(values[8]) if len(values) > 8 else 0,
        "avg_rank_candidates_per_sample": float(values[5] / max(1, values[0])) if len(values) > 5 else 0.0,
    }


def table_row_from_metrics(name: str, test: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": name,
        "engagement_ndcg@10": test.get("engagement_ndcg@10", test.get("ndcg@10")),
        "ndcg@10": test.get("ndcg@10"),
        "hr@10": test.get("hr@10", test.get("reaction_hr@10")),
        "ndcg@50": test.get("ndcg@50", test.get("reaction_ndcg@50")),
        "hr@50": test.get("hr@50", test.get("reaction_hr@50")),
        "mrr": test.get("mrr"),
        "hit@1": test.get("hit@1", test.get("hit1")),
    }


def table_row_from_payload(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    current = payload["split_eval"]["ranker_test"]
    row = table_row_from_metrics(name, current)
    next_test = payload.get("next_round_split_eval", {}).get("ranker_test", {})
    if next_test:
        row.update(
            {
                "current_ndcg@10": current.get("ndcg@10"),
                "next_ndcg@10": next_test.get("ndcg@10"),
                "next_hr@10": next_test.get("hr@10"),
                "next_ndcg@50": next_test.get("ndcg@50"),
                "next_mrr": next_test.get("mrr"),
                "next_hit@1": next_test.get("hit@1", next_test.get("hit1")),
            }
        )
    return row


def fair_row_from_payload(name: str, mode: str, payload: dict[str, Any]) -> dict[str, Any]:
    current = payload["split_eval"]["ranker_test"]
    next_test = payload.get("fair_next_round_eval", {}).get(mode, {})
    rollout = payload.get("rollout_eval", {}).get(mode, {})
    steps = rollout.get("steps", {}) if isinstance(rollout, dict) else {}
    avg = rollout.get("avg", {}) if isinstance(rollout, dict) else {}
    row = {
        "model": name,
        "current_ndcg@10": current.get("ndcg@10"),
        "next_ndcg@10": next_test.get("ndcg@10"),
        "next_mrr": next_test.get("mrr"),
        "next_hr@10": next_test.get("hr@10"),
        "rollout_avg_ndcg@10": avg.get("ndcg@10"),
        "rollout_avg_mrr": avg.get("mrr"),
    }
    for step in ["step+1", "step+2", "step+3"]:
        if step in steps:
            row[f"{step}_ndcg@10"] = steps[step].get("ndcg@10")
    return row


def compact_table(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    setting = results.get("setting", {})
    consequence_enabled = (
        float(setting.get("consequence_weight", 0.0)) > 0.0
        or float(setting.get("pretrain_consequence_weight", 0.0)) > 0.0
        or str(setting.get("target_mode", "")) == "current_future"
    )
    for name, payload in results["models"].items():
        display_name = "E1_HSTU_CONSEQUENCE_JEPA" if name == "E1_HSTU_JEPA" and consequence_enabled else name
        if not payload.get("skip_ranker_table"):
            rows.append(table_row_from_payload(display_name, payload))
            fair_modes: list[tuple[str, str]] = []
            if name == "R0_HSTU":
                fair_modes = [("R0_STALE", "stale"), ("HSTU_REENCODE_TRUE", "reencode_true"), ("HSTU_DIRECT_ALL", "direct_all")]
            elif name == "E5_HSTU_TRANSITION":
                fair_modes = [("WAM_TRANSITION_TRUE", "transition_true"), ("WAM_TRANSITION_PRED", "transition_pred")]
            for row_name, mode in fair_modes:
                if mode in payload.get("fair_next_round_eval", {}) or mode in payload.get("rollout_eval", {}):
                    rows.append(fair_row_from_payload(row_name, mode, payload))
        pretrain_test = payload.get("jepa_pretrain", {}).get("split_eval", {}).get("ranker_test")
        if pretrain_test and name == "E1_HSTU_JEPA":
            probe_name = "E4_CONSEQUENCE_REACTION_PRIMARY" if consequence_enabled else "E4_JEPA_RICH_REACTION_PRIMARY"
            rows.append(table_row_from_metrics(probe_name, pretrain_test))
    for name, payload in results.get("blends", {}).items():
        rows.append(table_row_from_metrics(name, payload["blend_test"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="EB-NeRD HSTU action prediction plus JEPA-style future consequence experiment.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qwen-raw-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples-per-split", type=int, default=1000)
    parser.add_argument("--target-mode", choices=["current_future", "future_only"], default="future_only")
    parser.add_argument("--include-current-neg", action="store_true", default=True)
    parser.add_argument("--rank-task", choices=["current_click", "future_engagement"], default="future_engagement")
    parser.add_argument("--future-horizon-hours", type=float, default=6.0)
    parser.add_argument("--max-future-impressions", type=int, default=8)
    parser.add_argument("--max-events-per-impression", type=int, default=20)
    parser.add_argument("--max-rank-candidates", type=int, default=256)
    parser.add_argument("--max-next-rank-candidates", type=int, default=128)
    parser.add_argument("--next-request-max-gap-hours", type=float, default=24.0)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--max-pos-events", type=int, default=16)
    parser.add_argument("--max-neg-events", type=int, default=32)
    parser.add_argument("--max-history-tokens", type=int, default=128)
    parser.add_argument("--user-profile-mode", choices=["none", "history_summary"], default="none")
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--jepa-ranker-epochs", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--use-item-id-emb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--primary-rank-metric", choices=["ndcg@10", "engagement_ndcg@10"], default="engagement_ndcg@10")
    parser.add_argument("--hstu-eval-ks", type=int, nargs="+", default=[10, 50, 200])
    parser.add_argument("--run-models", default="R0_HSTU,E1_HSTU_JEPA")
    parser.add_argument("--enable-reaction-blend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reaction-blend-lambda-grid", default="-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,3,5")
    parser.add_argument("--enable-rich-fusion-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fusion-epochs", type=int, default=5)
    parser.add_argument("--fusion-patience", type=int, default=2)
    parser.add_argument("--fusion-lr", type=float, default=1e-3)
    parser.add_argument("--fusion-hidden-dim", type=int, default=64)
    parser.add_argument("--fusion-residual-scale", type=float, default=0.5)
    parser.add_argument("--enable-lgbm-reranker", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lgbm-include-scalar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lgbm-binary-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lgbm-gain-scale", type=float, default=10.0)
    parser.add_argument("--lgbm-max-gain-label", type=int, default=20)
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.03)
    parser.add_argument("--lgbm-num-leaves", type=int, default=31)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=50)
    parser.add_argument("--lgbm-subsample", type=float, default=0.90)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.90)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=1.0)
    parser.add_argument("--lgbm-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--lgbm-n-jobs", type=int, default=8)
    parser.add_argument("--slot-rank-scale", type=float, default=1.0)
    parser.add_argument("--slot-query-scale", type=float, default=1.0)
    parser.add_argument("--disable-slot-self-attn", action="store_true")
    parser.add_argument("--current-rank-weight", type=float, default=1.0)
    parser.add_argument("--transition-weight", type=float, default=0.0)
    parser.add_argument("--transition-residual-scale", type=float, default=0.5)
    parser.add_argument("--transition-score-residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--transition-score-scale", type=float, default=0.5)
    parser.add_argument("--transition-init-from-r0", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-base-for-transition", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jepa-weight", type=float, default=0.25)
    parser.add_argument("--jepa-pretrain-epochs", type=int, default=0)
    parser.add_argument("--jepa-pretrain-patience", type=int, default=1)
    parser.add_argument("--jepa-pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--pretrain-event-weight", type=float, default=1.0)
    parser.add_argument("--pretrain-reaction-weight", type=float, default=1.0)
    parser.add_argument("--pretrain-consequence-weight", type=float, default=0.0)
    parser.add_argument("--reaction-weight", type=float, default=1.0)
    parser.add_argument("--consequence-weight", type=float, default=0.0)
    parser.add_argument("--consequence-neg-weight", type=float, default=0.35)
    parser.add_argument("--reaction-bce-weight", type=float, default=1.0)
    parser.add_argument("--reaction-listwise-weight", type=float, default=1.0)
    parser.add_argument("--reaction-pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--residual-teacher-weight", type=float, default=0.0)
    parser.add_argument("--enable-rich-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--horizon-bucket-hours", default="1,3,6")
    parser.add_argument("--rich-state-weight", type=float, default=2.0)
    parser.add_argument("--rich-item-weight", type=float, default=0.5)
    parser.add_argument("--rich-positive-weight", type=float, default=1.0)
    parser.add_argument("--freeze-jepa-after-pretrain", action="store_true")
    parser.add_argument("--event-temperature", type=float, default=0.07)
    parser.add_argument("--bpr-temperature", type=float, default=0.10)
    parser.add_argument("--candidate-temperature", type=float, default=0.07)
    parser.add_argument("--candidate-bce-temperature", type=float, default=0.10)
    parser.add_argument("--nce-weight", type=float, default=1.0)
    parser.add_argument("--slot-nce-weight", type=float, default=1.0)
    parser.add_argument("--assignment-weight", type=float, default=1.0)
    parser.add_argument("--bpr-weight", type=float, default=0.5)
    parser.add_argument("--candidate-kl-weight", type=float, default=0.5)
    parser.add_argument("--candidate-bce-weight", type=float, default=1.0)
    parser.add_argument("--rank-label-weight", type=float, default=0.0)
    parser.add_argument("--rank-head-weight", type=float, default=0.0)
    parser.add_argument("--diversity-weight", type=float, default=0.2)
    parser.add_argument("--diversity-margin", type=float, default=0.1)
    parser.add_argument("--slot-assignment-temperature", type=float, default=0.10)
    parser.add_argument("--slot-orth-weight", type=float, default=0.0)
    parser.add_argument("--slot-balance-weight", type=float, default=0.0)
    parser.add_argument("--slot-sharp-weight", type=float, default=0.0)
    args = parser.parse_args()
    args.rich_state_dim = rich_state_dim(args)

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    official = official_hstu_smoke()
    print(json.dumps({"stage": "load_samples", "path": args.samples_pkl}, ensure_ascii=False), flush=True)
    all_samples, prepare = load_samples(Path(args.samples_pkl))
    print(json.dumps({"stage": "select_samples", "max_samples_per_split": args.max_samples_per_split}, ensure_ascii=False), flush=True)
    samples, cache_idx = ep.select_ranker_samples(all_samples, args.max_samples_per_split)
    print(json.dumps({"stage": "load_qwen_raw_cache", "path": args.qwen_raw_cache}, ensure_ascii=False), flush=True)
    cache = np.load(args.qwen_raw_cache)
    article_ids = cache["article_ids"].astype(np.int64)
    idx_to_article = article_ids.astype(np.int64)
    item_emb = cache["article_emb"].astype(np.float32)
    article_to_idx = {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}
    source_emb = np.zeros((len(samples), item_emb.shape[1]), dtype=np.float32)
    data_dir = Path(args.data_dir)
    print(json.dumps({"stage": "load_articles_and_full_index", "data_dir": str(data_dir)}, ensure_ascii=False), flush=True)
    articles_df = pd.read_parquet(data_dir / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    articles_by_id = q.article_lookup(articles_df)
    full_index = q.FullSignalIndex(data_dir)
    users = sorted({s.user_id for s in samples})
    items = sorted(int(aid) for aid in article_ids.tolist())
    user_to_idx = {uid: idx + 1 for idx, uid in enumerate(users)}
    item_to_idx = {aid: idx + 1 for idx, aid in enumerate(items)}

    print("building event index", flush=True)
    event_index = ep.build_event_index(samples, article_to_idx, full_index, args)
    next_index, next_round_summary = build_next_round_index(samples, article_to_idx, full_index, args)
    event_index.update(next_index)
    rich_state_meta: dict[str, Any] = {"enabled": False, "state_dim": 0}
    if args.enable_rich_state:
        rich_state_meta = add_rich_event_state_index(samples, article_to_idx, full_index, event_index, args)
        patch_event_batch_builder_for_rich_state()
    patch_event_batch_builder_for_next_round()
    summary_values = event_index.pop("summary")
    event_summary = event_summary_from_values(summary_values)
    print(
        json.dumps(
            {"official_hstu_smoke": official, "event_summary": event_summary, "next_round_summary": next_round_summary},
            ensure_ascii=False,
        ),
        flush=True,
    )
    loaders = ep.make_loaders(
        samples,
        article_to_idx,
        item_emb,
        source_emb,
        event_index,
        articles_by_id,
        user_to_idx,
        item_to_idx,
        idx_to_article,
        args,
        ["ranker_train", "ranker_val", "ranker_test"],
    )
    results: dict[str, Any] = {
        "official_hstu_smoke": official,
        "prepare": prepare,
        "setting": vars(args),
        "split_counts": dict(Counter(s.split for s in samples)),
        "event_summary": event_summary,
        "next_round_summary": next_round_summary,
        "rich_state": rich_state_meta,
        "note": "No LLM UIH source token is used; source_emb is all zeros. Item/content embeddings come from the cached article embedding table.",
        "models": {},
    }
    q.write_json(out_dir / "results.partial.json", results)

    model_plan = {
        "R0_HSTU": {"use_jepa": False, "use_transition": False},
        "E1_HSTU_JEPA": {"use_jepa": True, "use_transition": False},
        "E5_HSTU_TRANSITION": {"use_jepa": False, "use_transition": True},
    }
    trained_models: dict[str, HstuJepaRanker] = {}
    pretrained_probe_models: dict[str, HstuJepaRanker] = {}
    requested_models = [m.strip() for m in args.run_models.split(",") if m.strip()]
    for name in requested_models:
        if name not in model_plan:
            raise ValueError(f"unknown model {name}; expected one of {sorted(model_plan)}")
        use_jepa = bool(model_plan[name]["use_jepa"])
        use_transition = bool(model_plan[name]["use_transition"])
        print(f"training {name}", flush=True)
        teacher_model = trained_models.get("R0_HSTU") if use_jepa and args.residual_teacher_weight > 0 else None
        init_model = trained_models.get("R0_HSTU") if use_transition and args.transition_init_from_r0 else None
        model, payload, pretrained_probe_model = train_model(
            name,
            use_jepa,
            use_transition,
            samples,
            loaders,
            event_index,
            item_emb,
            user_to_idx,
            item_to_idx,
            args,
            teacher_model=teacher_model,
            init_model=init_model,
        )
        trained_models[name] = model
        if pretrained_probe_model is not None:
            pretrained_probe_models[name] = pretrained_probe_model
        results["models"][name] = payload
        results["table"] = compact_table(results)
        q.write_json(out_dir / "results.partial.json", results)
        print(json.dumps({"table": results["table"]}, ensure_ascii=False, indent=2), flush=True)
    if (
        args.enable_reaction_blend
        and "R0_HSTU" in trained_models
        and "E1_HSTU_JEPA" in pretrained_probe_models
    ):
        print("evaluating E2_R0_PLUS_JEPA_REACTION blend", flush=True)
        blend = evaluate_reaction_blend(
            trained_models["R0_HSTU"],
            pretrained_probe_models["E1_HSTU_JEPA"],
            loaders,
            args,
            torch.device(args.device),
        )
        results.setdefault("blends", {})["E2_R0_PLUS_JEPA_REACTION"] = blend
        results["table"] = compact_table(results)
        q.write_json(out_dir / "results.partial.json", results)
        print(json.dumps({"table": results["table"], "blend": blend}, ensure_ascii=False, indent=2), flush=True)
    if (
        args.enable_rich_fusion_head
        and args.enable_rich_state
        and "R0_HSTU" in trained_models
        and "E1_HSTU_JEPA" in pretrained_probe_models
    ):
        print("training E3_R0_PLUS_RICH_JEPA_FUSION_HEAD", flush=True)
        fusion = train_fusion_head(
            "E3_R0_PLUS_RICH_JEPA_FUSION_HEAD",
            trained_models["R0_HSTU"],
            pretrained_probe_models["E1_HSTU_JEPA"],
            loaders,
            args,
            torch.device(args.device),
        )
        results.setdefault("blends", {})["E3_R0_PLUS_RICH_JEPA_FUSION_HEAD"] = fusion
        results["table"] = compact_table(results)
        q.write_json(out_dir / "results.partial.json", results)
        print(json.dumps({"table": results["table"], "fusion": fusion}, ensure_ascii=False, indent=2), flush=True)
    if (
        args.enable_lgbm_reranker
        and "R0_HSTU" in trained_models
        and "E1_HSTU_JEPA" in pretrained_probe_models
    ):
        print("training E5_LGBM_R0_RICH_JEPA_RERANKER", flush=True)
        lgbm_payload = train_lgbm_rich_jepa_reranker(
            "E5_LGBM_R0_RICH_JEPA_RERANKER",
            trained_models["R0_HSTU"],
            pretrained_probe_models["E1_HSTU_JEPA"],
            loaders,
            args,
            torch.device(args.device),
        )
        results.setdefault("blends", {})["E5_LGBM_R0_RICH_JEPA_RERANKER"] = lgbm_payload
        results["table"] = compact_table(results)
        q.write_json(out_dir / "results.partial.json", results)
        print(json.dumps({"table": results["table"], "lgbm": lgbm_payload}, ensure_ascii=False, indent=2), flush=True)
    q.write_json(out_dir / "results.json", results)
    print(json.dumps({"done": str(out_dir / "results.json"), "table": results["table"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
