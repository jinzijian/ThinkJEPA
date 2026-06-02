#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ebnerd_pipeline import article_lookup, first_example, standalone_jepa_eval, train_variant, write_json


@dataclass
class Sample:
    sample_id: int
    impression_id: int
    user_id: int
    time: pd.Timestamp
    source_split: str
    split: str
    current_inview: list[int]
    current_clicked: list[int]
    past_shown: list[int]
    past_clicked: list[int]
    past_not_clicked: list[int]
    future_shown: list[int]
    future_clicked: list[int]
    future_not_clicked: list[int]
    context: dict
    z_current: np.ndarray
    z_current_set: np.ndarray
    z_future_target: np.ndarray
    z_prior: np.ndarray | None = None
    delta_z: np.ndarray | None = None


KIND_PAD = 0
KIND_HISTORY_CLICK = 1
KIND_PAST_CLICK = 2
KIND_PAST_NOT_CLICK = 3
KIND_CURRENT_CANDIDATE = 4
KIND_FUTURE_CLICK = 5
KIND_FUTURE_NOT_CLICK = 6

FEEDBACK_PAD = 0
FEEDBACK_CLICK = 1
FEEDBACK_NOT_CLICK = 2
FEEDBACK_CANDIDATE = 3


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        out = pd.isna(value)
        if isinstance(out, (bool, np.bool_)):
            return bool(out)
    except (TypeError, ValueError):
        pass
    return False


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def clean_id(value: Any) -> int | None:
    if is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_float(value: Any, default: float = 0.0) -> float:
    if is_missing(value):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def ids(values: Iterable[Any]) -> list[int]:
    return [x for x in (clean_id(v) for v in as_list(values)) if x is not None]


def unit(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= 1e-8:
        return x.astype(np.float32)
    return (x / norm).astype(np.float32)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / den)


def load_cached_samples(path: Path) -> tuple[list[Sample], dict]:
    with path.open("rb") as f:
        packed = pickle.load(f)
    return packed["samples"], packed.get("prepare_summary", {})


def load_article_embeddings(path: Path) -> tuple[dict[int, int], np.ndarray]:
    packed = np.load(path)
    article_ids = packed["article_ids"].astype(np.int64)
    emb = packed["embeddings"].astype(np.float32)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    # 0 is reserved for padding in the neural model.
    article_to_idx = {int(aid): idx + 1 for idx, aid in enumerate(article_ids.tolist())}
    emb_table = np.vstack([np.zeros((1, emb.shape[1]), dtype=np.float32), emb]).astype(np.float32)
    return article_to_idx, emb_table


def load_behaviors(data_dir: Path) -> dict[int, list[Any]]:
    frames = []
    for split in ["train", "validation"]:
        frame = pd.read_parquet(data_dir / split / "behaviors.parquet").copy()
        frame["source_split"] = split
        frames.append(frame)
    behaviors = pd.concat(frames, ignore_index=True)
    behaviors["impression_time"] = pd.to_datetime(behaviors["impression_time"])
    by_user: dict[int, list[Any]] = {}
    for user_id, group in behaviors.sort_values(["user_id", "impression_time", "impression_id"]).groupby("user_id", sort=False):
        by_user[int(user_id)] = list(group.itertuples(index=False))
    return by_user


def load_history(data_dir: Path) -> dict[int, list[dict]]:
    by_user: dict[int, list[dict]] = defaultdict(list)
    for split in ["train", "validation"]:
        path = data_dir / split / "history.parquet"
        if not path.exists():
            continue
        history = pd.read_parquet(path)
        for row in history.itertuples(index=False):
            user_id = clean_id(getattr(row, "user_id"))
            if user_id is None:
                continue
            times = as_list(getattr(row, "impression_time_fixed"))
            article_ids = as_list(getattr(row, "article_id_fixed"))
            read_times = as_list(getattr(row, "read_time_fixed"))
            scrolls = as_list(getattr(row, "scroll_percentage_fixed"))
            for idx, aid in enumerate(article_ids):
                clean = clean_id(aid)
                if clean is None:
                    continue
                time = pd.to_datetime(times[idx]) if idx < len(times) else pd.NaT
                by_user[user_id].append(
                    {
                        "time": time,
                        "article_id": clean,
                        "read_time": clean_float(read_times[idx]) if idx < len(read_times) else 0.0,
                        "scroll": clean_float(scrolls[idx]) if idx < len(scrolls) else 0.0,
                    }
                )
    for user_id in list(by_user):
        by_user[user_id] = sorted(by_user[user_id], key=lambda ev: (ev["time"], ev["article_id"]))
    return by_user


def session_bucket(session_id: Any, buckets: int) -> int:
    sid = clean_id(session_id)
    if sid is None:
        return 0
    return 1 + (sid % buckets)


def event_nums(event_time: pd.Timestamp | None, current_time: pd.Timestamp, read_time: float, scroll: float) -> list[float]:
    if event_time is None or pd.isna(event_time):
        gap_hours = 0.0
    else:
        gap_hours = abs((current_time - event_time).total_seconds()) / 3600.0
    return [
        math.log1p(gap_hours),
        math.log1p(max(0.0, read_time)),
        max(0.0, min(1.0, scroll / 100.0)),
        1.0 / math.sqrt(1.0 + gap_hours),
    ]


def fill_event(
    arrays: dict[str, np.ndarray],
    row: int,
    pos: int,
    article_idx: int,
    kind: int,
    feedback: int,
    sess: int,
    nums: Sequence[float],
) -> None:
    arrays["article_idx"][row, pos] = article_idx
    arrays["kind"][row, pos] = kind
    arrays["feedback"][row, pos] = feedback
    arrays["session"][row, pos] = sess
    arrays["nums"][row, pos] = np.asarray(nums, dtype=np.float32)


def make_event_tuple(
    aid: int,
    kind: int,
    feedback: int,
    sess: int,
    event_time: pd.Timestamp | None,
    current_time: pd.Timestamp,
    read_time: float,
    scroll: float,
    article_to_idx: dict[int, int],
) -> tuple[int, int, int, int, list[float]] | None:
    article_idx = article_to_idx.get(aid)
    if article_idx is None:
        return None
    return (article_idx, kind, feedback, sess, event_nums(event_time, current_time, read_time, scroll))


def clicked_read_scroll(row: Any, clicked_ids: Sequence[int], aid: int, use_next: bool) -> tuple[float, float]:
    if aid not in set(clicked_ids):
        return 0.0, 0.0
    read_col = "next_read_time" if use_next else "read_time"
    scroll_col = "next_scroll_percentage" if use_next else "scroll_percentage"
    return clean_float(getattr(row, read_col, 0.0)), clean_float(getattr(row, scroll_col, 0.0))


def add_behavior_row_events(
    events: list[tuple[int, int, int, int, list[float]]],
    row: Any,
    current_time: pd.Timestamp,
    article_to_idx: dict[int, int],
    session_buckets: int,
    clicked_kind: int,
    not_clicked_kind: int,
    use_next: bool,
) -> None:
    inview = ids(getattr(row, "article_ids_inview"))
    clicked = ids(getattr(row, "article_ids_clicked"))
    clicked_set = set(clicked)
    event_time = pd.Timestamp(getattr(row, "impression_time"))
    sess = session_bucket(getattr(row, "session_id", None), session_buckets)
    for aid in inview:
        if aid in clicked_set:
            read, scroll = clicked_read_scroll(row, clicked, aid, use_next)
            ev = make_event_tuple(aid, clicked_kind, FEEDBACK_CLICK, sess, event_time, current_time, read, scroll, article_to_idx)
        else:
            ev = make_event_tuple(aid, not_clicked_kind, FEEDBACK_NOT_CLICK, sess, event_time, current_time, 0.0, 0.0, article_to_idx)
        if ev is not None:
            events.append(ev)


def build_sequence_arrays(
    samples: list[Sample],
    data_dir: Path,
    article_to_idx: dict[int, int],
    emb_table: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    by_user = load_behaviors(data_dir)
    histories = load_history(data_dir)
    row_index_by_user = {
        user_id: {int(getattr(row, "impression_id")): idx for idx, row in enumerate(rows)}
        for user_id, rows in by_user.items()
    }
    n = len(samples)
    source = {
        "article_idx": np.zeros((n, args.max_source_events), dtype=np.int32),
        "kind": np.zeros((n, args.max_source_events), dtype=np.int16),
        "feedback": np.zeros((n, args.max_source_events), dtype=np.int16),
        "session": np.zeros((n, args.max_source_events), dtype=np.int16),
        "nums": np.zeros((n, args.max_source_events, 4), dtype=np.float32),
    }
    target = {
        "article_idx": np.zeros((n, args.max_target_events), dtype=np.int32),
        "kind": np.zeros((n, args.max_target_events), dtype=np.int16),
        "feedback": np.zeros((n, args.max_target_events), dtype=np.int16),
        "session": np.zeros((n, args.max_target_events), dtype=np.int16),
        "nums": np.zeros((n, args.max_target_events, 4), dtype=np.float32),
    }
    anchors = np.zeros((n, emb_table.shape[1]), dtype=np.float32)
    stats = Counter()
    for out_idx, sample in enumerate(samples):
        rows = by_user.get(sample.user_id, [])
        idx = row_index_by_user.get(sample.user_id, {}).get(sample.impression_id)
        if idx is None:
            stats["missing_current_row"] += 1
            continue
        current_row = rows[idx]
        current_time = pd.Timestamp(getattr(current_row, "impression_time"))
        source_events: list[tuple[int, int, int, int, list[float]]] = []
        history_events = [ev for ev in histories.get(sample.user_id, []) if ev["time"] < current_time]
        for ev in history_events[-args.max_history_events:]:
            built = make_event_tuple(
                ev["article_id"],
                KIND_HISTORY_CLICK,
                FEEDBACK_CLICK,
                0,
                ev["time"],
                current_time,
                ev["read_time"],
                ev["scroll"],
                article_to_idx,
            )
            if built is not None:
                source_events.append(built)
        for past in rows[max(0, idx - args.max_past_impressions):idx]:
            add_behavior_row_events(
                source_events,
                past,
                current_time,
                article_to_idx,
                args.session_buckets,
                KIND_PAST_CLICK,
                KIND_PAST_NOT_CLICK,
                use_next=False,
            )
        current_events = []
        current_sess = session_bucket(getattr(current_row, "session_id", None), args.session_buckets)
        for aid in ids(getattr(current_row, "article_ids_inview"))[:args.max_current_events]:
            ev = make_event_tuple(
                aid,
                KIND_CURRENT_CANDIDATE,
                FEEDBACK_CANDIDATE,
                current_sess,
                current_time,
                current_time,
                0.0,
                0.0,
                article_to_idx,
            )
            if ev is not None:
                current_events.append(ev)
        keep_past = max(0, args.max_source_events - len(current_events))
        source_events = source_events[-keep_past:] + current_events
        for pos, ev in enumerate(source_events[:args.max_source_events]):
            fill_event(source, out_idx, pos, *ev)
        stats["source_events"] += len(source_events[:args.max_source_events])

        target_events: list[tuple[int, int, int, int, list[float]]] = []
        future_rows = []
        horizon_end = current_time + pd.Timedelta(hours=args.horizon_hours)
        for fut in rows[idx + 1:]:
            fut_time = pd.Timestamp(getattr(fut, "impression_time"))
            if fut_time > horizon_end:
                break
            future_rows.append(fut)
            if len(future_rows) >= args.max_future_impressions:
                break
        for fut in future_rows:
            add_behavior_row_events(
                target_events,
                fut,
                current_time,
                article_to_idx,
                args.session_buckets,
                KIND_FUTURE_CLICK,
                KIND_FUTURE_NOT_CLICK,
                use_next=True,
            )
        for pos, ev in enumerate(target_events[:args.max_target_events]):
            fill_event(target, out_idx, pos, *ev)
        stats["target_events"] += len(target_events[:args.max_target_events])
        anchor = np.zeros(emb_table.shape[1], dtype=np.float32)
        for article_idx, kind, _, _, nums in target_events[:args.max_target_events]:
            log_gap, log_read, scroll01, decay = nums
            if kind == KIND_FUTURE_CLICK:
                weight = (1.0 + 0.12 * log_read + 0.35 * scroll01) * decay
            else:
                weight = -0.28 * decay
            anchor += float(weight) * emb_table[article_idx]
        anchors[out_idx] = unit(anchor)
    stats["samples"] = n
    stats["avg_source_events"] = stats["source_events"] / max(1, n)
    stats["avg_target_events"] = stats["target_events"] / max(1, n)
    return {"source": source, "target": target, "anchor": anchors, "stats": dict(stats)}


class SequenceDataset(Dataset):
    def __init__(self, indices: Sequence[int], arrays: dict[str, Any]) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.arrays = arrays

    def __len__(self) -> int:
        return int(len(self.indices))

    def _pack(self, side: str, idx: int) -> dict[str, torch.Tensor]:
        block = self.arrays[side]
        return {
            f"{side}_article_idx": torch.from_numpy(block["article_idx"][idx].astype(np.int64)),
            f"{side}_kind": torch.from_numpy(block["kind"][idx].astype(np.int64)),
            f"{side}_feedback": torch.from_numpy(block["feedback"][idx].astype(np.int64)),
            f"{side}_session": torch.from_numpy(block["session"][idx].astype(np.int64)),
            f"{side}_nums": torch.from_numpy(block["nums"][idx].astype(np.float32)),
        }

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        out = {}
        out.update(self._pack("source", idx))
        out.update(self._pack("target", idx))
        out["anchor"] = torch.from_numpy(self.arrays["anchor"][idx].astype(np.float32))
        out["sample_index"] = torch.tensor(idx, dtype=torch.long)
        return out


class EventSequenceEncoder(nn.Module):
    def __init__(
        self,
        emb_table: np.ndarray,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        max_events: int,
        session_buckets: int,
    ) -> None:
        super().__init__()
        self.article_emb = nn.Embedding.from_pretrained(torch.from_numpy(emb_table), freeze=True, padding_idx=0)
        self.item_proj = nn.Linear(emb_table.shape[1], hidden_dim)
        self.kind_emb = nn.Embedding(16, hidden_dim)
        self.feedback_emb = nn.Embedding(8, hidden_dim)
        self.session_emb = nn.Embedding(session_buckets + 1, hidden_dim)
        self.pos_emb = nn.Embedding(max_events + 1, hidden_dim)
        self.num_proj = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        article_idx: torch.Tensor,
        kind: torch.Tensor,
        feedback: torch.Tensor,
        session: torch.Tensor,
        nums: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len = article_idx.shape
        pos = torch.arange(seq_len, device=article_idx.device).unsqueeze(0).expand(batch, -1)
        x = (
            self.item_proj(self.article_emb(article_idx))
            + self.kind_emb(kind)
            + self.feedback_emb(feedback)
            + self.session_emb(torch.clamp(session, min=0, max=self.session_emb.num_embeddings - 1))
            + self.pos_emb(pos)
            + self.num_proj(nums)
        )
        cls = self.cls.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pad = article_idx.eq(0)
        cls_pad = torch.zeros((batch, 1), dtype=torch.bool, device=article_idx.device)
        mask = torch.cat([cls_pad, pad], dim=1)
        h = self.encoder(x, src_key_padding_mask=mask)[:, 0]
        return self.norm(h)


class SequenceJepa(nn.Module):
    def __init__(self, emb_table: np.ndarray, args: argparse.Namespace) -> None:
        super().__init__()
        self.source_encoder = EventSequenceEncoder(
            emb_table,
            args.hidden_dim,
            args.layers,
            args.heads,
            args.ff_dim,
            args.dropout,
            args.max_source_events,
            args.session_buckets,
        )
        self.target_encoder = EventSequenceEncoder(
            emb_table,
            args.hidden_dim,
            args.layers,
            args.heads,
            args.ff_dim,
            args.dropout,
            args.max_target_events,
            args.session_buckets,
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, emb_table.shape[1]),
        )
        self.target_projector = nn.Sequential(
            nn.LayerNorm(args.hidden_dim),
            nn.Linear(args.hidden_dim, emb_table.shape[1]),
        )

    def encode_source(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.source_encoder(
            batch["source_article_idx"],
            batch["source_kind"],
            batch["source_feedback"],
            batch["source_session"],
            batch["source_nums"],
        )
        return F.normalize(self.predictor(h), dim=-1)

    def encode_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.target_encoder(
            batch["target_article_idx"],
            batch["target_kind"],
            batch["target_feedback"],
            batch["target_session"],
            batch["target_nums"],
        )
        return F.normalize(self.target_projector(h), dim=-1)


def split_indices(samples: list[Sample]) -> tuple[list[int], list[int]]:
    train = [idx for idx, s in enumerate(samples) if s.split == "jepa_train"]
    val = [idx for idx, s in enumerate(samples) if s.split == "jepa_val"]
    if len(train) < 10 or len(val) < 10:
        raise RuntimeError("Not enough JEPA train/val samples.")
    return train, val


def make_loader(indices: Sequence[int], arrays: dict[str, Any], args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        SequenceDataset(indices, arrays),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def contrastive_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def run_epoch(
    model: SequenceJepa,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    train = optimizer is not None
    model.train(train)
    rows = []
    with torch.set_grad_enabled(train):
        for batch_idx, batch in enumerate(loader):
            batch = move_batch(batch, device)
            anchor = F.normalize(batch["anchor"], dim=-1)
            pred = model.encode_source(batch)
            target = model.encode_target(batch)
            target_detached = target.detach()
            jepa_mse = F.mse_loss(pred, target_detached)
            jepa_cos = F.cosine_similarity(pred, target_detached, dim=-1).mean()
            pred_anchor_mse = F.mse_loss(pred, anchor)
            pred_anchor_cos = F.cosine_similarity(pred, anchor, dim=-1).mean()
            target_anchor_mse = F.mse_loss(target, anchor)
            target_anchor_cos = F.cosine_similarity(target, anchor, dim=-1).mean()
            nce = contrastive_loss(pred, target_detached, args.temperature)
            loss = (
                jepa_mse
                + 0.5 * (1.0 - jepa_cos)
                + args.anchor_weight * (pred_anchor_mse + 0.5 * (1.0 - pred_anchor_cos))
                + args.target_anchor_weight * (target_anchor_mse + 0.5 * (1.0 - target_anchor_cos))
                + args.contrastive_weight * nce
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "jepa_cos": float(jepa_cos.detach().cpu()),
                    "pred_anchor_cos": float(pred_anchor_cos.detach().cpu()),
                    "target_anchor_cos": float(target_anchor_cos.detach().cpu()),
                    "pred_anchor_mse": float(pred_anchor_mse.detach().cpu()),
                    "nce": float(nce.detach().cpu()),
                }
            )
            if args.max_batches and batch_idx + 1 >= args.max_batches:
                break
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def predict_all(model: SequenceJepa, arrays: dict[str, Any], args: argparse.Namespace, device: torch.device) -> np.ndarray:
    indices = np.arange(arrays["source"]["article_idx"].shape[0], dtype=np.int64)
    loader = make_loader(indices, arrays, args, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            preds.append(model.encode_source(batch).cpu().numpy().astype(np.float32))
    return np.vstack(preds).astype(np.float32)


def train_sequence_jepa(samples: list[Sample], arrays: dict[str, Any], emb_table: np.ndarray, args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_idx, val_idx = split_indices(samples)
    train_loader = make_loader(train_idx, arrays, args, shuffle=True)
    val_loader = make_loader(val_idx, arrays, args, shuffle=False)
    model = SequenceJepa(emb_table, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = -1e9
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args)
        val_metrics = run_epoch(model, val_loader, None, device, args)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["pred_anchor_cos"] > best_val:
            best_val = val_metrics["pred_anchor_cos"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    preds = predict_all(model, arrays, args, device)
    for sample, z_prior in zip(samples, preds):
        sample.z_prior = unit(z_prior)
        sample.delta_z = (sample.z_prior - sample.z_current).astype(np.float32)
    val_samples = [s for s in samples if s.split == "jepa_val"]
    val_indices = [idx for idx, s in enumerate(samples) if s.split == "jepa_val"]
    anchors = arrays["anchor"][val_indices]
    pred_val = preds[val_indices]
    metrics = {
        "predictor": "structured_sequence_jepa",
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "history": history,
        "best_val_pred_anchor_cos_loader": float(best_val),
        "val_anchor_cosine": float(np.mean([cosine_np(a, b) for a, b in zip(pred_val, anchors)])),
        "val_anchor_mse": float(np.mean((pred_val - anchors) ** 2)),
        "val_existing_future_cosine": float(np.mean([cosine_np(s.z_prior, s.z_future_target) for s in val_samples])),
        "val_existing_future_mse": float(np.mean([(s.z_prior - s.z_future_target) @ (s.z_prior - s.z_future_target) / len(s.z_prior) for s in val_samples])),
        "sequence_stats": arrays["stats"],
    }
    return {"model": model, "metrics": metrics}


def apply_loaded_sequence_jepa(samples: list[Sample], arrays: dict[str, Any], emb_table: np.ndarray, args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = SequenceJepa(emb_table, args).to(device)
    state = torch.load(args.load_model, map_location=device)
    model.load_state_dict(state)
    preds = predict_all(model, arrays, args, device)
    for sample, z_prior in zip(samples, preds):
        sample.z_prior = unit(z_prior)
        sample.delta_z = (sample.z_prior - sample.z_current).astype(np.float32)
    val_indices = [idx for idx, s in enumerate(samples) if s.split == "jepa_val"]
    val_samples = [samples[idx] for idx in val_indices]
    anchors = arrays["anchor"][val_indices]
    pred_val = preds[val_indices]
    metrics = {
        "predictor": "structured_sequence_jepa",
        "loaded_model": args.load_model,
        "num_val": len(val_indices),
        "val_anchor_cosine": float(np.mean([cosine_np(a, b) for a, b in zip(pred_val, anchors)])),
        "val_anchor_mse": float(np.mean((pred_val - anchors) ** 2)),
        "val_existing_future_cosine": float(np.mean([cosine_np(s.z_prior, s.z_future_target) for s in val_samples])),
        "val_existing_future_mse": float(np.mean([(s.z_prior - s.z_future_target) @ (s.z_prior - s.z_future_target) / len(s.z_prior) for s in val_samples])),
        "sequence_stats": arrays["stats"],
    }
    return {"model": model, "metrics": metrics}


def vector_stats(x: np.ndarray, max_eval: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    if len(idx) > max_eval:
        idx = rng.choice(idx, size=max_eval, replace=False)
    sub = x[idx]
    variance = np.var(sub, axis=0)
    if len(sub) > 1:
        pair_idx = rng.choice(len(sub), size=(min(2000, len(sub)), 2), replace=True)
        pair_cos = [cosine_np(sub[a], sub[b]) for a, b in pair_idx if a != b]
    else:
        pair_cos = [0.0]
    return {
        "mean_dim_variance": float(np.mean(variance)),
        "mean_pairwise_cosine": float(np.mean(pair_cos)),
        "mean_norm": float(np.mean(np.linalg.norm(sub, axis=1))),
    }


def self_retrieval_metrics(pred: np.ndarray, target: np.ndarray, max_eval: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pred)
    idx = np.arange(n)
    if n > max_eval:
        idx = rng.choice(idx, size=max_eval, replace=False)
    pred_sub = pred[idx]
    target_sub = target[idx]
    scores = pred_sub @ target_sub.T
    diag = np.diag(scores)
    ranks = 1 + np.sum(scores > diag[:, None], axis=1)
    return {
        "self_retrieval_num_eval": int(len(idx)),
        "self_retrieval_mrr": float(np.mean(1.0 / ranks)),
        "self_retrieval_r@1": float(np.mean(ranks <= 1)),
        "self_retrieval_r@10": float(np.mean(ranks <= 10)),
        "self_retrieval_r@100": float(np.mean(ranks <= 100)),
        "self_retrieval_median_rank": float(np.median(ranks)),
    }


def structured_jepa_eval(samples: list[Sample], arrays: dict[str, Any], args: argparse.Namespace) -> dict:
    train_idx = [idx for idx, s in enumerate(samples) if s.split == "jepa_train"]
    global_anchor = unit(arrays["anchor"][train_idx].mean(axis=0))
    out = {}
    rng = np.random.default_rng(args.seed)
    for split in ["jepa_val", "ranker_val", "ranker_test"]:
        idxs = [idx for idx, s in enumerate(samples) if s.split == split]
        if not idxs:
            continue
        anchor = arrays["anchor"][idxs]
        valid = np.where(np.linalg.norm(anchor, axis=1) > 1e-8)[0]
        raw_num_samples = len(idxs)
        if len(valid) == 0:
            continue
        idxs = [idxs[int(i)] for i in valid]
        anchor = anchor[valid]
        existing_future = np.vstack([samples[idx].z_future_target for idx in idxs]).astype(np.float32)
        predictors = {
            "past_uih": np.vstack([samples[idx].z_current for idx in idxs]).astype(np.float32),
            "current_candidate_set": np.vstack([samples[idx].z_current_set for idx in idxs]).astype(np.float32),
            "past_plus_current_set": np.vstack([unit(samples[idx].z_current + samples[idx].z_current_set) for idx in idxs]).astype(np.float32),
            "global_future_anchor_mean": np.repeat(global_anchor.reshape(1, -1), len(idxs), axis=0).astype(np.float32),
            "sequence_jepa_prior": np.vstack([samples[idx].z_prior for idx in idxs]).astype(np.float32),
        }
        split_out = {}
        shuffled = anchor.copy()
        rng.shuffle(shuffled)
        for name, pred in predictors.items():
            pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
            metrics = {
                "num_samples": int(raw_num_samples),
                "num_eval_nonempty_future": int(len(idxs)),
                "anchor_cosine": float(np.mean([cosine_np(a, b) for a, b in zip(pred, anchor)])),
                "anchor_mse": float(np.mean((pred - anchor) ** 2)),
                "existing_future_cosine": float(np.mean([cosine_np(a, b) for a, b in zip(pred, existing_future)])),
                "existing_future_mse": float(np.mean((pred - existing_future) ** 2)),
                "shuffled_anchor_cosine": float(np.mean([cosine_np(a, b) for a, b in zip(pred, shuffled)])),
            }
            metrics.update(self_retrieval_metrics(pred, anchor, args.retrieval_eval_max, args.seed))
            metrics["prediction_stats"] = vector_stats(pred, args.retrieval_eval_max, args.seed)
            split_out[name] = metrics
        split_out["target_anchor_stats"] = vector_stats(anchor, args.retrieval_eval_max, args.seed)
        out[split] = split_out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured UIH sequence JEPA on cached EB-NeRD small samples.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--article-embeddings", required=True)
    parser.add_argument("--baseline-results", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--anchor-weight", type=float, default=0.5)
    parser.add_argument("--target-anchor-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-weight", type=float, default=0.03)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--retrieval-eval-max", type=int, default=5000)
    parser.add_argument("--skip-ranker", action="store_true")
    parser.add_argument("--load-model", default="")
    parser.add_argument("--max-source-events", type=int, default=128)
    parser.add_argument("--max-target-events", type=int, default=96)
    parser.add_argument("--max-history-events", type=int, default=40)
    parser.add_argument("--max-past-impressions", type=int, default=8)
    parser.add_argument("--max-current-events", type=int, default=32)
    parser.add_argument("--max-future-impressions", type=int, default=8)
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--session-buckets", type=int, default=4096)
    parser.add_argument("--ranker", choices=["lightgbm_ranker", "sgd"], default="lightgbm_ranker")
    parser.add_argument("--lgbm-estimators", type=int, default=350)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.04)
    parser.add_argument("--lgbm-num-leaves", type=int, default=63)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=30)
    parser.add_argument("--lgbm-subsample", type=float, default=0.9)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=2.0)
    parser.add_argument("--lgbm-n-jobs", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, prepare_summary = load_cached_samples(Path(args.samples_pkl))
    article_to_idx, emb_table = load_article_embeddings(Path(args.article_embeddings))
    arrays = build_sequence_arrays(samples, Path(args.data_dir), article_to_idx, emb_table, args)
    if args.load_model:
        jepa = apply_loaded_sequence_jepa(samples, arrays, emb_table, args)
    else:
        jepa = train_sequence_jepa(samples, arrays, emb_table, args)
    # Drop the padding row before sending embeddings to ranker helpers.
    ranker_emb = emb_table[1:].copy()
    ranker_article_to_idx = {aid: idx - 1 for aid, idx in article_to_idx.items()}
    articles = pd.read_parquet(Path(args.data_dir) / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    articles_by_id = article_lookup(articles)
    results = {
        "prepare": prepare_summary,
        "sequence_jepa": jepa["metrics"],
        "structured_jepa_eval": structured_jepa_eval(samples, arrays, args),
        "standalone_jepa": standalone_jepa_eval(samples, ranker_article_to_idx, ranker_emb),
        "ranker_setting": {
            "task": "impression-level candidate reranking / pointwise CTR scoring within each exposed candidate set",
            "ranker": args.ranker,
            "input_rule": "current candidate set has item metadata only; no current feedback enters source UIH",
            "future_target_rule": "future target is a structured event sequence with clicked/not-clicked, read/scroll, time gap, and session features",
            "models": {
                "T1_future_prior": "B1 + structured-sequence JEPA predicted future UIH latent + delta_z",
                "T2_future_item_cross": "T1 + item x structured future-prior and item x delta_z cross features",
            },
        },
        "models": {},
        "example": first_example(samples, articles_by_id),
    }
    if args.baseline_results:
        baseline = json.loads(Path(args.baseline_results).read_text())
        results["baseline_models_from"] = args.baseline_results
        results["baseline_models"] = {
            k: baseline["models"][k]
            for k in ["B0_meta", "B1_text_history"]
            if k in baseline.get("models", {})
        }
    if not args.skip_ranker:
        for variant in ["T1_future_prior", "T2_future_item_cross"]:
            print(f"training {variant} with structured sequence prior", flush=True)
            results["models"][variant] = train_variant(variant, samples, articles_by_id, ranker_article_to_idx, ranker_emb, args.seed, args)
            print(json.dumps({variant: results["models"][variant]["ranker_test"]}, indent=2), flush=True)
    results["split_counts_after_load"] = dict(Counter(s.split for s in samples))
    write_json(out_dir / "results.json", results)
    write_json(out_dir / "uih_example.json", results["example"])
    torch.save(jepa["model"].state_dict(), out_dir / "sequence_jepa.pt")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
