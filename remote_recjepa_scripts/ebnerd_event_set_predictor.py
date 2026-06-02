#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import ebnerd_qwen_uih_text_jepa as q


@dataclass
class EventRecord:
    article_idx: int
    article_id: int
    weight: float
    event_kind: str
    clicked: bool
    read_time: float
    scroll: float
    next_read_time: float
    next_scroll: float
    dt_hours: float
    event_time: str
    session_id: int
    device_type: int
    candidate_count: int
    clicked_count: int


class SampleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "Sample":
            return q.Sample
        return super().find_class(module, name)


def load_samples(path: Path) -> tuple[list[q.Sample], dict[str, Any]]:
    with path.open("rb") as f:
        packed = SampleUnpickler(f).load()
    return packed["samples"], packed.get("prepare_summary", {})


def select_ranker_samples(samples: list[q.Sample], max_per_split: int) -> tuple[list[q.Sample], np.ndarray]:
    wanted = {"ranker_train", "ranker_val", "ranker_test"}
    counts: dict[str, int] = Counter()
    ranker_seen = 0
    out: list[q.Sample] = []
    cache_indices: list[int] = []
    for sample in samples:
        if sample.split not in wanted:
            continue
        cache_idx = ranker_seen
        ranker_seen += 1
        if max_per_split > 0 and counts[sample.split] >= max_per_split:
            continue
        counts[sample.split] += 1
        out.append(sample)
        cache_indices.append(cache_idx)
    for idx, sample in enumerate(out):
        sample.sample_id = idx
    return out, np.asarray(cache_indices, dtype=np.int64)


def weighted_article_mean(
    article_ids: list[int],
    article_to_idx: dict[int, int],
    item_emb: np.ndarray,
    max_items: int,
) -> np.ndarray:
    ids = [article_to_idx[int(aid)] for aid in article_ids[-max_items:] if int(aid) in article_to_idx]
    if not ids:
        return np.zeros((item_emb.shape[1],), dtype=np.float32)
    emb = item_emb[ids].astype(np.float32)
    weights = np.linspace(0.5, 1.0, len(ids), dtype=np.float32)
    weights = weights / max(1e-8, float(weights.sum()))
    return (emb * weights[:, None]).sum(axis=0).astype(np.float32)


def build_source_embeddings(
    samples: list[q.Sample],
    article_to_idx: dict[int, int],
    item_emb: np.ndarray,
    qwen_source_emb: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.source_mode == "qwen_cache":
        source = qwen_source_emb.astype(np.float32)
    else:
        source = np.zeros((len(samples), item_emb.shape[1]), dtype=np.float32)
        for sample in samples:
            clicked = weighted_article_mean(sample.past_clicked, article_to_idx, item_emb, args.max_source_history)
            if args.source_mode == "history_clicked_mean":
                vec = clicked
            else:
                not_clicked = weighted_article_mean(sample.past_not_clicked, article_to_idx, item_emb, args.max_source_history)
                shown = weighted_article_mean(sample.past_shown, article_to_idx, item_emb, args.max_source_history)
                vec = clicked + args.source_shown_weight * shown - args.source_negative_weight * not_clicked
            if not np.isfinite(vec).all() or float(np.linalg.norm(vec)) < 1e-8:
                vec = qwen_source_emb[sample.sample_id].astype(np.float32)
            source[sample.sample_id] = vec
    norms = np.linalg.norm(source, axis=1, keepdims=True)
    return (source / np.clip(norms, 1e-8, None)).astype(np.float32)


def event_weight(read_time: float, scroll: float, dt_hours: float, *, positive: bool) -> float:
    read_part = min(2.0, math.log1p(max(0.0, read_time)) / 3.0)
    scroll_part = min(1.0, max(0.0, scroll) / 100.0) * 0.5
    recency_part = math.exp(-max(0.0, dt_hours) / 24.0)
    base = 1.0 if positive else 0.5
    return float(base + read_part + scroll_part + 0.25 * recency_part)


def add_event(
    rows: list[EventRecord],
    article_to_idx: dict[int, int],
    aid: int,
    *,
    event_kind: str,
    row: Any | None,
    event_time: Any,
    read_time: float,
    scroll: float,
    dt_hours: float,
    positive: bool,
    max_events: int,
) -> None:
    # Future UIH is a set target, not an ordered sequence target. We collect all
    # available events first, then shuffle and truncate deterministically per sample.
    idx = article_to_idx.get(int(aid))
    if idx is None:
        return
    if event_time is None or pd.isna(event_time):
        event_time_text = ""
    else:
        event_time_text = pd.Timestamp(event_time).strftime("%Y-%m-%d %H:%M:%S")
    session_id = q.clean_id(getattr(row, "session_id", None)) if row is not None else None
    device_type = q.clean_id(getattr(row, "device_type", None)) if row is not None else None
    rows.append(
        EventRecord(
            article_idx=int(idx),
            article_id=int(aid),
            weight=event_weight(read_time, scroll, dt_hours, positive=positive),
            event_kind=event_kind,
            clicked=bool(positive),
            read_time=float(read_time),
            scroll=float(scroll),
            next_read_time=q.clean_float(getattr(row, "next_read_time", 0.0)) if row is not None else 0.0,
            next_scroll=q.clean_float(getattr(row, "next_scroll_percentage", 0.0)) if row is not None else 0.0,
            dt_hours=float(dt_hours),
            event_time=event_time_text,
            session_id=session_id or 0,
            device_type=device_type or 0,
            candidate_count=len(q.ids(getattr(row, "article_ids_inview"))) if row is not None else 0,
            clicked_count=len(q.ids(getattr(row, "article_ids_clicked"))) if row is not None else int(positive),
        )
    )


def collect_event_records(
    sample: q.Sample,
    article_to_idx: dict[int, int],
    full_index: q.FullSignalIndex,
    args: argparse.Namespace,
) -> tuple[list[EventRecord], list[EventRecord], dict[int, tuple[float, int]]]:
    rows, current_idx, current_row = full_index.current(sample)
    current_time = pd.Timestamp(sample.time if current_row is None else getattr(current_row, "impression_time"))
    pos_events: list[EventRecord] = []
    neg_events: list[EventRecord] = []
    rank_items: dict[int, tuple[float, int]] = {}

    if args.target_mode == "current_future":
        current_clicked = set(sample.current_clicked if current_row is None else q.ids(getattr(current_row, "article_ids_clicked")))
        current_inview = sample.current_inview if current_row is None else q.ids(getattr(current_row, "article_ids_inview"))
        read = q.clean_float(getattr(current_row, "read_time", 0.0)) if current_row is not None else 0.0
        scroll = q.clean_float(getattr(current_row, "scroll_percentage", 0.0)) if current_row is not None else 0.0
        for aid in current_inview:
            if aid in current_clicked:
                add_event(
                    pos_events,
                    article_to_idx,
                    aid,
                    event_kind="current_clicked",
                    row=current_row,
                    event_time=current_time,
                    read_time=read,
                    scroll=scroll,
                    dt_hours=0.0,
                    positive=True,
                    max_events=args.max_pos_events,
                )
            elif args.include_current_neg:
                add_event(
                    neg_events,
                    article_to_idx,
                    aid,
                    event_kind="current_not_clicked_exposure",
                    row=current_row,
                    event_time=current_time,
                    read_time=0.0,
                    scroll=0.0,
                    dt_hours=0.0,
                    positive=False,
                    max_events=args.max_neg_events,
                )

    if current_idx is not None:
        horizon_end = current_time + pd.Timedelta(hours=args.future_horizon_hours)
        future_rows = []
        for fut in rows[current_idx + 1 :]:
            fut_time = pd.Timestamp(getattr(fut, "impression_time"))
            if fut_time > horizon_end:
                break
            future_rows.append(fut)
            if len(future_rows) >= args.max_future_impressions:
                break
        for fut in future_rows:
            fut_time = pd.Timestamp(getattr(fut, "impression_time"))
            dt_h = max(0.0, (fut_time - current_time).total_seconds() / 3600.0)
            clicked = set(q.ids(getattr(fut, "article_ids_clicked")))
            read = q.clean_float(getattr(fut, "read_time", 0.0))
            scroll = q.clean_float(getattr(fut, "scroll_percentage", 0.0))
            for aid in q.ids(getattr(fut, "article_ids_inview"))[: args.max_events_per_impression]:
                article_idx = article_to_idx.get(int(aid))
                is_clicked = aid in clicked
                if article_idx is not None:
                    gain = event_weight(read, scroll, dt_h, positive=True) if is_clicked else 0.0
                    old_gain, old_label = rank_items.get(article_idx, (0.0, 0))
                    rank_items[article_idx] = (max(old_gain, gain), max(old_label, int(is_clicked)))
                if is_clicked:
                    add_event(
                        pos_events,
                        article_to_idx,
                        aid,
                        event_kind="future_clicked",
                        row=fut,
                        event_time=fut_time,
                        read_time=read,
                        scroll=scroll,
                        dt_hours=dt_h,
                        positive=True,
                        max_events=args.max_pos_events,
                    )
                else:
                    add_event(
                        neg_events,
                        article_to_idx,
                        aid,
                        event_kind="future_not_clicked_exposure",
                        row=fut,
                        event_time=fut_time,
                        read_time=0.0,
                        scroll=0.0,
                        dt_hours=dt_h,
                        positive=False,
                        max_events=args.max_neg_events,
                    )

    event_rng = np.random.default_rng(int(args.seed) + int(sample.sample_id) * 7919 + 17)
    event_rng.shuffle(pos_events)
    event_rng.shuffle(neg_events)
    return pos_events, neg_events, rank_items


def build_event_index(
    samples: list[q.Sample],
    article_to_idx: dict[int, int],
    full_index: q.FullSignalIndex,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    n = len(samples)
    pos_idx = np.zeros((n, args.max_pos_events), dtype=np.int32)
    neg_idx = np.zeros((n, args.max_neg_events), dtype=np.int32)
    pos_weight = np.zeros((n, args.max_pos_events), dtype=np.float32)
    neg_weight = np.zeros((n, args.max_neg_events), dtype=np.float32)
    pos_mask = np.zeros((n, args.max_pos_events), dtype=bool)
    neg_mask = np.zeros((n, args.max_neg_events), dtype=bool)
    rank_idx = np.zeros((n, args.max_rank_candidates), dtype=np.int32)
    rank_gain = np.zeros((n, args.max_rank_candidates), dtype=np.float32)
    rank_label = np.zeros((n, args.max_rank_candidates), dtype=np.int32)
    rank_mask = np.zeros((n, args.max_rank_candidates), dtype=bool)
    stats = Counter()

    for sample in samples:
        pos_events, neg_events, rank_items = collect_event_records(sample, article_to_idx, full_index, args)
        sid = sample.sample_id
        stats["samples"] += 1
        stats["pos_events"] += len(pos_events)
        stats["neg_events"] += len(neg_events)
        stats["empty_pos"] += int(len(pos_events) == 0)
        stats["empty_neg"] += int(len(neg_events) == 0)
        stats["rank_candidates"] += len(rank_items)
        stats["rank_positive"] += sum(label for _, label in rank_items.values())
        stats["rank_empty"] += int(len(rank_items) == 0)
        stats["rank_truncated"] += int(len(rank_items) > args.max_rank_candidates)
        for j, event in enumerate(pos_events[: args.max_pos_events]):
            pos_idx[sid, j] = event.article_idx
            pos_weight[sid, j] = event.weight
            pos_mask[sid, j] = True
        for j, event in enumerate(neg_events[: args.max_neg_events]):
            neg_idx[sid, j] = event.article_idx
            neg_weight[sid, j] = event.weight
            neg_mask[sid, j] = True
        rank_rows = [(idx, gain, label) for idx, (gain, label) in rank_items.items()]
        rank_rng = np.random.default_rng(int(args.seed) + int(sid) * 9973)
        if len(rank_rows) > args.max_rank_candidates:
            positives = [row for row in rank_rows if row[2] > 0]
            negatives = [row for row in rank_rows if row[2] <= 0]
            rank_rng.shuffle(positives)
            rank_rng.shuffle(negatives)
            if len(positives) >= args.max_rank_candidates:
                rank_rows = positives[: args.max_rank_candidates]
            else:
                rank_rows = positives + negatives[: args.max_rank_candidates - len(positives)]
        rank_rng.shuffle(rank_rows)
        for j, (idx, gain, label) in enumerate(rank_rows[: args.max_rank_candidates]):
            rank_idx[sid, j] = idx
            rank_gain[sid, j] = gain
            rank_label[sid, j] = label
            rank_mask[sid, j] = True

    return {
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "pos_weight": pos_weight,
        "neg_weight": neg_weight,
        "pos_mask": pos_mask,
        "neg_mask": neg_mask,
        "rank_idx": rank_idx,
        "rank_gain": rank_gain,
        "rank_label": rank_label,
        "rank_mask": rank_mask,
        "summary": np.asarray(
            [
                stats["samples"],
                stats["pos_events"],
                stats["neg_events"],
                stats["empty_pos"],
                stats["empty_neg"],
                stats["rank_candidates"],
                stats["rank_positive"],
                stats["rank_empty"],
                stats["rank_truncated"],
            ],
            dtype=np.int64,
        ),
    }


def future_rank_candidates(
    event_index: dict[str, np.ndarray],
    sample_id: int,
    idx_to_article: np.ndarray,
    seed: int,
) -> list[tuple[int, int, float, int]]:
    """Return shuffled future exposure candidates as (article_idx, article_id, train_gain, eval_label)."""
    if "rank_idx" in event_index:
        mask = event_index["rank_mask"][sample_id]
        out = [
            (
                int(idx),
                int(idx_to_article[int(idx)]),
                float(gain),
                int(label),
            )
            for idx, gain, label in zip(
                event_index["rank_idx"][sample_id][mask],
                event_index["rank_gain"][sample_id][mask],
                event_index["rank_label"][sample_id][mask],
            )
        ]
        rng = np.random.default_rng(int(seed) + int(sample_id) * 9973)
        rng.shuffle(out)
        return out
    rows: dict[int, tuple[int, int, float, int]] = {}
    neg_idx = event_index["neg_idx"][sample_id]
    neg_weight = event_index["neg_weight"][sample_id]
    neg_mask = event_index["neg_mask"][sample_id]
    for idx, weight in zip(neg_idx[neg_mask], neg_weight[neg_mask]):
        article_idx = int(idx)
        rows[article_idx] = (article_idx, int(idx_to_article[article_idx]), 0.0, 0)
    pos_idx = event_index["pos_idx"][sample_id]
    pos_weight = event_index["pos_weight"][sample_id]
    pos_mask = event_index["pos_mask"][sample_id]
    for idx, weight in zip(pos_idx[pos_mask], pos_weight[pos_mask]):
        article_idx = int(idx)
        prev = rows.get(article_idx)
        gain = float(max(float(weight), prev[2] if prev is not None else 0.0))
        rows[article_idx] = (article_idx, int(idx_to_article[article_idx]), gain, 1)
    out = list(rows.values())
    rng = np.random.default_rng(int(seed) + int(sample_id) * 9973)
    rng.shuffle(out)
    return out


def reaction_event_text(
    sample: q.Sample,
    event: EventRecord,
    rich_articles: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    article_text = q.compact_article(
        event.article_id,
        rich_articles,
        include_extra=True,
        body_chars=getattr(args, "reaction_body_chars", 0),
    )
    clicked = "yes" if event.clicked else "no"
    return "\n".join(
        [
            "Task: encode one user-item reaction event for future UIH prediction.",
            f"Anchor user: {sample.user_id}",
            f"Anchor time: {pd.Timestamp(sample.time).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Event kind: {event.event_kind}",
            (
                "Impression context: "
                f"dt_h={event.dt_hours:.2f} ; "
                f"event_time={event.event_time} ; "
                f"session_id={event.session_id} ; "
                f"device_type={event.device_type} ; "
                f"candidate_count={event.candidate_count} ; "
                f"clicked_count={event.clicked_count}"
            ),
            (
                "User reaction: "
                f"clicked={clicked} ; "
                f"read_s={event.read_time:.1f} ; "
                f"scroll_pct={event.scroll:.1f} ; "
                f"next_read_s={event.next_read_time:.1f} ; "
                f"next_scroll_pct={event.next_scroll:.1f} ; "
                f"engagement_weight={event.weight:.4f}"
            ),
            f"Article: {article_text}",
        ]
    )


def load_reaction_qwen_cache(
    path: Path,
    n_samples: int,
    max_pos_events: int,
    max_neg_events: int,
    emb_dim: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    packed = np.load(path)
    pos = packed["pos_emb"].astype(np.float16, copy=False)
    neg = packed["neg_emb"].astype(np.float16, copy=False)
    if pos.shape[:2] != (n_samples, max_pos_events):
        raise ValueError(f"reaction pos_emb shape {pos.shape} does not match {(n_samples, max_pos_events)}")
    if neg.shape[:2] != (n_samples, max_neg_events):
        raise ValueError(f"reaction neg_emb shape {neg.shape} does not match {(n_samples, max_neg_events)}")
    if pos.shape[-1] != emb_dim or neg.shape[-1] != emb_dim:
        raise ValueError(f"reaction emb dim {pos.shape[-1]}/{neg.shape[-1]} does not match item dim {emb_dim}")
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return pos, neg, meta


class EventDataset(Dataset):
    def __init__(self, samples: list[q.Sample], split: str):
        self.samples = [s for s in samples if s.split == split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> q.Sample:
        return self.samples[idx]


class EventBatchBuilder:
    base_scalar_dim = q.SetBatchBuilder.scalar_dim
    slot_scalar_dim = 3
    user_scalar_dim = 7
    user_profile_tokens = 3
    scalar_dim = base_scalar_dim + slot_scalar_dim

    def __init__(
        self,
        article_to_idx: dict[int, int],
        item_emb: np.ndarray,
        source_emb: np.ndarray,
        event_index: dict[str, np.ndarray],
        articles_by_id: dict[int, dict],
        user_to_idx: dict[int, int],
        item_to_idx: dict[int, int],
        idx_to_article: np.ndarray,
        rank_task: str,
        seed: int,
        max_history_tokens: int,
        user_profile_mode: str,
        rank_supervised_ids: set[int] | None = None,
        slots: np.ndarray | None = None,
        use_slot_features: bool = False,
    ) -> None:
        self.article_to_idx = article_to_idx
        self.item_emb = item_emb.astype(np.float32)
        self.source_emb = source_emb.astype(np.float32)
        self.event_index = event_index
        self.articles_by_id = articles_by_id
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_article = idx_to_article
        self.rank_task = rank_task
        self.seed = seed
        self.max_history_tokens = max_history_tokens
        self.user_profile_mode = user_profile_mode
        self.rank_supervised_ids = rank_supervised_ids
        self.slots = slots
        self.use_slot_features = use_slot_features

    def event_emb(self, key: str, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = self.event_index[f"{key}_idx"][sample_ids]
        mask = self.event_index[f"{key}_mask"][sample_ids]
        weight = self.event_index[f"{key}_weight"][sample_ids].astype(np.float32)
        if f"{key}_emb" in self.event_index:
            emb = self.event_index[f"{key}_emb"][sample_ids].astype(np.float32)
        else:
            emb = self.item_emb[idx].astype(np.float32)
        emb[~mask] = 0.0
        return emb, mask, weight

    def __call__(self, samples: list[q.Sample]) -> dict[str, torch.Tensor | np.ndarray]:
        bsz = len(samples)
        if self.rank_task == "future_engagement":
            candidate_rows = [
                future_rank_candidates(self.event_index, s.sample_id, self.idx_to_article, self.seed)
                for s in samples
            ]
            max_c = max(1, max((len(rows) for rows in candidate_rows), default=0))
        else:
            candidate_rows = []
            max_c = max(len(s.current_inview) for s in samples)
        d = self.item_emb.shape[1]
        item_emb = np.zeros((bsz, max_c, d), dtype=np.float32)
        scalar = np.zeros((bsz, max_c, self.scalar_dim), dtype=np.float32)
        labels = np.zeros((bsz, max_c), dtype=np.float32)
        eval_labels = np.zeros((bsz, max_c), dtype=np.int32)
        mask = np.zeros((bsz, max_c), dtype=bool)
        item_idx = np.zeros((bsz, max_c), dtype=np.int64)
        user_idx = np.zeros((bsz,), dtype=np.int64)
        source = np.zeros((bsz, d), dtype=np.float32)
        candidate_set = np.zeros((bsz, d), dtype=np.float32)
        groups = np.zeros((bsz, max_c), dtype=np.int64)
        sample_ids = np.zeros((bsz,), dtype=np.int64)
        hist_emb = np.zeros((bsz, self.max_history_tokens, d), dtype=np.float32)
        hist_type = np.zeros((bsz, self.max_history_tokens), dtype=np.int64)
        hist_age = np.zeros((bsz, self.max_history_tokens, 1), dtype=np.float32)
        hist_mask = np.zeros((bsz, self.max_history_tokens), dtype=bool)
        user_profile_emb = np.zeros((bsz, self.user_profile_tokens, d), dtype=np.float32)
        user_profile_type = np.zeros((bsz, self.user_profile_tokens), dtype=np.int64)
        user_profile_mask = np.zeros((bsz, self.user_profile_tokens), dtype=bool)
        user_scalar = np.zeros((bsz, self.user_scalar_dim), dtype=np.float32)
        rank_supervised = np.ones((bsz,), dtype=bool)

        slot_dim = self.slots.shape[1] if self.slots is not None else 1
        slot_emb = np.zeros((bsz, slot_dim, d), dtype=np.float32)
        slot_mask = np.zeros((bsz, slot_dim), dtype=bool)

        for bi, sample in enumerate(samples):
            sid = sample.sample_id
            sample_ids[bi] = sid
            if self.rank_supervised_ids is not None:
                rank_supervised[bi] = sid in self.rank_supervised_ids
            z_source = self.source_emb[sid].astype(np.float32)
            source[bi] = z_source
            user_idx[bi] = self.user_to_idx.get(sample.user_id, 0)
            clicked = set(sample.current_clicked)
            current_items = []
            history_rows: list[tuple[int, int]] = []
            clicked_hist = {int(aid) for aid in sample.past_clicked}
            not_clicked_hist = {int(aid) for aid in sample.past_not_clicked}
            shown_only_hist = [
                int(aid)
                for aid in sample.past_shown
                if int(aid) not in clicked_hist and int(aid) not in not_clicked_hist
            ]
            if self.user_profile_mode != "none":
                profile_sources = [sample.past_clicked, sample.past_not_clicked, shown_only_hist]
                for pi, aids in enumerate(profile_sources):
                    vec = weighted_article_mean(aids, self.article_to_idx, self.item_emb, self.max_history_tokens)
                    if np.isfinite(vec).all() and float(np.linalg.norm(vec)) > 1e-8:
                        user_profile_emb[bi, pi] = vec
                        user_profile_type[bi, pi] = pi + 1
                        user_profile_mask[bi, pi] = True
                clicked_n = float(len(sample.past_clicked))
                skipped_n = float(len(sample.past_not_clicked))
                shown_n = float(len(sample.past_shown))
                shown_only_n = float(len(shown_only_hist))
                response_n = max(1.0, clicked_n + skipped_n)
                user_scalar[bi] = np.asarray(
                    [
                        math.log1p(clicked_n) / 8.0,
                        math.log1p(skipped_n) / 8.0,
                        math.log1p(shown_n) / 8.0,
                        clicked_n / response_n,
                        skipped_n / response_n,
                        shown_only_n / max(1.0, shown_n),
                        float(len(clicked_hist)) / max(1.0, clicked_n),
                    ],
                    dtype=np.float32,
                )
            for aid in sample.past_shown:
                aid = int(aid)
                if aid not in self.article_to_idx:
                    continue
                if aid in clicked_hist:
                    htype = 1
                elif aid in not_clicked_hist:
                    htype = 2
                else:
                    htype = 3
                history_rows.append((aid, htype))
            # Fallback for old caches where response-only history ids may not appear in past_shown.
            present = {aid for aid, _ in history_rows}
            history_rows.extend((int(aid), 1) for aid in sample.past_clicked if int(aid) in self.article_to_idx and int(aid) not in present)
            present = {aid for aid, _ in history_rows}
            history_rows.extend((int(aid), 2) for aid in sample.past_not_clicked if int(aid) in self.article_to_idx and int(aid) not in present)
            history_rows = history_rows[-self.max_history_tokens :]
            denom = max(1, len(history_rows) - 1)
            for hi, (aid, htype) in enumerate(history_rows):
                idx = self.article_to_idx.get(aid)
                if idx is None:
                    continue
                hist_emb[bi, hi] = self.item_emb[idx]
                hist_type[bi, hi] = htype
                hist_age[bi, hi, 0] = float(hi) / float(denom)
                hist_mask[bi, hi] = True
            if self.slots is not None:
                s = self.slots[sid].astype(np.float32)
                slot_emb[bi, : s.shape[0]] = s
                slot_mask[bi, : s.shape[0]] = np.linalg.norm(s, axis=1) > 1e-6
            norm_slots = slot_emb[bi] / np.clip(np.linalg.norm(slot_emb[bi], axis=1, keepdims=True), 1e-8, None)
            if self.rank_task == "future_engagement":
                iterable = candidate_rows[bi]
            else:
                clicked = set(sample.current_clicked)
                iterable = [
                    (self.article_to_idx.get(aid), aid, float(aid in clicked), int(aid in clicked))
                    for aid in sample.current_inview
                ]
            for ci, (idx, aid, gain, eval_label) in enumerate(iterable):
                if idx is None:
                    continue
                item = self.item_emb[idx].astype(np.float32) if idx is not None else np.zeros(d, dtype=np.float32)
                current_items.append(item)
                item_emb[bi, ci] = item
                position_feature = 0.0 if self.rank_task == "future_engagement" else float(ci) / max(1.0, float(len(sample.current_inview) - 1))
                candidate_count = len(iterable) if self.rank_task == "future_engagement" else len(sample.current_inview)
                base = [q.cosine(item, z_source), 0.0, 0.0, position_feature]
                base += q.article_dense_meta(self.articles_by_id, aid, sample, candidate_count)
                if self.use_slot_features and slot_mask[bi].any():
                    scores = norm_slots @ (item / max(1e-8, float(np.linalg.norm(item))))
                    scores = scores[slot_mask[bi]]
                    top = np.sort(scores)[-min(3, len(scores)) :]
                    prob = np.exp(scores - scores.max())
                    prob = prob / max(1e-8, float(prob.sum()))
                    entropy = -float(np.sum(prob * np.log(np.clip(prob, 1e-8, None)))) / max(1.0, math.log(len(prob)))
                    slot_feats = [float(scores.max()), float(top.mean()), entropy]
                else:
                    slot_feats = [0.0, 0.0, 0.0]
                scalar[bi, ci] = np.asarray(base + slot_feats, dtype=np.float32)
                labels[bi, ci] = float(gain)
                eval_labels[bi, ci] = int(eval_label)
                mask[bi, ci] = True
                item_idx[bi, ci] = self.item_to_idx.get(aid, 0)
                groups[bi, ci] = sample.sample_id if self.rank_task == "future_engagement" else sample.impression_id
            if current_items:
                candidate_set[bi] = np.mean(current_items, axis=0).astype(np.float32)

        pos_emb, pos_mask, pos_weight = self.event_emb("pos", sample_ids)
        neg_emb, neg_mask, neg_weight = self.event_emb("neg", sample_ids)
        return {
            "item_emb": torch.from_numpy(item_emb),
            "scalar": torch.from_numpy(scalar),
            "labels": torch.from_numpy(labels),
            "eval_labels": torch.from_numpy(eval_labels),
            "mask": torch.from_numpy(mask),
            "item_idx": torch.from_numpy(item_idx),
            "user_idx": torch.from_numpy(user_idx),
            "source": torch.from_numpy(source),
            "candidate_set": torch.from_numpy(candidate_set),
            "hist_emb": torch.from_numpy(hist_emb),
            "hist_type": torch.from_numpy(hist_type),
            "hist_age": torch.from_numpy(hist_age),
            "hist_mask": torch.from_numpy(hist_mask),
            "user_profile_emb": torch.from_numpy(user_profile_emb),
            "user_profile_type": torch.from_numpy(user_profile_type),
            "user_profile_mask": torch.from_numpy(user_profile_mask),
            "user_scalar": torch.from_numpy(user_scalar),
            "slot_emb": torch.from_numpy(slot_emb),
            "slot_mask": torch.from_numpy(slot_mask),
            "pos_emb": torch.from_numpy(pos_emb),
            "pos_mask": torch.from_numpy(pos_mask),
            "pos_weight": torch.from_numpy(pos_weight),
            "neg_emb": torch.from_numpy(neg_emb),
            "neg_mask": torch.from_numpy(neg_mask),
            "neg_weight": torch.from_numpy(neg_weight),
            "rank_supervised": torch.from_numpy(rank_supervised),
            "group_ids": groups,
            "sample_ids": sample_ids,
        }


class EventSlotPredictor(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        scalar_dim: int,
        num_users: int,
        num_items: int,
        num_slots: int,
        d_model: int,
        nhead: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        use_item_id_emb: bool,
        score_mode: str,
        slot_residual_scale: float,
        init_slot_dot_scale: float,
        slot_output_mode: str,
        user_profile_mode: str,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.use_item_id_emb = use_item_id_emb
        self.score_mode = score_mode
        self.slot_residual_scale = float(slot_residual_scale)
        self.user_profile_mode = user_profile_mode
        self.slot_output_mode = slot_output_mode
        self.slot_dot_scale = nn.Parameter(torch.tensor(float(init_slot_dot_scale), dtype=torch.float32))
        self.vec_proj = nn.Linear(emb_dim, d_model)
        self.scalar_proj = nn.Linear(scalar_dim, d_model)
        self.hist_age_proj = nn.Linear(1, d_model)
        self.user_scalar_proj = nn.Linear(EventBatchBuilder.user_scalar_dim, d_model)
        self.user_emb = nn.Embedding(num_users + 1, d_model, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(8, d_model)
        self.hist_type_emb = nn.Embedding(4, d_model, padding_idx=0)
        self.profile_type_emb = nn.Embedding(4, d_model, padding_idx=0)
        self.slot_queries = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
        )
        self.rank_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.slot_residual_head = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        self.slot_gate_head = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.slot_residual_head[-1].weight)
        nn.init.zeros_(self.slot_residual_head[-1].bias)
        nn.init.zeros_(self.slot_gate_head[-1].weight)
        nn.init.zeros_(self.slot_gate_head[-1].bias)

    def slot_item_features(self, slots: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        slot_scores = torch.einsum("bkd,bcd->bck", F.normalize(slots, dim=-1), F.normalize(item_emb, dim=-1))
        top = torch.topk(slot_scores, k=min(3, slot_scores.shape[-1]), dim=-1).values
        prob = F.softmax(slot_scores, dim=-1)
        entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=-1) / max(1.0, math.log(slot_scores.shape[-1]))
        return torch.stack([slot_scores.max(dim=-1).values, top.mean(dim=-1), entropy], dim=-1)

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int, int]:
        user_tok = self.user_emb(batch["user_idx"]) + self.type_emb.weight[0]
        source_tok = self.vec_proj(batch["source"]) + self.type_emb.weight[1]
        set_tok = self.vec_proj(batch["candidate_set"]) + self.type_emb.weight[2]
        tokens = [user_tok.unsqueeze(1), source_tok.unsqueeze(1), set_tok.unsqueeze(1)]
        masks = [torch.zeros((batch["source"].shape[0], 3), dtype=torch.bool, device=batch["source"].device)]
        if self.user_profile_mode != "none":
            user_scalar_tok = self.user_scalar_proj(batch["user_scalar"]) + self.type_emb.weight[3]
            profile_tok = (
                self.vec_proj(batch["user_profile_emb"])
                + self.profile_type_emb(batch["user_profile_type"])
                + self.type_emb.weight[4]
            )
            tokens.extend([user_scalar_tok.unsqueeze(1), profile_tok])
            masks.extend(
                [
                    torch.zeros((batch["source"].shape[0], 1), dtype=torch.bool, device=batch["source"].device),
                    ~batch["user_profile_mask"],
                ]
            )
        hist_tok = (
            self.vec_proj(batch["hist_emb"])
            + self.hist_type_emb(batch["hist_type"])
            + self.hist_age_proj(batch["hist_age"])
            + self.type_emb.weight[5]
        )
        item_tok = (
            self.vec_proj(batch["item_emb"])
            + self.scalar_proj(batch["scalar"][..., : EventBatchBuilder.base_scalar_dim])
            + self.type_emb.weight[6]
        )
        if self.use_item_id_emb:
            item_tok = item_tok + self.item_id_emb(batch["item_idx"])
        slot_tok = self.slot_queries.unsqueeze(0).repeat(item_tok.shape[0], 1, 1) + self.type_emb.weight[7]
        tokens.append(hist_tok)
        masks.append(~batch["hist_mask"])
        item_start = sum(token.shape[1] for token in tokens)
        tokens.append(item_tok)
        masks.append(~batch["mask"])
        tokens.append(slot_tok)
        masks.append(torch.zeros((item_tok.shape[0], self.num_slots), dtype=torch.bool, device=item_tok.device))
        x = torch.cat(tokens, dim=1)
        pad_mask = torch.cat(masks, dim=1)
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        return out, item_start, item_tok.shape[1]

    def make_slots(self, out: torch.Tensor, item_start: int, num_items: int, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        slot_hidden = out[:, -self.num_slots :]
        raw_slots = self.out(slot_hidden)
        if self.slot_output_mode == "direct":
            return F.normalize(raw_slots, dim=-1)
        item_hidden = out[:, item_start : item_start + num_items]
        attn_logits = torch.einsum("bkd,bcd->bkc", slot_hidden, item_hidden) / math.sqrt(float(slot_hidden.shape[-1]))
        attn_logits = attn_logits.masked_fill(~batch["mask"].unsqueeze(1), -1e4)
        attn = F.softmax(attn_logits, dim=-1)
        candidate_slots = torch.einsum("bkc,bcd->bkd", attn, batch["item_emb"])
        if self.slot_output_mode == "candidate_attention":
            return F.normalize(candidate_slots, dim=-1)
        if self.slot_output_mode == "residual_candidate_attention":
            return F.normalize(candidate_slots + raw_slots, dim=-1)
        raise ValueError(f"unknown slot_output_mode: {self.slot_output_mode}")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out, item_start, num_items = self.encode(batch)
        return self.make_slots(out, item_start, num_items, batch)

    def forward_with_scores(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        out, item_start, num_items = self.encode(batch)
        item_out = out[:, item_start : item_start + num_items]
        slots = self.make_slots(out, item_start, num_items, batch)
        rank_score = self.rank_head(item_out).squeeze(-1)
        if self.score_mode != "plain":
            slot_feats = self.slot_item_features(slots, batch["item_emb"])
            if self.score_mode in {"slot_dot", "slot_dot_residual", "gated_slot_dot_residual"}:
                rank_score = rank_score + self.slot_residual_scale * self.slot_dot_scale * slot_feats[..., 0]
            if self.score_mode in {"slot_residual", "gated_slot_residual", "slot_dot_residual", "gated_slot_dot_residual"}:
                residual = self.slot_residual_head(slot_feats).squeeze(-1)
                if self.score_mode in {"gated_slot_residual", "gated_slot_dot_residual"}:
                    residual = torch.sigmoid(self.slot_gate_head(slot_feats).squeeze(-1)) * residual
                rank_score = rank_score + self.slot_residual_scale * residual
        return slots, rank_score


def slot_event_score(slots: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bkd,bpd->bkp", F.normalize(slots, dim=-1), F.normalize(events, dim=-1))
    return scores.max(dim=1).values


def event_nce_loss(slots: torch.Tensor, pos_emb: torch.Tensor, pos_mask: torch.Tensor, temp: float) -> torch.Tensor:
    bsz = slots.shape[0]
    flat = pos_emb[pos_mask]
    if flat.numel() == 0:
        return slots.sum() * 0.0
    owners = torch.arange(bsz, device=slots.device).unsqueeze(1).expand_as(pos_mask)[pos_mask]
    scores = torch.einsum("bkd,md->bkm", F.normalize(slots, dim=-1), F.normalize(flat, dim=-1)).max(dim=1).values / temp
    same = owners.unsqueeze(0) == torch.arange(bsz, device=slots.device).unsqueeze(1)
    valid = same.any(dim=1)
    num = torch.logsumexp(scores.masked_fill(~same, -1e4), dim=1)
    den = torch.logsumexp(scores, dim=1)
    return (den[valid] - num[valid]).mean() if valid.any() else slots.sum() * 0.0


def slotwise_event_nce_loss(slots: torch.Tensor, pos_emb: torch.Tensor, pos_mask: torch.Tensor, temp: float) -> torch.Tensor:
    flat_events = pos_emb[pos_mask]
    if flat_events.numel() == 0:
        return slots.sum() * 0.0
    bsz, num_slots, dim = slots.shape
    event_owner = torch.arange(bsz, device=slots.device).unsqueeze(1).expand_as(pos_mask)[pos_mask]
    slot_flat = F.normalize(slots.reshape(bsz * num_slots, dim), dim=-1)
    slot_owner = torch.arange(bsz, device=slots.device).unsqueeze(1).expand(bsz, num_slots).reshape(-1)
    scores = slot_flat @ F.normalize(flat_events, dim=-1).T / temp
    same = slot_owner.unsqueeze(1) == event_owner.unsqueeze(0)
    valid = same.any(dim=1)
    if not valid.any():
        return slots.sum() * 0.0
    num = torch.logsumexp(scores.masked_fill(~same, -1e4), dim=1)
    den = torch.logsumexp(scores, dim=1)
    return (den[valid] - num[valid]).mean()


def slot_assignment_ce_loss(
    slots: torch.Tensor,
    pos_emb: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_emb: torch.Tensor,
    neg_mask: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    event_emb = torch.cat([pos_emb, neg_emb], dim=1)
    event_mask = torch.cat([pos_mask, neg_mask], dim=1)
    positive_mask = torch.cat([pos_mask, torch.zeros_like(neg_mask)], dim=1)
    bsz, num_slots, dim = slots.shape
    event = F.normalize(event_emb, dim=-1)
    logits = torch.einsum("bkd,bmd->bkm", F.normalize(slots, dim=-1), event) / temp
    logits = logits.masked_fill(~event_mask.unsqueeze(1), -1e4)
    has_pos = positive_mask.any(dim=1).unsqueeze(1).expand(bsz, num_slots).reshape(-1)
    if not has_pos.any():
        return slots.sum() * 0.0
    flat_logits = logits.reshape(bsz * num_slots, -1)
    flat_pos = positive_mask.unsqueeze(1).expand(bsz, num_slots, -1).reshape(bsz * num_slots, -1)
    num = torch.logsumexp(flat_logits.masked_fill(~flat_pos, -1e4), dim=1)
    den = torch.logsumexp(flat_logits, dim=1)
    return (den[has_pos] - num[has_pos]).mean()


def event_bpr_loss(
    slots: torch.Tensor,
    pos_emb: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_emb: torch.Tensor,
    neg_mask: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    pos_score = slot_event_score(slots, pos_emb) / temp
    neg_score = slot_event_score(slots, neg_emb) / temp
    valid = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
    if not valid.any():
        return slots.sum() * 0.0
    pair = F.softplus(neg_score.unsqueeze(1) - pos_score.unsqueeze(2))
    return pair[valid].mean()


def candidate_kl_loss(
    slots: torch.Tensor,
    item_emb: torch.Tensor,
    mask: torch.Tensor,
    pos_emb: torch.Tensor,
    pos_mask: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    pred = torch.einsum("bkd,bcd->bkc", F.normalize(slots, dim=-1), F.normalize(item_emb, dim=-1)).max(dim=1).values / temp
    oracle = torch.einsum("bpd,bcd->bpc", F.normalize(pos_emb, dim=-1), F.normalize(item_emb, dim=-1)).masked_fill(
        ~pos_mask.unsqueeze(-1), -1e4
    )
    oracle = oracle.max(dim=1).values / temp
    pred = pred.masked_fill(~mask, -1e4)
    oracle = oracle.masked_fill(~mask, -1e4)
    valid = pos_mask.any(dim=1)
    if not valid.any():
        return slots.sum() * 0.0
    teacher = F.softmax(oracle[valid], dim=1).detach()
    return -(teacher * F.log_softmax(pred[valid], dim=1)).sum(dim=1).mean()


def candidate_rank_loss(slots: torch.Tensor, item_emb: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = torch.einsum("bkd,bcd->bkc", F.normalize(slots, dim=-1), F.normalize(item_emb, dim=-1)).max(dim=1).values
    return q.listwise_loss(pred, labels, mask)


def candidate_bce_loss(
    slots: torch.Tensor,
    item_emb: torch.Tensor,
    eval_labels: torch.Tensor,
    mask: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    pred = torch.einsum("bkd,bcd->bkc", F.normalize(slots, dim=-1), F.normalize(item_emb, dim=-1)).max(dim=1).values / temp
    valid = mask
    if not valid.any():
        return slots.sum() * 0.0
    target = eval_labels.float()
    y = target[valid]
    pos = y.sum()
    neg = y.numel() - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(pred[valid], y, pos_weight=pos_weight)


def supervised_rank_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "rank_supervised" not in batch:
        return batch["mask"]
    return batch["mask"] & batch["rank_supervised"].unsqueeze(1)


def slot_diversity_loss(slots: torch.Tensor, margin: float) -> torch.Tensor:
    s = F.normalize(slots, dim=-1)
    sim = torch.einsum("bkd,bld->bkl", s, s)
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool).unsqueeze(0)
    return (sim.masked_select(~eye) - margin).clamp_min(0.0).pow(2).mean()


def event_loss(
    slots: torch.Tensor,
    batch: dict[str, torch.Tensor],
    variant: str,
    args: argparse.Namespace,
    rank_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    nce = event_nce_loss(slots, batch["pos_emb"], batch["pos_mask"], args.event_temperature)
    slot_nce = slotwise_event_nce_loss(slots, batch["pos_emb"], batch["pos_mask"], args.event_temperature)
    assignment = slot_assignment_ce_loss(
        slots,
        batch["pos_emb"],
        batch["pos_mask"],
        batch["neg_emb"],
        batch["neg_mask"],
        args.event_temperature,
    )
    bpr = event_bpr_loss(slots, batch["pos_emb"], batch["pos_mask"], batch["neg_emb"], batch["neg_mask"], args.bpr_temperature)
    cand = candidate_kl_loss(slots, batch["item_emb"], batch["mask"], batch["pos_emb"], batch["pos_mask"], args.candidate_temperature)
    rank_mask = supervised_rank_mask(batch)
    rank = candidate_rank_loss(slots, batch["item_emb"], batch["labels"], rank_mask) if rank_mask.any() else slots.sum() * 0.0
    cand_bce = (
        candidate_bce_loss(slots, batch["item_emb"], batch["eval_labels"], rank_mask, args.candidate_bce_temperature)
        if rank_mask.any()
        else slots.sum() * 0.0
    )
    rank_head = q.listwise_loss(rank_scores, batch["labels"], rank_mask) if rank_scores is not None and rank_mask.any() else slots.sum() * 0.0
    div = slot_diversity_loss(slots, args.diversity_margin)
    total = (
        args.nce_weight * nce
        + args.slot_nce_weight * slot_nce
        + args.assignment_weight * assignment
        + args.diversity_weight * div
    )
    if variant in {"E2_event_nce_bpr", "E3_event_hybrid"}:
        total = total + args.bpr_weight * bpr
    if variant == "E3_event_hybrid":
        total = total + args.candidate_kl_weight * cand
    if args.rank_label_weight > 0:
        total = total + args.rank_label_weight * rank
    if args.candidate_bce_weight > 0:
        total = total + args.candidate_bce_weight * cand_bce
    if args.rank_head_weight > 0:
        total = total + args.rank_head_weight * rank_head
    return total, {
        "nce": float(nce.detach().cpu()),
        "slot_nce": float(slot_nce.detach().cpu()),
        "assignment": float(assignment.detach().cpu()),
        "bpr": float(bpr.detach().cpu()),
        "candidate_kl": float(cand.detach().cpu()),
        "rank_label": float(rank.detach().cpu()),
        "candidate_bce": float(cand_bce.detach().cpu()),
        "rank_head": float(rank_head.detach().cpu()),
        "diversity": float(div.detach().cpu()),
    }


@torch.no_grad()
def predict_slots(model: nn.Module, loader: DataLoader, device: torch.device, n_samples: int, num_slots: int, emb_dim: int) -> np.ndarray:
    model.eval()
    slots = np.zeros((n_samples, num_slots, emb_dim), dtype=np.float16)
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(moved).detach().cpu().numpy().astype(np.float16)
        slots[batch["sample_ids"]] = out
    return slots


@torch.no_grad()
def predict_event_rank_head(model: EventSlotPredictor, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    model.eval()
    labels, gains, scores, groups = [], [], [], []
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        _, out = model.forward_with_scores(moved)
        pred = out.detach().cpu().numpy()
        lab = batch["eval_labels"].numpy()
        gain = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        group = batch["group_ids"]
        labels.append(lab[mask])
        gains.append(gain[mask])
        scores.append(pred[mask])
        groups.append(group[mask])
    return ranking_metrics_for_task(
        np.concatenate(labels),
        np.concatenate(scores),
        np.concatenate(groups),
        args,
        gain_labels=np.concatenate(gains),
    )


@torch.no_grad()
def predict_rank_head_slot_components(
    model: EventSlotPredictor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, gains, rank_scores, slot_scores, groups = [], [], [], [], []
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        slots, out = model.forward_with_scores(moved)
        slot_score = torch.einsum(
            "bkd,bcd->bck",
            F.normalize(slots, dim=-1),
            F.normalize(moved["item_emb"], dim=-1),
        ).max(dim=-1).values
        pred = out.detach().cpu().numpy()
        slot = slot_score.detach().cpu().numpy()
        lab = batch["eval_labels"].numpy()
        gain = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        group = batch["group_ids"]
        labels.append(lab[mask])
        gains.append(gain[mask])
        rank_scores.append(pred[mask])
        slot_scores.append(slot[mask])
        groups.append(group[mask])
    return (
        np.concatenate(labels),
        np.concatenate(gains),
        np.concatenate(rank_scores),
        np.concatenate(slot_scores),
        np.concatenate(groups),
    )


def rank_head_slot_blend_metrics(
    model: EventSlotPredictor,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    y_val, y_val_gain, rank_val, slot_val, g_val = predict_rank_head_slot_components(model, val_loader, device)
    y_test, y_test_gain, rank_test, slot_test, g_test = predict_rank_head_slot_components(model, test_loader, device)
    rank_val = normalize_per_group(rank_val.astype(np.float32), g_val)
    slot_val = normalize_per_group(slot_val.astype(np.float32), g_val)
    rank_test = normalize_per_group(rank_test.astype(np.float32), g_test)
    slot_test = normalize_per_group(slot_test.astype(np.float32), g_test)
    rank_val_metric = ranking_metrics_for_task(y_val, rank_val, g_val, args, gain_labels=y_val_gain)
    slot_val_metric = ranking_metrics_for_task(y_val, slot_val, g_val, args, gain_labels=y_val_gain)
    best_lam = 0.0
    best_val = rank_val_metric
    best_metric = primary_metric(rank_val_metric, args)
    grid = [float(x) for x in args.rank_slot_blend_lambda_grid.split(",") if x.strip()]
    for lam in grid:
        out = ranking_metrics_for_task(y_val, rank_val + lam * slot_val, g_val, args, gain_labels=y_val_gain)
        metric = primary_metric(out, args)
        if metric > best_metric:
            best_metric = metric
            best_lam = lam
            best_val = out
    return {
        "ranker_val": {
            "rank_head": rank_val_metric,
            "slot": slot_val_metric,
            "best_blend": best_val,
            "lambda": best_lam,
        },
        "ranker_test": {
            "rank_head": ranking_metrics_for_task(y_test, rank_test, g_test, args, gain_labels=y_test_gain),
            "slot": ranking_metrics_for_task(y_test, slot_test, g_test, args, gain_labels=y_test_gain),
            "best_blend": ranking_metrics_for_task(y_test, rank_test + best_lam * slot_test, g_test, args, gain_labels=y_test_gain),
            "lambda": best_lam,
        },
    }


def make_loaders(
    samples: list[q.Sample],
    article_to_idx: dict[int, int],
    item_emb: np.ndarray,
    source_emb: np.ndarray,
    event_index: dict[str, np.ndarray],
    articles_by_id: dict[int, dict],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    idx_to_article: np.ndarray,
    args: argparse.Namespace,
    splits: list[str],
    rank_supervised_ids: set[int] | None = None,
    slots: np.ndarray | None = None,
    use_slot_features: bool = False,
) -> dict[str, DataLoader]:
    builder = EventBatchBuilder(
        article_to_idx,
        item_emb,
        source_emb,
        event_index,
        articles_by_id,
        user_to_idx,
        item_to_idx,
        idx_to_article,
        args.rank_task,
        args.seed,
        args.max_history_tokens,
        args.user_profile_mode,
        rank_supervised_ids=rank_supervised_ids,
        slots=slots,
        use_slot_features=use_slot_features,
    )
    return {
        split: DataLoader(
            EventDataset(samples, split),
            batch_size=args.batch_size,
            shuffle=split == "ranker_train",
            num_workers=args.num_workers,
            pin_memory=torch.device(args.device).type == "cuda",
            collate_fn=builder,
        )
        for split in splits
    }


def ranking_metrics_for_task(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_ids: np.ndarray,
    args: argparse.Namespace,
    gain_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    out = q.ranking_metrics(y_true, y_score, group_ids)
    if args.rank_task == "future_engagement" and gain_labels is not None:
        out.update({f"engagement_{k}": v for k, v in engagement_ranking_metrics(gain_labels, y_score, group_ids).items()})
    if args.rank_task == "future_engagement":
        out["group_unit"] = "anchor_user_time_query"
        out["num_queries"] = out.get("num_impressions", 0)
    else:
        out["group_unit"] = "logged_current_impression"
    return out


def engagement_ranking_metrics(y_gain: np.ndarray, y_score: np.ndarray, group_ids: np.ndarray) -> dict[str, Any]:
    gain = np.asarray(y_gain, dtype=np.float32)
    score = np.asarray(y_score, dtype=np.float32)
    groups = np.asarray(group_ids, dtype=np.int64)
    mrr: list[float] = []
    hit1: list[float] = []
    ndcg5: list[float] = []
    ndcg10: list[float] = []
    mean_gain1: list[float] = []
    for gid in np.unique(groups):
        idx = np.where(groups == gid)[0]
        labels = gain[idx]
        if labels.max(initial=0.0) <= 0:
            continue
        order = np.argsort(-score[idx])
        ranked = labels[order]
        pos = np.where(ranked > 0)[0]
        mrr.append(1.0 / float(pos[0] + 1) if len(pos) else 0.0)
        hit1.append(float(ranked[0] > 0))
        mean_gain1.append(float(ranked[0]))
        for k, bucket in [(5, ndcg5), (10, ndcg10)]:
            gains = ranked[:k]
            discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
            dcg = float(np.sum(gains * discounts))
            ideal = np.sort(labels)[::-1][:k]
            idcg = float(np.sum(ideal * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))))
            bucket.append(dcg / idcg if idcg > 0 else 0.0)
    return {
        "mrr": float(np.mean(mrr)) if mrr else float("nan"),
        "hit1": float(np.mean(hit1)) if hit1 else float("nan"),
        "ndcg@5": float(np.mean(ndcg5)) if ndcg5 else float("nan"),
        "ndcg@10": float(np.mean(ndcg10)) if ndcg10 else float("nan"),
        "mean_gain@1": float(np.mean(mean_gain1)) if mean_gain1 else float("nan"),
    }


def primary_metric(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    key = getattr(args, "primary_rank_metric", "ndcg@10")
    value = metrics.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        value = metrics.get("ndcg@10", -float("inf"))
    return float(value)


def normalize_per_group(values: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    group_ids = group_ids.astype(np.int64, copy=False)
    if values.size == 0:
        return values.astype(np.float32)
    order = np.argsort(group_ids, kind="mergesort")
    sorted_groups = group_ids[order]
    sorted_values = values[order]
    _, starts, counts = np.unique(sorted_groups, return_index=True, return_counts=True)
    sums = np.add.reduceat(sorted_values, starts)
    sums_sq = np.add.reduceat(sorted_values * sorted_values, starts)
    means = sums / counts
    var = np.maximum(sums_sq / counts - means * means, 1e-12)
    std = np.sqrt(var)
    normalized = (sorted_values - np.repeat(means, counts)) / np.repeat(np.maximum(std, 1e-6), counts)
    out = np.empty_like(normalized, dtype=np.float32)
    out[order] = normalized.astype(np.float32)
    return out


def sample_event_metrics(
    samples: list[q.Sample],
    split: str,
    item_emb: np.ndarray,
    event_index: dict[str, np.ndarray],
    slots: np.ndarray,
) -> dict[str, Any]:
    aucs: list[float] = []
    mrrs: list[float] = []
    recalls = {1: [], 3: [], 5: []}
    sims = []
    for sample in samples:
        if sample.split != split:
            continue
        sid = sample.sample_id
        slot = slots[sid].astype(np.float32)
        slot = slot / np.clip(np.linalg.norm(slot, axis=1, keepdims=True), 1e-8, None)
        sim = slot @ slot.T
        if slot.shape[0] > 1:
            sims.append(float(sim[np.triu_indices(slot.shape[0], k=1)].mean()))
        pos_ids = event_index["pos_idx"][sid][event_index["pos_mask"][sid]]
        neg_ids = event_index["neg_idx"][sid][event_index["neg_mask"][sid]]
        if len(pos_ids) == 0:
            continue
        labels = np.concatenate([np.ones(len(pos_ids), dtype=np.int32), np.zeros(len(neg_ids), dtype=np.int32)])
        if len(labels) == 0:
            continue
        if "pos_emb" in event_index:
            pos_ev = event_index["pos_emb"][sid][event_index["pos_mask"][sid]].astype(np.float32)
            neg_ev = event_index["neg_emb"][sid][event_index["neg_mask"][sid]].astype(np.float32)
            ev = np.concatenate([pos_ev, neg_ev], axis=0)
        else:
            ids = np.concatenate([pos_ids, neg_ids])
            ev = item_emb[ids].astype(np.float32)
        ev = ev / np.clip(np.linalg.norm(ev, axis=1, keepdims=True), 1e-8, None)
        scores = (ev @ slot.T).max(axis=1)
        order = np.argsort(-scores)
        ranked = labels[order]
        first = np.where(ranked == 1)[0]
        if len(first):
            mrrs.append(1.0 / float(first[0] + 1))
        for k in recalls:
            recalls[k].append(float(ranked[:k].max() if len(ranked) else 0.0))
        if len(pos_ids) > 0 and len(neg_ids) > 0:
            pos_s = scores[labels == 1]
            neg_s = scores[labels == 0]
            aucs.append(float(((pos_s[:, None] > neg_s[None, :]).mean() + 0.5 * (pos_s[:, None] == neg_s[None, :]).mean())))
    return {
        "num_samples": int(sum(1 for s in samples if s.split == split)),
        "num_scored": int(len(mrrs)),
        "event_auc": float(np.mean(aucs)) if aucs else 0.0,
        "event_mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "event_recall@1": float(np.mean(recalls[1])) if recalls[1] else 0.0,
        "event_recall@3": float(np.mean(recalls[3])) if recalls[3] else 0.0,
        "event_recall@5": float(np.mean(recalls[5])) if recalls[5] else 0.0,
        "slot_pairwise_cosine": float(np.mean(sims)) if sims else 0.0,
    }


def candidate_slot_metrics(
    samples: list[q.Sample],
    split: str,
    article_to_idx: dict[int, int],
    item_emb: np.ndarray,
    slots: np.ndarray,
    event_index: dict[str, np.ndarray],
    idx_to_article: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    labels, gains, scores, groups = [], [], [], []
    for sample in samples:
        if sample.split != split:
            continue
        slot = slots[sample.sample_id].astype(np.float32)
        slot = slot / np.clip(np.linalg.norm(slot, axis=1, keepdims=True), 1e-8, None)
        if args.rank_task == "future_engagement":
            rows = future_rank_candidates(event_index, sample.sample_id, idx_to_article, args.seed)
            idxs = [row[0] for row in rows]
            gain_rows = [row[2] for row in rows]
            label_rows = [row[3] for row in rows]
        else:
            clicked = set(sample.current_clicked)
            idxs = [article_to_idx.get(aid) for aid in sample.current_inview]
            gain_rows = [float(aid in clicked) for aid in sample.current_inview]
            label_rows = [int(aid in clicked) for aid in sample.current_inview]
        valid = [idx for idx in idxs if idx is not None]
        if not valid:
            continue
        item = item_emb[valid].astype(np.float32)
        item = item / np.clip(np.linalg.norm(item, axis=1, keepdims=True), 1e-8, None)
        s = (item @ slot.T).max(axis=1)
        vi = 0
        for label, gain, idx in zip(label_rows, gain_rows, idxs):
            if idx is None:
                continue
            labels.append(int(label))
            gains.append(float(gain))
            scores.append(float(s[vi]))
            groups.append(sample.sample_id if args.rank_task == "future_engagement" else sample.impression_id)
            vi += 1
    if not labels:
        return {"auc": float("nan"), "logloss": float("nan"), "mrr": float("nan"), "hit1": float("nan"), "ndcg@5": float("nan"), "ndcg@10": float("nan"), "num_rows": 0, "num_impressions": 0, "positive_rate": 0.0}
    return ranking_metrics_for_task(np.asarray(labels), np.asarray(scores), np.asarray(groups), args, gain_labels=np.asarray(gains))


def candidate_blend_metrics(
    samples: list[q.Sample],
    split: str,
    item_emb: np.ndarray,
    source_emb: np.ndarray,
    slots: np.ndarray,
    event_index: dict[str, np.ndarray],
    idx_to_article: np.ndarray,
    args: argparse.Namespace,
    tune: dict[str, float] | None = None,
) -> dict[str, Any]:
    labels, gains, source_scores, slot_scores, groups = [], [], [], [], []
    for sample in samples:
        if sample.split != split:
            continue
        rows = future_rank_candidates(event_index, sample.sample_id, idx_to_article, args.seed)
        if not rows:
            continue
        idxs = [row[0] for row in rows]
        gain_rows = [row[2] for row in rows]
        label_rows = [row[3] for row in rows]
        item = item_emb[idxs].astype(np.float32)
        item_norm = item / np.clip(np.linalg.norm(item, axis=1, keepdims=True), 1e-8, None)
        source = source_emb[sample.sample_id].astype(np.float32)
        source = source / max(1e-8, float(np.linalg.norm(source)))
        slot = slots[sample.sample_id].astype(np.float32)
        slot = slot / np.clip(np.linalg.norm(slot, axis=1, keepdims=True), 1e-8, None)
        source_s = item_norm @ source
        slot_s = (item_norm @ slot.T).max(axis=1)
        labels.extend(label_rows)
        gains.extend(gain_rows)
        source_scores.extend(source_s.tolist())
        slot_scores.extend(slot_s.tolist())
        groups.extend([sample.sample_id] * len(rows))
    if not labels:
        empty = {"auc": float("nan"), "logloss": float("nan"), "mrr": float("nan"), "hit1": float("nan"), "ndcg@5": float("nan"), "ndcg@10": float("nan"), "num_rows": 0, "num_impressions": 0, "positive_rate": 0.0}
        return {"source": empty, "slot": empty, "best_blend": empty, "lambda": 0.0}
    y = np.asarray(labels, dtype=np.int32)
    y_gain = np.asarray(gains, dtype=np.float32)
    g = np.asarray(groups, dtype=np.int64)
    s_source = normalize_per_group(np.asarray(source_scores, dtype=np.float32), g)
    s_slot = normalize_per_group(np.asarray(slot_scores, dtype=np.float32), g)
    source_metric = ranking_metrics_for_task(y, s_source, g, args, gain_labels=y_gain)
    slot_metric = ranking_metrics_for_task(y, s_slot, g, args, gain_labels=y_gain)
    if tune is None:
        best_lam = 0.0
        best_metric = -float("inf")
        best_out = source_metric
        grid = [float(x) for x in args.blend_lambda_grid.split(",") if x.strip()]
        for lam in grid:
            out = ranking_metrics_for_task(y, s_source + lam * s_slot, g, args, gain_labels=y_gain)
            metric = primary_metric(out, args)
            if metric > best_metric:
                best_metric = metric
                best_lam = lam
                best_out = out
    else:
        best_lam = float(tune["lambda"])
        best_out = ranking_metrics_for_task(y, s_source + best_lam * s_slot, g, args, gain_labels=y_gain)
    return {"source": source_metric, "slot": slot_metric, "best_blend": best_out, "lambda": best_lam}


class SlotSetRanker(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        scalar_dim: int,
        num_users: int,
        num_items: int,
        d_model: int,
        nhead: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        use_slots: bool,
        init_slot_dot_scale: float,
    ) -> None:
        super().__init__()
        self.use_slots = use_slots
        self.slot_dot_scale = nn.Parameter(torch.tensor(float(init_slot_dot_scale), dtype=torch.float32))
        self.vec_proj = nn.Linear(emb_dim, d_model)
        self.scalar_proj = nn.Linear(scalar_dim, d_model)
        self.user_emb = nn.Embedding(num_users + 1, d_model, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(5, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.scalar_head = nn.Sequential(nn.LayerNorm(scalar_dim), nn.Linear(scalar_dim, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.scalar_head[-1].weight)
        nn.init.zeros_(self.scalar_head[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_tok = self.user_emb(batch["user_idx"]) + self.type_emb.weight[0]
        source_tok = self.vec_proj(batch["source"]) + self.type_emb.weight[1]
        set_tok = self.vec_proj(batch["candidate_set"]) + self.type_emb.weight[2]
        tokens = [user_tok.unsqueeze(1), source_tok.unsqueeze(1), set_tok.unsqueeze(1)]
        masks = [torch.zeros((batch["source"].shape[0], 3), dtype=torch.bool, device=batch["source"].device)]
        if self.use_slots:
            tokens.append(self.vec_proj(batch["slot_emb"]) + self.type_emb.weight[3])
            masks.append(~batch["slot_mask"])
        item_tok = (
            self.vec_proj(batch["item_emb"])
            + self.scalar_proj(batch["scalar"])
            + self.item_id_emb(batch["item_idx"])
            + self.type_emb.weight[4]
        )
        tokens.append(item_tok)
        masks.append(~batch["mask"])
        x = torch.cat(tokens, dim=1)
        pad_mask = torch.cat(masks, dim=1)
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        ctx_len = 3 + (batch["slot_emb"].shape[1] if self.use_slots else 0)
        item_out = out[:, ctx_len:]
        score = self.head(item_out).squeeze(-1) + self.scalar_head(batch["scalar"]).squeeze(-1)
        if self.use_slots:
            score = score + self.slot_dot_scale * batch["scalar"][..., -3]
        return score


class CandidateAwareAttentionRanker(nn.Module):
    """NRMS/DIN-style listwise ranker for the fixed future-engagement task."""

    def __init__(
        self,
        emb_dim: int,
        scalar_dim: int,
        num_users: int,
        num_items: int,
        d_model: int,
        nhead: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        use_slots: bool,
        init_slot_dot_scale: float,
        use_item_id_emb: bool,
        user_profile_mode: str,
        use_scalar_head: bool,
    ) -> None:
        super().__init__()
        self.use_slots = use_slots
        self.use_item_id_emb = use_item_id_emb
        self.user_profile_mode = user_profile_mode
        self.use_scalar_head = use_scalar_head
        self.slot_dot_scale = nn.Parameter(torch.tensor(float(init_slot_dot_scale), dtype=torch.float32))
        self.vec_proj = nn.Linear(emb_dim, d_model)
        self.scalar_proj = nn.Linear(scalar_dim, d_model)
        self.hist_age_proj = nn.Linear(1, d_model)
        self.user_scalar_proj = nn.Linear(EventBatchBuilder.user_scalar_dim, d_model)
        self.user_emb = nn.Embedding(num_users + 1, d_model, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.hist_type_emb = nn.Embedding(4, d_model, padding_idx=0)
        self.profile_type_emb = nn.Embedding(4, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(8, d_model)
        hist_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        item_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(hist_layer, num_layers=max(1, layers))
        self.item_encoder = nn.TransformerEncoder(item_layer, num_layers=max(1, layers))
        self.item_gate = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.profile_gate = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        feature_dim = d_model * 10 + scalar_dim
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
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(scalar_dim),
            nn.Linear(scalar_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.scalar_head[-1].weight)
        nn.init.zeros_(self.scalar_head[-1].bias)

    def masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask.unsqueeze(-1).float()).sum(dim=1) / denom

    def attend(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bcd,btd->bct", query, key) / math.sqrt(float(query.shape[-1]))
        logits = logits.masked_fill(~mask.unsqueeze(1), -1e4)
        weights = F.softmax(logits, dim=-1)
        weights = weights * mask.unsqueeze(1).float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bct,btd->bcd", weights, value)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user = self.user_emb(batch["user_idx"]) + self.type_emb.weight[0]
        source = self.vec_proj(batch["source"]) + self.type_emb.weight[1]
        candidate_set = self.vec_proj(batch["candidate_set"]) + self.type_emb.weight[2]
        hist = (
            self.vec_proj(batch["hist_emb"])
            + self.hist_type_emb(batch["hist_type"])
            + self.hist_age_proj(batch["hist_age"])
            + self.type_emb.weight[3]
        )
        hist_mask = batch["hist_mask"]
        hist = self.history_encoder(hist, src_key_padding_mask=~hist_mask)
        item = self.vec_proj(batch["item_emb"]) + self.scalar_proj(batch["scalar"]) + self.type_emb.weight[4]
        if self.use_item_id_emb:
            item = item + self.item_id_emb(batch["item_idx"])
        item = self.item_encoder(item, src_key_padding_mask=~batch["mask"])
        hist_ctx = self.attend(item, hist, hist, hist_mask)
        hist_mean = self.masked_mean(hist, hist_mask).unsqueeze(1).expand_as(item)
        gate = self.item_gate(torch.cat([item, hist_ctx], dim=-1))
        item_hist = gate * hist_ctx + (1.0 - gate) * hist_mean

        if self.user_profile_mode != "none":
            profile = (
                self.vec_proj(batch["user_profile_emb"])
                + self.profile_type_emb(batch["user_profile_type"])
                + self.type_emb.weight[5]
            )
            profile_mask = batch["user_profile_mask"]
            profile_ctx = self.attend(item, profile, profile, profile_mask)
            profile_mean = self.masked_mean(profile, profile_mask).unsqueeze(1).expand_as(item)
            profile_gate = self.profile_gate(torch.cat([item, profile_ctx], dim=-1))
            profile_out = profile_gate * profile_ctx + (1.0 - profile_gate) * profile_mean
        else:
            profile_out = torch.zeros_like(item)

        if self.use_slots:
            slot = self.vec_proj(batch["slot_emb"]) + self.type_emb.weight[6]
            slot_mask = batch["slot_mask"]
            slot_ctx = self.attend(item, slot, slot, slot_mask)
            slot_dot = torch.einsum(
                "bcd,bkd->bck",
                F.normalize(batch["item_emb"], dim=-1),
                F.normalize(batch["slot_emb"], dim=-1),
            ).masked_fill(~slot_mask.unsqueeze(1), -1e4).max(dim=-1).values
            slot_dot = torch.where(torch.isfinite(slot_dot), slot_dot, torch.zeros_like(slot_dot))
        else:
            slot_ctx = torch.zeros_like(item)
            slot_dot = torch.zeros_like(batch["labels"])

        user_expand = user.unsqueeze(1).expand_as(item)
        source_expand = source.unsqueeze(1).expand_as(item)
        set_expand = candidate_set.unsqueeze(1).expand_as(item)
        features = torch.cat(
            [
                item,
                item_hist,
                item * item_hist,
                torch.abs(item - item_hist),
                source_expand,
                item * source_expand,
                set_expand,
                user_expand,
                profile_out,
                slot_ctx,
                batch["scalar"],
            ],
            dim=-1,
        )
        score = self.head(features).squeeze(-1)
        if self.use_scalar_head:
            score = score + self.scalar_head(batch["scalar"]).squeeze(-1)
        if self.use_slots:
            score = score + self.slot_dot_scale * slot_dot
        return score


class LinearFeatureRanker(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        d_model: int,
        dropout: float,
        use_slots: bool,
        init_slot_dot_scale: float,
    ) -> None:
        super().__init__()
        self.use_slots = use_slots
        self.slot_dot_scale = nn.Parameter(torch.tensor(float(init_slot_dot_scale), dtype=torch.float32))
        hidden = max(32, d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(scalar_dim),
            nn.Linear(scalar_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        score = self.net(batch["scalar"]).squeeze(-1)
        if self.use_slots:
            score = score + self.slot_dot_scale * batch["scalar"][..., -3]
        return score


@torch.no_grad()
def predict_ranker_full(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, gains, scores, groups = [], [], [], []
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(moved).detach().cpu().numpy()
        label_tensor = batch["eval_labels"] if "eval_labels" in batch else batch["labels"]
        lab = label_tensor.numpy()
        gain = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        group = batch["group_ids"]
        labels.append(lab[mask])
        gains.append(gain[mask])
        scores.append(out[mask])
        groups.append(group[mask])
    return np.concatenate(labels), np.concatenate(gains), np.concatenate(scores), np.concatenate(groups)


@torch.no_grad()
def predict_ranker(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, _, scores, groups = predict_ranker_full(model, loader, device)
    return labels, scores, groups


def lgbm_flat_features(loader: DataLoader, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    eval_ys: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    group_sizes: list[int] = []
    for batch in loader:
        scalar = batch["scalar"].numpy().astype(np.float32, copy=False)
        if args.lgbm_full_slot_features:
            item = batch["item_emb"].numpy().astype(np.float32, copy=False)
            slot = batch["slot_emb"].numpy().astype(np.float32, copy=False)
            slot_mask = batch["slot_mask"].numpy()
            item_norm = item / np.clip(np.linalg.norm(item, axis=-1, keepdims=True), 1e-8, None)
            slot_norm = slot / np.clip(np.linalg.norm(slot, axis=-1, keepdims=True), 1e-8, None)
            slot_scores = np.einsum("bcd,bkd->bck", item_norm, slot_norm).astype(np.float32)
            slot_scores = np.where(slot_mask[:, None, :], slot_scores, 0.0)
            slot_scores = np.sort(slot_scores, axis=-1)[:, :, ::-1]
            k = min(args.lgbm_slot_score_features, slot_scores.shape[-1])
            if k < args.lgbm_slot_score_features:
                padded = np.zeros((*slot_scores.shape[:2], args.lgbm_slot_score_features), dtype=np.float32)
                padded[:, :, :k] = slot_scores[:, :, :k]
                slot_scores = padded
            else:
                slot_scores = slot_scores[:, :, : args.lgbm_slot_score_features]
            scalar = np.concatenate([scalar, slot_scores], axis=-1)
        labels = batch["labels"].numpy().astype(np.float32, copy=False)
        eval_labels = batch["eval_labels"].numpy().astype(np.int32, copy=False)
        mask = batch["mask"].numpy()
        group_ids = batch["group_ids"]
        for bi in range(scalar.shape[0]):
            m = mask[bi]
            size = int(m.sum())
            if size <= 0:
                continue
            xs.append(scalar[bi, m])
            ys.append(labels[bi, m])
            eval_ys.append(eval_labels[bi, m])
            groups.append(group_ids[bi, m])
            group_sizes.append(size)
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(eval_ys, axis=0),
        np.concatenate(groups, axis=0),
        group_sizes,
    )


def train_lgbm_slot_ranker(name: str, loaders: dict[str, DataLoader], args: argparse.Namespace, use_slots: bool) -> dict[str, Any]:
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("LightGBM is required for --ranker-kind lgbm") from exc

    print(json.dumps({name: {"stage": "extract_lgbm_features"}}, ensure_ascii=False), flush=True)
    x_train, y_train_gain, y_train_eval, g_train, train_group = lgbm_flat_features(loaders["ranker_train"], args)
    x_val, y_val_gain, y_val_eval, g_val, val_group = lgbm_flat_features(loaders["ranker_val"], args)
    x_test, y_test_gain, y_test_eval, g_test, _ = lgbm_flat_features(loaders["ranker_test"], args)
    y_train = y_train_eval if args.lgbm_binary_labels else y_train_gain
    if not args.lgbm_binary_labels:
        y_train = np.clip(
            np.rint(y_train_gain * args.lgbm_gain_scale),
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
                    "features": int(x_train.shape[1]),
                    "use_slots": bool(use_slots),
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
        eval_set=[(x_val, y_val_eval)],
        eval_group=[val_group],
        eval_at=[10],
        callbacks=callbacks,
    )
    val_score = model.predict(x_val, num_iteration=model.best_iteration_)
    test_score = model.predict(x_test, num_iteration=model.best_iteration_)
    val = ranking_metrics_for_task(y_val_eval, val_score, g_val, args, gain_labels=y_val_gain)
    test = ranking_metrics_for_task(y_test_eval, test_score, g_test, args, gain_labels=y_test_gain)
    print(
        json.dumps(
            {
                name: {
                    "stage": "done_lgbm",
                    "val_ndcg10": val["ndcg@10"],
                    "test_ndcg10": test["ndcg@10"],
                    "val_primary": primary_metric(val, args),
                    "test_primary": primary_metric(test, args),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return {
        "history": [{"ranker_val": val}],
        "use_slots": use_slots,
        "kind": "lgbm",
        "best_iteration": int(model.best_iteration_ or args.lgbm_estimators),
        "ranker_val": val,
        "ranker_test": test,
    }


def train_slot_ranker(
    name: str,
    samples: list[q.Sample],
    article_to_idx: dict[int, int],
    item_emb: np.ndarray,
    source_emb: np.ndarray,
    event_index: dict[str, np.ndarray],
    articles_by_id: dict[int, dict],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    idx_to_article: np.ndarray,
    slots: np.ndarray | None,
    use_slots: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device)
    loaders = make_loaders(
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
        slots=slots,
        use_slot_features=use_slots,
    )
    if args.ranker_kind == "lgbm":
        return train_lgbm_slot_ranker(name, loaders, args, use_slots)
    if args.ranker_kind == "linear":
        model = LinearFeatureRanker(
            scalar_dim=EventBatchBuilder.scalar_dim,
            d_model=args.d_model,
            dropout=args.dropout,
            use_slots=use_slots,
            init_slot_dot_scale=args.init_slot_dot_scale,
        ).to(device)
    elif args.ranker_kind == "attention":
        model = CandidateAwareAttentionRanker(
            emb_dim=item_emb.shape[1],
            scalar_dim=EventBatchBuilder.scalar_dim,
            num_users=len(user_to_idx),
            num_items=len(item_to_idx),
            d_model=args.d_model,
            nhead=args.heads,
            layers=args.layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            use_slots=use_slots,
            init_slot_dot_scale=args.init_slot_dot_scale,
            use_item_id_emb=args.use_item_id_emb,
            user_profile_mode=args.user_profile_mode,
            use_scalar_head=args.attention_scalar_head,
        ).to(device)
    else:
        model = SlotSetRanker(
            emb_dim=item_emb.shape[1],
            scalar_dim=EventBatchBuilder.scalar_dim,
            num_users=len(user_to_idx),
            num_items=len(item_to_idx),
            d_model=args.d_model,
            nhead=args.heads,
            layers=args.layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            use_slots=use_slots,
            init_slot_dot_scale=args.init_slot_dot_scale,
        ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y0, y0_gain, s0, g0 = predict_ranker_full(model, loaders["ranker_val"], device)
    initial = ranking_metrics_for_task(y0, s0, g0, args, gain_labels=y0_gain)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_metric = primary_metric(initial, args)
    history = [{"epoch": 0, "loss": None, "ranker_val": initial}]
    patience = args.patience
    print(json.dumps({name: {"epoch": 0, "val_mrr": initial["mrr"], "val_ndcg10": initial["ndcg@10"]}}, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["ranker_train"]:
            moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            score = model(moved)
            loss = q.listwise_loss(score, moved["labels"], moved["mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        y, y_gain, s, g = predict_ranker_full(model, loaders["ranker_val"], device)
        val = ranking_metrics_for_task(y, s, g, args, gain_labels=y_gain)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "ranker_val": val}
        history.append(row)
        print(json.dumps({name: {"epoch": epoch, "loss": row["loss"], "val_mrr": val["mrr"], "val_ndcg10": val["ndcg@10"]}}, ensure_ascii=False), flush=True)
        if primary_metric(val, args) > best_metric:
            best_metric = primary_metric(val, args)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    out: dict[str, Any] = {"history": history, "use_slots": use_slots}
    for split, loader in loaders.items():
        y, y_gain, s, g = predict_ranker_full(model, loader, device)
        out[split] = ranking_metrics_for_task(y, s, g, args, gain_labels=y_gain)
    return out


def oracle_slots(event_index: dict[str, np.ndarray], item_emb: np.ndarray, num_slots: int) -> np.ndarray:
    n = event_index["pos_idx"].shape[0]
    slots = np.zeros((n, num_slots, item_emb.shape[1]), dtype=np.float16)
    for start in range(0, n, 2048):
        end = min(n, start + 2048)
        idx = event_index["pos_idx"][start:end, :num_slots]
        mask = event_index["pos_mask"][start:end, :num_slots]
        if "pos_emb" in event_index:
            part = event_index["pos_emb"][start:end, :num_slots].astype(np.float32)
        else:
            part = item_emb[idx].astype(np.float32)
        part[~mask] = 0.0
        slots[start:end] = part.astype(np.float16)
    return slots


def train_event_predictor(
    variant: str,
    samples: list[q.Sample],
    article_to_idx: dict[int, int],
    item_emb: np.ndarray,
    source_emb: np.ndarray,
    event_index: dict[str, np.ndarray],
    articles_by_id: dict[int, dict],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    idx_to_article: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = torch.device(args.device)
    train_ids = [s.sample_id for s in samples if s.split == "ranker_train"]
    if args.rank_supervision_fraction < 1.0:
        rng = np.random.default_rng(args.seed + 17)
        keep = max(1, int(round(len(train_ids) * args.rank_supervision_fraction)))
        supervised = set(int(x) for x in rng.choice(np.asarray(train_ids, dtype=np.int64), size=keep, replace=False).tolist())
    else:
        supervised = set(int(x) for x in train_ids)
    loaders = make_loaders(
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
        rank_supervised_ids=supervised,
    )
    model = EventSlotPredictor(
        emb_dim=item_emb.shape[1],
        scalar_dim=EventBatchBuilder.base_scalar_dim,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        num_slots=args.num_slots,
        d_model=args.predictor_d_model,
        nhead=args.predictor_heads,
        layers=args.predictor_layers,
        ff_dim=args.predictor_ff_dim,
        dropout=args.dropout,
        use_item_id_emb=args.use_item_id_emb,
        score_mode=args.score_mode,
        slot_residual_scale=args.slot_residual_scale,
        init_slot_dot_scale=args.init_slot_dot_scale,
        slot_output_mode=args.slot_output_mode,
        user_profile_mode=args.user_profile_mode,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.predictor_lr, weight_decay=args.weight_decay)
    best_state = None
    best_metric = -float("inf")
    patience = args.predictor_patience
    history = []
    for epoch in range(1, args.predictor_epochs + 1):
        model.train()
        losses = []
        parts: dict[str, list[float]] = defaultdict(list)
        for batch in loaders["ranker_train"]:
            moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            slots, rank_scores = model.forward_with_scores(moved)
            loss, loss_parts = event_loss(slots, moved, variant, args, rank_scores)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in loss_parts.items():
                parts[key].append(value)
        val_slots = predict_slots(model, loaders["ranker_val"], device, len(samples), args.num_slots, item_emb.shape[1])
        val_metrics = sample_event_metrics(samples, "ranker_val", item_emb, event_index, val_slots)
        val_candidate = candidate_slot_metrics(samples, "ranker_val", article_to_idx, item_emb, val_slots, event_index, idx_to_article, args)
        val_rank_head = predict_event_rank_head(model, loaders["ranker_val"], device, args)
        if args.predictor_selection_metric == "rank_head_ndcg":
            metric = primary_metric(val_rank_head, args)
        elif args.predictor_selection_metric == "candidate_ndcg":
            metric = primary_metric(val_candidate, args)
        elif args.predictor_selection_metric == "event_auc":
            metric = val_metrics["event_auc"]
        elif args.predictor_selection_metric == "event_mrr":
            metric = val_metrics["event_mrr"]
        elif args.predictor_selection_metric == "rank_candidate_mix":
            metric = primary_metric(val_rank_head, args) + args.predictor_selection_aux_weight * primary_metric(val_candidate, args)
        else:
            metric = primary_metric(val_rank_head, args) if args.rank_head_weight > 0 else (primary_metric(val_candidate, args) if args.rank_task == "future_engagement" else val_metrics["event_mrr"])
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "loss_parts": {k: float(np.mean(v)) for k, v in parts.items()},
            "event_val": val_metrics,
            "candidate_val": val_candidate,
            "rank_head_val": val_rank_head,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    variant: {
                        "epoch": epoch,
                        "loss": row["loss"],
                        "event_mrr": val_metrics["event_mrr"],
                        "event_auc": val_metrics["event_auc"],
                        "recall@3": val_metrics["event_recall@3"],
                        "slot_pairwise": val_metrics["slot_pairwise_cosine"],
                        "candidate_ndcg10": val_candidate["ndcg@10"],
                        "rank_head_ndcg10": val_rank_head["ndcg@10"],
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.predictor_patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    eval_splits = ["ranker_val", "ranker_test"] if args.skip_train_eval else ["ranker_train", "ranker_val", "ranker_test"]
    slot_splits = (
        ["ranker_train", "ranker_val", "ranker_test"]
        if not args.skip_predicted_slot_ranker
        else eval_splits
    )
    all_slots = np.zeros((len(samples), args.num_slots, item_emb.shape[1]), dtype=np.float16)
    for split in slot_splits:
        loader = loaders[split]
        pred = predict_slots(model, loader, device, len(samples), args.num_slots, item_emb.shape[1])
        ids = [s.sample_id for s in samples if s.split == split]
        all_slots[ids] = pred[ids]
    split_eval = {}
    for split in eval_splits:
        split_eval[split] = {
            "event": sample_event_metrics(samples, split, item_emb, event_index, all_slots),
            "candidate": candidate_slot_metrics(samples, split, article_to_idx, item_emb, all_slots, event_index, idx_to_article, args),
            "rank_head": predict_event_rank_head(model, loaders[split], device, args),
        }
    if args.rank_task == "future_engagement":
        if not args.skip_rank_slot_blend:
            split_eval["rank_head_slot_blend"] = rank_head_slot_blend_metrics(
                model,
                loaders["ranker_val"],
                loaders["ranker_test"],
                device,
                args,
            )
        if not args.skip_source_slot_blend:
            val_blend = candidate_blend_metrics(samples, "ranker_val", item_emb, source_emb, all_slots, event_index, idx_to_article, args)
            test_blend = candidate_blend_metrics(
                samples,
                "ranker_test",
                item_emb,
                source_emb,
                all_slots,
                event_index,
                idx_to_article,
                args,
                tune={"lambda": val_blend["lambda"]},
            )
            split_eval["blend"] = {"ranker_val": val_blend, "ranker_test": test_blend}
    return all_slots, {
        "history": history,
        "split_eval": split_eval,
        "rank_supervision": {
            "fraction": args.rank_supervision_fraction,
            "num_supervised_train_queries": len(supervised),
            "num_train_queries": len(train_ids),
        },
    }


def compact_summary(results: dict[str, Any]) -> dict[str, Any]:
    rankers = {}
    for name, payload in results.get("ranker_models", {}).items():
        test = payload.get("ranker_test", {})
        rankers[name] = {
            key: test.get(key)
            for key in ["auc", "mrr", "hit1", "ndcg@10", "engagement_ndcg@10", "engagement_mean_gain@1", "logloss"]
            if key in test
        }
    predictors = {}
    for name, payload in results.get("predictors", {}).items():
        test = payload.get("split_eval", {}).get("ranker_test", {})
        event = test.get("event", {})
        candidate = test.get("candidate", {})
        rank_head = test.get("rank_head", {})
        rank_slot = payload.get("split_eval", {}).get("rank_head_slot_blend", {}).get("ranker_test", {})
        rank_slot_best = rank_slot.get("best_blend", {})
        blend = payload.get("split_eval", {}).get("blend", {}).get("ranker_test", {})
        best_blend = blend.get("best_blend", {})
        predictors[name] = {
            "event_auc": event.get("event_auc"),
            "event_mrr": event.get("event_mrr"),
            "event_recall@1": event.get("event_recall@1"),
            "event_recall@3": event.get("event_recall@3"),
            "event_recall@5": event.get("event_recall@5"),
            "slot_pairwise_cosine": event.get("slot_pairwise_cosine"),
            "candidate_ndcg@10": candidate.get("ndcg@10"),
            "candidate_engagement_ndcg@10": candidate.get("engagement_ndcg@10"),
            "rank_head_ndcg@10": rank_head.get("ndcg@10"),
            "rank_head_engagement_ndcg@10": rank_head.get("engagement_ndcg@10"),
            "rank_head_mrr": rank_head.get("mrr"),
            "rank_slot_lambda": rank_slot.get("lambda"),
            "rank_slot_blend_ndcg@10": rank_slot_best.get("ndcg@10"),
            "rank_slot_blend_engagement_ndcg@10": rank_slot_best.get("engagement_ndcg@10"),
            "rank_slot_blend_mrr": rank_slot_best.get("mrr"),
            "blend_lambda": blend.get("lambda"),
            "blend_ndcg@10": best_blend.get("ndcg@10"),
            "blend_engagement_ndcg@10": best_blend.get("engagement_ndcg@10"),
            "blend_mrr": best_blend.get("mrr"),
        }
    return {
        "setting": results.get("setting", {}),
        "split_counts": results.get("split_counts", {}),
        "event_summary": results.get("event_summary", {}),
        "predictors": predictors,
        "rankers": rankers,
        "oracle_event_metrics": results.get("oracle_event_metrics", {}),
        "oracle_blend_metrics": results.get("oracle_blend_metrics", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-pooled future UIH event-set predictor for EB-NeRD.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qwen-raw-cache", required=True)
    parser.add_argument("--reaction-qwen-cache", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-svd-oracle-results", default="")
    parser.add_argument("--target-mode", choices=["current_future", "future_only"], default="current_future")
    parser.add_argument("--event-target-embedding", choices=["article_qwen", "reaction_qwen_text"], default="article_qwen")
    parser.add_argument("--rank-task", choices=["current_click", "future_engagement"], default="current_click")
    parser.add_argument("--loss-variants", default="E1_event_nce,E2_event_nce_bpr,E3_event_hybrid")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--max-pos-events", type=int, default=16)
    parser.add_argument("--max-neg-events", type=int, default=32)
    parser.add_argument("--max-rank-candidates", type=int, default=256)
    parser.add_argument("--max-future-impressions", type=int, default=8)
    parser.add_argument("--max-events-per-impression", type=int, default=20)
    parser.add_argument("--future-horizon-hours", type=float, default=24.0)
    parser.add_argument("--include-current-neg", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--source-mode", choices=["qwen_cache", "history_clicked_mean", "history_weighted"], default="qwen_cache")
    parser.add_argument("--max-source-history", type=int, default=80)
    parser.add_argument("--max-history-tokens", type=int, default=96)
    parser.add_argument("--reaction-body-chars", type=int, default=0)
    parser.add_argument("--user-profile-mode", choices=["none", "history_summary"], default="none")
    parser.add_argument("--source-negative-weight", type=float, default=0.25)
    parser.add_argument("--source-shown-weight", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--predictor-epochs", type=int, default=6)
    parser.add_argument("--predictor-patience", type=int, default=2)
    parser.add_argument("--predictor-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ranker-kind", choices=["transformer", "linear", "lgbm", "attention"], default="transformer")
    parser.add_argument("--attention-scalar-head", action="store_true")
    parser.add_argument("--use-item-id-emb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=768)
    parser.add_argument("--predictor-d-model", type=int, default=384)
    parser.add_argument("--predictor-heads", type=int, default=8)
    parser.add_argument("--predictor-layers", type=int, default=3)
    parser.add_argument("--predictor-ff-dim", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--event-temperature", type=float, default=0.07)
    parser.add_argument("--bpr-temperature", type=float, default=0.10)
    parser.add_argument("--candidate-temperature", type=float, default=0.07)
    parser.add_argument("--candidate-bce-temperature", type=float, default=0.10)
    parser.add_argument("--nce-weight", type=float, default=1.0)
    parser.add_argument("--slot-nce-weight", type=float, default=1.0)
    parser.add_argument("--assignment-weight", type=float, default=1.0)
    parser.add_argument("--bpr-weight", type=float, default=0.5)
    parser.add_argument("--candidate-kl-weight", type=float, default=0.5)
    parser.add_argument("--candidate-bce-weight", type=float, default=0.0)
    parser.add_argument("--rank-label-weight", type=float, default=0.0)
    parser.add_argument("--rank-head-weight", type=float, default=0.0)
    parser.add_argument("--rank-supervision-fraction", type=float, default=1.0)
    parser.add_argument(
        "--score-mode",
        choices=[
            "plain",
            "slot_residual",
            "gated_slot_residual",
            "slot_dot",
            "slot_dot_residual",
            "gated_slot_dot_residual",
        ],
        default="plain",
    )
    parser.add_argument("--slot-residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--slot-output-mode",
        choices=["direct", "candidate_attention", "residual_candidate_attention"],
        default="direct",
    )
    parser.add_argument(
        "--predictor-selection-metric",
        choices=["auto", "rank_head_ndcg", "candidate_ndcg", "event_auc", "event_mrr", "rank_candidate_mix"],
        default="auto",
    )
    parser.add_argument("--predictor-selection-aux-weight", type=float, default=0.25)
    parser.add_argument("--diversity-weight", type=float, default=0.02)
    parser.add_argument("--diversity-margin", type=float, default=0.10)
    parser.add_argument("--init-slot-dot-scale", type=float, default=5.0)
    parser.add_argument("--lgbm-estimators", type=int, default=250)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.04)
    parser.add_argument("--lgbm-num-leaves", type=int, default=63)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=50)
    parser.add_argument("--lgbm-subsample", type=float, default=0.9)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=2.0)
    parser.add_argument("--lgbm-n-jobs", type=int, default=16)
    parser.add_argument("--lgbm-early-stopping-rounds", type=int, default=30)
    parser.add_argument("--lgbm-binary-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lgbm-gain-scale", type=float, default=2.0)
    parser.add_argument("--lgbm-max-gain-label", type=int, default=15)
    parser.add_argument("--lgbm-full-slot-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lgbm-slot-score-features", type=int, default=8)
    parser.add_argument("--primary-rank-metric", choices=["ndcg@10", "engagement_ndcg@10"], default="ndcg@10")
    parser.add_argument("--blend-lambda-grid", default="-2,-1,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,3,4")
    parser.add_argument("--rank-slot-blend-lambda-grid", default="-2,-1,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,3,4")
    parser.add_argument("--skip-rank-slot-blend", action="store_true")
    parser.add_argument("--skip-source-slot-blend", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-baseline-rankers", action="store_true")
    parser.add_argument("--skip-predicted-slot-ranker", action="store_true")
    parser.add_argument("--skip-train-eval", action="store_true")
    parser.add_argument("--save-predicted-slots", action="store_true")
    parser.add_argument("--load-predicted-slots", default="")
    parser.add_argument("--loaded-slot-name", default="loaded")
    args = parser.parse_args()

    q.set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_samples, prepare = load_samples(Path(args.samples_pkl))
    samples, cache_idx = select_ranker_samples(all_samples, args.max_samples_per_split)
    cache = np.load(args.qwen_raw_cache)
    article_ids = cache["article_ids"].astype(np.int64)
    idx_to_article = article_ids.astype(np.int64)
    item_emb = cache["article_emb"].astype(np.float32)
    qwen_source_emb = cache["source_emb"].astype(np.float32)[cache_idx]
    article_to_idx = {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}
    source_emb = build_source_embeddings(samples, article_to_idx, item_emb, qwen_source_emb, args)
    data_dir = Path(args.data_dir)
    articles_df = pd.read_parquet(data_dir / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    articles_by_id = q.article_lookup(articles_df)
    full_index = q.FullSignalIndex(data_dir)
    users = sorted({s.user_id for s in samples})
    items = sorted(int(aid) for aid in article_ids.tolist())
    user_to_idx = {uid: idx + 1 for idx, uid in enumerate(users)}
    item_to_idx = {aid: idx + 1 for idx, aid in enumerate(items)}

    print("building event index", flush=True)
    event_index = build_event_index(samples, article_to_idx, full_index, args)
    summary_values = event_index.pop("summary")
    reaction_meta: dict[str, Any] | None = None
    if args.event_target_embedding == "reaction_qwen_text":
        if not args.reaction_qwen_cache:
            raise ValueError("--reaction-qwen-cache is required when --event-target-embedding reaction_qwen_text")
        pos_emb, neg_emb, reaction_meta = load_reaction_qwen_cache(
            Path(args.reaction_qwen_cache),
            len(samples),
            args.max_pos_events,
            args.max_neg_events,
            item_emb.shape[1],
        )
        event_index["pos_emb"] = pos_emb
        event_index["neg_emb"] = neg_emb
        print(
            json.dumps(
                {
                    "reaction_qwen_cache": str(args.reaction_qwen_cache),
                    "pos_emb": list(pos_emb.shape),
                    "neg_emb": list(neg_emb.shape),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    event_summary = {
        "samples": int(summary_values[0]),
        "pos_events": int(summary_values[1]),
        "neg_events": int(summary_values[2]),
        "empty_pos": int(summary_values[3]),
        "empty_neg": int(summary_values[4]),
        "avg_pos_per_sample": float(summary_values[1] / max(1, summary_values[0])),
        "avg_neg_per_sample": float(summary_values[2] / max(1, summary_values[0])),
        "rank_candidates": int(summary_values[5]) if len(summary_values) > 5 else 0,
        "rank_positive": int(summary_values[6]) if len(summary_values) > 6 else 0,
        "rank_empty": int(summary_values[7]) if len(summary_values) > 7 else 0,
        "rank_truncated": int(summary_values[8]) if len(summary_values) > 8 else 0,
        "avg_rank_candidates_per_sample": float(summary_values[5] / max(1, summary_values[0])) if len(summary_values) > 5 else 0.0,
    }
    print(json.dumps({"event_summary": event_summary}, ensure_ascii=False), flush=True)

    results: dict[str, Any] = {
        "prepare": prepare,
        "qwen_raw_cache": str(args.qwen_raw_cache),
        "reaction_qwen_cache": str(args.reaction_qwen_cache) if args.reaction_qwen_cache else "",
        "no_svd_oracle_results": json.loads(Path(args.no_svd_oracle_results).read_text()) if args.no_svd_oracle_results else None,
        "setting": {
            "target_mode": args.target_mode,
            "event_target_embedding": args.event_target_embedding,
            "rank_task": args.rank_task,
            "future_horizon_hours": args.future_horizon_hours,
            "max_future_impressions": args.max_future_impressions,
            "max_events_per_impression": args.max_events_per_impression,
            "max_rank_candidates": args.max_rank_candidates,
            "source_mode": args.source_mode,
            "max_history_tokens": args.max_history_tokens,
            "user_profile_mode": args.user_profile_mode,
            "ranker_kind": args.ranker_kind,
            "use_item_id_emb": args.use_item_id_emb,
            "score_mode": args.score_mode,
            "slot_residual_scale": args.slot_residual_scale,
            "init_slot_dot_scale": args.init_slot_dot_scale,
            "slot_output_mode": args.slot_output_mode,
            "rank_supervision_fraction": args.rank_supervision_fraction,
            "predictor_selection_metric": args.predictor_selection_metric,
            "predictor_selection_aux_weight": args.predictor_selection_aux_weight,
            "primary_rank_metric": args.primary_rank_metric,
            "target": "future UIH event set, not pooled future vector",
            "event_repr": (
                "frozen Qwen embedding of full reaction-event text"
                if args.event_target_embedding == "reaction_qwen_text"
                else "raw Qwen article embedding with read/scroll/time-gap derived event weights"
            ),
            "reaction_qwen_meta": reaction_meta,
            "rank_label": (
                "current impression click label"
                if args.rank_task == "current_click"
                else "flattened future exposed item engagement label for each anchor user-time query"
            ),
            "rank_unit": (
                "current logged impression"
                if args.rank_task == "current_click"
                else "anchor user-time query; candidates are all later exposed items in the future window"
            ),
            "num_slots": args.num_slots,
            "loss_variants": [v.strip() for v in args.loss_variants.split(",") if v.strip()],
            "anti_collapse": {
                "slot_nce_weight": args.slot_nce_weight,
                "assignment_weight": args.assignment_weight,
                "candidate_bce_weight": args.candidate_bce_weight,
                "diversity_weight": args.diversity_weight,
                "diversity_margin": args.diversity_margin,
            },
            "eval_shortcuts": {
                "skip_oracle": args.skip_oracle,
                "skip_baseline_rankers": args.skip_baseline_rankers,
                "skip_predicted_slot_ranker": args.skip_predicted_slot_ranker,
                "skip_train_eval": args.skip_train_eval,
                "skip_rank_slot_blend": args.skip_rank_slot_blend,
                "skip_source_slot_blend": args.skip_source_slot_blend,
            },
        },
        "split_counts": dict(Counter(s.split for s in samples)),
        "event_summary": event_summary,
        "predictors": {},
        "ranker_models": {},
    }
    q.write_json(out_dir / "results.partial.json", results)

    o_slots = None
    if args.skip_oracle:
        print("skipping oracle slots", flush=True)
    else:
        print("building oracle slots", flush=True)
        o_slots = oracle_slots(event_index, item_emb, args.num_slots)
        results["oracle_event_metrics"] = {
            split: {
                "event": sample_event_metrics(samples, split, item_emb, event_index, o_slots),
                "candidate": candidate_slot_metrics(samples, split, article_to_idx, item_emb, o_slots, event_index, idx_to_article, args),
            }
            for split in ["ranker_val", "ranker_test"]
        }
        if args.rank_task == "future_engagement":
            oracle_val_blend = candidate_blend_metrics(samples, "ranker_val", item_emb, source_emb, o_slots, event_index, idx_to_article, args)
            oracle_test_blend = candidate_blend_metrics(
                samples,
                "ranker_test",
                item_emb,
                source_emb,
                o_slots,
                event_index,
                idx_to_article,
                args,
                tune={"lambda": oracle_val_blend["lambda"]},
            )
            results["oracle_blend_metrics"] = {"ranker_val": oracle_val_blend, "ranker_test": oracle_test_blend}
        q.write_json(out_dir / "results.partial.json", results)

    def add_ranker(name: str, slots: np.ndarray | None, use_slots: bool) -> None:
        print(f"training {name}", flush=True)
        results["ranker_models"][name] = train_slot_ranker(
            name,
            samples,
            article_to_idx,
            item_emb,
            source_emb,
            event_index,
            articles_by_id,
            user_to_idx,
            item_to_idx,
            idx_to_article,
            slots,
            use_slots,
            args,
        )
        print(json.dumps({name: results["ranker_models"][name]["ranker_test"]}, indent=2), flush=True)
        q.write_json(out_dir / "results.partial.json", results)

    if args.skip_baseline_rankers:
        print("skipping baseline/oracle rankers", flush=True)
    else:
        add_ranker("S0_source_only_slot_ranker", None, False)
        if o_slots is not None:
            add_ranker("O_event_true_future_slots", o_slots, True)
    del o_slots

    if args.load_predicted_slots:
        slot_path = Path(args.load_predicted_slots)
        print(f"loading predicted slots {slot_path}", flush=True)
        loaded = np.load(slot_path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded_slots = loaded["slots"]
        else:
            loaded_slots = loaded
        loaded_slots = loaded_slots.astype(np.float16, copy=False)
        loaded_name = args.loaded_slot_name.strip() or "loaded"
        slot_eval = {}
        for split in ["ranker_val", "ranker_test"]:
            slot_eval[split] = {
                "event": sample_event_metrics(samples, split, item_emb, event_index, loaded_slots),
                "candidate": candidate_slot_metrics(samples, split, article_to_idx, item_emb, loaded_slots, event_index, idx_to_article, args),
            }
        results.setdefault("loaded_slots", {})[loaded_name] = {
            "slot_cache": str(slot_path),
            "split_eval": slot_eval,
        }
        add_ranker(f"P_{loaded_name}_slots", loaded_slots, True)
        del loaded_slots

    for variant in [v.strip() for v in args.loss_variants.split(",") if v.strip()]:
        print(f"training predictor {variant}", flush=True)
        slots, metrics = train_event_predictor(
            variant,
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
        )
        results["predictors"][variant] = metrics
        if args.save_predicted_slots:
            slot_path = out_dir / f"{variant}_slots.fp16.npy"
            np.save(slot_path, slots.astype(np.float16, copy=False))
            results["predictors"][variant]["slot_cache"] = str(slot_path)
        q.write_json(out_dir / "results.partial.json", results)
        if args.skip_predicted_slot_ranker:
            print(f"skipping P_{variant}_slots ranker", flush=True)
        else:
            add_ranker(f"P_{variant}_slots", slots, True)
        del slots

    q.write_json(out_dir / "results.json", results)
    print(json.dumps({"summary": compact_summary(results)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
