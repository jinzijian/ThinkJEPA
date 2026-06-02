#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

import ebnerd_event_set_predictor as ep
import ebnerd_qwen_uih_text_jepa as q


@dataclass
class CandidateReaction:
    article_idx: int
    article_id: int
    gain: float
    label: int
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
    return ep.select_ranker_samples(samples, max_per_split)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_logloss(y: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1]))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(float(mean_squared_error(y, pred))))


def unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8, None)


def torch_unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


def collect_candidate_reactions(
    sample: q.Sample,
    article_to_idx: dict[int, int],
    full_index: q.FullSignalIndex,
    args: argparse.Namespace,
) -> list[CandidateReaction]:
    rows, current_idx, current_row = full_index.current(sample)
    if current_idx is None:
        return []
    current_time = pd.Timestamp(sample.time if current_row is None else getattr(current_row, "impression_time"))
    horizon_end = current_time + pd.Timedelta(hours=args.future_horizon_hours)
    by_idx: dict[int, CandidateReaction] = {}
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
        inview = q.ids(getattr(fut, "article_ids_inview"))[: args.max_events_per_impression]
        read = q.clean_float(getattr(fut, "read_time", 0.0))
        scroll = q.clean_float(getattr(fut, "scroll_percentage", 0.0))
        event_time_text = fut_time.strftime("%Y-%m-%d %H:%M:%S")
        for aid in inview:
            idx = article_to_idx.get(int(aid))
            if idx is None:
                continue
            is_click = int(aid) in clicked
            gain = ep.event_weight(read, scroll, dt_h, positive=True) if is_click else 0.0
            rec = CandidateReaction(
                article_idx=int(idx),
                article_id=int(aid),
                gain=float(gain),
                label=int(is_click),
                event_kind="future_clicked" if is_click else "future_not_clicked_exposure",
                clicked=bool(is_click),
                read_time=float(read if is_click else 0.0),
                scroll=float(scroll if is_click else 0.0),
                next_read_time=q.clean_float(getattr(fut, "next_read_time", 0.0)) if is_click else 0.0,
                next_scroll=q.clean_float(getattr(fut, "next_scroll_percentage", 0.0)) if is_click else 0.0,
                dt_hours=float(dt_h),
                event_time=event_time_text,
                session_id=q.clean_id(getattr(fut, "session_id", None)) or 0,
                device_type=q.clean_id(getattr(fut, "device_type", None)) or 0,
                candidate_count=len(inview),
                clicked_count=len(clicked),
            )
            prev = by_idx.get(rec.article_idx)
            if prev is None:
                by_idx[rec.article_idx] = rec
            elif (rec.label, rec.gain, -rec.dt_hours) > (prev.label, prev.gain, -prev.dt_hours):
                by_idx[rec.article_idx] = rec
    rows_out = list(by_idx.values())
    rng = np.random.default_rng(int(args.seed) + int(sample.sample_id) * 1299721)
    if len(rows_out) > args.max_candidates:
        pos = [r for r in rows_out if r.label > 0]
        neg = [r for r in rows_out if r.label <= 0]
        pos.sort(key=lambda r: (-r.gain, r.dt_hours, r.article_id))
        rng.shuffle(neg)
        if len(pos) >= args.max_candidates:
            rows_out = pos[: args.max_candidates]
        else:
            rows_out = pos + neg[: args.max_candidates - len(pos)]
    rng.shuffle(rows_out)
    return rows_out


def build_candidate_index(
    samples: list[q.Sample],
    article_to_idx: dict[int, int],
    idx_to_article: np.ndarray,
    full_index: q.FullSignalIndex,
    args: argparse.Namespace,
) -> dict[str, Any]:
    n = len(samples)
    c = args.max_candidates
    cand_idx = np.zeros((n, c), dtype=np.int32)
    cand_aid = np.zeros((n, c), dtype=np.int64)
    gain = np.zeros((n, c), dtype=np.float32)
    label = np.zeros((n, c), dtype=np.int32)
    read_time = np.zeros((n, c), dtype=np.float32)
    scroll = np.zeros((n, c), dtype=np.float32)
    dt_hours = np.zeros((n, c), dtype=np.float32)
    mask = np.zeros((n, c), dtype=bool)
    shuffled_order = np.full((n, c), -1, dtype=np.int32)
    oracle_order = np.full((n, c), -1, dtype=np.int32)
    records: list[list[CandidateReaction]] = []
    stats = Counter()
    for sample in samples:
        sid = sample.sample_id
        rows = collect_candidate_reactions(sample, article_to_idx, full_index, args)
        records.append(rows)
        stats["samples"] += 1
        stats["candidates"] += len(rows)
        stats["positives"] += sum(r.label for r in rows)
        stats["empty"] += int(len(rows) == 0)
        stats["truncated"] += int(len(rows) >= args.max_candidates)
        for j, rec in enumerate(rows[:c]):
            cand_idx[sid, j] = rec.article_idx
            cand_aid[sid, j] = rec.article_id
            gain[sid, j] = rec.gain
            label[sid, j] = rec.label
            read_time[sid, j] = rec.read_time
            scroll[sid, j] = rec.scroll
            dt_hours[sid, j] = rec.dt_hours
            mask[sid, j] = True
        valid = np.arange(len(rows[:c]), dtype=np.int32)
        shuffled_order[sid, : len(valid)] = valid
        oracle = sorted(valid.tolist(), key=lambda j: (-gain[sid, j], -label[sid, j], int(cand_aid[sid, j])))
        oracle_order[sid, : len(oracle)] = np.asarray(oracle, dtype=np.int32)
    summary = {
        "samples": int(stats["samples"]),
        "candidates": int(stats["candidates"]),
        "positives": int(stats["positives"]),
        "empty": int(stats["empty"]),
        "truncated": int(stats["truncated"]),
        "avg_candidates_per_sample": float(stats["candidates"] / max(1, stats["samples"])),
        "avg_positives_per_sample": float(stats["positives"] / max(1, stats["samples"])),
        "max_candidates": int(args.max_candidates),
    }
    return {
        "records": records,
        "cand_idx": cand_idx,
        "cand_aid": cand_aid,
        "gain": gain,
        "label": label,
        "read_time": read_time,
        "scroll": scroll,
        "dt_hours": dt_hours,
        "mask": mask,
        "orders": {
            "shuffled": shuffled_order,
            "oracle": oracle_order,
            "baseline": shuffled_order.copy(),
        },
        "summary": summary,
        "idx_to_article": idx_to_article,
    }


def make_reaction_target_text(
    sample: q.Sample,
    rec: CandidateReaction,
    ordered: list[CandidateReaction],
    rank: int,
    rich_articles: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    clicked = "yes" if rec.clicked else "no"
    context_lines = []
    for pos, item in enumerate(ordered[: args.target_list_context_items], start=1):
        context_lines.append(
            f"candidate_{pos:02d}: {q.compact_article(item.article_id, rich_articles, include_extra=False, body_chars=0)}"
        )
    if not context_lines:
        context_lines.append("candidate_list: none")
    article_text = q.compact_article(
        rec.article_id,
        rich_articles,
        include_extra=True,
        body_chars=args.reaction_body_chars,
    )
    return "\n".join(
        [
            "Task: encode one ordered-list user-item reaction for future UIH prediction.",
            f"User: {sample.user_id}",
            f"Anchor time: {pd.Timestamp(sample.time).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Target candidate position: {rank + 1} of {len(ordered)}",
            "Ordered candidate list, feedback hidden:",
            *context_lines,
            (
                "Impression context: "
                f"event_kind={rec.event_kind} ; "
                f"dt_h={rec.dt_hours:.2f} ; "
                f"event_time={rec.event_time} ; "
                f"session_id={rec.session_id} ; "
                f"device_type={rec.device_type} ; "
                f"candidate_count={rec.candidate_count} ; "
                f"clicked_count={rec.clicked_count}"
            ),
            (
                "User reaction to target item: "
                f"clicked={clicked} ; "
                f"read_s={rec.read_time:.1f} ; "
                f"scroll_pct={rec.scroll:.1f} ; "
                f"next_read_s={rec.next_read_time:.1f} ; "
                f"next_scroll_pct={rec.next_scroll:.1f} ; "
                f"engagement_gain={rec.gain:.4f}"
            ),
            f"Target item: {article_text}",
        ]
    )


def ordered_records(index: dict[str, Any], sample_id: int, order_name: str) -> list[CandidateReaction]:
    order = index["orders"][order_name][sample_id]
    records = index["records"][sample_id]
    out = []
    for pos in order:
        if pos < 0:
            continue
        if int(pos) < len(records):
            out.append(records[int(pos)])
    return out


def build_or_load_target_cache(
    samples: list[q.Sample],
    index: dict[str, Any],
    rich_articles: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    cache_path = Path(args.reaction_target_cache)
    orders = [name.strip() for name in args.target_order_caches.split(",") if name.strip()]
    if args.target_cache_format == "npy_dir":
        cache_path.mkdir(parents=True, exist_ok=True)
        arrays = {name: cache_path / f"target_{name}.npy" for name in orders}
        success_path = cache_path / "_SUCCESS"
        if not args.rebuild_target_cache and success_path.exists() and all(path.exists() for path in arrays.values()):
            out = {name: np.load(path, mmap_mode="r") for name, path in arrays.items()}
            print(
                json.dumps(
                    {
                        "reaction_target_cache_dir": str(cache_path),
                        "loaded_orders": sorted(out),
                        "format": "npy_dir",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return out
    else:
        arrays = {}
    if args.target_cache_format != "npy_dir" and cache_path.exists() and not args.rebuild_target_cache:
        packed = np.load(cache_path)
        out = {name: packed[f"target_{name}"].astype(np.float16, copy=False) for name in orders if f"target_{name}" in packed}
        if all(name in out for name in orders):
            print(json.dumps({"reaction_target_cache": str(cache_path), "loaded_orders": sorted(out)}, ensure_ascii=False), flush=True)
            return out
    from sentence_transformers import SentenceTransformer

    emb_dim = int(args.embedding_dim)
    dtype = np.float16 if args.save_float16_targets else np.float32
    if args.target_cache_format == "npy_dir":
        targets = {
            name: np.lib.format.open_memmap(
                arrays[name],
                mode="w+",
                dtype=dtype,
                shape=(len(samples), args.max_candidates, emb_dim),
            )
            for name in orders
        }
    else:
        targets = {name: np.zeros((len(samples), args.max_candidates, emb_dim), dtype=dtype) for name in orders}
    examples: dict[str, list[str]] = {name: [] for name in orders}
    pending_texts: list[str] = []
    pending_locs: list[tuple[str, int, int]] = []

    print(json.dumps({"building_reaction_target_cache": str(cache_path), "orders": orders}, ensure_ascii=False), flush=True)
    model = SentenceTransformer(args.qwen_model, device=args.qwen_device, local_files_only=True)
    model.max_seq_length = args.max_seq_length
    pool = None
    if args.qwen_multi_process_devices:
        devices = [device.strip() for device in args.qwen_multi_process_devices.split(",") if device.strip()]
        print(json.dumps({"qwen_multi_process_devices": devices}, ensure_ascii=False), flush=True)
        pool = model.start_multi_process_pool(target_devices=devices)

    def flush(desc: str) -> None:
        nonlocal pending_texts, pending_locs
        if not pending_texts:
            return
        encoded = q.encode_texts(model, pending_texts, args.qwen_batch_size, True, desc, pool=pool).astype(dtype, copy=False)
        for vec, (order_name, sid, rank) in zip(encoded, pending_locs):
            targets[order_name][sid, rank] = vec
        pending_texts = []
        pending_locs = []

    try:
        for order_name in orders:
            for sample in samples:
                ordered = ordered_records(index, sample.sample_id, order_name)
                for rank, rec in enumerate(ordered[: args.max_candidates]):
                    text = make_reaction_target_text(sample, rec, ordered, rank, rich_articles, args)
                    pending_texts.append(text)
                    pending_locs.append((order_name, sample.sample_id, rank))
                    if len(examples[order_name]) < 2:
                        examples[order_name].append(text)
                if len(pending_texts) >= args.encode_text_chunk_size:
                    flush(f"reaction_targets_{order_name}_until_{sample.sample_id}")
        flush("reaction_targets_final")
    finally:
        if pool is not None:
            SentenceTransformer.stop_multi_process_pool(pool)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.target_cache_format == "npy_dir":
        for value in targets.values():
            value.flush()
        meta_path = cache_path / "meta.json"
        example_path = cache_path / "examples.json"
        success_path = cache_path / "_SUCCESS"
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **{f"target_{name}": value for name, value in targets.items()})
        meta_path = cache_path.with_suffix(".json")
        example_path = cache_path.with_suffix(".examples.json")
    meta = {
        "target": "per-candidate frozen Qwen reaction-event text embedding",
        "orders": orders,
        "num_samples": len(samples),
        "max_candidates": args.max_candidates,
        "embedding_dim": emb_dim,
        "qwen_model": args.qwen_model,
        "future_horizon_hours": args.future_horizon_hours,
        "target_list_context_items": args.target_list_context_items,
        "format": args.target_cache_format,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    example_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.target_cache_format == "npy_dir":
        success_path.write_text("ok\n", encoding="utf-8")
    print(json.dumps({"wrote": str(cache_path), "meta": meta}, ensure_ascii=False), flush=True)
    return targets


class ReactionDataset(Dataset):
    def __init__(self, samples: list[q.Sample], split: str):
        self.samples = [s for s in samples if s.split == split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> q.Sample:
        return self.samples[idx]


class ReactionBatchBuilder:
    base_scalar_dim = q.SetBatchBuilder.scalar_dim
    user_scalar_dim = ep.EventBatchBuilder.user_scalar_dim
    user_profile_tokens = ep.EventBatchBuilder.user_profile_tokens

    def __init__(
        self,
        samples: list[q.Sample],
        article_to_idx: dict[int, int],
        item_emb: np.ndarray,
        source_emb: np.ndarray,
        candidate_index: dict[str, Any],
        target_emb: np.ndarray | None,
        reaction_emb: np.ndarray | None,
        articles_by_id: dict[int, dict[str, Any]],
        user_to_idx: dict[int, int],
        item_to_idx: dict[int, int],
        order_name: str,
        max_history_tokens: int,
        user_profile_mode: str,
        input_mode: str,
    ) -> None:
        self.samples = samples
        self.article_to_idx = article_to_idx
        self.item_emb = item_emb.astype(np.float32, copy=False)
        self.source_emb = source_emb.astype(np.float32, copy=False)
        self.candidate_index = candidate_index
        self.target_emb = target_emb
        self.reaction_emb = reaction_emb
        self.articles_by_id = articles_by_id
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.order_name = order_name
        self.max_history_tokens = max_history_tokens
        self.user_profile_mode = user_profile_mode
        self.input_mode = input_mode

    def __call__(self, samples: list[q.Sample]) -> dict[str, torch.Tensor | np.ndarray]:
        bsz = len(samples)
        c = self.candidate_index["cand_idx"].shape[1]
        d = self.item_emb.shape[1]
        item = np.zeros((bsz, c, d), dtype=np.float32)
        target = np.zeros((bsz, c, d), dtype=np.float32)
        reaction = np.zeros((bsz, c, d), dtype=np.float32)
        scalar = np.zeros((bsz, c, self.base_scalar_dim), dtype=np.float32)
        labels = np.zeros((bsz, c), dtype=np.float32)
        gains = np.zeros((bsz, c), dtype=np.float32)
        read_time = np.zeros((bsz, c), dtype=np.float32)
        scroll = np.zeros((bsz, c), dtype=np.float32)
        mask = np.zeros((bsz, c), dtype=bool)
        item_idx = np.zeros((bsz, c), dtype=np.int64)
        group_ids = np.zeros((bsz, c), dtype=np.int64)
        user_idx = np.zeros((bsz,), dtype=np.int64)
        source = np.zeros((bsz, d), dtype=np.float32)
        candidate_set = np.zeros((bsz, d), dtype=np.float32)
        hist_emb = np.zeros((bsz, self.max_history_tokens, d), dtype=np.float32)
        hist_type = np.zeros((bsz, self.max_history_tokens), dtype=np.int64)
        hist_age = np.zeros((bsz, self.max_history_tokens, 1), dtype=np.float32)
        hist_mask = np.zeros((bsz, self.max_history_tokens), dtype=bool)
        user_profile_emb = np.zeros((bsz, self.user_profile_tokens, d), dtype=np.float32)
        user_profile_type = np.zeros((bsz, self.user_profile_tokens), dtype=np.int64)
        user_profile_mask = np.zeros((bsz, self.user_profile_tokens), dtype=bool)
        user_scalar = np.zeros((bsz, self.user_scalar_dim), dtype=np.float32)
        sample_ids = np.zeros((bsz,), dtype=np.int64)

        for bi, sample in enumerate(samples):
            sid = sample.sample_id
            sample_ids[bi] = sid
            source_vec = self.source_emb[sid].astype(np.float32)
            source[bi] = source_vec
            user_idx[bi] = self.user_to_idx.get(sample.user_id, 0)
            clicked_hist = {int(aid) for aid in sample.past_clicked}
            not_clicked_hist = {int(aid) for aid in sample.past_not_clicked}
            shown_only_hist = [int(aid) for aid in sample.past_shown if int(aid) not in clicked_hist and int(aid) not in not_clicked_hist]

            if self.user_profile_mode != "none":
                for pi, aids in enumerate([sample.past_clicked, sample.past_not_clicked, shown_only_hist]):
                    vec = ep.weighted_article_mean(aids, self.article_to_idx, self.item_emb, self.max_history_tokens)
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

            history_rows: list[tuple[int, int]] = []
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

            order = self.candidate_index["orders"][self.order_name][sid]
            valid_positions = [int(pos) for pos in order if int(pos) >= 0 and self.candidate_index["mask"][sid, int(pos)]]
            item_rows = []
            for rank, pos in enumerate(valid_positions[:c]):
                idx = int(self.candidate_index["cand_idx"][sid, pos])
                aid = int(self.candidate_index["cand_aid"][sid, pos])
                emb = self.item_emb[idx].astype(np.float32)
                item_rows.append(emb)
                item[bi, rank] = emb
                if self.target_emb is not None:
                    target[bi, rank] = self.target_emb[sid, rank].astype(np.float32)
                if self.reaction_emb is not None:
                    reaction[bi, rank] = self.reaction_emb[sid, rank].astype(np.float32)
                labels[bi, rank] = float(self.candidate_index["label"][sid, pos])
                gains[bi, rank] = float(self.candidate_index["gain"][sid, pos])
                read_time[bi, rank] = float(self.candidate_index["read_time"][sid, pos])
                scroll[bi, rank] = float(self.candidate_index["scroll"][sid, pos])
                mask[bi, rank] = True
                item_idx[bi, rank] = self.item_to_idx.get(aid, 0)
                group_ids[bi, rank] = sid
                position = float(rank) / max(1.0, float(len(valid_positions) - 1))
                base = [q.cosine(emb, source_vec), 0.0, 0.0, position]
                base += q.article_dense_meta(self.articles_by_id, aid, sample, len(valid_positions))
                scalar[bi, rank] = np.asarray(base, dtype=np.float32)
            if item_rows:
                candidate_set[bi] = np.mean(item_rows, axis=0).astype(np.float32)

            if self.input_mode == "no_user":
                user_idx[bi] = 0
                source[bi] = 0.0
                hist_emb[bi] = 0.0
                hist_mask[bi] = False
                user_profile_emb[bi] = 0.0
                user_profile_mask[bi] = False
                user_scalar[bi] = 0.0
            elif self.input_mode == "position_only":
                user_idx[bi] = 0
                source[bi] = 0.0
                candidate_set[bi] = 0.0
                hist_emb[bi] = 0.0
                hist_mask[bi] = False
                user_profile_emb[bi] = 0.0
                user_profile_mask[bi] = False
                user_scalar[bi] = 0.0
                item[bi] = 0.0
                scalar[bi, :, :3] = 0.0
                scalar[bi, :, 4:] = 0.0

        return {
            "item_emb": torch.from_numpy(item),
            "target_emb": torch.from_numpy(target),
            "reaction_emb": torch.from_numpy(reaction),
            "scalar": torch.from_numpy(scalar),
            "labels": torch.from_numpy(labels),
            "gains": torch.from_numpy(gains),
            "read_time": torch.from_numpy(read_time),
            "scroll": torch.from_numpy(scroll),
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
            "group_ids": group_ids,
            "sample_ids": sample_ids,
        }


def make_loader(
    samples: list[q.Sample],
    split: str,
    builder: ReactionBatchBuilder,
    args: argparse.Namespace,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        ReactionDataset(samples, split),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.device(args.device).type == "cuda",
        collate_fn=builder,
    )


class ReactionJepaModel(nn.Module):
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
        user_profile_mode: str,
        use_item_id_emb: bool,
    ) -> None:
        super().__init__()
        self.user_profile_mode = user_profile_mode
        self.use_item_id_emb = use_item_id_emb
        self.vec_proj = nn.Linear(emb_dim, d_model)
        self.scalar_proj = nn.Linear(scalar_dim, d_model)
        self.hist_age_proj = nn.Linear(1, d_model)
        self.user_scalar_proj = nn.Linear(ReactionBatchBuilder.user_scalar_dim, d_model)
        self.user_emb = nn.Embedding(num_users + 1, d_model, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(8, d_model)
        self.hist_type_emb = nn.Embedding(4, d_model, padding_idx=0)
        self.profile_type_emb = nn.Embedding(4, d_model, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
        )
        self.gain_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int, int]:
        bsz = batch["item_emb"].shape[0]
        user = self.user_emb(batch["user_idx"]) + self.type_emb.weight[0]
        source = self.vec_proj(batch["source"]) + self.type_emb.weight[1]
        cand = self.vec_proj(batch["candidate_set"]) + self.type_emb.weight[2]
        tokens = [user.unsqueeze(1), source.unsqueeze(1), cand.unsqueeze(1)]
        masks = [torch.zeros((bsz, 3), dtype=torch.bool, device=batch["item_emb"].device)]
        if self.user_profile_mode != "none":
            user_scalar = self.user_scalar_proj(batch["user_scalar"]) + self.type_emb.weight[3]
            profile = (
                self.vec_proj(batch["user_profile_emb"])
                + self.profile_type_emb(batch["user_profile_type"])
                + self.type_emb.weight[4]
            )
            tokens.extend([user_scalar.unsqueeze(1), profile])
            masks.extend(
                [
                    torch.zeros((bsz, 1), dtype=torch.bool, device=batch["item_emb"].device),
                    ~batch["user_profile_mask"],
                ]
            )
        hist = (
            self.vec_proj(batch["hist_emb"])
            + self.hist_type_emb(batch["hist_type"])
            + self.hist_age_proj(batch["hist_age"])
            + self.type_emb.weight[5]
        )
        item = self.vec_proj(batch["item_emb"]) + self.scalar_proj(batch["scalar"]) + self.type_emb.weight[6]
        if self.use_item_id_emb:
            item = item + self.item_id_emb(batch["item_idx"])
        tokens.append(hist)
        masks.append(~batch["hist_mask"])
        item_start = sum(tok.shape[1] for tok in tokens)
        tokens.append(item)
        masks.append(~batch["mask"])
        x = torch.cat(tokens, dim=1)
        pad = torch.cat(masks, dim=1)
        out = self.encoder(x, src_key_padding_mask=pad)
        return out, item_start, item.shape[1]

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, item_start, num_items = self.encode(batch)
        item_hidden = out[:, item_start : item_start + num_items]
        delta = self.delta_head(item_hidden)
        pred = torch_unit(batch["item_emb"] + delta)
        gain_score = self.gain_head(item_hidden).squeeze(-1)
        return pred, delta, gain_score


class ReactionAwareRanker(nn.Module):
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
        user_profile_mode: str,
        use_item_id_emb: bool,
        use_reaction: bool,
    ) -> None:
        super().__init__()
        self.user_profile_mode = user_profile_mode
        self.use_item_id_emb = use_item_id_emb
        self.use_reaction = use_reaction
        self.vec_proj = nn.Linear(emb_dim, d_model)
        self.reaction_proj = nn.Linear(emb_dim, d_model)
        self.scalar_proj = nn.Linear(scalar_dim, d_model)
        self.hist_age_proj = nn.Linear(1, d_model)
        self.user_scalar_proj = nn.Linear(ReactionBatchBuilder.user_scalar_dim, d_model)
        self.user_emb = nn.Embedding(num_users + 1, d_model, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.hist_type_emb = nn.Embedding(4, d_model, padding_idx=0)
        self.profile_type_emb = nn.Embedding(4, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(8, d_model)
        hist_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        item_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(hist_layer, num_layers=max(1, layers))
        self.item_encoder = nn.TransformerEncoder(item_layer, num_layers=max(1, layers))
        self.attn_gate = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        feat_dim = d_model * 9 + scalar_dim + 4
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

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
        cand = self.vec_proj(batch["candidate_set"]) + self.type_emb.weight[2]
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
        gate = self.attn_gate(torch.cat([item, hist_ctx], dim=-1))
        hist_out = gate * hist_ctx + (1.0 - gate) * hist_mean
        if self.user_profile_mode != "none":
            profile = (
                self.vec_proj(batch["user_profile_emb"])
                + self.profile_type_emb(batch["user_profile_type"])
                + self.type_emb.weight[5]
            )
            profile_mask = batch["user_profile_mask"]
            profile_out = self.attend(item, profile, profile, profile_mask)
        else:
            profile_out = torch.zeros_like(item)
        if self.use_reaction:
            react = self.reaction_proj(batch["reaction_emb"]) + self.type_emb.weight[6]
            react_dot = torch.einsum("bcd,bcd->bc", torch_unit(batch["item_emb"]), torch_unit(batch["reaction_emb"]))
            src_react_dot = torch.einsum("bd,bcd->bc", torch_unit(batch["source"]), torch_unit(batch["reaction_emb"]))
            delta_norm = torch.linalg.norm(batch["reaction_emb"] - batch["item_emb"], dim=-1)
            react_item_delta = torch.linalg.norm(batch["reaction_emb"] - batch["item_emb"], dim=-1)
        else:
            react = torch.zeros_like(item)
            react_dot = torch.zeros_like(batch["labels"])
            src_react_dot = torch.zeros_like(batch["labels"])
            delta_norm = torch.zeros_like(batch["labels"])
            react_item_delta = torch.zeros_like(batch["labels"])
        user_ex = user.unsqueeze(1).expand_as(item)
        source_ex = source.unsqueeze(1).expand_as(item)
        cand_ex = cand.unsqueeze(1).expand_as(item)
        extra = torch.stack([react_dot, src_react_dot, delta_norm, react_item_delta], dim=-1)
        feats = torch.cat(
            [
                item,
                hist_out,
                item * hist_out,
                torch.abs(item - hist_out),
                source_ex,
                item * source_ex,
                cand_ex,
                user_ex,
                profile_out + react,
                batch["scalar"],
                extra,
            ],
            dim=-1,
        )
        return self.head(feats).squeeze(-1)


def reaction_loss(
    pred: torch.Tensor,
    delta: torch.Tensor,
    gain_score: torch.Tensor,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    if not mask.any():
        zero = pred.sum() * 0.0
        return zero, {"target_nce": 0.0, "delta_nce": 0.0, "delta_l1": 0.0, "variance": 0.0, "gain_aux": 0.0}
    p = torch_unit(pred[mask])
    t = torch_unit(batch["target_emb"][mask])
    logits = p @ t.T / args.temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    target_nce = F.cross_entropy(logits, labels)
    target_delta = batch["target_emb"] - batch["item_emb"]
    d_pred = torch_unit(delta[mask])
    d_true = torch_unit(target_delta[mask])
    delta_nce = F.cross_entropy((d_pred @ d_true.T) / args.delta_temperature, labels)
    delta_l1 = F.smooth_l1_loss(delta[mask], target_delta[mask])
    std = torch.sqrt(p.var(dim=0) + 1e-4)
    variance = torch.mean(F.relu(args.variance_gamma - std))
    gain_aux = q.listwise_loss(gain_score, batch["gains"], mask)
    loss = (
        args.target_nce_weight * target_nce
        + args.delta_nce_weight * delta_nce
        + args.delta_l1_weight * delta_l1
        + args.variance_weight * variance
        + args.predictor_gain_aux_weight * gain_aux
    )
    return loss, {
        "target_nce": float(target_nce.detach().cpu()),
        "delta_nce": float(delta_nce.detach().cpu()),
        "delta_l1": float(delta_l1.detach().cpu()),
        "variance": float(variance.detach().cpu()),
        "gain_aux": float(gain_aux.detach().cpu()),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_rank_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, gains, scores, groups, sample_ids = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            out = model(moved).detach().cpu().numpy()
            m = batch["mask"].numpy()
            labels.append(batch["labels"].numpy()[m])
            gains.append(batch["gains"].numpy()[m])
            scores.append(out[m])
            groups.append(batch["group_ids"][m])
            sid = np.repeat(batch["sample_ids"], m.shape[1]).reshape(m.shape)
            sample_ids.append(sid[m])
    return np.concatenate(labels), np.concatenate(gains), np.concatenate(scores), np.concatenate(groups), np.concatenate(sample_ids)


def ranking_metrics(labels: np.ndarray, gains: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    out = q.ranking_metrics(labels, scores, groups)
    out.update({f"engagement_{k}": v for k, v in ep.engagement_ranking_metrics(gains, scores, groups).items()})
    out["group_unit"] = "anchor_user_time_query"
    out["num_queries"] = out.get("num_impressions", 0)
    return out


def train_ranker(
    name: str,
    samples: list[q.Sample],
    make_builder,
    args: argparse.Namespace,
    device: torch.device,
    num_users: int,
    num_items: int,
    emb_dim: int,
    use_reaction: bool,
    epochs: int,
) -> tuple[nn.Module, dict[str, Any]]:
    model = ReactionAwareRanker(
        emb_dim=emb_dim,
        scalar_dim=ReactionBatchBuilder.base_scalar_dim,
        num_users=num_users,
        num_items=num_items,
        d_model=args.ranker_d_model,
        heads=args.ranker_heads,
        layers=args.ranker_layers,
        ff_dim=args.ranker_ff_dim,
        dropout=args.dropout,
        user_profile_mode=args.user_profile_mode,
        use_item_id_emb=args.use_item_id_emb,
        use_reaction=use_reaction,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.ranker_lr, weight_decay=args.weight_decay)
    train_loader = make_loader(samples, "ranker_train", make_builder("ranker_train"), args, shuffle=True)
    val_loader = make_loader(samples, "ranker_val", make_builder("ranker_val"), args)
    best_state = None
    best_val = -float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            moved = move_batch(batch, device)
            score = model(moved)
            loss = q.listwise_loss(score, moved["gains"], moved["mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        y, g, s, groups, _ = predict_rank_scores(model, val_loader, device)
        val = ranking_metrics(y, g, s, groups)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"), "ranker_val": val})
        score = float(val.get(args.primary_rank_metric, val.get("ndcg@10", -float("inf"))))
        print(json.dumps({"ranker": name, "epoch": epoch, "loss": history[-1]["loss"], "val": val}, ensure_ascii=False), flush=True)
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    split_eval = {}
    for split in ["ranker_train", "ranker_val", "ranker_test"]:
        loader = make_loader(samples, split, make_builder(split), args)
        y, g, s, groups, _ = predict_rank_scores(model, loader, device)
        split_eval[split] = ranking_metrics(y, g, s, groups)
    return model, {"history": history, "split_eval": split_eval, "best_val": best_val}


def set_baseline_order_from_model(
    model: nn.Module,
    samples: list[q.Sample],
    make_builder,
    candidate_index: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    order = np.full_like(candidate_index["orders"]["shuffled"], -1)
    for split in ["ranker_train", "ranker_val", "ranker_test"]:
        loader = make_loader(samples, split, make_builder(split), args)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                moved = move_batch(batch, device)
                scores = model(moved).detach().cpu().numpy()
                masks = batch["mask"].numpy()
                for bi, sid in enumerate(batch["sample_ids"]):
                    sid = int(sid)
                    base_order = candidate_index["orders"]["shuffled"][sid]
                    valid_j = np.where(masks[bi])[0]
                    if len(valid_j) == 0:
                        continue
                    ranked_j = valid_j[np.argsort(-scores[bi, valid_j])]
                    order[sid, : len(ranked_j)] = base_order[ranked_j]
    candidate_index["orders"]["baseline"] = order


def train_predictor(
    samples: list[q.Sample],
    make_builder,
    args: argparse.Namespace,
    device: torch.device,
    num_users: int,
    num_items: int,
    emb_dim: int,
) -> tuple[ReactionJepaModel, dict[str, Any]]:
    model = ReactionJepaModel(
        emb_dim=emb_dim,
        scalar_dim=ReactionBatchBuilder.base_scalar_dim,
        num_users=num_users,
        num_items=num_items,
        d_model=args.predictor_d_model,
        heads=args.predictor_heads,
        layers=args.predictor_layers,
        ff_dim=args.predictor_ff_dim,
        dropout=args.dropout,
        user_profile_mode=args.user_profile_mode,
        use_item_id_emb=args.use_item_id_emb,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.predictor_lr, weight_decay=args.weight_decay)
    train_loader = make_loader(samples, "ranker_train", make_builder("ranker_train", order_name="oracle", target_name="oracle"), args, shuffle=True)
    val_loader = make_loader(samples, "ranker_val", make_builder("ranker_val", order_name="baseline", target_name="baseline"), args)
    history = []
    best_state = None
    best_val = float("inf")
    for epoch in range(1, args.predictor_epochs + 1):
        model.train()
        losses = []
        parts = Counter()
        for batch in train_loader:
            moved = move_batch(batch, device)
            pred, delta, gain_score = model(moved)
            loss, detail = reaction_loss(pred, delta, gain_score, moved, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in detail.items():
                parts[key] += float(value)
        denom = max(1, len(losses))
        val_metrics = predictor_eval(model, val_loader, device)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "loss_parts": {k: float(v / denom) for k, v in parts.items()},
            "ranker_val": val_metrics,
        }
        history.append(row)
        print(json.dumps({"predictor_epoch": row}, ensure_ascii=False), flush=True)
        val_loss = float(row["loss"]) if math.isfinite(float(row["loss"])) else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    split_eval = {}
    for split in ["ranker_train", "ranker_val", "ranker_test"]:
        order = "oracle" if split == "ranker_train" else "baseline"
        target = "oracle" if split == "ranker_train" else "baseline"
        loader = make_loader(samples, split, make_builder(split, order_name=order, target_name=target), args)
        split_eval[split] = predictor_eval(model, loader, device)
    return model, {"history": history, "split_eval": split_eval}


@torch.no_grad()
def predictor_eval(model: ReactionJepaModel, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    cos_pred, cos_item, pair_sims = [], [], []
    mrrs, r1, r3, r5 = [], [], [], []
    delta_cos = []
    for batch in loader:
        moved = move_batch(batch, device)
        pred, delta, _ = model(moved)
        pred_np = torch_unit(pred).detach().cpu().numpy()
        target_np = unit(batch["target_emb"].numpy().astype(np.float32))
        item_np = unit(batch["item_emb"].numpy().astype(np.float32))
        mask = batch["mask"].numpy()
        target_delta = batch["target_emb"].numpy().astype(np.float32) - batch["item_emb"].numpy().astype(np.float32)
        delta_np = delta.detach().cpu().numpy()
        for bi in range(mask.shape[0]):
            valid = np.where(mask[bi])[0]
            if len(valid) == 0:
                continue
            p = pred_np[bi, valid]
            t = target_np[bi, valid]
            it = item_np[bi, valid]
            scores = p @ t.T
            ranks = []
            for j in range(len(valid)):
                rank = int(np.where(np.argsort(-scores[j]) == j)[0][0]) + 1
                ranks.append(rank)
            mrrs.extend([1.0 / r for r in ranks])
            r1.extend([float(r <= 1) for r in ranks])
            r3.extend([float(r <= 3) for r in ranks])
            r5.extend([float(r <= 5) for r in ranks])
            cos_pred.extend(np.sum(p * t, axis=1).tolist())
            cos_item.extend(np.sum(it * t, axis=1).tolist())
            d1 = unit(delta_np[bi, valid])
            d2 = unit(target_delta[bi, valid])
            delta_cos.extend(np.sum(d1 * d2, axis=1).tolist())
            if len(valid) > 1:
                sim = p @ p.T
                pair_sims.append(float(sim[np.triu_indices(len(valid), k=1)].mean()))
    return {
        "reaction_mrr": float(np.mean(mrrs)) if mrrs else float("nan"),
        "reaction_recall@1": float(np.mean(r1)) if r1 else float("nan"),
        "reaction_recall@3": float(np.mean(r3)) if r3 else float("nan"),
        "reaction_recall@5": float(np.mean(r5)) if r5 else float("nan"),
        "pred_target_cosine": float(np.mean(cos_pred)) if cos_pred else float("nan"),
        "item_target_cosine": float(np.mean(cos_item)) if cos_item else float("nan"),
        "delta_target_cosine": float(np.mean(delta_cos)) if delta_cos else float("nan"),
        "pred_pairwise_cosine": float(np.mean(pair_sims)) if pair_sims else float("nan"),
    }


@torch.no_grad()
def predict_reaction_arrays(
    model: ReactionJepaModel,
    samples: list[q.Sample],
    make_builder,
    args: argparse.Namespace,
    device: torch.device,
    n_samples: int,
    emb_dim: int,
    order_name: str,
) -> np.ndarray:
    pred = np.zeros((n_samples, args.max_candidates, emb_dim), dtype=np.float16)
    for split in ["ranker_train", "ranker_val", "ranker_test"]:
        loader = make_loader(samples, split, make_builder(split, order_name=order_name, target_name="baseline"), args)
        model.eval()
        for batch in loader:
            moved = move_batch(batch, device)
            out, _, _ = model(moved)
            arr = out.detach().cpu().numpy().astype(np.float16)
            pred[batch["sample_ids"]] = arr
    return pred


class ProbeMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def flatten_for_probe(
    emb: np.ndarray,
    candidate_index: dict[str, Any],
    split_samples: list[q.Sample],
    order_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, click, gain, scroll = [], [], [], []
    for sample in split_samples:
        sid = sample.sample_id
        order = candidate_index["orders"][order_name][sid]
        for rank, pos in enumerate(order):
            if pos < 0 or not candidate_index["mask"][sid, int(pos)]:
                continue
            xs.append(emb[sid, rank].astype(np.float32))
            click.append(candidate_index["label"][sid, int(pos)])
            gain.append(candidate_index["gain"][sid, int(pos)])
            scroll.append(candidate_index["scroll"][sid, int(pos)] / 100.0)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(click, dtype=np.float32),
        np.asarray(gain, dtype=np.float32),
        np.asarray(scroll, dtype=np.float32),
    )


def train_probe(
    name: str,
    emb: np.ndarray,
    candidate_index: dict[str, Any],
    samples: list[q.Sample],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    train_samples = [s for s in samples if s.split == "ranker_train"]
    x_train, y_train, g_train, sc_train = flatten_for_probe(emb, candidate_index, train_samples, "baseline")
    if x_train.size == 0:
        return {"error": "empty train probe data"}
    model = ProbeMLP(x_train.shape[1], args.probe_hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.probe_lr, weight_decay=args.weight_decay)
    target_train = np.stack([y_train, np.log1p(g_train), sc_train], axis=1).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(target_train))
    loader = DataLoader(ds, batch_size=args.probe_batch_size, shuffle=True)
    for _ in range(args.probe_epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = F.binary_cross_entropy_with_logits(out[:, 0], yb[:, 0]) + F.smooth_l1_loss(out[:, 1:], yb[:, 1:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    out = {}
    model.eval()
    for split in ["ranker_val", "ranker_test"]:
        split_samples = [s for s in samples if s.split == split]
        x, y, g, sc = flatten_for_probe(emb, candidate_index, split_samples, "baseline")
        if x.size == 0:
            out[split] = {"error": "empty split"}
            continue
        preds = []
        for start in range(0, len(x), args.probe_batch_size):
            xb = torch.from_numpy(x[start : start + args.probe_batch_size]).to(device)
            with torch.no_grad():
                preds.append(model(xb).detach().cpu().numpy())
        p = np.vstack(preds)
        click_prob = 1.0 / (1.0 + np.exp(-p[:, 0]))
        out[split] = {
            "click_auc": safe_auc(y, click_prob),
            "click_logloss": safe_logloss(y, click_prob),
            "log_gain_rmse": rmse(np.log1p(g), p[:, 1]),
            "scroll_rmse": rmse(sc, p[:, 2]),
        }
    print(json.dumps({"probe": name, "metrics": out}, ensure_ascii=False), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Ranker-conditioned per-candidate reaction JEPA for EB-NeRD.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qwen-raw-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reaction-target-cache", default="")
    parser.add_argument("--max-samples-per-split", type=int, default=1000)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--max-future-impressions", type=int, default=8)
    parser.add_argument("--max-events-per-impression", type=int, default=20)
    parser.add_argument("--future-horizon-hours", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--source-mode", choices=["qwen_cache", "history_clicked_mean", "history_weighted"], default="history_weighted")
    parser.add_argument("--max-source-history", type=int, default=80)
    parser.add_argument("--source-negative-weight", type=float, default=0.25)
    parser.add_argument("--source-shown-weight", type=float, default=0.10)
    parser.add_argument("--max-history-tokens", type=int, default=128)
    parser.add_argument("--user-profile-mode", choices=["none", "history_summary"], default="history_summary")
    parser.add_argument("--input-mode", choices=["full", "no_user", "position_only"], default="full")
    parser.add_argument("--target-order-caches", default="oracle,baseline")
    parser.add_argument("--target-cache-format", choices=["npz", "npy_dir"], default="npz")
    parser.add_argument("--target-list-context-items", type=int, default=30)
    parser.add_argument("--reaction-body-chars", type=int, default=0)
    parser.add_argument("--qwen-model", default="models/Qwen3-Embedding-4B")
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--qwen-batch-size", type=int, default=24)
    parser.add_argument("--qwen-multi-process-devices", default="")
    parser.add_argument("--embedding-dim", type=int, default=2560)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--encode-text-chunk-size", type=int, default=4096)
    parser.add_argument("--save-float16-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-target-cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-item-id-emb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--baseline-epochs", type=int, default=2)
    parser.add_argument("--ranker-epochs", type=int, default=2)
    parser.add_argument("--ranker-lr", type=float, default=3e-4)
    parser.add_argument("--ranker-d-model", type=int, default=256)
    parser.add_argument("--ranker-heads", type=int, default=8)
    parser.add_argument("--ranker-layers", type=int, default=2)
    parser.add_argument("--ranker-ff-dim", type=int, default=768)
    parser.add_argument("--predictor-epochs", type=int, default=3)
    parser.add_argument("--predictor-lr", type=float, default=2e-4)
    parser.add_argument("--predictor-d-model", type=int, default=384)
    parser.add_argument("--predictor-heads", type=int, default=8)
    parser.add_argument("--predictor-layers", type=int, default=3)
    parser.add_argument("--predictor-ff-dim", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--delta-temperature", type=float, default=0.10)
    parser.add_argument("--target-nce-weight", type=float, default=1.0)
    parser.add_argument("--delta-nce-weight", type=float, default=1.0)
    parser.add_argument("--delta-l1-weight", type=float, default=0.1)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--variance-gamma", type=float, default=0.01)
    parser.add_argument("--predictor-gain-aux-weight", type=float, default=0.0)
    parser.add_argument("--probe-epochs", type=int, default=2)
    parser.add_argument("--probe-lr", type=float, default=5e-4)
    parser.add_argument("--probe-hidden", type=int, default=512)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--primary-rank-metric", choices=["ndcg@10", "engagement_ndcg@10"], default="engagement_ndcg@10")
    parser.add_argument("--skip-final-rankers", action="store_true")
    args = parser.parse_args()

    q.set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.reaction_target_cache:
        args.reaction_target_cache = str(out_dir / "reaction_candidate_targets.npz")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    all_samples, prepare = load_samples(Path(args.samples_pkl))
    samples, cache_idx = select_ranker_samples(all_samples, args.max_samples_per_split)
    raw = np.load(args.qwen_raw_cache)
    article_ids = raw["article_ids"].astype(np.int64)
    idx_to_article = article_ids.astype(np.int64)
    article_to_idx = {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}
    item_emb = raw["article_emb"].astype(np.float32)
    qwen_source_emb = raw["source_emb"].astype(np.float32)[cache_idx]
    source_emb = ep.build_source_embeddings(samples, article_to_idx, item_emb, qwen_source_emb, args)
    emb_dim = item_emb.shape[1]
    args.embedding_dim = emb_dim

    data_dir = Path(args.data_dir)
    articles_df = pd.read_parquet(data_dir / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    articles_by_id = q.article_lookup(articles_df)
    rich_articles = q.rich_article_lookup(articles_df, include_body=args.reaction_body_chars > 0)
    full_index = q.FullSignalIndex(data_dir)
    users = sorted({s.user_id for s in samples})
    items = sorted(int(aid) for aid in article_ids.tolist())
    user_to_idx = {uid: idx + 1 for idx, uid in enumerate(users)}
    item_to_idx = {aid: idx + 1 for idx, aid in enumerate(items)}

    print("building candidate reaction index", flush=True)
    candidate_index = build_candidate_index(samples, article_to_idx, idx_to_article, full_index, args)
    print(json.dumps({"candidate_summary": candidate_index["summary"]}, ensure_ascii=False), flush=True)

    def builder_factory(
        split: str,
        *,
        order_name: str = "shuffled",
        target_name: str | None = None,
        reaction_emb: np.ndarray | None = None,
    ) -> ReactionBatchBuilder:
        del split
        target = target_caches.get(target_name, None) if target_name else None
        return ReactionBatchBuilder(
            samples=samples,
            article_to_idx=article_to_idx,
            item_emb=item_emb,
            source_emb=source_emb,
            candidate_index=candidate_index,
            target_emb=target,
            reaction_emb=reaction_emb,
            articles_by_id=articles_by_id,
            user_to_idx=user_to_idx,
            item_to_idx=item_to_idx,
            order_name=order_name,
            max_history_tokens=args.max_history_tokens,
            user_profile_mode=args.user_profile_mode,
            input_mode=args.input_mode,
        )

    target_caches: dict[str, np.ndarray] = {}

    print("training baseline order ranker on shuffled candidate sets", flush=True)
    shuffled_factory = lambda split: builder_factory(split, order_name="shuffled")
    baseline_model, baseline_order_metrics = train_ranker(
        "S0_order_ranker_shuffled",
        samples,
        shuffled_factory,
        args,
        device,
        len(users),
        len(items),
        emb_dim,
        use_reaction=False,
        epochs=args.baseline_epochs,
    )
    set_baseline_order_from_model(baseline_model, samples, shuffled_factory, candidate_index, args, device)
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_caches = build_or_load_target_cache(samples, candidate_index, rich_articles, args)

    print("training ranker-conditioned reaction JEPA", flush=True)
    predictor_model, predictor_metrics = train_predictor(
        samples,
        builder_factory,
        args,
        device,
        len(users),
        len(items),
        emb_dim,
    )
    pred_baseline = predict_reaction_arrays(predictor_model, samples, builder_factory, args, device, len(samples), emb_dim, "baseline")
    np.save(out_dir / "predicted_reaction_baseline_order.fp16.npy", pred_baseline)
    item_only_reaction = np.zeros_like(pred_baseline)
    for sample in samples:
        sid = sample.sample_id
        order = candidate_index["orders"]["baseline"][sid]
        for rank, pos in enumerate(order):
            if pos < 0 or not candidate_index["mask"][sid, int(pos)]:
                continue
            item_only_reaction[sid, rank] = item_emb[int(candidate_index["cand_idx"][sid, int(pos)])].astype(np.float16)

    probes = {
        "item_only": train_probe("item_only", item_only_reaction, candidate_index, samples, args, device),
        "oracle_reaction": train_probe("oracle_reaction", target_caches["baseline"], candidate_index, samples, args, device),
        "predicted_reaction": train_probe("predicted_reaction", pred_baseline, candidate_index, samples, args, device),
    }

    ranker_results = {"S0_order_ranker_shuffled": baseline_order_metrics}
    if not args.skip_final_rankers:
        def baseline_order_factory(reaction: np.ndarray | None, use_input_mode: str | None = None):
            def make(split: str) -> ReactionBatchBuilder:
                old_mode = args.input_mode
                if use_input_mode is not None:
                    args.input_mode = use_input_mode
                try:
                    return builder_factory(split, order_name="baseline", reaction_emb=reaction)
                finally:
                    args.input_mode = old_mode
            return make

        _, ranker_results["R0_baseline_order"] = train_ranker(
            "R0_baseline_order",
            samples,
            baseline_order_factory(None),
            args,
            device,
            len(users),
            len(items),
            emb_dim,
            use_reaction=False,
            epochs=args.ranker_epochs,
        )
        _, ranker_results["R0_user_profile_only_control"] = train_ranker(
            "R0_user_profile_only_control",
            samples,
            baseline_order_factory(None, use_input_mode="full"),
            args,
            device,
            len(users),
            len(items),
            emb_dim,
            use_reaction=False,
            epochs=args.ranker_epochs,
        )
        _, ranker_results["B_item_only_reaction"] = train_ranker(
            "B_item_only_reaction",
            samples,
            baseline_order_factory(item_only_reaction),
            args,
            device,
            len(users),
            len(items),
            emb_dim,
            use_reaction=True,
            epochs=args.ranker_epochs,
        )
        _, ranker_results["O_oracle_reaction"] = train_ranker(
            "O_oracle_reaction",
            samples,
            baseline_order_factory(target_caches["baseline"]),
            args,
            device,
            len(users),
            len(items),
            emb_dim,
            use_reaction=True,
            epochs=args.ranker_epochs,
        )
        _, ranker_results["P_predicted_reaction"] = train_ranker(
            "P_predicted_reaction",
            samples,
            baseline_order_factory(pred_baseline),
            args,
            device,
            len(users),
            len(items),
            emb_dim,
            use_reaction=True,
            epochs=args.ranker_epochs,
        )

    results = {
        "prepare": prepare,
        "setting": {
            "task": "ranker-conditioned per-candidate reaction JEPA",
            "unit": "anchor user-time query",
            "train_predictor_order": "oracle future engagement order",
            "eval_predictor_order": "baseline ranker order",
            "target": "frozen Qwen reaction-event text per candidate, no pooling",
            "max_samples_per_split": args.max_samples_per_split,
            "max_candidates": args.max_candidates,
            "future_horizon_hours": args.future_horizon_hours,
            "user_profile_mode": args.user_profile_mode,
            "input_mode": args.input_mode,
            "source_mode": args.source_mode,
            "primary_rank_metric": args.primary_rank_metric,
        },
        "split_counts": dict(Counter(s.split for s in samples)),
        "candidate_summary": candidate_index["summary"],
        "baseline_order_ranker": baseline_order_metrics,
        "predictor": predictor_metrics,
        "probes": probes,
        "rankers": ranker_results,
        "artifacts": {
            "reaction_target_cache": str(args.reaction_target_cache),
            "predicted_reaction_baseline_order": str(out_dir / "predicted_reaction_baseline_order.fp16.npy"),
        },
    }
    write_json(out_dir / "results.json", results)
    compact = {
        "candidate_summary": candidate_index["summary"],
        "predictor_test": predictor_metrics.get("split_eval", {}).get("ranker_test", {}),
        "probe_test": {name: payload.get("ranker_test", {}) for name, payload in probes.items()},
        "ranker_test": {
            name: payload.get("split_eval", {}).get("ranker_test", {})
            for name, payload in ranker_results.items()
        },
    }
    write_json(out_dir / "summary.json", compact)
    print(json.dumps({"summary": compact}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
