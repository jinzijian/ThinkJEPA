#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader

import ebnerd_event_set_predictor as ep
import ebnerd_qwen_uih_text_jepa as q
import ebnerd_ranker_conditioned_reaction_jepa as prev


TARGET_SCHEMA_VERSION = "logged_reaction_multisignal_v3_reaction_first"


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
    return float(log_loss(y, np.clip(prob, 1e-6, 1.0 - 1e-6), labels=[0, 1]))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if y.size == 0:
        return float("nan")
    return float(math.sqrt(float(mean_squared_error(y, pred))))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if y.size == 0:
        return float("nan")
    return float(mean_absolute_error(y, pred))


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y, dtype=np.float32)
    prob = np.asarray(prob, dtype=np.float32)
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(prob[mask].mean()))
    return float(ece)


def ranks_ordinal(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    out = np.empty_like(order, dtype=np.float32)
    out[order] = np.arange(len(x), dtype=np.float32)
    return out


def slate_structure_metrics(gain: np.ndarray, score: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    gain = np.asarray(gain, dtype=np.float32)
    score = np.asarray(score, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int64)
    if gain.size == 0:
        nan = float("nan")
        return {
            "pairwise_gain_agreement": nan,
            "within_slate_spearman": nan,
            "gain_ndcg@5": nan,
            "gain_ndcg@10": nan,
            "gain_mean@1": nan,
        }
    order = np.argsort(groups, kind="mergesort")
    sorted_groups = groups[order]
    starts = np.r_[0, np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    pair_correct = 0
    pair_total = 0
    diagnostic_group_limit = 5000
    diagnostic_groups = 0
    spearman = []
    ndcg5, ndcg10, mean_gain1 = [], [], []
    for start, end in zip(starts, ends):
        idx = order[start:end]
        if len(idx) < 2:
            continue
        g = gain[idx]
        s = score[idx]
        if diagnostic_groups < diagnostic_group_limit:
            diagnostic_groups += 1
            tri_i, tri_j = np.triu_indices(len(idx), k=1)
            diff_g = g[tri_i] - g[tri_j]
            diff_s = s[tri_i] - s[tri_j]
            keep = np.abs(diff_g) > 1e-8
            if keep.any():
                pair_total += int(keep.sum())
                pair_correct += int((np.sign(diff_g[keep]) == np.sign(diff_s[keep])).sum())
            if float(np.std(g)) > 1e-8 and float(np.std(s)) > 1e-8:
                rg = ranks_ordinal(g)
                rs = ranks_ordinal(s)
                rg = rg - float(rg.mean())
                rs = rs - float(rs.mean())
                denom = float(np.sqrt(np.sum(rg * rg) * np.sum(rs * rs)))
                if denom > 1e-8:
                    spearman.append(float(np.sum(rg * rs) / denom))
        if g.max(initial=0.0) <= 0:
            continue
        ranked = g[np.argsort(-s)]
        mean_gain1.append(float(ranked[0]))
        for k, bucket in [(5, ndcg5), (10, ndcg10)]:
            top = ranked[:k]
            discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
            dcg = float(np.sum(top * discounts))
            ideal = np.sort(g)[::-1][:k]
            idcg = float(np.sum(ideal * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))))
            bucket.append(dcg / idcg if idcg > 0 else 0.0)
    return {
        "pairwise_gain_agreement": float(pair_correct / pair_total) if pair_total else float("nan"),
        "within_slate_spearman": float(np.mean(spearman)) if spearman else float("nan"),
        "gain_ndcg@5": float(np.mean(ndcg5)) if ndcg5 else float("nan"),
        "gain_ndcg@10": float(np.mean(ndcg10)) if ndcg10 else float("nan"),
        "gain_mean@1": float(np.mean(mean_gain1)) if mean_gain1 else float("nan"),
    }


def finite_metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = float(metrics.get(key, default))
    return value if math.isfinite(value) else default


def checkpoint_score(metrics: dict[str, Any], metric_name: str) -> tuple[float, str]:
    if metric_name == "reaction_proxy":
        score = (
            finite_metric(metrics, "click_logloss")
            + 0.25 * finite_metric(metrics, "log_read_rmse")
            + 0.25 * finite_metric(metrics, "log_gain_rmse")
        )
        return score, "min"
    if metric_name == "click_logloss":
        return finite_metric(metrics, "click_logloss"), "min"
    if metric_name == "multi_signal_macro_auc":
        return finite_metric(metrics, "multi_signal_macro_auc", default=float("-inf")), "max"
    if metric_name == "gain_ndcg10":
        return finite_metric(metrics, "gain_ndcg@10", default=float("-inf")), "max"
    if metric_name == "gain_mean1":
        return finite_metric(metrics, "gain_mean@1", default=float("-inf")), "max"
    if metric_name == "hybrid_reaction_gain":
        score = (
            finite_metric(metrics, "gain_ndcg@10", default=float("-inf"))
            + 0.10 * finite_metric(metrics, "gain_mean@1")
            + 0.05 * finite_metric(metrics, "multi_signal_macro_auc")
            - 0.05 * finite_metric(metrics, "click_logloss")
        )
        return score, "max"
    raise ValueError(f"unknown checkpoint metric: {metric_name}")


def is_better_checkpoint(score: float, best_score: float, direction: str) -> bool:
    if not math.isfinite(score):
        return False
    if direction == "min":
        return score < best_score
    if direction == "max":
        return score > best_score
    raise ValueError(f"unknown checkpoint direction: {direction}")


def article_brief(aid: int, articles: dict[int, dict[str, Any]]) -> str:
    art = articles.get(int(aid), {})
    bits = [f"id={aid}", f"title={art.get('title', f'article {aid}')}"]
    if art.get("subtitle"):
        bits.append(f"subtitle={art['subtitle']}")
    if art.get("category"):
        bits.append(f"category={art['category']}")
    if art.get("topics"):
        bits.append(f"topics={art['topics']}")
    return " ; ".join(bits)


def is_observed(value: Any) -> bool:
    return not q.is_missing(value)


def yn(value: bool) -> str:
    return "yes" if value else "no"


def build_logged_candidate_index(
    samples: list[q.Sample],
    article_to_idx: dict[int, int],
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
    next_read_time = np.zeros((n, c), dtype=np.float32)
    next_scroll = np.zeros((n, c), dtype=np.float32)
    read_mask = np.zeros((n, c), dtype=bool)
    scroll_mask = np.zeros((n, c), dtype=bool)
    next_read_mask = np.zeros((n, c), dtype=bool)
    next_scroll_mask = np.zeros((n, c), dtype=bool)
    mask = np.zeros((n, c), dtype=bool)
    logged_order = np.full((n, c), -1, dtype=np.int32)
    records: list[list[prev.CandidateReaction]] = []
    stats = Counter()
    for sample in samples:
        sid = sample.sample_id
        _, _, current_row = full_index.current(sample)
        current_time = pd.Timestamp(sample.time if current_row is None else getattr(current_row, "impression_time"))
        inview = q.ids(getattr(current_row, "article_ids_inview")) if current_row is not None else list(sample.current_inview)
        clicked = set(q.ids(getattr(current_row, "article_ids_clicked")) if current_row is not None else sample.current_clicked)
        raw_read = getattr(current_row, "read_time", 0.0) if current_row is not None else 0.0
        raw_scroll = getattr(current_row, "scroll_percentage", 0.0) if current_row is not None else 0.0
        raw_next_read = getattr(current_row, "next_read_time", 0.0) if current_row is not None else 0.0
        raw_next_scroll = getattr(current_row, "next_scroll_percentage", 0.0) if current_row is not None else 0.0
        read_obs = is_observed(raw_read)
        scroll_obs = is_observed(raw_scroll)
        next_read_obs = is_observed(raw_next_read)
        next_scroll_obs = is_observed(raw_next_scroll)
        read = q.clean_float(raw_read, 0.0)
        scroll_val = q.clean_float(raw_scroll, 0.0)
        next_read = q.clean_float(raw_next_read, 0.0)
        next_scroll_val = q.clean_float(raw_next_scroll, 0.0)
        session_id = q.clean_id(getattr(current_row, "session_id", None)) if current_row is not None else None
        device_type = q.clean_id(getattr(current_row, "device_type", None)) if current_row is not None else None
        rows: list[prev.CandidateReaction] = []
        for aid in inview[:c]:
            idx = article_to_idx.get(int(aid))
            if idx is None:
                continue
            is_click = int(aid) in clicked
            item_read = read if is_click and read_obs else 0.0
            item_scroll = scroll_val if is_click and scroll_obs else 0.0
            item_next_read = next_read if is_click and next_read_obs else 0.0
            item_next_scroll = next_scroll_val if is_click and next_scroll_obs else 0.0
            item_gain = ep.event_weight(item_read, item_scroll, 0.0, positive=True) if is_click else 0.0
            rows.append(
                prev.CandidateReaction(
                    article_idx=int(idx),
                    article_id=int(aid),
                    gain=float(item_gain),
                    label=int(is_click),
                    event_kind="logged_clicked" if is_click else "logged_not_clicked_exposure",
                    clicked=bool(is_click),
                    read_time=float(item_read),
                    scroll=float(item_scroll),
                    next_read_time=float(item_next_read),
                    next_scroll=float(item_next_scroll),
                    dt_hours=0.0,
                    event_time=current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    session_id=session_id or 0,
                    device_type=device_type or 0,
                    candidate_count=len(inview),
                    clicked_count=len(clicked),
                )
            )
        records.append(rows)
        stats["samples"] += 1
        stats["candidates"] += len(rows)
        stats["positives"] += sum(r.label for r in rows)
        stats["empty"] += int(len(rows) == 0)
        stats["truncated"] += int(len(inview) > c)
        for j, rec in enumerate(rows[:c]):
            cand_idx[sid, j] = rec.article_idx
            cand_aid[sid, j] = rec.article_id
            gain[sid, j] = rec.gain
            label[sid, j] = rec.label
            read_time[sid, j] = rec.read_time
            scroll[sid, j] = rec.scroll
            next_read_time[sid, j] = rec.next_read_time
            next_scroll[sid, j] = rec.next_scroll
            read_mask[sid, j] = bool((not rec.clicked) or read_obs)
            scroll_mask[sid, j] = bool(scroll_obs)
            next_read_mask[sid, j] = bool((not rec.clicked) or next_read_obs)
            next_scroll_mask[sid, j] = bool(next_scroll_obs)
            mask[sid, j] = True
        valid = np.arange(len(rows[:c]), dtype=np.int32)
        logged_order[sid, : len(valid)] = valid
    summary = {
        "task_unit": "logged impression/slate",
        "samples": int(stats["samples"]),
        "candidates": int(stats["candidates"]),
        "positives": int(stats["positives"]),
        "empty": int(stats["empty"]),
        "truncated": int(stats["truncated"]),
        "avg_candidates_per_sample": float(stats["candidates"] / max(1, stats["samples"])),
        "avg_positives_per_sample": float(stats["positives"] / max(1, stats["samples"])),
        "click_rate": float(stats["positives"] / max(1, stats["candidates"])),
        "max_candidates": int(c),
    }
    return {
        "records": records,
        "cand_idx": cand_idx,
        "cand_aid": cand_aid,
        "gain": gain,
        "label": label,
        "read_time": read_time,
        "scroll": scroll,
        "next_read_time": next_read_time,
        "next_scroll": next_scroll,
        "read_mask": read_mask,
        "scroll_mask": scroll_mask,
        "next_read_mask": next_read_mask,
        "next_scroll_mask": next_scroll_mask,
        "dt_hours": np.zeros_like(gain, dtype=np.float32),
        "mask": mask,
        "orders": {"logged": logged_order},
        "summary": summary,
    }


def ordered_records(index: dict[str, Any], sample_id: int) -> list[prev.CandidateReaction]:
    records = index["records"][sample_id]
    order = index["orders"]["logged"][sample_id]
    out = []
    for pos in order:
        if pos < 0:
            continue
        out.append(records[int(pos)])
    return out


def make_logged_reaction_target_text(
    sample: q.Sample,
    rec: prev.CandidateReaction,
    ordered: list[prev.CandidateReaction],
    rank: int,
    rich_articles: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    read_observed: bool,
    scroll_observed: bool,
    next_read_observed: bool,
    next_scroll_observed: bool,
) -> str:
    start = max(0, rank - args.target_neighbor_context)
    end = min(len(ordered), rank + args.target_neighbor_context + 1)
    context = []
    for pos, item in enumerate(ordered[start:end], start=start + 1):
        prefix = "target" if pos == rank + 1 else ("before" if pos <= rank else "after")
        context.append(f"{prefix}_position_{pos}: {article_brief(item.article_id, rich_articles)}")
    read_positive = read_observed and rec.read_time > 1e-8
    scroll_positive = scroll_observed and rec.scroll > 1e-8
    next_read_positive = next_read_observed and rec.next_read_time > 1e-8
    next_scroll_positive = next_scroll_observed and rec.next_scroll > 1e-8
    return "\n".join(
        [
            "Task: encode one logged-slate user reaction event for user-conditioned multi-signal reaction prediction.",
            "Important: click, read, scroll, next_read, and next_scroll are not mutually exclusive labels.",
            f"User id: {sample.user_id}",
            f"Impression id: {sample.impression_id}",
            f"Display time: {rec.event_time}",
            f"Slate length: {len(ordered)}",
            f"Target position: {rank + 1}",
            f"Target item: {article_brief(rec.article_id, rich_articles)}",
            (
                "Multi-signal reaction flags for target item: "
                f"clicked={yn(rec.clicked)} ; "
                f"read_observed={yn(read_observed)} ; "
                f"read_positive={yn(read_positive)} ; "
                f"scroll_observed={yn(scroll_observed)} ; "
                f"scroll_positive={yn(scroll_positive)} ; "
                f"next_read_observed={yn(next_read_observed)} ; "
                f"next_read_positive={yn(next_read_positive)} ; "
                f"next_scroll_observed={yn(next_scroll_observed)} ; "
                f"next_scroll_positive={yn(next_scroll_positive)}"
            ),
            (
                "Multi-signal reaction values for target item: "
                f"read_s={rec.read_time:.1f} ; "
                f"scroll_pct={rec.scroll:.1f} ; "
                f"next_read_s={rec.next_read_time:.1f} ; "
                f"next_scroll_pct={rec.next_scroll:.1f} ; "
                f"gain={rec.gain:.4f}"
            ),
            (
                "Impression context: "
                f"session_id={rec.session_id} ; "
                f"device_type={rec.device_type} ; "
                f"clicked_count={rec.clicked_count}"
            ),
            "Ordered slate local context:",
            *context,
        ]
    )


def make_compact_reaction_target_text(
    rec: prev.CandidateReaction,
    rich_articles: dict[int, dict[str, Any]],
    read_observed: bool,
    scroll_observed: bool,
    next_read_observed: bool,
    next_scroll_observed: bool,
) -> str:
    read_positive = read_observed and rec.read_time > 1e-8
    scroll_positive = scroll_observed and rec.scroll > 1e-8
    next_read_positive = next_read_observed and rec.next_read_time > 1e-8
    next_scroll_positive = next_scroll_observed and rec.next_scroll > 1e-8
    return "\n".join(
        [
            "Task: encode the target article plus the concrete user reaction outcome.",
            f"Target item: {article_brief(rec.article_id, rich_articles)}",
            (
                "Reaction flags: "
                f"clicked={yn(rec.clicked)} ; "
                f"read_observed={yn(read_observed)} ; "
                f"read_positive={yn(read_positive)} ; "
                f"scroll_observed={yn(scroll_observed)} ; "
                f"scroll_positive={yn(scroll_positive)} ; "
                f"next_read_observed={yn(next_read_observed)} ; "
                f"next_read_positive={yn(next_read_positive)} ; "
                f"next_scroll_observed={yn(next_scroll_observed)} ; "
                f"next_scroll_positive={yn(next_scroll_positive)}"
            ),
            (
                "Reaction values: "
                f"read_s={rec.read_time:.1f} ; "
                f"scroll_pct={rec.scroll:.1f} ; "
                f"next_read_s={rec.next_read_time:.1f} ; "
                f"next_scroll_pct={rec.next_scroll:.1f} ; "
                f"gain={rec.gain:.4f}"
            ),
        ]
    )


def build_or_load_target_cache(
    samples: list[q.Sample],
    index: dict[str, Any],
    rich_articles: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> np.ndarray:
    cache_dir = Path(args.reaction_target_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / "target_logged.npy"
    success_path = cache_dir / "_SUCCESS"
    meta_path = cache_dir / "meta.json"
    examples_path = cache_dir / "examples.json"
    expected_meta = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "num_samples": len(samples),
        "max_candidates": args.max_candidates,
        "embedding_dim": args.embedding_dim,
        "qwen_model": args.qwen_model,
        "target_neighbor_context": args.target_neighbor_context,
        "reaction_body_chars": args.reaction_body_chars,
        "max_seq_length": args.max_seq_length,
        "target_text_mode": args.target_text_mode,
    }
    if not args.rebuild_target_cache and success_path.exists() and target_path.exists():
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        stale = {k: {"expected": v, "found": meta.get(k)} for k, v in expected_meta.items() if meta.get(k) != v}
        if not stale:
            arr = np.load(target_path, mmap_mode="r")
            print(json.dumps({"reaction_target_cache_dir": str(cache_dir), "loaded": "target_logged.npy", "schema_version": TARGET_SCHEMA_VERSION}, ensure_ascii=False), flush=True)
            return arr
        print(json.dumps({"reaction_target_cache_stale": str(cache_dir), "rebuild_reason": stale}, ensure_ascii=False), flush=True)
    from sentence_transformers import SentenceTransformer

    dtype = np.float16 if args.save_float16_targets else np.float32
    targets = np.lib.format.open_memmap(target_path, mode="w+", dtype=dtype, shape=(len(samples), args.max_candidates, args.embedding_dim))
    model = SentenceTransformer(args.qwen_model, device=args.qwen_device, local_files_only=True)
    model.max_seq_length = args.max_seq_length
    pool = None
    if args.qwen_multi_process_devices:
        devices = [device.strip() for device in args.qwen_multi_process_devices.split(",") if device.strip()]
        print(json.dumps({"qwen_multi_process_devices": devices}, ensure_ascii=False), flush=True)
        pool = model.start_multi_process_pool(target_devices=devices)
    pending_texts: list[str] = []
    pending_locs: list[tuple[int, int]] = []
    examples: list[str] = []

    def flush(desc: str) -> None:
        nonlocal pending_texts, pending_locs
        if not pending_texts:
            return
        encoded = q.encode_texts(model, pending_texts, args.qwen_batch_size, True, desc, pool=pool).astype(dtype, copy=False)
        for vec, (sid, rank) in zip(encoded, pending_locs):
            targets[sid, rank] = vec
        pending_texts = []
        pending_locs = []

    print(
        json.dumps(
            {"building_logged_reaction_target_cache": str(cache_dir), "target_text_mode": args.target_text_mode},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.target_text_mode == "compact_reaction":
        text_to_id: dict[str, int] = {}
        unique_texts: list[str] = []
        locs_by_id: list[list[tuple[int, int]]] = []
        total_locs = 0
        for sample in samples:
            ordered = ordered_records(index, sample.sample_id)
            for rank, rec in enumerate(ordered[: args.max_candidates]):
                text = make_compact_reaction_target_text(
                    rec,
                    rich_articles,
                    read_observed=bool(index["read_mask"][sample.sample_id, rank]),
                    scroll_observed=bool(index["scroll_mask"][sample.sample_id, rank]),
                    next_read_observed=bool(index["next_read_mask"][sample.sample_id, rank]),
                    next_scroll_observed=bool(index["next_scroll_mask"][sample.sample_id, rank]),
                )
                text_id = text_to_id.get(text)
                if text_id is None:
                    text_id = len(unique_texts)
                    text_to_id[text] = text_id
                    unique_texts.append(text)
                    locs_by_id.append([])
                    if len(examples) < 3:
                        examples.append(text)
                locs_by_id[text_id].append((sample.sample_id, rank))
                total_locs += 1
        print(
            json.dumps(
                {
                    "compact_reaction_unique_texts": len(unique_texts),
                    "target_locs": total_locs,
                    "dedup_ratio": float(total_locs / max(1, len(unique_texts))),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            for start in range(0, len(unique_texts), args.encode_text_chunk_size):
                end = min(len(unique_texts), start + args.encode_text_chunk_size)
                encoded = q.encode_texts(
                    model,
                    unique_texts[start:end],
                    args.qwen_batch_size,
                    True,
                    f"compact_reaction_targets_{start}_{end}",
                    pool=pool,
                ).astype(dtype, copy=False)
                for vec, locs in zip(encoded, locs_by_id[start:end]):
                    for sid, rank in locs:
                        targets[sid, rank] = vec
        finally:
            if pool is not None:
                from sentence_transformers import SentenceTransformer

                SentenceTransformer.stop_multi_process_pool(pool)
        if hasattr(targets, "flush"):
            targets.flush()
        meta = {
            **expected_meta,
            "target": "deduplicated frozen Qwen compact article-plus-reaction text embedding",
            "order": "logged display order from article_ids_inview",
            "reaction_semantics": "multi-signal per displayed item; click/read/scroll/next-read/next-scroll can be simultaneously true; missing flags are explicit in text",
            "format": "npy_dir",
            "index_unit": "target_logged[sample_id, displayed_rank]",
            "unique_texts": len(unique_texts),
            "target_locs": total_locs,
        }
        write_json(meta_path, meta)
        write_json(examples_path, {"logged_reaction_target_examples": examples})
        success_path.write_text("ok\n", encoding="utf-8")
        print(json.dumps({"wrote": str(cache_dir), "meta": meta}, ensure_ascii=False), flush=True)
        return np.load(target_path, mmap_mode="r")

    try:
        for sample in samples:
            ordered = ordered_records(index, sample.sample_id)
            for rank, rec in enumerate(ordered[: args.max_candidates]):
                text = make_logged_reaction_target_text(
                    sample,
                    rec,
                    ordered,
                    rank,
                    rich_articles,
                    args,
                    read_observed=bool(index["read_mask"][sample.sample_id, rank]),
                    scroll_observed=bool(index["scroll_mask"][sample.sample_id, rank]),
                    next_read_observed=bool(index["next_read_mask"][sample.sample_id, rank]),
                    next_scroll_observed=bool(index["next_scroll_mask"][sample.sample_id, rank]),
                )
                pending_texts.append(text)
                pending_locs.append((sample.sample_id, rank))
                if len(examples) < 3:
                    examples.append(text)
            if len(pending_texts) >= args.encode_text_chunk_size:
                flush(f"logged_reaction_targets_until_{sample.sample_id}")
        flush("logged_reaction_targets_final")
    finally:
        if pool is not None:
            from sentence_transformers import SentenceTransformer

            SentenceTransformer.stop_multi_process_pool(pool)
    if hasattr(targets, "flush"):
        targets.flush()
    meta = {
        **expected_meta,
        "target": "per-candidate frozen Qwen logged reaction-event text embedding",
        "order": "logged display order from article_ids_inview",
        "reaction_semantics": "multi-signal per displayed item; click/read/scroll/next-read/next-scroll can be simultaneously true; missing flags are explicit in text",
        "format": "npy_dir",
        "index_unit": "target_logged[sample_id, displayed_rank]",
    }
    write_json(meta_path, meta)
    write_json(examples_path, {"logged_reaction_target_examples": examples})
    success_path.write_text("ok\n", encoding="utf-8")
    print(json.dumps({"wrote": str(cache_dir), "meta": meta}, ensure_ascii=False), flush=True)
    return np.load(target_path, mmap_mode="r")


class LoggedBatchBuilder(prev.ReactionBatchBuilder):
    def __call__(self, samples: list[q.Sample]) -> dict[str, torch.Tensor | np.ndarray]:
        batch = super().__call__(samples)
        bsz = len(samples)
        c = self.candidate_index["cand_idx"].shape[1]
        next_read = np.zeros((bsz, c), dtype=np.float32)
        next_scroll = np.zeros((bsz, c), dtype=np.float32)
        read_mask = np.zeros((bsz, c), dtype=bool)
        scroll_mask = np.zeros((bsz, c), dtype=bool)
        next_read_mask = np.zeros((bsz, c), dtype=bool)
        next_scroll_mask = np.zeros((bsz, c), dtype=bool)
        for bi, sample in enumerate(samples):
            sid = sample.sample_id
            order = self.candidate_index["orders"][self.order_name][sid]
            for rank, pos in enumerate(order[:c]):
                if pos < 0 or not self.candidate_index["mask"][sid, int(pos)]:
                    continue
                next_read[bi, rank] = float(self.candidate_index["next_read_time"][sid, int(pos)])
                next_scroll[bi, rank] = float(self.candidate_index["next_scroll"][sid, int(pos)])
                read_mask[bi, rank] = bool(self.candidate_index["read_mask"][sid, int(pos)])
                scroll_mask[bi, rank] = bool(self.candidate_index["scroll_mask"][sid, int(pos)])
                next_read_mask[bi, rank] = bool(self.candidate_index["next_read_mask"][sid, int(pos)])
                next_scroll_mask[bi, rank] = bool(self.candidate_index["next_scroll_mask"][sid, int(pos)])
        batch["next_read_time"] = torch.from_numpy(next_read)
        batch["next_scroll"] = torch.from_numpy(next_scroll)
        batch["read_mask"] = torch.from_numpy(read_mask)
        batch["scroll_mask"] = torch.from_numpy(scroll_mask)
        batch["next_read_mask"] = torch.from_numpy(next_read_mask)
        batch["next_scroll_mask"] = torch.from_numpy(next_scroll_mask)
        return batch


SIDECAR_FEATURE_NAMES = [
    "sidecar_click_prob",
    "sidecar_log_read",
    "sidecar_scroll",
    "sidecar_log_next_read",
    "sidecar_next_scroll",
    "sidecar_log_gain",
    "sidecar_gain_score",
    "sidecar_pred_item_cosine",
    "sidecar_delta_norm",
    "sidecar_delta_item_cosine",
    "sidecar_log_gain_zscore",
    "sidecar_gain_score_zscore",
    "sidecar_click_zscore",
    "sidecar_log_read_zscore",
    "sidecar_log_next_read_zscore",
    "sidecar_click_rank_pct",
    "sidecar_log_gain_rank_pct",
    "sidecar_gain_score_rank_pct",
    "sidecar_click_top1_margin",
    "sidecar_log_gain_top1_margin",
    "sidecar_gain_score_top1_margin",
    "sidecar_click_top3_flag",
    "sidecar_log_gain_top3_flag",
    "sidecar_gain_score_top3_flag",
]


class SidecarBatchBuilder(LoggedBatchBuilder):
    def __init__(self, *args, sidecar_features: np.ndarray, concat_to_scalar: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sidecar_features = sidecar_features
        self.concat_to_scalar = concat_to_scalar

    def __call__(self, samples: list[q.Sample]) -> dict[str, torch.Tensor | np.ndarray]:
        batch = super().__call__(samples)
        sidecar = np.zeros((len(samples), self.candidate_index["cand_idx"].shape[1], self.sidecar_features.shape[-1]), dtype=np.float32)
        for bi, sample in enumerate(samples):
            sidecar[bi] = self.sidecar_features[sample.sample_id].astype(np.float32)
        if self.concat_to_scalar:
            batch["scalar"] = torch.cat([batch["scalar"], torch.from_numpy(sidecar)], dim=-1)
        batch["sidecar"] = torch.from_numpy(sidecar)
        return batch


class LoggedReactionModel(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        scalar_dim: int,
        num_users: int,
        num_items: int,
        args: argparse.Namespace,
        head_feature_mode: str = "hidden",
        sidecar_dim: int = 0,
    ) -> None:
        super().__init__()
        if head_feature_mode not in {"hidden", "pred", "delta", "pred_delta"}:
            raise ValueError(f"unknown head_feature_mode={head_feature_mode}")
        self.head_feature_mode = head_feature_mode
        if args.sidecar_fusion_mode not in {"scalar_concat", "gated", "both"}:
            raise ValueError(f"unknown sidecar_fusion_mode={args.sidecar_fusion_mode}")
        self.sidecar_fusion_mode = args.sidecar_fusion_mode if sidecar_dim > 0 else "scalar_concat"
        self.core = prev.ReactionJepaModel(
            emb_dim=emb_dim,
            scalar_dim=scalar_dim,
            num_users=num_users,
            num_items=num_items,
            d_model=args.predictor_d_model,
            heads=args.predictor_heads,
            layers=args.predictor_layers,
            ff_dim=args.predictor_ff_dim,
            dropout=args.dropout,
            user_profile_mode=args.user_profile_mode,
            use_item_id_emb=args.use_item_id_emb,
        )
        latent_in = 0
        if head_feature_mode in {"pred", "pred_delta"}:
            latent_in += emb_dim
        if head_feature_mode in {"delta", "pred_delta"}:
            latent_in += emb_dim
        self.latent_feature_proj = nn.Sequential(
            nn.LayerNorm(latent_in),
            nn.Linear(latent_in, args.predictor_d_model),
            nn.GELU(),
            nn.Dropout(args.dropout),
        ) if latent_in else None
        if sidecar_dim > 0 and self.sidecar_fusion_mode in {"gated", "both"}:
            self.sidecar_adapter = nn.Sequential(
                nn.LayerNorm(sidecar_dim),
                nn.Linear(sidecar_dim, args.predictor_d_model),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.predictor_d_model, args.predictor_d_model),
            )
            self.sidecar_gate = nn.Sequential(
                nn.LayerNorm(sidecar_dim),
                nn.Linear(sidecar_dim, args.predictor_d_model),
            )
        else:
            self.sidecar_adapter = None
            self.sidecar_gate = None
        head_in = args.predictor_d_model * (2 if latent_in else 1)
        self.reaction_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, args.predictor_ff_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.predictor_ff_dim, 6),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out, item_start, num_items = self.core.encode(batch)
        item_hidden = out[:, item_start : item_start + num_items]
        if self.sidecar_adapter is not None and "sidecar" in batch:
            sidecar = batch["sidecar"].float()
            item_hidden = item_hidden + torch.sigmoid(self.sidecar_gate(sidecar)) * self.sidecar_adapter(sidecar)
        delta = self.core.delta_head(item_hidden)
        pred = prev.torch_unit(batch["item_emb"] + delta)
        gain_score = self.core.gain_head(item_hidden).squeeze(-1)
        head_input = item_hidden
        if self.latent_feature_proj is not None:
            pieces = []
            if self.head_feature_mode in {"pred", "pred_delta"}:
                pieces.append(pred)
            if self.head_feature_mode in {"delta", "pred_delta"}:
                pieces.append(delta)
            latent_features = self.latent_feature_proj(torch.cat(pieces, dim=-1))
            head_input = torch.cat([item_hidden, latent_features], dim=-1)
        reaction_out = self.reaction_head(head_input)
        return pred, delta, gain_score, reaction_out


class OracleLatentProbe(nn.Module):
    def __init__(self, emb_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def head_targets(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "click": batch["labels"],
        "log_read": torch.log1p(batch["read_time"].clamp_min(0.0)),
        "scroll": (batch["scroll"].clamp_min(0.0) / 100.0).clamp(0.0, 1.0),
        "log_next_read": torch.log1p(batch["next_read_time"].clamp_min(0.0)),
        "next_scroll": (batch["next_scroll"].clamp_min(0.0) / 100.0).clamp(0.0, 1.0),
        "log_gain": torch.log1p(batch["gains"].clamp_min(0.0)),
    }


def logged_head_loss(out: torch.Tensor, gain_score: torch.Tensor, batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    targets = head_targets(batch)
    if not mask.any():
        zero = out.sum() * 0.0
        return zero, {"click": 0.0, "read": 0.0, "scroll": 0.0, "next_read": 0.0, "next_scroll": 0.0, "gain": 0.0, "slate": 0.0}
    click_loss = F.binary_cross_entropy_with_logits(out[..., 0][mask], targets["click"][mask])
    read_mask = mask & batch["read_mask"]
    scroll_mask = mask & batch["scroll_mask"]
    next_read_mask = mask & batch["next_read_mask"]
    next_scroll_mask = mask & batch["next_scroll_mask"]
    read_loss = F.smooth_l1_loss(out[..., 1][read_mask], targets["log_read"][read_mask]) if read_mask.any() else out.sum() * 0.0
    scroll_loss = F.smooth_l1_loss(out[..., 2][scroll_mask], targets["scroll"][scroll_mask]) if scroll_mask.any() else out.sum() * 0.0
    next_read_loss = (
        F.smooth_l1_loss(out[..., 3][next_read_mask], targets["log_next_read"][next_read_mask])
        if next_read_mask.any()
        else out.sum() * 0.0
    )
    next_scroll_loss = (
        F.smooth_l1_loss(out[..., 4][next_scroll_mask], targets["next_scroll"][next_scroll_mask])
        if next_scroll_mask.any()
        else out.sum() * 0.0
    )
    gain_loss = F.smooth_l1_loss(out[..., 5][mask], targets["log_gain"][mask])
    slate_loss = q.listwise_loss(gain_score, batch["gains"], mask)
    total = (
        args.click_loss_weight * click_loss
        + args.read_loss_weight * read_loss
        + args.scroll_loss_weight * scroll_loss
        + args.next_read_loss_weight * next_read_loss
        + args.next_scroll_loss_weight * next_scroll_loss
        + args.gain_loss_weight * gain_loss
        + args.slate_aux_weight * slate_loss
    )
    return total, {
        "click": float(click_loss.detach().cpu()),
        "read": float(read_loss.detach().cpu()),
        "scroll": float(scroll_loss.detach().cpu()),
        "next_read": float(next_read_loss.detach().cpu()),
        "next_scroll": float(next_scroll_loss.detach().cpu()),
        "gain": float(gain_loss.detach().cpu()),
        "slate": float(slate_loss.detach().cpu()),
    }


def make_loader(samples: list[q.Sample], split: str, builder: LoggedBatchBuilder, args: argparse.Namespace, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        prev.ReactionDataset(samples, split),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.device(args.device).type == "cuda",
        collate_fn=builder,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def collect_model_predictions(
    model: LoggedReactionModel,
    loader: DataLoader,
    device: torch.device,
    *,
    include_latent: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    model.eval()
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    latent_mrr, r1, r3, r5, cos_pred, cos_item, delta_cos, pair_sims = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            pred, delta, gain_score, out = model(moved)
            mask = batch["mask"].numpy()
            heads = out.detach().cpu().numpy()
            rows["click"].append(batch["labels"].numpy()[mask])
            rows["click_prob"].append(1.0 / (1.0 + np.exp(-heads[..., 0][mask])))
            rows["log_read"].append(np.log1p(batch["read_time"].numpy()[mask]))
            rows["pred_log_read"].append(heads[..., 1][mask])
            rows["scroll"].append(np.clip(batch["scroll"].numpy()[mask] / 100.0, 0.0, 1.0))
            rows["pred_scroll"].append(heads[..., 2][mask])
            rows["log_next_read"].append(np.log1p(batch["next_read_time"].numpy()[mask]))
            rows["pred_log_next_read"].append(heads[..., 3][mask])
            rows["next_scroll"].append(np.clip(batch["next_scroll"].numpy()[mask] / 100.0, 0.0, 1.0))
            rows["pred_next_scroll"].append(heads[..., 4][mask])
            rows["log_gain"].append(np.log1p(batch["gains"].numpy()[mask]))
            rows["pred_log_gain"].append(heads[..., 5][mask])
            rows["gain"].append(batch["gains"].numpy()[mask])
            rows["pred_gain"].append(np.expm1(np.clip(heads[..., 5][mask], -10.0, 10.0)))
            rows["gain_score"].append(gain_score.detach().cpu().numpy()[mask])
            rows["read_mask"].append(batch["read_mask"].numpy()[mask])
            rows["scroll_mask"].append(batch["scroll_mask"].numpy()[mask])
            rows["next_read_mask"].append(batch["next_read_mask"].numpy()[mask])
            rows["next_scroll_mask"].append(batch["next_scroll_mask"].numpy()[mask])
            rows["groups"].append(batch["group_ids"][mask])
            if include_latent:
                pred_np = prev.unit(pred.detach().cpu().numpy().astype(np.float32))
                target_np = prev.unit(batch["target_emb"].numpy().astype(np.float32))
                item_np = prev.unit(batch["item_emb"].numpy().astype(np.float32))
                target_delta = batch["target_emb"].numpy().astype(np.float32) - batch["item_emb"].numpy().astype(np.float32)
                delta_np = delta.detach().cpu().numpy().astype(np.float32)
                for bi in range(mask.shape[0]):
                    valid = np.where(mask[bi])[0]
                    if len(valid) == 0:
                        continue
                    sim = pred_np[bi, valid] @ target_np[bi, valid].T
                    ranks = np.argsort(-sim, axis=1)
                    for row_idx in range(len(valid)):
                        rank = int(np.where(ranks[row_idx] == row_idx)[0][0]) + 1
                        latent_mrr.append(1.0 / rank)
                        r1.append(float(rank <= 1))
                        r3.append(float(rank <= 3))
                        r5.append(float(rank <= 5))
                    cos_pred.extend(np.sum(pred_np[bi, valid] * target_np[bi, valid], axis=1).tolist())
                    cos_item.extend(np.sum(item_np[bi, valid] * target_np[bi, valid], axis=1).tolist())
                    d_pred = prev.unit(delta_np[bi, valid])
                    d_true = prev.unit(target_delta[bi, valid])
                    delta_cos.extend(np.sum(d_pred * d_true, axis=1).tolist())
                    if len(valid) > 1:
                        sims = pred_np[bi, valid] @ pred_np[bi, valid].T
                        tri = sims[np.triu_indices(len(valid), k=1)]
                        pair_sims.extend(tri.tolist())
    out = {k: np.concatenate(v) if v else np.asarray([], dtype=np.float32) for k, v in rows.items()}
    latent = {}
    if include_latent:
        latent = {
            "reaction_latent_mrr": float(np.mean(latent_mrr)) if latent_mrr else float("nan"),
            "reaction_recall@1": float(np.mean(r1)) if r1 else float("nan"),
            "reaction_recall@3": float(np.mean(r3)) if r3 else float("nan"),
            "reaction_recall@5": float(np.mean(r5)) if r5 else float("nan"),
            "pred_target_cosine": float(np.mean(cos_pred)) if cos_pred else float("nan"),
            "item_target_cosine": float(np.mean(cos_item)) if cos_item else float("nan"),
            "delta_target_cosine": float(np.mean(delta_cos)) if delta_cos else float("nan"),
            "pred_pairwise_cosine": float(np.mean(pair_sims)) if pair_sims else float("nan"),
        }
    return out, latent


def metrics_from_arrays(arr: dict[str, np.ndarray]) -> dict[str, float]:
    y = arr["click"].astype(np.float32)
    p = arr["click_prob"].astype(np.float32)
    read_mask = arr["read_mask"].astype(bool)
    scroll_mask = arr["scroll_mask"].astype(bool)
    next_read_mask = arr["next_read_mask"].astype(bool)
    next_scroll_mask = arr["next_scroll_mask"].astype(bool)
    clicked = y > 0.5
    out = {
        "click_auc": safe_auc(y, p),
        "click_logloss": safe_logloss(y, p),
        "click_brier": float(np.mean((p - y) ** 2)) if y.size else float("nan"),
        "click_ece": expected_calibration_error(y, p),
        "log_read_mae": mae(arr["log_read"][read_mask], arr["pred_log_read"][read_mask]),
        "log_read_rmse": rmse(arr["log_read"][read_mask], arr["pred_log_read"][read_mask]),
        "clicked_log_read_mae": mae(arr["log_read"][read_mask & clicked], arr["pred_log_read"][read_mask & clicked]),
        "clicked_log_read_rmse": rmse(arr["log_read"][read_mask & clicked], arr["pred_log_read"][read_mask & clicked]),
        "scroll_mae": mae(arr["scroll"][scroll_mask], arr["pred_scroll"][scroll_mask]),
        "scroll_rmse": rmse(arr["scroll"][scroll_mask], arr["pred_scroll"][scroll_mask]),
        "clicked_scroll_mae": mae(arr["scroll"][scroll_mask & clicked], arr["pred_scroll"][scroll_mask & clicked]),
        "clicked_scroll_rmse": rmse(arr["scroll"][scroll_mask & clicked], arr["pred_scroll"][scroll_mask & clicked]),
        "log_next_read_mae": mae(arr["log_next_read"][next_read_mask], arr["pred_log_next_read"][next_read_mask]),
        "log_next_read_rmse": rmse(arr["log_next_read"][next_read_mask], arr["pred_log_next_read"][next_read_mask]),
        "next_scroll_mae": mae(arr["next_scroll"][next_scroll_mask], arr["pred_next_scroll"][next_scroll_mask]),
        "next_scroll_rmse": rmse(arr["next_scroll"][next_scroll_mask], arr["pred_next_scroll"][next_scroll_mask]),
        "log_gain_mae": mae(arr["log_gain"], arr["pred_log_gain"]),
        "log_gain_rmse": rmse(arr["log_gain"], arr["pred_log_gain"]),
    }
    read_event_auc = safe_auc((arr["log_read"][read_mask] > 1e-8).astype(np.float32), arr["pred_log_read"][read_mask])
    scroll_event_auc = safe_auc((arr["scroll"][scroll_mask] > 1e-8).astype(np.float32), arr["pred_scroll"][scroll_mask])
    next_read_event_auc = safe_auc((arr["log_next_read"][next_read_mask] > 1e-8).astype(np.float32), arr["pred_log_next_read"][next_read_mask])
    next_scroll_event_auc = safe_auc((arr["next_scroll"][next_scroll_mask] > 1e-8).astype(np.float32), arr["pred_next_scroll"][next_scroll_mask])
    event_aucs = np.asarray(
        [out["click_auc"], read_event_auc, scroll_event_auc, next_read_event_auc, next_scroll_event_auc],
        dtype=np.float32,
    )
    out.update(
        {
            "read_event_auc": read_event_auc,
            "scroll_event_auc": scroll_event_auc,
            "next_read_event_auc": next_read_event_auc,
            "next_scroll_event_auc": next_scroll_event_auc,
            "multi_signal_macro_auc": float(np.nanmean(event_aucs)) if np.isfinite(event_aucs).any() else float("nan"),
        }
    )
    out.update(slate_structure_metrics(arr["gain"], arr["pred_gain"], arr["groups"].astype(np.int64)))
    return out


def eval_model(model: LoggedReactionModel, samples: list[q.Sample], builder_factory, args: argparse.Namespace, device: torch.device, include_latent: bool) -> dict[str, Any]:
    out = {}
    for split in ["ranker_val", "ranker_test"]:
        loader = make_loader(samples, split, builder_factory(split), args)
        arr, latent = collect_model_predictions(model, loader, device, include_latent=include_latent)
        metrics = metrics_from_arrays(arr)
        metrics.update(latent)
        out[split] = metrics
    return out


def sidecar_zscore(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    count = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * mask_f).sum(dim=1, keepdim=True) / count
    var = (((values - mean) ** 2) * mask_f).sum(dim=1, keepdim=True) / count
    return (values - mean) / torch.sqrt(var + 1e-6)


def sidecar_rank_pct(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask, float("-inf"))
    order = torch.argsort(masked, dim=1, descending=True, stable=True)
    ranks = torch.zeros_like(values)
    rank_values = torch.arange(values.shape[1], device=values.device, dtype=values.dtype).unsqueeze(0).expand_as(values)
    ranks.scatter_(1, order, rank_values)
    count = mask.float().sum(dim=1, keepdim=True)
    pct = torch.where(count > 1.0, 1.0 - ranks / (count - 1.0).clamp_min(1.0), torch.ones_like(ranks))
    return torch.where(mask, pct, torch.zeros_like(pct))


def sidecar_top1_margin(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask, float("-inf"))
    k = min(2, values.shape[1])
    top = torch.topk(masked, k=k, dim=1).values
    top1 = top[:, :1]
    if k == 1:
        return torch.zeros_like(values)
    top2 = top[:, 1:2]
    top1_idx = torch.argmax(masked, dim=1, keepdim=True)
    pos = torch.arange(values.shape[1], device=values.device).unsqueeze(0).expand_as(values)
    max_other = torch.where(pos == top1_idx, top2, top1)
    margin = values - max_other
    return torch.where(mask, margin, torch.zeros_like(margin))


def sidecar_topk_flag(values: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    rank_pct = sidecar_rank_pct(values, mask)
    count = mask.float().sum(dim=1, keepdim=True)
    threshold = torch.where(count > 1.0, 1.0 - (float(k) - 1.0) / (count - 1.0).clamp_min(1.0), torch.ones_like(count))
    return torch.where(mask & (rank_pct >= threshold), torch.ones_like(values), torch.zeros_like(values))


def normalize_sidecar_features(
    features: np.ndarray,
    samples: list[q.Sample],
    candidate_index: dict[str, Any],
    chunk_size: int = 4096,
) -> dict[str, Any]:
    train_ids = np.asarray([sample.sample_id for sample in samples if sample.split == "ranker_train"], dtype=np.int64)
    if train_ids.size == 0:
        return {"applied": False, "reason": "empty_train_split"}
    dim = int(features.shape[-1])
    total = np.zeros(dim, dtype=np.float64)
    total_sq = np.zeros(dim, dtype=np.float64)
    count = 0
    for start in range(0, len(train_ids), chunk_size):
        ids = train_ids[start : start + chunk_size]
        valid = candidate_index["mask"][ids]
        if not valid.any():
            continue
        values = np.asarray(features[ids], dtype=np.float32)[valid]
        total += values.sum(axis=0, dtype=np.float64)
        total_sq += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += int(values.shape[0])
    if count == 0:
        return {"applied": False, "reason": "empty_train_valid_items"}
    mean = total / float(count)
    var = np.maximum(total_sq / float(count) - mean * mean, 1e-8)
    std = np.sqrt(var)
    all_ids = np.arange(len(samples), dtype=np.int64)
    for start in range(0, len(all_ids), chunk_size):
        ids = all_ids[start : start + chunk_size]
        valid = candidate_index["mask"][ids]
        arr = np.asarray(features[ids], dtype=np.float32)
        arr = (arr - mean.astype(np.float32)) / std.astype(np.float32)
        arr[~valid] = 0.0
        features[ids] = arr.astype(features.dtype, copy=False)
    if hasattr(features, "flush"):
        features.flush()
    return {
        "applied": True,
        "source_split": "ranker_train",
        "count": int(count),
        "mean": [float(x) for x in mean.tolist()],
        "std": [float(x) for x in std.tolist()],
    }


def generate_sidecar_features(
    name: str,
    predictor: LoggedReactionModel,
    samples: list[q.Sample],
    candidate_index: dict[str, Any],
    builder_factory,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> np.ndarray:
    cache_path = Path(args.sidecar_feature_cache) if args.sidecar_feature_cache else out_dir / f"{name}_sidecar_features.npy"
    meta_path = cache_path.with_suffix(".json")
    success_path = cache_path.with_suffix(".success")
    expected = {
        "name": name,
        "num_samples": len(samples),
        "max_candidates": args.max_candidates,
        "feature_names": SIDECAR_FEATURE_NAMES,
        "train_normalized": bool(args.sidecar_train_normalize),
    }
    if not args.rebuild_sidecar_cache and success_path.exists() and cache_path.exists():
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        stale = {k: {"expected": v, "found": meta.get(k)} for k, v in expected.items() if meta.get(k) != v}
        if not stale:
            print(json.dumps({"sidecar_feature_cache": str(cache_path), "loaded": True}, ensure_ascii=False), flush=True)
            return np.load(cache_path, mmap_mode="r")
        print(json.dumps({"sidecar_feature_cache_stale": str(cache_path), "rebuild_reason": stale}, ensure_ascii=False), flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16 if args.save_float16_sidecar else np.float32,
        shape=(len(samples), args.max_candidates, len(SIDECAR_FEATURE_NAMES)),
    )
    features[:] = 0
    predictor.eval()
    print(json.dumps({"building_sidecar_features": str(cache_path), "feature_names": SIDECAR_FEATURE_NAMES}, ensure_ascii=False), flush=True)
    with torch.no_grad():
        for split in ["ranker_train", "ranker_val", "ranker_test"]:
            loader = make_loader(samples, split, builder_factory(split), args)
            for batch in loader:
                moved = move_batch(batch, device)
                pred, delta, gain_score, heads = predictor(moved)
                mask = moved["mask"]
                item_unit = F.normalize(moved["item_emb"].float(), dim=-1)
                pred_unit = F.normalize(pred.float(), dim=-1)
                delta_unit = F.normalize(delta.float(), dim=-1)
                pred_item_cos = torch.sum(pred_unit * item_unit, dim=-1)
                delta_item_cos = torch.sum(delta_unit * item_unit, dim=-1)
                delta_norm = torch.linalg.vector_norm(delta.float(), dim=-1) / math.sqrt(float(delta.shape[-1]))
                log_gain_z = sidecar_zscore(heads[..., 5], mask)
                gain_score_z = sidecar_zscore(gain_score, mask)
                click_prob = torch.sigmoid(heads[..., 0])
                click_z = sidecar_zscore(click_prob, mask)
                log_read_z = sidecar_zscore(heads[..., 1], mask)
                log_next_read_z = sidecar_zscore(heads[..., 3], mask)
                click_rank = sidecar_rank_pct(click_prob, mask)
                log_gain_rank = sidecar_rank_pct(heads[..., 5], mask)
                gain_score_rank = sidecar_rank_pct(gain_score, mask)
                click_margin = sidecar_top1_margin(click_prob, mask)
                log_gain_margin = sidecar_top1_margin(heads[..., 5], mask)
                gain_score_margin = sidecar_top1_margin(gain_score, mask)
                click_top3 = sidecar_topk_flag(click_prob, mask, 3)
                log_gain_top3 = sidecar_topk_flag(heads[..., 5], mask, 3)
                gain_score_top3 = sidecar_topk_flag(gain_score, mask, 3)
                stacked = torch.stack(
                    [
                        click_prob,
                        heads[..., 1],
                        heads[..., 2],
                        heads[..., 3],
                        heads[..., 4],
                        heads[..., 5],
                        gain_score,
                        pred_item_cos,
                        delta_norm,
                        delta_item_cos,
                        log_gain_z,
                        gain_score_z,
                        click_z,
                        log_read_z,
                        log_next_read_z,
                        click_rank,
                        log_gain_rank,
                        gain_score_rank,
                        click_margin,
                        log_gain_margin,
                        gain_score_margin,
                        click_top3,
                        log_gain_top3,
                        gain_score_top3,
                    ],
                    dim=-1,
                )
                stacked = torch.where(mask.unsqueeze(-1), stacked, torch.zeros_like(stacked))
                arr = stacked.detach().cpu().numpy().astype(features.dtype, copy=False)
                for bi, sid in enumerate(batch["sample_ids"]):
                    features[int(sid)] = arr[bi]
    if hasattr(features, "flush"):
        features.flush()
    norm_meta = normalize_sidecar_features(features, samples, candidate_index) if args.sidecar_train_normalize else {"applied": False}
    meta = {
        **expected,
        "dtype": str(features.dtype),
        "source": "frozen JEPA predictor inference only; no oracle target features",
        "normalization": norm_meta,
    }
    write_json(meta_path, meta)
    success_path.write_text("ok\n", encoding="utf-8")
    print(json.dumps({"wrote_sidecar_features": str(cache_path), "meta": meta}, ensure_ascii=False), flush=True)
    return np.load(cache_path, mmap_mode="r")


def train_model_variant(
    name: str,
    samples: list[q.Sample],
    builder_factory,
    args: argparse.Namespace,
    device: torch.device,
    num_users: int,
    num_items: int,
    emb_dim: int,
    *,
    use_latent_loss: bool,
    latent_loss_scale: float = 1.0,
    head_feature_mode: str = "hidden",
    scalar_dim: int | None = None,
    sidecar_dim: int = 0,
) -> tuple[LoggedReactionModel, dict[str, Any]]:
    model = LoggedReactionModel(
        emb_dim,
        LoggedBatchBuilder.base_scalar_dim if scalar_dim is None else scalar_dim,
        num_users,
        num_items,
        args,
        head_feature_mode=head_feature_mode,
        sidecar_dim=sidecar_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.predictor_lr, weight_decay=args.weight_decay)
    train_loader = make_loader(samples, "ranker_train", builder_factory("ranker_train"), args, shuffle=True)
    best_state = None
    _, checkpoint_direction = checkpoint_score({}, args.checkpoint_metric)
    best_score = float("inf") if checkpoint_direction == "min" else float("-inf")
    history = []
    for epoch in range(1, args.predictor_epochs + 1):
        model.train()
        losses = []
        parts = Counter()
        for batch in train_loader:
            moved = move_batch(batch, device)
            pred, delta, gain_score, heads = model(moved)
            head_loss, head_detail = logged_head_loss(heads, gain_score, moved, args)
            if use_latent_loss:
                latent_loss, latent_detail = prev.reaction_loss(pred, delta, gain_score, moved, args)
                latent_loss = latent_loss_scale * latent_loss
            else:
                latent_loss = head_loss * 0.0
                latent_detail = {"target_nce": 0.0, "delta_nce": 0.0, "delta_l1": 0.0, "variance": 0.0, "gain_aux": 0.0}
            loss = head_loss + latent_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in {**{f"head_{k}": v for k, v in head_detail.items()}, **{f"latent_{k}": v for k, v in latent_detail.items()}}.items():
                parts[key] += float(value)
        val_eval = eval_model(model, samples, builder_factory, args, device, include_latent=use_latent_loss)["ranker_val"]
        score, checkpoint_direction = checkpoint_score(val_eval, args.checkpoint_metric)
        if is_better_checkpoint(score, best_score, checkpoint_direction):
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        denom = max(1, len(losses))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "head_feature_mode": head_feature_mode,
            "latent_loss_scale": float(latent_loss_scale if use_latent_loss else 0.0),
            "checkpoint_metric": args.checkpoint_metric,
            "checkpoint_score": float(score),
            "checkpoint_direction": checkpoint_direction,
            "loss_parts": {k: float(v / denom) for k, v in parts.items()},
            "ranker_val": val_eval,
        }
        history.append(row)
        print(json.dumps({"variant": name, "epoch": row}, ensure_ascii=False), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    split_eval = eval_model(model, samples, builder_factory, args, device, include_latent=use_latent_loss)
    return model, {"history": history, "split_eval": split_eval}


def train_oracle_probe(
    samples: list[q.Sample],
    builder_factory,
    args: argparse.Namespace,
    device: torch.device,
    emb_dim: int,
) -> dict[str, Any]:
    model = OracleLatentProbe(emb_dim, args.probe_hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.probe_lr, weight_decay=args.weight_decay)
    train_loader = make_loader(samples, "ranker_train", builder_factory("ranker_train"), args, shuffle=True)
    for epoch in range(1, args.probe_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            moved = move_batch(batch, device)
            out = model(moved["target_emb"])
            loss, _ = logged_head_loss(out, out[..., 5], moved, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(json.dumps({"oracle_latent_probe_epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan")}), flush=True)
    out: dict[str, Any] = {}
    model.eval()
    for split in ["ranker_val", "ranker_test"]:
        loader = make_loader(samples, split, builder_factory(split), args)
        rows: dict[str, list[np.ndarray]] = defaultdict(list)
        with torch.no_grad():
            for batch in loader:
                moved = move_batch(batch, device)
                heads = model(moved["target_emb"]).detach().cpu().numpy()
                mask = batch["mask"].numpy()
                rows["click"].append(batch["labels"].numpy()[mask])
                rows["click_prob"].append(1.0 / (1.0 + np.exp(-heads[..., 0][mask])))
                rows["log_read"].append(np.log1p(batch["read_time"].numpy()[mask]))
                rows["pred_log_read"].append(heads[..., 1][mask])
                rows["scroll"].append(np.clip(batch["scroll"].numpy()[mask] / 100.0, 0.0, 1.0))
                rows["pred_scroll"].append(heads[..., 2][mask])
                rows["log_next_read"].append(np.log1p(batch["next_read_time"].numpy()[mask]))
                rows["pred_log_next_read"].append(heads[..., 3][mask])
                rows["next_scroll"].append(np.clip(batch["next_scroll"].numpy()[mask] / 100.0, 0.0, 1.0))
                rows["pred_next_scroll"].append(heads[..., 4][mask])
                rows["log_gain"].append(np.log1p(batch["gains"].numpy()[mask]))
                rows["pred_log_gain"].append(heads[..., 5][mask])
                rows["gain"].append(batch["gains"].numpy()[mask])
                rows["pred_gain"].append(np.expm1(np.clip(heads[..., 5][mask], -10.0, 10.0)))
                rows["read_mask"].append(batch["read_mask"].numpy()[mask])
                rows["scroll_mask"].append(batch["scroll_mask"].numpy()[mask])
                rows["next_read_mask"].append(batch["next_read_mask"].numpy()[mask])
                rows["next_scroll_mask"].append(batch["next_scroll_mask"].numpy()[mask])
                rows["groups"].append(batch["group_ids"][mask])
        arr = {k: np.concatenate(v) if v else np.asarray([], dtype=np.float32) for k, v in rows.items()}
        out[split] = metrics_from_arrays(arr)
    print(json.dumps({"oracle_latent_probe": out}, ensure_ascii=False), flush=True)
    return out


def flatten_labels(samples: list[q.Sample], candidate_index: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    split_samples = [sample for sample in samples if sample.split == split]
    if not split_samples:
        return {
            "rank": np.asarray([], dtype=np.int32),
            "aid": np.asarray([], dtype=np.int64),
            "user": np.asarray([], dtype=np.int64),
            "click": np.asarray([], dtype=np.float32),
            "log_read": np.asarray([], dtype=np.float32),
            "scroll": np.asarray([], dtype=np.float32),
            "log_next_read": np.asarray([], dtype=np.float32),
            "next_scroll": np.asarray([], dtype=np.float32),
            "log_gain": np.asarray([], dtype=np.float32),
            "gain": np.asarray([], dtype=np.float32),
            "read_mask": np.asarray([], dtype=bool),
            "scroll_mask": np.asarray([], dtype=bool),
            "next_read_mask": np.asarray([], dtype=bool),
            "next_scroll_mask": np.asarray([], dtype=bool),
            "groups": np.asarray([], dtype=np.int64),
            "user_ctr": np.asarray([], dtype=np.float32),
        }
    sample_ids = np.asarray([sample.sample_id for sample in split_samples], dtype=np.int64)
    valid = candidate_index["mask"][sample_ids]
    c = valid.shape[1]
    # Logged-order arrays are already stored by displayed rank, so flattening can
    # stay vectorized instead of walking every displayed item in Python.
    ranks = np.broadcast_to(np.arange(c, dtype=np.int32), valid.shape)
    groups = np.broadcast_to(sample_ids[:, None], valid.shape)
    users = np.broadcast_to(np.asarray([sample.user_id for sample in split_samples], dtype=np.int64)[:, None], valid.shape)
    user_ctr = []
    for sample in split_samples:
        hist_total = max(1, len(sample.past_clicked) + len(sample.past_not_clicked))
        user_ctr.append(float(len(sample.past_clicked)) / hist_total)
    user_ctr_arr = np.broadcast_to(np.asarray(user_ctr, dtype=np.float32)[:, None], valid.shape)
    return {
        "rank": ranks[valid],
        "aid": candidate_index["cand_aid"][sample_ids][valid].astype(np.int64, copy=False),
        "user": users[valid],
        "click": candidate_index["label"][sample_ids][valid].astype(np.float32, copy=False),
        "log_read": np.log1p(candidate_index["read_time"][sample_ids][valid]).astype(np.float32, copy=False),
        "scroll": (candidate_index["scroll"][sample_ids][valid] / 100.0).astype(np.float32, copy=False),
        "log_next_read": np.log1p(candidate_index["next_read_time"][sample_ids][valid]).astype(np.float32, copy=False),
        "next_scroll": (candidate_index["next_scroll"][sample_ids][valid] / 100.0).astype(np.float32, copy=False),
        "log_gain": np.log1p(candidate_index["gain"][sample_ids][valid]).astype(np.float32, copy=False),
        "gain": candidate_index["gain"][sample_ids][valid].astype(np.float32, copy=False),
        "read_mask": candidate_index["read_mask"][sample_ids][valid],
        "scroll_mask": candidate_index["scroll_mask"][sample_ids][valid],
        "next_read_mask": candidate_index["next_read_mask"][sample_ids][valid],
        "next_scroll_mask": candidate_index["next_scroll_mask"][sample_ids][valid],
        "groups": groups[valid].astype(np.int64, copy=False),
        "user_ctr": user_ctr_arr[valid],
    }


def logged_policy_heuristic(samples: list[q.Sample], candidate_index: dict[str, Any]) -> dict[str, Any]:
    train = flatten_labels(samples, candidate_index, "ranker_train")
    pos_click, pos_gain, pos_read, pos_scroll, pos_next_read, pos_next_scroll = {}, {}, {}, {}, {}, {}
    for rank in np.unique(train["rank"]):
        m = train["rank"] == rank
        pos_click[int(rank)] = float(train["click"][m].mean())
        pos_gain[int(rank)] = float(train["log_gain"][m].mean())
        pos_read[int(rank)] = float(train["log_read"][m].mean())
        pos_scroll[int(rank)] = float(train["scroll"][m].mean())
        pos_next_read[int(rank)] = float(train["log_next_read"][m].mean())
        pos_next_scroll[int(rank)] = float(train["next_scroll"][m].mean())
    article_seen = Counter(train["aid"].tolist())
    article_click = Counter(train["aid"][train["click"] > 0.5].tolist())
    global_click = float(train["click"].mean()) if train["click"].size else 0.0

    out = {}
    for split in ["ranker_val", "ranker_test"]:
        data = flatten_labels(samples, candidate_index, split)
        click_prob = []
        pred_log_gain, pred_log_read, pred_scroll, pred_log_next_read, pred_next_scroll = [], [], [], [], []
        for rank, aid, user_ctr in zip(data["rank"], data["aid"], data["user_ctr"]):
            art_ctr = (article_click[int(aid)] + global_click * 5.0) / (article_seen[int(aid)] + 5.0)
            pos_ctr = pos_click.get(int(rank), global_click)
            click_prob.append(float(np.clip(0.55 * pos_ctr + 0.30 * art_ctr + 0.15 * user_ctr, 1e-6, 1 - 1e-6)))
            pred_log_gain.append(pos_gain.get(int(rank), float(train["log_gain"].mean())))
            pred_log_read.append(pos_read.get(int(rank), float(train["log_read"].mean())))
            pred_scroll.append(pos_scroll.get(int(rank), float(train["scroll"].mean())))
            pred_log_next_read.append(pos_next_read.get(int(rank), float(train["log_next_read"].mean())))
            pred_next_scroll.append(pos_next_scroll.get(int(rank), float(train["next_scroll"].mean())))
        arr = {
            "click": data["click"].astype(np.float32),
            "click_prob": np.asarray(click_prob, dtype=np.float32),
            "log_read": data["log_read"].astype(np.float32),
            "pred_log_read": np.asarray(pred_log_read, dtype=np.float32),
            "scroll": data["scroll"].astype(np.float32),
            "pred_scroll": np.asarray(pred_scroll, dtype=np.float32),
            "log_next_read": data["log_next_read"].astype(np.float32),
            "pred_log_next_read": np.asarray(pred_log_next_read, dtype=np.float32),
            "next_scroll": data["next_scroll"].astype(np.float32),
            "pred_next_scroll": np.asarray(pred_next_scroll, dtype=np.float32),
            "log_gain": data["log_gain"].astype(np.float32),
            "pred_log_gain": np.asarray(pred_log_gain, dtype=np.float32),
            "gain": data["gain"].astype(np.float32),
            "pred_gain": np.expm1(np.asarray(pred_log_gain, dtype=np.float32)),
            "read_mask": data["read_mask"].astype(bool),
            "scroll_mask": data["scroll_mask"].astype(bool),
            "next_read_mask": data["next_read_mask"].astype(bool),
            "next_scroll_mask": data["next_scroll_mask"].astype(bool),
            "groups": data["groups"].astype(np.int64),
        }
        out[split] = metrics_from_arrays(arr)
    print(json.dumps({"logged_policy_heuristic": out}, ensure_ascii=False), flush=True)
    return out


def parse_float_list(raw: str) -> list[float]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Logged-slate per-item reaction JEPA for EB-NeRD.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qwen-raw-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reaction-target-cache", default="")
    parser.add_argument("--max-samples-per-split", type=int, default=10000)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--source-mode", choices=["qwen_cache", "history_clicked_mean", "history_weighted"], default="history_weighted")
    parser.add_argument("--max-source-history", type=int, default=80)
    parser.add_argument("--source-negative-weight", type=float, default=0.25)
    parser.add_argument("--source-shown-weight", type=float, default=0.10)
    parser.add_argument("--max-history-tokens", type=int, default=128)
    parser.add_argument("--user-profile-mode", choices=["none", "history_summary"], default="history_summary")
    parser.add_argument("--target-neighbor-context", type=int, default=3)
    parser.add_argument("--target-text-mode", choices=["full_context", "compact_reaction"], default="full_context")
    parser.add_argument("--reaction-body-chars", type=int, default=0)
    parser.add_argument("--qwen-model", default="models/Qwen3-Embedding-4B")
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--qwen-batch-size", type=int, default=32)
    parser.add_argument("--qwen-multi-process-devices", default="")
    parser.add_argument("--embedding-dim", type=int, default=2560)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--encode-text-chunk-size", type=int, default=8192)
    parser.add_argument("--save-float16-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-target-cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-item-id-emb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
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
    parser.add_argument("--click-loss-weight", type=float, default=1.0)
    parser.add_argument("--read-loss-weight", type=float, default=0.5)
    parser.add_argument("--scroll-loss-weight", type=float, default=0.25)
    parser.add_argument("--next-read-loss-weight", type=float, default=0.25)
    parser.add_argument("--next-scroll-loss-weight", type=float, default=0.15)
    parser.add_argument("--gain-loss-weight", type=float, default=0.5)
    parser.add_argument("--slate-aux-weight", type=float, default=0.05)
    parser.add_argument(
        "--checkpoint-metric",
        choices=[
            "reaction_proxy",
            "click_logloss",
            "multi_signal_macro_auc",
            "gain_ndcg10",
            "gain_mean1",
            "hybrid_reaction_gain",
        ],
        default="reaction_proxy",
    )
    parser.add_argument("--probe-epochs", type=int, default=2)
    parser.add_argument("--probe-lr", type=float, default=5e-4)
    parser.add_argument("--probe-hidden", type=int, default=512)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--skip-heuristic", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-oracle-probe", action="store_true")
    parser.add_argument("--skip-standalone-jepa", action="store_true")
    parser.add_argument("--run-additive-ablation", action="store_true")
    parser.add_argument("--additive-latent-scales", default="0.02,0.05")
    parser.add_argument("--run-sidecar-ablation", action="store_true")
    parser.add_argument("--sidecar-latent-scale", type=float, default=0.01)
    parser.add_argument("--sidecar-predictor-head-feature-mode", choices=["hidden", "pred", "delta", "pred_delta"], default="hidden")
    parser.add_argument("--sidecar-feature-cache", default="")
    parser.add_argument("--rebuild-sidecar-cache", action="store_true")
    parser.add_argument("--save-float16-sidecar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sidecar-train-normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sidecar-only-from-cache", action="store_true")
    parser.add_argument("--sidecar-fusion-mode", choices=["scalar_concat", "gated", "both"], default="scalar_concat")
    args = parser.parse_args()

    q.set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.reaction_target_cache:
        args.reaction_target_cache = str(out_dir / "logged_reaction_targets")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    all_samples, prepare = prev.load_samples(Path(args.samples_pkl))
    samples, cache_idx = prev.select_ranker_samples(all_samples, args.max_samples_per_split)
    raw = np.load(args.qwen_raw_cache)
    article_ids = raw["article_ids"].astype(np.int64)
    article_to_idx = {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}
    item_emb = raw["article_emb"].astype(np.float32)
    qwen_source_emb = raw["source_emb"].astype(np.float32)[cache_idx]
    source_emb = ep.build_source_embeddings(samples, article_to_idx, item_emb, qwen_source_emb, args)
    emb_dim = int(item_emb.shape[1])
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

    print("building logged slate reaction index", flush=True)
    candidate_index = build_logged_candidate_index(samples, article_to_idx, full_index, args)
    print(json.dumps({"candidate_summary": candidate_index["summary"]}, ensure_ascii=False), flush=True)

    target_cache: np.ndarray | None = None

    def builder_factory(input_mode: str, target: np.ndarray | None = None):
        def make_builder(split: str) -> LoggedBatchBuilder:
            del split
            return LoggedBatchBuilder(
                samples=samples,
                article_to_idx=article_to_idx,
                item_emb=item_emb,
                source_emb=source_emb,
                candidate_index=candidate_index,
                target_emb=target,
                reaction_emb=None,
                articles_by_id=articles_by_id,
                user_to_idx=user_to_idx,
                item_to_idx=item_to_idx,
                order_name="logged",
                max_history_tokens=args.max_history_tokens,
                user_profile_mode=args.user_profile_mode,
                input_mode=input_mode,
            )

        return make_builder

    if args.skip_heuristic:
        heuristic_results: dict[str, Any] = {}
        print(json.dumps({"skipping_logged_policy_heuristic": True}, ensure_ascii=False), flush=True)
    else:
        print(json.dumps({"computing_logged_policy_heuristic": True}, ensure_ascii=False), flush=True)
        heuristic_results = logged_policy_heuristic(samples, candidate_index)

    results: dict[str, Any] = {
        "prepare": prepare,
        "setting": {
            "task": "logged-slate per-item reaction prediction",
            "unit": "one logged impression / slate",
            "input_order": "original article_ids_inview display order",
            "not_input_order": "engagement-sorted order",
            "target": "frozen Qwen reaction-event text per displayed item, no pooling",
            "max_samples_per_split": args.max_samples_per_split,
            "max_candidates": args.max_candidates,
            "checkpoint_metric": args.checkpoint_metric,
            "slate_aux_weight": args.slate_aux_weight,
            "predictor_gain_aux_weight": args.predictor_gain_aux_weight,
            "primary_claim": "JEPA learns a user-conditioned reaction model over logged recommendation slates.",
        },
        "split_counts": dict(Counter(s.split for s in samples)),
        "candidate_summary": candidate_index["summary"],
        "heuristic": heuristic_results,
        "models": {},
        "oracle_probe": {},
        "artifacts": {"reaction_target_cache": str(args.reaction_target_cache)},
    }

    if not args.skip_baselines:
        baseline_specs = [
            ("position_prior_only", "position_only", False),
            ("item_only", "no_user", False),
            ("user_history_item_supervised", "full", False),
        ]
        for name, input_mode, use_latent in baseline_specs:
            print(json.dumps({"training_baseline": name, "input_mode": input_mode}, ensure_ascii=False), flush=True)
            _, metrics = train_model_variant(
                name,
                samples,
                builder_factory(input_mode, None),
                args,
                device,
                len(users),
                len(items),
                emb_dim,
                use_latent_loss=use_latent,
                latent_loss_scale=1.0,
                head_feature_mode="hidden",
            )
            results["models"][name] = metrics
            write_json(out_dir / "partial_results.json", results)

    target_cache = build_or_load_target_cache(samples, candidate_index, rich_articles, args)
    if args.run_additive_ablation:
        additive_specs: list[tuple[str, bool, float, str]] = [
            ("user_history_item_latent_adapter_control", False, 0.0, "pred_delta"),
        ]
        for scale in parse_float_list(args.additive_latent_scales):
            tag = str(scale).replace(".", "p")
            additive_specs.extend(
                [
                    (f"user_history_item_jepa_aux_w{tag}", True, scale, "hidden"),
                    (f"user_history_item_jepa_aux_adapter_w{tag}", True, scale, "pred_delta"),
                ]
            )
        for name, use_latent, scale, head_feature_mode in additive_specs:
            print(
                json.dumps(
                    {
                        "training_additive_ablation": name,
                        "input_mode": "full",
                        "head_feature_mode": head_feature_mode,
                        "latent_loss_scale": scale if use_latent else 0.0,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            _, metrics = train_model_variant(
                name,
                samples,
                builder_factory("full", target_cache),
                args,
                device,
                len(users),
                len(items),
                emb_dim,
                use_latent_loss=use_latent,
                latent_loss_scale=scale,
                head_feature_mode=head_feature_mode,
            )
            results["models"][name] = metrics
            write_json(out_dir / "partial_results.json", results)

    if args.run_sidecar_ablation:
        scale_tag = str(args.sidecar_latent_scale).replace(".", "p")
        predictor_name = f"two_stage_jepa_predictor_w{scale_tag}"
        if args.sidecar_only_from_cache:
            if not args.sidecar_feature_cache:
                raise ValueError("--sidecar-only-from-cache requires --sidecar-feature-cache")
            sidecar_features = np.load(args.sidecar_feature_cache, mmap_mode="r")
            if int(sidecar_features.shape[-1]) != len(SIDECAR_FEATURE_NAMES):
                raise ValueError(
                    f"sidecar feature dim mismatch: cache has {sidecar_features.shape[-1]}, expected {len(SIDECAR_FEATURE_NAMES)}"
                )
            print(
                json.dumps(
                    {
                        "using_cached_sidecar_features": args.sidecar_feature_cache,
                        "shape": list(sidecar_features.shape),
                        "skip_sidecar_predictor_training": True,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            results["models"][predictor_name] = {"skipped": "loaded sidecar features from cache"}
        else:
            print(
                json.dumps(
                    {
                        "training_sidecar_predictor": predictor_name,
                        "latent_loss_scale": args.sidecar_latent_scale,
                        "head_feature_mode": args.sidecar_predictor_head_feature_mode,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            sidecar_predictor, predictor_metrics = train_model_variant(
                predictor_name,
                samples,
                builder_factory("full", target_cache),
                args,
                device,
                len(users),
                len(items),
                emb_dim,
                use_latent_loss=True,
                latent_loss_scale=args.sidecar_latent_scale,
                head_feature_mode=args.sidecar_predictor_head_feature_mode,
            )
            results["models"][predictor_name] = predictor_metrics
            write_json(out_dir / "partial_results.json", results)

            sidecar_features = generate_sidecar_features(
                predictor_name,
                sidecar_predictor,
                samples,
                candidate_index,
                builder_factory("full", target_cache),
                args,
                device,
                out_dir,
            )
        results["artifacts"]["sidecar_feature_cache"] = str(
            Path(args.sidecar_feature_cache) if args.sidecar_feature_cache else out_dir / f"{predictor_name}_sidecar_features.npy"
        )
        results["artifacts"]["sidecar_feature_names"] = SIDECAR_FEATURE_NAMES
        results["artifacts"]["sidecar_fusion_mode"] = args.sidecar_fusion_mode

        def sidecar_builder_factory(input_mode: str):
            def make_builder(split: str) -> SidecarBatchBuilder:
                del split
                return SidecarBatchBuilder(
                    samples=samples,
                    article_to_idx=article_to_idx,
                    item_emb=item_emb,
                    source_emb=source_emb,
                    candidate_index=candidate_index,
                    target_emb=None,
                    reaction_emb=None,
                    articles_by_id=articles_by_id,
                    user_to_idx=user_to_idx,
                    item_to_idx=item_to_idx,
                    order_name="logged",
                    max_history_tokens=args.max_history_tokens,
                    user_profile_mode=args.user_profile_mode,
                    input_mode=input_mode,
                    sidecar_features=sidecar_features,
                    concat_to_scalar=args.sidecar_fusion_mode in {"scalar_concat", "both"},
                )

            return make_builder

        sidecar_name = f"two_stage_jepa_sidecar_supervised_{args.sidecar_fusion_mode}_w{scale_tag}"
        sidecar_scalar_dim = LoggedBatchBuilder.base_scalar_dim + (
            len(SIDECAR_FEATURE_NAMES) if args.sidecar_fusion_mode in {"scalar_concat", "both"} else 0
        )
        print(
            json.dumps(
                {
                    "training_sidecar_supervised": sidecar_name,
                    "sidecar_dim": len(SIDECAR_FEATURE_NAMES),
                    "sidecar_fusion_mode": args.sidecar_fusion_mode,
                    "sidecar_scalar_dim": sidecar_scalar_dim,
                    "feature_names": SIDECAR_FEATURE_NAMES,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        _, sidecar_metrics = train_model_variant(
            sidecar_name,
            samples,
            sidecar_builder_factory("full"),
            args,
            device,
            len(users),
            len(items),
            emb_dim,
            use_latent_loss=False,
            latent_loss_scale=0.0,
            head_feature_mode="hidden",
            scalar_dim=sidecar_scalar_dim,
            sidecar_dim=len(SIDECAR_FEATURE_NAMES),
        )
        results["models"][sidecar_name] = sidecar_metrics
        write_json(out_dir / "partial_results.json", results)

    if not args.skip_standalone_jepa:
        print(json.dumps({"training_main_model": "logged_slate_reaction_jepa"}, ensure_ascii=False), flush=True)
        _, jepa_metrics = train_model_variant(
            "logged_slate_reaction_jepa",
            samples,
            builder_factory("full", target_cache),
            args,
            device,
            len(users),
            len(items),
            emb_dim,
            use_latent_loss=True,
            latent_loss_scale=1.0,
            head_feature_mode="hidden",
        )
        results["models"]["logged_slate_reaction_jepa"] = jepa_metrics
        write_json(out_dir / "partial_results.json", results)

    if not args.skip_oracle_probe:
        results["oracle_probe"]["true_reaction_latent_probe"] = train_oracle_probe(
            samples,
            builder_factory("full", target_cache),
            args,
            device,
            emb_dim,
        )

    compact = {
        "candidate_summary": candidate_index["summary"],
        "heuristic_test": results["heuristic"].get("ranker_test", {}),
        "model_test": {name: payload.get("split_eval", {}).get("ranker_test", {}) for name, payload in results["models"].items()},
        "oracle_probe_test": {
            name: payload.get("ranker_test", {}) for name, payload in results["oracle_probe"].items()
        },
    }
    results["summary"] = compact
    write_json(out_dir / "results.json", results)
    write_json(out_dir / "summary.json", compact)
    print(json.dumps({"summary": compact}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
