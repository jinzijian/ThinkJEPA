#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ebnerd_qwen_uih_text_jepa as q
from ebnerd_event_set_predictor import (
    collect_event_records,
    load_samples,
    reaction_event_text,
    select_ranker_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen-Qwen reaction-event latent cache for EB-NeRD future UIH.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qwen-raw-cache", required=True)
    parser.add_argument("--out-cache", required=True)
    parser.add_argument("--target-mode", choices=["current_future", "future_only"], default="future_only")
    parser.add_argument("--max-samples-per-split", type=int, default=1000)
    parser.add_argument("--max-pos-events", type=int, default=16)
    parser.add_argument("--max-neg-events", type=int, default=32)
    parser.add_argument("--max-rank-candidates", type=int, default=256)
    parser.add_argument("--max-future-impressions", type=int, default=8)
    parser.add_argument("--max-events-per-impression", type=int, default=20)
    parser.add_argument("--future-horizon-hours", type=float, default=6.0)
    parser.add_argument("--include-current-neg", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reaction-body-chars", type=int, default=0)
    parser.add_argument("--qwen-model", default="models/Qwen3-Embedding-4B")
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--qwen-batch-size", type=int, default=24)
    parser.add_argument("--qwen-multi-process-devices", default="")
    parser.add_argument("--embedding-dim", type=int, default=2560)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--encode-text-chunk-size", type=int, default=4096)
    parser.add_argument("--save-float16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    out_cache = Path(args.out_cache)
    if out_cache.exists() and not args.rebuild:
        print(json.dumps({"cache_exists": str(out_cache)}, ensure_ascii=False), flush=True)
        return

    q.set_seed(args.seed)
    all_samples, prepare = load_samples(Path(args.samples_pkl))
    samples, cache_idx = select_ranker_samples(all_samples, args.max_samples_per_split)
    raw = np.load(args.qwen_raw_cache)
    article_ids = raw["article_ids"].astype(np.int64)
    emb_dim = int(args.embedding_dim)
    article_to_idx = {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}
    data_dir = Path(args.data_dir)
    articles_df = pd.read_parquet(data_dir / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    rich_articles = q.rich_article_lookup(articles_df, include_body=args.reaction_body_chars > 0)
    full_index = q.FullSignalIndex(data_dir)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.qwen_model, device=args.qwen_device, local_files_only=True)
    model.max_seq_length = args.max_seq_length
    pool = None
    if args.qwen_multi_process_devices:
        devices = [device.strip() for device in args.qwen_multi_process_devices.split(",") if device.strip()]
        print(json.dumps({"qwen_multi_process_devices": devices}), flush=True)
        pool = model.start_multi_process_pool(target_devices=devices)

    dtype = np.float16 if args.save_float16 else np.float32
    pos_emb = np.zeros((len(samples), args.max_pos_events, emb_dim), dtype=dtype)
    neg_emb = np.zeros((len(samples), args.max_neg_events, emb_dim), dtype=dtype)
    pos_mask = np.zeros((len(samples), args.max_pos_events), dtype=bool)
    neg_mask = np.zeros((len(samples), args.max_neg_events), dtype=bool)
    stats = Counter()
    examples: dict[str, Any] = {"positive": [], "negative": []}

    pending_texts: list[str] = []
    pending_locs: list[tuple[str, int, int]] = []

    def flush(desc: str) -> None:
        nonlocal pending_texts, pending_locs
        if not pending_texts:
            return
        encoded = q.encode_texts(
            model,
            pending_texts,
            args.qwen_batch_size,
            True,
            desc,
            pool=pool,
        ).astype(dtype, copy=False)
        for vec, (kind, sid, pos) in zip(encoded, pending_locs):
            if kind == "pos":
                pos_emb[sid, pos] = vec
            else:
                neg_emb[sid, pos] = vec
        pending_texts = []
        pending_locs = []

    try:
        for sample in samples:
            pos_events, neg_events, _ = collect_event_records(sample, article_to_idx, full_index, args)
            stats["samples"] += 1
            stats["pos_events"] += len(pos_events)
            stats["neg_events"] += len(neg_events)
            stats["empty_pos"] += int(len(pos_events) == 0)
            stats["empty_neg"] += int(len(neg_events) == 0)
            for j, event in enumerate(pos_events[: args.max_pos_events]):
                text = reaction_event_text(sample, event, rich_articles, args)
                pos_mask[sample.sample_id, j] = True
                pending_texts.append(text)
                pending_locs.append(("pos", sample.sample_id, j))
                if len(examples["positive"]) < 3:
                    examples["positive"].append(text)
            for j, event in enumerate(neg_events[: args.max_neg_events]):
                text = reaction_event_text(sample, event, rich_articles, args)
                neg_mask[sample.sample_id, j] = True
                pending_texts.append(text)
                pending_locs.append(("neg", sample.sample_id, j))
                if len(examples["negative"]) < 3:
                    examples["negative"].append(text)
            if len(pending_texts) >= args.encode_text_chunk_size:
                flush(f"reaction_events_until_sample_{sample.sample_id}")
        flush("reaction_events_final")
    finally:
        if pool is not None:
            SentenceTransformer.stop_multi_process_pool(pool)

    out_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_cache,
        pos_emb=pos_emb,
        neg_emb=neg_emb,
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        cache_idx=cache_idx.astype(np.int64),
    )
    meta = {
        "model": args.qwen_model,
        "target": "future UIH reaction event set encoded directly by frozen Qwen",
        "num_samples": int(len(samples)),
        "split_counts": dict(Counter(s.split for s in samples)),
        "raw_embedding_dim": emb_dim,
        "target_mode": args.target_mode,
        "future_horizon_hours": float(args.future_horizon_hours),
        "max_future_impressions": int(args.max_future_impressions),
        "max_events_per_impression": int(args.max_events_per_impression),
        "max_pos_events": int(args.max_pos_events),
        "max_neg_events": int(args.max_neg_events),
        "reaction_body_chars": int(args.reaction_body_chars),
        "event_summary": {
            "samples": int(stats["samples"]),
            "pos_events": int(stats["pos_events"]),
            "neg_events": int(stats["neg_events"]),
            "empty_pos": int(stats["empty_pos"]),
            "empty_neg": int(stats["empty_neg"]),
            "avg_pos_per_sample": float(stats["pos_events"] / max(1, stats["samples"])),
            "avg_neg_per_sample": float(stats["neg_events"] / max(1, stats["samples"])),
        },
        "prepare": prepare,
    }
    out_cache.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    out_cache.with_suffix(".examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"wrote": str(out_cache), "meta": meta}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
