#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def select_ranker_samples(samples: list[q.Sample], max_per_split: int) -> list[q.Sample]:
    wanted = {"ranker_train", "ranker_val", "ranker_test"}
    counts: dict[str, int] = Counter()
    out: list[q.Sample] = []
    for sample in samples:
        if sample.split not in wanted:
            continue
        if max_per_split > 0 and counts[sample.split] >= max_per_split:
            continue
        counts[sample.split] += 1
        out.append(sample)
    for idx, sample in enumerate(out):
        sample.sample_id = idx
    return out


def unit_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x.astype(np.float32), axis=1, keepdims=True)
    return x.astype(np.float32) / np.clip(norm, 1e-8, None)


def build_or_load_raw_qwen_cache(
    samples: list[q.Sample],
    articles_df: pd.DataFrame,
    rich_articles: dict[int, dict[str, Any]],
    full_index: q.FullSignalIndex,
    args: argparse.Namespace,
) -> tuple[dict[int, int], np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, str]]:
    cache_path = Path(args.qwen_raw_cache)
    example_path = cache_path.with_suffix(".examples.json")
    meta_path = cache_path.with_suffix(".json")
    if cache_path.exists() and not args.rebuild_qwen_cache:
        packed = np.load(cache_path)
        article_ids = packed["article_ids"].astype(np.int64)
        article_emb = packed["article_emb"].astype(np.float32)
        source_emb = packed["source_emb"].astype(np.float32)
        target_emb = packed["target_emb"].astype(np.float32)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        examples = json.loads(example_path.read_text()) if example_path.exists() else {}
        return {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}, article_emb, source_emb, target_emb, meta, examples

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.qwen_model, device=args.qwen_device, local_files_only=True)
    model.max_seq_length = args.max_seq_length
    pool = None
    if args.qwen_multi_process_devices:
        devices = [device.strip() for device in args.qwen_multi_process_devices.split(",") if device.strip()]
        print(json.dumps({"qwen_multi_process_devices": devices}), flush=True)
        pool = model.start_multi_process_pool(target_devices=devices)

    def source_text(sample: q.Sample) -> str:
        return q.build_full_signal_source_text(sample, rich_articles, args, full_index)

    def target_text(sample: q.Sample) -> str:
        return q.build_full_signal_target_text(sample, rich_articles, args, full_index)

    try:
        articles_df = articles_df.drop_duplicates("article_id").reset_index(drop=True)
        article_ids = articles_df["article_id"].astype("int64").to_numpy()
        article_texts = [q.article_text_from_row(row, include_body=args.include_body) for _, row in articles_df.iterrows()]
        article_emb = q.encode_texts(model, article_texts, args.qwen_batch_size, True, "raw_articles_no_svd", pool=pool)
        article_emb = unit_rows(article_emb)

        source_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        examples = {
            "source_text": source_text(samples[0]) if samples else "",
            "target_text": target_text(samples[0]) if samples else "",
        }
        for start in range(0, len(samples), args.encode_text_chunk_size):
            end = min(len(samples), start + args.encode_text_chunk_size)
            chunk = samples[start:end]
            source_raw = q.encode_texts(
                model,
                [source_text(sample) for sample in chunk],
                args.qwen_batch_size,
                True,
                f"raw_source_uih_{start}_{end}",
                pool=pool,
            )
            target_raw = q.encode_texts(
                model,
                [target_text(sample) for sample in chunk],
                args.qwen_batch_size,
                True,
                f"raw_target_uih_{start}_{end}",
                pool=pool,
            )
            source_parts.append(unit_rows(source_raw).astype(np.float32))
            target_parts.append(unit_rows(target_raw).astype(np.float32))
        source_emb = np.vstack(source_parts).astype(np.float32)
        target_emb = np.vstack(target_parts).astype(np.float32)
    finally:
        if pool is not None:
            SentenceTransformer.stop_multi_process_pool(pool)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if args.save_float16 else np.float32
    np.savez(
        cache_path,
        article_ids=article_ids,
        article_emb=article_emb.astype(dtype),
        source_emb=source_emb.astype(dtype),
        target_emb=target_emb.astype(dtype),
    )
    meta = {
        "model": args.qwen_model,
        "no_svd": True,
        "raw_dim": int(article_emb.shape[1]),
        "num_articles": int(len(article_ids)),
        "num_samples": int(len(samples)),
        "max_seq_length": int(args.max_seq_length),
        "include_body": bool(args.include_body),
        "compact_body_chars": int(args.compact_body_chars),
        "full_signal_uih": True,
        "source_text_rule": "full signal past UIH plus current candidate set, no current feedback",
        "target_text_rule": "current response with read/scroll plus multi-horizon future UIH",
        "ranker_only_splits": True,
        "max_samples_per_ranker_split": int(args.max_samples_per_split),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    example_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    return {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}, article_emb, source_emb, target_emb, meta, examples


def main() -> None:
    parser = argparse.ArgumentParser(description="No-SVD raw Qwen oracle UIH ranker ablation for EB-NeRD.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qwen-raw-cache", required=True)
    parser.add_argument("--reference-results", default="")
    parser.add_argument("--qwen-model", default="models/Qwen3-Embedding-4B")
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--qwen-batch-size", type=int, default=16)
    parser.add_argument("--qwen-multi-process-devices", default="")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--encode-text-chunk-size", type=int, default=4096)
    parser.add_argument("--include-body", action="store_true")
    parser.add_argument("--compact-body-chars", type=int, default=120)
    parser.add_argument("--full-signal-max-history-events", type=int, default=40)
    parser.add_argument("--full-signal-max-past-impressions", type=int, default=8)
    parser.add_argument("--full-signal-max-future-impressions", type=int, default=8)
    parser.add_argument("--full-signal-horizon-hours", type=float, default=24.0)
    parser.add_argument("--full-signal-max-events-per-impression", type=int, default=20)
    parser.add_argument("--max-current-text", type=int, default=40)
    parser.add_argument("--save-float16", action="store_true")
    parser.add_argument("--rebuild-qwen-cache", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--init-post-dot-scale", type=float, default=5.0)
    args = parser.parse_args()

    q.set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_samples, prepare = load_samples(Path(args.samples_pkl))
    samples = select_ranker_samples(all_samples, args.max_samples_per_split)
    data_dir = Path(args.data_dir)
    articles_df = pd.read_parquet(data_dir / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    rich_articles = q.rich_article_lookup(articles_df, include_body=args.include_body)
    articles_by_id = q.article_lookup(articles_df)
    full_index = q.FullSignalIndex(data_dir)
    article_to_idx, item_emb, source_emb, target_emb, qwen_meta, text_examples = build_or_load_raw_qwen_cache(
        samples, articles_df, rich_articles, full_index, args
    )
    users = sorted({s.user_id for s in samples})
    items = sorted({aid for s in samples for aid in s.current_inview})
    user_to_idx = {uid: idx + 1 for idx, uid in enumerate(users)}
    item_to_idx = {aid: idx + 1 for idx, aid in enumerate(items)}
    zeros = np.zeros_like(target_emb, dtype=np.float32)

    results: dict[str, Any] = {
        "prepare": prepare,
        "qwen_raw": qwen_meta,
        "reference_results": json.loads(Path(args.reference_results).read_text()) if args.reference_results else None,
        "setting": {
            "task": "ranker-only no-SVD oracle sanity check",
            "source_text": qwen_meta.get("source_text_rule"),
            "target_text": qwen_meta.get("target_text_rule"),
            "ranker": "same set transformer, raw 2560-dim frozen Qwen vectors, no SVD",
        },
        "text_example": text_examples,
        "split_counts": dict(Counter(s.split for s in samples)),
        "candidate_dot": {
            "oracle_raw_qwen_target_uih": {
                split: q.candidate_dot_metrics(samples, split, article_to_idx, item_emb, target_emb)
                for split in ["ranker_val", "ranker_test"]
            }
        },
        "models": {},
    }
    q.write_json(out_dir / "results.partial.json", results)

    def add_ranker_result(
        name: str,
        post_vectors: np.ndarray,
        *,
        use_source_token: bool,
        use_post_token: bool,
        use_source_feature: bool,
        use_post_feature: bool,
        use_delta_feature: bool,
    ) -> None:
        print(f"training {name}", flush=True)
        results["models"][name] = q.train_ranker(
            name,
            samples,
            article_to_idx,
            item_emb,
            source_emb,
            post_vectors,
            articles_by_id,
            user_to_idx,
            item_to_idx,
            use_source_token,
            use_post_token,
            use_source_feature,
            use_post_feature,
            use_delta_feature,
            args,
        )
        print(json.dumps({name: results["models"][name]["ranker_test"]}, indent=2), flush=True)
        q.write_json(out_dir / "results.partial.json", results)

    add_ranker_result(
        "S0_source_only_raw_qwen_no_svd",
        zeros,
        use_source_token=True,
        use_post_token=False,
        use_source_feature=True,
        use_post_feature=False,
        use_delta_feature=False,
    )
    add_ranker_result(
        "O0_oracle_only_raw_qwen_no_svd",
        target_emb,
        use_source_token=False,
        use_post_token=True,
        use_source_feature=False,
        use_post_feature=True,
        use_delta_feature=False,
    )
    add_ranker_result(
        "O1_source_oracle_raw_qwen_no_svd",
        target_emb,
        use_source_token=True,
        use_post_token=True,
        use_source_feature=True,
        use_post_feature=True,
        use_delta_feature=False,
    )
    add_ranker_result(
        "O2_source_oracle_delta_raw_qwen_no_svd",
        target_emb,
        use_source_token=True,
        use_post_token=True,
        use_source_feature=True,
        use_post_feature=True,
        use_delta_feature=True,
    )
    q.write_json(out_dir / "results.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
