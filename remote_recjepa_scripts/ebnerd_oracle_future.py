#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from ebnerd_pipeline import article_lookup, ranking_metrics, train_variant, write_json


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


def unit(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= 1e-8:
        return x.astype(np.float32)
    return (x / norm).astype(np.float32)


def load_cached(path: Path) -> tuple[list[Sample], dict]:
    with path.open("rb") as f:
        packed = pickle.load(f)
    return packed["samples"], packed.get("prepare_summary", {})


def load_article_embeddings(path: Path) -> tuple[dict[int, int], np.ndarray]:
    packed = np.load(path)
    article_ids = packed["article_ids"].astype(np.int64)
    dense = packed["embeddings"].astype(np.float32)
    dense = normalize(dense).astype(np.float32)
    return {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}, dense


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / den)


def set_prior(samples: list[Sample], vectors: np.ndarray, name: str) -> dict:
    nonzero = 0
    for sample, vec in zip(samples, vectors):
        prior = unit(vec.astype(np.float32))
        sample.z_prior = prior
        sample.delta_z = (prior - sample.z_current).astype(np.float32)
        nonzero += int(np.linalg.norm(prior) > 1e-8)
    return {"oracle": name, "num_samples": len(samples), "num_nonzero_prior": nonzero}


def candidate_dot_metrics(
    samples: list[Sample],
    split: str,
    article_to_idx: dict[int, int],
    emb: np.ndarray,
    vector_getter,
) -> dict:
    labels = []
    scores = []
    group_ids = []
    nonzero_groups = 0
    for sample in samples:
        if sample.split != split:
            continue
        z = unit(vector_getter(sample).astype(np.float32))
        if np.linalg.norm(z) > 1e-8:
            nonzero_groups += 1
        clicked = set(sample.current_clicked)
        for aid in sample.current_inview:
            idx = article_to_idx.get(aid)
            score = cosine(emb[idx], z) if idx is not None else 0.0
            labels.append(int(aid in clicked))
            scores.append(score)
            group_ids.append(sample.impression_id)
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float32)
    g = np.asarray(group_ids)
    out = ranking_metrics(y, s, g)
    out["nonzero_prior_impressions"] = int(nonzero_groups)
    return out


def load_structured_anchor(samples: list[Sample], args: argparse.Namespace) -> np.ndarray:
    from ebnerd_sequence_jepa import build_sequence_arrays, load_article_embeddings as load_padded

    article_to_idx, emb_table = load_padded(Path(args.article_embeddings))
    arrays = build_sequence_arrays(samples, Path(args.data_dir), article_to_idx, emb_table, args)
    return arrays["anchor"].astype(np.float32), arrays["stats"]


def run_oracle_ranker(
    samples: list[Sample],
    articles_by_id: dict[int, dict],
    article_to_idx: dict[int, int],
    emb: np.ndarray,
    args: argparse.Namespace,
    oracle_name: str,
    vectors: np.ndarray,
) -> dict:
    out = {"prior": set_prior(samples, vectors, oracle_name), "candidate_dot": {}, "models": {}}
    for split in ["ranker_val", "ranker_test"]:
        out["candidate_dot"][split] = {
            "dot_item_oracle": candidate_dot_metrics(samples, split, article_to_idx, emb, lambda s: s.z_prior),
            "dot_item_past_uih": candidate_dot_metrics(samples, split, article_to_idx, emb, lambda s: s.z_current),
            "dot_item_current_set": candidate_dot_metrics(samples, split, article_to_idx, emb, lambda s: s.z_current_set),
        }
    for variant in ["T1_future_prior", "T2_future_item_cross"]:
        print(f"training {variant} with {oracle_name}", flush=True)
        out["models"][variant] = train_variant(variant, samples, articles_by_id, article_to_idx, emb, args.seed, args)
        print(json.dumps({variant: out["models"][variant]["ranker_test"]}, indent=2), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle future UIH upper-bound tests for EB-NeRD RecJEPA.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--article-embeddings", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--oracle", choices=["pooled", "structured", "both"], default="both")
    parser.add_argument("--seed", type=int, default=7)
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
    samples, prepare = load_cached(Path(args.samples_pkl))
    article_to_idx, emb = load_article_embeddings(Path(args.article_embeddings))
    articles = pd.read_parquet(Path(args.data_dir) / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    articles_by_id = article_lookup(articles)
    baseline = json.loads(Path(args.baseline_results).read_text())
    results = {
        "prepare": prepare,
        "baseline_models_from": args.baseline_results,
        "baseline_test": {
            name: baseline["models"][name]["ranker_test"]
            for name in ["B0_meta", "B1_text_history", "T1_future_prior", "T2_future_item_cross"]
            if name in baseline.get("models", {})
        },
        "oracle_results": {},
        "split_counts": dict(Counter(s.split for s in samples)),
    }
    if args.oracle in ("pooled", "both"):
        pooled = np.vstack([s.z_future_target for s in samples]).astype(np.float32)
        results["oracle_results"]["pooled_future_uih"] = run_oracle_ranker(
            samples, articles_by_id, article_to_idx, emb, args, "pooled_future_uih", pooled
        )
    if args.oracle in ("structured", "both"):
        structured, stats = load_structured_anchor(samples, args)
        results["structured_anchor_stats"] = stats
        results["oracle_results"]["structured_full_signal_anchor"] = run_oracle_ranker(
            samples, articles_by_id, article_to_idx, emb, args, "structured_full_signal_anchor", structured
        )
    write_json(out_dir / "results.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
