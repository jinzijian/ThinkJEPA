#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

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


def repeat_vector(vec: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(vec.reshape(1, -1).astype(np.float32), n, axis=0)


def mean_vector(x: np.ndarray) -> np.ndarray:
    return q.unit_rows(x.mean(axis=0, keepdims=True))[0]


def user_mean_prediction(samples: list[q.Sample], target_emb: np.ndarray, train_idxs: list[int], global_mean: np.ndarray) -> np.ndarray:
    sums: dict[int, np.ndarray] = {}
    counts: Counter[int] = Counter()
    train_set = set(train_idxs)
    for sample in samples:
        if sample.sample_id not in train_set:
            continue
        sums.setdefault(sample.user_id, np.zeros(target_emb.shape[1], dtype=np.float64))
        sums[sample.user_id] += target_emb[sample.sample_id]
        counts[sample.user_id] += 1
    means = {uid: mean_vector((total / counts[uid]).reshape(1, -1)) for uid, total in sums.items()}
    out = np.zeros_like(target_emb, dtype=np.float32)
    for sample in samples:
        out[sample.sample_id] = means.get(sample.user_id, global_mean)
    return out


def source_cluster_mean_prediction(
    samples: list[q.Sample],
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    train_idxs: list[int],
    global_mean: np.ndarray,
    clusters: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    k = min(clusters, max(2, len(train_idxs)))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init="auto")
    train_source = source_emb[train_idxs].astype(np.float32)
    labels_train = km.fit_predict(train_source)
    sums = np.zeros((k, target_emb.shape[1]), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for sample_idx, label in zip(train_idxs, labels_train):
        sums[label] += target_emb[sample_idx]
        counts[label] += 1
    means = np.repeat(global_mean.reshape(1, -1), k, axis=0).astype(np.float32)
    nonempty = counts > 0
    means[nonempty] = q.unit_rows((sums[nonempty] / counts[nonempty, None]).astype(np.float32))
    all_labels = km.predict(source_emb.astype(np.float32))
    out = means[all_labels].astype(np.float32)
    meta = {
        "clusters": int(k),
        "empty_clusters": int((counts == 0).sum()),
        "min_cluster_count": int(counts[counts > 0].min()) if np.any(counts > 0) else 0,
        "max_cluster_count": int(counts.max()) if counts.size else 0,
    }
    return out, meta


def vector_variance(name: str, vectors: np.ndarray, idxs: list[int]) -> dict[str, Any]:
    x = vectors[idxs].astype(np.float32)
    if len(x) == 0:
        return {"name": name, "num_samples": 0}
    pairwise_n = min(len(x), 5000)
    sub = x[:pairwise_n]
    pair = sub @ sub.T
    tri = pair[np.triu_indices(pairwise_n, k=1)] if pairwise_n > 1 else np.array([0.0])
    centered = x - x.mean(axis=0, keepdims=True)
    var = centered.var(axis=0)
    return {
        "name": name,
        "num_samples": int(len(x)),
        "mean_pairwise_cosine": float(np.mean(tri)),
        "std_pairwise_cosine": float(np.std(tri)),
        "mean_dim_variance": float(var.mean()),
        "max_dim_variance": float(var.max()),
        "mean_l2_to_split_mean": float(np.linalg.norm(centered, axis=1).mean()),
    }


def predictor_eval(
    name: str,
    samples: list[q.Sample],
    article_to_idx: dict[int, int],
    item_emb: np.ndarray,
    pred: np.ndarray,
    target_emb: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "splits": {}}
    for split in ["jepa_val", "ranker_train", "ranker_val", "ranker_test"]:
        idxs = [s.sample_id for s in samples if s.split == split]
        row = q.cosine_summary(split, pred, target_emb, idxs)
        row["candidate_dot"] = q.candidate_dot_metrics(samples, split, article_to_idx, item_emb, pred)
        row["variance"] = vector_variance(split, pred, idxs)
        out["splits"][split] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Mean-collapse diagnostics for full-signal Qwen UIH JEPA on EB-NeRD.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qwen-uih-cache", required=True)
    parser.add_argument("--reference-results", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
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

    samples, prepare = load_samples(Path(args.samples_pkl))
    cache = np.load(args.qwen_uih_cache)
    article_ids = cache["article_ids"].astype(np.int64)
    item_emb = cache["article_emb"].astype(np.float32)
    source_emb = cache["source_emb"].astype(np.float32)
    target_emb = cache["target_emb"].astype(np.float32)
    article_to_idx = {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}

    articles_df = pd.read_parquet(Path(args.data_dir) / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    articles_by_id = q.article_lookup(articles_df)
    users = sorted({s.user_id for s in samples})
    items = sorted({aid for s in samples for aid in s.current_inview})
    user_to_idx = {uid: idx + 1 for idx, uid in enumerate(users)}
    item_to_idx = {aid: idx + 1 for idx, aid in enumerate(items)}

    train_idxs = [s.sample_id for s in samples if s.split == "jepa_train"]
    global_mean = mean_vector(target_emb[train_idxs])
    predictors: dict[str, np.ndarray] = {
        "global_mean": repeat_vector(global_mean, len(samples)),
        "user_mean_jepa_train": user_mean_prediction(samples, target_emb, train_idxs, global_mean),
        "source_as_post": source_emb.astype(np.float32),
    }
    cluster_pred, cluster_meta = source_cluster_mean_prediction(
        samples,
        source_emb,
        target_emb,
        train_idxs,
        global_mean,
        clusters=args.clusters,
        seed=args.seed,
    )
    predictors[f"source_cluster{cluster_meta['clusters']}_target_mean"] = cluster_pred

    results: dict[str, Any] = {
        "prepare": prepare,
        "qwen_uih_cache": str(args.qwen_uih_cache),
        "reference_results": json.loads(Path(args.reference_results).read_text()) if args.reference_results else None,
        "split_counts": dict(Counter(s.split for s in samples)),
        "mean_train_split": "jepa_train",
        "cluster_meta": cluster_meta,
        "target_variance": {
            split: vector_variance(split, target_emb, [s.sample_id for s in samples if s.split == split])
            for split in ["jepa_train", "jepa_val", "ranker_train", "ranker_val", "ranker_test"]
        },
        "predictor_eval": {},
        "ranker_models": {},
    }
    q.write_json(out_dir / "results.partial.json", results)

    for name, pred in predictors.items():
        print(f"evaluating {name}", flush=True)
        results["predictor_eval"][name] = predictor_eval(name, samples, article_to_idx, item_emb, pred, target_emb)
        q.write_json(out_dir / "results.partial.json", results)

    def add_ranker(name: str, post_vectors: np.ndarray, *, p0_like: bool) -> None:
        print(f"training {name}", flush=True)
        results["ranker_models"][name] = q.train_ranker(
            name,
            samples,
            article_to_idx,
            item_emb,
            source_emb,
            post_vectors,
            articles_by_id,
            user_to_idx,
            item_to_idx,
            use_source_token=not p0_like,
            use_post_token=True,
            use_source_feature=not p0_like,
            use_post_feature=True,
            use_delta_feature=not p0_like,
            args=args,
        )
        print(json.dumps({name: results["ranker_models"][name]["ranker_test"]}, indent=2), flush=True)
        q.write_json(out_dir / "results.partial.json", results)

    for base_name in ["global_mean", "user_mean_jepa_train", f"source_cluster{cluster_meta['clusters']}_target_mean"]:
        add_ranker(f"{base_name}_P0_like", predictors[base_name], p0_like=True)
        add_ranker(f"{base_name}_P2_like", predictors[base_name], p0_like=False)

    add_ranker("source_as_post_P2_like", predictors["source_as_post"], p0_like=False)

    q.write_json(out_dir / "results.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
