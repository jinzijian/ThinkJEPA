#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset

from ebnerd_pipeline import article_lookup, cosine, mean_embedding, ranking_metrics, unit, write_json


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


def load_cached(path: Path) -> tuple[list[Sample], dict[str, Any]]:
    with path.open("rb") as f:
        packed = pickle.load(f)
    return packed["samples"], packed.get("prepare_summary", {})


def load_article_embeddings(path: Path) -> tuple[dict[int, int], np.ndarray]:
    packed = np.load(path)
    article_ids = packed["article_ids"].astype(np.int64)
    dense = packed["embeddings"].astype(np.float32)
    dense = normalize(dense).astype(np.float32)
    return {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}, dense


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_answer_anchor(samples: list[Sample], article_to_idx: dict[int, int], emb: np.ndarray) -> np.ndarray:
    vectors = []
    for sample in samples:
        clicked = list(sample.current_clicked)
        clicked_set = set(clicked)
        not_clicked = [aid for aid in sample.current_inview if aid not in clicked_set]
        shown = mean_embedding(sample.current_inview, article_to_idx, emb)
        pos = mean_embedding(clicked, article_to_idx, emb)
        neg = mean_embedding(not_clicked, article_to_idx, emb)
        vectors.append(unit(pos + 0.15 * shown - 0.35 * neg))
    return np.vstack(vectors).astype(np.float32)


def load_structured_anchor(samples: list[Sample], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    from ebnerd_sequence_jepa import build_sequence_arrays, load_article_embeddings as load_padded

    article_to_idx, emb_table = load_padded(Path(args.article_embeddings))
    arrays = build_sequence_arrays(samples, Path(args.data_dir), article_to_idx, emb_table, args)
    return arrays["anchor"].astype(np.float32), arrays["stats"]


def oracle_vectors(
    name: str,
    samples: list[Sample],
    article_to_idx: dict[int, int],
    emb: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if name == "pooled_future":
        return np.vstack([s.z_future_target for s in samples]).astype(np.float32), {"oracle": name}
    if name == "structured_future":
        vecs, stats = load_structured_anchor(samples, args)
        return vecs, {"oracle": name, "structured_anchor_stats": stats}
    if name == "answer_current":
        return make_answer_anchor(samples, article_to_idx, emb), {
            "oracle": name,
            "note": "Diagnostic only: includes current clicked/not-clicked labels and is not deployable.",
        }
    if name == "pooled_future_plus_answer":
        future = np.vstack([s.z_future_target for s in samples]).astype(np.float32)
        answer = make_answer_anchor(samples, article_to_idx, emb)
        mixed = np.vstack([unit(f + args.answer_mix_weight * a) for f, a in zip(future, answer)]).astype(np.float32)
        return mixed, {
            "oracle": name,
            "answer_mix_weight": args.answer_mix_weight,
            "note": "Diagnostic only: mixes true future UIH with current labels.",
        }
    if name == "post_current_uih":
        future = np.vstack([s.z_future_target for s in samples]).astype(np.float32)
        current_response = make_answer_anchor(samples, article_to_idx, emb)
        mixed = np.vstack(
            [
                unit(args.current_response_weight * current + args.future_weight * fut)
                for current, fut in zip(current_response, future)
            ]
        ).astype(np.float32)
        return mixed, {
            "oracle": name,
            "current_response_weight": args.current_response_weight,
            "future_weight": args.future_weight,
            "note": "Oracle target: current impression response plus later future UIH; not available at serving time.",
        }
    raise ValueError(f"unknown oracle: {name}")


def article_dense_meta(articles_by_id: dict[int, dict], aid: int, sample: Sample, pos: int, n: int) -> list[float]:
    art = articles_by_id.get(aid, {})
    published = art.get("published_time", pd.NaT)
    age_hours = 0.0
    if not pd.isna(published):
        age_hours = max(0.0, (sample.time - published).total_seconds() / 3600.0)
    hour = sample.time.hour + sample.time.minute / 60.0
    dow = sample.time.dayofweek
    past_shown = Counter(sample.past_shown)
    past_clicked = Counter(sample.past_clicked)
    return [
        math.log1p(len(sample.current_inview)),
        math.log1p(len(sample.past_shown)),
        math.log1p(len(sample.past_clicked)),
        math.log1p(len(sample.past_not_clicked)),
        float(pos) / max(1.0, float(n - 1)),
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
        math.sin(2 * math.pi * dow / 7.0),
        math.cos(2 * math.pi * dow / 7.0),
        float(sample.context.get("is_sso_user", 0)),
        float(sample.context.get("is_subscriber", 0)),
        float(sample.context.get("age", -1.0)) / 100.0,
        float(art.get("premium", 0)),
        math.log1p(age_hours),
        math.log1p(float(art.get("total_inviews", 0.0))),
        math.log1p(float(art.get("total_pageviews", 0.0))),
        math.log1p(float(art.get("total_read_time", 0.0))),
        float(art.get("sentiment_score", 0.0)),
        float(past_shown[aid] > 0),
        float(past_clicked[aid] > 0),
        math.log1p(past_shown[aid]),
        math.log1p(past_clicked[aid]),
    ]


class ImpressionDataset(Dataset):
    def __init__(self, samples: list[Sample], split: str):
        self.samples = [s for s in samples if s.split == split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


class BatchBuilder:
    def __init__(
        self,
        article_to_idx: dict[int, int],
        emb: np.ndarray,
        oracle: np.ndarray,
        articles_by_id: dict[int, dict],
        user_to_idx: dict[int, int],
        item_to_idx: dict[int, int],
        include_future: bool,
    ) -> None:
        self.article_to_idx = article_to_idx
        self.emb = emb
        self.oracle = oracle
        self.articles_by_id = articles_by_id
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.include_future = include_future
        self.emb_dim = emb.shape[1]
        self.scalar_dim = 28

    def __call__(self, samples: list[Sample]) -> dict[str, torch.Tensor | np.ndarray]:
        bsz = len(samples)
        max_c = max(len(s.current_inview) for s in samples)
        d = self.emb_dim
        dense_dim = 10 * d + self.scalar_dim
        dense = np.zeros((bsz, max_c, dense_dim), dtype=np.float32)
        labels = np.zeros((bsz, max_c), dtype=np.float32)
        mask = np.zeros((bsz, max_c), dtype=bool)
        user_idx = np.zeros((bsz, max_c), dtype=np.int64)
        item_idx = np.zeros((bsz, max_c), dtype=np.int64)
        group_ids = np.zeros((bsz, max_c), dtype=np.int64)
        row_labels = []
        for bi, sample in enumerate(samples):
            z_past = sample.z_current.astype(np.float32)
            z_set = sample.z_current_set.astype(np.float32)
            z_future = self.oracle[sample.sample_id].astype(np.float32) if self.include_future else np.zeros(d, dtype=np.float32)
            z_delta = (z_future - z_past).astype(np.float32)
            clicked = set(sample.current_clicked)
            n = len(sample.current_inview)
            uidx = self.user_to_idx.get(sample.user_id, 0)
            for ci, aid in enumerate(sample.current_inview):
                idx = self.article_to_idx.get(aid)
                item = self.emb[idx].astype(np.float32) if idx is not None else np.zeros(d, dtype=np.float32)
                scalar = np.array(
                    [
                        cosine(item, z_future),
                        cosine(item, z_delta),
                        cosine(item, z_past),
                        cosine(item, z_set),
                        cosine(z_future, z_past),
                        cosine(z_future, z_set),
                    ]
                    + article_dense_meta(self.articles_by_id, aid, sample, ci, n),
                    dtype=np.float32,
                )
                vec = np.concatenate(
                    [
                        item,
                        z_past,
                        z_set,
                        z_future,
                        z_delta,
                        item * z_future,
                        item * z_delta,
                        np.abs(item - z_future),
                        item * z_past,
                        np.abs(item - z_past),
                        scalar,
                    ]
                )
                dense[bi, ci] = vec
                labels[bi, ci] = float(aid in clicked)
                mask[bi, ci] = True
                user_idx[bi, ci] = uidx
                item_idx[bi, ci] = self.item_to_idx.get(aid, 0)
                group_ids[bi, ci] = sample.impression_id
                row_labels.append(int(aid in clicked))
        return {
            "dense": torch.from_numpy(dense),
            "labels": torch.from_numpy(labels),
            "mask": torch.from_numpy(mask),
            "user_idx": torch.from_numpy(user_idx),
            "item_idx": torch.from_numpy(item_idx),
            "group_ids": group_ids,
        }


class NeuralOracleRanker(nn.Module):
    def __init__(
        self,
        dense_dim: int,
        num_users: int,
        num_items: int,
        user_emb_dim: int,
        item_emb_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users + 1, user_emb_dim, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, item_emb_dim, padding_idx=0)
        in_dim = dense_dim + user_emb_dim + item_emb_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, dense: torch.Tensor, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        features = torch.cat([dense, self.user_emb(user_idx), self.item_id_emb(item_idx)], dim=-1)
        return self.net(features).squeeze(-1)


def listwise_bce_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, bce_weight: float) -> torch.Tensor:
    scores = scores.masked_fill(~mask, -1e4)
    pos = labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    target = labels / pos
    listwise = -(target * F.log_softmax(scores, dim=1)).sum(dim=1)
    if bce_weight <= 0.0:
        return listwise.mean()
    valid = mask.float()
    neg = ((1.0 - labels) * valid).sum().clamp_min(1.0)
    pos_count = (labels * valid).sum().clamp_min(1.0)
    pos_weight = (neg / pos_count).detach().clamp(max=20.0)
    bce = F.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weight, reduction="none")
    bce = (bce * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return (listwise + bce_weight * bce).mean()


@torch.no_grad()
def predict_split(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    scores = []
    groups = []
    for batch in loader:
        dense = batch["dense"].to(device)
        user_idx = batch["user_idx"].to(device)
        item_idx = batch["item_idx"].to(device)
        mask = batch["mask"].to(device)
        out = model(dense, user_idx, item_idx).detach().cpu().numpy()
        lab = batch["labels"].numpy()
        m = batch["mask"].numpy()
        group = batch["group_ids"]
        labels.append(lab[m])
        scores.append(out[m])
        groups.append(group[m])
    return np.concatenate(labels), np.concatenate(scores), np.concatenate(groups)


def train_model(
    name: str,
    samples: list[Sample],
    article_to_idx: dict[int, int],
    emb: np.ndarray,
    oracle: np.ndarray,
    articles_by_id: dict[int, dict],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    include_future: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device)
    builder = BatchBuilder(article_to_idx, emb, oracle, articles_by_id, user_to_idx, item_to_idx, include_future)
    loaders = {
        split: DataLoader(
            ImpressionDataset(samples, split),
            batch_size=args.batch_size,
            shuffle=(split == "ranker_train"),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=builder,
        )
        for split in ["ranker_train", "ranker_val", "ranker_test"]
    }
    dense_dim = 10 * emb.shape[1] + builder.scalar_dim
    model = NeuralOracleRanker(
        dense_dim=dense_dim,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        user_emb_dim=args.user_emb_dim,
        item_emb_dim=args.item_emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_metric = -float("inf")
    history = []
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["ranker_train"]:
            dense = batch["dense"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            user_idx = batch["user_idx"].to(device)
            item_idx = batch["item_idx"].to(device)
            scores = model(dense, user_idx, item_idx)
            loss = listwise_bce_loss(scores, labels, mask, args.bce_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        y_val, s_val, g_val = predict_split(model, loaders["ranker_val"], device)
        val_metrics = ranking_metrics(y_val, s_val, g_val)
        metric = val_metrics["ndcg@10"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "ranker_val": val_metrics})
        print(
            json.dumps(
                {
                    name: {
                        "epoch": epoch,
                        "loss": float(np.mean(losses)),
                        "val_auc": val_metrics["auc"],
                        "val_mrr": val_metrics["mrr"],
                        "val_ndcg10": val_metrics["ndcg@10"],
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    out = {"history": history, "include_future": include_future}
    for split, loader in loaders.items():
        y, s, g = predict_split(model, loader, device)
        out[split] = ranking_metrics(y, s, g)
    return out


def candidate_dot_metrics(samples: list[Sample], split: str, article_to_idx: dict[int, int], emb: np.ndarray, oracle: np.ndarray) -> dict:
    labels = []
    scores = []
    groups = []
    for sample in samples:
        if sample.split != split:
            continue
        z = unit(oracle[sample.sample_id])
        clicked = set(sample.current_clicked)
        for aid in sample.current_inview:
            idx = article_to_idx.get(aid)
            score = cosine(emb[idx], z) if idx is not None else 0.0
            labels.append(int(aid in clicked))
            scores.append(score)
            groups.append(sample.impression_id)
    return ranking_metrics(np.asarray(labels), np.asarray(scores), np.asarray(groups))


def summarize_baseline(path: str) -> dict[str, Any]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    if "models" not in raw:
        return raw.get("baseline_test", {})
    return {
        k: v["ranker_test"]
        for k, v in raw["models"].items()
        if isinstance(v, dict) and "ranker_test" in v
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural oracle adapter for EB-NeRD future-UIH reranking.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--article-embeddings", required=True)
    parser.add_argument("--baseline-results", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--oracle",
        choices=["pooled_future", "structured_future", "answer_current", "pooled_future_plus_answer", "post_current_uih"],
        default="pooled_future",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--user-emb-dim", type=int, default=32)
    parser.add_argument("--item-emb-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--bce-weight", type=float, default=0.20)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--skip-no-future", action="store_true")
    parser.add_argument("--answer-mix-weight", type=float, default=1.0)
    parser.add_argument("--current-response-weight", type=float, default=1.0)
    parser.add_argument("--future-weight", type=float, default=1.0)
    parser.add_argument("--max-source-events", type=int, default=128)
    parser.add_argument("--max-target-events", type=int, default=96)
    parser.add_argument("--max-history-events", type=int, default=40)
    parser.add_argument("--max-past-impressions", type=int, default=8)
    parser.add_argument("--max-current-events", type=int, default=32)
    parser.add_argument("--max-future-impressions", type=int, default=8)
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--session-buckets", type=int, default=4096)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, prepare = load_cached(Path(args.samples_pkl))
    article_to_idx, emb = load_article_embeddings(Path(args.article_embeddings))
    articles = pd.read_parquet(Path(args.data_dir) / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    articles_by_id = article_lookup(articles)
    oracle, oracle_meta = oracle_vectors(args.oracle, samples, article_to_idx, emb, args)
    oracle = np.vstack([unit(x) for x in oracle]).astype(np.float32)
    users = sorted({s.user_id for s in samples if s.split.startswith("ranker_")})
    items = sorted({aid for s in samples if s.split.startswith("ranker_") for aid in s.current_inview})
    user_to_idx = {uid: idx + 1 for idx, uid in enumerate(users)}
    item_to_idx = {aid: idx + 1 for idx, aid in enumerate(items)}

    results: dict[str, Any] = {
        "prepare": prepare,
        "oracle": oracle_meta,
        "baseline_test": summarize_baseline(args.baseline_results),
        "split_counts": dict(Counter(s.split for s in samples)),
        "candidate_dot": {
            split: candidate_dot_metrics(samples, split, article_to_idx, emb, oracle)
            for split in ["ranker_val", "ranker_test"]
        },
        "models": {},
    }
    variants = []
    if not args.skip_no_future:
        variants.append(("N0_neural_no_future", False))
    variants.append((f"O1_neural_{args.oracle}", True))
    for model_name, include_future in variants:
        print(f"training {model_name}", flush=True)
        results["models"][model_name] = train_model(
            model_name,
            samples,
            article_to_idx,
            emb,
            oracle,
            articles_by_id,
            user_to_idx,
            item_to_idx,
            include_future,
            args,
        )
        print(json.dumps({model_name: results["models"][model_name]["ranker_test"]}, indent=2), flush=True)
        write_json(out_dir / "results.partial.json", results)
    write_json(out_dir / "results.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
