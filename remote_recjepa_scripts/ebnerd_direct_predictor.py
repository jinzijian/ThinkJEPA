#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from torch.utils.data import DataLoader

import ebnerd_qwen_uih_text_jepa as q
from ebnerd_raw_delta_jepa import raw_delta_loss, vector_summary


class SampleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "Sample":
            return q.Sample
        return super().find_class(module, name)


def load_samples(path: Path) -> tuple[list[q.Sample], dict[str, Any]]:
    with path.open("rb") as f:
        packed = SampleUnpickler(f).load()
    return packed["samples"], packed.get("prepare_summary", {})


class DirectSetDeltaPredictor(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        scalar_dim: int,
        num_users: int,
        num_items: int,
        d_model: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.source_proj = nn.Linear(emb_dim, d_model)
        self.item_proj = nn.Linear(emb_dim, d_model)
        self.scalar_proj = nn.Linear(scalar_dim, d_model)
        self.user_emb = nn.Embedding(num_users + 1, d_model, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(
            nn.LayerNorm(4 * emb_dim + d_model),
            nn.Linear(4 * emb_dim + d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        source = batch["source"]
        candidate_set = batch["candidate_set"]
        item_h = self.item_proj(batch["item_emb"]) + self.scalar_proj(batch["scalar"]) + self.item_id_emb(batch["item_idx"])
        query = self.source_proj(source) + self.user_emb(batch["user_idx"])
        item_h = self.norm(item_h)
        query = self.norm(query)
        logits = torch.einsum("bcd,bd->bc", item_h, query) / math.sqrt(float(item_h.shape[-1]))
        logits = logits.masked_fill(~batch["mask"], -1e4)
        weight = F.softmax(logits, dim=1)
        item_mix = torch.einsum("bc,bcd->bd", weight, batch["item_emb"])
        user_ctx = self.user_emb(batch["user_idx"])
        x = torch.cat([source, candidate_set, item_mix, source * item_mix, user_ctx], dim=-1)
        return self.out(x)


@torch.no_grad()
def predict_delta(
    model: nn.Module,
    loader: DataLoader,
    source_emb: np.ndarray,
    device: torch.device,
    n_samples: int,
    emb_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_delta = np.zeros((n_samples, emb_dim), dtype=np.float32)
    pred_future = np.zeros((n_samples, emb_dim), dtype=np.float32)
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        delta = model(moved).detach().cpu().numpy().astype(np.float32)
        sample_ids = batch["sample_ids"]
        pred_delta[sample_ids] = delta
        pred_future[sample_ids] = source_emb[sample_ids].astype(np.float32) + delta
    return pred_delta, pred_future


def train_direct_predictor(
    samples: list[q.Sample],
    article_to_idx: dict[int, int],
    item_emb: np.ndarray,
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    articles_by_id: dict[int, dict],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    device = torch.device(args.device)
    loaders = q.make_loaders(
        samples,
        article_to_idx,
        item_emb,
        source_emb,
        target_emb,
        articles_by_id,
        user_to_idx,
        item_to_idx,
        args,
        ["jepa_train", "jepa_val", "ranker_train", "ranker_val", "ranker_test"],
        use_source_feature=True,
        use_post_feature=False,
        use_delta_feature=False,
    )
    model = DirectSetDeltaPredictor(
        emb_dim=item_emb.shape[1],
        scalar_dim=q.SetBatchBuilder.scalar_dim,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        d_model=args.d_model,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.predictor_lr, weight_decay=args.weight_decay)
    best_state = None
    best_metric = -float("inf")
    history = []
    patience = args.patience
    target_delta = target_emb - source_emb
    for epoch in range(1, args.predictor_epochs + 1):
        model.train()
        losses = []
        parts: dict[str, list[float]] = defaultdict(list)
        for batch in loaders["jepa_train"]:
            moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            pred_delta = model(moved)
            loss, loss_parts = raw_delta_loss(pred_delta, moved, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in loss_parts.items():
                parts[key].append(value)
        val_delta, val_future = predict_delta(model, loaders["jepa_val"], source_emb, device, len(samples), item_emb.shape[1])
        val_idxs = [s.sample_id for s in samples if s.split == "jepa_val"]
        val_future_summary = vector_summary("jepa_val_future", val_future, target_emb, val_idxs)
        val_delta_summary = vector_summary("jepa_val_delta", val_delta, target_delta, val_idxs)
        val_dot = q.candidate_dot_metrics(samples, "jepa_val", article_to_idx, item_emb, val_future)
        metric = val_dot["ndcg@10"]
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "loss_parts": {k: float(np.mean(v)) for k, v in parts.items()},
            "jepa_val_future": val_future_summary,
            "jepa_val_delta": val_delta_summary,
            "jepa_val_candidate_dot": val_dot,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "direct_predictor": {
                        "epoch": epoch,
                        "loss": row["loss"],
                        "future_cos": val_future_summary["cosine_to_target"],
                        "delta_cos": val_delta_summary["cosine_to_target"],
                        "future_pairwise": val_future_summary["pred_mean_pairwise_cosine"],
                        "val_dot_mrr": val_dot["mrr"],
                        "val_dot_ndcg10": val_dot["ndcg@10"],
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    pred_delta_all = np.zeros_like(target_emb, dtype=np.float32)
    pred_future_all = np.zeros_like(target_emb, dtype=np.float32)
    for split, loader in loaders.items():
        split_delta, split_future = predict_delta(model, loader, source_emb, device, len(samples), item_emb.shape[1])
        idxs = [s.sample_id for s in samples if s.split == split]
        pred_delta_all[idxs] = split_delta[idxs]
        pred_future_all[idxs] = split_future[idxs]

    split_eval = {}
    for split in ["jepa_val", "ranker_train", "ranker_val", "ranker_test"]:
        idxs = [s.sample_id for s in samples if s.split == split]
        split_eval[split] = {
            "future": vector_summary(split, pred_future_all, target_emb, idxs),
            "delta": vector_summary(split, pred_delta_all, target_delta, idxs),
            "candidate_dot": q.candidate_dot_metrics(samples, split, article_to_idx, item_emb, pred_future_all),
        }
    return pred_delta_all, pred_future_all, {"history": history, "split_eval": split_eval}


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-JEPA direct future-UIH predictor baseline for EB-NeRD Qwen UIH.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qwen-uih-cache", required=True)
    parser.add_argument("--reference-results", default="")
    parser.add_argument("--raw-delta-results", default="")
    parser.add_argument("--lgbm-results", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--predictor-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--predictor-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--init-post-dot-scale", type=float, default=5.0)
    parser.add_argument("--candidate-temperature", type=float, default=0.07)
    parser.add_argument("--inbatch-temperature", type=float, default=0.07)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--future-weight", type=float, default=0.2)
    parser.add_argument("--candidate-kl-weight", type=float, default=1.0)
    parser.add_argument("--candidate-label-weight", type=float, default=1.0)
    parser.add_argument("--inbatch-nce-weight", type=float, default=0.1)
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

    results: dict[str, Any] = {
        "prepare": prepare,
        "qwen_uih_cache": str(args.qwen_uih_cache),
        "reference_results": json.loads(Path(args.reference_results).read_text()) if args.reference_results else None,
        "raw_delta_results": json.loads(Path(args.raw_delta_results).read_text()) if args.raw_delta_results else None,
        "lgbm_results": json.loads(Path(args.lgbm_results).read_text()) if args.lgbm_results else None,
        "setting": {
            "predictor": "non-JEPA direct attention/MLP predictor",
            "target": "raw_delta = target_future - source_uih",
            "pred_future": "source_uih + pred_raw_delta",
            "predictor_input_post_delta_scalar_leakage": False,
        },
        "split_counts": dict(Counter(s.split for s in samples)),
        "direct_predictor": {},
        "ranker_models": {},
    }
    q.write_json(out_dir / "results.partial.json", results)

    print("training non-JEPA direct predictor", flush=True)
    pred_delta, pred_future, predictor_metrics = train_direct_predictor(
        samples,
        article_to_idx,
        item_emb,
        source_emb,
        target_emb,
        articles_by_id,
        user_to_idx,
        item_to_idx,
        args,
    )
    results["direct_predictor"] = predictor_metrics
    q.write_json(out_dir / "results.partial.json", results)

    def add_ranker(
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
            use_source_token,
            use_post_token,
            use_source_feature,
            use_post_feature,
            use_delta_feature,
            args,
        )
        print(json.dumps({name: results["ranker_models"][name]["ranker_test"]}, indent=2), flush=True)
        q.write_json(out_dir / "results.partial.json", results)

    add_ranker(
        "M0_direct_predicted_only",
        pred_future,
        use_source_token=False,
        use_post_token=True,
        use_source_feature=False,
        use_post_feature=True,
        use_delta_feature=False,
    )
    add_ranker(
        "M1_source_direct_predicted",
        pred_future,
        use_source_token=True,
        use_post_token=True,
        use_source_feature=True,
        use_post_feature=True,
        use_delta_feature=False,
    )

    q.write_json(out_dir / "results.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
