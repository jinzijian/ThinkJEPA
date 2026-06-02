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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def raw_candidate_distribution_loss(
    pred_future: torch.Tensor,
    target_future: torch.Tensor,
    item_emb: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher = torch.einsum("bcd,bd->bc", item_emb, target_future) / temperature
    student = torch.einsum("bcd,bd->bc", item_emb, pred_future) / temperature
    teacher = teacher.masked_fill(~mask, -1e4)
    student = student.masked_fill(~mask, -1e4)
    teacher_prob = F.softmax(teacher, dim=1).detach()
    distill = -(teacher_prob * F.log_softmax(student, dim=1)).sum(dim=1).mean()
    label_rank = q.listwise_loss(student, labels, mask)
    return distill, label_rank


def raw_inbatch_nce(pred_future: torch.Tensor, target_future: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = pred_future @ target_future.T / temperature
    labels = torch.arange(pred_future.shape[0], device=pred_future.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def raw_delta_loss(pred_delta: torch.Tensor, batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    target_future = batch["post"]
    source = batch["source"]
    target_delta = target_future - source
    pred_future = source + pred_delta

    delta_mse = F.mse_loss(pred_delta, target_delta)
    future_mse = F.mse_loss(pred_future, target_future)
    distill, label_rank = raw_candidate_distribution_loss(
        pred_future,
        target_future,
        batch["item_emb"],
        batch["labels"],
        batch["mask"],
        args.candidate_temperature,
    )
    nce = raw_inbatch_nce(pred_future, target_future.detach(), args.inbatch_temperature)
    total = (
        args.delta_weight * delta_mse
        + args.future_weight * future_mse
        + args.candidate_kl_weight * distill
        + args.candidate_label_weight * label_rank
        + args.inbatch_nce_weight * nce
    )
    parts = {
        "delta_mse": float(delta_mse.detach().cpu()),
        "future_mse": float(future_mse.detach().cpu()),
        "candidate_kl": float(distill.detach().cpu()),
        "candidate_label": float(label_rank.detach().cpu()),
        "inbatch_nce": float(nce.detach().cpu()),
    }
    return total, parts


@torch.no_grad()
def predict_delta(
    model: torch.nn.Module,
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


def vector_summary(name: str, pred: np.ndarray, target: np.ndarray, idxs: list[int]) -> dict[str, Any]:
    if not idxs:
        return {"name": name, "num_samples": 0}
    p = pred[idxs].astype(np.float32)
    t = target[idxs].astype(np.float32)
    pairwise_n = min(len(idxs), 5000)
    sub = p[:pairwise_n]
    sub_norm = sub / np.clip(np.linalg.norm(sub, axis=1, keepdims=True), 1e-8, None)
    pair = sub_norm @ sub_norm.T
    tri = pair[np.triu_indices(pairwise_n, k=1)] if pairwise_n > 1 else np.array([0.0])
    return {
        "name": name,
        "num_samples": int(len(idxs)),
        "cosine_to_target": float(np.mean([q.cosine(a, b) for a, b in zip(p, t)])),
        "mse_to_target": float(np.mean((p - t) ** 2)),
        "pred_mean_pairwise_cosine": float(np.mean(tri)),
        "pred_norm_mean": float(np.linalg.norm(p, axis=1).mean()),
        "target_norm_mean": float(np.linalg.norm(t, axis=1).mean()),
    }


def train_raw_delta_jepa(
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
    model = q.QwenTextJepaPredictor(
        emb_dim=item_emb.shape[1],
        scalar_dim=q.SetBatchBuilder.scalar_dim,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.jepa_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        use_candidate_mix=args.jepa_candidate_mix,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.jepa_lr, weight_decay=args.weight_decay)
    best_state = None
    best_metric = -float("inf")
    history = []
    patience = args.patience
    for epoch in range(1, args.jepa_epochs + 1):
        model.train()
        losses = []
        parts: dict[str, list[float]] = defaultdict(list)
        for batch in loaders["jepa_train"]:
            moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            pred_delta = model(moved)
            loss, loss_parts = raw_delta_loss(pred_delta, moved, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in loss_parts.items():
                parts[key].append(value)
        val_delta, val_future = predict_delta(model, loaders["jepa_val"], source_emb, device, len(samples), item_emb.shape[1])
        val_idxs = [s.sample_id for s in samples if s.split == "jepa_val"]
        target_delta = target_emb - source_emb
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
                    "raw_delta_jepa": {
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

    target_delta = target_emb - source_emb
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
    parser = argparse.ArgumentParser(description="Raw unnormalized delta JEPA for EB-NeRD full-signal Qwen UIH.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qwen-uih-cache", required=True)
    parser.add_argument("--reference-results", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--jepa-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--jepa-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--jepa-layers", type=int, default=2)
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
    parser.add_argument("--jepa-candidate-mix", action="store_true")
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
        "setting": {
            "jepa_target": "raw_delta = target_future - source_uih",
            "delta_normalized": False,
            "pred_future": "source_uih + pred_raw_delta",
            "predictor_input_post_delta_scalar_leakage": False,
        },
        "split_counts": dict(Counter(s.split for s in samples)),
        "raw_delta_jepa": {},
        "ranker_models": {},
    }
    q.write_json(out_dir / "results.partial.json", results)

    print("training raw unnormalized delta JEPA", flush=True)
    pred_delta, pred_future, jepa_metrics = train_raw_delta_jepa(
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
    results["raw_delta_jepa"] = jepa_metrics
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
        "D0_raw_delta_predicted_only",
        pred_future,
        use_source_token=False,
        use_post_token=True,
        use_source_feature=False,
        use_post_feature=True,
        use_delta_feature=False,
    )
    add_ranker(
        "D1_source_raw_delta_predicted",
        pred_future,
        use_source_token=True,
        use_post_token=True,
        use_source_feature=True,
        use_post_feature=True,
        use_delta_feature=False,
    )
    add_ranker(
        "D2_source_raw_delta_predicted_delta",
        pred_future,
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
