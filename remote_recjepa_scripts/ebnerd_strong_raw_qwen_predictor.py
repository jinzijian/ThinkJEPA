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


class StrongSetTransitionPredictor(nn.Module):
    """Non-mask JEPA-style predictor over frozen Qwen source/candidate latents."""

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
        residual_blocks: int,
    ) -> None:
        super().__init__()
        self.vec_proj = nn.Linear(emb_dim, d_model)
        self.scalar_proj = nn.Linear(scalar_dim, d_model)
        self.user_emb = nn.Embedding(num_users + 1, d_model, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(4, d_model)
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
        self.query = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.raw_mix_proj = nn.Linear(emb_dim, d_model)
        blocks: list[nn.Module] = []
        for _ in range(residual_blocks):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, d_model),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_tok = self.user_emb(batch["user_idx"]) + self.type_emb.weight[0]
        source_tok = self.vec_proj(batch["source"]) + self.type_emb.weight[1]
        set_tok = self.vec_proj(batch["candidate_set"]) + self.type_emb.weight[2]
        item_tok = (
            self.vec_proj(batch["item_emb"])
            + self.scalar_proj(batch["scalar"])
            + self.item_id_emb(batch["item_idx"])
            + self.type_emb.weight[3]
        )
        x = torch.cat([user_tok.unsqueeze(1), source_tok.unsqueeze(1), set_tok.unsqueeze(1), item_tok], dim=1)
        ctx_mask = torch.zeros((x.shape[0], 3), dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([ctx_mask, ~batch["mask"]], dim=1)
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        user_out, source_out, set_out = out[:, 0], out[:, 1], out[:, 2]
        item_out = out[:, 3:]

        query = self.query(torch.cat([user_out, source_out, set_out], dim=-1))
        logits = torch.einsum("bcd,bd->bc", item_out, query) / math.sqrt(float(item_out.shape[-1]))
        logits = logits.masked_fill(~batch["mask"], -1e4)
        weight = F.softmax(logits, dim=1)
        item_mix = torch.einsum("bc,bcd->bd", weight, item_out)
        raw_item_mix = torch.einsum("bc,bcd->bd", weight, batch["item_emb"])

        h = source_out + set_out + user_out + item_mix + self.raw_mix_proj(raw_item_mix)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


def strong_delta_loss(
    pred_delta: torch.Tensor,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss, parts = raw_delta_loss(pred_delta, batch, args)
    target_future = batch["post"]
    source = batch["source"]
    target_delta = target_future - source
    pred_future = source + pred_delta
    delta_norm = F.mse_loss(pred_delta.norm(dim=-1), target_delta.norm(dim=-1))
    future_norm = F.mse_loss(pred_future.norm(dim=-1), target_future.norm(dim=-1))
    loss = loss + args.delta_norm_weight * delta_norm + args.future_norm_weight * future_norm
    parts["delta_norm"] = float(delta_norm.detach().cpu())
    parts["future_norm"] = float(future_norm.detach().cpu())
    return loss, parts


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


def train_strong_predictor(
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
        ["ranker_train", "ranker_val", "ranker_test"],
        use_source_feature=True,
        use_post_feature=False,
        use_delta_feature=False,
    )
    model = StrongSetTransitionPredictor(
        emb_dim=item_emb.shape[1],
        scalar_dim=q.SetBatchBuilder.scalar_dim,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        d_model=args.predictor_d_model,
        nhead=args.predictor_heads,
        layers=args.predictor_layers,
        ff_dim=args.predictor_ff_dim,
        dropout=args.dropout,
        residual_blocks=args.predictor_residual_blocks,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.predictor_lr, weight_decay=args.weight_decay)
    best_state = None
    best_metric = -float("inf")
    history = []
    patience = args.predictor_patience
    target_delta = target_emb - source_emb
    for epoch in range(1, args.predictor_epochs + 1):
        model.train()
        losses = []
        parts: dict[str, list[float]] = defaultdict(list)
        for batch in loaders["ranker_train"]:
            moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            pred_delta = model(moved)
            loss, loss_parts = strong_delta_loss(pred_delta, moved, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in loss_parts.items():
                parts[key].append(value)

        val_delta, val_future = predict_delta(model, loaders["ranker_val"], source_emb, device, len(samples), item_emb.shape[1])
        val_idxs = [s.sample_id for s in samples if s.split == "ranker_val"]
        val_future_summary = vector_summary("ranker_val_future", val_future, target_emb, val_idxs)
        val_delta_summary = vector_summary("ranker_val_delta", val_delta, target_delta, val_idxs)
        val_dot = q.candidate_dot_metrics(samples, "ranker_val", article_to_idx, item_emb, val_future)
        metric = val_dot["ndcg@10"]
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "loss_parts": {k: float(np.mean(v)) for k, v in parts.items()},
            "ranker_val_future": val_future_summary,
            "ranker_val_delta": val_delta_summary,
            "ranker_val_candidate_dot": val_dot,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "strong_raw_qwen_predictor": {
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
            patience = args.predictor_patience
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
    for split in ["ranker_train", "ranker_val", "ranker_test"]:
        idxs = [s.sample_id for s in samples if s.split == split]
        split_eval[split] = {
            "future": vector_summary(split, pred_future_all, target_emb, idxs),
            "delta": vector_summary(split, pred_delta_all, target_delta, idxs),
            "candidate_dot": q.candidate_dot_metrics(samples, split, article_to_idx, item_emb, pred_future_all),
        }
    return pred_delta_all, pred_future_all, {"history": history, "split_eval": split_eval}


def main() -> None:
    parser = argparse.ArgumentParser(description="Strong non-mask JEPA-style predictor on raw no-SVD Qwen UIH latents.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qwen-raw-cache", required=True)
    parser.add_argument("--reference-results", default="")
    parser.add_argument("--direct-results", default="")
    parser.add_argument("--no-svd-oracle-results", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--predictor-epochs", type=int, default=8)
    parser.add_argument("--predictor-patience", type=int, default=3)
    parser.add_argument("--predictor-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=768)
    parser.add_argument("--predictor-d-model", type=int, default=512)
    parser.add_argument("--predictor-heads", type=int, default=8)
    parser.add_argument("--predictor-layers", type=int, default=4)
    parser.add_argument("--predictor-ff-dim", type=int, default=2048)
    parser.add_argument("--predictor-residual-blocks", type=int, default=2)
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
    parser.add_argument("--delta-norm-weight", type=float, default=0.2)
    parser.add_argument("--future-norm-weight", type=float, default=0.2)
    args = parser.parse_args()

    q.set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples, prepare = load_samples(Path(args.samples_pkl))
    samples, sample_cache_idx = select_ranker_samples(all_samples, args.max_samples_per_split)
    cache = np.load(args.qwen_raw_cache)
    article_ids = cache["article_ids"].astype(np.int64)
    item_emb = cache["article_emb"].astype(np.float32)
    source_emb_all = cache["source_emb"].astype(np.float32)
    target_emb_all = cache["target_emb"].astype(np.float32)
    source_emb = source_emb_all[sample_cache_idx]
    target_emb = target_emb_all[sample_cache_idx]
    if len(samples) != source_emb.shape[0]:
        raise ValueError(f"sample/cache mismatch: samples={len(samples)} source_emb={source_emb.shape[0]}")
    article_to_idx = {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}

    articles_df = pd.read_parquet(Path(args.data_dir) / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    articles_by_id = q.article_lookup(articles_df)
    users = sorted({s.user_id for s in samples})
    items = sorted({aid for s in samples for aid in s.current_inview})
    user_to_idx = {uid: idx + 1 for idx, uid in enumerate(users)}
    item_to_idx = {aid: idx + 1 for idx, aid in enumerate(items)}

    results: dict[str, Any] = {
        "prepare": prepare,
        "qwen_raw_cache": str(args.qwen_raw_cache),
        "reference_results": json.loads(Path(args.reference_results).read_text()) if args.reference_results else None,
        "direct_results": json.loads(Path(args.direct_results).read_text()) if args.direct_results else None,
        "no_svd_oracle_results": json.loads(Path(args.no_svd_oracle_results).read_text()) if args.no_svd_oracle_results else None,
        "setting": {
            "encoder": "frozen Qwen raw 2560-dim source/target UIH latents, no SVD",
            "predictor": "non-mask set-transition transformer predictor",
            "input": "z_source plus current candidate set",
            "target": "raw delta = z_future - z_source; pred_future = z_source + pred_delta",
            "predictor_train_split": "ranker_train",
            "predictor_early_stop_split": "ranker_val",
            "predictor_input_post_delta_scalar_leakage": False,
        },
        "split_counts": dict(Counter(s.split for s in samples)),
        "strong_predictor": {},
        "ranker_models": {},
    }
    q.write_json(out_dir / "results.partial.json", results)

    print("training strong raw Qwen non-mask predictor", flush=True)
    pred_delta, pred_future, predictor_metrics = train_strong_predictor(
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
    results["strong_predictor"] = predictor_metrics
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
        "P0_strong_predicted_only_raw_qwen_no_svd",
        pred_future,
        use_source_token=False,
        use_post_token=True,
        use_source_feature=False,
        use_post_feature=True,
        use_delta_feature=False,
    )
    add_ranker(
        "P1_source_strong_predicted_raw_qwen_no_svd",
        pred_future,
        use_source_token=True,
        use_post_token=True,
        use_source_feature=True,
        use_post_feature=True,
        use_delta_feature=False,
    )
    add_ranker(
        "P2_source_strong_predicted_delta_raw_qwen_no_svd",
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
