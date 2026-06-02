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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ebnerd_pipeline import (
    article_lookup,
    first_example,
    standalone_jepa_eval,
    train_variant,
    write_json,
)


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


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / den)


def load_article_embeddings(path: Path) -> tuple[dict[int, int], np.ndarray]:
    packed = np.load(path)
    article_ids = packed["article_ids"].astype(np.int64)
    emb = packed["embeddings"].astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-8)
    article_to_idx = {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}
    return article_to_idx, emb.astype(np.float32)


def load_cached_samples(path: Path) -> tuple[list[Sample], dict]:
    with path.open("rb") as f:
        packed = pickle.load(f)
    return packed["samples"], packed.get("prepare_summary", {})


class FuturePriorTokenDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        article_to_idx: dict[int, int],
        emb: np.ndarray,
        max_clicked: int,
        max_not_clicked: int,
        max_current: int,
    ) -> None:
        self.samples = samples
        self.article_to_idx = article_to_idx
        self.emb = emb
        self.max_clicked = max_clicked
        self.max_not_clicked = max_not_clicked
        self.max_current = max_current
        self.max_tokens = 2 + max_clicked + max_not_clicked + max_current
        self.emb_dim = emb.shape[1]

    def __len__(self) -> int:
        return len(self.samples)

    def _fill_articles(
        self,
        tokens: np.ndarray,
        types: np.ndarray,
        mask: np.ndarray,
        pos: int,
        article_ids: list[int],
        limit: int,
        type_id: int,
    ) -> int:
        kept = [aid for aid in article_ids[-limit:] if aid in self.article_to_idx]
        for aid in kept:
            tokens[pos] = self.emb[self.article_to_idx[aid]]
            types[pos] = type_id
            mask[pos] = False
            pos += 1
        return pos

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        tokens = np.zeros((self.max_tokens, self.emb_dim), dtype=np.float32)
        types = np.zeros(self.max_tokens, dtype=np.int64)
        mask = np.ones(self.max_tokens, dtype=np.bool_)
        tokens[0] = sample.z_current.astype(np.float32)
        tokens[1] = sample.z_current_set.astype(np.float32)
        types[0] = 1
        types[1] = 2
        mask[0] = False
        mask[1] = False
        pos = 2
        pos = self._fill_articles(tokens, types, mask, pos, sample.past_clicked, self.max_clicked, 3)
        pos = self._fill_articles(tokens, types, mask, pos, sample.past_not_clicked, self.max_not_clicked, 4)
        self._fill_articles(tokens, types, mask, pos, sample.current_inview, self.max_current, 5)
        nums = np.array(
            [
                math.log1p(len(sample.past_shown)),
                math.log1p(len(sample.past_clicked)),
                math.log1p(len(sample.past_not_clicked)),
                math.log1p(len(sample.current_inview)),
                math.log1p(len(sample.future_shown)),
                float(sample.time.hour) / 23.0,
            ],
            dtype=np.float32,
        )
        return {
            "tokens": torch.from_numpy(tokens),
            "types": torch.from_numpy(types),
            "padding_mask": torch.from_numpy(mask),
            "nums": torch.from_numpy(nums),
            "target": torch.from_numpy(sample.z_future_target.astype(np.float32)),
        }


class FuturePriorTransformer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        num_features: int = 6,
    ) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.input_proj = nn.Linear(emb_dim, hidden_dim)
        self.type_emb = nn.Embedding(8, hidden_dim)
        self.num_proj = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        types: torch.Tensor,
        padding_mask: torch.Tensor,
        nums: torch.Tensor,
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        x = self.input_proj(tokens) + self.type_emb(types)
        cls = self.cls.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.zeros((batch, 1), dtype=torch.bool, device=padding_mask.device)
        mask = torch.cat([cls_mask, padding_mask], dim=1)
        h = self.encoder(x, src_key_padding_mask=mask)[:, 0]
        n = self.num_proj(nums)
        return self.head(torch.cat([h, n], dim=-1))


def split_samples(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    train = [s for s in samples if s.split == "jepa_train"]
    val = [s for s in samples if s.split == "jepa_val"]
    if len(train) < 10 or len(val) < 10:
        raise RuntimeError("Not enough JEPA train/val samples in cached sample file.")
    return train, val


def make_loader(
    samples: list[Sample],
    article_to_idx: dict[int, int],
    emb: np.ndarray,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = FuturePriorTokenDataset(
        samples,
        article_to_idx,
        emb,
        max_clicked=args.max_clicked_tokens,
        max_not_clicked=args.max_not_clicked_tokens,
        max_current=args.max_current_tokens,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int = 0,
) -> dict:
    train = optimizer is not None
    model.train(train)
    losses = []
    mses = []
    cosines = []
    with torch.set_grad_enabled(train):
        for batch_idx, batch in enumerate(loader):
            tokens = batch["tokens"].to(device, non_blocking=True)
            types = batch["types"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            nums = batch["nums"].to(device, non_blocking=True)
            target = F.normalize(batch["target"].to(device, non_blocking=True), dim=-1)
            pred = F.normalize(model(tokens, types, padding_mask, nums), dim=-1)
            mse = F.mse_loss(pred, target)
            cos = F.cosine_similarity(pred, target, dim=-1).mean()
            loss = mse + 0.5 * (1.0 - cos)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mses.append(float(mse.detach().cpu()))
            cosines.append(float(cos.detach().cpu()))
            if max_batches and batch_idx + 1 >= max_batches:
                break
    return {
        "loss": float(np.mean(losses)),
        "mse": float(np.mean(mses)),
        "cosine": float(np.mean(cosines)),
    }


def predict_samples(
    model: nn.Module,
    samples: list[Sample],
    article_to_idx: dict[int, int],
    emb: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    loader = make_loader(samples, article_to_idx, emb, args, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["tokens"].to(device, non_blocking=True),
                batch["types"].to(device, non_blocking=True),
                batch["padding_mask"].to(device, non_blocking=True),
                batch["nums"].to(device, non_blocking=True),
            )
            out = F.normalize(out, dim=-1).cpu().numpy().astype(np.float32)
            preds.append(out)
    return np.vstack(preds).astype(np.float32)


def fit_transformer_prior(samples: list[Sample], article_to_idx: dict[int, int], emb: np.ndarray, args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_samples, val_samples = split_samples(samples)
    train_loader = make_loader(train_samples, article_to_idx, emb, args, shuffle=True)
    val_loader = make_loader(val_samples, article_to_idx, emb, args, shuffle=False)
    model = FuturePriorTransformer(
        emb_dim=emb.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = -1e9
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.max_train_batches)
        val_metrics = run_epoch(model, val_loader, None, device, args.max_val_batches)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["cosine"] > best_val:
            best_val = val_metrics["cosine"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = predict_samples(model, samples, article_to_idx, emb, args, device)
    for sample, z_prior in zip(samples, pred):
        sample.z_prior = unit(z_prior)
        sample.delta_z = (sample.z_prior - sample.z_current).astype(np.float32)
    val = [s for s in samples if s.split == "jepa_val"]
    metrics = {
        "predictor": "transformer_token_prior",
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "history": history,
        "best_val_cosine_loader": float(best_val),
    }
    if val:
        metrics["val_mse"] = float(np.mean([(s.z_prior - s.z_future_target) @ (s.z_prior - s.z_future_target) / len(s.z_prior) for s in val]))
        metrics["val_cosine"] = float(np.mean([cosine_np(s.z_prior, s.z_future_target) for s in val]))
    return {"model": model, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a token Transformer JEPA future prior on cached EB-NeRD samples.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--article-embeddings", required=True)
    parser.add_argument("--baseline-results", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-clicked-tokens", type=int, default=24)
    parser.add_argument("--max-not-clicked-tokens", type=int, default=24)
    parser.add_argument("--max-current-tokens", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
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
    samples, prepare_summary = load_cached_samples(Path(args.samples_pkl))
    article_to_idx, emb = load_article_embeddings(Path(args.article_embeddings))
    articles = pd.read_parquet(Path(args.data_dir) / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    articles_by_id = article_lookup(articles)
    jepa = fit_transformer_prior(samples, article_to_idx, emb, args)
    results = {
        "prepare": prepare_summary,
        "transformer_jepa": jepa["metrics"],
        "standalone_jepa": standalone_jepa_eval(samples, article_to_idx, emb),
        "ranker_setting": {
            "task": "impression-level candidate reranking / pointwise CTR scoring within each exposed candidate set",
            "ranker": args.ranker,
            "input_rule": "current candidate set has no clicked/not-clicked feedback; current feedback is label only",
            "future_target_rule": "future target uses the same exposure-aware UIH schema as past UIH",
            "models": {
                "T1_future_prior": "B1 + Transformer-predicted future UIH latent + delta_z",
                "T2_future_item_cross": "T1 + item x Transformer future-prior and item x delta_z cross features",
            },
        },
        "models": {},
        "example": first_example(samples, articles_by_id),
    }
    if args.baseline_results:
        baseline = json.loads(Path(args.baseline_results).read_text())
        results["baseline_models_from"] = args.baseline_results
        results["baseline_models"] = {
            k: baseline["models"][k]
            for k in ["B0_meta", "B1_text_history"]
            if k in baseline.get("models", {})
        }
    for variant in ["T1_future_prior", "T2_future_item_cross"]:
        print(f"training {variant} with transformer prior", flush=True)
        results["models"][variant] = train_variant(variant, samples, articles_by_id, article_to_idx, emb, args.seed, args)
        print(json.dumps({variant: results["models"][variant]["ranker_test"]}, indent=2), flush=True)
    counts = Counter(s.split for s in samples)
    results["split_counts_after_load"] = dict(counts)
    write_json(out_dir / "results.json", results)
    torch.save(jepa["model"].state_dict(), out_dir / "transformer_prior.pt")
    write_json(out_dir / "uih_example.json", results["example"])
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
