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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cached(path: Path) -> tuple[list[Sample], dict[str, Any]]:
    with path.open("rb") as f:
        packed = pickle.load(f)
    return packed["samples"], packed.get("prepare_summary", {})


def load_article_embeddings(path: Path) -> tuple[dict[int, int], np.ndarray]:
    packed = np.load(path)
    article_ids = packed["article_ids"].astype(np.int64)
    dense = normalize(packed["embeddings"].astype(np.float32)).astype(np.float32)
    return {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}, dense


def make_current_response_anchor(samples: list[Sample], article_to_idx: dict[int, int], emb: np.ndarray) -> np.ndarray:
    out = []
    for sample in samples:
        clicked = list(sample.current_clicked)
        clicked_set = set(clicked)
        not_clicked = [aid for aid in sample.current_inview if aid not in clicked_set]
        shown = mean_embedding(sample.current_inview, article_to_idx, emb)
        pos = mean_embedding(clicked, article_to_idx, emb)
        neg = mean_embedding(not_clicked, article_to_idx, emb)
        out.append(unit(pos + 0.15 * shown - 0.35 * neg))
    return np.vstack(out).astype(np.float32)


def make_post_current_uih(
    samples: list[Sample],
    article_to_idx: dict[int, int],
    emb: np.ndarray,
    current_response_weight: float,
    future_weight: float,
) -> np.ndarray:
    current = make_current_response_anchor(samples, article_to_idx, emb)
    future = np.vstack([s.z_future_target for s in samples]).astype(np.float32)
    return np.vstack([unit(current_response_weight * c + future_weight * f) for c, f in zip(current, future)]).astype(np.float32)


def article_dense_meta(articles_by_id: dict[int, dict], aid: int, sample: Sample, n: int) -> list[float]:
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
        math.log1p(n),
        math.log1p(len(sample.past_shown)),
        math.log1p(len(sample.past_clicked)),
        math.log1p(len(sample.past_not_clicked)),
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


class SetBatchBuilder:
    scalar_dim = 25

    def __init__(
        self,
        article_to_idx: dict[int, int],
        emb: np.ndarray,
        z_post: np.ndarray,
        articles_by_id: dict[int, dict],
        user_to_idx: dict[int, int],
        item_to_idx: dict[int, int],
    ) -> None:
        self.article_to_idx = article_to_idx
        self.emb = emb
        self.z_post = z_post
        self.articles_by_id = articles_by_id
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx

    def __call__(self, samples: list[Sample]) -> dict[str, torch.Tensor | np.ndarray]:
        bsz = len(samples)
        max_c = max(len(s.current_inview) for s in samples)
        d = self.emb.shape[1]
        item_emb = np.zeros((bsz, max_c, d), dtype=np.float32)
        scalar = np.zeros((bsz, max_c, self.scalar_dim), dtype=np.float32)
        labels = np.zeros((bsz, max_c), dtype=np.float32)
        mask = np.zeros((bsz, max_c), dtype=bool)
        item_idx = np.zeros((bsz, max_c), dtype=np.int64)
        user_idx = np.zeros((bsz,), dtype=np.int64)
        past = np.zeros((bsz, d), dtype=np.float32)
        candidate_set = np.zeros((bsz, d), dtype=np.float32)
        post = np.zeros((bsz, d), dtype=np.float32)
        groups = np.zeros((bsz, max_c), dtype=np.int64)
        for bi, sample in enumerate(samples):
            z_past = sample.z_current.astype(np.float32)
            z_set = sample.z_current_set.astype(np.float32)
            z_post = self.z_post[sample.sample_id].astype(np.float32)
            z_delta = (z_post - z_past).astype(np.float32)
            past[bi] = z_past
            candidate_set[bi] = z_set
            post[bi] = z_post
            user_idx[bi] = self.user_to_idx.get(sample.user_id, 0)
            clicked = set(sample.current_clicked)
            n = len(sample.current_inview)
            for ci, aid in enumerate(sample.current_inview):
                idx = self.article_to_idx.get(aid)
                item = self.emb[idx].astype(np.float32) if idx is not None else np.zeros(d, dtype=np.float32)
                item_emb[bi, ci] = item
                scalar[bi, ci] = np.array(
                    [
                        cosine(item, z_past),
                        cosine(item, z_set),
                        cosine(item, z_post),
                        cosine(item, z_delta),
                    ]
                    + article_dense_meta(self.articles_by_id, aid, sample, n),
                    dtype=np.float32,
                )
                labels[bi, ci] = float(aid in clicked)
                mask[bi, ci] = True
                item_idx[bi, ci] = self.item_to_idx.get(aid, 0)
                groups[bi, ci] = sample.impression_id
        return {
            "item_emb": torch.from_numpy(item_emb),
            "scalar": torch.from_numpy(scalar),
            "labels": torch.from_numpy(labels),
            "mask": torch.from_numpy(mask),
            "item_idx": torch.from_numpy(item_idx),
            "user_idx": torch.from_numpy(user_idx),
            "past": torch.from_numpy(past),
            "candidate_set": torch.from_numpy(candidate_set),
            "post": torch.from_numpy(post),
            "group_ids": groups,
        }


class SetTransformerRanker(nn.Module):
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
        use_post_token: bool,
        init_post_dot_scale: float,
    ) -> None:
        super().__init__()
        self.use_post_token = use_post_token
        self.post_dot_scale = nn.Parameter(torch.tensor(float(init_post_dot_scale), dtype=torch.float32))
        self.vec_proj = nn.Linear(emb_dim, d_model)
        self.scalar_proj = nn.Linear(scalar_dim, d_model)
        self.user_emb = nn.Embedding(num_users + 1, d_model, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(5, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(scalar_dim),
            nn.Linear(scalar_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.scalar_head[-1].weight)
        nn.init.zeros_(self.scalar_head[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_tok = self.user_emb(batch["user_idx"]) + self.type_emb.weight[0]
        past_tok = self.vec_proj(batch["past"]) + self.type_emb.weight[1]
        set_tok = self.vec_proj(batch["candidate_set"]) + self.type_emb.weight[2]
        tokens = [user_tok.unsqueeze(1), past_tok.unsqueeze(1), set_tok.unsqueeze(1)]
        if self.use_post_token:
            post_tok = self.vec_proj(batch["post"]) + self.type_emb.weight[3]
            tokens.append(post_tok.unsqueeze(1))
        item_tok = self.vec_proj(batch["item_emb"]) + self.scalar_proj(batch["scalar"]) + self.item_id_emb(batch["item_idx"]) + self.type_emb.weight[4]
        tokens.append(item_tok)
        x = torch.cat(tokens, dim=1)
        ctx_len = 4 if self.use_post_token else 3
        ctx_mask = torch.zeros((x.shape[0], ctx_len), dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([ctx_mask, ~batch["mask"]], dim=1)
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        item_out = out[:, ctx_len:]
        learned = self.head(item_out).squeeze(-1) + self.scalar_head(batch["scalar"]).squeeze(-1)
        # scalar[..., 2] is cosine(item, post_current_uih). This direct adapter makes
        # the intended item-post compatibility available without forcing attention
        # layers to rediscover a dot product from scratch.
        if self.use_post_token:
            learned = learned + self.post_dot_scale * batch["scalar"][..., 2]
        return learned


class SetJepaPredictor(nn.Module):
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
    ) -> None:
        super().__init__()
        self.vec_proj = nn.Linear(emb_dim, d_model)
        self.scalar_proj = nn.Linear(scalar_dim, d_model)
        self.user_emb = nn.Embedding(num_users + 1, d_model, padding_idx=0)
        self.item_id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(4, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_tok = self.user_emb(batch["user_idx"]) + self.type_emb.weight[0]
        past_tok = self.vec_proj(batch["past"]) + self.type_emb.weight[1]
        set_tok = self.vec_proj(batch["candidate_set"]) + self.type_emb.weight[2]
        item_tok = self.vec_proj(batch["item_emb"]) + self.scalar_proj(batch["scalar"]) + self.item_id_emb(batch["item_idx"]) + self.type_emb.weight[3]
        x = torch.cat([user_tok.unsqueeze(1), past_tok.unsqueeze(1), set_tok.unsqueeze(1), item_tok], dim=1)
        ctx_mask = torch.zeros((x.shape[0], 3), dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([ctx_mask, ~batch["mask"]], dim=1)
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        item_out = out[:, 3:]
        item_mask = batch["mask"].float().unsqueeze(-1)
        item_pool = (item_out * item_mask).sum(dim=1) / item_mask.sum(dim=1).clamp_min(1.0)
        pooled = torch.cat([out[:, 0], out[:, 1], item_pool], dim=-1)
        return self.out(pooled)


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = scores.masked_fill(~mask, -1e4)
    pos = labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    target = labels / pos
    return (-(target * F.log_softmax(scores, dim=1)).sum(dim=1)).mean()


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def inbatch_contrastive_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    pred_u = F.normalize(pred, dim=-1)
    target_u = F.normalize(target, dim=-1)
    logits = pred_u @ target_u.T / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def vicreg_regularizer(z: torch.Tensor, variance_gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    z = F.normalize(z, dim=-1)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(variance_gamma - std))
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    cov_loss = off_diagonal(cov).pow_(2).sum() / z.shape[1]
    return var_loss, cov_loss


def candidate_distribution_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    item_emb: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    item = F.normalize(item_emb, dim=-1)
    pred_u = F.normalize(pred, dim=-1)
    target_u = F.normalize(target, dim=-1)
    teacher = torch.einsum("bcd,bd->bc", item, target_u) / temperature
    student = torch.einsum("bcd,bd->bc", item, pred_u) / temperature
    teacher = teacher.masked_fill(~mask, -1e4)
    student = student.masked_fill(~mask, -1e4)
    teacher_prob = F.softmax(teacher, dim=1).detach()
    distill = -(teacher_prob * F.log_softmax(student, dim=1)).sum(dim=1).mean()
    label_rank = listwise_loss(student, labels, mask)
    return distill, label_rank


def jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    item_emb: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_u = F.normalize(pred, dim=-1)
    target_u = F.normalize(target, dim=-1)
    latent = F.mse_loss(pred_u, target_u) + (1.0 - (pred_u * target_u).sum(dim=-1).mean())
    distill, label_rank = candidate_distribution_loss(
        pred,
        target,
        item_emb,
        labels,
        mask,
        args.candidate_temperature,
    )
    nce = inbatch_contrastive_loss(pred, target.detach(), args.inbatch_temperature)
    var_loss, cov_loss = vicreg_regularizer(pred, args.variance_gamma)
    total = (
        args.latent_weight * latent
        + args.candidate_kl_weight * distill
        + args.candidate_label_weight * label_rank
        + args.inbatch_nce_weight * nce
        + args.variance_weight * var_loss
        + args.covariance_weight * cov_loss
    )
    parts = {
        "latent": float(latent.detach().cpu()),
        "candidate_kl": float(distill.detach().cpu()),
        "candidate_label": float(label_rank.detach().cpu()),
        "inbatch_nce": float(nce.detach().cpu()),
        "variance": float(var_loss.detach().cpu()),
        "covariance": float(cov_loss.detach().cpu()),
    }
    return total, parts


@torch.no_grad()
def predict_ranker(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    scores = []
    groups = []
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(moved).detach().cpu().numpy()
        lab = batch["labels"].numpy()
        mask = batch["mask"].numpy()
        group = batch["group_ids"]
        labels.append(lab[mask])
        scores.append(out[mask])
        groups.append(group[mask])
    return np.concatenate(labels), np.concatenate(scores), np.concatenate(groups)


@torch.no_grad()
def predict_latents(model: nn.Module, loader: DataLoader, device: torch.device, num_samples: int, emb_dim: int) -> np.ndarray:
    model.eval()
    pred = np.zeros((num_samples, emb_dim), dtype=np.float32)
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = F.normalize(model(moved), dim=-1).detach().cpu().numpy().astype(np.float32)
        sample_ids = batch["sample_ids"]
        pred[sample_ids] = out
    return pred


class JepaBatchBuilder(SetBatchBuilder):
    def __call__(self, samples: list[Sample]) -> dict[str, torch.Tensor | np.ndarray]:
        batch = super().__call__(samples)
        batch["target"] = batch["post"]
        batch["sample_ids"] = np.array([s.sample_id for s in samples], dtype=np.int64)
        return batch


def cosine_summary(name: str, pred: np.ndarray, target: np.ndarray, idxs: list[int]) -> dict[str, Any]:
    if not idxs:
        return {"name": name, "num_samples": 0}
    p = pred[idxs]
    t = target[idxs]
    pairwise_n = min(len(idxs), 5000)
    sample = p[:pairwise_n]
    pair_cos = sample @ sample.T
    tri = pair_cos[np.triu_indices(pairwise_n, k=1)] if pairwise_n > 1 else np.array([0.0])
    return {
        "name": name,
        "num_samples": int(len(idxs)),
        "cosine_to_post_current_uih": float(np.mean([cosine(a, b) for a, b in zip(p, t)])),
        "mse_to_post_current_uih": float(np.mean((p - t) ** 2)),
        "pred_mean_pairwise_cosine": float(np.mean(tri)),
    }


def split_candidate_dot_metrics(samples: list[Sample], split: str, article_to_idx: dict[int, int], emb: np.ndarray, vectors: np.ndarray) -> dict[str, Any]:
    return candidate_dot_metrics(samples, split, article_to_idx, emb, vectors)


def train_jepa_predictor(
    samples: list[Sample],
    article_to_idx: dict[int, int],
    emb: np.ndarray,
    post: np.ndarray,
    articles_by_id: dict[int, dict],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = torch.device(args.device)
    builder = JepaBatchBuilder(article_to_idx, emb, post, articles_by_id, user_to_idx, item_to_idx)
    loaders = {
        split: DataLoader(
            ImpressionDataset(samples, split),
            batch_size=args.batch_size,
            shuffle=(split == "jepa_train"),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=builder,
        )
        for split in ["jepa_train", "jepa_val", "ranker_train", "ranker_val", "ranker_test"]
    }
    model = SetJepaPredictor(
        emb_dim=emb.shape[1],
        scalar_dim=SetBatchBuilder.scalar_dim,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.jepa_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.jepa_lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = -float("inf")
    history = []
    patience = args.patience
    for epoch in range(1, args.jepa_epochs + 1):
        model.train()
        losses = []
        loss_parts: dict[str, list[float]] = {
            "latent": [],
            "candidate_kl": [],
            "candidate_label": [],
            "inbatch_nce": [],
            "variance": [],
            "covariance": [],
        }
        for batch in loaders["jepa_train"]:
            moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            pred = model(moved)
            loss, parts = jepa_loss(pred, moved["target"], moved["item_emb"], moved["labels"], moved["mask"], args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                loss_parts[key].append(value)
        val_pred = predict_latents(model, loaders["jepa_val"], device, len(samples), emb.shape[1])
        val_idxs = [s.sample_id for s in samples if s.split == "jepa_val"]
        val_cos = cosine_summary("jepa_val", val_pred, post, val_idxs)
        val_dot = split_candidate_dot_metrics(samples, "jepa_val", article_to_idx, emb, val_pred)
        metric = val_dot["ndcg@10"]
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "loss_parts": {k: float(np.mean(v)) for k, v in loss_parts.items() if v},
            "jepa_val": val_cos,
            "jepa_val_candidate_dot": val_dot,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "set_jepa": {
                        "epoch": epoch,
                        "loss": row["loss"],
                        "val_cos": val_cos["cosine_to_post_current_uih"],
                        "val_pairwise": val_cos["pred_mean_pairwise_cosine"],
                        "val_dot_mrr": val_dot["mrr"],
                        "val_dot_ndcg10": val_dot["ndcg@10"],
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if metric > best_val:
            best_val = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred_all = np.zeros((len(samples), emb.shape[1]), dtype=np.float32)
    for split, loader in loaders.items():
        split_pred = predict_latents(model, loader, device, len(samples), emb.shape[1])
        idxs = [s.sample_id for s in samples if s.split == split]
        pred_all[idxs] = split_pred[idxs]
    split_eval = {}
    for split in ["jepa_val", "ranker_train", "ranker_val", "ranker_test"]:
        idxs = [s.sample_id for s in samples if s.split == split]
        split_eval[split] = cosine_summary(split, pred_all, post, idxs)
        split_eval[split]["candidate_dot"] = split_candidate_dot_metrics(samples, split, article_to_idx, emb, pred_all)
    return pred_all, {"history": history, "split_eval": split_eval}


def train_ranker(
    model_name: str,
    samples: list[Sample],
    article_to_idx: dict[int, int],
    emb: np.ndarray,
    post: np.ndarray,
    articles_by_id: dict[int, dict],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    use_post_token: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device)
    builder = SetBatchBuilder(article_to_idx, emb, post, articles_by_id, user_to_idx, item_to_idx)
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
    model = SetTransformerRanker(
        emb_dim=emb.shape[1],
        scalar_dim=SetBatchBuilder.scalar_dim,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        use_post_token=use_post_token,
        init_post_dot_scale=args.init_post_dot_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y_val, s_val, g_val = predict_ranker(model, loaders["ranker_val"], device)
    initial_metrics = ranking_metrics(y_val, s_val, g_val)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_metric = initial_metrics["ndcg@10"]
    patience = args.patience
    history = [{"epoch": 0, "loss": None, "ranker_val": initial_metrics}]
    print(
        json.dumps(
            {
                model_name: {
                    "epoch": 0,
                    "loss": None,
                    "val_mrr": initial_metrics["mrr"],
                    "val_hit1": initial_metrics["hit1"],
                    "val_ndcg10": initial_metrics["ndcg@10"],
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["ranker_train"]:
            moved = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            scores = model(moved)
            loss = listwise_loss(scores, moved["labels"], moved["mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        y_val, s_val, g_val = predict_ranker(model, loaders["ranker_val"], device)
        val_metrics = ranking_metrics(y_val, s_val, g_val)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "ranker_val": val_metrics}
        history.append(row)
        print(
            json.dumps(
                {
                    model_name: {
                        "epoch": epoch,
                        "loss": row["loss"],
                        "val_mrr": val_metrics["mrr"],
                        "val_hit1": val_metrics["hit1"],
                        "val_ndcg10": val_metrics["ndcg@10"],
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        metric = val_metrics["ndcg@10"]
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
    out = {"history": history, "use_post_token": use_post_token}
    for split, loader in loaders.items():
        y, s, g = predict_ranker(model, loader, device)
        out[split] = ranking_metrics(y, s, g)
    return out


def candidate_dot_metrics(samples: list[Sample], split: str, article_to_idx: dict[int, int], emb: np.ndarray, post: np.ndarray) -> dict[str, Any]:
    labels = []
    scores = []
    groups = []
    for sample in samples:
        if sample.split != split:
            continue
        z = post[sample.sample_id]
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
        name: value["ranker_test"]
        for name, value in raw["models"].items()
        if isinstance(value, dict) and "ranker_test" in value
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Set-transformer post-current UIH oracle/student experiment for EB-NeRD.")
    parser.add_argument("--samples-pkl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--article-embeddings", required=True)
    parser.add_argument("--baseline-results", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--jepa-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
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
    parser.add_argument("--current-response-weight", type=float, default=1.0)
    parser.add_argument("--future-weight", type=float, default=1.0)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-predicted", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, prepare = load_cached(Path(args.samples_pkl))
    article_to_idx, emb = load_article_embeddings(Path(args.article_embeddings))
    articles = pd.read_parquet(Path(args.data_dir) / "articles.parquet").drop_duplicates("article_id").reset_index(drop=True)
    articles_by_id = article_lookup(articles)
    post = make_post_current_uih(samples, article_to_idx, emb, args.current_response_weight, args.future_weight)
    zeros = np.zeros_like(post, dtype=np.float32)
    users = sorted({s.user_id for s in samples if s.split.startswith("ranker_") or s.split.startswith("jepa_")})
    items = sorted({aid for s in samples for aid in s.current_inview})
    user_to_idx = {uid: idx + 1 for idx, uid in enumerate(users)}
    item_to_idx = {aid: idx + 1 for idx, aid in enumerate(items)}

    results: dict[str, Any] = {
        "prepare": prepare,
        "setting": {
            "task": "one impression is one sample; set/list transformer scores all candidates in the impression",
            "loss": "pure listwise softmax over candidates; no pointwise CTR/BCE loss",
            "input_tokens": "user token, past UIH token, current candidate-set token, optional post-current UIH token, unordered item tokens",
            "post_current_uih": "current impression response UIH plus later future UIH; oracle is not available at serving time",
            "current_response_weight": args.current_response_weight,
            "future_weight": args.future_weight,
        },
        "baseline_test": summarize_baseline(args.baseline_results),
        "split_counts": dict(Counter(s.split for s in samples)),
        "candidate_dot": {
            "oracle_post_current_uih": {
                split: candidate_dot_metrics(samples, split, article_to_idx, emb, post)
                for split in ["ranker_val", "ranker_test"]
            }
        },
        "models": {},
    }

    if not args.skip_baseline:
        print("training S0_set_transformer_no_post", flush=True)
        results["models"]["S0_set_transformer_no_post"] = train_ranker(
            "S0_set_transformer_no_post",
            samples,
            article_to_idx,
            emb,
            zeros,
            articles_by_id,
            user_to_idx,
            item_to_idx,
            False,
            args,
        )
        print(json.dumps({"S0_set_transformer_no_post": results["models"]["S0_set_transformer_no_post"]["ranker_test"]}, indent=2), flush=True)
        write_json(out_dir / "results.partial.json", results)

    if not args.skip_oracle:
        print("training O1_set_transformer_oracle_post", flush=True)
        results["models"]["O1_set_transformer_oracle_post"] = train_ranker(
            "O1_set_transformer_oracle_post",
            samples,
            article_to_idx,
            emb,
            post,
            articles_by_id,
            user_to_idx,
            item_to_idx,
            True,
            args,
        )
        print(json.dumps({"O1_set_transformer_oracle_post": results["models"]["O1_set_transformer_oracle_post"]["ranker_test"]}, indent=2), flush=True)
        write_json(out_dir / "results.partial.json", results)

    if not args.skip_predicted:
        print("training set JEPA post-current predictor", flush=True)
        pred_post, jepa_metrics = train_jepa_predictor(
            samples,
            article_to_idx,
            emb,
            post,
            articles_by_id,
            user_to_idx,
            item_to_idx,
            args,
        )
        results["set_jepa"] = jepa_metrics
        results["candidate_dot"]["predicted_post_current_uih"] = {
            split: candidate_dot_metrics(samples, split, article_to_idx, emb, pred_post)
            for split in ["ranker_val", "ranker_test"]
        }
        print("training T1_set_transformer_predicted_post", flush=True)
        results["models"]["T1_set_transformer_predicted_post"] = train_ranker(
            "T1_set_transformer_predicted_post",
            samples,
            article_to_idx,
            emb,
            pred_post,
            articles_by_id,
            user_to_idx,
            item_to_idx,
            True,
            args,
        )
        print(json.dumps({"T1_set_transformer_predicted_post": results["models"]["T1_set_transformer_predicted_post"]["ranker_test"]}, indent=2), flush=True)

    write_json(out_dir / "results.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
