#!/usr/bin/env python3
"""Mechanistic analysis for dynamic-K latent smoothing.

This script is intended to run from the TTJepa repository on the experiment
machine. It reuses the episode-aligned rows produced by k_refinement_analysis.py
and computes two diagnostics:

1. Latent spectrum / effective rank across recurrent depths.
2. Linear state-probe quality across recurrent depths.

It does not run policy evaluation. It only reloads the checkpoint, recomputes
predicted latents for the already-analyzed evaluation windows, and joins them
with privileged state labels from the HDF5 datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LEWM_ROOT = os.environ.get("LEWM_ROOT", "/home/robotuser/zijian/le-wm")
if LEWM_ROOT not in sys.path:
    sys.path.insert(1, LEWM_ROOT)

from utils import get_column_normalizer, get_img_preprocessor


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(h5_path: Path, *, frameskip: int, num_steps: int, img_size: int):
    dataset = swm.data.HDF5Dataset(
        path=h5_path,
        frameskip=frameskip,
        num_steps=num_steps,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=img_size),
        get_column_normalizer(dataset, "action", "action"),
    )
    return dataset


def stack_batch(items):
    keys = items[0].keys()
    batch = {}
    for key in keys:
        values = [item[key] for item in items]
        if torch.is_tensor(values[0]):
            batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = torch.as_tensor(np.stack(values, axis=0))
    return batch


def sorted_block_ids(keys: set[str]) -> list[int]:
    block_ids = []
    for key in keys:
        match = re.match(r"privileged_block_(\d+)_pos$", key)
        if match:
            block_ids.append(int(match.group(1)))
    return sorted(block_ids)


def quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    return float(min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2)))


def state_label_specs(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        keys = set(f.keys())
        block_ids = sorted_block_ids(keys)
        specs = {
            "has_qpos": "qpos" in keys,
            "has_observation": "observation" in keys,
            "block_ids": block_ids,
            "has_goal_blocks": all(
                f"goal_privileged_block_{block_id}_pos" in keys
                for block_id in block_ids
            ),
        }
    return specs


def collect_state_labels(
    h5_path: Path,
    global_indices: np.ndarray,
    *,
    frameskip: int,
    horizon_steps: int,
) -> dict[str, np.ndarray]:
    """Collect future state labels aligned with predicted latent steps."""

    specs = state_label_specs(h5_path)
    labels: dict[str, list[np.ndarray]] = defaultdict(list)

    with h5py.File(h5_path, "r") as f:
        max_index = len(next(iter(f.values()))) - 1
        for global_idx in global_indices:
            for step in range(1, horizon_steps + 1):
                idx = int(min(global_idx + frameskip * step, max_index))

                if specs["has_qpos"]:
                    labels["qpos"].append(np.asarray(f["qpos"][idx], dtype=np.float64).ravel())

                if specs["has_observation"]:
                    obs = np.asarray(f["observation"][idx], dtype=np.float64).ravel()
                    if obs.size <= 256:
                        labels["observation"].append(obs)

                block_pos = []
                block_quat = []
                goal_rel = []
                for block_id in specs["block_ids"]:
                    pos_key = f"privileged_block_{block_id}_pos"
                    quat_key = f"privileged_block_{block_id}_quat"
                    goal_key = f"goal_privileged_block_{block_id}_pos"
                    if pos_key in f:
                        pos = np.asarray(f[pos_key][idx], dtype=np.float64).ravel()
                        block_pos.append(pos)
                        if specs["has_goal_blocks"] and goal_key in f:
                            goal = np.asarray(f[goal_key][idx], dtype=np.float64).ravel()
                            goal_rel.append(goal - pos)
                    if quat_key in f:
                        block_quat.append(np.asarray(f[quat_key][idx], dtype=np.float64).ravel())

                if block_pos:
                    positions = np.stack(block_pos, axis=0)
                    labels["block_pos"].append(positions.ravel())
                    if len(block_pos) >= 2:
                        pairwise = []
                        for i in range(len(block_pos)):
                            for j in range(i + 1, len(block_pos)):
                                pairwise.append(np.linalg.norm(block_pos[i] - block_pos[j]))
                        labels["block_pairwise_dist"].append(np.asarray(pairwise, dtype=np.float64))
                if block_quat:
                    labels["block_quat"].append(np.concatenate(block_quat))
                if goal_rel:
                    labels["goal_relative_block_pos"].append(np.concatenate(goal_rel))

    return {key: np.stack(value, axis=0) for key, value in labels.items() if value}


def compute_pred_latents(
    *,
    model,
    dataset,
    rows: list[dict],
    batch_size: int,
    history_size: int,
    max_depth: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes = np.array([int(row["episode"]) for row in rows], dtype=np.int64)
    starts = np.array([int(row["start_step"]) for row in rows], dtype=np.int64)
    global_indices = np.array([int(row["global_idx"]) for row in rows], dtype=np.int64)

    all_preds = []
    model.eval()
    for lo in range(0, len(rows), batch_size):
        hi = min(lo + batch_size, len(rows))
        chunk = dataset.load_chunk(
            episodes[lo:hi],
            starts[lo:hi],
            starts[lo:hi] + dataset.span,
        )
        batch = stack_batch(chunk)
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)

        with torch.inference_mode():
            output = model.encode(batch)
            emb = output["emb"]
            act_emb = output["act_emb"]
            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            pred = model.predict(
                ctx_emb,
                ctx_act,
                return_all=True,
                predictor_depth=max_depth,
            )
            preds = pred["preds"].detach().float().cpu().numpy()
            # Expected [K, B, T, D]. Flatten any latent dimensions after T.
            preds = preds.reshape(preds.shape[0], preds.shape[1], preds.shape[2], -1)
            all_preds.append(preds)

    pred_all = np.concatenate(all_preds, axis=1)
    return pred_all, global_indices, episodes


def effective_rank_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x).all(axis=1)
    x = x[finite]
    if x.shape[0] < 2:
        return {
            "n": int(x.shape[0]),
            "dim": int(x.shape[1]) if x.ndim == 2 else 0,
            "total_variance": math.nan,
            "mean_feature_std": math.nan,
            "top1_var_frac": math.nan,
            "top5_var_frac": math.nan,
            "participation_rank": math.nan,
            "entropy_rank": math.nan,
        }
    x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    eig = (singular**2) / max(x.shape[0] - 1, 1)
    total = float(eig.sum())
    if total <= 1e-20:
        probs = np.ones_like(eig) / max(len(eig), 1)
    else:
        probs = eig / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-20)))
    return {
        "n": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "total_variance": total,
        "mean_feature_std": float(np.std(x, axis=0).mean()),
        "top1_var_frac": float(probs[0]) if len(probs) else math.nan,
        "top5_var_frac": float(probs[:5].sum()) if len(probs) else math.nan,
        "participation_rank": float((eig.sum() ** 2) / np.sum(eig**2 + 1e-20)),
        "entropy_rank": float(np.exp(entropy)),
    }


def spectrum_by_category(
    pred_all: np.ndarray,
    rows: list[dict],
) -> list[dict]:
    categories = np.array([row["category"] for row in rows])
    out = []
    max_depth, _, horizon_steps, _ = pred_all.shape
    for depth_idx in range(max_depth):
        depth = depth_idx + 1
        x_depth = pred_all[depth_idx].reshape(len(rows), horizon_steps, -1)
        for category in ["all", *sorted(set(categories.tolist()))]:
            if category == "all":
                mask = np.ones(len(rows), dtype=bool)
            else:
                mask = categories == category
            x = x_depth[mask].reshape(-1, x_depth.shape[-1])
            stats = effective_rank_stats(x)
            out.append({"depth": depth, "category": category, **stats})
    return out


def standardize_train_test(x_train, x_test):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def ridge_fit_predict(x_train, y_train, x_test, alpha: float):
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_centered = y_train - y_mean
    xtx = x_train.T @ x_train
    reg = alpha * np.eye(xtx.shape[0], dtype=np.float64)
    xty = x_train.T @ y_centered
    try:
        weight = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        weight = np.linalg.pinv(xtx + reg) @ xty
    return x_test @ weight + y_mean


def probe_metrics(
    pred_all: np.ndarray,
    labels: dict[str, np.ndarray],
    rows: list[dict],
    *,
    seed: int,
    train_frac: float,
    alpha: float,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    n_rows = len(rows)
    max_depth, _, horizon_steps, latent_dim = pred_all.shape
    flat_categories = np.repeat([row["category"] for row in rows], horizon_steps)

    probe_rows = []
    category_rows = []
    n_flat = n_rows * horizon_steps
    order = rng.permutation(n_flat)
    n_train = max(4, min(n_flat - 1, int(round(train_frac * n_flat))))
    train_idx = order[:n_train]
    test_idx = order[n_train:]

    for label_name, y in labels.items():
        if y.shape[0] != n_flat:
            continue
        finite_y = np.isfinite(y).all(axis=1)
        for depth_idx in range(max_depth):
            depth = depth_idx + 1
            x = pred_all[depth_idx].reshape(n_flat, latent_dim)
            finite = finite_y & np.isfinite(x).all(axis=1)
            tr = train_idx[finite[train_idx]]
            te = test_idx[finite[test_idx]]
            if len(tr) < 4 or len(te) < 2:
                continue
            x_train, x_test = standardize_train_test(x[tr], x[te])
            y_train = y[tr]
            y_test = y[te]
            pred = ridge_fit_predict(x_train, y_train, x_test, alpha)
            err = pred - y_test
            mse = float(np.mean(err**2))
            y_var = float(np.mean((y_test - y_test.mean(axis=0, keepdims=True)) ** 2))
            r2 = float(1.0 - mse / max(y_var, 1e-12))
            probe_rows.append(
                {
                    "label": label_name,
                    "depth": depth,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "target_dim": int(y.shape[1]),
                    "mse": mse,
                    "target_variance": y_var,
                    "r2": r2,
                    "normalized_mse": float(mse / max(y_var, 1e-12)),
                }
            )

            per_sample_mse = np.mean(err**2, axis=1)
            for category in sorted(set(flat_categories[te].tolist())):
                mask = flat_categories[te] == category
                if not np.any(mask):
                    continue
                category_rows.append(
                    {
                        "label": label_name,
                        "depth": depth,
                        "category": category,
                        "n_test": int(mask.sum()),
                        "mse": float(per_sample_mse[mask].mean()),
                    }
                )

    return probe_rows, category_rows


def summarize_findings(dataset: str, spectrum_rows, probe_rows, category_probe_rows) -> str:
    lines = [f"# Smoothing Analysis: {dataset}", ""]

    all_spectrum = [row for row in spectrum_rows if row["category"] == "all"]
    by_depth = {int(row["depth"]): row for row in all_spectrum}
    if 1 in by_depth and 4 in by_depth:
        k1 = by_depth[1]
        k4 = by_depth[4]
        er_ratio = k4["entropy_rank"] / max(k1["entropy_rank"], 1e-12)
        var_ratio = k4["total_variance"] / max(k1["total_variance"], 1e-12)
        top1_delta = k4["top1_var_frac"] - k1["top1_var_frac"]
        lines.append(
            f"- Spectrum all episodes: K4/K1 entropy-rank ratio = {er_ratio:.3f}, "
            f"variance ratio = {var_ratio:.3f}, top1 variance fraction delta = {top1_delta:+.3f}."
        )

    for label in sorted({row["label"] for row in probe_rows}):
        label_rows = {int(row["depth"]): row for row in probe_rows if row["label"] == label}
        if 1 in label_rows and 4 in label_rows:
            delta = label_rows[4]["r2"] - label_rows[1]["r2"]
            lines.append(
                f"- Probe {label}: R2 K1={label_rows[1]['r2']:.3f}, "
                f"K4={label_rows[4]['r2']:.3f}, delta={delta:+.3f}."
            )

    if category_probe_rows:
        lines.append("")
        lines.append("Category probe MSE is written to category_probe_mse.csv for helped/hurt comparisons.")
    lines.append("")
    return "\n".join(lines)


def default_specs(root: Path) -> dict[str, dict]:
    data = Path(os.environ.get("STABLEWM_HOME", "/vepfs/zijian/lewm_data"))
    analysis = root / "analysis" / "k_refinement_all_20260620_024634"
    return {
        "reacher": {
            "name": "reacher",
            "rows": analysis / "reacher_dynamic_oracle_rows.csv",
            "h5": data / "reacher.h5",
            "policy": "ttjepa_reacher_dynamic_oracle_k4_10e/weights_epoch_10.pt",
        },
        "cube_single": {
            "name": "cube_single",
            "rows": analysis / "cube_single_dynamic_oracle_rows.csv",
            "h5": data / "datasets" / "ogbench" / "cube_single_expert.h5",
            "policy": "ttjepa_cube_dynamic_oracle_k4_10e/weights_epoch_10.pt",
        },
        "cube_double": {
            "name": "cube_double",
            "rows": analysis / "cube_double_dynamic_oracle_rows.csv",
            "h5": data / "datasets" / "ogbench" / "visual_cube_double_play.h5",
            "policy": "ttjepa_cube_double_dynamic_oracle_k4_10e/weights_epoch_10.pt",
        },
        "cube_triple": {
            "name": "cube_triple",
            "rows": analysis / "cube_triple_dynamic_oracle_rows.csv",
            "h5": data / "datasets" / "ogbench" / "visual_cube_triple_play.h5",
            "policy": "ttjepa_cube_triple_dynamic_oracle_k4_10e/weights_epoch_10.pt",
        },
    }


def resolve_existing_path(path: Path) -> Path:
    if path.exists():
        return path
    # Compatibility for the copied cube-single dataset layout.
    alt = Path(str(path).replace("/datasets/ogbench/cube_single_expert.h5", "/datasets/ogbench--cube_single_expert.h5"))
    if alt.exists():
        return alt
    raise FileNotFoundError(path)


def run_one(spec: dict, args, output_dir: Path) -> dict:
    rows = read_csv_rows(resolve_existing_path(Path(spec["rows"])))
    if args.max_rows and len(rows) > args.max_rows:
        rows = rows[: args.max_rows]

    h5_path = resolve_existing_path(Path(spec["h5"]))
    dataset = make_dataset(
        h5_path,
        frameskip=args.frameskip,
        num_steps=args.history_size + 1,
        img_size=args.img_size,
    )
    model = swm.wm.utils.load_pretrained(spec["policy"]).to(args.device)
    model.eval()

    pred_all, global_indices, _ = compute_pred_latents(
        model=model,
        dataset=dataset,
        rows=rows,
        batch_size=args.batch_size,
        history_size=args.history_size,
        max_depth=args.max_depth,
        device=args.device,
    )
    labels = collect_state_labels(
        h5_path,
        global_indices,
        frameskip=args.frameskip,
        horizon_steps=pred_all.shape[2],
    )

    spectrum_rows = spectrum_by_category(pred_all, rows)
    probe_rows, category_probe_rows = probe_metrics(
        pred_all,
        labels,
        rows,
        seed=args.seed,
        train_frac=args.train_frac,
        alpha=args.ridge_alpha,
    )

    dataset_dir = output_dir / spec["name"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_csv(dataset_dir / "spectrum_by_depth.csv", spectrum_rows)
    write_csv(dataset_dir / "probe_by_depth.csv", probe_rows)
    write_csv(dataset_dir / "category_probe_mse.csv", category_probe_rows)
    (dataset_dir / "spectrum_by_depth.json").write_text(json.dumps(spectrum_rows, indent=2, sort_keys=True))
    (dataset_dir / "probe_by_depth.json").write_text(json.dumps(probe_rows, indent=2, sort_keys=True))
    (dataset_dir / "category_probe_mse.json").write_text(json.dumps(category_probe_rows, indent=2, sort_keys=True))
    findings = summarize_findings(spec["name"], spectrum_rows, probe_rows, category_probe_rows)
    (dataset_dir / "findings.md").write_text(findings)

    return {
        "name": spec["name"],
        "rows": len(rows),
        "h5": str(h5_path),
        "policy": spec["policy"],
        "labels": sorted(labels.keys()),
        "output_dir": str(dataset_dir),
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=["reacher", "cube_single", "cube_double", "cube_triple"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=1e-2)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("STABLEWM_HOME", "/vepfs/zijian/lewm_data")
    os.environ.setdefault("LOCAL_DATASET_DIR", "/vepfs/zijian/lewm_data")

    output_dir = Path(args.output_dir or f"analysis/k_smoothing_{os.environ.get('RUN_TAG', 'latest')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = default_specs(REPO_ROOT)
    summaries = []
    for key in args.datasets:
        if key not in specs:
            raise KeyError(f"Unknown dataset {key}. Choices: {sorted(specs)}")
        print(f"RUN {key}", flush=True)
        summary = run_one(specs[key], args, output_dir)
        summaries.append(summary)
        print(summary["findings"], flush=True)

    (output_dir / "combined_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True))
    (output_dir / "findings.md").write_text("\n".join(summary["findings"] for summary in summaries))
    print(f"WROTE {output_dir}", flush=True)


if __name__ == "__main__":
    main()
