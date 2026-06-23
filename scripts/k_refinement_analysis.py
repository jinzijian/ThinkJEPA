#!/usr/bin/env python3
"""Analyze whether latent refinement metrics identify K-needed eval cases.

This is an analysis-only script intended to run from the TTJepa repository on
the remote experiment machine. It aligns with eval.py's dataset-driven episode
sampling, reuses the trained checkpoint to compute per-depth latent prediction
errors on the corresponding dataset windows, and joins those metrics with
existing fixed-K evaluation success arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LEWM_ROOT = os.environ.get("LEWM_ROOT", "/home/robotuser/zijian/le-wm")
if LEWM_ROOT not in sys.path:
    sys.path.insert(1, LEWM_ROOT)

from utils import get_column_normalizer, get_img_preprocessor


SUCCESS_RE = re.compile(r"'episode_successes'\s*:\s*array\(\[(.*?)\]\)", re.S)


def parse_successes(path: Path) -> np.ndarray:
    text = path.read_text(errors="ignore")
    match = SUCCESS_RE.search(text)
    if not match:
        raise ValueError(f"Could not parse episode_successes from {path}")
    body = match.group(1)
    values = re.findall(r"\bTrue\b|\bFalse\b", body)
    if not values:
        raise ValueError(f"No boolean successes found in {path}")
    return np.array([v == "True" for v in values], dtype=bool)


def eval_global_indices(h5_path: Path, seed: int, num_eval: int, goal_offset: int):
    with h5py.File(h5_path, "r") as f:
        ep_col = "episode_idx" if "episode_idx" in f else "ep_idx"
        ep_idx_arr = f[ep_col][:]
        step_idx = f["step_idx"][:]
        ep_indices, first_indices = np.unique(ep_idx_arr, return_index=True)
        lengths = f["ep_len"][:]
        if len(lengths) == len(ep_idx_arr):
            ep_lengths = lengths[first_indices]
        elif int(np.max(ep_indices)) < len(lengths):
            ep_lengths = lengths[ep_indices]
        else:
            ep_lengths = lengths[: len(ep_indices)]
        max_start_idx = ep_lengths - int(goal_offset) - 1
        max_start = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
        max_start_per_row = np.array([max_start[ep_id] for ep_id in ep_idx_arr])
        valid_mask = step_idx <= max_start_per_row
        valid_indices = np.nonzero(valid_mask)[0]
        rng = np.random.default_rng(seed)
        chosen_positions = rng.choice(
            len(valid_indices) - 1, size=num_eval, replace=False
        )
        global_indices = np.sort(valid_indices[chosen_positions])
        episodes = ep_idx_arr[global_indices].astype(int)
        starts = step_idx[global_indices].astype(int)
        offsets = f["ep_offset"][:]
    return global_indices, episodes, starts, offsets


def make_dataset(h5_path: Path, *, frameskip: int, num_steps: int, img_size: int):
    dataset = swm.data.HDF5Dataset(
        path=h5_path,
        frameskip=frameskip,
        num_steps=num_steps,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=img_size),
        get_column_normalizer(dataset, "action", "action"),
    ]
    dataset.transform = spt.data.transforms.Compose(*transforms)
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


def compute_latent_metrics(
    *,
    model,
    dataset,
    episodes: np.ndarray,
    starts: np.ndarray,
    batch_size: int,
    history_size: int,
    max_depth: int,
    device: str,
):
    all_metrics = []
    model.eval()
    for lo in range(0, len(episodes), batch_size):
        hi = min(lo + batch_size, len(episodes))
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
            target = emb[:, 1 : history_size + 1]
            pred = model.predict(
                ctx_emb,
                ctx_act,
                return_all=True,
                predictor_depth=max_depth,
            )
            preds_all = pred["preds"]
            pred_err = (preds_all - target.unsqueeze(0)).pow(2).mean(dim=-1)
            pred_err_ep = pred_err.mean(dim=-1).transpose(0, 1).detach().cpu().numpy()
            residual_ep = (
                pred["residuals"].mean(dim=-1).transpose(0, 1).detach().cpu().numpy()
            )

        for row in range(hi - lo):
            err = pred_err_ep[row]
            residual = residual_ep[row]
            best = float(np.min(err))
            first_good = int(np.argmax(err <= best * 1.02) + 1)
            all_metrics.append(
                {
                    "latent_mse_k1": float(err[0]),
                    "latent_mse_k4": float(err[min(3, len(err) - 1)]),
                    "latent_mse_best": best,
                    "latent_mse_improvement_abs": float(err[0] - err[min(3, len(err) - 1)]),
                    "latent_mse_improvement_rel": float(
                        (err[0] - err[min(3, len(err) - 1)]) / max(err[0], 1e-12)
                    ),
                    "latent_oracle_depth_2pct": first_good,
                    "residual_mean_k1": float(residual[0]),
                    "residual_mean_k4": float(residual[min(3, len(residual) - 1)]),
                    "residual_max": float(np.max(residual)),
                }
            )
    return all_metrics


def quat_distance(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    return float(min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2)))


def state_metrics(h5_path: Path, global_indices, episodes, starts, offsets, goal_offset):
    metrics = []
    with h5py.File(h5_path, "r") as f:
        keys = set(f.keys())
        ep_col = "episode_idx" if "episode_idx" in f else "ep_idx"
        offset_is_per_row = len(offsets) == len(f[ep_col])
        block_ids = []
        for key in keys:
            m = re.match(r"privileged_block_(\d+)_pos$", key)
            if m:
                block_ids.append(int(m.group(1)))
        block_ids = sorted(block_ids)

        for global_idx, ep, start in zip(global_indices, episodes, starts):
            ep_offset = offsets[int(global_idx)] if offset_is_per_row else offsets[int(ep)]
            goal_idx = int(ep_offset + int(start) + int(goal_offset))
            row = {
                "global_idx": int(global_idx),
                "episode": int(ep),
                "start_step": int(start),
                "goal_idx": int(goal_idx),
            }
            pos_dists = []
            quat_dists = []
            start_positions = []
            goal_positions = []
            for block_id in block_ids:
                pos_key = f"privileged_block_{block_id}_pos"
                quat_key = f"privileged_block_{block_id}_quat"
                if pos_key in keys:
                    start_pos = np.asarray(f[pos_key][global_idx], dtype=np.float64)
                    goal_pos = np.asarray(f[pos_key][goal_idx], dtype=np.float64)
                    start_positions.append(start_pos)
                    goal_positions.append(goal_pos)
                    pos_dists.append(float(np.linalg.norm(goal_pos - start_pos)))
                if quat_key in keys:
                    quat_dists.append(quat_distance(f[quat_key][global_idx], f[quat_key][goal_idx]))

            if pos_dists:
                row["goal_pos_dist_mean"] = float(np.mean(pos_dists))
                row["goal_pos_dist_max"] = float(np.max(pos_dists))
                row["goal_pos_dist_sum"] = float(np.sum(pos_dists))
            if quat_dists:
                row["goal_quat_dist_mean"] = float(np.mean(quat_dists))
                row["goal_quat_dist_max"] = float(np.max(quat_dists))
            if len(start_positions) >= 2:
                pairwise = []
                for i in range(len(start_positions)):
                    for j in range(i + 1, len(start_positions)):
                        pairwise.append(float(np.linalg.norm(start_positions[i] - start_positions[j])))
                row["start_block_pairwise_min"] = float(np.min(pairwise))
                row["start_block_pairwise_mean"] = float(np.mean(pairwise))

            if "qpos" in keys:
                start_qpos = np.asarray(f["qpos"][global_idx], dtype=np.float64)
                goal_qpos = np.asarray(f["qpos"][goal_idx], dtype=np.float64)
                row["goal_qpos_dist"] = float(np.linalg.norm(goal_qpos - start_qpos))
            if "observation" in keys:
                start_obs = np.asarray(f["observation"][global_idx], dtype=np.float64)
                goal_obs = np.asarray(f["observation"][goal_idx], dtype=np.float64)
                row["goal_observation_dist"] = float(np.linalg.norm(goal_obs - start_obs))

            if "action" in keys:
                actions = np.asarray(f["action"][global_idx:goal_idx], dtype=np.float64)
                if actions.size:
                    action_norms = np.linalg.norm(actions.reshape(actions.shape[0], -1), axis=1)
                    row["action_norm_mean_to_goal"] = float(np.mean(action_norms))
                    row["action_norm_max_to_goal"] = float(np.max(action_norms))
            metrics.append(row)
    return metrics


def category(k1_success: bool, k4_success: bool):
    if k1_success and k4_success:
        return "easy_both_success"
    if (not k1_success) and k4_success:
        return "depth_helped_k1_fail_k4_success"
    if k1_success and (not k4_success):
        return "depth_hurt_k1_success_k4_fail"
    return "hard_both_fail"


def mean_or_none(values):
    values = [v for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return None
    return float(np.mean(values))


def summarize(rows, metrics):
    summary = {
        "num_rows": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "category_metric_means": {},
        "auc_depth_helped_vs_easy": {},
    }
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)
    for cat, cat_rows in by_cat.items():
        summary["category_metric_means"][cat] = {
            metric: mean_or_none([row.get(metric) for row in cat_rows])
            for metric in metrics
        }

    subset = [
        row
        for row in rows
        if row["category"]
        in {"easy_both_success", "depth_helped_k1_fail_k4_success"}
    ]
    if len({row["category"] for row in subset}) == 2:
        labels = np.array(
            [row["category"] == "depth_helped_k1_fail_k4_success" for row in subset],
            dtype=int,
        )
        for metric in metrics:
            values = np.array([row.get(metric, np.nan) for row in subset], dtype=float)
            finite = np.isfinite(values)
            if finite.sum() >= 4 and len(np.unique(labels[finite])) == 2:
                try:
                    auc = roc_auc_score(labels[finite], values[finite])
                except ValueError:
                    continue
                summary["auc_depth_helped_vs_easy"][metric] = float(auc)
    return summary


def run_dataset(spec, args, output_dir: Path):
    h5_path = Path(spec["h5_path"])
    dataset = make_dataset(
        h5_path,
        frameskip=args.frameskip,
        num_steps=args.history_size + 1,
        img_size=spec.get("img_size", args.img_size),
    )
    model = swm.wm.utils.load_pretrained(spec["policy"]).to(args.device)
    model.eval()

    all_rows = []
    for seed, k1_path, k4_path in spec["evals"]:
        k1_success = parse_successes(Path(k1_path))
        k4_success = parse_successes(Path(k4_path))
        if len(k1_success) != len(k4_success):
            raise ValueError(f"Mismatched success arrays for {spec['name']} seed {seed}")
        global_indices, episodes, starts, offsets = eval_global_indices(
            h5_path, int(seed), len(k1_success), args.goal_offset
        )
        latent = compute_latent_metrics(
            model=model,
            dataset=dataset,
            episodes=episodes,
            starts=starts,
            batch_size=args.batch_size,
            history_size=args.history_size,
            max_depth=args.max_depth,
            device=args.device,
        )
        state = state_metrics(
            h5_path,
            global_indices,
            episodes,
            starts,
            offsets,
            args.goal_offset,
        )
        for i in range(len(k1_success)):
            row = {
                "dataset": spec["name"],
                "seed": int(seed),
                "eval_index": int(i),
                "k1_success": bool(k1_success[i]),
                "k4_success": bool(k4_success[i]),
                "category": category(bool(k1_success[i]), bool(k4_success[i])),
                **latent[i],
                **state[i],
            }
            all_rows.append(row)

    metrics = [
        "latent_mse_k1",
        "latent_mse_k4",
        "latent_mse_improvement_abs",
        "latent_mse_improvement_rel",
        "latent_oracle_depth_2pct",
        "residual_mean_k1",
        "residual_mean_k4",
        "residual_max",
        "goal_pos_dist_mean",
        "goal_pos_dist_max",
        "goal_quat_dist_mean",
        "goal_quat_dist_max",
        "goal_qpos_dist",
        "goal_observation_dist",
        "start_block_pairwise_min",
        "action_norm_mean_to_goal",
        "action_norm_max_to_goal",
    ]
    summary = summarize(all_rows, metrics)
    summary["spec"] = {
        "name": spec["name"],
        "policy": spec["policy"],
        "h5_path": str(h5_path),
        "num_seeds": len(spec["evals"]),
    }

    csv_path = output_dir / f"{spec['name']}_rows.csv"
    json_path = output_dir / f"{spec['name']}_summary.json"
    fieldnames = sorted({key for row in all_rows for key in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary, csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="analysis/k_refinement")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    os.environ.setdefault("STABLEWM_HOME", "/home/robotuser/zijian/lewm_data")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "name": "cube_single_dynamic_oracle",
            "img_size": 224,
            "h5_path": "/home/robotuser/zijian/lewm_data/datasets/ogbench/cube_single_expert.h5",
            "policy": "ttjepa_cube_dynamic_oracle_k4_10e/weights_epoch_10.pt",
            "evals": [
                (
                    42,
                    "/home/robotuser/zijian/lewm_data/ttjepa_cube_dynamic_oracle_k4_10e/ttjepa_cube_dynamic_oracle_k4_10e_depth_sweep_seed42_fixed_k1_results.txt",
                    "/home/robotuser/zijian/lewm_data/ttjepa_cube_dynamic_oracle_k4_10e/ttjepa_cube_dynamic_oracle_k4_10e_depth_sweep_seed42_fixed_k4_results.txt",
                ),
                (
                    43,
                    "/home/robotuser/zijian/lewm_data/ttjepa_cube_dynamic_oracle_k4_10e/ttjepa_cube_dynamic_oracle_k4_10e_depth_sweep_seed43_fixed_k1_results.txt",
                    "/home/robotuser/zijian/lewm_data/ttjepa_cube_dynamic_oracle_k4_10e/ttjepa_cube_dynamic_oracle_k4_10e_depth_sweep_seed43_fixed_k4_results.txt",
                ),
                (
                    44,
                    "/home/robotuser/zijian/lewm_data/ttjepa_cube_dynamic_oracle_k4_10e/ttjepa_cube_dynamic_oracle_k4_10e_depth_sweep_seed44_fixed_k1_results.txt",
                    "/home/robotuser/zijian/lewm_data/ttjepa_cube_dynamic_oracle_k4_10e/ttjepa_cube_dynamic_oracle_k4_10e_depth_sweep_seed44_fixed_k4_results.txt",
                ),
            ],
        },
        {
            "name": "cube_triple_dynamic_oracle",
            "img_size": 224,
            "h5_path": "/home/robotuser/zijian/lewm_data/datasets/ogbench/visual_cube_triple_play.h5",
            "policy": "ttjepa_cube_triple_dynamic_oracle_k4_10e/weights_epoch_10.pt",
            "evals": [
                (
                    42,
                    "/home/robotuser/zijian/lewm_data/ttjepa_cube_triple_dynamic_oracle_k4_10e/ttjepa_cube_triple_dynamic_fixed_k1_results.txt",
                    "/home/robotuser/zijian/lewm_data/ttjepa_cube_triple_dynamic_oracle_k4_10e/ttjepa_cube_triple_dynamic_fixed_k4_results.txt",
                )
            ],
        },
        {
            "name": "reacher_dynamic_oracle",
            "img_size": 224,
            "h5_path": "/home/robotuser/zijian/lewm_data/reacher.h5",
            "policy": "ttjepa_reacher_dynamic_oracle_k4_10e/weights_epoch_10.pt",
            "evals": [
                (
                    42,
                    "/home/robotuser/zijian/lewm_data/ttjepa_reacher_dynamic_oracle_k4_10e/ttjepa_reacher_dynamic_oracle_k4_10e_seed42_fixed_k1_results.txt",
                    "/home/robotuser/zijian/lewm_data/ttjepa_reacher_dynamic_oracle_k4_10e/ttjepa_reacher_dynamic_oracle_k4_10e_seed42_fixed_k4_results.txt",
                )
            ],
        },
        {
            "name": "cube_double_dynamic_oracle",
            "img_size": 224,
            "h5_path": "/home/robotuser/zijian/lewm_data/datasets/ogbench/visual_cube_double_play.h5",
            "policy": "ttjepa_cube_double_dynamic_oracle_k4_10e/weights_epoch_10.pt",
            "evals": [
                (
                    42,
                    "/home/robotuser/zijian/lewm_data/ttjepa_cube_double_dynamic_oracle_k4_10e/ttjepa_cube_double_dynamic_oracle_k4_10e_fixed_k1_results.txt",
                    "/home/robotuser/zijian/lewm_data/ttjepa_cube_double_dynamic_oracle_k4_10e/ttjepa_cube_double_dynamic_oracle_k4_10e_fixed_k4_results.txt",
                )
            ],
        },
    ]

    all_summaries = {}
    for spec in specs:
        print(f"RUN {spec['name']}", flush=True)
        summary, csv_path, json_path = run_dataset(spec, args, output_dir)
        all_summaries[spec["name"]] = summary
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"WROTE {csv_path}", flush=True)
        print(f"WROTE {json_path}", flush=True)

    combined = output_dir / "combined_summary.json"
    combined.write_text(json.dumps(all_summaries, indent=2, sort_keys=True))
    print(f"WROTE {combined}", flush=True)


if __name__ == "__main__":
    main()
