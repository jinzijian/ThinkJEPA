#!/usr/bin/env python3
"""Post-hoc sweeps for raw latent-MSE K allocation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1"})


def summarize_policy(df: pd.DataFrame, continue_mask: np.ndarray) -> dict:
    k1 = bool_series(df["k1_success"]).to_numpy()
    k4 = bool_series(df["k4_success"]).to_numpy()
    cont = np.asarray(continue_mask, dtype=bool)
    success = np.where(cont, k4, k1)
    helped = (~k1) & k4
    hurt = k1 & (~k4)
    return {
        "success_rate": float(success.mean()),
        "mean_k": float(1.0 + 3.0 * cont.mean()),
        "continue_rate": float(cont.mean()),
        "helped_recall": float((cont & helped).sum() / max(helped.sum(), 1)),
        "helped_precision": float((cont & helped).sum() / max(cont.sum(), 1)),
        "hurt_selected_rate": float((cont & hurt).sum() / max(hurt.sum(), 1)),
        "selected_count": int(cont.sum()),
    }


def top_fraction_mask(values: pd.Series, frac: float, descending: bool = True) -> np.ndarray:
    n = len(values)
    k = int(round(frac * n))
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    order = values.to_numpy().argsort()
    if descending:
        order = order[::-1]
    mask[order[:k]] = True
    return mask


def run_dataset(df: pd.DataFrame) -> dict:
    k1 = bool_series(df["k1_success"]).to_numpy()
    k4 = bool_series(df["k4_success"]).to_numpy()
    helped = (~k1) & k4
    hurt = k1 & (~k4)

    out = {
        "n": int(len(df)),
        "fixed_k1": {"success_rate": float(k1.mean()), "mean_k": 1.0},
        "fixed_k4": {"success_rate": float(k4.mean()), "mean_k": 4.0},
        "perfect_k1_k4_selector": {
            **summarize_policy(df, helped),
            "success_rate": float((k1 | k4).mean()),
            "note": "Use K4 exactly on K1-fail/K4-success episodes.",
        },
        "category_counts": df["category"].value_counts().to_dict(),
        "raw_mse_label_tolerance": [],
        "raw_mse_k4_better_margin": [],
        "top_fraction_gating": [],
    }

    for tol in [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.10]:
        cont = df["latent_mse_k1"].to_numpy() > (1.0 + tol) * df["latent_mse_best"].to_numpy()
        row = {"tol": tol, **summarize_policy(df, cont)}
        row["helped_label_rate"] = float((cont & helped).sum() / max(helped.sum(), 1))
        row["hurt_label_rate"] = float((cont & hurt).sum() / max(hurt.sum(), 1))
        out["raw_mse_label_tolerance"].append(row)

    improvement = (
        (df["latent_mse_k1"].to_numpy() - df["latent_mse_k4"].to_numpy())
        / np.maximum(df["latent_mse_k1"].to_numpy(), 1e-12)
    )
    for margin in [-0.005, 0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05]:
        cont = improvement > margin
        out["raw_mse_k4_better_margin"].append(
            {"margin": margin, **summarize_policy(df, cont)}
        )

    score_specs = [
        ("latent_mse_k1", True),
        ("latent_mse_improvement_rel", True),
        ("residual_mean_k1", True),
        ("goal_pos_dist_mean", True),
        ("goal_quat_dist_mean", True),
        ("goal_qpos_dist", True),
        ("goal_observation_dist", True),
    ]
    if "start_block_pairwise_min" in df:
        score_specs.append(("start_block_pairwise_min", False))

    for score, descending in score_specs:
        if score not in df or df[score].isna().all():
            continue
        values = df[score].fillna(df[score].median())
        for frac in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 1.0]:
            cont = top_fraction_mask(values, frac, descending=descending)
            out["top_fraction_gating"].append(
                {"score": score, "fraction": frac, **summarize_policy(df, cont)}
            )

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        default="analysis/k_refinement_20260620_010330",
        help="Directory containing *_rows.csv files.",
    )
    args = parser.parse_args()
    analysis_dir = Path(args.analysis_dir)

    summaries = {}
    table_rows = []
    for csv_path in sorted(analysis_dir.glob("*_rows.csv")):
        name = csv_path.name.replace("_rows.csv", "")
        df = pd.read_csv(csv_path)
        summary = run_dataset(df)
        summaries[name] = summary
        for row in summary["top_fraction_gating"]:
            table_rows.append({"dataset": name, **row})

    out_json = analysis_dir / "raw_mse_k_sweep_summary.json"
    out_csv = analysis_dir / "raw_mse_k_gating_sweep.csv"
    out_json.write_text(json.dumps(summaries, indent=2, sort_keys=True))
    pd.DataFrame(table_rows).to_csv(out_csv, index=False)

    for name, summary in summaries.items():
        print(f"\n{name}")
        print("fixed_k1", summary["fixed_k1"])
        print("fixed_k4", summary["fixed_k4"])
        print("perfect", summary["perfect_k1_k4_selector"])
        print("raw_mse_label_tolerance")
        for row in summary["raw_mse_label_tolerance"]:
            print(row)
        print("raw_mse_k4_better_margin")
        for row in summary["raw_mse_k4_better_margin"]:
            print(row)

    print(f"\nWROTE {out_json}")
    print(f"WROTE {out_csv}")


if __name__ == "__main__":
    main()
