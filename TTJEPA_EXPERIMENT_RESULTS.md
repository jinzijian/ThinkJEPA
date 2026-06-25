# TTJepa Experiment Results Ledger

Last updated: 2026-06-23 PT

This file preserves the full experiment record. The README now keeps the
public-facing motivation plus the main fixed-depth, post-hoc raw-MSE, and
raw-MSE-supervised learned-head result tables. This ledger is broader: it also
keeps checkpoint paths, logs, planner-feature selectors, whitened/probe-weighted
variants, and exploratory training-time regularization leads.

## Scope Map

| Thread | Included in Paper 1? | Role |
| --- | --- | --- |
| Fixed-depth recurrent TTJepa, `K1/K2/K3/K4` | Yes | Establishes whether transition depth matters |
| Raw latent MSE dynamic K | Yes | Main method for Paper 1 |
| Hindsight K1/K4 chooser | Yes | Upper bound / diagnostic only |
| Latent spectrum and state-probe analysis | Yes | Mechanistic/failure analysis |
| Planner-feature selector | No | Diagnostic evidence that planner traces contain useful signal |
| Learned continue head / joint marginal-depth training | No | Future work / separate paper |
| Strong joint-depth `rel0005` 80% result | No | Training-time regularization lead |
| Whitened / probe-weighted halt labels | No | Exploratory alternatives |

## Main Fixed-Depth And Raw-MSE Results

The raw-MSE analysis uses the recurrent checkpoints associated with the
K-refinement rows under `analysis/k_refinement_all_20260620_024634`.

| Dataset / run | LeWM baseline | Fixed K1 | Fixed K2 | Fixed K3 | Fixed K4 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reacher seed42 | 80% | 88% | n/a | n/a | 86% | Current checkpoint does not need large K |
| Cube single seed42 | 72% | 80% | n/a | n/a | 78% | K4 slightly lower than K1 |
| Cube single seed43 | 72% | 88% | n/a | n/a | 90% | K4 improves by 2 points |
| Cube single seed44 | 72% | 66% | n/a | n/a | 64% | K4 slightly lower than K1 |
| Cube single 3-seed avg | 72% | 78% | n/a | n/a | 77.3% | Seed-level gain but not stable mean gain |
| Cube single original rerun `20260621_refixed_k1234` | 72% | 80% | 76% | 78% | 78% | Same original checkpoint; K1 best |
| Cube double original rerun `20260621_refixed_k1234` | 66% | 72% | 70% | 68% | 70% | Extra depth does not help |
| Cube triple original | 74% | 70% | 76% | 76% | 78% | Cleanest fixed-depth K gain |
| Cube triple whitened checkpoint | 74% | 72% | 72% | 76% | 76% | Exploratory checkpoint; depth still helps |

Important distinction: `LeWM baseline` is the original non-recurrent baseline.
`Fixed K1` is TTJepa's recurrent transition predictor stopped after one
refinement step. They are not the same model.

## Raw Latent MSE Dynamic K

This is the Paper 1 dynamic-K method.

| Dataset | Fixed K1 | Fixed K4 | Best raw-MSE dynamic K | Hindsight K1/K4 chooser | K1 fail / K4 success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reacher | 88%@K1.00 | 86%@K4.00 | 88%@K1.06 to K2.32 | 92%@K1.12 | 2 / 50 |
| Cube single | 78%@K1.00 | 77.3%@K4.00 | 77.3%@K2.72 to K2.96 | 80.7%@K1.08 | 4 / 150 |
| Cube double | 72%@K1.00 | 70%@K4.00 | 72%@K1.00 to K2.62 | 72%@K1.00 | 0 / 50 |
| Cube triple | 70%@K1.00 | 78%@K4.00 | 76%@K2.32 | 82%@K1.36 | 6 / 50 |

Cube-triple threshold sweep:

| Rule | Success | Mean K | Selected episodes | Notes |
| --- | ---: | ---: | ---: | --- |
| Fixed K1 | 70% | 1.00 | 0 / 50 | Shallow baseline |
| Raw latent MSE, tolerance 0 | 76% | 2.32 | 22 / 50 | Recovers part of the K4 gain |
| Raw latent MSE, tolerance 0.001 | 74% | 1.96 | 16 / 50 | Less compute, weaker success |
| Raw latent MSE, tolerance 0.003 | 72% | 1.54 | 9 / 50 | Too conservative |
| Fixed K4 | 78% | 4.00 | 50 / 50 | Stronger but expensive |

Cube-triple K1/K4 episode categories:

| Category | Count |
| --- | ---: |
| K1 fails, K4 succeeds | 6 |
| K1 succeeds, K4 fails | 2 |
| Both succeed | 33 |
| Both fail | 9 |

Local artifacts:

- `analysis/k_refinement_all_20260620_024634/raw_mse_k_gating_sweep.csv`
- `analysis/k_refinement_all_20260620_024634/k_gating_pareto_all.png`
- `analysis/paper1_figures/png_direct/raw_mse_tolerance_pareto.png`
- `analysis/paper1_figures/png_direct/k1_k4_outcome_split.png`
- `analysis/paper1_figures/png_direct/raw_mse_precision_recall_failure.png`

## Mechanistic Analysis: Spectrum And State Probes

Completed analysis pass: `analysis/k_smoothing_20260622`.

Result: there is no strong global latent-collapse signature from K1 to K4.

| Dataset | Spectrum summary | Probe summary |
| --- | --- | --- |
| Reacher | K4/K1 entropy-rank ratio `1.000`, variance ratio `1.000` | qpos R2 `-0.131 -> -0.131`; observation probe is poor |
| Cube single | entropy-rank ratio `1.000`, variance ratio `1.003` | block position R2 `0.991 -> 0.991` |
| Cube double | entropy-rank ratio `1.000`, variance ratio `1.000` | block position R2 `0.946 -> 0.946`; pairwise distance `0.900 -> 0.899` |
| Cube triple | entropy-rank ratio `1.000`, variance ratio `1.000` | block position R2 `0.902 -> 0.902`; pairwise distance `0.895 -> 0.894` |

Figures:

- `analysis/k_smoothing_20260622/figures/spectrum_k1_vs_k4_scatter.png`
- `analysis/k_smoothing_20260622/figures/probe_r2_k1_vs_k4_scatter.png`
- `analysis/k_smoothing_20260622/figures/category_probe_mse_k1_vs_k4_scatter.png`

Interpretation: the failure mode is likely more local than global spectrum or
simple linear state probes can see. The next decisive analysis is CEM candidate
ranking stability.

## Planner-Feature Selector Diagnostic

This is not Paper 1's method. It is a diagnostic showing that planner/result
features can contain stronger information about when K4 is useful.

| Threshold range | Success | Mean K |
| --- | ---: | ---: |
| 0.38 to 0.41 | 80% | 2.53 to 2.63 |
| 0.50 | 74% | 1.80 |
| 0.70 | 72% | 1.12 |

Interpretation: planner trajectory information can predict useful extra depth,
but this was a separate diagnostic selector rather than the raw-MSE Paper 1
method.

## Learned Continue Head / Joint Marginal-Depth Runs

These results are preserved for future work and should not be mixed into the
Paper 1 raw-MSE story.

| Run | Learned dynamic result | Fixed K1 sanity | Fixed K4 sanity | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `rel00005` | 78% at K=1.064 | 74% | 74% | Clean learned-selector gain |
| `rel0002` | 78% at K=1.035 | 78% | 72% | Avoids harmful over-refinement, but does not beat K1 |
| `rel0005` | 80% at K=1.000 to K=1.062 | 80% | 80% | Training-time regularization effect, not dynamic-compute evidence |
| `rel0001` | 74% at K=1.47 | not sanity-checked | not sanity-checked | Weaker setting |
| `rel000` | 66% near K1 | not sanity-checked | not sanity-checked | No-margin target fails |

Preferred wording for `rel0005`:

> A stronger joint-depth training variant improves all depths to 80%,
> suggesting a separate training-time regularization effect; we exclude it from
> the main dynamic-compute comparison and discuss it separately.

Remote checkpoints:

- `/vepfs/zijian/lewm_data/checkpoints/ttjepa_cube_triple_joint_marginal_rel00005_k4_10e/weights_epoch_10.pt`
- `/vepfs/zijian/lewm_data/checkpoints/ttjepa_cube_triple_joint_marginal_rel0002_k4_10e/weights_epoch_10.pt`
- `/vepfs/zijian/lewm_data/checkpoints/ttjepa_cube_triple_joint_marginal_rel0005_k4_10e/weights_epoch_10.pt`

Remote result directories:

- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_joint_marginal_rel00005_k4_10e`
- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_joint_marginal_rel0002_k4_10e`
- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_joint_marginal_rel0005_k4_10e`

Important logs:

- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_joint_marginal_rel00005_k4_10e_20260620_085100_fine.log`
- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_joint_marginal_rel0002_k4_10e_20260620_085100_fine.log`
- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_joint_marginal_rel0005_k4_10e_20260620_085100.log`

## Exploratory Halt-Label Variants

These are alternative label/score designs explored during debugging. They are
not part of Paper 1.

| Variant | Result summary | Status |
| --- | --- | --- |
| Raw learned halt thresholds | Best learned result around 74% at mean depth about 1.40 | Useful diagnosis; weaker than fixed K4 |
| Whitened latent MSE | Fixed K1/K2/K3/K4 = 72/72/76/76; learned best 74% | Whitening did not clearly fix halting |
| Probe-weighted latent MSE | Ran as an exploratory queue; keep result files under remote experiment directory | Needs exact final table copied from remote artifacts |

Known remote directories:

- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_dynamic_whitened_oracle_k4_10e`
- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_dynamic_probe_weighted_oracle_k4_10e`

Known logs:

- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_whitened_oracle_20260617_20260617_063655.log`
- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_probe_weighted_oracle_20260617_193620.log`

## Remote Layout

Remote machine:

- SSH: `ssh -p 20747 root@115.190.235.210`
- Repo: `/vepfs/zijian/TTJepa`
- Data/results root: `/vepfs/zijian/lewm_data`

Important raw recurrent checkpoint:

- `/vepfs/zijian/lewm_data/checkpoints/ttjepa_cube_triple_dynamic_oracle_k4_10e/weights_epoch_10.pt`

Important raw recurrent result directory:

- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_dynamic_oracle_k4_10e`

Local analysis scripts:

- `scripts/k_refinement_analysis.py`
- `scripts/k_raw_mse_sweep.py`
- `scripts/k_smoothing_analysis.py`
