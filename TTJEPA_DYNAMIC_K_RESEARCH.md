# TTJepa Dynamic K Research Notes

Last updated: 2026-06-23 PT

This document tracks the current TTJepa direction for dynamic test-time
refinement depth (`K`) in latent world-model planning. It is a working research
record, not a polished paper draft. The full result ledger is
[TTJEPA_EXPERIMENT_RESULTS.md](TTJEPA_EXPERIMENT_RESULTS.md).

## Current Thesis

Paper 1 should focus on one question:

> Can a simple raw latent prediction-error signal decide when a latent
> world-model planner should spend extra recurrent transition refinement?

The clean framing is dynamic test-time compute along the transition-depth axis,
with raw latent MSE as the only dynamic-K rule in the main paper. Learned
selectors, joint-depth training, planner-feature selectors, and label variants
are useful follow-up experiments, but they should stay outside the Paper 1 main
tables.

Recommended paper claim:

> We study per-transition test-time scaling in latent world-model planning by
> exposing recurrent transition refinement depth `K` and using raw latent
> prediction error as a simple dynamic stopping signal. Raw latent MSE recovers
> much of the cube-triple fixed-depth gain at lower average compute, while its
> failures reveal a gap between latent prediction error and planner benefit.

Avoid these claims for now:

- Do not claim dynamic K is universally better than sampling width, CEM
  iterations, or horizon until equal-budget studies are complete.
- Do not use learned-selector or joint-depth results as Paper 1 evidence.
  Preserve them in the result ledger and frame them as future work.
- Do not describe checkpoint/result path names in paper text as "oracle"; use
  "hindsight", "fixed-depth target", "teacher target", or "learned selector"
  depending on context.

## Main Evidence So Far

### 1. K matters on cube-triple

The original recurrent TTJepa cube-triple checkpoint shows that deeper
transition refinement can help:

| Model / rule | Success | Mean K | Interpretation |
| --- | ---: | ---: | --- |
| LeWM baseline | 74% | n/a | Non-recurrent baseline |
| TTJepa fixed K1 | 70% | 1.00 | Shallow recurrent transition is not enough |
| TTJepa fixed K2 | 76% | 2.00 | Most of the fixed-depth gain appears by K2 |
| TTJepa fixed K3 | 76% | 3.00 | Similar to K2 |
| TTJepa fixed K4 | 78% | 4.00 | Best fixed-depth result in this run |
| Hindsight K1/K4 chooser | 82% | 1.36 | Upper bound: use K4 exactly on K1-fail/K4-success episodes |

This is the core reason to keep focusing on K: cube-triple contains cases where
extra recurrent refinement changes success, and the hindsight upper bound says
only a small fraction of episodes need deep compute.

### 2. Raw latent MSE is a reasonable v0 signal, but incomplete

Using raw latent MSE to decide whether to continue from K1 to K4 is not
meaningless:

| Rule | Success | Mean K | Selected episodes | Notes |
| --- | ---: | ---: | ---: | --- |
| Fixed K1 | 70% | 1.00 | 0 | Baseline shallow depth |
| Raw latent MSE, tolerance 0 | 76% | 2.32 | 22 / 50 | Recovers half of the depth-helped cases |
| Raw latent MSE, tolerance 0.001 | 74% | 1.96 | 16 / 50 | Less compute, weaker success |
| Raw latent MSE, tolerance 0.003 | 72% | 1.54 | 9 / 50 | Too conservative |
| Fixed K4 | 78% | 4.00 | 50 / 50 | Stronger but expensive |

Episode categories for cube-triple:

| Category | Count |
| --- | ---: |
| K1 fails, K4 succeeds | 6 |
| K1 succeeds, K4 fails | 2 |
| Both succeed | 33 |
| Both fail | 9 |

Conclusion: raw latent MSE has signal, but it is too blunt. It recovers some
depth-helped cases, but it cannot match fixed K4 or the hindsight upper bound.
This supports the analysis that latent MSE is smoothed and not fully aligned
with planner-relevant contact details.

### 3. Planner-feature selector is a diagnostic, not Paper 1

A separate diagnostic selector using planner/result features reached stronger
cube-triple numbers:

| Threshold range | Success | Mean K |
| --- | ---: | ---: |
| 0.38 to 0.41 | 80% | 2.53 to 2.63 |
| 0.50 | 74% | 1.80 |
| 0.70 | 72% | 1.12 |

Interpretation:

- The planner trajectory contains useful information about whether deeper K is
  worth paying for.
- This is not yet the preferred final method because it was trained as a
  separate diagnostic selector rather than learned jointly inside the model.
- The result is useful motivation for a planner-aware learned selector, but it
  should not be presented as the clean main method without integration.

### 4. Jointly learned selector: future work, not Paper 1

The latest cube-triple joint marginal-benefit runs train a `continue_head`
together with the recurrent predictor. The label asks whether the next depth
still gives enough marginal improvement. This is closer to the desired model:
the model learns its own dynamic K behavior during training rather than using a
separate post-hoc selector.

Key results:

| Run | Learned dynamic result | Fixed K1 sanity | Fixed K4 sanity | Main interpretation |
| --- | ---: | ---: | ---: | --- |
| `rel00005` | 78% at K=1.064 | 74% | 74% | Clean dynamic-K gain: +4 over both shallow and deep fixed depth |
| `rel0002` | 78% at K=1.035 | 78% | 72% | Learned halting avoids harmful over-refinement |
| `rel0005` | 80% at K=1.000 | 80% | 80% | Strong training-time regularization effect, not clean dynamic-K evidence |
| `rel0001` | 74% at K=1.47 | not sanity-checked here | not sanity-checked here | Weaker setting |
| `rel000` | 66% near K1 | not sanity-checked here | not sanity-checked here | No-margin target fails |

The strongest clean learned-selector result from this batch is `rel00005`:
learned dynamic K gets 78% while using almost K1 compute, and both fixed K1 and
fixed K4 are 74%. This is promising future work, but it is not part of Paper 1
after the decision to make Paper 1 raw-MSE-only.

The most important caveat is `rel0005`: it improves all depths to 80%. That is
interesting, but it is not a dynamic-compute win because K1 already gets 80%.
The correct wording is:

> A stronger joint-depth training variant improves all depths to 80%, suggesting
> a separate training-time regularization effect; we exclude it from the main
> dynamic-compute comparison and discuss it separately.

This should become a separate analysis thread: joint marginal-depth supervision
may reduce latent smoothing or task-detail collapse.

## Current Paper Position

This is already enough for a promising raw-MSE Paper 1 direction, but not yet
enough for an ICLR oral-level empirical claim. The central evidence is coherent:

1. Fixed K helps on cube-triple, so there is useful transition-depth compute.
2. Hindsight K1/K4 selection suggests only a small subset needs deeper compute.
3. Raw latent MSE partially works and exposes the gap between generic latent
   prediction error and planner benefit.
4. Spectrum/probe analysis shows the failure is not explained by a broad global
   latent-collapse signature.
5. CEM candidate-ranking stability is the next decisive mechanism analysis.

The main risk is still statistical and mechanism-related: most key
cube-triple numbers are 50-episode, single-seed results, and the CEM-ranking
analysis is not yet complete.

## Recommended Main-Text Story

Use this sequence:

1. Start from latent world-model planning and the cost of imagined transitions.
2. Introduce recurrent transition refinement depth `K`.
3. Show fixed K helps on cube-triple but wastes compute on easy transitions.
4. Show raw latent MSE is a reasonable first attempt but misses planner-relevant
   contact detail.
5. Test whether the failure is global smoothing/collapse using spectrum and
   state-probe analysis.
6. Show that global spectrum/probe metrics are nearly unchanged, pointing to
   local planner alignment rather than broad latent collapse.
7. Identify CEM ranking stability as the next required diagnostic.

Good contribution wording:

- We formulate transition refinement depth as a test-time compute axis in
  latent world-model planning.
- We show that raw latent prediction error is not enough to identify when deep
  refinement affects planning success.
- We show that broad latent spectrum/probe metrics do not explain the failure,
  motivating planner-ranking analysis.

## Main Analysis Module: Latent Smoothing And Planner Alignment

This analysis should be part of the main dynamic-K paper, even if the stronger
joint-depth training result remains a follow-up. It explains why fixed or
dynamic deeper `K` is not uniformly better.

Hypothesis tested: deeper recurrent refinement may reduce raw latent prediction
error while globally smoothing away task-relevant contact details. Planning
success depends on whether the latent preserves information that changes CEM
candidate ranking, not only on whether the latent is easier to reconstruct.

Completed analysis pass: `analysis/k_smoothing_20260622`.

- Recomputed predicted latents for `K1/K2/K3/K4` on reacher, cube-single,
  cube-double, and cube-triple, using the same windows as the K-refinement
  eval.
- Measured per-depth variance, singular-value spectrum, entropy effective
  rank, and top singular-value concentration.
- Trained lightweight ridge probes from predicted latents to available
  task-state labels: `qpos`, low-dimensional observations, block position,
  block quaternion, goal-relative block position, and pairwise block distance.
- Reported both all-window statistics and K1/K4 outcome categories: easy,
  depth-helped, depth-hurt, and hard.

Observed result:

- Global spectrum is nearly invariant to depth. `K4/K1` entropy-rank ratio is
  essentially `1.000` on all four datasets, and variance/top-direction metrics
  barely move.
- State probes are nearly invariant to depth. Examples: cube-single block
  position `R2=0.991 -> 0.991`, cube-double block position `0.946 -> 0.946`,
  cube-triple block position `0.902 -> 0.902`. Most other probe deltas are
  around the third decimal place.
- Depth-helped and depth-hurt subsets do not show a clean global-rank collapse
  pattern. The subset changes are small and noisy rather than a broad
  compression effect.

Interpretation: the current evidence does not support a strong global latent
collapse/smoothing claim. The more plausible failure mode is local planner
alignment: small transition changes can alter CEM elite ranking or selected
actions without changing global spectrum or simple linear state probes.

Remaining decisive analysis: CEM candidate ranking.

- For the same sampled candidate action sequences, evaluate terminal costs with
  `K1/K2/K3/K4` rollouts.
- Measure elite-set overlap, Kendall rank correlation, and selected-action
  changes across depth.
- Target result: on cube-triple, larger `K` should correct rankings in the
  small subset of depth-helped episodes. On cube-single/cube-double, larger `K`
  may preserve raw latent quality but fail to improve, or may destabilize, the
  planner ranking.

Target conclusion: raw latent MSE is a useful first dynamic-K signal, but the
failure mode is planner alignment. The better diagnostic for test-time compute
is whether extra refinement preserves task-state information and improves CEM
ranking quality.

## Separate Follow-Up Thread: Latent Smoothing / Collapse

Do not let this become the main paper unless dynamic K stalls.

The `rel0005` run is important because it changes the predictor itself: fixed
K1, learned dynamic K, and fixed K4 all reach 80%. That suggests the marginal
depth supervision may act as training-time regularization, possibly forcing the
latent dynamics to keep task-relevant contact details that the plain latent MSE
objective smooths away.

Required checks before making this a strong claim:

- Repeat `rel0005` across at least 3 seeds.
- Compare latent spectrum / effective rank against raw recurrent training.
- Train lightweight probes for block pose, goal-relative pose, and contact-like
  state variables.
- Measure CEM candidate ranking quality by depth, not just latent MSE.
- Compare against ordinary intermediate-depth MSE supervision to show the
  effect is from marginal-benefit supervision, not merely extra loss.

## Next Experiments

Priority order:

1. Replicate the raw-MSE dynamic-K analysis with additional seeds where
   practical, especially cube-triple.
2. Report raw-MSE precision/recall over K1-fail/K4-success and
   K1-success/K4-fail episodes.
3. Add wall-clock latency and average number of recurrent transition calls.
4. Add CEM candidate-ranking stability: elite-set overlap, Kendall tau, and
   selected-action changes for K1/K2/K3/K4.
5. Keep learned-selector, planner-feature, and joint-depth training results in
   [TTJEPA_EXPERIMENT_RESULTS.md](TTJEPA_EXPERIMENT_RESULTS.md), not in Paper 1
   main tables.

## Paths And Artifacts

Remote machine:

- SSH: `ssh -p 20747 root@115.190.235.210`
- Repo: `/vepfs/zijian/TTJepa`
- Data/results root: `/vepfs/zijian/lewm_data`

Important checkpoints:

- Raw recurrent cube-triple:
  `/vepfs/zijian/lewm_data/checkpoints/ttjepa_cube_triple_dynamic_oracle_k4_10e/weights_epoch_10.pt`
- Joint marginal `rel00005`:
  `/vepfs/zijian/lewm_data/checkpoints/ttjepa_cube_triple_joint_marginal_rel00005_k4_10e/weights_epoch_10.pt`
- Joint marginal `rel0002`:
  `/vepfs/zijian/lewm_data/checkpoints/ttjepa_cube_triple_joint_marginal_rel0002_k4_10e/weights_epoch_10.pt`
- Joint marginal `rel0005`:
  `/vepfs/zijian/lewm_data/checkpoints/ttjepa_cube_triple_joint_marginal_rel0005_k4_10e/weights_epoch_10.pt`

Important result directories:

- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_dynamic_oracle_k4_10e`
- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_joint_marginal_rel00005_k4_10e`
- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_joint_marginal_rel0002_k4_10e`
- `/vepfs/zijian/lewm_data/ttjepa_cube_triple_joint_marginal_rel0005_k4_10e`

Important logs:

- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_joint_marginal_rel00005_k4_10e_20260620_085100_fine.log`
- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_joint_marginal_rel0002_k4_10e_20260620_085100_fine.log`
- `/vepfs/zijian/TTJepa/logs/ttjepa_cube_triple_joint_marginal_rel0005_k4_10e_20260620_085100.log`
- `/vepfs/zijian/TTJepa/logs/*fixed_k*_20260621_002537_fixed_sanity.log`

Local analysis artifacts:

- `analysis/k_refinement_all_20260620_024634/raw_mse_k_sweep_summary.json`
- `analysis/k_refinement_all_20260620_024634/combined_summary.json`
- `analysis/k_refinement_all_20260620_024634/k_gating_pareto_all.png`
