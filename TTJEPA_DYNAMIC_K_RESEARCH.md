# TTJepa Dynamic K Research Notes

Last updated: 2026-06-20 PT / 2026-06-21 UTC

This document tracks the current TTJepa direction for dynamic test-time
refinement depth (`K`) in latent world-model planning. It is a working research
record, not a polished paper draft.

## Current Thesis

The main paper should focus on one question:

> Can a latent world-model planner learn to spend more recurrent transition
> refinement only when deeper imagined dynamics are useful?

The clean framing is dynamic test-time compute along the transition-depth axis.
The paper should not become primarily about sampling width, CEM iterations, or
training-time latent regularization unless those are used as controlled
comparisons or follow-up analysis.

Recommended paper claim:

> We study per-transition test-time scaling in latent world-model planning by
> adding a recurrent transition predictor and a learned depth selector. On
> contact-heavy cube manipulation, learned dynamic K can match or improve fixed
> depth while using close to K=1 average compute.

Avoid these claims for now:

- Do not claim dynamic K is universally better than sampling width, CEM
  iterations, or horizon until equal-budget studies are complete.
- Do not use the strongest joint-depth result as headline dynamic-compute
  evidence when fixed K1 is already equally strong.
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

### 3. Planner-feature selector is a diagnostic, not the final method

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

### 4. Jointly learned selector: current best clean evidence

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

The best main-paper result from this batch is `rel00005`: learned dynamic K
gets 78% while using almost K1 compute, and both fixed K1 and fixed K4 are 74%.
That is the cleanest evidence that model-internal dynamic K selection can matter.

The most important caveat is `rel0005`: it improves all depths to 80%. That is
interesting, but it is not a dynamic-compute win because K1 already gets 80%.
The correct wording is:

> A stronger joint-depth training variant improves all depths to 80%, suggesting
> a separate training-time regularization effect; we exclude it from the main
> dynamic-compute comparison and discuss it separately.

This should become a separate analysis thread: joint marginal-depth supervision
may reduce latent smoothing or task-detail collapse.

## Current Paper Position

This is already enough for a promising paper direction, but not yet enough for
an ICLR oral-level empirical claim. The central evidence is coherent:

1. Fixed K helps on cube-triple, so there is useful transition-depth compute.
2. Hindsight K1/K4 selection suggests only a small subset needs deeper compute.
3. Raw latent MSE partially works but exposes why generic latent error is weak.
4. A jointly trained selector can beat fixed K at almost K1 average compute in
   the clean `rel00005` setting.
5. A stronger joint-depth setting (`rel0005`) suggests a second phenomenon:
   depth-supervised training itself may improve the latent dynamics.

The main risk is still statistical and scope-related: most key cube-triple
numbers are 50-episode, single-seed results. The paper needs multi-seed
replication and tighter separation between dynamic test-time compute and
training-time regularization.

## Recommended Main-Text Story

Use this sequence:

1. Start from latent world-model planning and the cost of imagined transitions.
2. Introduce recurrent transition refinement depth `K`.
3. Show fixed K helps on cube-triple but wastes compute on easy transitions.
4. Show raw latent MSE is a reasonable first attempt but misses planner-relevant
   contact detail.
5. Train a selector jointly with the recurrent predictor using marginal-depth
   benefit targets.
6. Report the clean dynamic-K result (`rel00005`) as the main method.
7. Put `rel0005` in a separate section or appendix as a training-regularization
   observation, not as the headline dynamic-compute result.

Good contribution wording:

- We formulate transition refinement depth as a test-time compute axis in
  latent world-model planning.
- We show that raw latent prediction error is not enough to identify when deep
  refinement affects planning success.
- We train an internal selector that chooses per-transition recurrent depth and
  improves success at near-K1 average compute on cube-triple.
- We identify a separate effect where joint depth supervision can improve all
  fixed depths, suggesting a connection to latent smoothing and task-relevant
  detail preservation.

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

1. Replicate `rel00005` and `rel0002` on cube-triple with at least 3 seeds.
2. For each replicated run, evaluate fixed K1/K2/K3/K4 and learned thresholds
   on a held-out validation split before reporting test success.
3. Report selector precision/recall over K1-fail/K4-success and
   K1-success/K4-fail episodes.
4. Add wall-clock latency and average number of recurrent transition calls.
5. Run the same clean setting on reacher, cube-single, and cube-double to show
   when dynamic K is useful and when K1 is enough.
6. Keep the `rel0005` analysis in a separate section: useful phenomenon, not
   the headline dynamic-K comparison.

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

