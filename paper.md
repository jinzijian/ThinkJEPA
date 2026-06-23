# When Should a Latent Planner Refine?

Working title:

**When Should a Latent Planner Refine? Dynamic Transition Depth via Raw Latent Error**

## Scope

This draft is for **Paper 1 only**. The paper focuses on recurrent transition
depth `K` and the raw latent-MSE dynamic-K analysis.

Included in the main paper:

- LeWM baseline comparisons.
- TTJepa fixed-depth recurrent transition refinement: `K1/K2/K3/K4`.
- Raw latent-MSE based K allocation.
- Hindsight `K1/K4` chooser as an analysis upper bound.
- Failure analysis of raw MSE.
- Latent spectrum / state-probe analysis.
- CEM candidate-ranking analysis as the next required analysis.

Excluded from the main paper:

- Learned continue-head / joint marginal-depth selector.
- Planner-feature selector.
- Whitened latent MSE.
- Probe-weighted latent MSE.
- The `rel0005` joint-depth result where all fixed depths reach 80%.

Those experiments should stay in [TTJEPA_EXPERIMENT_RESULTS.md](TTJEPA_EXPERIMENT_RESULTS.md)
as supporting records and possible future-paper leads.

## One-Sentence Thesis

Latent world-model planners usually spend test-time compute on CEM sampling
width, optimizer iterations, or rollout horizon. This paper studies a different
axis: **how many recurrent refinement steps should be spent on each imagined
latent transition?**

## Motivation Example

A robot reaching toward an object does not need to think very hard about every
free-space movement. But when the hand makes contact, starts lifting, or needs
to manipulate several objects, a small dynamics error can change the selected
action. A latent planner should therefore spend little transition-compute on
easy free-space transitions and more transition-compute on contact-rich or
planning-sensitive transitions.

This is not generic "robot reasoning." The question is narrower:

> In latent MPC/CEM planning, when should the transition model refine an imagined
> transition more deeply?

## Claim Boundaries

Strong claims supported by current evidence:

1. `K` is a real compute axis inside latent world-model planning.
2. Fixed deeper `K` can change success, especially on cube-triple.
3. Larger `K` is not universally better, so conditional allocation matters.
4. Raw latent MSE is a useful first allocation signal, not an empty heuristic.
5. Raw latent MSE is incomplete because it does not perfectly match planner
   benefit.
6. The failure is not explained by a simple global latent-collapse story.

Claims to avoid:

- Do not claim `K4` is always better than `K1`.
- Do not claim raw latent MSE is optimal.
- Do not claim the current results prove a fully deployable learned halting
  policy.
- Do not use the learned selector or joint-depth results as main Paper 1
  evidence.
- Use neutral terms for the raw-MSE target, such as "hindsight",
  "fixed-depth target", "post-hoc allocation analysis", or "teacher target"
  only when a qualification is needed.

Important deployability note:

The current raw-MSE sweep is a controlled analysis of whether latent prediction
error contains signal for allocating `K`. If the final paper claims fully
test-time dynamic `K`, we must either use a test-time residual proxy or train a
small predictor of this raw-MSE target. If we keep Paper 1 strictly raw-MSE-only,
the safest framing is: **raw latent error reveals when deeper transition
refinement is useful and exposes the remaining gap between prediction error and
planner benefit.**

## Contributions

1. We formulate recurrent transition depth `K` as a transition-level
   test-time compute axis in latent world-model planning.
2. We show that fixed-depth recurrent refinement changes planning success, with
   the clearest gain on visual cube-triple.
3. We evaluate raw latent MSE as the first simple signal for allocating `K`.
4. We show that raw MSE recovers much of the cube-triple fixed-depth gain while
   using lower average compute.
5. We analyze why raw MSE is incomplete through outcome splits, hindsight
   `K1/K4` selection, latent spectrum, state probes, and planned CEM-ranking
   diagnostics.

## Main Experimental Results

### Table 1: LeWM Baseline and Fixed-Depth TTJepa

`LeWM baseline` and `Fixed K1` are not the same model. LeWM baseline uses the
original non-recurrent transition predictor. Fixed `K1` uses the TTJepa
recurrent transition predictor stopped after one refinement step. For dynamic-K
analysis, the fairest comparison is inside the same TTJepa checkpoint:
`K1/K2/K3/K4`.

| Dataset / run | LeWM baseline | Fixed K1 | Fixed K2 | Fixed K3 | Fixed K4 | Observation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reacher seed42 | 80% | 88% | n/a | n/a | 86% | K4 is worse than K1 |
| Cube single seed42 | 72% | 80% | n/a | n/a | 78% | K4 is slightly lower |
| Cube single seed43 | 72% | 88% | n/a | n/a | 90% | K4 improves by 2 points |
| Cube single seed44 | 72% | 66% | n/a | n/a | 64% | K4 is slightly lower |
| Cube single 3-seed avg | 72% | 78% | n/a | n/a | 77.3% | K4 gain is unstable |
| Cube single original rerun | 72% | 80% | 76% | 78% | 78% | K1 best, K3/K4 recover |
| Cube double original rerun | 66% | 72% | 70% | 68% | 70% | Extra depth does not help |
| Cube triple original | 74% | 70% | 76% | 76% | 78% | Clearest K gain |

Core conclusion:

`K` matters, but deeper `K` is dataset- and checkpoint-dependent. The right
claim is not "always use K4." The right claim is "transition depth changes
success and should be allocated conditionally."

Current figure artifact:

- `analysis/paper1_figures/png_direct/main_success_vs_lewm.png`

### Table 2: Raw Latent-MSE Dynamic K

This table is a `K1/K4` dynamic-selection analysis. Each episode either stays at
`K1` or switches to `K4` based on raw latent MSE.

| Dataset | Fixed K1 | Fixed K4 | Best raw-MSE dynamic K | Hindsight K1/K4 chooser | K1 fail / K4 success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reacher | 88%@K1.00 | 86%@K4.00 | 88%@K1.06 to K2.32 | 92%@K1.12 | 2 / 50 |
| Cube single | 78%@K1.00 | 77.3%@K4.00 | 77.3%@K2.72 to K2.96 | 80.7%@K1.08 | 4 / 150 |
| Cube double | 72%@K1.00 | 70%@K4.00 | 72%@K1.00 to K2.62 | 72%@K1.00 | 0 / 50 |
| Cube triple | 70%@K1.00 | 78%@K4.00 | 76%@K2.32 | 82%@K1.36 | 6 / 50 |

Core conclusion:

Raw latent MSE is a reasonable v0 signal. On cube-triple, it improves
`70%@K1` to `76%@K2.32`, recovering much of the fixed `K4` gain while using
less average compute. It is not weak; it is incomplete.

Current figure artifact:

- `analysis/paper1_figures/png_direct/raw_mse_tolerance_pareto.png`

### Cube-Triple Threshold Sweep

| Policy | Success | Mean K | Selected K4 count | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Fixed K1 | 70% | 1.00 | 0 / 50 | Shallow recurrent depth |
| Raw latent MSE, tolerance 0 | 76% | 2.32 | 22 / 50 | Recovers part of K4 gain |
| Raw latent MSE, tolerance 0.001 | 74% | 1.96 | 16 / 50 | Less compute, lower success |
| Raw latent MSE, tolerance 0.003 | 72% | 1.54 | 9 / 50 | Too conservative |
| Fixed K4 | 78% | 4.00 | 50 / 50 | Stronger but expensive |

Outcome split:

| Category | Count |
| --- | ---: |
| K1 fails, K4 succeeds | 6 |
| K1 succeeds, K4 fails | 2 |
| Both succeed | 33 |
| Both fail | 9 |

Core conclusion:

The gap between raw MSE and the hindsight `K1/K4` chooser is the most important
analysis signal. It says useful dynamic-K structure exists, but raw latent MSE
does not perfectly identify it.

Current figure artifacts:

- `analysis/paper1_figures/png_direct/k1_k4_outcome_split.png`
- `analysis/paper1_figures/png_direct/raw_mse_precision_recall_failure.png`

## Mechanistic Analysis

### Analysis 1: Is the Failure Just Global Latent Smoothing?

Hypothesis:

Deeper recurrent refinement might reduce raw latent MSE by globally smoothing
the latent representation, destroying contact-relevant detail.

Result:

This hypothesis is not supported by the first analysis pass. Across reacher,
cube-single, cube-double, and cube-triple, `K4` and `K1` have nearly identical
global spectrum and state-probe behavior.

Key facts:

- `K4/K1` entropy-rank ratio is essentially `1.000` across the analyzed tasks.
- Cube-single block-position probe stays `R2=0.991 -> 0.991`.
- Cube-double block-position probe stays `R2=0.946 -> 0.946`.
- Cube-triple block-position probe stays `R2=0.902 -> 0.902`.
- Category-level probe MSE does not cleanly separate helped and hurt episodes.

Interpretation:

The raw-MSE failure is not a simple global latent-collapse story. The more
likely issue is local planner alignment: small latent changes can alter CEM
elite ranking or selected actions without changing global spectrum or simple
linear state probes.

Current figure artifacts:

- `analysis/k_smoothing_20260622/figures/spectrum_k1_vs_k4_scatter.png`
- `analysis/k_smoothing_20260622/figures/probe_r2_k1_vs_k4_scatter.png`
- `analysis/k_smoothing_20260622/figures/category_probe_mse_k1_vs_k4_scatter.png`

### Analysis 2: CEM Candidate-Ranking Stability

This is the next required analysis.

Question:

Does extra `K` change the ordering of candidate action sequences in the
planner?

Why it matters:

The planner does not care about latent MSE directly. It cares about whether the
terminal latent cost ranks action candidates correctly enough for CEM/MPC to
select a good action.

Planned metrics:

- Top-elite overlap between `K1` and deeper `K`.
- Kendall tau rank correlation between terminal costs.
- Whether the selected action changes.
- Whether `K1 fail / K4 success` episodes show ranking correction at larger
  `K`.
- Whether `K1 success / K4 fail` episodes show harmful ranking shifts.

Target conclusion:

Raw latent MSE is useful because it correlates with transition-improvement
pressure, but it is incomplete because planner benefit depends on candidate
ranking and selected actions.

## Section-by-Section Paper Outline

### 1. Introduction

Core argument:

Latent planning compute is currently allocated mostly through candidate count,
CEM iterations, or horizon length. Recurrent transition refinement depth `K` is
another compute axis, and it may be especially useful for contact-rich
transitions.

Key figure:

Fig. 1: Robot motivation cartoon.

Figure design:

- Left: free-space reaching, labeled "easy transition, K=1".
- Right: contact / lifting / multi-object interaction, labeled "refine more".
- Bottom: CEM rollout with transition arrows annotated by different `K`.

### 2. Background: Latent World-Model Planning

Core argument:

LeWM encodes visual observations and goals into latent space, rolls out
candidate action sequences, and optimizes terminal goal distance with CEM. We
keep the planner fixed and study transition depth.

Key figure:

Fig. 2: LeWM-style latent planner with `K` highlighted inside the transition
predictor.

### 3. Method: Recurrent Transition Refinement

Core argument:

The transition predictor can be run for multiple weight-tied refinement steps.
Fixed `K` applies the same depth everywhere. Dynamic allocation asks when an
imagined transition deserves deeper refinement.

Method pieces:

- Same encoder / action space / planner objective as LeWM-style planning.
- Recurrent transition predictor produces `z_hat^(1) ... z_hat^(K)`.
- Fixed-depth evaluation runs `K1/K2/K3/K4`.
- Raw latent-MSE analysis checks when deeper predictions reduce latent error
  enough to justify larger `K`.

Key figure:

Fig. 3: Recurrent transition cell and raw-MSE allocation rule.

### 4. Main Results: Does K Matter?

Core argument:

Fixed-depth results establish that `K` is a meaningful axis. Cube-triple gives
the cleanest evidence: `K1=70%`, `K2=76%`, `K3=76%`, `K4=78%`.

Key table:

Table 1: LeWM baseline plus fixed TTJepa `K1/K2/K3/K4`.

### 5. Raw Latent-MSE Dynamic K

Core argument:

Raw latent MSE is a clean first rule. It improves cube-triple from `70%@K1` to
`76%@K2.32`, close to `78%@K4`, but with lower average `K`.

Key table:

Table 2: Raw-MSE dynamic-K results across reacher, cube-single, cube-double,
and cube-triple.

Key figure:

Fig. 4: Success vs mean `K` Pareto.

### 6. Where Does Raw MSE Fail?

Core argument:

Raw MSE has signal, but it does not perfectly identify which episodes benefit
from deeper `K`. Hindsight `K1/K4` selection shows remaining headroom:
cube-triple can reach `82%@K1.36`.

Key figures:

- Fig. 5: `K1/K4` outcome split.
- Fig. 6: Precision / recall of raw-MSE selection over helped and hurt cases.

### 7. Mechanistic Analysis

Core argument:

The failure is not explained by global latent smoothing. Spectrum and linear
probe metrics stay near the `K1=K4` diagonal. The next explanation to test is
planner alignment through CEM candidate ranking.

Key figures:

- Fig. 7: Spectrum `K1` vs `K4` scatter.
- Fig. 8: State-probe `K1` vs `K4` scatter.
- Fig. 9: Category probe MSE scatter.
- Fig. 10: CEM ranking stability by outcome category.

### 8. Discussion

Core argument:

The paper's clean contribution is not "bigger K always wins." It is:

> Transition depth is a real compute axis; raw latent MSE can allocate it
> partially; the remaining gap reveals that latent prediction error and planner
> benefit are related but not identical.

## Recommended Figure Order

1. Motivation: robot free-space vs contact-rich transition compute.
2. Method: latent planner with recurrent transition depth `K`.
3. Main results table or grouped bar: LeWM / `K1/K2/K3/K4`.
4. Raw-MSE success vs mean-`K` Pareto.
5. `K1/K4` outcome split.
6. Raw-MSE precision / recall failure analysis.
7. Spectrum `K1` vs `K4` scatter.
8. State-probe `K1` vs `K4` scatter.
9. Category probe MSE `K1` vs `K4` scatter.
10. CEM candidate-ranking stability.

## Short Abstract Draft

Latent world-model planners typically allocate test-time compute by increasing
the number of sampled action sequences, optimizer iterations, or rollout
horizon. We study a complementary axis: the recurrent refinement depth used for
each imagined latent transition. Using TTJepa, a recurrent transition predictor
built on a LeWM-style latent planner, we evaluate fixed-depth rollouts and a
simple raw latent-error allocation analysis. On visual cube-triple, fixed
transition depth improves success from `70%` at `K1` to `78%` at `K4`, while
raw latent-MSE allocation reaches `76%` at mean `K=2.32`. Across four tasks, we
show that large `K` is not universally useful, motivating conditional
allocation. We further analyze the failure modes of raw latent MSE: a hindsight
`K1/K4` chooser reaches `82%` at mean `K=1.36`, while spectrum and state-probe
analyses show no broad global latent-collapse signature. These results identify
transition refinement depth as a meaningful compute axis in latent
world-model planning and show that raw latent prediction error is useful but
not fully aligned with planner benefit.

## ICLR-Strength Checklist

Required before submission:

- Multi-seed raw-MSE dynamic-K validation.
- Wall-clock latency and recurrent transition-call counts.
- CEM ranking stability analysis.
- Cleaner main figures with consistent style.
- Clear deployability wording for the raw-MSE rule.

Strongest current story:

1. `K` is a real latent-planning compute axis.
2. Cube-triple proves deeper refinement can matter.
3. Raw MSE gets a non-trivial dynamic-compute win.
4. The gap to hindsight selection gives a clean analysis problem.
5. Global smoothing is not enough to explain the gap.
6. Planner ranking is the next mechanistic explanation.
