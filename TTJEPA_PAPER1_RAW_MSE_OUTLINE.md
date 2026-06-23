# TTJepa Paper 1 Outline: Raw Latent MSE Dynamic K

Working title:

When Should a Latent Planner Refine? Dynamic Transition Depth via Raw Latent Error

One-sentence thesis:

Latent world-model planning usually spends test-time compute on CEM sampling width, optimizer iterations, or rollout horizon. This paper studies another axis: recurrent transition refinement depth K, and asks whether a simple raw latent prediction-error signal can decide when extra refinement is worth paying for.

## Paper Scope

This paper only uses raw latent MSE as the dynamic-K method.

Included:

- Fixed-depth recurrent TTJepa: K1/K2/K3/K4.
- Raw latent MSE dynamic K.
- Hindsight K1/K4 chooser as an upper-bound diagnostic.
- Failure analysis of raw MSE.
- Latent spectrum and state-probe analysis.
- CEM candidate-ranking analysis as the next required figure.

Excluded from main paper:

- Learned continue-head / joint marginal-depth selector.
- Planner-feature diagnostic selector.
- Whitened and probe-weighted halt-label variants.
- The rel0005 80% training-time regularization lead.

All excluded experiments remain in TTJEPA_EXPERIMENT_RESULTS.md.

## Core Contribution Claims

1. We identify transition refinement depth K as a test-time compute axis inside latent world-model planning.
2. We show that fixed K changes planning success, especially on cube-triple, but large K is not uniformly useful.
3. We propose raw latent MSE stopping as a simple dynamic-K rule.
4. We show raw MSE is a real signal: on cube-triple it improves from 70%@K1 to 76%@K2.32, recovering much of fixed K4's 78% success at lower average compute.
5. We analyze why raw MSE is incomplete: it does not perfectly align with planner benefit, and global latent spectrum/probe metrics do not explain the failure.
6. We identify CEM candidate ranking stability as the key next mechanism for understanding when extra K is useful.

## Section 1: Introduction

Core argument:

Manipulation planning has uneven transition difficulty. Free-space reaching is often easy; contact-rich interaction, grasping, object sliding, occlusion, and multi-object binding are harder. A latent planner should not use the same imagined-transition compute everywhere.

What to say:

In latent MPC/CEM, test-time compute is usually allocated to:

- number of candidate action sequences N,
- number of CEM iterations I,
- rollout horizon H.

This paper studies a different axis:

- recurrent refinement depth K inside each imagined latent transition.

Main figure:

Fig. 1: Motivation cartoon.

Figure design:

- Left: robot moving hand through free space toward an object. Label: "easy transition, K=1".
- Right: robot making contact / lifting / manipulating object. Label: "contact-rich transition, larger K".
- Bottom: CEM rollout with several transition arrows, each annotated with K1 or K4.

What the figure proves:

The paper is not about generic robot reasoning. It is about where to spend compute inside latent dynamics prediction.

## Section 2: Background: Latent World-Model Planning

Core argument:

LeWM-style planners encode visual state and goal into latent space, roll out candidate action sequences, and optimize terminal goal-matching cost with CEM. We keep this planner fixed and study only the transition predictor's refinement depth.

What to say:

Pipeline:

1. Encode current observation and goal into latent representations.
2. Sample action sequences.
3. Roll each candidate through a latent dynamics model.
4. Score terminal latent distance to goal.
5. Use CEM/MPC to choose the next action.

Main figure:

Fig. 2: Latent planner schematic.

Figure design:

- Observation and goal encoder.
- Latent transition predictor.
- CEM candidate rollouts.
- Terminal latent goal cost.
- Highlight K as a new compute axis inside each transition.

What the figure proves:

The method changes the imagined transition computation, not the action space, dataset, encoder target, or CEM objective.

## Section 3: Method: Recurrent Transition Refinement and Raw-MSE Dynamic K

Core argument:

The transition predictor can be run for multiple recurrent refinement steps. Fixed K applies the same depth everywhere; dynamic K uses raw latent prediction improvement to decide whether deeper refinement is useful.

Fixed-depth setup:

- K1: one recurrent refinement step.
- K2/K3/K4: more transition refinement steps.
- Used as sanity checks to establish whether depth matters.

Raw-MSE dynamic rule:

- Compare shallow and deeper latent prediction error.
- Continue/refine when deeper K improves raw latent MSE enough.
- Stop early when raw latent MSE suggests extra refinement is not worth paying for.

Main figure:

Fig. 3: Recurrent refinement cell and raw-MSE stopping rule.

Figure design:

- z_t, action, context/goal enter recurrent predictor.
- Predictions z_hat^(1), z_hat^(2), z_hat^(3), z_hat^(4).
- Raw latent MSE improvement score is measured across depth.
- Decision: stay at K1 or pay for K4.

What the figure proves:

Raw MSE is simple, model-internal, and does not require a separate learned selector, planner features, task-specific probes, or extra labels.

## Section 4: Main Results: Does K Matter?

Core argument:

K is a real test-time compute axis because fixed-depth changes success. But fixed larger K is not universally better, so dynamic allocation is necessary.

Main table:

Table 1: LeWM baseline and fixed-depth TTJepa.

Current working numbers:

| Dataset / run | LeWM baseline | Fixed K1 | Fixed K2 | Fixed K3 | Fixed K4 | Observation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reacher seed42 | 80% | 88% | n/a | n/a | 86% | K4 worse than K1 |
| Cube single seed42 | 72% | 80% | n/a | n/a | 78% | K4 slightly lower |
| Cube single seed43 | 72% | 88% | n/a | n/a | 90% | K4 improves by 2 points |
| Cube single seed44 | 72% | 66% | n/a | n/a | 64% | K4 slightly lower |
| Cube single 3-seed avg | 72% | 78% | n/a | n/a | 77.3% | K4 gain unstable |
| Cube single original rerun | 72% | 80% | 76% | 78% | 78% | K1 best, K3/K4 recover |
| Cube double original rerun | 66% | 72% | 70% | 68% | 70% | extra depth does not help |
| Cube triple original | 74% | 70% | 76% | 76% | 78% | clearest K gain |

Current artifact:

- analysis/paper1_figures/png_direct/main_success_vs_lewm.png

What this section proves:

K matters, especially on cube-triple, but the correct claim is not "larger K is always better." The correct claim is "transition depth changes success and must be allocated conditionally."

## Section 5: Raw Latent MSE Dynamic K

Core argument:

Raw latent MSE is a reasonable and effective first dynamic-K rule. It is not perfect, but it is not weak: it recovers a substantial part of the fixed-K4 gain on cube-triple at lower mean K.

Main table:

Table 2: Raw-MSE dynamic K across datasets.

| Dataset | Fixed K1 | Fixed K4 | Best raw-MSE dynamic K | Hindsight K1/K4 chooser | K1 fail / K4 success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reacher | 88%@K1.00 | 86%@K4.00 | 88%@K1.06 to K2.32 | 92%@K1.12 | 2 / 50 |
| Cube single | 78%@K1.00 | 77.3%@K4.00 | 77.3%@K2.72 to K2.96 | 80.7%@K1.08 | 4 / 150 |
| Cube double | 72%@K1.00 | 70%@K4.00 | 72%@K1.00 to K2.62 | 72%@K1.00 | 0 / 50 |
| Cube triple | 70%@K1.00 | 78%@K4.00 | 76%@K2.32 | 82%@K1.36 | 6 / 50 |

Main figure:

Fig. 4: Success vs mean K Pareto for raw-MSE thresholds.

Current artifact:

- analysis/paper1_figures/png_direct/raw_mse_tolerance_pareto.png

What the figure proves:

Raw latent MSE gives a real success/compute tradeoff. On cube-triple, it moves the planner from 70%@K1 to 76%@K2.32, while fixed K4 is 78%@K4.00.

## Section 6: Failure Analysis: Where Does Raw MSE Fail?

Core argument:

Raw MSE has signal, but it does not perfectly select the episodes where deeper K changes planning success. The hindsight K1/K4 chooser shows there is remaining dynamic-K headroom.

Main observations:

- Cube-triple has 6 episodes where K1 fails and K4 succeeds.
- It also has 2 episodes where K1 succeeds and K4 fails.
- Hindsight K1/K4 chooser reaches 82%@K1.36.
- Raw MSE tolerance 0 reaches 76%@K2.32.

Main figures:

Fig. 5: K1/K4 outcome split.

Current artifact:

- analysis/paper1_figures/png_direct/k1_k4_outcome_split.png

Fig. 6: Raw-MSE precision/recall or selected-case analysis.

Current artifact:

- analysis/paper1_figures/png_direct/raw_mse_precision_recall_failure.png

What these figures prove:

Raw MSE is useful but not planner-perfect. It sometimes spends compute on cases that do not need it, and it misses some cases where extra K changes success.

## Section 7: Mechanistic Analysis: Is the Failure Just Global Latent Smoothing?

Core argument:

A natural hypothesis is that deeper K globally smooths or collapses the latent representation, removing task-relevant details. We tested this, and the result is more subtle: global spectrum and simple state probes are almost unchanged from K1 to K4.

Completed analyses:

1. Latent spectrum / effective rank by K.
2. Linear state-probe quality by K.
3. Category-level probe MSE for easy, helped, hurt, and hard episodes.

Main figures:

Fig. 7: K1 vs K4 latent spectrum scatter.

Current artifact:

- analysis/k_smoothing_20260622/figures/spectrum_k1_vs_k4_scatter.png

Fig. 8: K1 vs K4 state-probe R2 scatter.

Current artifact:

- analysis/k_smoothing_20260622/figures/probe_r2_k1_vs_k4_scatter.png

Fig. 9: K1 vs K4 category probe MSE scatter.

Current artifact:

- analysis/k_smoothing_20260622/figures/category_probe_mse_k1_vs_k4_scatter.png

Key empirical facts:

- K4/K1 entropy-rank ratio is essentially 1.000 across reacher, cube-single, cube-double, and cube-triple.
- Cube-single block position R2 stays 0.991 -> 0.991.
- Cube-double block position R2 stays 0.946 -> 0.946.
- Cube-triple block position R2 stays 0.902 -> 0.902.

What this section proves:

Raw-MSE failure is not well explained by a broad global latent-rank collapse. The more likely issue is local planner alignment: small imagined-transition changes can affect CEM elite ranking or selected action without changing global spectrum or simple linear probes.

## Section 8: Required Next Mechanistic Figure: CEM Candidate-Ranking Stability

Core argument:

The decisive question is whether extra K changes the ordering of candidate action sequences in the planner. If raw latent MSE improves but CEM ranking does not improve, the extra refinement is not useful planning compute.

Planned analysis:

For the same sampled CEM action candidates, compare K1/K2/K3/K4 rollouts.

Metrics:

- Top-elite overlap between depths.
- Kendall tau rank correlation between terminal costs.
- Whether the selected action changes.
- Whether depth-helped episodes show ranking correction at larger K.
- Whether depth-hurt episodes show harmful ranking shifts.

Main figure to add:

Fig. 10: CEM ranking stability by depth and outcome category.

Figure design:

- Panel A: elite-set overlap K1 vs K4 for easy/helped/hurt/hard episodes.
- Panel B: Kendall tau K1 vs K4 for the same categories.
- Panel C: selected-action changed or unchanged.

What the figure should prove:

The reason raw MSE is incomplete is planner alignment, not simply latent reconstruction quality.

## Section 9: Discussion and Limitations

Core argument:

The safe claim is not that larger K is always better. The safe claim is that K is a real transition-level compute axis, and raw latent MSE is a simple rule that partially allocates this compute but leaves planner-alignment headroom.

Claims to make:

- K is a meaningful compute axis in latent planning.
- Raw latent MSE can recover useful depth gains in contact-heavy cube-triple.
- The method is simple and does not require a learned selector.
- Failure analysis is part of the contribution: raw latent error and planner benefit are not the same.

Claims to avoid:

- Do not claim K4 is universally better.
- Do not claim raw MSE is optimal.
- Do not claim global latent collapse explains everything.
- Do not use learned-selector or joint-depth training results as Paper 1 evidence.

Remaining requirements for a strong ICLR submission:

- Multi-seed raw-MSE validation.
- Wall-clock latency and recurrent transition-call count.
- CEM candidate-ranking stability analysis.
- Cleaner figure set with one consistent visual style.

## Main Figure Order

1. Fig. 1: Motivation cartoon: free-space vs contact-rich transition compute.
2. Fig. 2: Method schematic: latent planner with recurrent transition depth K.
3. Fig. 3: Fixed-depth result table or grouped bar: LeWM / K1 / K2 / K3 / K4.
4. Fig. 4: Raw-MSE success vs mean-K Pareto.
5. Fig. 5: Hindsight K1/K4 outcome split.
6. Fig. 6: Raw-MSE precision/recall failure analysis.
7. Fig. 7: Spectrum K1-vs-K4 scatter.
8. Fig. 8: State-probe K1-vs-K4 scatter.
9. Fig. 9: Category probe MSE K1-vs-K4 scatter.
10. Fig. 10: CEM ranking stability, to be added.

## Short Abstract Draft

Latent world-model planners typically allocate test-time compute by increasing the number of sampled action sequences, optimizer iterations, or rollout horizon. We study a complementary axis: the recurrent refinement depth used for each imagined latent transition. Using TTJepa, a recurrent transition predictor built on a LeWM-style latent planner, we evaluate fixed-depth rollouts and a simple dynamic rule based on raw latent prediction error. On visual cube-triple, fixed transition depth improves success from 70% at K1 to 78% at K4, while raw latent-MSE stopping reaches 76% at mean K=2.32. Across four tasks, we show that large K is not universally useful, motivating dynamic allocation. We further analyze the failure modes of raw latent MSE: a hindsight K1/K4 chooser reaches 82% at mean K=1.36, while spectrum and state-probe analyses show no broad global latent-collapse signature. These results identify transition refinement depth as a meaningful test-time compute axis and show that raw latent prediction error is a useful but incomplete proxy for planner benefit.
