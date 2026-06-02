# RecJEPA Future UIH Experiments

## Core Claim

The paper claim we need to prove is:

> A JEPA-style predictor can learn a user-specific future UIH representation, and this predicted future UIH improves a listwise future-engagement ranking task beyond same-capacity rank-only baselines.

Current logged-slate reaction framing:

> JEPA should be evaluated as an incremental world-modeling objective on top of a strong supervised reaction predictor, not as a standalone model that must beat every baseline by itself.

Reaction target semantics:

- One article/item can have many different reactions across users, slates, positions, and times.
- Therefore the reaction target is indexed by `(sample_id, displayed_rank)`, not by `article_id`.
- `click`, `read`, `scroll`, `next_read`, and `next_scroll` are multi-signal labels; they are not mutually exclusive classes.
- Missingness is part of the reaction event. A missing scroll value is not the same thing as an observed zero-scroll reaction.

For the logged-slate task, the decisive comparison is therefore:

- Baseline: user history + item + position + logged slate context, trained only with supervised reaction heads.
- Treatment: same model and same inputs, plus JEPA reaction-latent prediction loss and/or latent adapter features.
- Success: treatment improves full-label reaction prediction metrics, especially click LogLoss/calibration, read/scroll/gain regression, and slate-level gain nDCG.

The decisive setting is the full-label offline setting: all available logged future-engagement labels are used to train the ranker. Low-label experiments are useful diagnostics, but they are not the main claim.

This means a run is not considered successful just because a larger candidate-conditioned rank head beats a small baseline. The decisive comparison is:

- Same input: past/source UIH + shuffled future candidate set.
- Same ranking architecture and capacity.
- Control: rank-only listwise training, no future UIH event-set objective.
- Treatment: same rank head plus JEPA future UIH event-set losses.
- Success: treatment improves test listwise metrics, especially nDCG@10 and MRR, while also showing better future-event diagnostics and lower slot collapse.

## Actual Data Source and Split Construction

All EB-NeRD small experiments in this note use the cached sample file:

```text
outputs/ebnerd_small_qwen_lgbm/samples.pkl
```

That cache was built from EB-NeRD small:

```text
data/ebnerd/ebnerd_small/
  train/
    behaviors.parquet
    history.parquet
  validation/
    behaviors.parquet
    history.parquet
  articles.parquet
```

The cache contains 346,369 impression-level samples:

| cached split | source EB-NeRD split | samples | time range |
|---|---|---:|---|
| `jepa_train` | `train` | 141,877 | 2023-05-18 07:00:01 -> 2023-05-22 20:49:59 |
| `jepa_val` | `train` | 21,285 | 2023-05-23 20:50:02 -> 2023-05-24 17:01:00 |
| `ranker_train` | `validation` | 109,923 | 2023-05-27 05:01:01 -> 2023-05-30 08:16:37 |
| `ranker_val` | `validation` | 36,642 | 2023-05-30 08:16:39 -> 2023-05-31 07:14:57 |
| `ranker_test` | `validation` | 36,642 | 2023-05-31 07:14:58 -> 2023-06-01 06:59:59 |

Important:

- `ranker_train`, `ranker_val`, and `ranker_test` are not random splits.
- They are consecutive time blocks from the original EB-NeRD small `validation` split, approximately `60% / 20% / 20%`.
- The official EB-NeRD testset is not used in these offline experiments because it does not provide labels for this analysis.
- The latest logged-slate reaction experiments use only `ranker_train`, `ranker_val`, and `ranker_test`; `jepa_train/jepa_val` are left over from earlier pretrain-style experiments.

For full-small runs:

```text
MAX_SAMPLES_PER_SPLIT=0
```

means use all ranker samples:

```text
ranker_train = 109,923
ranker_val   = 36,642
ranker_test  = 36,642
```

For smoke runs such as `1k` or `10k`, the script does not create a new random
split. It keeps the same cached split labels and takes the first `N` samples
encountered from each of `ranker_train`, `ranker_val`, and `ranker_test`.

In the latest logged-slate reaction task, one sample is one real logged
impression/slate:

```text
input:
  user history before the impression
  logged ordered slate = article_ids_inview in original display order
  article content embedding, position, user/profile features, slate context

target:
  per displayed item reaction
  click label from article_ids_clicked
  read_time / scroll_percentage / next_read_time / next_scroll_percentage
  not-clicked exposure items receive zero engagement targets
```

## Final Task Definition

- Unit: anchor query `(user, time t)`.
- Input: source UIH before/at `t` and shuffled future candidate set `C_future(t)`.
- History UIH: ordered by logged exposure time. History is a sequence because user state depends on temporal context.
- Future candidate set: unordered. All later exposed items in `(t, t + horizon]` are flattened, deduplicated, and deterministically shuffled by `sample_id`.
- Future event-set target: unordered. Positive/negative future events are deterministically shuffled before truncation, and set losses must not align `slot_id` to future event position.
- Label: future clicked/read item is positive; not-clicked future exposure is negative.
- Metric group: `sample_id`, not request/impression id.
- Metrics: AUC, MRR, Hit@1, nDCG@10, LogLoss.

Important leakage rule:

```text
past/history UIH: ordered sequence
future candidate UIH/items: shuffled set
future prediction target: event set, no chronological slot supervision
```

## Current Full Small Baseline Result

Run path:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_full_rankhead_noid/results.json
```

Setting:

- Dataset: EB-NeRD small.
- Target mode: `future_only`.
- Rank task: `future_engagement`.
- Source mode: `history_weighted`.
- Candidate candidates: shuffled future exposed item set.
- Item-id embedding: disabled.
- Split sizes: train `109923`, val `36642`, test `36642`.
- Candidate stats: `8753949` total rank candidates, `1065159` positives, average `47.78` candidates/query.

| model | AUC | MRR | Hit@1 | nDCG@10 | LogLoss |
|---|---:|---:|---:|---:|---:|
| S0 source-only linear ranker | 0.679991 | 0.451018 | 0.243725 | 0.406343 | 0.799489 |
| Oracle true future slots | 0.958294 | 1.000000 | 1.000000 | 0.981963 | 0.134848 |
| E3 JEPA rank-head | 0.701839 | 0.492481 | 0.292090 | 0.431311 | 0.755708 |
| R0 rank-head only, same architecture | 0.712829 | 0.485879 | 0.281550 | 0.431214 | 0.807927 |
| E3 predicted slots -> separate ranker | 0.648485 | 0.433033 | 0.229272 | 0.384537 | 0.632923 |

Interpretation:

- E3 and R0 both beat S0 by about `+0.025` nDCG@10.
- E3 does not yet clearly beat same-capacity R0 on nDCG@10 (`+0.00010` only).
- E3 has better MRR/Hit@1 than R0, but the nDCG gap is too small for the core claim.
- The separate slot-ranker path is weaker than S0, so the useful route is direct listwise scoring with slot-aware residual/gating.

## Future UIH Representation Diagnostics

| diagnostic | E3 JEPA | R0 rank-only |
|---|---:|---:|
| future event AUC | 0.573184 | 0.491520 |
| future event MRR | 0.429988 | 0.356554 |
| future event Recall@1 | 0.251798 | 0.183904 |
| future event Recall@3 | 0.487592 | 0.385646 |
| future event Recall@5 | 0.658594 | 0.565214 |
| slot pairwise cosine | 0.603603 | 0.999761 |
| slot-only candidate nDCG@10 | 0.317936 | 0.257841 |

Interpretation:

- JEPA losses clearly reduce collapse and improve future-event retrieval.
- Rank-only control collapses its slots almost completely.
- The missing piece is showing that this better future UIH representation improves the final listwise ranker beyond rank-only capacity.

## Current Predictor/Ranker Focus

The current bottleneck is split into two questions:

1. Can the predictor learn a strong user-specific future UIH event set?
2. Can the ranker use those predicted future UIH slots better than a same-capacity rank-only model?

Recent correction:

- History tokens are ordered using `past_shown` and a recency/position scalar.
- Future candidates are deterministic shuffled sets.
- Future event targets are now shuffled before truncation.
- `slotwise_event_nce_loss` and `slot_assignment_ce_loss` were changed from slot-to-event-position supervision into order-invariant set losses.
- The 1k h3 set-loss smoke test passed: event AUC `0.6185`, slot nDCG@10 `0.3510`, slot pairwise cosine `0.6360`.

This matters because the paper claim should be about future-state-aware set ranking, not reconstructing a logged future chronology.

## Latest Predictor Ablations

Small 10k, future-only, future engagement ranking:

| setting | horizon | event AUC | event MRR | slot cosine | slot nDCG@10 | rank-head nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| direct latent slots | 24h | 0.5623 | 0.4125 | 0.9943 | 0.2989 | 0.1970 |
| candidate-attention slots | 24h | 0.6319 | 0.5017 | 0.7703 | 0.3676 | 0.3902 |
| candidate-attention slots | 12h | 0.6310 | 0.4842 | 0.7423 | 0.3832 | 0.4051 |
| candidate-attention slots | 6h | 0.6388 | 0.4786 | 0.6808 | 0.4097 | 0.4352 |
| candidate-attention slots | 3h | 0.6509 | 0.4682 | 0.6335 | 0.4257 | 0.4527 |

Interpretation:

- Direct generation of raw future slots collapses or learns weak slots.
- Candidate-attention slots are much better because predicted future UIH has to interact with the shuffled future candidate set.
- Shorter horizon is easier and gives stronger slot ranking signal. The current best diagnostic recipe is 3h candidate-attention.

Full EB-NeRD small, 24h candidate-attention, seed 7:

| event AUC | slot nDCG@10 | rank-head MRR | rank-head nDCG@10 |
|---:|---:|---:|---:|
| 0.6630 | 0.4024 | 0.4825 | 0.4239 |

Interpretation:

- More data improves predictor quality versus 10k/24h.
- The predictor is no longer just averaging, but the full-label ranker comparison is still not solved.
- Next decisive check is the fixed separate LGBM slot-ranker path, because an earlier run accidentally trained the slot ranker with zero train-slot features when `--skip-train-eval` was enabled.

## Ordered-History / Unordered-Future 10k h3 Control

Run paths:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_10k_h3_e3_lgbm_slots_setloss/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_10k_h3_r0_lgbm_slots_setloss/results.json
```

Setting:

- `max_samples_per_split=10000`
- `future_horizon_hours=3`
- history UIH is ordered
- future candidate set is shuffled
- future event targets are shuffled before truncation
- future set losses are order-invariant
- same S0 LGBM baseline in both runs

Predictor diagnostics on test:

| model | event AUC | event MRR | slot nDCG@10 | rank-head MRR | rank-head nDCG@10 |
|---|---:|---:|---:|---:|---:|
| E3 future-UIH JEPA | 0.656012 | 0.484462 | 0.438482 | 0.475412 | 0.465898 |
| R0 rank-only control | 0.628805 | 0.442485 | 0.407140 | 0.418463 | 0.430281 |

LGBM ranker with and without predicted slot features:

| model | AUC | MRR | Hit@1 | nDCG@5 | nDCG@10 | LogLoss |
|---|---:|---:|---:|---:|---:|---:|
| S0 source-only LGBM | 0.711364 | 0.468695 | 0.267245 | 0.391645 | 0.473987 | 0.550398 |
| P_E3 predicted future slots | 0.711254 | 0.479367 | 0.282084 | 0.393337 | 0.474158 | 0.455105 |
| P_R0 rank-only slots | 0.708888 | 0.463396 | 0.253197 | 0.388637 | 0.474245 | 0.475321 |

Interpretation:

- Predictor-side claim is now much stronger: E3 learns a better future UIH event-set representation than same-architecture R0 without using future order.
- Ranker-side result is mixed:
  - E3 improves MRR, Hit@1, nDCG@5, and LogLoss over both S0 and R0.
  - nDCG@10 is essentially tied across S0/E3/R0; R0 is microscopically highest on nDCG@10.
- Current bottleneck is not whether JEPA can learn future UIH; it is how to inject the predicted future UIH into a ranker objective that optimizes nDCG, not just top-1 / calibration.

## Next Experiment: Slot Residual/Gated Fusion

Purpose:

Test whether predicted future UIH slots add ranking signal on top of the same candidate-conditioned rank head.

Important correction:

The first predictor versions used only a pooled `source_emb` plus a trainable `user_id` embedding. That is weak user information for a user-specific future UIH predictor. The next version adds structured past UIH history tokens:

- past clicked article tokens
- past not-clicked exposure tokens
- past shown-only exposure tokens
- token type embeddings
- coarse within-history recency/position feature

Both R0 and E3 receive the same structured history tokens, so the comparison remains about the future-UIH JEPA objective rather than about extra user information.

Score:

```text
score(item) = rank_head(item | source UIH, candidate set)
            + residual(item, predicted future slots)
```

Residual features:

- `max_slot_score`
- `top3_slot_score_mean`
- `slot_score_entropy`

Comparisons:

| name | future UIH JEPA losses | rank loss | score mode | desired outcome |
|---|---|---|---|---|
| R0_residual_control | off | on | gated slot residual | same-capacity control |
| E3_residual_jepa | on | on | gated slot residual | should beat R0 on test nDCG@10/MRR |

Acceptance criterion:

- E3 residual test nDCG@10 should exceed R0 residual by a meaningful margin.
- E3 residual should keep better event diagnostics than R0.

## Ranker-Conditioned Reaction JEPA Smoke

Run path:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_ranker_conditioned_reaction_jepa_1k_h6/results.json
```

Setting:

- `max_samples_per_split=1000`
- horizon `6h`
- candidate list max length `30`
- train predictor order: oracle future-engagement order
- validation/test predictor order: baseline ranker order
- target: per-candidate frozen Qwen reaction text embedding, not pooled
- user information enabled: `user_id`, source UIH, ordered history tokens, history-summary profile tokens/scalars

Candidate summary:

| samples | candidates | positives | empty queries | truncated | avg candidates/query | avg positives/query |
|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 44709 | 6260 | 923 | 814 | 14.903 | 2.087 |

Predictor diagnostics on test:

| metric | value |
|---|---:|
| reaction MRR | 0.1505 |
| Recall@1 | 0.0438 |
| Recall@3 | 0.1212 |
| Recall@5 | 0.1948 |
| predicted-target cosine | 0.0948 |
| item-target cosine | 0.2652 |
| delta-target cosine | 0.0652 |
| predicted pairwise cosine | 0.7107 |

Reaction probe on test:

| feature | click AUC | click LogLoss | log-gain RMSE | scroll RMSE |
|---|---:|---:|---:|---:|
| item only | 0.5804 | 0.4264 | 0.4346 | 0.2255 |
| oracle reaction Qwen target | 0.6357 | 0.3837 | 0.4240 | 0.2306 |
| predicted reaction | 0.6906 | 0.3647 | 0.4311 | 0.2266 |

Future-engagement ranker on test:

| model | AUC | MRR | Hit@1 | nDCG@10 | engagement nDCG@10 | engagement gain@1 |
|---|---:|---:|---:|---:|---:|---:|
| S0 shuffled-order baseline ranker | 0.6972 | 0.4430 | 0.2380 | 0.4667 | 0.4578 | 0.5781 |
| R0 baseline-order ranker | 0.6800 | 0.4425 | 0.2319 | 0.4652 | 0.4564 | 0.5617 |
| B item-only reaction control | 0.7012 | 0.4480 | 0.2440 | 0.4668 | 0.4580 | 0.5920 |
| O oracle reaction target | 0.6897 | 0.4456 | 0.2395 | 0.4690 | 0.4603 | 0.5849 |
| P predicted reaction | 0.6896 | 0.4505 | 0.2470 | 0.4684 | 0.4593 | 0.6020 |

Interpretation:

- The new reaction-only task validates that user-conditioned predicted reaction contains useful reaction signal: the frozen probe on predicted reaction beats item-only and oracle reaction target on click AUC/LogLoss in this smoke.
- The final listwise ranker sees small gains: predicted reaction improves MRR, Hit@1, nDCG@10, engagement nDCG@10, and gain@1 over R0, but the margin is small and item-only reaction is also strong.
- The predicted embedding is not yet close to the Qwen reaction target (`pred_target_cosine < item_target_cosine`), so the current predictor is learning a useful proxy but not a faithful reaction latent.
- This is a good pipeline smoke, not yet paper-level evidence. The next fix should strengthen the JEPA objective around per-item reaction deltas and add a same-capacity direct-reaction predictor control.
- Oracle true future slots should remain far above both, preserving headroom.

## Smoke: Structured History Tokens + Gated Residual

Run paths:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_smoke_hist_residual_e3/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_smoke_hist_residual_r0/results.json
```

Setting:

- `max_samples_per_split=1000`
- `max_history_tokens=96`
- `score_mode=gated_slot_residual`
- Oracle/S0/slot-ranker skipped for speed.

| model | event AUC | event MRR | Recall@1 | Recall@3 | Recall@5 | slot cosine | slot nDCG@10 | rank-head MRR | rank-head nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 structured-history JEPA | 0.572080 | 0.408940 | 0.216883 | 0.488312 | 0.671429 | 0.998715 | 0.287092 | 0.401026 | 0.365670 |
| R0 structured-history rank-only | 0.516928 | 0.291638 | 0.092208 | 0.338961 | 0.541558 | 0.999832 | 0.222248 | 0.397949 | 0.363165 |

Interpretation:

- E3 improves future-event diagnostics and rank-head nDCG over R0 on smoke.
- The rank-head delta is tiny (`+0.0025` nDCG@10), so this is not enough for a paper claim.
- Slot collapse is still severe on 1k; full small is needed because earlier full E3 reduced slot cosine to about `0.60`.

## Full Small Seed 7: Structured History Tokens + Gated Residual

Run paths:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_full_hist_residual_e3_seed7/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_full_hist_residual_r0_seed7/results.json
```

| model | event AUC | event MRR | Recall@1 | Recall@3 | Recall@5 | slot cosine | slot nDCG@10 | rank-head MRR | rank-head nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 structured-history JEPA | 0.576746 | 0.425160 | 0.243761 | 0.484208 | 0.663106 | 0.649590 | 0.318878 | 0.484838 | 0.426012 |
| R0 structured-history rank-only | 0.492301 | 0.349117 | 0.173541 | 0.384588 | 0.553370 | 0.985114 | 0.253794 | 0.487520 | 0.427962 |

Interpretation:

- E3 clearly learns a better future UIH representation than R0.
- But E3 does not yet improve the final listwise rank-head over R0 on test nDCG@10 (`-0.00195`).
- This is not enough for the core paper claim.
- Next test: keep the rank-head score and explicitly evaluate `rank_score + lambda * slot_item_score`, with lambda tuned on validation. This directly tests whether the learned future UIH slots add residual ranking signal.

## Next Pivot: Controlled Label-Budget Listwise Ranking

The direct full-supervision setting is hard because a strong candidate-conditioned rank head can learn the future ranking task directly. This does not mean the dataset lacks labels. EB-NeRD has complete logged future behavior labels for the offline split. Instead, we use a controlled label-budget probe:

- The full offline labels exist.
- We intentionally restrict how many train queries contribute to the supervised listwise rank loss.
- The JEPA future-UIH event-set objective can still use all train queries as sequence-level auxiliary/self-supervised supervision.
- This tests whether future-UIH prediction improves sample efficiency and representation quality, not whether the raw dataset is missing labels.

Paper wording should be:

```text
We have full offline labels, but restrict the ranking supervision budget as a controlled probe.
If future-UIH JEPA learns useful user-state representations, it should improve ranking performance
when the supervised rank loss cannot fully absorb the future behavior signal.
```

This is a mechanism/supporting claim, not the final main claim. The main paper claim still needs to hold when the ranker uses all available labels:

```text
Given all logged future engagement labels for supervised ranking, predicted future-UIH representations
should still add incremental signal over an equally trained ranker baseline.
```

Therefore the controlled label-budget table below is supplementary evidence for representation quality and sample efficiency. The next main experiment is a full-label reranking test with a stronger `LightGBM LambdaRank` ranker:

- Baseline: full-label LGBM on source/candidate/meta features.
- Treatment: same full-label LGBM plus predicted future-UIH slot interaction features.
- Control: same full-label LGBM plus R0/rank-only slots.
- Accept: E3 slots improve over baseline and R0 slots under the same full-label training budget.

## Full-Label Main Probe: Joint Ranker + Future-UIH Auxiliary

Important correction:

- Low rank-label-budget runs are supporting/sample-efficiency evidence only.
- The main claim needs full-label ranking.
- The current strongest full-label evidence is the joint neural ranker, where E3 and R0 both use all rank labels and the same model interface, but E3 also has future-UIH event-set supervision.

Full small, seed 7, matched current code:

| model | future event AUC | slot cosine | rank-head MRR | rank-head nDCG@10 |
|---|---:|---:|---:|---:|
| E3 joint ranker + future-UIH aux | 0.582741 | 0.567903 | 0.492255 | 0.429493 |
| R0 rank-only joint ranker | 0.504780 | 0.999942 | 0.484480 | 0.424207 |

Interpretation:

- Under full labels, E3 improves nDCG@10 by `+0.00529` and MRR by `+0.00777` over a matched R0 rank-only model.
- E3 also learns non-collapsed future-UIH slots, while R0 slots collapse.
- This is directionally the main result we want, but it is not yet enough alone: older unmatched runs had R0 around `0.431`, so we need same-code multi-seed repeats before making a strong claim.

Full small, seed 13, matched current code:

| model | future event AUC | future event MRR | slot cosine | slot nDCG@10 | rank AUC | rank MRR | rank Hit@1 | rank nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 joint ranker + future-UIH aux | 0.568828 | 0.419272 | 0.540557 | 0.314826 | 0.704295 | 0.491004 | 0.286978 | 0.433631 |
| R0 rank-only joint ranker | 0.505379 | 0.369708 | 0.998601 | 0.260776 | 0.714753 | 0.483239 | 0.280175 | 0.428950 |
| E3 - R0 | +0.063449 | +0.049564 | -0.458044 | +0.054050 | -0.010457 | +0.007765 | +0.006803 | +0.004681 |

Interpretation:

- This directly matches the desired full-label setting: both models train with all available rank labels.
- E3 improves the listwise metrics we care about most for ranking quality: `+0.00468` nDCG@10, `+0.00776` MRR, and `+0.00680` Hit@1.
- E3 also strongly improves future-UIH diagnostics: future-event AUC `+0.06345` and slot-only nDCG@10 `+0.05405`; R0 slots remain collapsed.
- Rank AUC is lower for E3 (`-0.01046`), so the result should be described as a modest listwise-ranking improvement, not a universal pointwise classification improvement.
- Current full-label evidence is positive on seeds 7 and 13, but the margin is small. Seed 21 is the stability check below.

Full small, seed 21, matched current code:

| model | future event AUC | future event MRR | slot cosine | slot nDCG@10 | rank AUC | rank MRR | rank Hit@1 | rank nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 joint ranker + future-UIH aux | 0.573113 | 0.417506 | 0.565638 | 0.314489 | 0.706114 | 0.484605 | 0.282819 | 0.426727 |
| R0 rank-only joint ranker | 0.501590 | 0.379122 | 0.962300 | 0.265931 | 0.708776 | 0.490421 | 0.290574 | 0.430367 |
| E3 - R0 | +0.071522 | +0.038384 | -0.396662 | +0.048558 | -0.002662 | -0.005816 | -0.007755 | -0.003640 |

Interpretation:

- Seed 21 is the counterexample: E3 still learns a much better future-UIH representation, but the final rank head is worse than R0 on MRR/Hit@1/nDCG@10.
- This means the full-label ranking gain is not yet stable enough for the main paper claim.

Three-seed full-label summary:

| seed | delta future event AUC | delta rank MRR | delta rank nDCG@10 |
|---:|---:|---:|---:|
| 7 | +0.077961 | +0.007775 | +0.005286 |
| 13 | +0.063449 | +0.007765 | +0.004681 |
| 21 | +0.071522 | -0.005816 | -0.003640 |
| mean | +0.070977 | +0.003241 | +0.002109 |

Current conclusion:

- The JEPA objective reliably learns a non-collapsed future UIH representation under full labels.
- However, converting that representation into final listwise rank improvement is weak and seed-sensitive.
- We should not claim the full-label ranker result is solved yet. The next research step is to improve how future UIH is injected into the full-label ranker, not to retreat to the low-label setting as the main story.

## Full-Label LGBM Slot Injection Probe

Full-label `LightGBM LambdaRank` was added as a stronger reranker probe.

| model | nDCG@10 | MRR | note |
|---|---:|---:|---|
| S0 LGBM source/candidate/meta | 0.411093 | 0.455454 | full-label baseline |
| LGBM + R0 slots | 0.407974 | 0.451021 | collapsed slots do not help |
| LGBM + pure JEPA slots | 0.407846 | 0.450951 | predictor uses no rank loss; current compressed slot features do not help |
| LGBM + joint E3 slots | 0.409846 | 0.453992 | joint E3 slots still do not beat S0 LGBM |

Interpretation:

- Full-label LGBM does not yet benefit from the current compressed slot features.
- This means the full-label main claim should currently use the joint neural ranker result, not the post-hoc LGBM slot injection result.
- Next LGBM retry adds per-slot candidate similarity features instead of only `max/top3/entropy`, because the current three slot features appear too compressed.

## Supplementary Label-Budget Probe

This is not the main paper setting. It is a controlled probe for whether future-UIH prediction improves representation quality and sample efficiency when supervised rank loss is intentionally restricted.

- E3 uses all train queries for future UIH event-set prediction.
- E3 and R0 use the same limited fraction of train queries for supervised listwise ranking loss.
- R0 has no future UIH event-set objective.
- If JEPA has learned useful future UIH, E3 should beat R0 under the same controlled rank-label budget.

New control knob:

```text
--rank-supervision-fraction
```

Initial fractions to test:

- `0.01`
- `0.05`
- `0.10`
- `1.00`

Primary table:

| fraction | R0 nDCG@10 | E3 nDCG@10 | delta | R0 MRR | E3 MRR | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.374245 | 0.413866 | +0.039621 | 0.416300 | 0.469408 | +0.053107 |
| 0.05 | 0.383645 | 0.430106 | +0.046462 | 0.419158 | 0.484566 | +0.065408 |

## Full Small Seed 7: 1 Percent Label-Efficient Result

Run paths:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_full_lowlabel001_e3_seed7/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_full_lowlabel001_r0_seed7/results.json
```

Setting:

- `rank_supervision_fraction=0.01`
- `max_history_tokens=96`
- `score_mode=plain`
- `skip_source_slot_blend=true`
- JEPA event-set loss sees all train queries.
- Listwise rank loss sees only 1 percent of train queries.

| model | event AUC | event MRR | Recall@1 | Recall@3 | Recall@5 | slot cosine | slot nDCG@10 | rank-head MRR | rank-head nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 low-label JEPA | 0.586910 | 0.431901 | 0.250035 | 0.493902 | 0.673364 | 0.534513 | 0.327141 | 0.469408 | 0.413866 |
| R0 low-label rank-only | 0.500293 | 0.365299 | 0.191836 | 0.400592 | 0.572758 | 0.999931 | 0.258414 | 0.416300 | 0.374245 |

Interpretation:

- E3 improves test nDCG@10 by `+0.03962` and MRR by `+0.05311`.
- E3 has clearly stronger future-event prediction than R0 (`event AUC 0.5869` vs `0.5003`).
- R0 slots collapse (`slot cosine=0.99993`), while E3 maintains non-trivial slot diversity (`0.53451`).
- This supports the mechanism/sample-efficiency story, but does not replace the full-label main comparison.

## Full Small Seed 7: 5 Percent Label-Efficient Result

Run paths:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_full_lowlabel005_e3_seed7/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_full_lowlabel005_r0_seed7/results.json
```

Setting:

- `rank_supervision_fraction=0.05`
- `max_history_tokens=96`
- `score_mode=plain`
- `skip_source_slot_blend=true`
- JEPA event-set loss sees all train queries.
- Listwise rank loss sees only 5 percent of train queries.

| model | event AUC | event MRR | Recall@1 | Recall@3 | Recall@5 | slot cosine | slot nDCG@10 | rank-head MRR | rank-head nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 low-label JEPA | 0.588552 | 0.443051 | 0.259800 | 0.511598 | 0.684786 | 0.521412 | 0.334505 | 0.484566 | 0.430106 |
| R0 low-label rank-only | 0.478495 | 0.300231 | 0.126798 | 0.316624 | 0.493902 | 0.999958 | 0.231741 | 0.419158 | 0.383645 |

Interpretation:

- E3 improves test nDCG@10 by `+0.04646` and MRR by `+0.06541`.
- E3 also learns a non-collapsed future UIH representation (`slot cosine=0.52` vs R0 `1.00`).
- This is strong supplementary evidence that the future-UIH objective learns useful ranking signal.
- The core claim still depends on full-label E3 vs full-label R0 under matched capacity and multi-seed repeats.

## New Hypothesis: Add Explicit User Profile Information

Observation:

- The predictor already has a learned `user_id` token.
- But this is only a transductive identity embedding.
- It does not explicitly expose a stable user profile derived from pre-anchor behavior.
- The past UIH token sequence can encode this in principle, but the model may spend capacity recovering coarse user state before learning future UIH slots.

Patch added:

```text
--user-profile-mode history_summary
```

When enabled, every query gets extra pre-anchor user profile inputs:

- clicked-profile token: weighted raw-Qwen mean of historical clicked articles
- skipped-profile token: weighted raw-Qwen mean of historical not-clicked exposures
- shown-profile token: weighted raw-Qwen mean of historical shown-only exposures
- scalar user profile: log history counts, click/skip ratio, shown-only ratio, unique-click ratio

Important fairness constraint:

- E3 and R0 both receive the same user profile inputs.
- The only difference remains whether the model has JEPA future-UIH event-set supervision.
- This means any E3 gain still supports the claim that future UIH prediction adds useful listwise signal beyond the same user/candidate architecture.

Smoke result on 1k per split:

| model | event AUC | event MRR | Recall@1 | Recall@3 | Recall@5 | slot cosine | slot nDCG@10 | rank-head MRR | rank-head nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 + user profile | 0.594988 | 0.455723 | 0.270130 | 0.537662 | 0.702597 | 0.999326 | 0.306514 | 0.368239 | 0.330963 |
| R0 + user profile | 0.592481 | 0.403477 | 0.209091 | 0.471429 | 0.681818 | 0.997798 | 0.293931 | 0.349918 | 0.306818 |

Interpretation:

- Shape/loss/ranker path works with user profile tokens.
- On tiny 1k, E3 improves rank-head nDCG@10 by `+0.02415`.
- Slot cosine is still collapsed after one epoch, so this is only a pipeline smoke, not a paper result.
- Next: run full small with `rank_supervision_fraction` in `{0.01, 0.05, 0.10}` and compare E3 vs R0 under identical user-profile inputs.

## Full Small Seed 7: 5 Percent With Explicit User Profile

Run paths:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_full_lowlabel005_userprofile_e3_seed7/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_full_lowlabel005_userprofile_r0_seed7/results.json
```

Setting:

- `rank_supervision_fraction=0.05`
- `user_profile_mode=history_summary`
- `max_history_tokens=96`
- `score_mode=plain`
- E3 and R0 both receive the same user-id token, history tokens, and explicit user-profile tokens.

| model | event AUC | event MRR | Recall@1 | Recall@3 | Recall@5 | slot cosine | slot nDCG@10 | rank-head MRR | rank-head nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 + user profile | 0.581230 | 0.444637 | 0.265158 | 0.509800 | 0.680591 | 0.560959 | 0.329002 | 0.478964 | 0.424478 |
| R0 + user profile | 0.472346 | 0.313138 | 0.130076 | 0.345530 | 0.531232 | 0.999640 | 0.243415 | 0.432700 | 0.393062 |

Interpretation:

- With explicit user-profile information, E3 still improves nDCG@10 by `+0.03142` and MRR by `+0.04626`.
- This addresses the concern that the predictor lacked user information.
- The user-profile E3 result is slightly below the no-profile 5 percent E3 run (`0.42448` vs `0.43011` nDCG@10), so the explicit profile is best treated as a robustness/control ablation rather than the main recipe.
- The mechanism remains: JEPA future-UIH supervision gives non-collapsed slots and better listwise ranking under limited rank labels.

## Clean 10k Future-Engagement Ranking: Explicit h6 / Engagement nDCG

Purpose:

- Move away from hidden/implicit old horizon settings.
- Define a reproducible future-engagement listwise task.
- Test the core full-label claim: predicted future UIH slots can improve a listwise ranker.

Task:

- Input history is ordered pre-anchor UIH.
- Future candidate item set is unordered/shuffled.
- Rank unit is one anchor user-time query.
- Candidate set contains later exposed items in the next `6` hours.
- `max_events_per_impression=20`
- `max_rank_candidates=256`
- Train label is graded future engagement gain from click/read/scroll/recency.
- Report both binary click nDCG and graded `engagement_ndcg`.

Shared event summary:

```text
samples=30000
rank_candidates=561142
rank_positive=64313
avg_rank_candidates_per_sample=18.7047
```

Run paths:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_10k_h6_e3_lgbm_engmetric_clean/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_10k_h6_r0_rankonly_lgbm_engmetric_clean/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_10k_h6_e3_rankjoint_lgbm_engmetric_clean/results.json
```

Ranker test metrics:

| model | future-event AUC | slot nDCG@10 | slot engagement nDCG@10 | ranker MRR | Hit@1 | nDCG@10 | engagement nDCG@10 | mean gain@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 source-only LGBM | - | - | - | 0.478321 | 0.272346 | 0.472809 | 0.464199 | 0.651782 |
| P_E3 future-UIH slots | 0.628158 | 0.418919 | 0.411825 | 0.483366 | 0.280894 | 0.477440 | 0.468353 | 0.667090 |
| P_R0 rank-only slots | 0.473432 | 0.260559 | 0.256686 | 0.479651 | 0.270396 | 0.478669 | 0.470103 | 0.643566 |
| P_E3 + rank-head joint | 0.644209 | 0.430880 | 0.423302 | 0.480267 | 0.274145 | 0.467515 | 0.458695 | 0.661921 |

Interpretation:

- Positive full-label result: predicted E3 future-UIH slots improve over S0 on all main listwise metrics:
  - `nDCG@10`: `0.472809 -> 0.477440` (`+0.004631`)
  - `engagement_ndCG@10`: `0.464199 -> 0.468353` (`+0.004154`)
  - `MRR`: `0.478321 -> 0.483366`
  - `Hit@1`: `0.272346 -> 0.280894`
  - `mean_gain@1`: `0.651782 -> 0.667090`
- Mechanism check: E3 learns substantially better future UIH than rank-only R0:
  - future-event AUC `0.628158` vs `0.473432`
  - slot engagement nDCG@10 `0.411825` vs `0.256686`
- Caveat: under full rank labels, R0 rank-only slots slightly outperform E3 on `nDCG@10` and `engagement_ndCG@10`.
  - This means the full-label claim should be phrased as: future-UIH prediction learns useful/additive ranking features.
  - It should not yet be phrased as: JEPA future-UIH objective dominates a fully supervised rank-only representation.
- E3 + rank-head joint loss improves future-UIH metrics but hurts the downstream LGBM adapter, so the current bottleneck is not just adding rank supervision to the predictor. The adapter/objective alignment still needs work.

Current paper-safe takeaway:

```text
Predicted future UIH slots provide incremental signal for future-engagement listwise ranking,
and they are measurably future-state-aware. However, in the full-label setting, direct
rank-only supervision remains a strong competing control. The clearest JEPA advantage is
currently in mechanism metrics and low-label/sample-efficiency settings.
```

## Attention Ranker Baseline: Same h6 Future-Engagement Task

Motivation:

- LGBM is a useful engineering adapter, but a paper baseline should also include a neural attention ranker.
- The task is unchanged: ordered past UIH, unordered future candidate set, listwise future engagement ranking.
- The attention ranker uses:
  - ordered history self-attention,
  - candidate-aware target attention over history,
  - candidate-set self-attention,
  - optional future-slot attention.

References for baseline family:

- NRMS-style neural news recommendation: multi-head self-attention over clicked history.
- DIN-style candidate-aware interest attention: candidate attends to user behavior history.

### 1k Smoke

Run:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_attention_h6_1k_smoke_profile_slotattn/results.json
```

Setting:

- `max_samples_per_split=1000`
- `ranker_kind=attention`
- `user_profile_mode=history_summary`
- `init_slot_dot_scale=0.0`
- `epochs=3`
- no item-id embeddings

| model | MRR | Hit@1 | nDCG@10 | engagement nDCG@10 | mean gain@1 |
|---|---:|---:|---:|---:|---:|
| S0 attention | 0.405276 | 0.200301 | 0.424854 | 0.417271 | 0.476860 |
| S0 + E3 slots attention | 0.402232 | 0.191265 | 0.425082 | 0.417673 | 0.456997 |

Interpretation:

- Shape/loss/ranker path works.
- E3 slots give a tiny nDCG/engagement-nDCG gain, but MRR/Hit@1/mean-gain drop.
- This is a smoke test only.

### 10k Attention, 5 Epochs

Run paths:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_10k_h6_attention_e3_cached/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_10k_h6_attention_r0_cached/results.json
```

| model | MRR | Hit@1 | nDCG@10 | engagement nDCG@10 | mean gain@1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| S0 attention | 0.454678 | 0.248950 | 0.449394 | 0.441417 | 0.600926 | 0.687236 |
| S0 + E3 slots attention | 0.462164 | 0.265147 | 0.453150 | 0.445192 | 0.637862 | 0.693448 |
| S0 + R0 slots attention | 0.464228 | 0.267397 | 0.455978 | 0.447910 | 0.641468 | 0.694115 |

### 10k Attention, 10 Epochs

Run paths:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_10k_h6_attention10e_e3_cached/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_10k_h6_attention10e_r0_cached/results.json
```

| model | MRR | Hit@1 | nDCG@10 | engagement nDCG@10 | mean gain@1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| S0 attention | 0.454217 | 0.246551 | 0.451315 | 0.443624 | 0.596230 | 0.692827 |
| S0 + E3 slots attention | 0.466719 | 0.266647 | 0.457645 | 0.449833 | 0.645972 | 0.687366 |
| S0 + R0 slots attention | 0.469826 | 0.265297 | 0.462166 | 0.454157 | 0.639086 | 0.690332 |

Interpretation:

- Attention ranker confirms the main adapter finding:
  - E3 future-UIH slots improve S0 attention by `+0.00633` nDCG@10 and `+0.00621` engagement nDCG@10.
  - Mean gain@1 improves strongly: `0.59623 -> 0.64597`.
- R0 rank-only slots still slightly outperform E3 on nDCG/engagement nDCG.
- Attention S0 is currently weaker than LGBM S0 (`0.4513` vs `0.4728` nDCG@10), so it is a neural baseline/control, not yet a stronger baseline.
- Longer training from 5 to 10 epochs improves S0 only modestly, so the next ranker improvement should be architectural/training-objective, not just more epochs.

Current interpretation with attention included:

```text
Across LGBM and attention rankers, predicted future UIH slots consistently add signal to S0.
The remaining gap is not whether future UIH is useful; it is whether the JEPA future-UIH
objective can beat a direct rank-only latent control under full-label supervision.
```

## Next Target: Qwen-Encoded Reaction Event UIH

The previous E3 event target still represented an event as:

```text
raw Qwen article embedding + read/scroll/time-gap scalar weights
```

That is not expressive enough for the desired UIH target, because the user's concrete reaction is mostly outside the Qwen latent. The updated target is:

```text
reaction_event_text =
  article title/subtitle/category/topics/sentiment
  + event kind: future_clicked or future_not_clicked_exposure
  + clicked yes/no
  + read_time / scroll_percentage
  + next_read_time / next_scroll_percentage
  + session_id / device / candidate_count
  + time gap from anchor t

z_reaction_event = frozen Qwen(reaction_event_text)
future UIH target = unordered set of z_reaction_event
```

The task itself stays fixed:

- input: ordered past/source UIH plus shuffled future candidate set
- output: ranking over shuffled future candidate items
- E3 treatment: predicts Qwen-encoded future reaction-event slots
- R0 baseline/control: same architecture and same full rank labels, but no future-UIH event-set objective

New scripts:

```text
ebnerd_reaction_qwen_cache.py
run_ebnerd_h6_reaction_qwen_text_smoke.sh
run_ebnerd_h6_reaction_qwen_text_10k_pair.sh
```

The decisive comparison now treats R0 as a formal baseline:

```text
S0: source/candidate/meta ranker, no slots
O_event: true Qwen reaction-event future UIH slots
P_E3: predicted Qwen reaction-event future UIH slots
P_R0: rank-only same-architecture slots
```

Acceptance criterion:

```text
P_E3 should beat both S0 and P_R0 on listwise future-engagement metrics
while also showing better future-reaction-event retrieval and lower slot collapse.
```

### 1k Reaction-Qwen Smoke

Run paths:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_reaction_qwen_text_h6_1k/reaction_qwen_events.npz
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_1k_h6_reaction_qwen_e3/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_1k_h6_reaction_qwen_e3_joint/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_future_item_ranking_1k_h6_reaction_qwen_r0/results.json
```

Cache summary:

```text
num_samples=3000
ranker_train=1000, ranker_val=1000, ranker_test=1000
pos_events=6381
neg_events=74000
avg_pos_per_sample=2.127
avg_neg_per_sample=24.667
rank_candidates=56688
rank_positive=6260
avg_rank_candidates_per_sample=18.896
```

Test ranker results:

| model | AUC | MRR | Hit@1 | nDCG@10 | engagement nDCG@10 | mean gain@1 |
|---|---:|---:|---:|---:|---:|---:|
| S0 source/candidate/meta | 0.701333 | 0.469575 | 0.272590 | 0.446446 | 0.439448 | 0.662421 |
| O_event true Qwen reaction slots | 0.982888 | 0.984079 | 0.972892 | 0.971333 | 0.953289 | 2.412192 |
| P_E3 reaction slots | 0.695800 | 0.432746 | 0.213855 | 0.435259 | 0.428641 | 0.519589 |
| P_E3_joint reaction slots | 0.694619 | 0.461818 | 0.272590 | 0.444681 | 0.437301 | 0.660150 |
| P_R0 rank-only slots | 0.706809 | 0.464327 | 0.274096 | 0.457096 | 0.450102 | 0.683014 |

Predictor diagnostics:

| predictor | future-event AUC | future-event MRR | slot cosine | slot nDCG@10 | slot engagement nDCG@10 | rank-head nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| E3 reaction | 0.570158 | 0.395942 | 0.684999 | 0.360603 | 0.354832 | 0.368930 |
| E3_joint reaction | 0.574250 | 0.397479 | 0.684999 | 0.363095 | 0.357383 | 0.385280 |
| R0 rank-only | 0.624782 | 0.423717 | 0.684995 | 0.363848 | 0.357720 | 0.408423 |

Interpretation:

- The new Qwen reaction-event target is wired correctly and has a very high oracle ceiling.
  - `O_event` is far above S0 on all listwise metrics.
  - This means the target contains future ranking signal when used as true future slots.
- The current E3 losses do not yet learn that target well enough.
  - On this 1k smoke, E3 is below both S0 and R0.
  - Adding rank-head supervision in `E3_joint` closes some MRR/Hit@1 gap but still does not beat R0.
- R0 is now a formal baseline/control.
  - It uses the same architecture and rank labels but no future-UIH event-set loss.
  - Because R0 beats E3 in this smoke, the next research step is not another full-size run with the same loss; it is improving the reaction-event predictor objective/adapter.

Immediate next hypotheses:

```text
1. The reaction text target is semantically rich but item-dot scoring may be poorly aligned:
   Qwen(reaction event text) includes clicked/read/session tokens, while candidate item embeddings are pure article text.

2. Candidate KL from reaction slots to article embeddings may poison the event objective.
   Try E1/E2 without candidate KL, then add a learned reaction-to-item projection.

3. E3 should predict reaction-event slots and ranker should consume them through a learned cross-attention adapter,
   not only raw cosine/slot-dot features.
```

### Strict JEPA-Style Direct-Slot Follow-Up

The next run removes the candidate-mix output shortcut:

```text
context encoder input:
  ordered past UIH + shuffled future candidate set

predictor output:
  K raw future reaction slots

target encoder:
  frozen Qwen(reaction_event_text)

loss:
  event-set InfoNCE + BPR + assignment + diversity
  optional rank-head supervision for a matched full-label comparison

not allowed:
  predicted_slot = soft mix of candidate item embeddings
  candidate KL that forces reaction slots back toward pure article embeddings
```

R0 remains the matched rank-only baseline:

```text
R0_direct:
  same context encoder
  same raw slot queries/output path
  same rank-head supervision
  no future-reaction event-set loss
```

Script:

```text
run_ebnerd_h6_reaction_qwen_text_jepa_direct_1k.sh
```

## Ranker-Conditioned Reaction JEPA 10k

Run path:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_ranker_conditioned_reaction_jepa_10k_h6/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_ranker_conditioned_reaction_jepa_10k_h6/summary.json
```

Setting:

- EB-NeRD small, 10k samples per split: 30k user-time queries total.
- Fixed future candidate list length up to 30.
- History is ordered; future candidate order is produced by a baseline ranker for eval/test.
- Predictor train order uses oracle future-engagement order as teacher-forced list context.
- Target is not pooled: each candidate has a frozen Qwen reaction-event text embedding.
- Reaction target cache is `npy_dir` memmap with `_SUCCESS` marker:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_ranker_conditioned_reaction_jepa_10k_h6/reaction_candidate_targets
```

Candidate summary:

| samples | candidates | positives | empty queries | truncated | avg candidates/query | avg positives/query |
|---:|---:|---:|---:|---:|---:|---:|
| 30000 | 445497 | 64313 | 9023 | 7879 | 14.850 | 2.144 |

Predictor diagnostics on test:

| metric | value |
|---|---:|
| reaction MRR | 0.2503 |
| Recall@1 | 0.1170 |
| Recall@3 | 0.2508 |
| Recall@5 | 0.3567 |
| predicted-target cosine | 0.2394 |
| item-target cosine | 0.2671 |
| delta-target cosine | 0.3650 |
| predicted pairwise cosine | 0.5288 |

Reaction probe on test:

| feature | click AUC | click LogLoss | log-gain RMSE | scroll RMSE |
|---|---:|---:|---:|---:|
| item only | 0.5759 | 0.4224 | 0.4381 | 0.2190 |
| oracle reaction Qwen target | 0.6900 | 0.3723 | 0.4176 | 0.2170 |
| predicted reaction | 0.6832 | 0.3924 | 0.4280 | 0.2175 |

Future-engagement ranker on test:

| model | AUC | MRR | Hit@1 | nDCG@10 | engagement nDCG@10 | engagement gain@1 |
|---|---:|---:|---:|---:|---:|---:|
| S0 shuffled-order baseline ranker | 0.6732 | 0.4786 | 0.2809 | 0.4765 | 0.4678 | 0.6806 |
| R0 baseline-order ranker | 0.6903 | 0.4736 | 0.2705 | 0.4783 | 0.4694 | 0.6538 |
| R0 user-profile-only control | 0.6821 | 0.4819 | 0.2815 | 0.4787 | 0.4700 | 0.6860 |
| B item-only reaction control | 0.6965 | 0.4906 | 0.2900 | 0.4870 | 0.4779 | 0.7021 |
| O oracle reaction target | 0.6983 | 0.5830 | 0.4262 | 0.5545 | 0.5456 | 1.0204 |
| P predicted reaction | 0.6824 | 0.4856 | 0.2836 | 0.4830 | 0.4745 | 0.6896 |

Interpretation:

- 10k fixes the "1k may be too small" concern: the predictor now learns a real user-conditioned reaction signal.
  - Predicted reaction probe click AUC is `0.6832`, far above item-only `0.5759` and close to oracle reaction `0.6900`.
  - Delta-target cosine reaches `0.3650`, showing the model is learning a reaction delta beyond the article embedding.
- The listwise ranker can use the predicted reaction feature in the full-label setting.
  - `P_predicted_reaction` beats `R0_baseline_order` on MRR, Hit@1, nDCG@10, engagement nDCG@10, and engagement gain@1.
  - The absolute nDCG@10 gain is modest: `0.4830 - 0.4783 = +0.0047`.
- The oracle reaction target has large headroom.
  - `O_oracle_reaction` reaches nDCG@10 `0.5545` and engagement nDCG@10 `0.5456`.
  - This proves the per-candidate Qwen reaction UIH target is useful if predicted well.
- The item-only reaction control is still strong.
  - `B_item_only_reaction` beats `P_predicted_reaction` on test nDCG@10.
  - This means the current predicted reaction helps over R0, but the ranker still extracts a lot from item content alone.

Research conclusion:

```text
The ranker-conditioned reaction JEPA direction works at 10k scale:
user + ordered history + ordered candidate list can predict per-item future reaction latent,
and the predicted reaction latent improves full-label listwise future-engagement ranking over R0.

For paper-level evidence, the next improvement should focus on making P beat item-only controls,
not merely R0. The highest-value next changes are stronger reaction delta prediction,
better reaction-to-ranker adapter/cross-attention, and multi-seed 10k/full-small verification.
```

## Logged-Slate Reaction JEPA 10k

Run path:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_logged_slate_reaction_jepa_10k_maskfix/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_logged_slate_reaction_jepa_10k_maskfix/summary.json
```

Script:

```text
ebnerd_logged_slate_reaction_jepa.py
```

Task definition:

- Unit: one real logged EB-NeRD impression/slate.
- Input slate: original `article_ids_inview` logged display order.
- Not used as input: engagement-sorted order, future sampled candidate order, baseline-ranker order.
- Input user context: ordered past UIH before `impression_time`, user id, history summary, source UIH embedding.
- Per-item input: article Qwen embedding, position embedding, full logged-slate Transformer context.
- Target: one frozen Qwen reaction-event text latent for each displayed item occurrence, indexed by `(sample_id, displayed_rank)`.
- Explicit heads predict click, log read time, scroll, log next read time, next scroll, and log gain.
- Scroll and next-scroll losses/metrics use observed-scroll masks. This `maskfix` run fixes an earlier smoke issue where not-clicked rows with missing scroll still contributed zero-scroll loss.
- Reaction is multi-signal, not a mutually exclusive class label. The same displayed item event may have `clicked=yes`, positive read time, positive scroll, and positive next-read/next-scroll simultaneously.
- A repeated article has repeated reaction targets. We do not collapse targets by `article_id`, because the same article can produce different reactions for different users, positions, slate contexts, and times.

Metric correction after this run:

- Keep click LogLoss/Brier/ECE and read/scroll/gain regression as primary reaction-prediction metrics.
- Treat gain nDCG as a slate-structure diagnostic, not the main task by itself.
- Add multi-signal event AUC diagnostics for `read_event`, `scroll_event`, `next_read_event`, and `next_scroll_event`, plus a macro AUC over click/read/scroll/next signals.
- Target-cache schema was updated to `logged_reaction_multisignal_v2`; old target caches without explicit observed/missing flags should be rebuilt before rerunning.

Target cache:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_logged_slate_reaction_jepa_10k/logged_reaction_targets
```

Candidate summary:

| samples | candidates | positives | truncated | avg candidates/slate | avg positives/slate | click rate |
|---:|---:|---:|---:|---:|---:|---:|
| 30000 | 344697 | 29729 | 1569 | 11.490 | 0.991 | 0.0862 |

Main reaction-prediction results on test:

| model | click AUC | click LogLoss | Brier | log-read RMSE | clicked log-read RMSE | scroll RMSE | clicked scroll RMSE | log-gain RMSE | gain nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logged policy heuristic | 0.6341 | 0.2994 | 0.0805 | 0.9462 | 3.0644 | 0.2853 | 0.9706 | 0.3490 | 0.4377 |
| Position prior only | 0.6824 | 0.2806 | 0.0772 | 0.9580 | 3.2144 | 0.2747 | 0.8684 | 0.3459 | 0.4366 |
| Item only | 0.8167 | 0.2428 | 0.0700 | 0.9162 | 3.0167 | 0.2582 | 0.7794 | 0.3286 | 0.6252 |
| User history + item supervised | 0.8160 | 0.2459 | 0.0713 | 0.9072 | 2.9315 | 0.2695 | 0.7040 | 0.3371 | 0.6275 |
| Logged-slate reaction JEPA | 0.7390 | 0.2736 | 0.0763 | 0.9451 | 3.1506 | 0.2716 | 0.8455 | 0.3419 | 0.5508 |
| True reaction latent probe | 1.0000 | 0.0006 | 0.0000 | 0.2660 | 0.8955 | 0.0410 | 0.1301 | 0.1860 | 1.0000 |

Slate-structure results on test:

| model | pairwise gain agreement | within-slate Spearman | gain nDCG@5 | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|---:|
| Logged policy heuristic | 0.5024 | -0.4653 | 0.3513 | 0.4377 | 0.2864 |
| Position prior only | 0.5003 | 0.7645 | 0.3500 | 0.4366 | 0.2895 |
| Item only | 0.7606 | 0.1503 | 0.5865 | 0.6252 | 0.7437 |
| User history + item supervised | 0.7645 | 0.1976 | 0.5894 | 0.6275 | 0.7459 |
| Logged-slate reaction JEPA | 0.6711 | 0.0711 | 0.4879 | 0.5508 | 0.5410 |
| True reaction latent probe | 1.0000 | 0.3915 | 1.0000 | 1.0000 | 2.4667 |

JEPA latent diagnostics on test:

| metric | value |
|---|---:|
| reaction latent MRR | 0.8823 |
| Recall@1 | 0.7813 |
| Recall@3 | 0.9843 |
| Recall@5 | 0.9947 |
| predicted-target cosine | 0.3818 |
| item-target cosine | 0.4125 |
| delta-target cosine | 0.2664 |
| predicted pairwise cosine | 0.3319 |

Interpretation:

- This run correctly changes the main task from reranking to logged-slate reaction world modeling.
- The oracle/probe result is intentionally an upper bound, not deployable: the true reaction text includes the reaction label and response values, so a probe can almost directly recover click/read/scroll.
- The frozen-Qwen target itself is highly identifiable. The JEPA predictor reaches reaction-latent MRR `0.8823` and Recall@1 `0.7813`.
- However, the current JEPA training does not convert that latent retrieval signal into stronger reaction-head prediction.
  - It beats position-only, but it is far below item-only and supervised user-history+item baselines.
  - Test click AUC gap: `0.7390` versus item-only `0.8167`.
  - Test gain nDCG@10 gap: `0.5508` versus item-only `0.6252`.
- User history helps only slightly in the supervised baseline:
  - Item-only gain nDCG@10 is `0.6252`.
  - User-history+item supervised gain nDCG@10 is `0.6275`.
  - This suggests logged-slate short-term reaction is dominated by item/position/logged-policy signals at this scale, and user-conditioned gains need stronger modeling or more data.

Research conclusion:

```text
The logged-slate formulation is now correct and paper-clean:
the model predicts concrete per-item reactions for the real displayed slate,
not a reordered or sampled future candidate list.

The 10k result is not yet a paper win for the JEPA objective:
the predictor can retrieve the Qwen reaction latent, but the reaction heads
still underperform strong item-only and supervised baselines.

Next work should focus on coupling the predicted reaction latent to the heads:
stronger latent-to-head adapter, balanced head-vs-latent loss, contrastive hard
negatives within the same slate, and possibly caching/logging candidate index
so multi-seed ablations are cheap.
```

## Logged-Slate Reaction JEPA 10k Additive Ablation

Run path:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_logged_slate_reaction_jepa_10k_additive/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_logged_slate_reaction_jepa_10k_additive/summary.json
```

Script:

```text
ebnerd_logged_slate_reaction_jepa.py
run_ebnerd_logged_slate_reaction_jepa_additive_10k.sh
```

Purpose:

This run changes the framing from "standalone JEPA must beat baselines" to
"JEPA as an increment over the strong supervised reaction predictor." All
additive variants use the same logged-slate reaction task and the same 10k
train/val/test split as the maskfix run.

Additive variants:

- `user_history_item_supervised`: strong supervised baseline, no JEPA latent loss.
- `user_history_item_latent_adapter_control`: latent adapter path enabled, but no JEPA latent loss.
- `user_history_item_jepa_aux_w0p02`: supervised baseline plus JEPA latent auxiliary loss, weight `0.02`.
- `user_history_item_jepa_aux_adapter_w0p02`: JEPA latent auxiliary loss plus predicted-latent/delta features fed into heads, weight `0.02`.
- `user_history_item_jepa_aux_w0p05`: same as above without latent head features, weight `0.05`.
- `user_history_item_jepa_aux_adapter_w0p05`: JEPA latent auxiliary loss plus predicted-latent/delta features fed into heads, weight `0.05`.

Candidate summary:

| samples | candidates | positives | truncated | avg candidates/slate | avg positives/slate | click rate |
|---:|---:|---:|---:|---:|---:|---:|
| 30000 | 344697 | 29729 | 1569 | 11.490 | 0.991 | 0.0862 |

Main test results:

| model | click AUC | click LogLoss | log-read RMSE | clicked log-read RMSE | log-gain RMSE | gain nDCG@5 | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| User history + item supervised | 0.8160 | 0.2459 | 0.9072 | 2.9315 | 0.3371 | 0.5894 | 0.6275 | 0.7459 |
| Latent adapter control, no JEPA loss | 0.8087 | 0.2429 | 0.9235 | 3.0675 | 0.3277 | 0.5856 | 0.6252 | 0.7574 |
| JEPA aux, w=0.02 | 0.8133 | 0.2422 | 0.9263 | 3.0748 | 0.3304 | 0.5851 | 0.6260 | 0.7436 |
| JEPA aux + latent adapter, w=0.02 | 0.8157 | 0.2412 | 0.9345 | 3.1272 | 0.3300 | 0.5857 | 0.6241 | 0.7381 |
| JEPA aux, w=0.05 | 0.8110 | 0.2461 | 0.9211 | 3.0457 | 0.3321 | 0.5780 | 0.6185 | 0.7134 |
| JEPA aux + latent adapter, w=0.05 | 0.8016 | 0.2462 | 0.9437 | 3.1646 | 0.3335 | 0.5712 | 0.6142 | 0.7055 |

JEPA latent diagnostics on test:

| model | reaction latent MRR | Recall@1 | predicted-target cosine | delta-target cosine | predicted pairwise cosine |
|---|---:|---:|---:|---:|---:|
| JEPA aux, w=0.02 | 0.8800 | 0.7793 | 0.3203 | 0.1847 | 0.3585 |
| JEPA aux + latent adapter, w=0.02 | 0.8413 | 0.7243 | 0.2091 | 0.0758 | 0.5624 |
| JEPA aux, w=0.05 | 0.8829 | 0.7830 | 0.3599 | 0.2482 | 0.3502 |
| JEPA aux + latent adapter, w=0.05 | 0.8655 | 0.7581 | 0.3092 | 0.1715 | 0.4651 |

Incremental interpretation:

- JEPA auxiliary loss gives a real reaction-prediction increment on some metrics:
  - `w=0.02` improves click LogLoss from `0.2459` to `0.2422`.
  - `w=0.02 + adapter` improves click LogLoss further to `0.2412`.
  - `w=0.02` improves log-gain RMSE from `0.3371` to `0.3304`.
- It does not yet improve the main slate-level engagement ordering metric:
  - supervised baseline gain nDCG@10 is `0.6275`.
  - best JEPA additive gain nDCG@10 is `0.6260`.
- The latent adapter path by itself is a confounder:
  - adapter control improves click LogLoss and log-gain RMSE without JEPA loss.
  - therefore the cleanest JEPA-only comparison is `user_history_item_supervised` vs `user_history_item_jepa_aux_w0p02`.
- Weight `0.05` over-regularizes the reaction heads. It improves latent diagnostics slightly, but hurts the downstream reaction metrics.
- Directly feeding predicted latent/delta into the heads is unstable. It helps click LogLoss at `w=0.02`, but hurts slate-level gain ranking.

Research conclusion:

```text
The additive JEPA framing is partially supported:
JEPA latent prediction can act as an auxiliary world-modeling loss that improves
click calibration and gain-value regression over the supervised reaction model.

It is not yet enough for the stronger paper claim that JEPA improves full-label
slate-level engagement ordering. The next step should keep JEPA as an increment,
but use a cleaner two-stage adapter:
1. train JEPA reaction latent predictor,
2. freeze/export predicted reaction features,
3. train the supervised reaction model with residual/cross features from those
   frozen predictions.

This would separate "can predict future/reaction UIH" from "can the reaction
head exploit the prediction", and avoids the current joint-loss interference.
```

## Logged-Slate Reaction JEPA 10k Additive Multi-Signal v2

Run path:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_logged_slate_reaction_jepa_10k_additive_multisignal_v2/results.json
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_logged_slate_reaction_jepa_10k_additive_multisignal_v2/summary.json
```

Target cache:

```text
/home/robotuser/zijian/RecJepa/outputs/ebnerd_small_logged_slate_reaction_jepa_10k_multisignal_v2/logged_reaction_targets
```

Target schema:

```text
logged_reaction_multisignal_v2
index_unit = target_logged[sample_id, displayed_rank]
reaction = multi-signal per displayed item
click/read/scroll/next-read/next-scroll are not mutually exclusive
missing flags are explicit in the reaction text
```

This run rebuilds the frozen-Qwen reaction target texts after the correction
that one displayed item can have multiple simultaneous reaction signals. The
evaluation now reports event AUC for read, scroll, next-read, and next-scroll,
plus a macro AUC over all reaction-event signals.

Candidate summary:

| samples | candidates | positives | truncated | avg candidates/slate | avg positives/slate | click rate |
|---:|---:|---:|---:|---:|---:|---:|
| 30000 | 344697 | 29729 | 1569 | 11.490 | 0.991 | 0.0862 |

Main test results:

| model | click AUC | click LogLoss | Brier | log-read RMSE | clicked log-read RMSE | scroll RMSE | clicked scroll RMSE | log-gain RMSE | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logged policy heuristic | 0.6341 | 0.2994 | 0.0805 | 0.9462 | 3.0644 | 0.2853 | 0.9706 | 0.3490 | 0.4377 | 0.2864 |
| Position prior only | 0.6824 | 0.2806 | 0.0772 | 0.9580 | 3.2144 | 0.2748 | 0.8684 | 0.3459 | 0.4366 | 0.2895 |
| Item only | 0.8167 | 0.2428 | 0.0700 | 0.9162 | 3.0167 | 0.2582 | 0.7794 | 0.3286 | 0.6252 | 0.7437 |
| User history + item supervised | 0.8160 | 0.2459 | 0.0713 | 0.9072 | 2.9315 | 0.2695 | 0.7040 | 0.3371 | 0.6275 | 0.7459 |
| Latent adapter control, no JEPA loss | 0.8087 | 0.2429 | 0.0691 | 0.9235 | 3.0675 | 0.2581 | 0.8049 | 0.3277 | 0.6252 | 0.7574 |
| JEPA aux, w=0.02 | 0.8127 | 0.2427 | 0.0694 | 0.9245 | 3.0658 | 0.2608 | 0.7655 | 0.3299 | 0.6256 | 0.7436 |
| JEPA aux + latent adapter, w=0.02 | 0.8198 | 0.2380 | 0.0684 | 0.9322 | 3.1196 | 0.2585 | 0.8193 | 0.3272 | 0.6324 | 0.7656 |
| JEPA aux, w=0.05 | 0.8126 | 0.2452 | 0.0705 | 0.9191 | 3.0343 | 0.2591 | 0.7813 | 0.3305 | 0.6197 | 0.7137 |
| JEPA aux + latent adapter, w=0.05 | 0.7984 | 0.2497 | 0.0708 | 0.9350 | 3.1218 | 0.2622 | 0.8101 | 0.3326 | 0.6093 | 0.7092 |

Multi-signal event AUC results:

| model | click AUC | read event AUC | scroll event AUC | next-read event AUC | next-scroll event AUC | macro event AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logged policy heuristic | 0.6341 | 0.6102 | 0.6182 | 0.6117 | 0.6090 | 0.6166 |
| Position prior only | 0.6824 | 0.6824 | 0.6862 | 0.6860 | 0.6808 | 0.6835 |
| Item only | 0.8167 | 0.8080 | 0.8191 | 0.8153 | 0.8124 | 0.8143 |
| User history + item supervised | 0.8160 | 0.8142 | 0.8237 | 0.8139 | 0.8160 | 0.8168 |
| Latent adapter control, no JEPA loss | 0.8087 | 0.8050 | 0.8134 | 0.8040 | 0.8030 | 0.8068 |
| JEPA aux, w=0.02 | 0.8127 | 0.8031 | 0.8223 | 0.8010 | 0.8106 | 0.8099 |
| JEPA aux + latent adapter, w=0.02 | 0.8198 | 0.8119 | 0.8160 | 0.8054 | 0.8089 | 0.8124 |
| JEPA aux, w=0.05 | 0.8126 | 0.8116 | 0.8213 | 0.8124 | 0.8135 | 0.8143 |
| JEPA aux + latent adapter, w=0.05 | 0.7984 | 0.7932 | 0.8082 | 0.7906 | 0.7892 | 0.7959 |

JEPA latent diagnostics on test:

| model | reaction latent MRR | Recall@1 | Recall@3 | Recall@5 | predicted-target cosine | item-target cosine | delta-target cosine | predicted pairwise cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JEPA aux, w=0.02 | 0.8842 | 0.7929 | 0.9759 | 0.9917 | 0.2894 | 0.3508 | 0.2215 | 0.3495 |
| JEPA aux + latent adapter, w=0.02 | 0.8130 | 0.6898 | 0.9295 | 0.9698 | 0.1574 | 0.3508 | 0.0883 | 0.5739 |
| JEPA aux, w=0.05 | 0.8885 | 0.7994 | 0.9785 | 0.9927 | 0.3159 | 0.3508 | 0.2813 | 0.3419 |
| JEPA aux + latent adapter, w=0.05 | 0.8581 | 0.7528 | 0.9626 | 0.9866 | 0.2241 | 0.3508 | 0.1709 | 0.4404 |

Incremental interpretation:

- The cleanest positive result is `JEPA aux + latent adapter, w=0.02`.
  - It improves click AUC from `0.8160` to `0.8198`.
  - It improves click LogLoss from `0.2459` to `0.2380`.
  - It improves click Brier from `0.0713` to `0.0684`.
  - It improves gain nDCG@10 from `0.6275` to `0.6324`.
  - It improves gain@1 from `0.7459` to `0.7656`.
- This is the first logged-slate 10k run where the JEPA additive path improves
  both click prediction and slate-level gain ordering over the strong supervised
  user-history+item baseline.
- The improvement is not uniform across all reaction metrics.
  - Macro event AUC drops from `0.8168` to `0.8124`.
  - Read/next-read event AUC also drops, so the current adapter is more helpful
    for click/gain ranking than for the full multi-signal reaction state.
- The no-adapter JEPA auxiliary loss is weaker.
  - It learns a good latent retrieval signal, but the supervised heads do not
    consistently exploit that signal without explicit latent/delta features.
- Weight `0.05` still over-regularizes or misaligns the heads.
  - It improves latent diagnostics slightly over `w=0.02`, but hurts the
    engagement ordering metrics.

Research conclusion:

```text
This run gives a concrete incremental JEPA signal in the corrected logged-slate
reaction task: with the same user-history + item supervised model, adding the
JEPA reaction-latent auxiliary objective plus a latent adapter improves click
AUC/LogLoss/Brier and gain nDCG@10.

The result is promising but not yet the final paper claim, because the full
multi-signal reaction macro AUC still trails the supervised baseline. The next
paper-level step should preserve the positive w=0.02 adapter recipe, then run
multi-seed 10k and full-small, while improving the adapter so read/scroll/next
reaction metrics do not regress.
```

## Logged-Slate Reaction JEPA 10k Gain-Aware Multi-Seed

Change made after the first additive run:

```text
checkpoint_metric = gain_ndcg10
slate_aux_weight = 0.20
predictor_gain_aux_weight = 0.05
target = logged_reaction_multisignal_v2
```

Why this change matters:

```text
The previous checkpoint selection optimized a reaction proxy:
click LogLoss + read RMSE + gain RMSE.

That can improve reaction prediction while leaving slate-level engagement
ordering unstable. For the current claim, checkpoint selection must be aligned
with the listwise future-engagement signal.
```

Code/scripts:

```text
ebnerd_logged_slate_reaction_jepa.py
run_ebnerd_logged_slate_reaction_jepa_additive_10k_gainselect_multiseed.sh
```

### Main 3-Seed Result: JEPA Weight 0.02

Run paths:

```text
outputs/ebnerd_small_logged_slate_reaction_jepa_10k_additive_multisignal_v2_gain_ndcg10_slate0.20_seed7
outputs/ebnerd_small_logged_slate_reaction_jepa_10k_additive_multisignal_v2_gain_ndcg10_slate0.20_seed11
outputs/ebnerd_small_logged_slate_reaction_jepa_10k_additive_multisignal_v2_gain_ndcg10_slate0.20_seed13
```

Mean test metrics over seeds 7/11/13:

| model | click AUC | click LogLoss | macro event AUC | log-gain RMSE | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|---:|---:|
| Item only | 0.8136 | 0.2424 | 0.8011 | 0.3309 | 0.6215 | 0.7564 |
| User history + item supervised | 0.8118 | 0.2450 | 0.8082 | 0.3334 | 0.6278 | 0.7522 |
| Latent adapter control, no JEPA | 0.8065 | 0.2464 | 0.8048 | 0.3325 | 0.6263 | 0.7477 |
| JEPA aux, w=0.02 | 0.8091 | 0.2452 | 0.8061 | 0.3376 | 0.6266 | 0.7451 |
| JEPA aux + latent adapter, w=0.02 | 0.8134 | 0.2452 | 0.8043 | 0.3301 | 0.6292 | 0.7591 |

Delta vs same latent-adapter control:

| model | click AUC | click LogLoss | macro event AUC | log-gain RMSE | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|---:|---:|
| JEPA aux, w=0.02 | +0.0026 | -0.0012 | +0.0013 | +0.0051 | +0.0003 | -0.0026 |
| JEPA aux + latent adapter, w=0.02 | +0.0069 | -0.0012 | -0.0005 | -0.0024 | +0.0029 | +0.0114 |

Per-seed gain nDCG@10:

| seed | supervised | adapter control | JEPA aux | JEPA aux + adapter |
|---:|---:|---:|---:|---:|
| 7 | 0.6275 | 0.6266 | 0.6254 | 0.6333 |
| 11 | 0.6245 | 0.6178 | 0.6295 | 0.6313 |
| 13 | 0.6313 | 0.6346 | 0.6248 | 0.6231 |

Interpretation:

```text
JEPA w=0.02 gives a positive mean listwise increment when the predicted
reaction latent is explicitly exposed to the reaction head:

  adapter control -> JEPA+adapter
  gain nDCG@10: 0.6263 -> 0.6292
  gain@1:       0.7477 -> 0.7591
  click AUC:    0.8065 -> 0.8134

This supports the core idea that reaction-UIH latent prediction can add useful
signal to logged-slate future-engagement ordering.

But the effect is not yet fully stable: seed 13 regresses. So this is a
promising 10k result, not yet a final paper-level result.
```

### Low JEPA Weight Sweep: 0.01

Run paths:

```text
outputs/ebnerd_small_logged_slate_reaction_jepa_10k_additive_multisignal_v2_gain_ndcg10_slate0.20_scales0p01_seed7
outputs/ebnerd_small_logged_slate_reaction_jepa_10k_additive_multisignal_v2_gain_ndcg10_slate0.20_scales0p01_seed11
outputs/ebnerd_small_logged_slate_reaction_jepa_10k_additive_multisignal_v2_gain_ndcg10_slate0.20_scales0p01_seed13
```

This run skipped the ordinary baselines and only compared the latent-adapter
control with JEPA variants.

Mean test metrics over seeds 7/11/13:

| model | click AUC | click LogLoss | macro event AUC | log-gain RMSE | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|---:|---:|
| Latent adapter control, no JEPA | 0.8066 | 0.2462 | 0.8057 | 0.3324 | 0.6227 | 0.7448 |
| JEPA aux, w=0.01 | 0.8164 | 0.2427 | 0.8079 | 0.3306 | 0.6220 | 0.7577 |
| JEPA aux + latent adapter, w=0.01 | 0.8103 | 0.2437 | 0.7981 | 0.3321 | 0.6247 | 0.7544 |

Delta vs same latent-adapter control:

| model | click AUC | click LogLoss | macro event AUC | log-gain RMSE | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|---:|---:|
| JEPA aux, w=0.01 | +0.0098 | -0.0036 | +0.0022 | -0.0018 | -0.0007 | +0.0129 |
| JEPA aux + latent adapter, w=0.01 | +0.0037 | -0.0025 | -0.0076 | -0.0003 | +0.0020 | +0.0096 |

Interpretation:

```text
Lowering the JEPA weight improves click/reaction prediction and gain@1, but
does not clearly improve gain nDCG@10 through the no-adapter path. The adapter
path still gives a positive mean nDCG increment, but macro event AUC drops.

This suggests the JEPA predictor is learning a useful user-conditioned reaction
latent, while the current fusion method is too high-variance.
```

### Current Research Conclusion

```text
The strongest current claim is not "final recommender lift" yet.

The strongest supported claim is:
  A frozen-Qwen reaction-event JEPA target can be learned from logged slates,
  and adding the predicted reaction latent as an auxiliary/user-state signal
  gives a small positive mean increment on listwise future-engagement ranking.

Evidence:
  - reaction latent retrieval is consistently non-random
  - w=0.02 JEPA+adapter improves mean gain nDCG@10 over adapter control
  - w=0.01 JEPA variants improve click AUC/LogLoss and gain@1
  - improvements are not stable enough across all seeds to claim solved

Next paper-level step:
  replace direct pred/delta concatenation with a conservative residual fusion:
  train the reaction predictor first, freeze it, then feed only calibrated
  scalar sidecar features into the supervised slate model:
    predicted click/read/scroll/gain
    predicted-vs-item reaction cosine
    predicted delta norm/cosine
    within-slate normalized predicted gain

This will test the core hypothesis more cleanly:
  JEPA predicts future/reaction UIH, and a downstream listwise model can use
  that predicted UIH as an increment.
```

## Full EB-NeRD Small: Logged-Slate Reaction JEPA With Compact Qwen Target

Run date: 2026-05-15.

This run uses the full EB-NeRD small logged-slate task:

```text
unit: one logged impression / slate
samples: 183,207
displayed item reactions: 2,111,078
click positives: 181,443
click rate: 8.59%
max slate length: 30
input order: original article_ids_inview logged display order
task: per-item multi-signal reaction prediction + slate gain ordering
checkpoint metric: validation gain nDCG@10
```

The direct full-context Qwen target was too slow for full small. The first
attempt encoded every displayed item reaction text directly:

```text
full-context target texts: 2,111,078
Qwen3-Embedding-4B, max_seq=1024, batch=32
estimated runtime: about 11 hours for target cache alone
```

So the completed full-small run uses a compact, deduplicated target:

```text
target_text_mode: compact_reaction
target text: article brief + concrete reaction flags/values
predictor input still includes: user history, user/profile features, position, and logged slate context
unique Qwen target texts: 188,885
target locations filled: 2,111,078
dedup ratio: 11.18x
Qwen3-Embedding-4B, max_seq=256, batch=512
```

Run paths:

```text
baseline partial:
outputs/ebnerd_small_logged_slate_reaction_jepa_full_gain_ndcg10_slate0.20_scales0p02x0p01_seed7/partial_results.json

compact JEPA:
outputs/ebnerd_small_logged_slate_reaction_jepa_full_gain_ndcg10_slate0.20_scales0p02x0p01_compact_qwen256_bs512_seed7/summary.json

target cache:
outputs/ebnerd_small_logged_slate_reaction_jepa_full_multisignal_v3_compact_reaction_qwen256/logged_reaction_targets
```

Test metrics:

| model | click AUC | macro event AUC | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|
| Position prior only | 0.6822 | 0.6834 | 0.4349 | 0.2854 |
| Item only | 0.8137 | 0.8050 | 0.6268 | 0.7503 |
| User history + item supervised | 0.8125 | 0.8136 | 0.6320 | 0.7702 |
| Latent adapter control, no JEPA | 0.8034 | 0.7940 | 0.6222 | 0.7604 |
| JEPA aux, w=0.02 | 0.8092 | 0.8080 | 0.6250 | 0.7669 |
| JEPA aux + adapter, w=0.02 | 0.8134 | 0.8132 | 0.6297 | 0.7529 |
| JEPA aux, w=0.01 | 0.8108 | 0.8051 | 0.6259 | 0.7483 |
| JEPA aux + adapter, w=0.01 | 0.8151 | 0.8098 | 0.6181 | 0.7446 |

Deltas vs the same latent-adapter control:

| model | click AUC | macro event AUC | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|
| JEPA aux, w=0.02 | +0.0057 | +0.0140 | +0.0029 | +0.0066 |
| JEPA aux + adapter, w=0.02 | +0.0100 | +0.0192 | +0.0075 | -0.0075 |
| JEPA aux, w=0.01 | +0.0073 | +0.0111 | +0.0037 | -0.0121 |
| JEPA aux + adapter, w=0.01 | +0.0116 | +0.0158 | -0.0040 | -0.0158 |

Deltas vs the strongest supervised baseline:

| model | click AUC | macro event AUC | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|
| JEPA aux + adapter, w=0.02 | +0.0009 | -0.0004 | -0.0023 | -0.0174 |
| JEPA aux, w=0.01 | -0.0018 | -0.0085 | -0.0061 | -0.0220 |

Latent diagnostics on test:

| model | reaction Recall@1 | pred-target cosine | item-target cosine | delta-target cosine |
|---|---:|---:|---:|---:|
| JEPA aux, w=0.02 | 0.9972 | 0.6433 | 0.6575 | 0.2997 |
| JEPA aux + adapter, w=0.02 | 0.9983 | 0.5957 | 0.6575 | 0.2466 |
| JEPA aux, w=0.01 | 0.9988 | 0.6201 | 0.6575 | 0.2223 |
| JEPA aux + adapter, w=0.01 | 0.9987 | 0.4946 | 0.6575 | 0.1201 |

Interpretation:

```text
Full small gives a more conservative result than 10k.

Positive evidence:
  JEPA improves over the same latent-adapter control on click AUC,
  macro event AUC, and gain nDCG@10. The best control-relative gain is
  JEPA aux + adapter w=0.02: +0.0075 gain nDCG@10.

Limitation:
  JEPA does not beat the strongest user-history + item supervised baseline
  on test gain nDCG@10. The best JEPA variant reaches 0.6297 vs the strong
  supervised baseline at 0.6320.

Conclusion:
  The current experiment supports "JEPA adds information over a matched
  latent-adapter control", but it does not yet prove end-to-end lift over the
  strongest supervised logged-slate reaction model.
```

Next step:

```text
Do not keep pushing direct pred/delta concatenation as the final fusion.
It is high-variance and can hurt gain@1.

The better next experiment is a two-stage JEPA sidecar:
  1. train the reaction predictor with compact Qwen reaction target
  2. freeze it
  3. feed calibrated scalar sidecar features into the strong supervised model:
       predicted click/read/scroll/gain
       predicted-vs-item target cosine
       delta norm / delta cosine
       within-slate normalized predicted gain

This tests the core claim more cleanly:
  learned future/reaction UIH can provide incremental features to a strong
  listwise reaction model.
```

## Full EB-NeRD Small: Two-Stage JEPA Sidecar

Run date: 2026-05-16.

This run tests the next fusion idea:

```text
stage 1:
  train a JEPA reaction predictor with compact Qwen reaction targets
  checkpoint by validation gain nDCG@10

stage 2:
  freeze the predictor
  generate deployable per-item scalar sidecar features from predictor outputs
  train a supervised logged-slate reaction model with these sidecar scalars

important:
  sidecar features do not use oracle target embeddings or true reactions at inference
```

Sidecar scalar features:

```text
sidecar_click_prob
sidecar_log_read
sidecar_scroll
sidecar_log_next_read
sidecar_next_scroll
sidecar_log_gain
sidecar_gain_score
sidecar_pred_item_cosine
sidecar_delta_norm
sidecar_delta_item_cosine
sidecar_log_gain_zscore
sidecar_gain_score_zscore
```

Run paths:

```text
10k smoke:
outputs/ebnerd_small_logged_slate_reaction_jepa_sidecar_10000_gain_ndcg10_w0p01_qwen256_bs512_seed7

full small:
outputs/ebnerd_small_logged_slate_reaction_jepa_sidecar_full_gain_ndcg10_w0p01_qwen256_bs512_seed7

full sidecar feature cache:
outputs/ebnerd_small_logged_slate_reaction_jepa_sidecar_full_gain_ndcg10_w0p01_qwen256_bs512_seed7/two_stage_jepa_predictor_w0p01_sidecar_features.npy
```

10k smoke passed:

```text
candidate samples: 30,000
candidate item reactions: 344,697
compact target unique texts: 35,846
sidecar feature cache written: 30,000 x 30 x 12, float16
summary.json written successfully
```

Full-small test metrics:

| model | click AUC | macro event AUC | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|
| User history + item supervised | 0.8125 | 0.8136 | 0.6320 | 0.7702 |
| JEPA aux + adapter, w=0.02 | 0.8134 | 0.8132 | 0.6297 | 0.7529 |
| Two-stage JEPA predictor, w=0.01 | 0.8001 | 0.7909 | 0.6193 | 0.7640 |
| Two-stage JEPA sidecar supervised, w=0.01 | 0.8033 | 0.7958 | 0.6280 | 0.7765 |

Deltas:

| comparison | click AUC | macro event AUC | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|
| Sidecar supervised vs JEPA predictor | +0.0032 | +0.0048 | +0.0087 | +0.0126 |
| Sidecar supervised vs strongest supervised baseline | -0.0092 | -0.0179 | -0.0040 | +0.0063 |
| Sidecar supervised vs direct JEPA+adapter w=0.02 | -0.0101 | -0.0174 | -0.0017 | +0.0237 |

Interpretation:

```text
The two-stage sidecar is technically valid and deployable:
  it trains a JEPA predictor, freezes it, and feeds only predicted scalar
  reaction features to a downstream supervised model.

It improves over the frozen JEPA predictor itself:
  gain nDCG@10: 0.6193 -> 0.6280
  gain@1:       0.7640 -> 0.7765

But it still does not beat the strongest supervised baseline on gain nDCG@10:
  sidecar: 0.6280
  strong supervised: 0.6320

It does beat the strongest supervised baseline on gain@1:
  sidecar: 0.7765
  strong supervised: 0.7702
```

Conclusion:

```text
The current JEPA sidecar is useful for top-1 engagement strength, but not yet
for full slate nDCG@10. This suggests the sidecar signal is too coarse or too
weakly calibrated across the whole slate.
```

Next improvement:

```text
The sidecar feature route is the right place to improve, but the current
features need calibration/normalization.

Try:
  - train the sidecar ranker with sidecar features normalized from train split
  - add a sidecar gate instead of adding scalars directly to the item scalar projection
  - train the JEPA predictor longer or with a better predictor checkpoint metric
  - make sidecar features rank-relative: per-slate rank of predicted gain,
    percentile score, and top-k margin
```

## Full EB-NeRD Small: Rank-Relative Train-Normalized Sidecar

Run date: 2026-05-16.

This run keeps the same logged-slate reaction JEPA task, but expands the
deployable sidecar from 12 scalars to 24 scalars:

```text
base predicted reaction features:
  click/read/scroll/next_read/next_scroll/gain heads
  latent-item cosine features
  slate z-scores for predicted gain

new rank-relative features:
  per-slate rank percentile for predicted click/log_gain/gain_score
  top-1 margin for predicted click/log_gain/gain_score
  top-3 flags for predicted click/log_gain/gain_score

normalization:
  compute mean/std only on ranker_train valid items
  apply to train/val/test sidecar features
```

Run paths:

```text
10k smoke:
outputs/ebnerd_small_logged_slate_reaction_jepa_sidecar_10000_gain_ndcg10_reltnorm_w0p01_qwen256_bs512_seed7

full small:
outputs/ebnerd_small_logged_slate_reaction_jepa_sidecar_full_gain_ndcg10_reltnorm_w0p01_qwen256_bs512_seed7

full sidecar feature cache:
outputs/ebnerd_small_logged_slate_reaction_jepa_sidecar_full_gain_ndcg10_reltnorm_w0p01_qwen256_bs512_seed7/two_stage_jepa_predictor_w0p01_sidecar_features.npy
```

10k smoke passed:

```text
candidate samples: 30,000
candidate item reactions: 344,697
sidecar feature cache written: 30,000 x 30 x 24, float16
train-normalization valid items: 117,026
summary.json written successfully
```

Full-small candidate summary:

```text
logged slates: 183,207
item reactions: 2,111,078
clicked positives: 181,443
click rate: 0.08595
max candidates per slate: 30
train-normalization valid items: 1,276,925
```

Full-small test metrics:

| model | click AUC | macro event AUC | pairwise gain agreement | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|---:|
| User history + item supervised | 0.8125 | 0.8136 | n/a | 0.6320 | 0.7702 |
| Two-stage JEPA predictor, raw sidecar run | 0.8001 | 0.7909 | 0.7321 | 0.6193 | 0.7640 |
| Two-stage JEPA sidecar, 12 raw features | 0.8033 | 0.7958 | n/a | 0.6280 | 0.7765 |
| Two-stage JEPA sidecar, 24 rank-relative train-normalized features | 0.8028 | 0.8030 | 0.7578 | 0.6278 | 0.7731 |

Deltas:

| comparison | click AUC | macro event AUC | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|
| 24-feature sidecar vs predictor | +0.0027 | +0.0121 | +0.0085 | +0.0091 |
| 24-feature sidecar vs 12-feature sidecar | -0.0005 | +0.0072 | -0.0002 | -0.0035 |
| 24-feature sidecar vs supervised baseline | -0.0097 | -0.0106 | -0.0042 | +0.0029 |

Interpretation:

```text
Rank-relative sidecar features are useful for multi-signal reaction prediction:
  macro event AUC improves from 0.7958 to 0.8030 versus the 12-feature sidecar.
  pairwise gain agreement reaches 0.7578.

However, they do not improve the main full-slate nDCG@10 over the 12-feature
raw sidecar:
  12-feature sidecar nDCG@10: 0.6280
  24-feature rel+trainnorm sidecar nDCG@10: 0.6278

The JEPA signal is still additive over its own predictor and still beats the
supervised baseline on gain@1, but it does not yet beat the supervised baseline
on gain nDCG@10.
```

Conclusion:

```text
This attempt shows that calibration/rank-relative features help reaction
classification structure, but not enough for full-slate nDCG. The bottleneck is
probably not just scalar sidecar calibration; the downstream model needs a
stronger mechanism to fuse predicted reaction state into item hidden states.
```

Next improvement:

```text
Try a sidecar gate / residual adapter inside the transformer item hidden state:
  item_hidden <- item_hidden + gate(sidecar) * sidecar_adapter(sidecar)

This is stronger than scalar concatenation because the predicted reaction state
can modulate the item representation before the reaction heads and listwise gain
score, instead of being compressed through the generic scalar projection.
```

## Sidecar Gate Follow-Up

Run date: 2026-05-16.

Implementation:

```text
scalar_concat:
  append 24 sidecar features to the item scalar features

gated:
  keep base scalar features only
  inject sidecar through:
    item_hidden <- item_hidden + sigmoid(gate(sidecar)) * adapter(sidecar)

both:
  append sidecar features to scalar features
  also inject sidecar through the gated residual
```

To isolate downstream fusion, these runs reused existing sidecar feature caches
and skipped JEPA predictor retraining:

```text
10k cache:
outputs/ebnerd_small_logged_slate_reaction_jepa_sidecar_10000_gain_ndcg10_reltnorm_w0p01_qwen256_bs512_seed7/two_stage_jepa_predictor_w0p01_sidecar_features.npy

full cache:
outputs/ebnerd_small_logged_slate_reaction_jepa_sidecar_full_gain_ndcg10_reltnorm_w0p01_qwen256_bs512_seed7/two_stage_jepa_predictor_w0p01_sidecar_features.npy
```

10k smoke:

| fusion | click AUC | macro event AUC | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|
| scalar concat, 24 rel+trainnorm | 0.7807 | 0.7829 | 0.5836 | 0.6045 |
| gated only, 24 rel+trainnorm | 0.7677 | 0.7578 | 0.5419 | 0.5541 |
| both scalar+gate, 24 rel+trainnorm | 0.7706 | 0.7758 | 0.5915 | 0.6625 |

Full-small test:

| fusion | click AUC | macro event AUC | pairwise gain agreement | gain nDCG@10 | gain@1 |
|---|---:|---:|---:|---:|---:|
| scalar concat, 24 rel+trainnorm | 0.8028 | 0.8030 | 0.7578 | 0.6278 | 0.7731 |
| both scalar+gate, 24 rel+trainnorm | 0.8035 | 0.7960 | 0.7598 | 0.6279 | 0.7732 |

Interpretation:

```text
Pure gated sidecar is not stable enough on 10k.

Adding gated residual on top of scalar concat is safe but not materially better:
  gain nDCG@10: 0.6278 -> 0.6279
  gain@1:       0.7731 -> 0.7732
  click AUC:    0.8028 -> 0.8035

The best observed nDCG@10 is still the simpler 12-feature raw sidecar:
  12-feature raw sidecar: 0.6280
  24-feature rel+trainnorm scalar concat: 0.6278
  24-feature rel+trainnorm scalar+gate: 0.6279
```

Conclusion:

```text
The current bottleneck is not just how scalar sidecar features are injected.
The JEPA-derived reaction sidecar contains useful signal, especially for top-1
gain and pairwise gain agreement, but the nDCG@10 lift is saturating.

Next work should improve the JEPA predictor target/objective or train a direct
listwise objective over predicted reaction trajectories, instead of adding more
fusion plumbing around the same sidecar features.
```
