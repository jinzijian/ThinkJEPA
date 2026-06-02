#!/usr/bin/env bash
set -euo pipefail

cd /home/robotuser/zijian/RecJepa

cache=outputs/ebnerd_small_reaction_qwen_text_h6_1k/reaction_qwen_events.npz

common_args=(
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl
  --data-dir data/ebnerd/ebnerd_small
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz
  --reaction-qwen-cache "$cache"
  --event-target-embedding reaction_qwen_text
  --target-mode future_only
  --rank-task future_engagement
  --source-mode history_weighted
  --max-samples-per-split 1000
  --max-history-tokens 128
  --future-horizon-hours 6
  --max-events-per-impression 20
  --ranker-kind lgbm
  --no-lgbm-binary-labels
  --lgbm-gain-scale 2.0
  --lgbm-max-gain-label 15
  --primary-rank-metric engagement_ndcg@10
  --score-mode plain
  --slot-output-mode direct
  --no-use-item-id-emb
  --slot-nce-weight 1.0
  --assignment-weight 1.0
  --candidate-bce-weight 0.0
  --candidate-kl-weight 0.0
  --diversity-weight 2.0
  --diversity-margin 0.2
  --skip-train-eval
  --skip-rank-slot-blend
  --skip-source-slot-blend
  --skip-oracle
  --skip-baseline-rankers
  --save-predicted-slots
)

e2_out=outputs/ebnerd_small_future_item_ranking_1k_h6_reaction_qwen_e2_direct_joint
rm -rf "$e2_out"
mkdir -p "$e2_out"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" /home/robotuser/miniconda3/bin/python -u ebnerd_event_set_predictor.py \
  "${common_args[@]}" \
  --out-dir "$e2_out" \
  --loss-variants E2_event_nce_bpr \
  --predictor-epochs 3 \
  --predictor-patience 1 \
  --predictor-selection-metric rank_head_ndcg \
  --rank-head-weight 1.0 \
  2>&1 | tee "$e2_out/run.log"

r0_out=outputs/ebnerd_small_future_item_ranking_1k_h6_reaction_qwen_r0_direct
rm -rf "$r0_out"
mkdir -p "$r0_out"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" /home/robotuser/miniconda3/bin/python -u ebnerd_event_set_predictor.py \
  "${common_args[@]}" \
  --out-dir "$r0_out" \
  --loss-variants R0_rank_only \
  --predictor-epochs 3 \
  --predictor-patience 1 \
  --predictor-selection-metric rank_head_ndcg \
  --nce-weight 0.0 \
  --slot-nce-weight 0.0 \
  --assignment-weight 0.0 \
  --bpr-weight 0.0 \
  --candidate-kl-weight 0.0 \
  --rank-label-weight 0.0 \
  --candidate-bce-weight 0.0 \
  --diversity-weight 0.0 \
  --rank-head-weight 1.0 \
  2>&1 | tee "$r0_out/run.log"
