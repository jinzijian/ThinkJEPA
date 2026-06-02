#!/usr/bin/env bash
set -euo pipefail

cd /home/robotuser/zijian/RecJepa

out=outputs/ebnerd_small_future_item_ranking_10k_h6_e3_rankjoint_lgbm_engmetric_clean
rm -rf "$out"
mkdir -p "$out"

/home/robotuser/miniconda3/bin/python -u ebnerd_event_set_predictor.py \
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
  --data-dir data/ebnerd/ebnerd_small \
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
  --out-dir "$out" \
  --target-mode future_only \
  --rank-task future_engagement \
  --source-mode history_weighted \
  --max-samples-per-split 10000 \
  --max-history-tokens 128 \
  --future-horizon-hours 6 \
  --max-events-per-impression 20 \
  --ranker-kind lgbm \
  --no-lgbm-binary-labels \
  --lgbm-gain-scale 2.0 \
  --lgbm-max-gain-label 15 \
  --primary-rank-metric engagement_ndcg@10 \
  --loss-variants E3_event_hybrid \
  --predictor-selection-metric rank_candidate_mix \
  --predictor-selection-aux-weight 0.5 \
  --score-mode slot_dot \
  --init-slot-dot-scale 2.0 \
  --slot-output-mode candidate_attention \
  --no-use-item-id-emb \
  --slot-nce-weight 1.0 \
  --assignment-weight 1.0 \
  --candidate-bce-weight 2.0 \
  --rank-head-weight 1.0 \
  --diversity-weight 2.0 \
  --diversity-margin 0.2 \
  --skip-oracle \
  --skip-train-eval \
  --skip-rank-slot-blend \
  --skip-source-slot-blend \
  --save-predicted-slots \
  2>&1 | tee "$out/run.log"
