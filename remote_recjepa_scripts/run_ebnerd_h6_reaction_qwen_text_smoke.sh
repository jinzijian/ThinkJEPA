#!/usr/bin/env bash
set -euo pipefail

cd /home/robotuser/zijian/RecJepa

cache_dir=outputs/ebnerd_small_reaction_qwen_text_h6_1k
cache="$cache_dir/reaction_qwen_events.npz"
mkdir -p "$cache_dir"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" /home/robotuser/miniconda3/bin/python -u ebnerd_reaction_qwen_cache.py \
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
  --data-dir data/ebnerd/ebnerd_small \
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
  --out-cache "$cache" \
  --target-mode future_only \
  --max-samples-per-split 1000 \
  --future-horizon-hours 6 \
  --max-events-per-impression 20 \
  --qwen-model models/Qwen3-Embedding-4B \
  --qwen-device cuda \
  --qwen-batch-size 32 \
  --encode-text-chunk-size 2048 \
  2>&1 | tee "$cache_dir/cache.log"

e3_out=outputs/ebnerd_small_future_item_ranking_1k_h6_reaction_qwen_e3
rm -rf "$e3_out"
mkdir -p "$e3_out"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" /home/robotuser/miniconda3/bin/python -u ebnerd_event_set_predictor.py \
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
  --data-dir data/ebnerd/ebnerd_small \
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
  --reaction-qwen-cache "$cache" \
  --event-target-embedding reaction_qwen_text \
  --out-dir "$e3_out" \
  --target-mode future_only \
  --rank-task future_engagement \
  --source-mode history_weighted \
  --max-samples-per-split 1000 \
  --max-history-tokens 128 \
  --future-horizon-hours 6 \
  --max-events-per-impression 20 \
  --ranker-kind lgbm \
  --no-lgbm-binary-labels \
  --lgbm-gain-scale 2.0 \
  --lgbm-max-gain-label 15 \
  --primary-rank-metric engagement_ndcg@10 \
  --loss-variants E3_event_hybrid \
  --predictor-epochs 3 \
  --predictor-patience 1 \
  --predictor-selection-metric candidate_ndcg \
  --score-mode slot_dot \
  --init-slot-dot-scale 2.0 \
  --slot-output-mode candidate_attention \
  --no-use-item-id-emb \
  --slot-nce-weight 1.0 \
  --assignment-weight 1.0 \
  --candidate-bce-weight 2.0 \
  --diversity-weight 2.0 \
  --diversity-margin 0.2 \
  --skip-train-eval \
  --save-predicted-slots \
  2>&1 | tee "$e3_out/run.log"

e3_joint_out=outputs/ebnerd_small_future_item_ranking_1k_h6_reaction_qwen_e3_joint
rm -rf "$e3_joint_out"
mkdir -p "$e3_joint_out"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" /home/robotuser/miniconda3/bin/python -u ebnerd_event_set_predictor.py \
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
  --data-dir data/ebnerd/ebnerd_small \
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
  --reaction-qwen-cache "$cache" \
  --event-target-embedding reaction_qwen_text \
  --out-dir "$e3_joint_out" \
  --target-mode future_only \
  --rank-task future_engagement \
  --source-mode history_weighted \
  --max-samples-per-split 1000 \
  --max-history-tokens 128 \
  --future-horizon-hours 6 \
  --max-events-per-impression 20 \
  --ranker-kind lgbm \
  --no-lgbm-binary-labels \
  --lgbm-gain-scale 2.0 \
  --lgbm-max-gain-label 15 \
  --primary-rank-metric engagement_ndcg@10 \
  --loss-variants E3_event_hybrid \
  --predictor-epochs 3 \
  --predictor-patience 1 \
  --predictor-selection-metric rank_candidate_mix \
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
  --skip-train-eval \
  --save-predicted-slots \
  2>&1 | tee "$e3_joint_out/run.log"

r0_out=outputs/ebnerd_small_future_item_ranking_1k_h6_reaction_qwen_r0
rm -rf "$r0_out"
mkdir -p "$r0_out"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" /home/robotuser/miniconda3/bin/python -u ebnerd_event_set_predictor.py \
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
  --data-dir data/ebnerd/ebnerd_small \
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
  --reaction-qwen-cache "$cache" \
  --event-target-embedding reaction_qwen_text \
  --out-dir "$r0_out" \
  --target-mode future_only \
  --rank-task future_engagement \
  --source-mode history_weighted \
  --max-samples-per-split 1000 \
  --max-history-tokens 128 \
  --future-horizon-hours 6 \
  --max-events-per-impression 20 \
  --ranker-kind lgbm \
  --no-lgbm-binary-labels \
  --lgbm-gain-scale 2.0 \
  --lgbm-max-gain-label 15 \
  --primary-rank-metric engagement_ndcg@10 \
  --loss-variants R0_rank_only \
  --predictor-epochs 3 \
  --predictor-patience 1 \
  --predictor-selection-metric rank_head_ndcg \
  --score-mode slot_dot \
  --init-slot-dot-scale 2.0 \
  --slot-output-mode candidate_attention \
  --no-use-item-id-emb \
  --nce-weight 0.0 \
  --slot-nce-weight 0.0 \
  --assignment-weight 0.0 \
  --bpr-weight 0.0 \
  --candidate-kl-weight 0.0 \
  --rank-label-weight 0.0 \
  --candidate-bce-weight 0.0 \
  --diversity-weight 0.0 \
  --rank-head-weight 1.0 \
  --diversity-margin 0.2 \
  --skip-train-eval \
  --skip-rank-slot-blend \
  --skip-source-slot-blend \
  --save-predicted-slots \
  2>&1 | tee "$r0_out/run.log"
