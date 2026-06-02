#!/usr/bin/env bash
set -euo pipefail

cd /home/robotuser/zijian/RecJepa

out=outputs/ebnerd_small_future_item_ranking_10k_h6_attention_r0_cached
rm -rf "$out"
mkdir -p "$out"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" /home/robotuser/miniconda3/bin/python -u ebnerd_event_set_predictor.py \
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
  --ranker-kind attention \
  --epochs 5 \
  --patience 2 \
  --batch-size 64 \
  --d-model 128 \
  --heads 4 \
  --layers 1 \
  --ff-dim 384 \
  --dropout 0.10 \
  --lr 3e-4 \
  --primary-rank-metric engagement_ndcg@10 \
  --loss-variants "" \
  --load-predicted-slots outputs/ebnerd_small_future_item_ranking_10k_h6_r0_rankonly_lgbm_engmetric_clean/R0_rank_only_slots.fp16.npy \
  --loaded-slot-name R0_h6_cached \
  --skip-oracle \
  --skip-train-eval \
  --no-use-item-id-emb \
  --user-profile-mode history_summary \
  --init-slot-dot-scale 0.0 \
  2>&1 | tee "$out/run.log"
