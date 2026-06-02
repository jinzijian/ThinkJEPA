#!/usr/bin/env bash
set -euo pipefail

cd /home/robotuser/zijian/RecJepa

out=outputs/ebnerd_small_ranker_conditioned_reaction_jepa_10k_h6
mkdir -p "$out"
rm -f "$out"/results.json "$out"/summary.json "$out"/predicted_reaction_baseline_order.fp16.npy "$out"/run.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}" /home/robotuser/miniconda3/bin/python -u ebnerd_ranker_conditioned_reaction_jepa.py \
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
  --data-dir data/ebnerd/ebnerd_small \
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
  --out-dir "$out" \
  --reaction-target-cache "$out/reaction_candidate_targets" \
  --target-cache-format npy_dir \
  --max-samples-per-split 10000 \
  --max-candidates 30 \
  --future-horizon-hours 6 \
  --max-future-impressions 8 \
  --max-events-per-impression 20 \
  --source-mode history_weighted \
  --max-history-tokens 128 \
  --user-profile-mode history_summary \
  --input-mode full \
  --target-order-caches oracle,baseline \
  --target-list-context-items 30 \
  --qwen-model models/Qwen3-Embedding-4B \
  --qwen-device cuda:0 \
  --qwen-batch-size 32 \
  --qwen-multi-process-devices cuda:0,cuda:1,cuda:2 \
  --max-seq-length 1024 \
  --encode-text-chunk-size 8192 \
  --device cuda \
  --batch-size 64 \
  --num-workers 4 \
  --baseline-epochs 2 \
  --predictor-epochs 5 \
  --ranker-epochs 2 \
  --probe-epochs 2 \
  --probe-batch-size 1024 \
  --primary-rank-metric engagement_ndcg@10 \
  --no-use-item-id-emb \
  2>&1 | tee "$out/run.log"
