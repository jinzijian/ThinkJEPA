#!/usr/bin/env bash
set -euo pipefail

cd /home/robotuser/zijian/RecJepa

out=outputs/ebnerd_small_logged_slate_reaction_jepa_full
mkdir -p "$out"
rm -f "$out"/results.json "$out"/summary.json "$out"/partial_results.json "$out"/run.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}" /home/robotuser/miniconda3/bin/python -u ebnerd_logged_slate_reaction_jepa.py \
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
  --data-dir data/ebnerd/ebnerd_small \
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
  --out-dir "$out" \
  --reaction-target-cache "$out/logged_reaction_targets" \
  --max-samples-per-split 0 \
  --max-candidates 30 \
  --source-mode history_weighted \
  --max-history-tokens 128 \
  --user-profile-mode history_summary \
  --target-neighbor-context 3 \
  --qwen-model models/Qwen3-Embedding-4B \
  --qwen-device cuda:0 \
  --qwen-batch-size 32 \
  --qwen-multi-process-devices cuda:0,cuda:1,cuda:2 \
  --max-seq-length 1024 \
  --encode-text-chunk-size 8192 \
  --device cuda \
  --batch-size 64 \
  --num-workers 4 \
  --predictor-epochs 3 \
  --probe-epochs 2 \
  --probe-batch-size 1024 \
  --no-use-item-id-emb \
  2>&1 | tee "$out/run.log"
