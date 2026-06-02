#!/usr/bin/env bash
set -euo pipefail

cd /home/robotuser/zijian/RecJepa

out=outputs/ebnerd_small_logged_slate_reaction_jepa_10k_additive_multisignal_v2
cache=outputs/ebnerd_small_logged_slate_reaction_jepa_10k_multisignal_v2/logged_reaction_targets
mkdir -p "$out" "$cache"
rm -f "$out"/results.json "$out"/summary.json "$out"/partial_results.json "$out"/run.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" /home/robotuser/miniconda3/bin/python -u ebnerd_logged_slate_reaction_jepa.py \
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
  --data-dir data/ebnerd/ebnerd_small \
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
  --out-dir "$out" \
  --reaction-target-cache "$cache" \
  --rebuild-target-cache \
  --max-samples-per-split 10000 \
  --max-candidates 30 \
  --source-mode history_weighted \
  --max-history-tokens 128 \
  --user-profile-mode history_summary \
  --target-neighbor-context 3 \
  --qwen-model models/Qwen3-Embedding-4B \
  --qwen-device cuda:0 \
  --qwen-batch-size 32 \
  --qwen-multi-process-devices cuda:0 \
  --max-seq-length 1024 \
  --encode-text-chunk-size 8192 \
  --device cuda \
  --batch-size 64 \
  --num-workers 4 \
  --predictor-epochs 3 \
  --probe-epochs 0 \
  --probe-batch-size 1024 \
  --run-additive-ablation \
  --additive-latent-scales 0.02,0.05 \
  --skip-standalone-jepa \
  --skip-oracle-probe \
  --no-use-item-id-emb \
  2>&1 | tee "$out/run.log"
