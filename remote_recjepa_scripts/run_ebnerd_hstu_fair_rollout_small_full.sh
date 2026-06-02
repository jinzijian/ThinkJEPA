#!/usr/bin/env bash
set -euo pipefail

cd /home/robotuser/zijian/RecJepa_hstu_jepa

out=outputs/ebnerd_hstu_fair_rollout_small_full
rm -rf "$out"
mkdir -p "$out"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" /home/robotuser/miniconda3/bin/python -u ebnerd_hstu_jepa.py \
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
  --data-dir data/ebnerd/ebnerd_small \
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
  --out-dir "$out" \
  --max-samples-per-split 0 \
  --run-models R0_HSTU,E5_HSTU_TRANSITION \
  --rank-task current_click \
  --target-mode current_future \
  --future-horizon-hours 6 \
  --next-request-max-gap-hours 24 \
  --rollout-steps 3 \
  --max-next-rank-candidates 128 \
  --max-history-tokens 128 \
  --max-rank-candidates 256 \
  --max-events-per-impression 20 \
  --num-slots 8 \
  --epochs 3 \
  --patience 2 \
  --batch-size 64 \
  --d-model 128 \
  --heads 4 \
  --layers 1 \
  --ff-dim 384 \
  --dropout 0.10 \
  --primary-rank-metric ndcg@10 \
  --no-use-item-id-emb \
  --current-rank-weight 0.0 \
  --transition-weight 1.0 \
  --transition-residual-scale 0.5 \
  --transition-score-residual \
  --transition-score-scale 0.5 \
  --transition-init-from-r0 \
  --freeze-base-for-transition \
  --jepa-pretrain-epochs 0 \
  --jepa-weight 0.0 \
  --reaction-weight 0.0 \
  --consequence-weight 0.0 \
  --candidate-bce-weight 0.0 \
  --diversity-weight 0.0 \
  --slot-orth-weight 0.0 \
  --slot-balance-weight 0.0 \
  --slot-sharp-weight 0.0 \
  --no-enable-reaction-blend \
  --no-enable-rich-fusion-head \
  --no-enable-lgbm-reranker \
  2>&1 | tee "$out/run.log"
