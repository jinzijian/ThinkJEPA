#!/usr/bin/env bash
set -euo pipefail

cd /home/robotuser/zijian/RecJepa_hstu_jepa

out=outputs/ebnerd_hstu_jepa_10k_h6_rich_primary_tinyjepa_list10
rm -rf "$out"
mkdir -p "$out"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" /home/robotuser/miniconda3/bin/python -u ebnerd_hstu_jepa.py \
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
  --data-dir data/ebnerd/ebnerd_small \
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
  --out-dir "$out" \
  --max-samples-per-split 10000 \
  --run-models E1_HSTU_JEPA \
  --future-horizon-hours 6 \
  --horizon-bucket-hours 1,3,6 \
  --max-history-tokens 128 \
  --max-rank-candidates 256 \
  --max-events-per-impression 20 \
  --num-slots 8 \
  --jepa-pretrain-epochs 3 \
  --jepa-pretrain-patience 2 \
  --pretrain-event-weight 0.02 \
  --pretrain-reaction-weight 1.0 \
  --rich-state-weight 0.5 \
  --rich-item-weight 0.1 \
  --rich-positive-weight 1.0 \
  --epochs 5 \
  --jepa-ranker-epochs 0 \
  --batch-size 64 \
  --d-model 128 \
  --heads 4 \
  --layers 1 \
  --ff-dim 384 \
  --dropout 0.10 \
  --primary-rank-metric engagement_ndcg@10 \
  --no-use-item-id-emb \
  --slot-rank-scale 0.25 \
  --slot-query-scale 5.0 \
  --disable-slot-self-attn \
  --reaction-bce-weight 0.0 \
  --reaction-listwise-weight 10.0 \
  --candidate-bce-weight 0.0 \
  --diversity-weight 0.0 \
  --slot-orth-weight 0.0 \
  --slot-balance-weight 0.0 \
  --slot-sharp-weight 0.0 \
  --no-enable-reaction-blend \
  --no-enable-rich-fusion-head \
  2>&1 | tee "$out/run.log"
