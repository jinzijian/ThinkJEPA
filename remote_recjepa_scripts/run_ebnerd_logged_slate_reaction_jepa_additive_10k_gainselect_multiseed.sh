#!/usr/bin/env bash
set -euo pipefail

cd /home/robotuser/zijian/RecJepa

cache=outputs/ebnerd_small_logged_slate_reaction_jepa_10k_multisignal_v2/logged_reaction_targets
seeds="${SEEDS:-7 11 13}"
metric="${CHECKPOINT_METRIC:-gain_ndcg10}"
slate_aux_weight="${SLATE_AUX_WEIGHT:-0.20}"
predictor_gain_aux_weight="${PREDICTOR_GAIN_AUX_WEIGHT:-0.05}"
additive_latent_scales="${ADDITIVE_LATENT_SCALES:-0.02}"
scale_tag="${additive_latent_scales//./p}"
scale_tag="${scale_tag//,/x}"
skip_baselines_arg="${SKIP_BASELINES_ARG:-}"

for seed in $seeds; do
  out=outputs/ebnerd_small_logged_slate_reaction_jepa_10k_additive_multisignal_v2_${metric}_slate${slate_aux_weight}_scales${scale_tag}_seed${seed}
  mkdir -p "$out"
  rm -f "$out"/results.json "$out"/summary.json "$out"/partial_results.json "$out"/run.log

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" /home/robotuser/miniconda3/bin/python -u ebnerd_logged_slate_reaction_jepa.py \
    --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
    --data-dir data/ebnerd/ebnerd_small \
    --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
    --out-dir "$out" \
    --reaction-target-cache "$cache" \
    --max-samples-per-split 10000 \
    --max-candidates 30 \
    --seed "$seed" \
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
    --checkpoint-metric "$metric" \
    --slate-aux-weight "$slate_aux_weight" \
    --predictor-gain-aux-weight "$predictor_gain_aux_weight" \
    --run-additive-ablation \
    --additive-latent-scales "$additive_latent_scales" \
    $skip_baselines_arg \
    --skip-standalone-jepa \
    --skip-oracle-probe \
    --no-use-item-id-emb \
    2>&1 | tee "$out/run.log"
done
