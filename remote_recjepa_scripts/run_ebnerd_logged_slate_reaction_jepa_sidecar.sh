#!/usr/bin/env bash
set -euo pipefail

cd /home/robotuser/zijian/RecJepa

seed="${SEED:-7}"
metric="${CHECKPOINT_METRIC:-gain_ndcg10}"
max_samples="${MAX_SAMPLES_PER_SPLIT:-0}"
epochs="${PREDICTOR_EPOCHS:-3}"
sidecar_latent_scale="${SIDECAR_LATENT_SCALE:-0.01}"
slate_aux_weight="${SLATE_AUX_WEIGHT:-0.20}"
predictor_gain_aux_weight="${PREDICTOR_GAIN_AUX_WEIGHT:-0.05}"
qwen_batch_size="${QWEN_BATCH_SIZE:-512}"
max_seq_length="${MAX_SEQ_LENGTH:-256}"
encode_text_chunk_size="${ENCODE_TEXT_CHUNK_SIZE:-131072}"
sidecar_train_normalize="${SIDECAR_TRAIN_NORMALIZE:-1}"
sidecar_fusion_mode="${SIDECAR_FUSION_MODE:-scalar_concat}"
sidecar_only_from_cache="${SIDECAR_ONLY_FROM_CACHE:-0}"
sidecar_feature_cache="${SIDECAR_FEATURE_CACHE:-}"
scale_tag="${sidecar_latent_scale//./p}"
sidecar_tag="relraw"
sidecar_norm_flag=()
if [[ "$sidecar_train_normalize" == "1" || "$sidecar_train_normalize" == "true" || "$sidecar_train_normalize" == "TRUE" ]]; then
  sidecar_tag="reltnorm"
  sidecar_norm_flag=(--sidecar-train-normalize)
fi
sidecar_cache_flag=()
if [[ -n "$sidecar_feature_cache" ]]; then
  sidecar_cache_flag=(--sidecar-feature-cache "$sidecar_feature_cache")
fi
sidecar_only_flag=()
if [[ "$sidecar_only_from_cache" == "1" || "$sidecar_only_from_cache" == "true" || "$sidecar_only_from_cache" == "TRUE" ]]; then
  sidecar_only_flag=(--sidecar-only-from-cache)
fi

if [[ "$max_samples" == "0" ]]; then
  split_tag="full"
  cache=outputs/ebnerd_small_logged_slate_reaction_jepa_full_multisignal_v3_compact_reaction_qwen${max_seq_length}/logged_reaction_targets
else
  split_tag="${max_samples}"
  cache=outputs/ebnerd_small_logged_slate_reaction_jepa_${split_tag}_multisignal_v3_compact_reaction_qwen${max_seq_length}/logged_reaction_targets
fi

out=outputs/ebnerd_small_logged_slate_reaction_jepa_sidecar_${split_tag}_${metric}_${sidecar_tag}_${sidecar_fusion_mode}_w${scale_tag}_qwen${max_seq_length}_bs${qwen_batch_size}_seed${seed}
mkdir -p "$out" "$cache"
rm -f "$out"/results.json "$out"/summary.json "$out"/partial_results.json "$out"/run.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}" /home/robotuser/miniconda3/bin/python -u ebnerd_logged_slate_reaction_jepa.py \
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
  --data-dir data/ebnerd/ebnerd_small \
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
  --out-dir "$out" \
  --reaction-target-cache "$cache" \
  --max-samples-per-split "$max_samples" \
  --max-candidates 30 \
  --seed "$seed" \
  --source-mode history_weighted \
  --max-history-tokens 128 \
  --user-profile-mode history_summary \
  --target-neighbor-context 3 \
  --target-text-mode compact_reaction \
  --qwen-model models/Qwen3-Embedding-4B \
  --qwen-device cuda:0 \
  --qwen-batch-size "$qwen_batch_size" \
  --qwen-multi-process-devices cuda:0,cuda:1,cuda:2 \
  --max-seq-length "$max_seq_length" \
  --encode-text-chunk-size "$encode_text_chunk_size" \
  --device cuda \
  --batch-size 64 \
  --num-workers 4 \
  --predictor-epochs "$epochs" \
  --probe-epochs 0 \
  --probe-batch-size 1024 \
  --skip-heuristic \
  --skip-baselines \
  --checkpoint-metric "$metric" \
  --slate-aux-weight "$slate_aux_weight" \
  --predictor-gain-aux-weight "$predictor_gain_aux_weight" \
  --run-sidecar-ablation \
  --sidecar-latent-scale "$sidecar_latent_scale" \
  "${sidecar_norm_flag[@]}" \
  "${sidecar_cache_flag[@]}" \
  "${sidecar_only_flag[@]}" \
  --sidecar-fusion-mode "$sidecar_fusion_mode" \
  --skip-standalone-jepa \
  --skip-oracle-probe \
  --no-use-item-id-emb \
  2>&1 | tee "$out/run.log"
