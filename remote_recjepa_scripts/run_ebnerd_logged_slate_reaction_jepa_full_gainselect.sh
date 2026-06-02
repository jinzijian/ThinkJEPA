#!/usr/bin/env bash
set -euo pipefail

cd /home/robotuser/zijian/RecJepa

seed="${SEED:-7}"
metric="${CHECKPOINT_METRIC:-gain_ndcg10}"
slate_aux_weight="${SLATE_AUX_WEIGHT:-0.20}"
predictor_gain_aux_weight="${PREDICTOR_GAIN_AUX_WEIGHT:-0.05}"
additive_latent_scales="${ADDITIVE_LATENT_SCALES:-0.02,0.01}"
qwen_batch_size="${QWEN_BATCH_SIZE:-512}"
max_seq_length="${MAX_SEQ_LENGTH:-256}"
encode_text_chunk_size="${ENCODE_TEXT_CHUNK_SIZE:-131072}"
scale_tag="${additive_latent_scales//./p}"
scale_tag="${scale_tag//,/x}"

out=outputs/ebnerd_small_logged_slate_reaction_jepa_full_${metric}_slate${slate_aux_weight}_scales${scale_tag}_compact_qwen${max_seq_length}_bs${qwen_batch_size}_seed${seed}
cache=outputs/ebnerd_small_logged_slate_reaction_jepa_full_multisignal_v3_compact_reaction_qwen${max_seq_length}/logged_reaction_targets
mkdir -p "$out" "$cache"
rm -f "$out"/results.json "$out"/summary.json "$out"/partial_results.json "$out"/run.log

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}" /home/robotuser/miniconda3/bin/python -u ebnerd_logged_slate_reaction_jepa.py \
  --samples-pkl outputs/ebnerd_small_qwen_lgbm/samples.pkl \
  --data-dir data/ebnerd/ebnerd_small \
  --qwen-raw-cache outputs/ebnerd_small_qwen_uih_no_svd_oracle/qwen_uih_raw_cache.npz \
  --out-dir "$out" \
  --reaction-target-cache "$cache" \
  --max-samples-per-split 0 \
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
  --predictor-epochs 3 \
  --probe-epochs 0 \
  --probe-batch-size 1024 \
  --skip-heuristic \
  --skip-baselines \
  --checkpoint-metric "$metric" \
  --slate-aux-weight "$slate_aux_weight" \
  --predictor-gain-aux-weight "$predictor_gain_aux_weight" \
  --run-additive-ablation \
  --additive-latent-scales "$additive_latent_scales" \
  --skip-standalone-jepa \
  --skip-oracle-probe \
  --no-use-item-id-emb \
  2>&1 | tee "$out/run.log"
