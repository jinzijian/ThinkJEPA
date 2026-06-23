# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/logo-dark.png">
    <img src="logo/logo.png" alt="ThinkJEPA logo" width="420">
  </picture>
</p>

<p align="center"><strong>Official implementation of ThinkJEPA.</strong></p>

<p align="center">
  <a href="README.zh-CN.md"><strong>中文研究草稿</strong></a>
</p>

<p align="center">
  Haichao Zhang<sup>1</sup>, Yijiang Li<sup>2</sup>, Shwai He<sup>3</sup>, Tushar Nagarajan<sup>4</sup>, Mingfei Chen<sup>5</sup>,<br>
  Jianglin Lu<sup>1</sup>, Ang Li<sup>3</sup>, and Yun Fu<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> Northeastern University
  &nbsp;&nbsp;
  <sup>2</sup> University of California San Diego
  &nbsp;&nbsp;
  <sup>3</sup> University of Maryland
  &nbsp;&nbsp;
  <sup>4</sup> The University of Texas at Austin
  &nbsp;&nbsp;
  <sup>5</sup> University of Washington
</p>

<p align="center">
  For questions about this public release, or if you encounter any issues reproducing the released setup, please contact Haichao Zhang:
  <a href="mailto:zhang.haich@northeastern.edu">zhang.haich@northeastern.edu</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.22281"><strong>Paper</strong></a> |
  <a href="https://github.com/Hai-chao-Zhang/ThinkJEPA"><strong>GitHub</strong></a> |
  <a href="https://huggingface.co/datasets/haichaozhang/cache"><strong>Released Preprocessed Cache</strong></a> |
  <a href="#citation"><strong>Citation</strong></a> |
  <a href="#license"><strong>License</strong></a>
</p>

ThinkJEPA is a dual-path embodied prediction framework in which a vision-language model acts as a cortex-like reasoner for high-level semantics and long-horizon intent, while a JEPA branch acts as a cerebellum-like controller for low-level dynamics, physical consistency, and rapid local correction. This repository is the public ThinkJEPA release for reproducing the released training and evaluation setup on EgoDex-style data and cache inputs.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/thinkjepa-dark.png">
    <img src="logo/thinkjepa.png" alt="ThinkJEPA architecture" width="1100">
  </picture>
</p>

## Overview

- The VLM-thinker branch provides high-level reasoning guidance from Qwen3-VL-Thinking features.
- The dense JEPA branch models video dynamics and supplies low-level embodied features for prediction.
- The released training path predicts future trajectory outputs from JEPA features conditioned by pyramid guidance from the VLM branch.
- This public snapshot is intentionally minimal: it includes the core train/eval code, preprocessing scripts, retained EgoDex helpers, and a bundled `vjepa2/` dependency subtree required by the released path.

## Research Draft: Dynamic K For Latent World-Model Planning

This repository is also tracking an ongoing TTJepa research direction on
dynamic test-time compute for latent world-model planning. The working question
is:

> Can raw latent prediction error decide when an imagined transition needs more
> recurrent refinement steps?

### Motivation

Latent model-predictive control usually spends test-time compute on wider CEM
sampling, more optimizer iterations, or longer rollout horizons. We study a
different axis: the number of recurrent refinement steps `K` used inside each
imagined transition.

This matters because contact-heavy manipulation has uneven transition
difficulty. Some transitions are free-space and should be cheap. Others involve
object contact, occlusion, or multi-object binding and may need additional
latent dynamics refinement. A fixed large `K` wastes compute on easy
transitions; a fixed small `K` misses hard contact cases.

### Method

We start from LeWM-style latent planning: encode the current visual state and
goal, imagine action sequences in latent space, and choose actions with CEM
using a goal-matching cost. TTJepa changes only the transition predictor:

1. The predictor is recurrent and weight-tied across refinement depth.
2. A fixed-depth run uses the same `K` for every imagined transition, for
   example `K=1`, `K=2`, or `K=4`.
3. A dynamic-depth run decides whether to continue refining after each depth.
4. The paper focus is this dynamic choice of `K`, not a new action space,
   tokenizer, or planner.

The core hypothesis is that useful test-time compute should be allocated to the
small subset of transitions where deeper latent dynamics changes planning
success.

### LeWM Baselines

Current working-run results are below. The source matters: different
checkpoints and sweeps should not be merged into one fixed-depth comparison.
The table prioritizes the recurrent checkpoints used by the raw-MSE analysis;
depths not evaluated for the same checkpoint are marked `n/a`.

`LeWM baseline` and `Fixed K1` are not the same model. LeWM baseline uses the
original non-recurrent transition predictor. Fixed K1 uses the TTJepa recurrent
predictor but stops after its first refinement step. They share the latent
planning / CEM evaluation setup, but the predictor architecture, training
objective, and checkpoint differ. For dynamic-K claims, the fair internal
comparison is within the same TTJepa checkpoint: fixed K1/K2/K3/K4 versus
dynamic K. LeWM baseline is an external reference.

| Dataset / run | LeWM baseline | Fixed K1 | Fixed K2 | Fixed K3 | Fixed K4 | Main observation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reacher seed42 | 80% | 88% | n/a | n/a | 86% | K4 is worse than K1 in this checkpoint |
| Cube single seed42 | 72% | 80% | n/a | n/a | 78% | K4 is slightly lower than K1 |
| Cube single seed43 | 72% | 88% | n/a | n/a | 90% | K4 improves over K1 by 2 points |
| Cube single seed44 | 72% | 66% | n/a | n/a | 64% | K4 is slightly lower than K1 |
| Cube single 3-seed avg | 72% | 78% | n/a | n/a | 77.3% | Average K4 is slightly lower, but seed43 shows K4 can help |
| Cube single original rerun `20260621_refixed_k1234` | 72% | 80% | 76% | 78% | 78% | Same original checkpoint with K1-4 completed; K1 is best, K3/K4 recover to 78% |
| Cube double original rerun `20260621_refixed_k1234` | 66% | 72% | 70% | 68% | 70% | Same original checkpoint; extra depth does not help |
| Cube triple original | 74% | 70% | 76% | 76% | 78% | Clearest setting where deeper K helps; most gain appears by K2 |

These are current working-run numbers, not final matched multi-seed benchmark
statistics. The raw-MSE analysis below uses 50 episodes for Reacher,
Cube double, and Cube triple, and 150 episodes for Cube single.

Additional source notes:

- Cube single results were not lost; they are scattered across several naming
  conventions. The table now lists seed42/43/44 separately and adds a
  same-source original-checkpoint rerun with `K1=80%`, `K2=76%`, `K3=78%`, and
  `K4=78%`. The conclusion is not that K4 never helps, but that its benefit is
  unstable in the current runs.
- Cube double's original raw-MSE checkpoint now has same-source `K1=72%`,
  `K2=70%`, `K3=68%`, and `K4=70%`.
- Cube triple has the cleanest same-source fixed-depth sweep:
  `K1=70%`, `K2=76%`, `K3=76%`, and `K4=78%`. Exploratory whitened,
  probe-weighted, and learned-selector checkpoints are preserved in
  [TTJEPA_EXPERIMENT_RESULTS.md](TTJEPA_EXPERIMENT_RESULTS.md), not mixed into
  the Paper 1 main table.

The accurate statement is not that K4 is always better. Fixed-depth gains are
dataset- and checkpoint-dependent. Cube-triple is the clearest current evidence
that transition refinement depth changes task success; Cube single has
seed-level gains, but the original-checkpoint rerun still favors K1; Reacher
and Cube double do not need large K in the current checkpoints.

### First Method: Raw Latent MSE Stopping

The first dynamic-K method uses raw latent MSE as the continue signal. For a
transition, compare the latent prediction error after a shallow depth against
the error after deeper refinement. If deeper refinement reduces latent MSE by
enough, use a deeper `K`; otherwise stop early.

This is a reasonable first attempt because it directly asks whether recurrent
refinement improves the learned latent prediction. It is also deliberately
simple: no planner features, no task-specific probes, and no extra learned
selector.

The table below is specifically a K1/K4 dynamic-selection analysis: each
episode either stays at K1 or switches to K4 based on raw latent MSE.

| Dataset | Fixed K1 | Fixed K4 | Best raw-MSE dynamic K | Hindsight K1/K4 chooser | Depth-helped cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reacher | 88%@K1.00 | 86%@K4.00 | 88%@K1.06 to K2.32 | 92%@K1.12 | 2 / 50 |
| Cube single | 78%@K1.00 | 77.3%@K4.00 | 77.3%@K2.72 to K2.96 | 80.7%@K1.08 | 4 / 150 |
| Cube double | 72%@K1.00 | 70%@K4.00 | 72%@K1.00 to K2.62 | 72%@K1.00 | 0 / 50 |
| Cube triple | 70%@K1.00 | 78%@K4.00 | 76%@K2.32 | 82%@K1.36 | 6 / 50 |

Analysis:

- Raw latent MSE is not a useless signal. On cube-triple it recovers a real
  portion of the K4 gain: `70% -> 76%`, while using mean `K=2.32`.
- It does not reach fixed K4 `78%`, and it is still below the hindsight K1/K4
  chooser `82%@K1.36`.
- On Reacher and Cube double, fixed K4 is not better than K1, so raw MSE mostly
  spends extra compute without improving success. Cube single is more mixed:
  one seed improves with K4, but the 3-seed average is slightly lower than K1.
- The failure mode is alignment: raw latent MSE measures representation error,
  not planner benefit. It can miss task-relevant contact details if the latent
  space is smoothed, anisotropic, or partially collapsed around features that
  are easy to predict but not decisive for CEM action selection.

The conclusion is that raw latent MSE is a useful v0 and a strong diagnostic,
but it is not a sufficient final answer for dynamic test-time compute.

### Mechanistic Analysis: Latent Smoothing And Planner Alignment

We tested the main failure hypothesis that deeper recurrent refinement may
reduce generic latent prediction error by globally smoothing away task-relevant
contact details. The first analysis pass recomputes `K1/K2/K3/K4` predicted
latents on the same K-refinement evaluation windows, then measures latent
spectrum/effective rank and linear state-probe quality. Artifacts are under
`analysis/k_smoothing_20260622`.

![Spectrum K1 vs K4 scatter](analysis/k_smoothing_20260622/figures/spectrum_k1_vs_k4_scatter.png)

![Probe R2 K1 vs K4 scatter](analysis/k_smoothing_20260622/figures/probe_r2_k1_vs_k4_scatter.png)

![Category probe MSE K1 vs K4 scatter](analysis/k_smoothing_20260622/figures/category_probe_mse_k1_vs_k4_scatter.png)

The result is useful but more subtle than the initial hypothesis:

- Global spectrum is almost unchanged with depth. Across reacher, cube-single,
  cube-double, and cube-triple, the `K4/K1` entropy-rank ratio is essentially
  `1.000`; total variance and top singular-vector concentration also barely
  move.
- Linear state probes are also nearly unchanged. Cube-single block position
  stays at `R2=0.991` for both `K1` and `K4`; cube-double block position changes
  from `0.946` to `0.946`; cube-triple block position changes from `0.902` to
  `0.902`. Other probe deltas are mostly at the third decimal place.
- Depth-helped and depth-hurt subsets do not show a clean global collapse
  signature. Their spectra and probe errors move only slightly, so the current
  evidence does not support a strong claim that large `K` globally compresses
  the latent representation.

This changes the interpretation. The failure mode is less likely to be a broad
latent-rank collapse, and more likely to be a local planner-alignment problem:
small changes in imagined transitions can change CEM elite ranking or selected
actions without visibly changing global spectrum or simple linear probes.

The remaining decisive analysis is therefore CEM ranking stability. For the
same candidate action sequences, evaluate terminal costs with `K1/K2/K3/K4` and
measure top-elite overlap, Kendall rank correlation, and whether the selected
action changes. If latent MSE improves while CEM ranking does not improve, the
extra recurrent depth is not useful planning compute. If cube-triple helped
episodes show ranking correction at larger `K`, that directly explains the
`70% -> 76%` raw-MSE dynamic gain.

### Complete Experiment Ledger

This README is now scoped to Paper 1: fixed-depth recurrent refinement, raw
latent-MSE dynamic K, and the associated failure/mechanistic analysis. Other
experiments are preserved separately and should not be mixed into the main
paper tables:

- learned continue-head / joint marginal-depth runs,
- planner-feature diagnostic selectors,
- whitened and probe-weighted halt-label variants,
- the stronger `rel0005` training-time regularization lead.

See [TTJEPA_EXPERIMENT_RESULTS.md](TTJEPA_EXPERIMENT_RESULTS.md) for the full
experiment ledger, checkpoint paths, result directories, and log paths.

### Current Paper Position

The current paper skeleton is:

1. Define transition refinement depth `K` as a test-time compute axis in latent
   world-model planning.
2. Show that fixed deeper `K` helps on cube-triple but not uniformly across all
   datasets.
3. Introduce raw latent MSE stopping as the first dynamic-K method and show its
   strengths and weaknesses across four datasets.
4. Use hindsight K1/K4 selection to estimate how much headroom raw MSE leaves.
5. Analyze why raw MSE is incomplete: global latent spectrum and linear state
   probes do not show a broad collapse, pointing instead to local
   planner-alignment failures.
6. Identify CEM candidate-ranking stability as the next decisive diagnostic.

### Proposed Paper Outline And Figure Plan

Working title:

> When Should a Latent Planner Refine? Dynamic Transition Depth via Raw Latent Error

Core thesis:

> Test-time compute in latent world-model planning should not only be allocated
> to CEM sampling width, optimizer iterations, or rollout horizon. It can also
> be allocated inside each imagined transition through recurrent refinement
> depth `K`.

| Section | Core argument | Figure / table | What it should show |
| --- | --- | --- | --- |
| 1. Introduction | Manipulation transitions have uneven difficulty: free-space motion is cheap, while contact, grasping, and multi-object interactions need more dynamics refinement. | Fig. 1 motivation cartoon: reaching vs contact-rich grasping; CEM rollout with variable `K`. | The paper is about compute allocation inside latent dynamics, not generic robot reasoning. |
| 2. Background | LeWM-style planning rolls out candidate actions in latent space and optimizes terminal goal cost with CEM. Prior compute axes are `N/I/H`; this paper studies `K`. | Fig. 2 LeWM planner plus recurrent transition-depth axis. | We keep the planner/action space fixed and expose a new test-time compute knob. |
| 3. Method | A weight-tied recurrent transition predictor produces `K1/K2/K3/K4` predictions; dynamic K decides whether another refinement step is worth paying for. | Fig. 3 recurrent refinement cell and stop/continue decision. | `K` is a controlled inference-depth variable, not simply a larger model. |
| 4. Main Results | Fixed deeper `K` clearly helps cube-triple, but not every dataset needs large `K`; therefore K matters, but must be allocated dynamically. | Table 1 LeWM / fixed K / dynamic K; `analysis/paper1_figures/png_direct/main_success_vs_lewm.png`. | Establish the baseline and show that transition depth changes planning success. |
| 5. Raw Latent MSE Dynamic K | Raw latent MSE is a clean first dynamic-K rule. On cube-triple it recovers much of the K4 gain: `70%@K1 -> 76%@K2.32`. | Fig. 4 success-vs-mean-K Pareto: `analysis/paper1_figures/png_direct/raw_mse_tolerance_pareto.png`. | Raw MSE is a real signal, not an empty heuristic. |
| 6. Failure Analysis | Raw MSE does not perfectly match planner benefit. Hindsight K1/K4 selection shows additional dynamic-K headroom. | Fig. 5 outcome split: `analysis/paper1_figures/png_direct/k1_k4_outcome_split.png`; Fig. 6 precision/recall: `analysis/paper1_figures/png_direct/raw_mse_precision_recall_failure.png`. | The failure mode is alignment between latent prediction improvement and action selection. |
| 7. Mechanistic Analysis | Deeper K does not show a broad global latent-collapse signature: spectrum and state probes stay near the `K1=K4` diagonal. | Fig. 7 spectrum scatter; Fig. 8 probe scatter; Fig. 9 category probe MSE scatter under `analysis/k_smoothing_20260622/figures/`. | The issue is likely local planner alignment, not simple global latent smoothing. |
| 8. Discussion | The safe claim is compute allocation with a simple raw-MSE rule, not universal superiority of large K. | Limitations / next-experiment table. | Multi-seed raw-MSE validation, wall-clock, and CEM-ranking traces remain required for strong ICLR claims. |

Open requirements before making strong ICLR-level claims:

- Replicate the raw-MSE dynamic-K analysis across seeds.
- Report raw-MSE precision/recall on K1-fail/K4-success and
  K1-success/K4-fail episodes.
- Add wall-clock latency and recurrent transition-call counts.
- Add CEM ranking stability by depth and outcome category.
- Keep learned-selector and joint-depth results in
  [TTJEPA_EXPERIMENT_RESULTS.md](TTJEPA_EXPERIMENT_RESULTS.md), not the Paper 1
  main tables.

See [TTJepa Dynamic K Research Notes](TTJEPA_DYNAMIC_K_RESEARCH.md) for working
research notes and [TTJepa Experiment Results Ledger](TTJEPA_EXPERIMENT_RESULTS.md)
for the complete result archive.

## Repository Layout

```text
thinkjepa/
├── cache_train/
│   ├── generate_egodex_split_manifest.py
│   ├── build_video_cache_splits.py
│   ├── qwen3_cache_extractor.py
│   ├── qwen3_parallel_cache_extractor.py
│   ├── thinker_train.py
│   ├── thinker_predictor.py
│   ├── omnijepa.py
│   ├── omnijepa_data.py
│   ├── omnijepa_toy_train.py
│   ├── models.py
│   ├── predictor.py
│   ├── hf_egodex.py
│   └── run_main_egodex_suite.py
├── egodex/
├── scripts/
│   ├── train.sh
│   └── eval_main.sh
├── vjepa2/
├── tests/
├── logo/
├── LICENSE
├── NOTICE
├── CITATION.cff
├── CITATION.bib
├── RELEASE_AUDIT.md
├── requirements-extraction.txt
└── requirements-public.txt
```

## Environment Setup

### Train / Eval Environment

For training and evaluation, we recommend a V-JEPA2-aligned environment.
On this machine, the known working training environment was a conda env similar to `vjepa` with:

- Python `3.11.15`
- `torch==2.10.0+cu128`
- `torchvision==0.25.0+cu128`
- `torchaudio==2.10.0+cu128`
- `decord==0.6.0`
- `numpy==2.3.5`
- `h5py==3.16.0`
- `opencv-python==4.13.0.92`
- `pillow==12.0.0`
- `pyyaml==6.0.3`
- `timm==1.0.25`
- `einops==0.8.2`

Example setup:

```bash
conda create -n thinkjepa-train python=3.11 -y
conda activate thinkjepa-train

# Install a PyTorch stack that matches your CUDA runtime and wheel index.
# The local working environment used CUDA 12.8 wheels.
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128

pip install -r requirements-public.txt
```

The release already bundles the `vjepa2/` source subtree used by the documented ThinkJEPA path. By default, the wrapper scripts point `VJEPA2_ROOT` to `./vjepa2`.

The upstream `vjepa2/requirements.txt` contains a broader research stack, including tools such as `tensorboard` and `wandb`. Those extras are not required by the released ThinkJEPA train/eval path.

### Qwen3-VL Extraction Environment

If you want to run cache extraction yourself, use a dedicated Qwen3-VL environment instead of reusing the lean train/eval env. On this machine, the known working extraction environment was a conda env similar to `qwen3vl` with:

- Python `3.10.19`
- `torch==2.10.0`
- `torchvision==0.25.0`
- `torchaudio==2.10.0+cu128`
- `torchcodec==0.10.0+cu128`
- `transformers==5.2.0`
- `qwen-vl-utils==0.0.14`
- `huggingface-hub==1.4.1`
- `decord==0.6.0`
- `numpy==2.2.6`
- `h5py==3.16.0`
- `pillow==12.1.1`
- `matplotlib==3.10.8`
- `pyyaml==6.0.3`
- `accelerate==1.12.0`
- `sentencepiece==0.2.1`
- `safetensors==0.7.0`

Example setup:

```bash
conda create -n qwen3vl python=3.10 -y
conda activate qwen3vl

# Install a PyTorch + torchcodec stack that matches your CUDA runtime and wheel index.
# The local working extraction environment used CUDA 12.8 wheels.
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
pip install torchcodec==0.10.0+cu128

pip install -r requirements-extraction.txt
```

The extraction scripts default to `--force_video_backend torchcodec`. If `torchcodec` is unavailable on your machine, switch the backend to `decord` explicitly.
If you only plan to reproduce training/evaluation from the released Hugging Face cache, you can skip this extraction environment entirely.

### Optional Checkpoint Setup

If you need pretrained JEPA weights, provide them through environment variables instead of editing local paths directly:

```bash
export THINKJEPA_JEPA_VITL_PT=<CHECKPOINT_PATH>
```

## Data And Cache Preparation

### Option A: Use The Released Prepared Cache

We provide a prepared cache release on Hugging Face:

- https://huggingface.co/datasets/haichaozhang/cache

This is the recommended path for reproducing the released ThinkJEPA setup without rebuilding Qwen3-VL features locally. The released scripts accept either:

- a remote Hugging Face reference such as `hf://datasets/haichaozhang/cache/part2`, or
- a local Hugging Face snapshot/cache path that already points to the downloaded `part2` tree.

### Option A1: Use The Remote Hugging Face Reference

```bash
HF_HOME=<HF_HOME> \
DATA_DIR=hf://datasets/haichaozhang/cache/part2 \
CACHE_DIR=hf://datasets/haichaozhang/cache/part2 \
TRAIN_MANIFEST=hf://datasets/haichaozhang/cache/egodex_part2_video_cache_subset2000_ratio0.9_seed42/splits/train_cache.txt \
TEST_MANIFEST=hf://datasets/haichaozhang/cache/egodex_part2_video_cache_subset2000_ratio0.9_seed42/splits/test_cache.txt \
VJEPA2_ROOT=$PWD/vjepa2 \
bash scripts/train.sh
```

### Option A2: Use A Local Hugging Face Snapshot Path

If you have already downloaded the released cache, you can point the scripts directly at the local snapshot directory instead of using `hf://...` references.

Minimal example:

```bash
LOCAL_CACHE_ROOT=<LOCAL_HF_SNAPSHOT>/part2

DATA_DIR=${LOCAL_CACHE_ROOT} \
CACHE_DIR=${LOCAL_CACHE_ROOT} \
VJEPA2_ROOT=$PWD/vjepa2 \
bash scripts/train.sh
```

If you want to use explicit train/test manifests, you can pass them as well:

```bash
LOCAL_CACHE_ROOT=<LOCAL_HF_SNAPSHOT>/part2

DATA_DIR=${LOCAL_CACHE_ROOT} \
CACHE_DIR=${LOCAL_CACHE_ROOT} \
TRAIN_MANIFEST=<OPTIONAL_LOCAL_MANIFEST> \
TEST_MANIFEST=<OPTIONAL_LOCAL_MANIFEST> \
VJEPA2_ROOT=$PWD/vjepa2 \
bash scripts/train.sh
```

The same local-path form also works for evaluation:

```bash
LOCAL_CACHE_ROOT=<LOCAL_HF_SNAPSHOT>/part2

DATA_DIR=${LOCAL_CACHE_ROOT} \
CACHE_DIR=${LOCAL_CACHE_ROOT} \
TRAIN_MANIFEST=<OPTIONAL_LOCAL_MANIFEST> \
TEST_MANIFEST=<OPTIONAL_LOCAL_MANIFEST> \
VJEPA2_ROOT=$PWD/vjepa2 \
bash scripts/eval_main.sh
```

On our side, we smoke-tested the public release with a local Hugging Face snapshot path in addition to the `hf://...` path.

### Option B: Download Raw EgoDex And Build Cache Locally

Raw EgoDex is distributed by the official EgoDex project:

- EgoDex repository: https://github.com/apple/ml-egodex
- EgoDex paper: https://arxiv.org/abs/2505.11709

Example download for `part2`:

```bash
curl "https://ml-site.cdn-apple.com/datasets/egodex/part2.zip" -o part2.zip
unzip part2.zip
```

Expected layout:

```text
<DATA_ROOT>/part2/<task_name>/<episode>.mp4
<DATA_ROOT>/part2/<task_name>/<episode>.hdf5
```

### Generate Split Manifests

```bash
python cache_train/generate_egodex_split_manifest.py \
  --data_root <DATA_ROOT>/part2 \
  --output_dir <SPLIT_ROOT>/part2_ratio0.9_seed42 \
  --glob_pattern "*.hdf5" \
  --train_ratio 0.9 \
  --split_seed 42
```

### Extract Qwen3-VL-Thinking Cache

The released ThinkJEPA setup uses Qwen3-VL-Thinking features. The public release includes:

- `cache_train/qwen3_cache_extractor.py`
- `cache_train/qwen3_parallel_cache_extractor.py`

These scripts are intended to run from the dedicated Qwen3-VL extraction environment described above.

Minimal parallel extraction example:

```bash
python cache_train/qwen3_parallel_cache_extractor.py \
  --file_dir <DATA_ROOT>/part2 \
  --output_dir <CACHE_ROOT>/part2 \
  --pretrained Qwen/Qwen3-VL-2B-Thinking \
  --layers 0 4 8 12 16 20 24 27 \
  --max_frames 32 \
  --max_new_token_num 16 \
  --batch_size 20 \
  --save_dtype fp16 \
  --res 256 \
  --prompt "Describe this video."
```

This produces per-video `.npz` cache files aligned with the EgoDex video tree.

### Build Cache-Aligned Train/Test Manifests

```bash
python cache_train/build_video_cache_splits.py \
  --dataset egodex \
  --data_root <DATA_ROOT>/part2 \
  --cache_root <CACHE_ROOT>/part2 \
  --output_dir <SPLIT_ROOT>/egodex_part2_video_cache_subset2000_ratio0.9_seed42 \
  --subset_size 2000 \
  --train_ratio 0.9 \
  --split_seed 42
```

## Training

The most reliable public-release training path is to invoke `cache_train/thinker_train.py` directly so you can pass `--no_preload_cache_to_memory` explicitly.

`train_batch_size` and `test_batch_size` are **per-GPU** batch sizes.

### Single-GPU Training

```bash
PROJECT_ROOT=/path/to/thinkjepa
cd "${PROJECT_ROOT}"
conda activate qwen3vl

export VJEPA2_ROOT="${PROJECT_ROOT}/vjepa2"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/cache_train:${PROJECT_ROOT}/vjepa2:$(dirname "${PROJECT_ROOT}/vjepa2"):${PYTHONPATH}"

LOCAL_CACHE_ROOT=<LOCAL_HF_SNAPSHOT>/part2

python cache_train/thinker_train.py \
  --data_dir "${LOCAL_CACHE_ROOT}" \
  --cache_dir "${LOCAL_CACHE_ROOT}" \
  --output_dir "${PROJECT_ROOT}/outputs/full_train_run_single" \
  --results_md "${PROJECT_ROOT}/outputs/full_train_run_single/test_results.md" \
  --output_mp4 "${PROJECT_ROOT}/outputs/full_train_run_single/vis/pred" \
  --epochs 50 \
  --predictor thinkjepa \
  --backbone vjepa \
  --optimize_together_downstream \
  --seed 42 \
  --train_ratio 0.9 \
  --split_seed 42 \
  --train_batch_size 16 \
  --test_batch_size 16 \
  --num_workers 4 \
  --prefetch_factor 1 \
  --past_T 32 \
  --future_T 32 \
  --temporal_stride 1 \
  --camera_mode auto \
  --thinkjepa_vlm_source both \
  --thinkjepa_vlm_layer_selector last \
  --thinkjepa_vlm_cond_mode film \
  --thinkjepa_dual_tower \
  --lambda_vlm 0.5 \
  --lambda_mutual 0.1 \
  --lr 1e-3 \
  --lr_pred 1e-4 \
  --lr_vlm 1e-4 \
  --max_visual_batches 1 \
  --use_npz_cache \
  --skip_vjepa \
  --no_preload_cache_to_memory
```

### Multi-GPU Training

The example below uses 4 GPUs with batch size `16` per GPU, so the effective global train batch is `64`.

```bash
PROJECT_ROOT=/path/to/thinkjepa
cd "${PROJECT_ROOT}"
conda activate qwen3vl

export VJEPA2_ROOT="${PROJECT_ROOT}/vjepa2"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/cache_train:${PROJECT_ROOT}/vjepa2:$(dirname "${PROJECT_ROOT}/vjepa2"):${PYTHONPATH}"
export NCCL_NVLS_ENABLE=0

LOCAL_CACHE_ROOT=<LOCAL_HF_SNAPSHOT>/part2

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 cache_train/thinker_train.py \
  --data_dir "${LOCAL_CACHE_ROOT}" \
  --cache_dir "${LOCAL_CACHE_ROOT}" \
  --output_dir "${PROJECT_ROOT}/outputs/full_train_run_4gpu_bs16" \
  --results_md "${PROJECT_ROOT}/outputs/full_train_run_4gpu_bs16/test_results.md" \
  --output_mp4 "${PROJECT_ROOT}/outputs/full_train_run_4gpu_bs16/vis/pred" \
  --epochs 500 \
  --auto_resume \
  --predictor thinkjepa \
  --backbone vjepa \
  --optimize_together_downstream \
  --seed 42 \
  --train_ratio 0.9 \
  --split_seed 42 \
  --train_batch_size 16 \
  --test_batch_size 16 \
  --num_workers 4 \
  --prefetch_factor 1 \
  --past_T 32 \
  --future_T 32 \
  --temporal_stride 1 \
  --camera_mode auto \
  --thinkjepa_vlm_source both \
  --thinkjepa_vlm_layer_selector last \
  --thinkjepa_vlm_cond_mode film \
  --thinkjepa_dual_tower \
  --lambda_vlm 0.5 \
  --lambda_mutual 0.1 \
  --lr 1e-3 \
  --lr_pred 1e-4 \
  --lr_vlm 1e-4 \
  --max_visual_batches 1 \
  --use_npz_cache \
  --skip_vjepa \
  --no_preload_cache_to_memory \
  --ddp
```

`DATA_DIR` and `CACHE_DIR` can still be either:

- `hf://datasets/haichaozhang/cache/part2`
- an absolute local Hugging Face snapshot path such as `<LOCAL_HF_SNAPSHOT>/part2`

If you prefer the lightweight wrapper, `scripts/train.sh` is still available:

```bash
DATA_DIR=<DATA_ROOT_OR_HF_SPEC> \
CACHE_DIR=<CACHE_ROOT_OR_HF_SPEC> \
TRAIN_MANIFEST=<TRAIN_MANIFEST_OPTIONAL> \
TEST_MANIFEST=<TEST_MANIFEST_OPTIONAL> \
VJEPA2_ROOT=$PWD/vjepa2 \
bash scripts/train.sh
```

`scripts/train.sh` enables the JEPA/VLM dual-tower path by default for `PREDICTOR=thinkjepa`.
Set `THINKJEPA_DUAL_TOWER=0` to return to the earlier one-way VLM-conditioning setup.

For the public release, we smoke-tested training with both:

- the remote Hugging Face form `hf://datasets/haichaozhang/cache/part2`
- a local Hugging Face snapshot path under `~/.cache/huggingface/.../part2`

## Evaluation

```bash
DATA_DIR=<DATA_ROOT_OR_HF_SPEC> \
CACHE_DIR=<CACHE_ROOT_OR_HF_SPEC> \
TRAIN_MANIFEST=<TRAIN_MANIFEST_OPTIONAL> \
TEST_MANIFEST=<TEST_MANIFEST_OPTIONAL> \
VJEPA2_ROOT=$PWD/vjepa2 \
bash scripts/eval_main.sh
```

The same two path forms are supported for evaluation:

- `hf://datasets/haichaozhang/cache/part2`
- an absolute local Hugging Face snapshot path such as `<LOCAL_HF_SNAPSHOT>/part2`

For the public release, we also smoke-tested `scripts/eval_main.sh` with both path forms above.

## OmniJEPA V1 Scaffold

This branch includes a lightweight implementation scaffold for the follow-up OmniJEPA direction:

```text
VLM mid-layer states -> JEPA task guidance -> predicted future JEPA latent
predicted future JEPA latent -> VLM/action conditioning tokens
```

The reusable modules live in `cache_train/omnijepa.py`:

- `VlmToJepaTaskAdapter`: converts VLM hidden states into ThinkJEPA-compatible VLM guidance streams.
- `JepaFutureTokenResampler`: compresses `[B,T,P,D]` future JEPA latents into a small set of VLM-space future tokens.
- `JepaLateFusionAdapter`: lets late VLM layers cross-attend to JEPA future tokens for text/QA/plan outputs.
- `FlowActionExpert`: flow-matching action chunk head for `<MODE=ACT>` robot outputs.
- `OmniJepaBridge`: backbone-agnostic wrapper around the adapters above.

Explicit mode routing is used by design:

```text
<MODE=QA>   -> LM head
<MODE=PLAN> -> LM head
<MODE=ACT>  -> flow action expert
```

Unified cache helpers are in `cache_train/omnijepa_data.py`. Each `.npz` sample should provide:

```text
episode_id
timestep
mode
instruction
obs_frames
current_jepa_latent
predicted_future_jepa_latent
oracle_future_jepa_latent
target_text or target_action_chunk or target_traj_tokens
```

Run the smoke tests with:

```bash
python -m unittest tests.test_omnijepa
```

For a synthetic end-to-end cache/training smoke run, use:

```bash
python cache_train/omnijepa_toy_train.py --generate_toy_data --cache_root /tmp/omnijepa_toy_cache
```

## Third-Party Sources

This release retains third-party components that are necessary for the released reproduction path.

- EgoDex-derived helpers under `egodex/` are adapted from Apple's EgoDex project.
  - Source repository: https://github.com/apple/ml-egodex
  - Retained notice files:
    - `egodex/LICENSE.txt`
    - `egodex/ACKNOWLEDGEMENTS.txt`
    - `egodex/utils/LICENSE.txt`
    - `egodex/utils/ACKNOWLEDGEMENTS.txt`
- The bundled `vjepa2/` subtree is derived from the V-JEPA2 repository.
  - Source repository: https://github.com/facebookresearch/vjepa2
  - Retained notice files:
    - `vjepa2/LICENSE`
    - `vjepa2/APACHE-LICENSE`

These third-party notices continue to apply to their respective subtrees and are not replaced by the root ThinkJEPA release license.

## Citation

If you use ThinkJEPA, please cite the paper and link to the original repository:

- Repository: https://github.com/Hai-chao-Zhang/ThinkJEPA

```bibtex
@article{zhang2026thinkjepa,
  title={ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model},
  author={Zhang, Haichao and Li, Yijiang and He, Shwai and Nagarajan, Tushar and Chen, Mingfei and Lu, Jianglin and Li, Ang and Fu, Yun},
  journal={arXiv preprint arXiv:2603.22281},
  year={2026}
}
```

See `CITATION.cff` and `CITATION.bib` for machine-readable and BibTeX citation metadata.

## Attribution

If you use, modify, or redistribute ThinkJEPA or derivative code, please:

- retain the `LICENSE` and `NOTICE` files
- retain applicable third-party notices that ship with the repository
- cite the ThinkJEPA paper where citation practices apply
- include a link to the original repository: https://github.com/Hai-chao-Zhang/ThinkJEPA

## License

The root repository is released under the custom `ThinkJEPA Attribution License (BSD-3-Clause-based, custom)`. Redistribution and modification are broadly permitted, provided that required attribution, notice retention, change-marking, and repository-link requirements are followed.

See:

- `LICENSE`
- `NOTICE`
- `RELEASE_AUDIT.md`

## Release Scope

This public release intentionally excludes private datasets, unpublished internal manifests, private checkpoints, experiment logs, notebook artifacts, and unrelated experimental code paths. Use the linked GitHub and Hugging Face resources where appropriate.
