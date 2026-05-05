#!/bin/bash
#SBATCH --partition=camas
#SBATCH --job-name=inf_syn_c
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=sn17
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=160000

set -euo pipefail

module load anaconda3
source activate /data/lab/yan/peihong_li/condaenvlist/bd3lm
cd /data/lab/yan/peihong_li/ACL/structure_ABD_rouge

CELL=${CELL:-main}  # cinf_only / c_only_train / main
BLOCK_SIZE=${BLOCK_SIZE:-4}
NUM_SAMPLES=${NUM_SAMPLES:-100}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
METRIC=${METRIC:-rouge}
DATA_CACHE_DIR=${DATA_CACHE_DIR:-/data/lab/yan/peihong_li/data_cache/ndp_pilot}
SYN_RAW_DIR=${SYN_RAW_DIR:-${DATA_CACHE_DIR}/raw/synthetic_topic_skeleton}
LOG_ROOT=${LOG_ROOT:-$PWD/sample_logs/ndp_pilot/synthetic_topic_skeleton}

BASELINE_CKPT=${BASELINE_CKPT:-/data/lab/yan/peihong_li/ACL/block_diff_cond/outputs/ndp_pilot/synthetic_topic_skeleton/baseline_bs${BLOCK_SIZE}/checkpoints/best.ckpt}
STRUCTURED_CKPT=${STRUCTURED_CKPT:-$PWD/outputs/ndp_pilot/synthetic_topic_skeleton/structured_bs${BLOCK_SIZE}/checkpoints/best.ckpt}

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${DATA_CACHE_DIR}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "$LOG_ROOT" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

COMMON_C_ARGS=(
  algo.structured_masking.enabled=True
  algo.structured_masking.r_low=0.15
  algo.structured_masking.r_high=0.85
  algo.structured_masking.global_t=True
  algo.structured_masking.b_max_tokens=32
  algo.structured_masking.full_bidir_attention=True
  algo.span_loss.enabled=False
)

CINF_ARGS=(
  algo.structured_inference.enabled=True
  algo.structured_inference.aggregation=mean
  algo.structured_inference.commitment=mixed
  algo.structured_inference.threshold=fixed_ratio
  sampling.first_hitting=False
)

if [[ "$CELL" == "cinf_only" ]]; then
  CHECKPOINT_PATH="$BASELINE_CKPT"
  EXTRA_ARGS=("${COMMON_C_ARGS[@]}" "${CINF_ARGS[@]}")
elif [[ "$CELL" == "c_only_train" ]]; then
  CHECKPOINT_PATH="$STRUCTURED_CKPT"
  EXTRA_ARGS=("${COMMON_C_ARGS[@]}" algo.structured_inference.enabled=False sampling.first_hitting=False)
elif [[ "$CELL" == "main" ]]; then
  CHECKPOINT_PATH="$STRUCTURED_CKPT"
  EXTRA_ARGS=("${COMMON_C_ARGS[@]}" "${CINF_ARGS[@]}")
else
  echo "CELL must be cinf_only, c_only_train, or main. Got: $CELL" >&2
  exit 2
fi

python -u main.py \
  model=small \
  mode=sample_eval \
  loader.eval_batch_size="$EVAL_BATCH_SIZE" \
  model.length=1024 \
  model.attn_backend=sdpa \
  block_size="$BLOCK_SIZE" \
  algo=bd3lm \
  algo.T=5000 \
  data=synthetic_topic_skeleton \
  data.cache_dir="$DATA_CACHE_DIR" \
  data.local_data_dir="$SYN_RAW_DIR" \
  data.prefix_max_tokens=64 \
  data.answer_max_tokens=957 \
  data.conditional_generation=True \
  wandb=null \
  sampling.nucleus_p=0.9 \
  sampling.kv_cache=False \
  sampling.num_eval_samples="$NUM_SAMPLES" \
  sampling.shuffle_valid=False \
  +sampling.context_size=1024 \
  sampling.logdir="$LOG_ROOT/${CELL}_${NUM_SAMPLES}samples.csv" \
  diagnostics.enabled=True \
  diagnostics.save_path="$LOG_ROOT/${CELL}_${NUM_SAMPLES}samples_diagnostics.json" \
  'diagnostics.snapshot_reveal_fractions=[0.1,0.3,0.5,0.7]' \
  'diagnostics.snapshot_step_fractions=[0.1,0.3,0.5,0.7]' \
  diagnostics.early_fraction=0.3 \
  eval.conditional_metric="$METRIC" \
  eval.checkpoint_path="$CHECKPOINT_PATH" \
  "${EXTRA_ARGS[@]}"
