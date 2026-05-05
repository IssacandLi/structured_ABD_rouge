#!/bin/bash
#SBATCH --partition=camas
#SBATCH --job-name=ndp_wp_c
#SBATCH --time=48:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --nodelist=sn17
#SBATCH --gres=gpu:tesla:2
#SBATCH --mem=300000

set -euo pipefail

module load anaconda3
source activate /data/lab/yan/peihong_li/condaenvlist/bd3lm
cd /data/lab/yan/peihong_li/ACL/structure_ABD_rouge

BLOCK_SIZE=${BLOCK_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-10000}
VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-50}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-kuleshov-group/bd3lm-owt-block_size1024-pretrain}
WP_SAMPLE_FRACTION=${WP_SAMPLE_FRACTION:-0.01}
TRAIN_LIMIT=${TRAIN_LIMIT:-1.0}
VAL_LIMIT=${VAL_LIMIT:-1.0}

export DATA_CACHE_DIR=/data/lab/yan/peihong_li/data_cache/ndp_pilot
export TOKENIZERS_PARALLELISM=false

export HF_HOME="${DATA_CACHE_DIR}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

RUN_ROOT=/data/lab/yan/peihong_li/ACL/structure_ABD_rouge/outputs/ndp_pilot/wp
SAVE_DIR="${RUN_ROOT}/structured_bs${BLOCK_SIZE}"
mkdir -p "$SAVE_DIR"

srun bash -lc '
  set -euo pipefail
  export TRITON_CACHE_DIR='"${DATA_CACHE_DIR}"'/triton/rank${SLURM_PROCID}
  export TORCHINDUCTOR_CACHE_DIR='"${DATA_CACHE_DIR}"'/torchinductor/rank${SLURM_PROCID}
  export XDG_CACHE_HOME='"${DATA_CACHE_DIR}"'/xdg/rank${SLURM_PROCID}
  mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
  export LANG=C.UTF-8
  export LC_ALL=C.UTF-8

  python -u main.py \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    model=small \
    algo=bd3lm \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    data=writing_prompts \
    data.cache_dir='"${DATA_CACHE_DIR}"' \
    data.sample_fraction='"${WP_SAMPLE_FRACTION}"' \
    data.prefix_max_tokens=128 \
    data.answer_max_tokens=893 \
    data.conditional_generation=True \
    model.length=1024 \
    block_size='"${BLOCK_SIZE}"' \
    wandb.name=bd3lm-wp-structured-block_size'"${BLOCK_SIZE}"' \
    mode=train \
    model.attn_backend=flex \
    training.resample=False \
    training.from_pretrained='"${PRETRAIN_CKPT}"' \
    loader.num_workers=0 \
    trainer.max_steps='"${MAX_STEPS}"' \
    trainer.val_check_interval='"${VAL_CHECK_INTERVAL}"' \
    trainer.limit_train_batches='"${TRAIN_LIMIT}"' \
    trainer.limit_val_batches='"${VAL_LIMIT}"' \
    hydra.run.dir='"${SAVE_DIR}"' \
    checkpointing.save_dir='"${SAVE_DIR}"' \
    algo.structured_masking.enabled=True \
    algo.structured_masking.r_low=0.15 \
    algo.structured_masking.r_high=0.85 \
    algo.structured_masking.b_max_tokens=32 \
    algo.structured_masking.full_bidir_attention=True
'
