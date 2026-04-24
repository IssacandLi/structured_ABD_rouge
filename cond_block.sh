#!/bin/bash
#SBATCH --partition=camas
#SBATCH --job-name=bd3lm_cnn_dm
#SBATCH --time=48:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --nodelist=sn18
#SBATCH --gres=gpu:tesla:2
#SBATCH --mem=300000

module load anaconda3
source activate /data/lab/yan/peihong_li/condaenvlist/bd3lm
cd /data/lab/yan/peihong_li/ACL/structure_ABD/

BLOCK_SIZE=4
PRETRAIN_CKPT=kuleshov-group/bd3lm-owt-block_size1024-pretrain

# ✅ 指向“离线 dat”目录（所有 rank 共用）
export DATA_CACHE_DIR=/data/lab/yan/peihong_li/data_cache/cdm_cnn_dm_dat
export TOKENIZERS_PARALLELISM=false

# ✅ HF cache 也放到可写目录（推荐继续保留）
export HF_HOME="${DATA_CACHE_DIR}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# ✅ triton/inductor cache 仍然 rank-specific（避免编译锁）
srun bash -lc '
  set -e
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
    data=cnn_dailymail \
    data.wrap=False \
    data.cnn_dm_version=3.0.0 \
    data.prefix_max_tokens=768 \
    data.answer_max_tokens=256 \
    data.loss_on_answer_eos=True \
    model.length=1024 \
    block_size='"${BLOCK_SIZE}"' \
    wandb.name=structured_abd_4_8_2026_11am \
    mode=train \
    model.attn_backend=flex \
    training.resample=False \
    training.from_pretrained='"${PRETRAIN_CKPT}"' \
    loader.num_workers=0 \
    trainer.max_steps=10000 \
    trainer.val_check_interval=50 \
    trainer.limit_train_batches=0.01 \
  	trainer.limit_val_batches=0.01 \
  	algo.structured_masking.enabled=True \
  	algo.structured_masking.r_low=0.3 \
    algo.structured_masking.r_high=0.7 \
    algo.structured_masking.b_max_tokens=32 \
    algo.structured_masking.global_t=True \
    algo.structured_masking.full_bidir_attention=True \
    algo.clip_search_widths=[] \
    algo.span_loss.enabled=False \
    algo.span_loss.lambda_span=1.0 \
    algo.span_loss.type=bow \
    data.cache_dir='"${DATA_CACHE_DIR}"'
'
