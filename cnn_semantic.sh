#!/bin/bash
#SBATCH --partition=camas
#SBATCH --job-name=bd3lm_cnn_dm
#SBATCH --time=48:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --nodelist=sn17
#SBATCH --gres=gpu:tesla:2
#SBATCH --mem=300000

module load anaconda3
source activate /data/lab/yan/peihong_li/condaenvlist/Semantic_bd3lm
cd /data/lab/yan/peihong_li/ACL/bd3lms3/bd3lms

python prepare_cnn_dailymail_dat_patched.py \
  --cache_dir /data/lab/yan/peihong_li/data_cache/cdm_cnn_dm_dat_semantic \
  --block_size 1024 \
  --cnn_dm_version 3.0.0 \
  --prefix_max_tokens 768 \
  --answer_max_tokens 256 \
  --loss_on_answer_eos \
  --insert_eos \
  --insert_special_tokens \
  --semantic_blocks \
  --qwen_model_name Qwen/Qwen2.5-7B-Instruct \
  --seg_end_id 0 \
  --seg_max_tokens 12 \
  --seg_min_tokens 4 \
  --qwen_batch_size 128 \
  --qwen_device_map auto \
  --qwen_dtype auto \
  --num_proc 1