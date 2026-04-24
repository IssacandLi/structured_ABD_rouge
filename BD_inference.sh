#!/bin/bash
#SBATCH --partition=camas
#SBATCH --job-name=bd3lm_cnn_dm
#SBATCH --time=48:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --nodelist=sn17
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=300000

module load anaconda3
source activate /data/lab/yan/peihong_li/condaenvlist/bd3lm
cd /data/lab/yan/peihong_li/ACL/structure_ABD_rouge/

BLOCK_SIZE=4

python -u main.py \
  model=small \
  mode=sample_eval \
  loader.eval_batch_size=1 \
  model.length=1024 \
  block_size=4 \
  algo=bd3lm \
  algo.T=5000 \
  data.train=cnn_dailymail \
  data.valid=cnn_dailymail \
  data.wrap=false \
  +data.cnn_dm_version=3.0.0 \
  +data.prefix_max_tokens=768 \
  +data.answer_max_tokens=256 \
  +data.loss_on_answer_eos=True \
  wandb=null \
  +data.conditional_generation=true \
  data.cache_dir=/data/lab/yan/peihong_li/data_cache/cdm_cnn_dm_dat \
  sampling.nucleus_p=0.9 \
  sampling.kv_cache=false \
  sampling.num_eval_samples=100 \
  +sampling.context_size=512 \
  sampling.logdir=$PWD/sample_logs/cnn_dm_cond_bs4_100samples.csv \
  diagnostics.enabled=True \
  diagnostics.save_path=$PWD/sample_logs/cnn_dm_cond_bs4_100samples_diagnostics.json \
  'diagnostics.snapshot_reveal_fractions=[0.1,0.3,0.5,0.7]' \
  diagnostics.early_fraction=0.3 \
  ++eval.conditional_metric=rouge \
  eval.checkpoint_path=/data/lab/yan/peihong_li/ACL/structure_ABD/outputs/cnn_dailymail/2026.04.21/165620/checkpoints/best.ckpt \
  algo.structured_masking.enabled=True \
  algo.structured_masking.r_low=0.3 \
  algo.structured_masking.r_high=0.7 \
  algo.structured_masking.global_t=True \
  algo.span_loss.enabled=False \
  algo.span_loss.lambda_span=1.0 \
  algo.span_loss.type=bow \
  algo.structured_inference.enabled=True \
  algo.structured_inference.aggregation=mean \
  algo.structured_inference.commitment=mixed \
  algo.structured_inference.threshold=fixed_ratio \
  algo.structured_masking.b_max_tokens=32 \
  algo.structured_masking.full_bidir_attention=True \
  sampling.kv_cache=False \
  sampling.first_hitting=False \
  model.attn_backend=sdpa
