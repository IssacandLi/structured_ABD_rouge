#!/usr/bin/env python3
"""Build tokenized dataset caches for NDP pilot tasks."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import datasets

import dataloader


TASK_DEFAULTS = {
  "writing_prompts": {
    "dataset_name": "writing_prompts",
    "hf_dataset_name": "euclaise/writingprompts",
    "prompt_field": "prompt",
    "answer_field": "story",
    "sample_fraction": 0.01,
    "prefix_max_tokens": 128,
    "answer_max_tokens": 893,
  },
  "synthetic_topic_skeleton": {
    "dataset_name": "ndp_topic_skeleton",
    "hf_dataset_name": None,
    "prompt_field": "prompt",
    "answer_field": "story",
    "sample_fraction": 1.0,
    "prefix_max_tokens": 64,
    "answer_max_tokens": 957,
  },
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--task",
    required=True,
    choices=sorted(TASK_DEFAULTS),
    help="Pilot task cache to build.")
  parser.add_argument("--cache-dir", required=True)
  parser.add_argument("--local-data-dir", default=None)
  parser.add_argument("--tokenizer", default="gpt2")
  parser.add_argument("--block-size", type=int, default=1024)
  parser.add_argument("--num-proc", type=int, default=8)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--sample-fraction", type=float, default=None)
  parser.add_argument("--max-train-samples", type=int, default=None)
  parser.add_argument("--max-valid-samples", type=int, default=None)
  parser.add_argument("--prefix-max-tokens", type=int, default=None)
  parser.add_argument("--answer-max-tokens", type=int, default=None)
  parser.add_argument(
    "--prefetch-full",
    action="store_true",
    help=(
      "Download/cache the full raw HF dataset before building the sampled "
      "tokenized cache. Only applies to HF-backed tasks such as WP."))
  return parser.parse_args()


def main():
  args = parse_args()
  defaults = TASK_DEFAULTS[args.task]

  if args.prefetch_full and defaults["hf_dataset_name"] is not None:
    for split in ["train", "validation"]:
      print(
        f"[prefetch] downloading full {defaults['hf_dataset_name']} split={split}",
        flush=True)
      datasets.load_dataset(
        defaults["hf_dataset_name"],
        split=split,
        cache_dir=args.cache_dir,
        trust_remote_code=True)

  tokenizer_cfg = SimpleNamespace(
    data=SimpleNamespace(tokenizer_name_or_path=args.tokenizer))
  tokenizer = dataloader.get_tokenizer(tokenizer_cfg)

  sample_fraction = (
    defaults["sample_fraction"]
    if args.sample_fraction is None else args.sample_fraction)
  prefix_max_tokens = (
    defaults["prefix_max_tokens"]
    if args.prefix_max_tokens is None else args.prefix_max_tokens)
  answer_max_tokens = (
    defaults["answer_max_tokens"]
    if args.answer_max_tokens is None else args.answer_max_tokens)

  split_specs = [
    ("train", args.max_train_samples),
    ("validation", args.max_valid_samples),
  ]
  for mode, max_samples in split_specs:
    print(f"[cache] building {args.task} split={mode}", flush=True)
    dataloader.get_dataset(
      dataset_name=defaults["dataset_name"],
      tokenizer=tokenizer,
      wrap=False,
      mode=mode,
      cache_dir=args.cache_dir,
      block_size=args.block_size,
      num_proc=args.num_proc,
      streaming=False,
      insert_eos=True,
      insert_special_tokens=True,
      prefix_max_tokens=prefix_max_tokens,
      answer_max_tokens=answer_max_tokens,
      loss_on_answer_eos=True,
      prompt_field=defaults["prompt_field"],
      answer_field=defaults["answer_field"],
      hf_dataset_name=defaults["hf_dataset_name"],
      local_data_dir=args.local_data_dir,
      local_train_file="train.jsonl",
      local_validation_file="validation.jsonl",
      sample_fraction=sample_fraction,
      max_samples=max_samples,
      data_seed=args.seed,
    )
  print("[cache] done", flush=True)


if __name__ == "__main__":
  main()
