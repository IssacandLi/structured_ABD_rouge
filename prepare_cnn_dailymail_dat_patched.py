#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import transformers

# 直接复用你项目里的 get_dataset (使用 patched 版本以支持 semantic_blocks)
import dataloader_patched as dataloader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--tokenizer_name", type=str, default="gpt2")
    ap.add_argument("--block_size", type=int, default=1024)

    ap.add_argument("--cnn_dm_version", type=str, default="3.0.0")
    ap.add_argument("--prefix_max_tokens", type=int, default=768)
    ap.add_argument("--answer_max_tokens", type=int, default=256)
    # --- semantic block segmentation on highlights (optional) ---
    ap.add_argument("--semantic_blocks", action="store_true", help="use Qwen to segment highlights into semantic blocks")
    ap.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seg_end_id", type=int, default=0, help="GPT-2 token id used as segment end marker")
    ap.add_argument("--seg_max_tokens", type=int, default=12)
    ap.add_argument("--seg_min_tokens", type=int, default=4)
    ap.add_argument("--qwen_max_new_tokens", type=int, default=256)
    ap.add_argument("--qwen_batch_size", type=int, default=128)
    ap.add_argument("--qwen_temperature", type=float, default=0.0)
    ap.add_argument("--qwen_device_map", type=str, default="auto")
    ap.add_argument("--qwen_dtype", type=str, default="auto")
    ap.add_argument("--loss_on_answer_eos", action="store_true")

    # 这俩要和你训练 config 保持一致（你 yaml 里是 True）
    ap.add_argument("--insert_eos", action="store_true")
    ap.add_argument("--insert_special_tokens", action="store_true")

    ap.add_argument("--num_proc", type=int, default=16)
    args = ap.parse_args()

    if args.semantic_blocks and args.num_proc != 1:
        print(f"[WARN] semantic_blocks=True: forcing --num_proc=1 to avoid loading Qwen multiple times (was {args.num_proc}).")
        args.num_proc = 1

    os.makedirs(args.cache_dir, exist_ok=True)

    tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    # GPT2 没 pad_token 的话要补一下，否则 padding='max_length' 会出问题
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 只要执行到这里，get_dataset 会：不存在 -> 生成 -> save_to_disk；存在 -> load_from_disk
    for split in ["train", "validation"]:
        print(f"\n[PREP] building split={split}")
        _ = dataloader.get_dataset(
            "cnn_dailymail",
            tok,
            wrap=False,
            mode=split,
            cache_dir=args.cache_dir,
            block_size=args.block_size,
            streaming=False,
            num_proc=args.num_proc,
            insert_eos=args.insert_eos,
            insert_special_tokens=args.insert_special_tokens,
            cnn_dm_version=args.cnn_dm_version,
            prefix_max_tokens=args.prefix_max_tokens,
            answer_max_tokens=args.answer_max_tokens,
            loss_on_answer_eos=args.loss_on_answer_eos,
            semantic_blocks=args.semantic_blocks,
            qwen_model_name=args.qwen_model_name,
            seg_end_ids=[args.seg_end_id],
            seg_max_tokens=args.seg_max_tokens,
            seg_min_tokens=args.seg_min_tokens,
            qwen_max_new_tokens=args.qwen_max_new_tokens,
            qwen_temperature=args.qwen_temperature,
            qwen_device_map=args.qwen_device_map,
            qwen_dtype=args.qwen_dtype,
            qwen_batch_size=args.qwen_batch_size
        )
        print(f"[PREP] done split={split}")

    print("\nAll done. Now training can load_from_disk without building any cache.")


if __name__ == "__main__":
    main()
