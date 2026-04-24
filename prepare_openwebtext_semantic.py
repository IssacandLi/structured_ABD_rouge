#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Offline preprocessing for openwebtext (semantic-aware version):

- 读取 openwebtext-train / openwebtext-valid split
- 对每条 text 进行“语义切分”（可选：用开源 LLM；否则用简单句子切分）
- 对整篇文档进行 tokenize（不加 special tokens），
  并为每个 token 记录其所属语义块 seg_id
- 仿照原始 dataloader._group_texts：
  * 把所有 input_ids 串成一个长序列
  * 每次取 (block_size - 2) 个 token，加 [BOS] / [EOS]，得到长度 = block_size 的块
  * attention_mask 全 1
- 额外保存一个 seg_ids，长度同为 block_size，BOS/EOS 的 seg_id 从邻近 token 继承
- 保存路径与原始 get_dataset 一致（不加后缀），因此会覆盖原来的 .dat

使用示例（注意保持和 openwebtext-split.yaml 一致）：

  # 生成 train 语义块
  python prepare_openwebtext_semantic.py \
  --dataset_name openwebtext-train \
  --mode train \
  --cache_dir /data/lab/yan/peihong_li/data_cache \
  --block_size 1024 \
  --tokenizer_name gpt2 \
  --insert_eos True \
  --insert_special True \
  --wrap True \
  --llm_model_name "Qwen/Qwen2.5-1.5B-Instruct"


  # 生成 valid 语义块
  python prepare_openwebtext_semantic.py \
  --dataset_name openwebtext-valid \
  --mode validation \
  --cache_dir /data/lab/yan/peihong_li/data_cache \
  --block_size 1024 \
  --tokenizer_name gpt2 \
  --insert_eos True \
  --insert_special True \
  --wrap True \
  --llm_model_name "Qwen/Qwen2.5-1.5B-Instruct"

"""

import argparse
import os
import typing
import itertools  # ★ 新增：用于拼接 token 序列

import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import omegaconf

import dataloader
import utils

LOGGER = utils.get_logger(__name__)

# ===================== 语义切分相关 ===================== #

_sem_tokenizer = None
_sem_model = None


def init_semantic_llm(model_name: str):
  """懒加载开源 LLM，只在第一次调用时加载。"""
  global _sem_tokenizer, _sem_model
  if _sem_model is None:
    LOGGER.info(f"[Semantic LLM] Loading model: {model_name}")
    _sem_tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      trust_remote_code=True,   # ★ 对 Qwen、LLaMA 等新模型一般需要
    )
    _sem_model = AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map="auto",        # 
      torch_dtype=torch.float16,
      trust_remote_code=True,
    )

    # 有些模型没有 pad_token，避免 generate 报错
    if _sem_tokenizer.pad_token is None:
      _sem_tokenizer.pad_token = _sem_tokenizer.eos_token

    _sem_model.eval()



@torch.no_grad()
def semantic_split_with_llm(text: str, llm_model_name: str) -> typing.List[str]:
  """
  使用开源 LLM 对一段文本做语义切分。
  返回若干段落：[seg1, seg2, ...]
  """
  init_semantic_llm(llm_model_name)

  prompt = (
    "You are a text segmenter.\n"
    "Split the following text into semantic chunks. "
    "Each chunk should be coherent and not too long. "
    "Output only the chunks, one per line, without explanations.\n\n"
    f"TEXT:\n{text}\n\n"
    "CHUNKS:\n"
  )

  inputs = _sem_tokenizer(prompt, return_tensors="pt").to(_sem_model.device)
  out = _sem_model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,
    eos_token_id=_sem_tokenizer.eos_token_id,
  )

  gen = _sem_tokenizer.decode(
    out[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
  )

  segments = [line.strip() for line in gen.split("\n") if line.strip()]
  if not segments:
    segments = [text.strip()]
  return segments


def simple_sentence_split(text: str) -> typing.List[str]:
  """
  没有 LLM 时的简易 fallback：按句号/问号/感叹号粗分。
  只是为了跑通流程，真正实验建议用 LLM 模式。
  """
  import re

  parts = re.split(r'([\.!?])', text)
  segments = []
  buf = ""
  for part in parts:
    if part in [".", "!", "?"]:
      buf += part
      if buf.strip():
        segments.append(buf.strip())
        buf = ""
    else:
      buf += part
  if buf.strip():
    segments.append(buf.strip())

  segments = [s for s in segments if s]
  return segments or [text.strip()]


# ===================== 项目内的 tokenizer & path 对齐 ===================== #

def build_project_tokenizer(tokenizer_name: str):
  """
  直接复用项目里的 dataloader.get_tokenizer，
  保证和训练时的 tokenizer 完全一致（包括 GPT2 的 post_processor）。
  """
  cfg = omegaconf.OmegaConf.create({
    "data": {
      "tokenizer_name_or_path": tokenizer_name
    }
  })
  tokenizer = dataloader.get_tokenizer(cfg)
  return tokenizer


def get_raw_openwebtext_split(dataset_name: str, cache_dir: str):
  """
  仿照 dataloader.get_dataset 里对 openwebtext 的切分：
    - openwebtext-train: train[:-100000]
    - openwebtext-valid: train[-100000:]
  """
  if dataset_name == "openwebtext-train":
    split = "train[:-100000]"
  elif dataset_name == "openwebtext-valid":
    split = "train[-100000:]"
  else:
    raise ValueError(
      f"Unsupported dataset_name: {dataset_name}. "
      f"Expected 'openwebtext-train' or 'openwebtext-valid'."
    )

  LOGGER.info(f"Loading raw openwebtext with split='{split}'")
  raw_ds = datasets.load_dataset(
    "openwebtext",
    split=split,
    cache_dir=cache_dir,
    streaming=False,
    trust_remote_code=True,
  )
  return raw_ds


def compute_output_path(
  cache_dir: str,
  dataset_name: str,
  mode: str,
  block_size: int,
  insert_eos: bool,
  insert_special_tokens: bool,
  wrap: bool,
  semantic_suffix: str = "",  # ★ 改：默认不加后缀，这样会覆盖原始 .dat
) -> str:
  """
  完全复刻 dataloader.get_dataset 里的 _path 逻辑：

    if wrap:
      filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped{eos_tag}.dat'
    else:
      filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped{eos_tag}.dat'

    eos_tag:
      if not insert_eos: '_eosFalse'
      if not insert_special_tokens: '_specialFalse'

  """
  eos_tag = ''
  if not insert_eos:
    eos_tag = '_eosFalse'
  if not insert_special_tokens:
    eos_tag = '_specialFalse'

  if wrap:
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped{eos_tag}{semantic_suffix}.dat'
  else:
    filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped{eos_tag}{semantic_suffix}.dat'

  return os.path.join(cache_dir, filename)


# ============ 仿照 _group_texts，但同时处理 seg_ids ============ #

def group_texts_with_segments(
  examples,
  block_size: int,
  bos_id: int,
  eos_id: int,
  insert_special_tokens: bool = True,
):
  """
  输入:
    examples["input_ids"]: List[List[int]]
    examples["seg_ids"]:   List[List[int]]

  输出:
    每一条:
      input_ids: [BOS] + 1022 tokens + [EOS]   (长度=block_size)
      seg_ids:   与 input_ids 等长，BOS/EOS 的 seg_id 从邻近 token 继承
      attention_mask: [1]*block_size
  """
  # 拼接所有文档
  concatenated_ids = list(itertools.chain(*examples["input_ids"]))
  concatenated_seg = list(itertools.chain(*examples["seg_ids"]))

  if len(concatenated_ids) == 0:
    return {
      "input_ids": [],
      "seg_ids": [],
      "attention_mask": [],
    }

  if insert_special_tokens:
    new_block_size = block_size - 2  # 预留 BOS/EOS
  else:
    new_block_size = block_size

  total_length = (len(concatenated_ids) // new_block_size) * new_block_size
  if total_length == 0:
    return {
      "input_ids": [],
      "seg_ids": [],
      "attention_mask": [],
    }

  input_blocks = []
  seg_blocks = []
  attn_blocks = []

  for i in range(0, total_length, new_block_size):
    core_ids = concatenated_ids[i : i + new_block_size]
    core_seg = concatenated_seg[i : i + new_block_size]

    if insert_special_tokens:
      # BOS/EOS 的 seg_id 分别继承首尾 token
      block_ids = [bos_id] + core_ids + [eos_id]
      block_seg = [core_seg[0]] + core_seg + [core_seg[-1]]
    else:
      block_ids = core_ids
      block_seg = core_seg

    input_blocks.append(block_ids)
    seg_blocks.append(block_seg)
    attn_blocks.append([1] * block_size)

  return {
    "input_ids": input_blocks,
    "seg_ids": seg_blocks,
    "attention_mask": attn_blocks,
  }


# ===================== 主流程 ===================== #

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_name", type=str, required=True,
                      help="openwebtext-train 或 openwebtext-valid")
  parser.add_argument("--mode", type=str, required=True,
                      help="'train' 或 'validation'，必须和 get_dataset 里的 mode 一致")
  parser.add_argument("--cache_dir", type=str, required=True)
  parser.add_argument("--block_size", type=int, default=1024)
  parser.add_argument("--tokenizer_name", type=str, default="gpt2")

  parser.add_argument("--insert_eos", type=lambda x: x.lower() == "true",
                      default=True)
  parser.add_argument("--insert_special", type=lambda x: x.lower() == "true",
                      default=True)
  parser.add_argument("--wrap", type=lambda x: x.lower() == "true",
                      default=True,
                      help="只影响保存路径的命名，要和 yaml 里的 wrap 一致")

  parser.add_argument(
        "--llm_model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct", 
        help=(
          "用于语义切分的 LLM 模型名称，例如："
          "'Qwen/Qwen2.5-1.5B-Instruct' 或 "
          "'meta-llama/Llama-3.1-8B-Instruct'；"
          "设为空则使用简单句子切分。"
        ),
    )

  args = parser.parse_args()

  LOGGER.info(f"Args: {args}")

  # 1) 计算输出路径（与 dataloader.get_dataset 完全一致，这里覆盖原文件）
  out_path = compute_output_path(
    cache_dir=args.cache_dir,
    dataset_name=args.dataset_name,
    mode=args.mode,
    block_size=args.block_size,
    insert_eos=args.insert_eos,
    insert_special_tokens=args.insert_special,
    wrap=args.wrap,
    semantic_suffix="",  # ★ 关键：不加 _semantic，直接覆盖原 dat
  )
  LOGGER.info(f"Output dataset path: {out_path}")

  if utils.fsspec_exists(out_path):
    LOGGER.warning(f"{out_path} already exists and will be overwritten.")

  # 2) tokenizer：直接用项目里的 get_tokenizer，保持行为一致
  tokenizer = build_project_tokenizer(args.tokenizer_name)
  tokenizer.padding_side = "right"
  tokenizer.truncation_side = "right"

  BOS = tokenizer.bos_token_id
  EOS = tokenizer.eos_token_id

  if BOS is None or EOS is None:
    raise ValueError("Tokenizer must have BOS/EOS tokens configured.")

  # 3) 加载 raw openwebtext split
  raw_ds = get_raw_openwebtext_split(args.dataset_name, args.cache_dir)

  # 4) 单样本处理：text -> 语义 segments -> 整篇 token 序列 + seg_ids
  def process_example(example):
    text = example["text"]

    if args.llm_model_name is not None:
      segments = semantic_split_with_llm(text, args.llm_model_name)
    else:
      segments = simple_sentence_split(text)

    doc_ids: typing.List[int] = []
    doc_seg: typing.List[int] = []

    seg_idx = 0
    for seg in segments:
      seg = seg.strip()
      if not seg:
        continue

      enc = tokenizer(
        seg,
        add_special_tokens=False,          # ★ 不在这里加 BOS/EOS
        return_attention_mask=False,
        return_token_type_ids=False,
      )
      ids = enc["input_ids"]
      if len(ids) == 0:
        continue

      doc_ids.extend(ids)
      doc_seg.extend([seg_idx] * len(ids))
      seg_idx += 1

    # 极端情况：LLM / 正则切出来全是空，fallback 为整段 text
    if len(doc_ids) == 0:
      enc = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
      )
      ids = enc["input_ids"]
      if len(ids) == 0:
        return {
          "input_ids": [[]],
          "seg_ids": [[]],
        }
      doc_ids = ids
      doc_seg = [0] * len(ids)
      seg_idx = 1

    # 文档级别 EOS（与原始 dataloader 中 insert_eos 的语义一致）
    if args.insert_eos and EOS is not None:
      doc_ids.append(EOS)
      # EOS 归到最后一个语义块
      doc_seg.append(max(0, seg_idx - 1))

    # 注意：为了兼容 HF map(batched=False) 的行为，
    # 返回 list[list[int]]，HF 会展开成多行样本
    return {
      "input_ids": [doc_ids],
      "seg_ids": [doc_seg],
    }

  LOGGER.info("Start semantic segmenting + tokenizing raw dataset...")
  tokenized = raw_ds.map(
    process_example,
    batched=False,
    num_proc=1,            # 使用 LLM 时不建议多进程
    remove_columns=["text"],
    desc=f"Semantic segmenting + tokenizing {args.dataset_name}",
  )

  LOGGER.info("Start grouping into fixed-length blocks (1024 with BOS/EOS)...")

  # 仿照 dataloader._group_texts，对 input_ids & seg_ids 同时分块
  grouped = tokenized.map(
    lambda batch: group_texts_with_segments(
      batch,
      block_size=args.block_size,
      bos_id=BOS,
      eos_id=EOS,
      insert_special_tokens=args.insert_special,
    ),
    batched=True,
    num_proc=1,
    desc="Grouping into blocks with seg_ids",
  )

  LOGGER.info("Saving processed dataset to disk...")
  grouped.save_to_disk(out_path)
  LOGGER.info("Done.")


if __name__ == "__main__":
  main()