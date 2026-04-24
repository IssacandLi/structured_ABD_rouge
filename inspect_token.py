import os
import datasets
from transformers import AutoTokenizer

DATA_DIR = "/data/lab/yan/peihong_li/data_cache/cdm_cnn_dm_dat_semantic"
SPLIT = "train"  # train / validation
DAT_NAME = f"cnn_dailymail_{SPLIT}_bs1024_unwrapped_cnn3.0.0_p768_a256_lossEOS1.dat"
DAT_PATH = os.path.join(DATA_DIR, DAT_NAME)

SEG_END_ID = 0
N_SHOW = 50
SEED = 123

OUT_TXT = os.path.join(DATA_DIR, f"segmentation_inspect_{SPLIT}_{N_SHOW}.txt")

# tokenizer 必须和离线处理一致（GPT-2）
tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

ds = datasets.load_from_disk(DAT_PATH)
sample = ds.shuffle(seed=SEED).select(range(min(N_SHOW, len(ds))))

def decode(ids):
    return tok.decode(ids, skip_special_tokens=False)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(f"Segmentation inspection for {SPLIT}\n")
    f.write(f"Dataset: {DAT_PATH}\n")
    f.write(f"N_SHOW = {N_SHOW}\n")
    f.write("=" * 100 + "\n\n")

    for idx, ex in enumerate(sample):
        ids = ex["input_ids"]
        am = ex["attention_mask"]  # answer mask: answer=1

        # 只取 answer 区域
        ans_ids = [t for t, m in zip(ids, am) if m == 1]

        # 按 seg_end 切块
        blocks = []
        cur = []
        for t in ans_ids:
            if t == SEG_END_ID:
                blocks.append(cur)
                cur = []
            else:
                cur.append(t)
        if cur:
            blocks.append(cur)

        block_lens = [len(b) for b in blocks if len(b) > 0]
        seg_cnt = len(block_lens)

        # answer 起点位置（大致）
        try:
            ans_start = am.index(1)
        except ValueError:
            ans_start = None

        # 写 header
        f.write("=" * 100 + "\n")
        f.write(f"[Sample {idx}]\n")
        f.write(f"seg_end_count = {seg_cnt}\n")
        f.write(f"block_lens    = {block_lens}\n")
        if ans_start is not None:
            f.write(f"ans_start_idx = {ans_start}\n")
        f.write(f"answer_tokens = {len(ans_ids)}\n\n")

        # 写每个 block（最多前 12 个，防止太长）
        MAX_BLOCKS = 12
        for bi, b in enumerate(blocks[:MAX_BLOCKS]):
            txt = decode(b).replace("\n", " ")
            if len(txt) > 300:
                txt = txt[:300] + " ..."
            f.write(f"  block{bi:02d} len={len(b):2d}: {txt}\n")

        if len(blocks) > MAX_BLOCKS:
            f.write(f"  ... ({len(blocks) - MAX_BLOCKS} more blocks)\n")

        f.write("\n")

print("Saved to:", OUT_TXT)
