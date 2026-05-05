#!/usr/bin/env python3
"""Topic-skeleton diagnostics for NDP synthetic outputs."""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

try:
  from prepare_topic_skeleton import TOPIC_BANK
except ImportError:
  TOPIC_BANK = {}


def read_jsonl(path):
  rows = []
  with open(path, "r", encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def load_diagnostics(path):
  with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
  return payload.get("records", [])


def tokenize(text):
  return [
    token.strip(".,;:!?\"'()[]{}").lower()
    for token in str(text or "").split()
    if token.strip(".,;:!?\"'()[]{}")
  ]


def topic_vector(text):
  counts = Counter(tokenize(text))
  return [
    float(sum(counts[word] for word in TOPIC_BANK[topic]))
    for topic in sorted(TOPIC_BANK)
  ]


def cosine(a, b):
  dot = sum(x * y for x, y in zip(a, b))
  na = math.sqrt(sum(x * x for x in a))
  nb = math.sqrt(sum(y * y for y in b))
  if na == 0.0 or nb == 0.0:
    return math.nan
  return dot / (na * nb)


def best_topic(text):
  counts = Counter(tokenize(text))
  scores = {
    topic: sum(counts[word] for word in words)
    for topic, words in TOPIC_BANK.items()
  }
  return max(scores.items(), key=lambda item: item[1])[0] if scores else None


def split_equal_words(text, n_segments):
  words = str(text or "").split()
  if n_segments <= 0:
    return []
  if not words:
    return ["" for _ in range(n_segments)]
  segments = []
  for idx in range(n_segments):
    start = round(idx * len(words) / n_segments)
    end = round((idx + 1) * len(words) / n_segments)
    segments.append(" ".join(words[start:end]))
  return segments


def score_text(text, topic_sequence):
  segments = split_equal_words(text, len(topic_sequence))
  predicted = [best_topic(segment) for segment in segments]
  correct = [
    pred == gold for pred, gold in zip(predicted, topic_sequence)
  ]
  vectors = [topic_vector(segment) for segment in segments]
  adjacent = [
    cosine(vectors[i], vectors[i + 1])
    for i in range(len(vectors) - 1)
  ]
  adjacent = [value for value in adjacent if not math.isnan(value)]
  return {
    "predicted_topic_sequence": predicted,
    "topic_recovery_accuracy": (
      sum(correct) / len(topic_sequence) if topic_sequence else math.nan),
    "final_topic_order_correct": bool(predicted == list(topic_sequence)),
    "cross_span_topic_consistency": (
      sum(adjacent) / len(adjacent) if adjacent else math.nan),
  }


def mean(values):
  values = [value for value in values if not math.isnan(value)]
  return sum(values) / len(values) if values else math.nan


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--diagnostics", required=True)
  parser.add_argument("--raw-validation-jsonl", required=True)
  parser.add_argument("--output-json", required=True)
  parser.add_argument("--output-csv", default=None)
  return parser.parse_args()


def main():
  args = parse_args()
  records = load_diagnostics(args.diagnostics)
  gold_rows = read_jsonl(args.raw_validation_jsonl)
  rows = []

  for idx, record in enumerate(records):
    if idx >= len(gold_rows):
      break
    gold = gold_rows[idx]
    topic_sequence = gold.get("topic_sequence", [])
    final_text = record.get("final_text") or record.get("pred_text") or ""
    final_scores = score_text(final_text, topic_sequence)
    rows.append({
      "sample_index": idx,
      "snapshot_source": "final",
      "target_fraction": 1.0,
      **final_scores,
    })

    for snapshot in record.get("step_snapshots", []):
      if not snapshot.get("captured", False):
        continue
      partial_text = snapshot.get("partial_text") or ""
      snapshot_scores = score_text(partial_text, topic_sequence)
      rows.append({
        "sample_index": idx,
        "snapshot_source": "step_fraction",
        "target_fraction": float(snapshot.get("target_step_fraction", math.nan)),
        **snapshot_scores,
      })

  grouped = defaultdict(list)
  for row in rows:
    grouped[(row["snapshot_source"], row["target_fraction"])].append(row)

  summary = []
  for (source, target), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
    summary.append({
      "snapshot_source": source,
      "target_fraction": target,
      "num_samples": len(group_rows),
      "topic_recovery_accuracy": mean([
        row["topic_recovery_accuracy"] for row in group_rows]),
      "final_topic_order_accuracy": mean([
        float(row["final_topic_order_correct"]) for row in group_rows]),
      "cross_span_topic_consistency": mean([
        row["cross_span_topic_consistency"] for row in group_rows]),
    })

  output_json_dir = os.path.dirname(args.output_json)
  if output_json_dir:
    os.makedirs(output_json_dir, exist_ok=True)
  with open(args.output_json, "w", encoding="utf-8") as handle:
    json.dump({"summary": summary, "rows": rows}, handle, indent=2, sort_keys=True)

  if args.output_csv:
    output_csv_dir = os.path.dirname(args.output_csv)
    if output_csv_dir:
      os.makedirs(output_csv_dir, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()) if summary else [])
      if summary:
        writer.writeheader()
        writer.writerows(summary)

  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
