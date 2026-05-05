#!/usr/bin/env python3
"""Generate synthetic topic-skeleton data for the NDP pilot."""

import argparse
import json
import math
import os
import random
from collections import Counter


TOPIC_BANK = {
  "harbor": [
    "harbor", "dock", "tide", "lantern", "anchor", "ferry", "sailor",
    "fog", "breakwater", "cargo", "compass", "mast", "buoy", "net",
    "wharf", "current", "storm", "rope", "deck", "pier",
  ],
  "archive": [
    "archive", "ledger", "ink", "catalog", "shelf", "manuscript",
    "index", "dust", "folio", "stamp", "cipher", "margin", "clerk",
    "record", "vault", "parchment", "seal", "library", "drawer", "file",
  ],
  "garden": [
    "garden", "seed", "orchard", "soil", "vine", "greenhouse", "petal",
    "root", "branch", "rain", "beehive", "shovel", "moss", "trellis",
    "pond", "mulch", "sprout", "sunlight", "leaf", "bloom",
  ],
  "observatory": [
    "observatory", "telescope", "orbit", "comet", "star", "lens",
    "eclipse", "astronomer", "dome", "planet", "signal", "meteor",
    "chart", "zenith", "constellation", "gravity", "night", "sky",
    "mirror", "galaxy",
  ],
  "court": [
    "court", "witness", "verdict", "judge", "jury", "evidence",
    "gavel", "appeal", "oath", "lawyer", "testimony", "trial",
    "sentence", "chamber", "case", "alibi", "bailiff", "statute",
    "hearing", "defense",
  ],
  "theater": [
    "theater", "stage", "curtain", "actor", "script", "balcony",
    "spotlight", "rehearsal", "costume", "mask", "audience", "dialogue",
    "prop", "scene", "director", "ticket", "applause", "monologue",
    "backdrop", "orchestra",
  ],
  "laboratory": [
    "laboratory", "beaker", "sample", "enzyme", "microscope", "protocol",
    "flask", "thermal", "assay", "reagent", "pipette", "culture",
    "data", "compound", "control", "sterile", "meter", "reaction",
    "vial", "sensor",
  ],
  "market": [
    "market", "stall", "vendor", "coin", "basket", "spice", "linen",
    "barter", "scale", "receipt", "crowd", "alley", "cart", "merchant",
    "price", "fruit", "awning", "morning", "crate", "cloth",
  ],
}

TRANSITIONS = {
  "continuation": [
    "The same problem followed them into the next hour.",
    "Nothing about the situation felt finished yet.",
  ],
  "contrast": [
    "Yet the next clue seemed to contradict everything before it.",
    "Still, the mood changed as soon as they crossed the threshold.",
  ],
  "escalation": [
    "Then the stakes rose faster than anyone expected.",
    "A small mistake suddenly became impossible to ignore.",
  ],
  "scene_shift": [
    "By dusk, the scene had moved somewhere else entirely.",
    "The next morning opened on a different noise and a different fear.",
  ],
  "callback": [
    "Only then did the earlier detail return with a new meaning.",
    "The forgotten sign from before finally mattered.",
  ],
}

TEMPLATES = [
  "The {agent} noticed the {w0} beside the {w1}, and the detail made the plan less certain.",
  "Near the {w2}, a quiet argument about the {w3} forced everyone to slow down.",
  "Someone had hidden a {w4} in plain sight, but only the {agent} understood why it mattered.",
  "The old story mentioned {w5} and {w6}, which now seemed more like a warning.",
  "When the {w7} appeared again, the group realized this place had shaped the choice all along.",
  "A trace of {w8} remained there, connecting the present choice to an earlier promise.",
]

AGENTS = [
  "narrator", "apprentice", "messenger", "stranger", "caretaker",
  "captain", "scribe", "student", "traveler", "investigator",
]


def write_jsonl(path, rows):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_span(rng, topic, target_words, transition):
  words = TOPIC_BANK[topic]
  agent = rng.choice(AGENTS)
  sentences = []
  if transition is not None:
    sentences.append(rng.choice(TRANSITIONS[transition]))
  cursor = 0
  while len(" ".join(sentences).split()) < target_words:
    picked = [words[(cursor + i) % len(words)] for i in range(9)]
    cursor += rng.randint(2, 5)
    sentence = rng.choice(TEMPLATES).format(
      agent=agent,
      **{f"w{i}": picked[i] for i in range(9)})
    if rng.random() < 0.35:
      sentence += f" The {topic} pattern was becoming too clear to dismiss."
    sentences.append(sentence)
  return " ".join(sentences)


def generate_example(rng, idx, min_topics, max_topics, min_span_words, max_span_words):
  n_topics = rng.randint(min_topics, max_topics)
  topic_sequence = rng.sample(list(TOPIC_BANK), n_topics)
  story_parts = []
  spans = []
  word_cursor = 0
  transition_names = list(TRANSITIONS)
  for span_idx, topic in enumerate(topic_sequence):
    transition = None if span_idx == 0 else rng.choice(transition_names)
    span_text = build_span(
      rng,
      topic,
      target_words=rng.randint(min_span_words, max_span_words),
      transition=transition)
    span_words = span_text.split()
    spans.append({
      "span_index": span_idx,
      "topic": topic,
      "transition": transition,
      "word_start": word_cursor,
      "word_end": word_cursor + len(span_words),
      "indicator_words": TOPIC_BANK[topic],
      "text": span_text,
    })
    story_parts.append(span_text)
    word_cursor += len(span_words)

  prompt = (
    "Write a coherent story that follows a hidden sequence of scenes. "
    "The exact scene order should become clear from repeated concrete details.")
  return {
    "id": f"topic_skeleton_{idx:06d}",
    "prompt": prompt,
    "story": " ".join(story_parts),
    "topic_sequence": topic_sequence,
    "spans": spans,
  }


def topic_scores(text):
  words = Counter(token.strip(".,;:!?\"'()[]{}").lower() for token in text.split())
  return {
    topic: sum(words[w] for w in indicators)
    for topic, indicators in TOPIC_BANK.items()
  }


def best_topic(text):
  scores = topic_scores(text)
  return max(scores.items(), key=lambda item: item[1])[0]


def sanity_check(rows, seed):
  rng = random.Random(seed)
  coherent_correct = 0
  random_correct = 0
  total = 0
  for row in rows:
    story_words = row["story"].split()
    for span in row["spans"]:
      span_text = span["text"]
      true_topic = span["topic"]
      coherent_correct += int(best_topic(span_text) == true_topic)
      sample_size = max(1, len(span_text.split()))
      random_text = " ".join(rng.sample(story_words, min(sample_size, len(story_words))))
      random_correct += int(best_topic(random_text) == true_topic)
      total += 1
  return {
    "num_spans": total,
    "coherent_span_accuracy": coherent_correct / total if total else math.nan,
    "random_token_accuracy": random_correct / total if total else math.nan,
  }


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--num-train", type=int, default=1000)
  parser.add_argument("--num-validation", type=int, default=200)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--min-topics", type=int, default=4)
  parser.add_argument("--max-topics", type=int, default=8)
  parser.add_argument("--min-span-words", type=int, default=80)
  parser.add_argument("--max-span-words", type=int, default=160)
  parser.add_argument("--fail-on-sanity", action="store_true")
  parser.add_argument("--sanity-margin", type=float, default=0.20)
  return parser.parse_args()


def main():
  args = parse_args()
  rng = random.Random(args.seed)
  train = [
    generate_example(
      rng, idx, args.min_topics, args.max_topics,
      args.min_span_words, args.max_span_words)
    for idx in range(args.num_train)
  ]
  validation = [
    generate_example(
      rng, args.num_train + idx, args.min_topics, args.max_topics,
      args.min_span_words, args.max_span_words)
    for idx in range(args.num_validation)
  ]

  write_jsonl(os.path.join(args.output_dir, "train.jsonl"), train)
  write_jsonl(os.path.join(args.output_dir, "validation.jsonl"), validation)
  write_jsonl(os.path.join(args.output_dir, "sanity_100.jsonl"), train[:100])

  report = sanity_check(train[:100], seed=args.seed + 17)
  report_path = os.path.join(args.output_dir, "sanity_report.json")
  with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
  print(json.dumps(report, indent=2, sort_keys=True))

  if args.fail_on_sanity:
    gap = report["coherent_span_accuracy"] - report["random_token_accuracy"]
    if gap < args.sanity_margin:
      raise SystemExit(
        f"Sanity gap too small: {gap:.3f} < {args.sanity_margin:.3f}")


if __name__ == "__main__":
  main()
