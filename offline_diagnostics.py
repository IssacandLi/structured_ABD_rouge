import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FUNCTION_WORDS = {
  'a', 'an', 'the', 'this', 'that', 'these', 'those',
  'of', 'in', 'on', 'at', 'to', 'for', 'from', 'by', 'with', 'about',
  'into', 'over', 'after', 'before', 'under', 'between', 'through',
  'and', 'or', 'but', 'nor', 'so', 'yet',
  'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
  'do', 'does', 'did', 'doing', 'have', 'has', 'had',
  'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
  'i', 'you', 'he', 'she', 'it', 'we', 'they',
  'me', 'him', 'her', 'us', 'them',
  'my', 'your', 'his', 'its', 'our', 'their',
  'mine', 'yours', 'hers', 'ours', 'theirs',
  'who', 'whom', 'whose', 'which', 'what',
  'as', 'if', 'than', 'because', 'while', 'although', 'though',
  'not', 'no', 'yes', 'up', 'down', 'out', 'off',
  'there', 'here', 'then', 'when', 'where', 'why', 'how',
}

SPECIAL_TOKENS = {'<|endoftext|>', '[mask]'}
BUCKET_LABELS = ['early', 'mid_early', 'mid_late', 'late']


def parse_args():
  parser = argparse.ArgumentParser(
    description='Offline analysis for saved sampling diagnostics.')
  parser.add_argument(
    '--run',
    action='append',
    required=True,
    help='Run spec in the form label=/path/to/diagnostics.json')
  parser.add_argument(
    '--output_dir',
    required=True,
    help='Directory to save analysis CSV/JSON summaries.')
  parser.add_argument(
    '--analysis_length',
    choices=['gt', 'final', 'full'],
    default='gt',
    help='Which token span to use for per-position analysis.')
  return parser.parse_args()


def normalize_piece(piece):
  if piece is None:
    return ''
  text = str(piece).replace('\n', ' ')
  text = re.sub(r'\s+', ' ', text).strip().lower()
  return text


def join_pieces(pieces):
  text = ''.join(piece for piece in pieces if piece is not None)
  text = re.sub(r'\s+', ' ', text).strip()
  return text


def analysis_length(record, mode):
  generation_len = int(record.get('generation_token_count', 0) or 0)
  gt_len = int(record.get('gt_answer_length', 0) or 0)
  final_len = int(record.get('final_answer_length', 0) or 0)
  if mode == 'gt' and gt_len > 0:
    return min(generation_len or gt_len, gt_len)
  if mode == 'final' and final_len > 0:
    return min(generation_len or final_len, final_len)
  return generation_len or gt_len or final_len


def classify_token(piece):
  token = normalize_piece(piece)
  if not token or token in SPECIAL_TOKENS:
    return 'OTHER'
  if not any(ch.isalpha() for ch in token):
    return 'OTHER'
  if token in FUNCTION_WORDS:
    return 'FUNCTION'
  return 'CONTENT'


def load_runs(run_specs):
  runs = []
  for spec in run_specs:
    if '=' not in spec:
      raise ValueError(f'Invalid --run spec: {spec!r}')
    label, path = spec.split('=', 1)
    with open(path, 'r', encoding='utf-8') as f:
      payload = json.load(f)
    runs.append({
      'label': label,
      'path': path,
      'meta': payload.get('meta', {}),
      'records': payload.get('records', []),
    })
  return runs


def build_diag1_rows(runs, analysis_mode):
  rows = []
  for run in runs:
    for record in run['records']:
      max_len = analysis_length(record, analysis_mode)
      gt_text = record.get('gt_text') or join_pieces(
        record.get('gt_answer_token_text', [])[:max_len])
      for snapshot in record.get('snapshots', []):
        if not snapshot.get('captured', False):
          continue
        clipped_pieces = []
        for pos, piece in zip(
          snapshot.get('revealed_positions', []),
          snapshot.get('revealed_token_text', []),
        ):
          if max_len and int(pos) >= max_len:
            continue
          clipped_pieces.append(piece)
        revealed_text = join_pieces(clipped_pieces)
        if not revealed_text:
          continue

        segments = []
        if clipped_pieces:
          split_indices = np.array_split(np.arange(len(clipped_pieces)), 4)
          for indices in split_indices:
            if len(indices) == 0:
              continue
            segment_text = join_pieces([clipped_pieces[idx] for idx in indices.tolist()])
            if segment_text:
              segments.append(segment_text)

        rows.append({
          'run': run['label'],
          'sample_index': int(record.get('global_sample_index', len(rows))),
          'target_reveal_fraction': float(snapshot['target_reveal_fraction']),
          'step_index': snapshot.get('step_index'),
          'revealed_fraction': snapshot.get('revealed_fraction'),
          'revealed_text': revealed_text,
          'gt_text': gt_text,
          'segments': segments,
        })
  return rows


def compute_diag1(diag1_rows):
  if not diag1_rows:
    return pd.DataFrame(), pd.DataFrame()

  text_pool = set()
  for row in diag1_rows:
    text_pool.add(row['revealed_text'])
    if row['gt_text']:
      text_pool.add(row['gt_text'])
    for segment in row['segments']:
      text_pool.add(segment)
  text_pool = sorted(text for text in text_pool if text)

  if not text_pool:
    return pd.DataFrame(), pd.DataFrame()

  vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
  matrix = vectorizer.fit_transform(text_pool)
  text_to_idx = {text: idx for idx, text in enumerate(text_pool)}

  def sim(text_a, text_b):
    if not text_a or not text_b:
      return np.nan
    idx_a = text_to_idx[text_a]
    idx_b = text_to_idx[text_b]
    return float(cosine_similarity(matrix[idx_a], matrix[idx_b])[0, 0])

  detail_rows = []
  for row in diag1_rows:
    segment_sims = []
    for i in range(len(row['segments'])):
      for j in range(i + 1, len(row['segments'])):
        segment_sims.append(sim(row['segments'][i], row['segments'][j]))
    coherence = float(np.mean(segment_sims)) if segment_sims else np.nan
    relevance = sim(row['revealed_text'], row['gt_text'])
    detail_rows.append({
      'run': row['run'],
      'sample_index': row['sample_index'],
      'target_reveal_fraction': row['target_reveal_fraction'],
      'step_index': row['step_index'],
      'revealed_fraction': row['revealed_fraction'],
      'coherence': coherence,
      'relevance_to_gt': relevance,
    })

  detail_df = pd.DataFrame(detail_rows)
  summary_df = detail_df.groupby(
    ['run', 'target_reveal_fraction'], as_index=False).agg(
      coherence=('coherence', 'mean'),
      relevance_to_gt=('relevance_to_gt', 'mean'),
      num_samples=('sample_index', 'count'))
  return detail_df, summary_df


def bucket_from_fraction(step_fraction):
  if step_fraction is None or math.isnan(step_fraction):
    return None
  if step_fraction < 0.25:
    return BUCKET_LABELS[0]
  if step_fraction < 0.50:
    return BUCKET_LABELS[1]
  if step_fraction < 0.75:
    return BUCKET_LABELS[2]
  return BUCKET_LABELS[3]


def compute_diag2(runs, analysis_mode):
  bucket_rows = []
  overall_rows = []

  for run in runs:
    counts = defaultdict(lambda: defaultdict(int))
    content_steps = []
    function_steps = []

    for record in run['records']:
      max_len = analysis_length(record, analysis_mode)
      step_list = record.get('first_unmask_step', [])[:max_len]
      step_frac_list = record.get('first_unmask_step_fraction', [])[:max_len]
      token_text_list = record.get('first_unmask_token_text', [])[:max_len]

      for step, step_frac, token_text in zip(step_list, step_frac_list, token_text_list):
        if step is None or int(step) < 0:
          continue
        step_fraction = float(step_frac) if step_frac is not None else np.nan
        bucket = bucket_from_fraction(step_fraction)
        if bucket is None:
          continue
        category = classify_token(token_text)
        counts[bucket][category] += 1
        if category == 'CONTENT':
          content_steps.append(step_fraction)
        elif category == 'FUNCTION':
          function_steps.append(step_fraction)

    for bucket in BUCKET_LABELS:
      content = counts[bucket]['CONTENT']
      function = counts[bucket]['FUNCTION']
      other = counts[bucket]['OTHER']
      total = content + function + other
      bucket_rows.append({
        'run': run['label'],
        'bucket': bucket,
        'content_count': content,
        'function_count': function,
        'other_count': other,
        'content_ratio': (content / total) if total else np.nan,
        'function_ratio': (function / total) if total else np.nan,
        'content_vs_function_ratio': (
          content / (content + function)) if (content + function) else np.nan,
      })

    mean_content = float(np.mean(content_steps)) if content_steps else np.nan
    mean_function = float(np.mean(function_steps)) if function_steps else np.nan
    overall_rows.append({
      'run': run['label'],
      'content_first_ratio': (
        mean_content / mean_function) if mean_function and not np.isnan(mean_function) else np.nan,
      'mean_content_step_fraction': mean_content,
      'mean_function_step_fraction': mean_function,
      'num_content_tokens': len(content_steps),
      'num_function_tokens': len(function_steps),
    })

  return pd.DataFrame(bucket_rows), pd.DataFrame(overall_rows)


def compute_diag3(runs, analysis_mode):
  sample_rows = []
  summary_rows = []

  for run in runs:
    early_fraction = float(run['meta'].get('early_fraction', 0.3))
    run_survivals = []

    for record in run['records']:
      max_len = analysis_length(record, analysis_mode)
      total_steps = int(record.get('total_sampling_steps', 0) or 0)
      if total_steps <= 0:
        continue
      early_cutoff = max(1, int(math.ceil(total_steps * early_fraction)))
      step_list = record.get('first_unmask_step', [])[:max_len]
      token_text_list = record.get('first_unmask_token_text', [])[:max_len]
      gt_tokens = [
        normalize_piece(piece)
        for piece in record.get('gt_answer_token_text', [])[:max_len]
      ]
      gt_tokens = [
        token for token in gt_tokens
        if token and token not in SPECIAL_TOKENS and any(ch.isalnum() for ch in token)]

      early_tokens = []
      for step, token_text in zip(step_list, token_text_list):
        if step is None or int(step) < 0 or int(step) > early_cutoff:
          continue
        token = normalize_piece(token_text)
        if not token or token in SPECIAL_TOKENS:
          continue
        if not any(ch.isalnum() for ch in token):
          continue
        early_tokens.append(token)

      if not early_tokens:
        survival = np.nan
      else:
        overlap = sum((Counter(early_tokens) & Counter(gt_tokens)).values())
        survival = float(overlap / len(early_tokens))
        run_survivals.append(survival)

      sample_rows.append({
        'run': run['label'],
        'sample_index': int(record.get('global_sample_index', len(sample_rows))),
        'early_token_count': len(early_tokens),
        'gt_token_count': len(gt_tokens),
        'survival_to_gt': survival,
      })

    summary_rows.append({
      'run': run['label'],
      'mean_survival_to_gt': float(np.mean(run_survivals)) if run_survivals else np.nan,
      'num_valid_samples': len(run_survivals),
    })

  return pd.DataFrame(sample_rows), pd.DataFrame(summary_rows)


def main():
  args = parse_args()
  os.makedirs(args.output_dir, exist_ok=True)
  runs = load_runs(args.run)

  diag1_rows = build_diag1_rows(runs, args.analysis_length)
  diag1_detail_df, diag1_summary_df = compute_diag1(diag1_rows)
  diag2_bucket_df, diag2_overall_df = compute_diag2(runs, args.analysis_length)
  diag3_sample_df, diag3_summary_df = compute_diag3(runs, args.analysis_length)

  outputs = {
    'diag1_detail.csv': diag1_detail_df,
    'diag1_summary.csv': diag1_summary_df,
    'diag2_bucket_summary.csv': diag2_bucket_df,
    'diag2_overall_summary.csv': diag2_overall_df,
    'diag3_per_sample.csv': diag3_sample_df,
    'diag3_summary.csv': diag3_summary_df,
  }
  for filename, df in outputs.items():
    df.to_csv(os.path.join(args.output_dir, filename), index=False)

  summary_payload = {
    'runs': [run['label'] for run in runs],
    'analysis_length': args.analysis_length,
    'diag1_summary': diag1_summary_df.to_dict(orient='records'),
    'diag2_overall_summary': diag2_overall_df.to_dict(orient='records'),
    'diag3_summary': diag3_summary_df.to_dict(orient='records'),
  }
  with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary_payload, f, indent=2, ensure_ascii=False)

  print('Saved offline diagnostics to', args.output_dir)
  if not diag1_summary_df.empty:
    print('\n[Diag1] Early-step global coherence / relevance')
    print(diag1_summary_df.to_string(index=False))
  if not diag2_overall_df.empty:
    print('\n[Diag2] Denoising order analysis')
    print(diag2_overall_df.to_string(index=False))
  if not diag3_summary_df.empty:
    print('\n[Diag3] Skeleton survival rate')
    print(diag3_summary_df.to_string(index=False))


if __name__ == '__main__':
  main()
