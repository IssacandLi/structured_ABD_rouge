import argparse
import functools
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize


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
CONTENT_POS_PREFIXES = ('NN', 'VB', 'JJ')
CONTENT_POS_TAGS = {'RB', 'RBR', 'RBS'}
FUNCTION_POS_TAGS = {
  'DT', 'PDT', 'WDT', 'IN', 'CC', 'TO', 'PRP', 'PRP$',
  'WP', 'WP$', 'WRB', 'MD', 'EX', 'POS', 'RP', 'UH',
}
CONTENT_UPOS_TAGS = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'NUM'}
FUNCTION_UPOS_TAGS = {
  'DET', 'ADP', 'CCONJ', 'SCONJ', 'PART', 'PRON', 'AUX',
}


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
  parser.add_argument(
    '--timing_mode',
    choices=['global_step', 'reveal_rank'],
    default='global_step',
    help=(
      'How to define early/late timing for Diag2/Diag3. '
      '"global_step" uses absolute denoising steps; '
      '"reveal_rank" uses per-sample answer-token reveal order.'))
  parser.add_argument(
    '--diag1_backend',
    choices=['auto', 'tfidf', 'lsa', 'transformer'],
    default='auto',
    help=(
      'Similarity backend for Diag1. '
      '"auto" tries a local transformer only if --diag1_encoder_model is set, '
      'otherwise falls back to LSA.'))
  parser.add_argument(
    '--diag1_encoder_model',
    default=None,
    help='Local path or cached HF model id for transformer-based Diag1 embeddings.')
  parser.add_argument(
    '--diag1_svd_components',
    type=int,
    default=128,
    help='Number of SVD dimensions for LSA-based Diag1 embeddings.')
  parser.add_argument(
    '--diag1_coherence_mode',
    choices=['position_bins', 'revealed_chunks'],
    default='position_bins',
    help=(
      'How to compute Diag1 coherence. '
      '"position_bins" splits the full answer span into 4 position quartiles; '
      '"revealed_chunks" preserves the older split-over-revealed-token behavior.'))
  parser.add_argument(
    '--transformers_cache',
    default=None,
    help='Writable cache directory to use when loading local transformer models.')
  parser.add_argument(
    '--diag2_backend',
    choices=['auto', 'wordlist', 'nltk_pos', 'spacy_pos'],
    default='auto',
    help=(
      'Classifier backend for Diag2. '
      '"auto" prefers spaCy POS tagging, then NLTK, then wordlist heuristics.'))
  parser.add_argument(
    '--nltk_data_dir',
    default=None,
    help='Optional local NLTK data directory containing the averaged perceptron tagger.')
  parser.add_argument(
    '--spacy_model',
    default='en_core_web_sm',
    help='spaCy model package/path to use for Diag2 POS tagging.')
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


def normalize_word(text):
  token = normalize_piece(text)
  token = re.sub(r'^[^\w]+', '', token)
  token = re.sub(r'[^\w]+$', '', token)
  return token


def normalize_surface_word(text):
  if text is None:
    return ''
  token = str(text).replace('\n', ' ').strip()
  token = re.sub(r'^[^\w]+', '', token)
  token = re.sub(r'[^\w]+$', '', token)
  token = re.sub(r'\s+', ' ', token).strip()
  return token


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


def _group_token_pieces(token_text_list):
  groups = []
  current = None
  for idx, piece in enumerate(token_text_list):
    raw = '' if piece is None else str(piece)
    if current is None or re.match(r'^\s', raw):
      if current is not None:
        groups.append(current)
      current = {
        'positions': [idx],
        'surface': raw,
      }
    else:
      current['positions'].append(idx)
      current['surface'] += raw

  if current is not None:
    groups.append(current)

  normalized_groups = []
  for group in groups:
    normalized_groups.append({
      'positions': list(group['positions']),
      'surface': group['surface'],
      'text': normalize_word(group['surface']),
    })
  return normalized_groups


def _classify_word_wordlist(word):
  if not word or word in SPECIAL_TOKENS:
    return 'OTHER'
  if not any(ch.isalpha() for ch in word):
    return 'OTHER'
  if word in FUNCTION_WORDS:
    return 'FUNCTION'
  return 'CONTENT'


def _classify_pos_tag(tag):
  if not tag:
    return 'OTHER'
  tag = str(tag)
  if tag in FUNCTION_POS_TAGS:
    return 'FUNCTION'
  if tag in CONTENT_POS_TAGS or tag.startswith(CONTENT_POS_PREFIXES):
    return 'CONTENT'
  return 'OTHER'


def _classify_spacy_token(token):
  upos = (getattr(token, 'pos_', None) or '').upper()
  if upos in FUNCTION_UPOS_TAGS:
    return 'FUNCTION'
  if upos in CONTENT_UPOS_TAGS:
    return 'CONTENT'
  tag = getattr(token, 'tag_', None)
  return _classify_pos_tag(tag)


def _nltk_tagger_available(nltk_data_dir=None):
  try:
    import nltk
    if nltk_data_dir:
      nltk.data.path.insert(0, nltk_data_dir)
    nltk.data.find('taggers/averaged_perceptron_tagger')
    return True
  except Exception:
    return False


@functools.lru_cache(maxsize=None)
def _load_spacy_model_cached(spacy_model):
  import spacy

  return spacy.load(spacy_model)


def _spacy_tagger_available(spacy_model='en_core_web_sm'):
  try:
    nlp = _load_spacy_model_cached(spacy_model)
  except Exception:
    return False
  return any(name in nlp.pipe_names for name in ('tagger', 'morphologizer'))


def _resolve_diag2_backend(requested_backend, nltk_data_dir=None, spacy_model='en_core_web_sm'):
  notes = []
  if requested_backend == 'spacy_pos':
    if _spacy_tagger_available(spacy_model=spacy_model):
      return 'spacy_pos', notes
    notes.append(
      f'Requested Diag2 backend "spacy_pos" but spaCy model '
      f'"{spacy_model}" is unavailable; falling back to NLTK/wordlist.')
    if _nltk_tagger_available(nltk_data_dir=nltk_data_dir):
      return 'nltk_pos', notes
    return 'wordlist', notes
  if requested_backend == 'wordlist':
    return 'wordlist', notes
  if requested_backend == 'nltk_pos':
    if _nltk_tagger_available(nltk_data_dir=nltk_data_dir):
      return 'nltk_pos', notes
    notes.append(
      'Requested Diag2 backend "nltk_pos" but local averaged_perceptron_tagger '
      'data is unavailable; falling back to wordlist heuristics.')
    return 'wordlist', notes
  if _spacy_tagger_available(spacy_model=spacy_model):
    notes.append(
      f'Diag2 auto backend selected spaCy POS tagging ({spacy_model}).')
    return 'spacy_pos', notes
  if _nltk_tagger_available(nltk_data_dir=nltk_data_dir):
    notes.append('Diag2 auto backend selected local NLTK POS tagging.')
    return 'nltk_pos', notes
  notes.append(
    'Diag2 auto backend could not find a usable spaCy/NLTK POS tagger; '
    'using wordlist heuristics.')
  return 'wordlist', notes


def _categorize_positions_wordlist(token_text_list):
  categories = ['OTHER'] * len(token_text_list)
  for group in _group_token_pieces(token_text_list):
    category = _classify_word_wordlist(group['text'])
    for position in group['positions']:
      categories[position] = category
  return categories


def _categorize_positions_nltk(token_text_list, nltk_data_dir=None):
  import nltk

  if nltk_data_dir:
    nltk.data.path.insert(0, nltk_data_dir)

  groups = _group_token_pieces(token_text_list)
  words = [group['text'] if group['text'] else group['surface'] for group in groups]
  words = [word if word else '' for word in words]
  tagged_words = nltk.pos_tag(words)

  categories = ['OTHER'] * len(token_text_list)
  for group, (_, tag) in zip(groups, tagged_words):
    category = _classify_pos_tag(tag)
    for position in group['positions']:
      categories[position] = category
  return categories


def _categorize_positions_spacy(token_text_list, spacy_model='en_core_web_sm'):
  from spacy.tokens import Doc

  nlp = _load_spacy_model_cached(spacy_model)
  groups = _group_token_pieces(token_text_list)
  categories = ['OTHER'] * len(token_text_list)
  taggable_groups = []
  words = []
  for group in groups:
    word = normalize_surface_word(group['surface'])
    if not word or not any(ch.isalnum() for ch in word):
      continue
    taggable_groups.append(group)
    words.append(word)

  if not words:
    return categories

  doc = Doc(nlp.vocab, words=words)
  for _, proc in nlp.pipeline:
    doc = proc(doc)

  for group, token in zip(taggable_groups, doc):
    category = _classify_spacy_token(token)
    for position in group['positions']:
      categories[position] = category
  return categories


def _categorize_positions(token_text_list, backend, nltk_data_dir=None, spacy_model='en_core_web_sm'):
  if backend == 'spacy_pos':
    try:
      return _categorize_positions_spacy(
        token_text_list, spacy_model=spacy_model)
    except Exception:
      if _nltk_tagger_available(nltk_data_dir=nltk_data_dir):
        return _categorize_positions_nltk(
          token_text_list, nltk_data_dir=nltk_data_dir)
      return _categorize_positions_wordlist(token_text_list)
  if backend == 'nltk_pos':
    try:
      return _categorize_positions_nltk(
        token_text_list, nltk_data_dir=nltk_data_dir)
    except Exception:
      return _categorize_positions_wordlist(token_text_list)
  return _categorize_positions_wordlist(token_text_list)


def _build_tfidf_similarity_backend(text_pool):
  vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
  matrix = vectorizer.fit_transform(text_pool)
  text_to_idx = {text: idx for idx, text in enumerate(text_pool)}

  def sim(text_a, text_b):
    if not text_a or not text_b:
      return np.nan
    idx_a = text_to_idx[text_a]
    idx_b = text_to_idx[text_b]
    return float(cosine_similarity(matrix[idx_a], matrix[idx_b])[0, 0])

  return sim


def _build_lsa_similarity_backend(text_pool, svd_components):
  vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
  matrix = vectorizer.fit_transform(text_pool)
  max_components = min(matrix.shape[0] - 1, matrix.shape[1] - 1, svd_components)
  if max_components < 2:
    return _build_tfidf_similarity_backend(text_pool), 'tfidf', [
      'Diag1 LSA backend had too few texts/features for SVD; falling back to TF-IDF.'
    ]

  svd = TruncatedSVD(n_components=max_components, random_state=0)
  embeddings = svd.fit_transform(matrix)
  embeddings = sk_normalize(embeddings)
  text_to_idx = {text: idx for idx, text in enumerate(text_pool)}

  def sim(text_a, text_b):
    if not text_a or not text_b:
      return np.nan
    idx_a = text_to_idx[text_a]
    idx_b = text_to_idx[text_b]
    return float(np.dot(embeddings[idx_a], embeddings[idx_b]))

  notes = [f'Diag1 using LSA sentence-like embeddings with {max_components} dimensions.']
  return sim, 'lsa', notes


def _build_transformer_similarity_backend(text_pool, encoder_model, transformers_cache):
  import torch
  from transformers import AutoModel, AutoTokenizer

  if transformers_cache:
    os.environ['TRANSFORMERS_CACHE'] = transformers_cache

  tokenizer = AutoTokenizer.from_pretrained(
    encoder_model, local_files_only=True)
  model = AutoModel.from_pretrained(
    encoder_model, local_files_only=True)
  model.eval()
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

  embeddings = []
  batch_size = 32
  with torch.no_grad():
    for start in range(0, len(text_pool), batch_size):
      batch = text_pool[start:start + batch_size]
      encoded = tokenizer(
        batch,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=256,
      )
      outputs = model(**encoded)
      hidden = outputs.last_hidden_state
      attn = encoded['attention_mask'].unsqueeze(-1)
      pooled = (hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
      pooled = torch.nn.functional.normalize(pooled, dim=-1)
      embeddings.append(pooled.cpu().numpy())
  embeddings = np.concatenate(embeddings, axis=0)
  text_to_idx = {text: idx for idx, text in enumerate(text_pool)}

  def sim(text_a, text_b):
    if not text_a or not text_b:
      return np.nan
    idx_a = text_to_idx[text_a]
    idx_b = text_to_idx[text_b]
    return float(np.dot(embeddings[idx_a], embeddings[idx_b]))

  notes = [f'Diag1 using local transformer encoder: {encoder_model}']
  return sim, 'transformer', notes


def _resolve_diag1_backend(
  requested_backend,
  text_pool,
  encoder_model=None,
  transformers_cache=None,
  svd_components=128,
):
  notes = []
  if requested_backend == 'tfidf':
    notes.append('Diag1 using TF-IDF cosine similarity.')
    return _build_tfidf_similarity_backend(text_pool), 'tfidf', notes
  if requested_backend == 'lsa':
    sim_fn, backend_name, lsa_notes = _build_lsa_similarity_backend(
      text_pool, svd_components)
    return sim_fn, backend_name, notes + lsa_notes
  if requested_backend == 'transformer':
    if not encoder_model:
      notes.append(
        'Requested Diag1 transformer backend without --diag1_encoder_model; '
        'falling back to LSA.')
      sim_fn, backend_name, lsa_notes = _build_lsa_similarity_backend(
        text_pool, svd_components)
      return sim_fn, backend_name, notes + lsa_notes
    try:
      return _build_transformer_similarity_backend(
        text_pool, encoder_model, transformers_cache)
    except Exception as exc:
      notes.append(
        f'Diag1 transformer backend failed locally ({exc}); falling back to LSA.')
      sim_fn, backend_name, lsa_notes = _build_lsa_similarity_backend(
        text_pool, svd_components)
      return sim_fn, backend_name, notes + lsa_notes

  if encoder_model:
    try:
      sim_fn, backend_name, tf_notes = _build_transformer_similarity_backend(
        text_pool, encoder_model, transformers_cache)
      return sim_fn, backend_name, ['Diag1 auto backend selected transformer.'] + tf_notes
    except Exception as exc:
      notes.append(
        f'Diag1 auto backend could not load local transformer encoder ({exc}); '
        'falling back to LSA.')
  else:
    notes.append(
      'Diag1 auto backend has no local encoder model specified; using LSA fallback.')
  sim_fn, backend_name, lsa_notes = _build_lsa_similarity_backend(
    text_pool, svd_components)
  return sim_fn, backend_name, notes + lsa_notes


def _build_position_bin_segments(revealed_items, max_len):
  if max_len <= 0:
    return []
  position_to_piece = {
    int(pos): piece for pos, piece in revealed_items
    if int(pos) < max_len
  }
  segments = []
  for indices in np.array_split(np.arange(max_len), 4):
    segment_text = join_pieces([
      position_to_piece[int(pos)]
      for pos in indices.tolist()
      if int(pos) in position_to_piece
    ])
    if segment_text:
      segments.append(segment_text)
  return segments


def _build_revealed_chunk_segments(clipped_pieces):
  segments = []
  if clipped_pieces:
    split_indices = np.array_split(np.arange(len(clipped_pieces)), 4)
    for indices in split_indices:
      if len(indices) == 0:
        continue
      segment_text = join_pieces([clipped_pieces[idx] for idx in indices.tolist()])
      if segment_text:
        segments.append(segment_text)
  return segments


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
        revealed_items = []
        clipped_pieces = []
        for pos, piece in zip(
          snapshot.get('revealed_positions', []),
          snapshot.get('revealed_token_text', []),
        ):
          if max_len and int(pos) >= max_len:
            continue
          revealed_items.append((int(pos), piece))
          clipped_pieces.append(piece)
        revealed_text = join_pieces(clipped_pieces)
        if not revealed_text:
          continue

        rows.append({
          'run': run['label'],
          'sample_index': int(record.get('global_sample_index', len(rows))),
          'snapshot_source': 'reveal_fraction',
          'target_reveal_fraction': float(snapshot['target_reveal_fraction']),
          'target_step_fraction': np.nan,
          'step_index': snapshot.get('step_index'),
          'revealed_fraction': snapshot.get('revealed_fraction'),
          'revealed_text': revealed_text,
          'gt_text': gt_text,
          'position_bin_segments': _build_position_bin_segments(
            revealed_items, max_len),
          'revealed_chunk_segments': _build_revealed_chunk_segments(
            clipped_pieces),
        })

      for snapshot in record.get('step_snapshots', []):
        if not snapshot.get('captured', False):
          continue
        revealed_items = []
        clipped_pieces = []
        for pos, piece in zip(
          snapshot.get('visible_positions', []),
          snapshot.get('visible_token_text', []),
        ):
          if max_len and int(pos) >= max_len:
            continue
          revealed_items.append((int(pos), piece))
          clipped_pieces.append(piece)
        revealed_text = snapshot.get('partial_text') or join_pieces(clipped_pieces)
        if not revealed_text:
          continue

        target_step_fraction = float(snapshot['target_step_fraction'])
        rows.append({
          'run': run['label'],
          'sample_index': int(record.get('global_sample_index', len(rows))),
          'snapshot_source': 'step_fraction',
          # Keep this legacy column populated so existing plotting scripts still work.
          'target_reveal_fraction': target_step_fraction,
          'target_step_fraction': target_step_fraction,
          'step_index': snapshot.get('step_index'),
          'revealed_fraction': snapshot.get('revealed_fraction'),
          'revealed_text': revealed_text,
          'gt_text': gt_text,
          'position_bin_segments': _build_position_bin_segments(
            revealed_items, max_len),
          'revealed_chunk_segments': _build_revealed_chunk_segments(
            clipped_pieces),
        })
  return rows


def compute_diag1(
  diag1_rows,
  diag1_backend='auto',
  diag1_encoder_model=None,
  diag1_svd_components=128,
  diag1_coherence_mode='position_bins',
  transformers_cache=None,
):
  if not diag1_rows:
    return pd.DataFrame(), pd.DataFrame(), {
      'requested_backend': diag1_backend,
      'actual_backend': None,
      'coherence_mode': diag1_coherence_mode,
      'notes': [],
    }

  text_pool = set()
  for row in diag1_rows:
    text_pool.add(row['revealed_text'])
    if row['gt_text']:
      text_pool.add(row['gt_text'])
    segment_key = (
      'position_bin_segments'
      if diag1_coherence_mode == 'position_bins'
      else 'revealed_chunk_segments'
    )
    for segment in row[segment_key]:
      text_pool.add(segment)
  text_pool = sorted(text for text in text_pool if text)

  if not text_pool:
    return pd.DataFrame(), pd.DataFrame(), {
      'requested_backend': diag1_backend,
      'actual_backend': None,
      'coherence_mode': diag1_coherence_mode,
      'notes': [],
    }

  sim, backend_name, backend_notes = _resolve_diag1_backend(
    diag1_backend,
    text_pool,
    encoder_model=diag1_encoder_model,
    transformers_cache=transformers_cache,
    svd_components=diag1_svd_components,
  )

  detail_rows = []
  for row in diag1_rows:
    segments = row[
      'position_bin_segments'
      if diag1_coherence_mode == 'position_bins'
      else 'revealed_chunk_segments'
    ]
    segment_sims = []
    for i in range(len(segments)):
      for j in range(i + 1, len(segments)):
        segment_sims.append(sim(segments[i], segments[j]))
    coherence = float(np.mean(segment_sims)) if segment_sims else np.nan
    relevance = sim(row['revealed_text'], row['gt_text'])
    detail_rows.append({
      'run': row['run'],
      'sample_index': row['sample_index'],
      'snapshot_source': row.get('snapshot_source', 'reveal_fraction'),
      'target_reveal_fraction': row['target_reveal_fraction'],
      'target_step_fraction': row.get('target_step_fraction', np.nan),
      'step_index': row['step_index'],
      'revealed_fraction': row['revealed_fraction'],
      'coherence': coherence,
      'relevance_to_gt': relevance,
    })

  detail_df = pd.DataFrame(detail_rows)
  summary_df = detail_df.groupby(
    ['run', 'snapshot_source', 'target_reveal_fraction'], as_index=False).agg(
      target_step_fraction=('target_step_fraction', 'mean'),
      coherence=('coherence', 'mean'),
      relevance_to_gt=('relevance_to_gt', 'mean'),
      num_samples=('sample_index', 'count'))
  return detail_df, summary_df, {
    'requested_backend': diag1_backend,
    'actual_backend': backend_name,
    'coherence_mode': diag1_coherence_mode,
    'notes': backend_notes,
  }


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


def _ranked_reveals(record, max_len):
  step_list = record.get('first_unmask_step', [])[:max_len]
  token_text_list = record.get('first_unmask_token_text', [])[:max_len]
  revealed = []
  for idx, (step, token_text) in enumerate(zip(step_list, token_text_list)):
    if step is None or int(step) < 0:
      continue
    revealed.append({
      'position': idx,
      'step': int(step),
      'token_text': token_text,
    })
  revealed.sort(key=lambda item: (item['step'], item['position']))
  total = len(revealed)
  if total == 0:
    return []
  ranked = []
  for rank, item in enumerate(revealed):
    rank_fraction = float((rank + 0.5) / total)
    ranked.append({
      'position': item['position'],
      'step': item['step'],
      'token_text': item['token_text'],
      'rank': rank,
      'rank_fraction': rank_fraction,
      'total_revealed': total,
    })
  return ranked


def compute_diag2(
  runs,
  analysis_mode,
  timing_mode='global_step',
  diag2_backend='auto',
  nltk_data_dir=None,
  spacy_model='en_core_web_sm',
):
  actual_backend, backend_notes = _resolve_diag2_backend(
    diag2_backend,
    nltk_data_dir=nltk_data_dir,
    spacy_model=spacy_model)
  bucket_rows = []
  overall_rows = []

  for run in runs:
    counts = defaultdict(lambda: defaultdict(int))
    content_steps = []
    function_steps = []

    for record in run['records']:
      max_len = analysis_length(record, analysis_mode)
      if timing_mode == 'reveal_rank':
        timed_tokens = [
          (
            item['step'],
            item['rank_fraction'],
            item['token_text'],
            item['position'],
          )
          for item in _ranked_reveals(record, max_len)
        ]
      else:
        step_list = record.get('first_unmask_step', [])[:max_len]
        step_frac_list = record.get('first_unmask_step_fraction', [])[:max_len]
        token_text_list = record.get('first_unmask_token_text', [])[:max_len]
        timed_tokens = [
          (step, step_frac, token_text, answer_pos)
          for answer_pos, (step, step_frac, token_text) in enumerate(zip(
            step_list, step_frac_list, token_text_list))
        ]

      token_categories = _categorize_positions(
        record.get('first_unmask_token_text', [])[:max_len],
        backend=actual_backend,
        nltk_data_dir=nltk_data_dir,
        spacy_model=spacy_model,
      )

      for step, step_frac, token_text, answer_pos in timed_tokens:
        if step is None or int(step) < 0:
          continue
        step_fraction = float(step_frac) if step_frac is not None else np.nan
        bucket = bucket_from_fraction(step_fraction)
        if bucket is None:
          continue
        category = (
          token_categories[answer_pos]
          if answer_pos < len(token_categories)
          else 'OTHER'
        )
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

  return pd.DataFrame(bucket_rows), pd.DataFrame(overall_rows), {
    'requested_backend': diag2_backend,
    'actual_backend': actual_backend,
    'spacy_model': spacy_model if actual_backend == 'spacy_pos' else None,
    'notes': backend_notes,
  }


def compute_diag3(runs, analysis_mode, timing_mode='global_step'):
  sample_rows = []
  summary_rows = []

  for run in runs:
    early_fraction = float(run['meta'].get('early_fraction', 0.3))
    run_survivals = []

    for record in run['records']:
      max_len = analysis_length(record, analysis_mode)
      if timing_mode == 'reveal_rank':
        ranked_reveals = _ranked_reveals(record, max_len)
        early_count = max(1, int(math.ceil(len(ranked_reveals) * early_fraction))) \
          if ranked_reveals else 0
        selected_reveals = ranked_reveals[:early_count]
      else:
        total_steps = int(record.get('total_sampling_steps', 0) or 0)
        if total_steps <= 0:
          continue
        early_cutoff = max(1, int(math.ceil(total_steps * early_fraction)))
        step_list = record.get('first_unmask_step', [])[:max_len]
        token_text_list = record.get('first_unmask_token_text', [])[:max_len]
        selected_reveals = []
        for step, token_text in zip(step_list, token_text_list):
          if step is None or int(step) < 0 or int(step) > early_cutoff:
            continue
          selected_reveals.append({
            'step': int(step),
            'token_text': token_text,
          })

      gt_tokens = [
        normalize_piece(piece)
        for piece in record.get('gt_answer_token_text', [])[:max_len]
      ]
      gt_tokens = [
        token for token in gt_tokens
        if token and token not in SPECIAL_TOKENS and any(ch.isalnum() for ch in token)]

      early_tokens = []
      for item in selected_reveals:
        token = normalize_piece(item['token_text'])
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
        'early_committed_fraction': (
          len(early_tokens) / max_len if max_len else np.nan),
        'survival_to_gt': survival,
      })

    summary_rows.append({
      'run': run['label'],
      'mean_survival_to_gt': float(np.mean(run_survivals)) if run_survivals else np.nan,
      'mean_early_committed_fraction': float(np.nanmean([
        row['early_committed_fraction']
        for row in sample_rows
        if row['run'] == run['label']
      ])) if any(row['run'] == run['label'] for row in sample_rows) else np.nan,
      'num_valid_samples': len(run_survivals),
    })

  return pd.DataFrame(sample_rows), pd.DataFrame(summary_rows)


def main():
  args = parse_args()
  os.makedirs(args.output_dir, exist_ok=True)
  runs = load_runs(args.run)

  diag1_rows = build_diag1_rows(runs, args.analysis_length)
  diag1_detail_df, diag1_summary_df, diag1_meta = compute_diag1(
    diag1_rows,
    diag1_backend=args.diag1_backend,
    diag1_encoder_model=args.diag1_encoder_model,
    diag1_svd_components=args.diag1_svd_components,
    diag1_coherence_mode=args.diag1_coherence_mode,
    transformers_cache=args.transformers_cache,
  )
  diag2_bucket_df, diag2_overall_df, diag2_meta = compute_diag2(
    runs,
    args.analysis_length,
    timing_mode=args.timing_mode,
    diag2_backend=args.diag2_backend,
    nltk_data_dir=args.nltk_data_dir,
    spacy_model=args.spacy_model,
  )
  diag3_sample_df, diag3_summary_df = compute_diag3(
    runs, args.analysis_length, timing_mode=args.timing_mode)

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
    'timing_mode': args.timing_mode,
    'diag1_backend': diag1_meta,
    'diag2_backend': diag2_meta,
    'diag1_summary': diag1_summary_df.to_dict(orient='records'),
    'diag2_overall_summary': diag2_overall_df.to_dict(orient='records'),
    'diag3_summary': diag3_summary_df.to_dict(orient='records'),
  }
  with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary_payload, f, indent=2, ensure_ascii=False)

  print('Saved offline diagnostics to', args.output_dir)
  if diag1_meta.get('actual_backend'):
    print(f"\n[Diag1 backend] requested={diag1_meta['requested_backend']} actual={diag1_meta['actual_backend']}")
    for note in diag1_meta.get('notes', []):
      print(' ', note)
  if diag2_meta.get('actual_backend'):
    print(f"\n[Diag2 backend] requested={diag2_meta['requested_backend']} actual={diag2_meta['actual_backend']}")
    for note in diag2_meta.get('notes', []):
      print(' ', note)
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
