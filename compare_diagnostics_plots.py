import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import offline_diagnostics as od


BUCKET_ORDER = ['early', 'mid_early', 'mid_late', 'late']


def parse_args():
  parser = argparse.ArgumentParser(
    description='Compare sampling diagnostics from multiple runs and plot them in one figure.')
  parser.add_argument(
    '--run',
    action='append',
    required=True,
    help='Run spec in the form label=/path/to/diagnostics.json')
  parser.add_argument(
    '--output_dir',
    required=True,
    help='Directory to save summary CSVs and plots.')
  parser.add_argument(
    '--analysis_length',
    choices=['gt', 'final', 'full'],
    default='gt',
    help='Which token span to use for per-position analysis.')
  parser.add_argument(
    '--timing_mode',
    choices=['global_step', 'reveal_rank'],
    default='reveal_rank',
    help=(
      'How to define early/late timing for Diag2/Diag3. '
      '"reveal_rank" is fairer for semi-AR baselines.'))
  parser.add_argument(
    '--diag1_backend',
    choices=['auto', 'tfidf', 'lsa', 'transformer'],
    default='auto',
    help='Similarity backend for Diag1.')
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
    '--transformers_cache',
    default=None,
    help='Writable cache directory to use when loading local transformer models.')
  parser.add_argument(
    '--diag2_backend',
    choices=['auto', 'wordlist', 'nltk_pos'],
    default='auto',
    help='Classifier backend for Diag2.')
  parser.add_argument(
    '--nltk_data_dir',
    default=None,
    help='Optional local NLTK data directory containing the averaged perceptron tagger.')
  parser.add_argument(
    '--title',
    default='Sampling Diagnostics Comparison',
    help='Figure title.')
  return parser.parse_args()


def _save_tables(output_dir, diag1_detail_df, diag1_summary_df,
                 diag2_bucket_df, diag2_overall_df,
                 diag3_sample_df, diag3_summary_df):
  outputs = {
    'diag1_detail.csv': diag1_detail_df,
    'diag1_summary.csv': diag1_summary_df,
    'diag2_bucket_summary.csv': diag2_bucket_df,
    'diag2_overall_summary.csv': diag2_overall_df,
    'diag3_per_sample.csv': diag3_sample_df,
    'diag3_summary.csv': diag3_summary_df,
  }
  for filename, df in outputs.items():
    df.to_csv(os.path.join(output_dir, filename), index=False)


def _save_summary_json(
  output_dir,
  runs,
  args,
  diag1_meta,
  diag2_meta,
  diag1_summary_df,
  diag2_overall_df,
  diag3_summary_df,
):
  payload = {
    'runs': [run['label'] for run in runs],
    'title': args.title,
    'analysis_length': args.analysis_length,
    'timing_mode': args.timing_mode,
    'diag1_backend': diag1_meta,
    'diag2_backend': diag2_meta,
    'diag1_summary': diag1_summary_df.to_dict(orient='records'),
    'diag2_overall_summary': diag2_overall_df.to_dict(orient='records'),
    'diag3_summary': diag3_summary_df.to_dict(orient='records'),
  }
  with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)


def _plot_diag1(ax, df, y_key, title, ylabel):
  if df.empty:
    ax.set_title(title)
    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax.set_axis_off()
    return

  for run_name in sorted(df['run'].unique()):
    run_df = df[df['run'] == run_name].sort_values('target_reveal_fraction')
    ax.plot(
      run_df['target_reveal_fraction'],
      run_df[y_key],
      marker='o',
      linewidth=2,
      label=run_name,
    )
  ax.set_title(title)
  ax.set_xlabel('Reveal Fraction')
  ax.set_ylabel(ylabel)
  ax.grid(alpha=0.25)
  ax.legend()


def _plot_diag2(ax, bucket_df, overall_df, timing_mode):
  if bucket_df.empty:
    ax.set_title('Diag2: Denoising Order')
    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax.set_axis_off()
    return

  x = np.arange(len(BUCKET_ORDER))
  for run_name in sorted(bucket_df['run'].unique()):
    run_df = bucket_df[bucket_df['run'] == run_name].copy()
    run_df['bucket'] = run_df['bucket'].astype(str)
    run_df = run_df.set_index('bucket').reindex(BUCKET_ORDER).reset_index()

    label = run_name
    overall_row = overall_df[overall_df['run'] == run_name]
    if not overall_row.empty:
      cfr = overall_row.iloc[0]['content_first_ratio']
      if cfr == cfr:
        label = f'{run_name} (CFR={cfr:.3f})'

    ax.plot(
      x,
      run_df['content_vs_function_ratio'],
      marker='o',
      linewidth=2,
      label=label,
    )

  if timing_mode == 'reveal_rank':
    ax.set_title('Diag2: Content-First by Reveal-Rank Bucket')
  else:
    ax.set_title('Diag2: Content-First by Step Bucket')
  ax.set_xlabel('Bucket')
  ax.set_ylabel('Content / (Content + Function)')
  ax.set_xticks(x)
  ax.set_xticklabels(BUCKET_ORDER, rotation=15)
  ax.set_ylim(0.0, 1.0)
  ax.grid(alpha=0.25)
  ax.legend()


def _plot_diag3(ax, summary_df, timing_mode):
  if summary_df.empty:
    ax.set_title('Diag3: Skeleton Survival')
    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax.set_axis_off()
    return

  run_names = summary_df['run'].tolist()
  values = summary_df['mean_survival_to_gt'].tolist()
  x = np.arange(len(run_names))
  ax.bar(x, values, width=0.6)
  if timing_mode == 'reveal_rank':
    ax.set_title('Diag3: Early-Reveal Skeleton Survival')
  else:
    ax.set_title('Diag3: Skeleton Survival Rate')
  ax.set_xlabel('Run')
  ax.set_ylabel('Mean Survival to GT')
  ax.set_xticks(x)
  ax.set_xticklabels(run_names, rotation=15)
  ax.set_ylim(0.0, max(1.0, np.nanmax(values) * 1.15 if len(values) else 1.0))
  ax.grid(axis='y', alpha=0.25)
  for idx, value in enumerate(values):
    if value == value:
      ax.text(idx, value + 0.02, f'{value:.3f}', ha='center', va='bottom')


def main():
  args = parse_args()
  os.makedirs(args.output_dir, exist_ok=True)

  runs = od.load_runs(args.run)
  diag1_rows = od.build_diag1_rows(runs, args.analysis_length)
  diag1_detail_df, diag1_summary_df, diag1_meta = od.compute_diag1(
    diag1_rows,
    diag1_backend=args.diag1_backend,
    diag1_encoder_model=args.diag1_encoder_model,
    diag1_svd_components=args.diag1_svd_components,
    transformers_cache=args.transformers_cache,
  )
  diag2_bucket_df, diag2_overall_df, diag2_meta = od.compute_diag2(
    runs,
    args.analysis_length,
    timing_mode=args.timing_mode,
    diag2_backend=args.diag2_backend,
    nltk_data_dir=args.nltk_data_dir,
  )
  diag3_sample_df, diag3_summary_df = od.compute_diag3(
    runs, args.analysis_length, timing_mode=args.timing_mode)

  _save_tables(
    args.output_dir,
    diag1_detail_df, diag1_summary_df,
    diag2_bucket_df, diag2_overall_df,
    diag3_sample_df, diag3_summary_df,
  )
  _save_summary_json(
    args.output_dir,
    runs,
    args,
    diag1_meta,
    diag2_meta,
    diag1_summary_df,
    diag2_overall_df,
    diag3_summary_df,
  )

  fig, axes = plt.subplots(2, 2, figsize=(14, 10))
  fig.suptitle(args.title, fontsize=15)

  _plot_diag1(
    axes[0, 0], diag1_summary_df,
    y_key='coherence',
    title='Diag1: Early-Step Global Coherence',
    ylabel='Coherence')
  _plot_diag1(
    axes[0, 1], diag1_summary_df,
    y_key='relevance_to_gt',
    title='Diag1: Relevance to GT',
    ylabel='Relevance to GT')
  _plot_diag2(axes[1, 0], diag2_bucket_df, diag2_overall_df, args.timing_mode)
  _plot_diag3(axes[1, 1], diag3_summary_df, args.timing_mode)

  plt.tight_layout(rect=[0, 0, 1, 0.96])

  png_path = os.path.join(args.output_dir, 'diagnostics_comparison.png')
  pdf_path = os.path.join(args.output_dir, 'diagnostics_comparison.pdf')
  fig.savefig(png_path, dpi=220, bbox_inches='tight')
  fig.savefig(pdf_path, bbox_inches='tight')
  plt.close(fig)

  print('Diag1 backend:', diag1_meta)
  print('Diag2 backend:', diag2_meta)
  print('Saved comparison tables and plots to', args.output_dir)
  print('PNG:', png_path)
  print('PDF:', pdf_path)


if __name__ == '__main__':
  main()
