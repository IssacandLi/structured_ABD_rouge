import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=False,
    weights_only=False).to('cuda')

@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    #==========disable validation=========
    if dl is None:
      print(f'Skipping {dl_type} dataloader batch (None).')
      continue
    print(f'Printing {dl_type} dataloader batch.')
    print(f'Printing {dl_type} dataloader batch.')

    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def _diagnostics_enabled(config):
  return bool(getattr(getattr(config, 'diagnostics', None), 'enabled', False))


def _diagnostics_save_path(config):
  diagnostics_cfg = getattr(config, 'diagnostics', None)
  custom_path = getattr(diagnostics_cfg, 'save_path', None)
  if custom_path:
    return str(custom_path)
  log_path = str(config.sampling.logdir)
  stem, ext = os.path.splitext(log_path)
  if ext:
    return f'{stem}_diagnostics.json'
  return f'{log_path}_diagnostics.json'

def generate_samples(config, logger, tokenizer):
  conditional = bool(getattr(config.data, "conditional_generation", False))
  conditional_metric = str(
    getattr(config.eval, 'conditional_metric', 'rouge')).lower()
  diagnostics_enabled = _diagnostics_enabled(config)
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  if conditional:
    logger.info('Conditional sampling: using dataloader prefixes and generating answer tokens.')
    _, valid_dl = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, valid_seed=config.seed)

    preds = []
    prefixes = []
    gts = []
    diagnostic_records = []

    num_batches = int(getattr(config.sampling, "num_cond_batches", 1))
    num_batches = max(1, num_batches)
    target_num_samples = int(getattr(config.sampling, 'num_eval_samples', 0) or 0)
    if target_num_samples > 0:
      per_batch = max(1, int(config.loader.eval_batch_size))
      num_batches = max(1, (target_num_samples + per_batch - 1) // per_batch)
      logger.info(
        f'Conditional evaluation target: {target_num_samples} samples '
        f'(eval_batch_size={per_batch}, num_batches={num_batches}).')

    model.backbone.eval()
    for b_idx, batch in enumerate(valid_dl):
      if b_idx >= num_batches:
        break
      # Move batch tensors to CPU/GPU handled inside model; keep here light.
      # Generate answer-only text (return_full_sequence=False)
      pred_i = model.restore_model_and_sample_conditional(
        num_steps=config.algo.T,
        batch=batch,
        token_mask_key='attention_mask',
        return_full_sequence=False)
      batch_diagnostics = model.pop_last_sampling_diagnostics() \
        if diagnostics_enabled else []

      if pred_i is None:
        logger.warning(f"[Batch {b_idx}] Sampling returned None, skipping")
        continue

      batch_size = min(len(pred_i), int(batch['input_ids'].shape[0]))
      if target_num_samples > 0:
        remaining = target_num_samples - len(preds)
        if remaining <= 0:
          break
        batch_size = min(batch_size, remaining)
      if batch_size <= 0:
        break

      # Extract prefix text + ground-truth answer from batch for logging
      input_ids = batch['input_ids']
      mask = batch['attention_mask']
      pad_id = tokenizer.pad_token_id

      for i in range(batch_size):
        ids = input_ids[i].tolist()
        m = mask[i].tolist()

        # prefix: up to first answer token (first m==1), ignoring PAD
        try:
          ans_start = m.index(1)
        except ValueError:
          ans_start = len(ids)
        prefix_ids = [t for t in ids[:ans_start] if t != pad_id]
        gt_ids = [t for t, mm in zip(ids, m) if mm == 1 and t != pad_id]

        # cut GT at first EOS if present
        if tokenizer.eos_token_id in gt_ids:
          gt_ids = gt_ids[:gt_ids.index(tokenizer.eos_token_id)]

        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)
        pred_text = pred_i[i]

        prefixes.append(prefix_text)
        gts.append(gt_text)

        if diagnostics_enabled:
          if i < len(batch_diagnostics):
            diag_record = batch_diagnostics[i]
          else:
            diag_record = {}
          diag_record['global_sample_index'] = len(diagnostic_records)
          diag_record['prefix_text'] = prefix_text
          diag_record['gt_text'] = gt_text
          diag_record['pred_text'] = pred_text
          diag_record['seed'] = int(config.seed)
          diagnostic_records.append(diag_record)

      preds.extend(pred_i[:batch_size])
      if target_num_samples > 0 and len(preds) >= target_num_samples:
        break

    logger.info(f'Collected {len(preds)} conditional samples for evaluation.')
    model.metrics.reset()
    if conditional_metric == 'rouge':
      model.metrics.record_rouge_scores(
        predictions=preds,
        references=gts,
      )
      print('Num evaluated samples:', len(preds))
      print('ROUGE-1:', float(model.metrics.gen_rouge1.compute()))
      print('ROUGE-2:', float(model.metrics.gen_rouge2.compute()))
      print('ROUGE-L:', float(model.metrics.gen_rougeL.compute()))
      metric_save_dict = {
        'rouge1': model.metrics.gen_rouge1s,
        'rouge2': model.metrics.gen_rouge2s,
        'rougeL': model.metrics.gen_rougeLs,
      }
    elif conditional_metric == 'ppl':
      model.metrics.record_conditional_perplexity(
          prefixes=prefixes,
          answers=preds,
          max_length=2048,
          device='cuda',
      )
      print('Num evaluated samples:', len(preds))
      print('Generative perplexity:', float(model.metrics.gen_ppl.compute()))
      metric_save_dict = {
        'gen_ppl': model.metrics.gen_ppls,
      }
    else:
      raise ValueError(
        f'Unsupported eval.conditional_metric={conditional_metric!r}. '
        f'Expected one of: rouge, ppl.')
    # Print a few examples
    k = min(3, len(preds))
    for i in range(k):
      print('\n=== Example', i, '===')
      print('PREFIX:\n', prefixes[i])
      print('GT:\n', gts[i])
      print('PRED:\n', preds[i])

    csv_path = config.sampling.logdir
    save_dict = {
      'prefix': [[p] for p in prefixes],
      'gt': [[g] for g in gts],
      'pred': [[p] for p in preds],
      'seed': [config.seed for _ in range(len(preds))],
    }
    save_dict.update(metric_save_dict)

    utils.update_and_save_csv(save_dict, csv_path)

    if diagnostics_enabled:
      for i, record in enumerate(diagnostic_records):
        eval_metrics = {}
        for metric_name, metric_values in metric_save_dict.items():
          if i < len(metric_values):
            eval_metrics[metric_name] = float(metric_values[i])
        record['eval_metrics'] = eval_metrics

      diagnostics_payload = {
        'meta': {
          'seed': int(config.seed),
          'data_valid': str(config.data.valid),
          'checkpoint_path': str(config.eval.checkpoint_path),
          'conditional_metric': conditional_metric,
          'num_samples': int(len(diagnostic_records)),
          'sampling_logdir': str(config.sampling.logdir),
          'snapshot_reveal_fractions': list(
            getattr(config.diagnostics, 'snapshot_reveal_fractions', [])),
          'early_fraction': float(getattr(config.diagnostics, 'early_fraction', 0.3)),
          'structured_inference_enabled': bool(
            getattr(getattr(config.algo, 'structured_inference', None), 'enabled', False)),
        },
        'records': diagnostic_records,
      }
      diagnostics_path = _diagnostics_save_path(config)
      utils.save_json(diagnostics_payload, diagnostics_path)
      logger.info(f'Saved sampling diagnostics to {diagnostics_path}')
    return preds


  text_samples = model.restore_model_and_sample(
    num_steps=config.algo.T)
  print('Text samples:', text_samples)
  print('Generative perplexity:',
        model.metrics.gen_ppl.compute())
  print('Entropy:', model.metrics.gen_entropy.compute())
  csv_path = config.sampling.logdir
  save_dict = {'gen_ppl': model.metrics.gen_ppls,
                'gen_nfes': model.metrics.gen_nfes,
                'gen_entropy': model.metrics.gen_entropies,
                'gen_lengths': model.metrics.gen_lengths,
                'samples': [[i] for i in text_samples],
                'seed': [config.seed for _ in range(len(text_samples))]}
  if config.sampling.var_length:
    print(text_samples)
    save_dict['samples'] = ['' for _ in range(len(text_samples))]
  utils.update_and_save_csv(save_dict, csv_path)
  return text_samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Eval.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  seed = config.seed
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  L.seed_everything(seed)
  config.seed = seed
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=seed)
  trainer.validate(model, valid_ds)

def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
    logger.info(f'Resuming training at {ckpt_path}')
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  # train_ds, valid_ds = dataloader.get_dataloaders(
  #   config, tokenizer)
  # _print_batch(train_ds, valid_ds, tokenizer)

  #-=========disable validation====
  disable_val = bool(config.training.get("disable_validation", False))

  if disable_val:
    train_ds, _ = dataloader.get_dataloaders(config, tokenizer, skip_valid=True)
    valid_ds = None
  else:
    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)

  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.from_pretrained is not None and ckpt_path is None:
    logger.info(f'Loading pretrained model from {config.training.from_pretrained}')
    # load pretraining checkpoint
    if 'kuleshov-group/' in config.training.from_pretrained:
      # load from hf
      model = diffusion.Diffusion(config, tokenizer=tokenizer)
      state_dict = transformers.AutoModelForMaskedLM.from_pretrained(
          config.training.from_pretrained,
          trust_remote_code=True
      ).state_dict()
      model.load_state_dict(state_dict)
    else:
      model = diffusion.Diffusion.load_from_checkpoint(
        config.training.from_pretrained,
        tokenizer=tokenizer,
        config=config,
        strict=False)
    # add buffers for grid search
    model.register_buffer('sampling_eps_min', torch.tensor(
      config.training.sampling_eps_min))
    model.register_buffer('sampling_eps_max', torch.tensor(
      config.training.sampling_eps_max))
  else:
    logger.info(f'Initializing new model')
    model = diffusion.Diffusion(
      config, tokenizer=valid_ds.tokenizer)
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)

  # trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
  #if disable validation=================

  if disable_val:
    trainer.fit(model, train_ds, ckpt_path=ckpt_path)
  else:
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
  
@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    config.wandb = None
    samples = generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    config.wandb = None
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()
