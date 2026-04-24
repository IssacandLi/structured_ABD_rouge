import itertools
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from collections import OrderedDict

import dataloader
import metrics
import models
import noise_schedule
import utils
import numpy as np
import itertools

import math

def _sample_categorical(categorical_probs):
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
  return samples

def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.algo.sampler
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.cross_attn = self.config.algo.cross_attn
    self._cond_stream = None
    self.ignore_bos = self.config.algo.ignore_bos
    self.mdlm_loss_scale = self.config.algo.mdlm_loss_scale
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    if hasattr(self.config, 'algo'):
      self.parameterization = self.config.algo.parameterization
    else:
      self.parameterization = self.config.parameterization
    if hasattr(self.config, 'block_size'):
      self.block_size = self.config.block_size
    else:
      self.block_size = self.config.model.length
    if self.parameterization == 'ar':
      self.block_size = 1

    # --- Mechanism C: Structured Masking ---
    self.structured_masking = getattr(
        getattr(self.config.algo, 'structured_masking', None), 'enabled', False)
    if self.structured_masking:
        self.sm_r_low = self.config.algo.structured_masking.r_low
        self.sm_r_high = self.config.algo.structured_masking.r_high
        self.sm_b_max_tokens = getattr(
            self.config.algo.structured_masking, 'b_max_tokens', 32)
        self.sm_global_t = self.config.algo.structured_masking.global_t
        self.sm_full_bidir = getattr(
            self.config.algo.structured_masking, 'full_bidir_attention', False)
        print("=" * 50)
        print("  [Mechanism C] Structured masking ENABLED (token-level)")
        print(f"  r_low={self.sm_r_low}, r_high={self.sm_r_high}, B_max={self.sm_b_max_tokens}")
        print(f"  full_bidir_attention={self.sm_full_bidir}")
        print("=" * 50)

    if self.config.algo.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
      #  egenerate mask if pretrained model uses flex attention mask
      # and current model uses sdpa mask
      if getattr(self.backbone.config, 'attn_backend', None) == 'flex' and \
        self.config.model.attn_backend == 'sdpa':
        self.backbone.config.attn_backend = 'sdpa'
        for i in self.backbone.backbone.blocks:
          i.attn_backend = 'sdpa'
        self.backbone.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend='sdpa')
    else:
      raise ValueError(f'Unknown backbone: {self.config.algo.backbone}')

    # 如果启用全双向 mask，覆盖默认的 block_diff_mask
    if self.structured_masking and getattr(self, 'sm_full_bidir', False):
        if hasattr(self.backbone, 'gen_mask'):
            self.backbone.gen_mask(
                seqlen=self.config.model.length,
                block_size=self.config.block_size,  # 传但被忽略
                attn_backend=self.config.model.attn_backend,
                full_bidir=True,
            )

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length

    self.noise = noise_schedule.get_noise(self.config)
    self.metrics = metrics.Metrics(config)

    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.var_min = self.config.algo.var_min
    if self.var_min:
      self.register_buffer('sampling_eps_min', torch.tensor(
        self.config.training.sampling_eps_min))
      self.register_buffer('sampling_eps_max', torch.tensor(
        self.config.training.sampling_eps_max))
      
    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
  
    # --- C-inf: Structured Unmasking at Inference ---

    self.structured_inference = getattr(
        getattr(self.config.algo, 'structured_inference', None), 'enabled', False)
    if self.structured_inference:
        self.cinf_aggregation = self.config.algo.structured_inference.aggregation
        self.cinf_commitment = self.config.algo.structured_inference.commitment
        self.cinf_threshold = self.config.algo.structured_inference.threshold
        # 验证 C 的参数也已加载（C-inf 复用 s(t) 公式）
        assert self.structured_masking, \
            "C-inf requires structured_masking to be enabled (need r_low, r_high, b_max_tokens)"
        print("=" * 50)
        print("  [C-inf] Structured inference ENABLED")
        print(f"  aggregation={self.cinf_aggregation}")
        print(f"  commitment={self.cinf_commitment}")
        print(f"  Reusing C params: r_low={self.sm_r_low}, r_high={self.sm_r_high}, B_max={self.sm_b_max_tokens}")
        print("=" * 50)

    # --- Mechanism A: Span-Level Loss ---
    self.span_loss_enabled = getattr(
        getattr(self.config.algo, 'span_loss', None), 'enabled', False)
    if self.span_loss_enabled:
        self.span_loss_lambda = self.config.algo.span_loss.lambda_span
        self.span_loss_type = self.config.algo.span_loss.type
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _get_parameters(self):
    parameters = [self.backbone.parameters(),
                  self.noise.parameters()]
    return itertools.chain(* parameters)

  def on_validation_model_zero_grad(self) -> None:
    '''
    Small hack to avoid first validation on resume. 
    This will NOT work if the gradient accumulation step should be performed at this point.
    '''
    super().on_validation_model_zero_grad()
    if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
        self.trainer.sanity_checking = True
        self._restarting_skip_val_flag = False

  def _validate_configuration(self):
    if self.config.mode == 'sample_eval' and \
        self.config.sampling.first_hitting:
      assert self.config.loader.eval_batch_size == 1
    assert self.config.algo.backbone in {
      'dit', 'ar', 'hf_dit'}
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
    if self.config.sampling.kv_cache:
      assert self.config.algo.name in {'ar', 'bd3lm'}
      
    if self.parameterization in {'sedd'}:
      assert self.time_conditioning
    
    if self.config.mode == 'sample_eval':
      assert self.config.model.attn_backend != 'flex', 'FlexAttention mask not supported at inference.'
    if self.config.model.attn_backend == 'flex':
      assert self.config.algo.name == 'bd3lm', 'Custom FlexAttention mask only supported for BD3LM.'
      
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    if hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'sdpa':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(*args, **kwargs)
    elif hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'flex':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
    if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
      self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
      self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
    return self

  def _replace_ckpt_keys(self, checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
      new_state_dict[k.replace('_orig_mod.', '')] = v
    checkpoint['state_dict'] = new_state_dict
    return checkpoint

  def on_load_checkpoint(self, checkpoint):
    print('Loading checkpoint at', checkpoint['global_step'])
    self._restarting_skip_val_flag = True

    # for models compiled with `torch.compile`
    if '_orig_mod.' in list(checkpoint['state_dict'].keys())[0]:
      checkpoint = self._replace_ckpt_keys(checkpoint)

    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    if 'sampling_eps_min' in checkpoint.keys():
      self.sampling_eps_min = checkpoint['sampling_eps_min']
      self.sampling_eps_max = checkpoint['sampling_eps_max']
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    if hasattr(self, 'sampling_eps_min'):
      checkpoint['sampling_eps_min'] = self.sampling_eps_min
      checkpoint['sampling_eps_max'] = self.sampling_eps_max
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          # persistent_workers=True
          persistent_workers=(self.config.loader.num_workers > 0),
          )
        )
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    # cause of overfitting for block size 1?
    if self.parameterization == 'ar':
      return None
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma, cond=None,sample_mode=False, store_kv=False):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    x = x.long()
    # if self.cross_attn:
    #   assert cond is not None, "cross_attn(two_stream) requires cond"
    #   cond = cond.long()
    #   # 允许 windowed forward：x 可能是 [:, -block_size:]，cond 对齐到同长度
    #   if x.shape[1] != cond.shape[1]:
    #     cond = cond[:, -x.shape[1]:]
    #   x_cat = torch.cat((x, cond), dim=-1)
    # else:
    #   x_cat = x
    use_two_stream = (self.cross_attn and (cond is not None) and (not sample_mode))

    if use_two_stream:
        cond = cond.long()
        if x.shape[1] != cond.shape[1]:
            cond = cond[:, -x.shape[1]:]
        x_cat = torch.cat([x, cond], dim=1)     # [B, 2L]
    else:
        x_cat = x                               # [B, L]


    with torch.amp.autocast('cuda', dtype=torch.float32):
      if self.config.algo.name == 'bd3lm':
        # print(f"[FORWARD] Before backbone, x_cat shape: {x_cat.shape}, batch_size: {x_cat.shape[0]}")
        logits = self.backbone(x_cat, sigma,
                              store_kv=store_kv,
                              sample_mode=sample_mode)

      elif self.config.algo.name == 'ar':
        if self.config.algo.backbone == 'hf_dit':
          logits = self.backbone(x, None)     
        else:
          logits = self.backbone(x, sigma, sample_mode=sample_mode, store_kv=store_kv)
        logits[:, :, self.mask_index] = self.neg_infinity
        logits = logits.log_softmax(-1)
      else:
        logits = self.backbone(x, sigma)

    if self.cross_attn:
      # x = x[:, :self.config.model.length]
      logits = logits[:, :self.config.model.length]
      x = x[:, :self.config.model.length]
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                      xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                        xt=x,
                                        sigma=sigma)
    return logits
    
  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    del batch_idx
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'])
    self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return losses.loss

  def on_validation_epoch_start(self):
    self.metrics.reset()
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.eval()
    self.backbone.eval()
    self.noise.eval()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0
    self.sampling_eps = self.config.training.sampling_eps

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
              on_epoch=True, sync_dist=True)
    if self.ema:
      self.ema.restore(self._get_parameters())
    if self.var_min and not self.trainer.sanity_checking:
      self._clipped_schedule_search()
      self.log('sampling_eps_min',
               self.sampling_eps_min,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
      self.log('sampling_eps_max',
               self.sampling_eps_max,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
  
  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    """Checks if the current sampling interval is valid for reporting likelihood."""
    if (sampling_eps_min == 1e-3 \
        and sampling_eps_max == 1 \
        and not (self.block_size == 1 and self.config.training.eval_nll)):
      return True # elbo
    elif (self.block_size == 1 and sampling_eps_min >= 1):
      return True # nll (block size 1)
    return False # not a valid elbo (biased estimate)
      
  def validation_step(self, batch, batch_idx):
    print(f"[VALIDATION] Step {batch_idx}, batch size: {batch['input_ids'].shape[0]}")
    if self.var_min:
      for noise_clip_start in self.metrics.valid_vars.keys():
        sampling_eps_min, sampling_eps_max = noise_clip_start
        if self._check_val_sampling_intvl(sampling_eps_min, sampling_eps_max) == True:
          # compute and record nelbo
          losses_clip = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=sampling_eps_min,
                            sampling_eps_max=sampling_eps_max)
          losses = Loss(
            nlls=losses_clip.nlls.clone(),
            token_mask=losses_clip.token_mask,
            loss=losses_clip.loss.clone())
        elif len(self.metrics.valid_vars[noise_clip_start]) < 100:
          # elbo from clipped schedule (biased estimate)
          losses_clip = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            sampling_eps_min=sampling_eps_min,
                            sampling_eps_max=sampling_eps_max)
        if len(self.metrics.valid_vars[noise_clip_start]) < 100:
          # only report variance over 100 batches
          nlls = losses_clip.nlls
          self.metrics.valid_vars[noise_clip_start].append(
            nlls.reshape(
              nlls.shape[0], -1, self.block_size).mean(-1))
    elif self.block_size == 1:
      # nll
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1,
                          sampling_eps_max=1)
    else:
      # nelbo
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1e-3,
                          sampling_eps_max=1)
    self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)
    return losses.loss

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]

  def _resample_q_xt(
    self,
    x,
    xt,
    move_indices,
    p,
    block_size,
    sampling_eps_min,
    sampling_eps_max,
    noised_mask=None,
    ):
    """
    Resample with conditional noised_mask support, without infinite loops.
    Enforces bounds on *eligible tokens per block*.
    """
    assert noised_mask is not None, "conditional resample requires noised_mask"

    device = x.device
    B, L = x.shape
    n_blocks = L // block_size

    # reshape
    x_blk = x.view(B, n_blocks, block_size)
    m_blk = noised_mask.to(device=device, dtype=torch.bool).view(B, n_blocks, block_size)

    # eligible count per block
    k = m_blk.sum(-1)  # [B, n_blocks]
    active = k > 0

    # current move_indices in block form
    mi_blk = move_indices.view(B, n_blocks, block_size)

    # ratio bounds -> count bounds per block
    k_f = k.to(torch.float32)
    min_k = torch.ceil(k_f * float(sampling_eps_min)).to(torch.long)
    max_k = torch.floor(k_f * float(sampling_eps_max)).to(torch.long)

    # clamp into [0, k]
    min_k = torch.clamp(min_k, min=0)
    max_k = torch.minimum(max_k, k)

    # infeasible blocks: min_k > max_k
    infeasible = (min_k > max_k) & active
    if infeasible.any():
        p_scalar = p[:, 0] if p.ndim == 2 else p  # [B] or scalar
        target = torch.round(p_scalar[:, None] * k_f).to(torch.long)
        target = torch.clamp(target, min=0)
        target = torch.minimum(target, k)
        min_k = torch.where(infeasible, target, min_k)
        max_k = torch.where(infeasible, target, max_k)

    # decide which blocks need resample (count-based)
    cur_cnt = (mi_blk & m_blk).sum(-1)  # [B, n_blocks]
    regen_blk = ((cur_cnt < min_k) | (cur_cnt > max_k)) & active
    if not regen_blk.any():
        return xt.view(B, n_blocks, block_size)

    # fresh bernoulli sample for eligible positions
    rand = torch.rand((B, L), device=device)
    base = (rand < p) & noised_mask.to(device=device, dtype=torch.bool)
    base_blk = base.view(B, n_blocks, block_size)

    # Only apply to regen blocks; keep original elsewhere
    mi_blk = torch.where(regen_blk[:, :, None], base_blk, mi_blk)

    # one-shot adjustment to satisfy [min_k, max_k]
    cnt = (mi_blk & m_blk).sum(-1)
    need_add = (min_k - cnt).clamp(min=0)
    need_drop = (cnt - max_k).clamp(min=0)

    for b in range(B):
        blk_ids = torch.where(regen_blk[b])[0].tolist()
        for j in blk_ids:
            elig_pos = torch.where(m_blk[b, j])[0]
            if elig_pos.numel() == 0:
                continue

            add = int(need_add[b, j].item())
            if add > 0:
                zeros = elig_pos[~mi_blk[b, j, elig_pos]]
                if zeros.numel() > 0:
                    pick = zeros[torch.randperm(zeros.numel(), device=device)[:add]]
                    mi_blk[b, j, pick] = True

            drop = int(need_drop[b, j].item())
            if drop > 0:
                ones = elig_pos[mi_blk[b, j, elig_pos]]
                if ones.numel() > 0:
                    pick = ones[torch.randperm(ones.numel(), device=device)[:drop]]
                    mi_blk[b, j, pick] = False

    move_indices[:] = mi_blk.view(B, L)
    xt = torch.where(move_indices, self.mask_index, x)
    return xt.view(B, n_blocks, block_size)


  def q_xt(
    self,
    x,
    p,
    block_size=None,
    sampling_eps_min=None,
    sampling_eps_max=None,
    noised_mask=None,
    ):
    if block_size is None:
        block_size = self.block_size

    B, L = x.shape
    assert noised_mask is not None, "conditional q_xt requires noised_mask"
    noised_mask = noised_mask.to(device=x.device, dtype=torch.bool)



    # move_indices = (torch.rand((B, L), device=x.device) <= p) & noised_mask
    # xt = torch.where(move_indices, self.mask_index, x)
    # --- Mechanism C: 结构化 masking ---
    if self.structured_masking:
        xt = self._structured_mask(x, p, block_size, noised_mask)
    else:
        # --- 原始逻辑：per-token 随机 mask ---
        move_indices = (torch.rand((B, L), device=x.device) <= p) & noised_mask
        xt = torch.where(move_indices, self.mask_index, x)


    if block_size == 1 and sampling_eps_min == 1.0:
        return torch.full_like(x, self.mask_index)

    # if self.config.training.resample and not (sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
    #     xt_blk = xt.view(B, -1, block_size)
    #     xt_blk = self._resample_q_xt(
    #         x=x,
    #         xt=xt_blk,
    #         move_indices=move_indices,
    #         p=p,
    #         block_size=block_size,
    #         sampling_eps_min=sampling_eps_min,
    #         sampling_eps_max=sampling_eps_max,
    #         noised_mask=noised_mask,
    #     )
    #     xt = xt_blk.view(B, L)
    if (not self.structured_masking) and \
       self.config.training.resample and \
       not (sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
        xt_blk = xt.view(B, -1, block_size)
        xt_blk = self._resample_q_xt(
            x=x, xt=xt_blk, move_indices=(xt == self.mask_index),
            p=p, block_size=block_size,
            sampling_eps_min=sampling_eps_min,
            sampling_eps_max=sampling_eps_max,
            noised_mask=noised_mask,
        )
        xt = xt_blk.view(B, L)

    return xt

  def _structured_mask(self, x, p, block_size, noised_mask):
    """
    Mechanism C:
    - span 按 token 切分，与 block_size 完全无关
    - span_size = 1 + s(t) * (B_max - 1)，随噪声线性增长
    - 用 span-level + token-level residual mixture 对齐目标 masking ratio
    """
    B, L = x.shape
    device = x.device

    # r_t = 当前 move_chance（noise schedule 给出）
    if p.dim() == 2:
        r_t = p[:, 0]  # [B]
    else:
        r_t = p.expand(B) if p.dim() == 0 else p

    s_t = torch.clamp(
        (r_t - self.sm_r_low) / (self.sm_r_high - self.sm_r_low + 1e-8),
        min=0.0, max=1.0
    )

    B_max = int(self.sm_b_max_tokens)
    xt = x.clone()

    for b in range(B):
        s = s_t[b].item()
        r = r_t[b].item()

        # answer 区域的 token 位置
        ans_token_ids = torch.where(noised_mask[b])[0]
        n_ans = len(ans_token_ids)
        if n_ans == 0:
            continue

        # noise-dependent span size
        span_size = max(1, int(round(1 + s * (B_max - 1))))
        n_spans = max(1, n_ans // span_size)

        # 让一部分 token 以整 span 方式被 mask，剩余质量由 token 级 residual 补齐，
        # 从而保证 E[masked ratio] = r。
        p_span = max(0.0, min(1.0, r * s))
        if p_span >= 1.0:
            p_token = 0.0
        else:
            p_token = max(0.0, min(1.0, (r - p_span) / max(1.0 - p_span, 1e-8)))
        span_mask_decisions = (torch.rand(n_spans, device=device) < p_span)

        for sp_idx in range(n_spans):
            sp_start = sp_idx * span_size
            sp_end = min(sp_start + span_size, n_ans) if sp_idx < n_spans - 1 else n_ans
            span_token_ids = ans_token_ids[sp_start:sp_end]

            if span_mask_decisions[sp_idx]:
                xt[b, span_token_ids] = self.mask_index
                continue

            token_mask = torch.rand(len(span_token_ids), device=device) < p_token
            to_mask = span_token_ids[token_mask]
            if len(to_mask) > 0:
                xt[b, to_mask] = self.mask_index

    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)

  @torch.no_grad()
  def _nucleus_sample(self, p_x0):
    p = self.config.sampling.nucleus_p
    if p == 1.0:
      return p_x0
    p_x0_ = p_x0[:, -self.block_size:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
    p_x0_ /= p_x0_.sum(-1, keepdim=True)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0

  # @torch.no_grad()
  # def _ddpm_caching_update(self, x, t, dt, p_x0=None, cond=None):
  #   _, move_chance_t = self.noise(t)
  #   _, move_chance_s = self.noise(t - dt)
  #   sigma_t = self._sigma_from_p(move_chance_t)
  #   move_chance_t = move_chance_t[:, None]
  #   move_chance_s = move_chance_s[:, None]
  #   mask_prob = move_chance_s / move_chance_t

  #   if p_x0 is None:
  #     if self.config.sampling.kv_cache:
  #       p_x0 = self.forward(x[:, -self.block_size:],
  #                       sigma_t,
  #                       cond=cond,
  #                       sample_mode=True).to(torch.float64)
  #     else:   
  #       p_x0 = self.forward(x,
  #                         sigma_t,cond=cond,
  #                         sample_mode=True).to(torch.float64)
  #       p_x0 = p_x0[:, -self.block_size:]
  #     p_x0 = p_x0.exp()
  #     p_x0 = self._nucleus_sample(p_x0)

  #   if self.config.sampling.first_hitting:
  #     x_block = _sample_categorical(p_x0)
  #     # randomly and uniformly select an index in the block (among masked tokens)
  #     num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
  #     ind = torch.randint(0, num_masked, (x_block.shape[0],))
  #     ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
  #     mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
  #     x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
  #   else:
  #     q_xs = p_x0 * (1 - mask_prob)
  #     q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
  #     x_block = _sample_categorical(q_xs)
  #   copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
  #   x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
  #   x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

  #   # compute kv cache if all tokens in a block are sampled
  #   if self.config.sampling.kv_cache and self.mask_index not in x_block:
  #     _ = self.forward(x_block, sigma_t, cond=cond,sample_mode=True, store_kv=True)

  #   if not torch.allclose(x_new, x):
  #     return None, x_new
  #   else:
  #     return p_x0, x_new
  @torch.no_grad()
  def _ddpm_caching_update(self, x, t, dt, p_x0=None, cond=None,
                           cinf_token_mask=None, cinf_num_steps=None):
      _, move_chance_t = self.noise(t)
      _, move_chance_s = self.noise(t - dt)
      sigma_t = self._sigma_from_p(move_chance_t)
      move_chance_t = move_chance_t[:, None]
      move_chance_s = move_chance_s[:, None]
      mask_prob = move_chance_s / move_chance_t

      use_two_stream_sampling = (
          cinf_token_mask is not None
          and cond is not None
          and not self.config.sampling.kv_cache
          and x.shape[1] == self.config.model.length
      )

      if p_x0 is None:
          if use_two_stream_sampling:
              p_x0 = self.forward(
                  x,
                  sigma_t,
                  cond=cond,
                  sample_mode=False).to(torch.float64)
              p_x0 = p_x0[:, -self.block_size:]
          elif self.config.sampling.kv_cache:
              p_x0 = self.forward(x[:, -self.block_size:],
                                  sigma_t, cond=cond, sample_mode=True).to(torch.float64)
          else:
              p_x0 = self.forward(x, sigma_t, cond=cond, sample_mode=True).to(torch.float64)
              p_x0 = p_x0[:, -self.block_size:]
          p_x0 = p_x0.exp()
          p_x0 = self._nucleus_sample(p_x0)

      if self.config.sampling.first_hitting:
          # C-inf 路径外层已 assert，不会进这里
          x_block = _sample_categorical(p_x0)
          num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
          ind = torch.randint(0, num_masked, (x_block.shape[0],))
          ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
          mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
          x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
      else:
          q_xs = p_x0 * (1 - mask_prob)
          q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)

          # ====== C-inf 分支：用 span-correlated sample 替代独立采样 ======
          if cinf_token_mask is not None:
              x_block = self._span_correlated_sample(
                  probs=q_xs,
                  x=x[:, -self.block_size:],
                  t=t,
                  token_mask=cinf_token_mask,
                  num_steps=cinf_num_steps,
              )
          else:
              x_block = _sample_categorical(q_xs)
          # =================================================================

      copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
      x_block = copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
      x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

      if self.config.sampling.kv_cache and self.mask_index not in x_block:
          _ = self.forward(x_block, sigma_t, cond=cond, sample_mode=True, store_kv=True)

      if not torch.allclose(x_new, x):
          return None, x_new
      else:
          return p_x0, x_new


  @torch.no_grad()
  def _ar_sampler(self, bsz, context_len=1024):
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache()

    with torch.amp.autocast('cuda', dtype=torch.float32):
      # precompute token buffer
      num_pred_tokens = self.num_tokens - 1
      x = torch.zeros(
        (bsz, num_pred_tokens + 1),
        dtype=torch.long,
        device=self.device)
      x[:, 0] = self.tokenizer.bos_token_id
      stop = False
      for i in tqdm(range(num_pred_tokens)):
        # need to sample a gumbel for each token
        # to save memory in variable-length sampling
        noise = (torch.distributions.Gumbel(0, 1)
                .sample((bsz, self.vocab_size))
                .to(self.device))
        next_logits = self.forward(
          x[:, :i + 1][:, -context_len:],
          None,
          store_kv=self.config.sampling.kv_cache)[:, -1:].to(torch.float64)
    
        next_logits = next_logits.exp()
        next_logits = self._nucleus_sample(next_logits).log()
        y = (next_logits[:, -1] + noise).argmax(-1)
        # check if we need to resample (or stop sampling for variable-length sampling)
        if (i+1) > 256:
          stop, x_out = self._check_stop_conds(x[:, :i+1])
          if stop:
            x = x_out
        if (stop and not self.config.sampling.var_length) \
          or (stop and x.shape[-1] == 1):
          return None
        elif stop:
          break
        x[:, i + 1] = y
      return x
  
  @torch.no_grad()
  def _sample(
    self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
    """Generate samples from the model."""
    if seqlen is None:
      seqlen = self.config.model.length
    if batch_size_per_gpu is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    samples = []
    if self.parameterization == 'ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i = self._ar_sampler(batch_size_per_gpu)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.gen_nfes.append(self.config.model.length)
      samples = torch.cat(samples, dim=0) 
      return self.tokenizer.batch_decode(samples)
    if self.sampler == 'semi_ar':
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          num_tries += 1
          sample_i, nfes = self._semi_ar_sampler(
            n_samples=batch_size_per_gpu,
            num_strides=(seqlen // self.block_size), 
            num_steps=num_steps,
            seqlen=seqlen)
          if num_tries > 10:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.nfes.update(nfes)
        self.metrics.gen_nfes.append(nfes)
    else:
      nfes = num_steps
      for _ in range(self.config.sampling.num_sample_batches):
        sample_i, num_tries = None, 0
        while sample_i is None:
          sample_i = self._analytic_sampler(
            n_samples=batch_size_per_gpu,
            num_steps=num_steps,
            seqlen=seqlen,
            eps=eps)
          num_tries += 1
          if num_tries > 10 and sample_i is None:
            raise ValueError('Sampling failed.')
        samples.append(sample_i)
        self.metrics.nfes.update(nfes)
        self.metrics.gen_nfes.append(nfes)
    samples = torch.cat(samples, dim=0) 
    return self.tokenizer.batch_decode(samples)

  def _sigma_from_p(self, p):
    return torch.min(- torch.log(1 - p), self.noise.sigma_max)
  @torch.no_grad()
  def restore_model_and_sample_conditional(
    self,
    batch,
    num_steps,
    eps=1e-5,
    seqlen=None,
    token_mask_key=None,
    return_full_sequence=False,
  ):
    # === C-inf 入口：对齐训练时的 full_bidir + block_size=seqlen ===
    if self.structured_inference:
        assert getattr(self, 'sm_full_bidir', False), \
            "C-inf 推理要求训练时 full_bidir_attention=True"
        inf_seqlen = self.config.model.length
        # 重生成全双向 mask（和训练一致）
        self.backbone.gen_mask(
            seqlen=inf_seqlen,
            block_size=inf_seqlen,   # 传任意值都可，full_bidir=True 时被忽略
            attn_backend=self.config.model.attn_backend,
            full_bidir=True,
        )
        # mask 搬到 device
        if hasattr(self.backbone, 'block_diff_mask'):
            dev = next(self.backbone.parameters()).device
            if torch.is_tensor(self.backbone.block_diff_mask):
                self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(dev)
        # block_size 扩到 seqlen，让 semi_ar 单 stride 并行
        self._orig_block_size = self.block_size
        self.block_size = inf_seqlen
        if hasattr(self.backbone, 'block_size'):
            self.backbone.block_size = inf_seqlen
    # kv_cache 的切片逻辑对 block_size=seqlen 会越界，必须关
    assert not self.config.sampling.kv_cache, \
        "C-inf 必须设置 sampling.kv_cache=False"
    # first_hitting 会强制只 unmask 1 个 token，与 span gating 冲突
    assert not self.config.sampling.first_hitting, \
        "C-inf 必须设置 sampling.first_hitting=False"

    if self.ema:
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()

    x0 = batch['input_ids'].to(self.device)
    if seqlen is None:
      seqlen = x0.shape[1]
    x0 = x0[:, :seqlen]

    # choose answer mask: 1 on answer positions
    token_mask = None
    if token_mask_key is not None and token_mask_key in batch:
      token_mask = batch[token_mask_key]
    elif 'token_mask' in batch:
      token_mask = batch['token_mask']
    elif 'attention_mask' in batch:
      token_mask = batch['attention_mask']
    if token_mask is not None:
      token_mask = token_mask.to(self.device)[:, :seqlen].bool()

    for b in range(x0.shape[0]):
    # 找到第一个 answer 位置（token_mask[b] 中第一个 True 的位置）
      mask_b = token_mask[b].cpu()
      if mask_b.any():
          ans_start = int(torch.where(mask_b)[0][0].item())
          prefix_length = ans_start
          answer_length = int(mask_b.sum().item())
      else:
          prefix_length = seqlen
          answer_length = 0
      
      print(f"[Batch {b}] Prefix length: {prefix_length}, Answer length: {answer_length}")
      print(f"[Batch {b}] Token mask (first 50): {mask_b[:50].tolist()}")
      print(f"[Batch {b}] Token mask (last 50): {mask_b[-50:].tolist()}")
      print(f"[Batch {b}] Total sequence length: {seqlen}")
      print(f"[Batch {b}] Prefix ratio: {prefix_length/seqlen:.2%}, Answer ratio: {answer_length/seqlen:.2%}")
      print("-" * 80)
    gen_mask = None
    if token_mask is not None:
        gen_mask = torch.zeros_like(token_mask, dtype=torch.bool)

        for b in range(x0.shape[0]):
            if token_mask[b].any():
                ans_start = int(torch.where(token_mask[b])[0][0].item())
            else:
                # fallback: if no answer mask, generate from position 1
                ans_start = 1

            # choose a max generation length (prefer config; fallback 256)
            gen_len = 256
            if hasattr(self.config, "data") and hasattr(self.config.data, "answer_max_tokens"):
                gen_len = int(self.config.data.answer_max_tokens)

            end = min(seqlen, ans_start + gen_len)
            gen_mask[b, ans_start:end] = True

    sample_token_mask = gen_mask if gen_mask is not None else token_mask

    x_init = x0.clone()
    if sample_token_mask is not None:
        x_init[sample_token_mask] = self.mask_index
    else:
        x_init[:, 1:] = self.mask_index

    # fixed cond stream (two-stream)
    cond = None
    if self.cross_attn:
      cond = x0.clone()
      if sample_token_mask is not None:
        cond[sample_token_mask] = self.mask_index
      else:
        cond[:, 1:] = self.mask_index

    try:
      # sample
      if self.structured_inference:
          # C-inf: 用 semi_ar，单 stride 覆盖整个序列，内部嵌入 span gating
          print("[C-inf] Using semi_ar with block_size=seqlen + span gating")
          x_out, _ = self._semi_ar_sampler(
              n_samples=x_init.shape[0],
              num_steps=num_steps,
              num_strides=1,                # 关键：只跑 1 个 stride
              seqlen=seqlen,
              context_size=seqlen,
              x_init=x_init,
              cond=cond,
              eps=eps,
              cinf_token_mask=sample_token_mask,
              cinf_num_steps=num_steps,
          )
      elif self.sampler == 'semi_ar':
          x_out, _ = self._semi_ar_sampler(
              n_samples=x_init.shape[0],
              num_steps=num_steps,
              num_strides=(seqlen // self.block_size),
              seqlen=seqlen,
              context_size=self.config.sampling.context_size,
              x_init=x_init,
              cond=None,
          )
      else:
        x_out = self._analytic_sampler(
          n_samples=x_init.shape[0],
          num_steps=num_steps,
          seqlen=seqlen,
          eps=eps,
          x_init=x_init,
          cond=cond,
        )

      if self.ema:
        self.ema.restore(self._get_parameters())

      if x_out is None:
        print("[DEBUG] x_out is None - sampling failed due to stop conditions")
        print("[DEBUG] Retrying without stop conditions is not implemented; returning empty strings")
        return [""] * batch['input_ids'].shape[0]
      print(f"[DEBUG] x_out shape: {x_out.shape}")
      print(f"[DEBUG] x_out sample (first 50): {x_out[0, :50].tolist()}")
      print(f"[DEBUG] x_out sample (last 50): {x_out[0, -50:].tolist()}")
      print(f"[DEBUG] token_mask shape: {sample_token_mask.shape if sample_token_mask is not None else None}")
      if sample_token_mask is not None:
          print(f"[DEBUG] token_mask has any True: {sample_token_mask[0].any().item()}")
          if sample_token_mask[0].any():
              ans_pos = int(torch.where(sample_token_mask[0])[0][0].item())
              print(f"[DEBUG] ans_pos: {ans_pos}")
              print(f"[DEBUG] answer region tokens: {x_out[0, ans_pos:ans_pos+20].tolist()}")

      if return_full_sequence:
        return self.tokenizer.batch_decode(x_out)

      # return only answer part (cut from first answer position, stop at EOS)
      if token_mask is None:
        return self.tokenizer.batch_decode(x_out)

      out_texts = []
      for b in range(x_out.shape[0]):
        ans_pos = int(torch.where(token_mask[b])[0][0].item()) if token_mask[b].any() else 0
        seq = x_out[b, ans_pos:]
        eos = torch.where(seq == self.tokenizer.eos_token_id)[0]
        if len(eos) > 0:
          seq = seq[: int(eos[0].item()) + 1]
        out_texts.append(self.tokenizer.decode(seq, skip_special_tokens=True))
      return out_texts
    finally:
      if self.structured_inference and hasattr(self, '_orig_block_size'):
        self.block_size = self._orig_block_size
        if hasattr(self.backbone, 'block_size'):
          self.backbone.block_size = self._orig_block_size


  def restore_model_and_sample(self, num_steps, eps=1e-5, seqlen=None):
    """Generate samples from the model."""
    if self.ema:  
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(
      seqlen=seqlen,
      batch_size_per_gpu=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps)
    self.metrics.record_generative_perplexity(
      samples,
      self.config.model.length,
      self.config.loader.eval_batch_size,
      self.device)
    return samples

  def get_score(self, x, sigma, cond=None):
    model_output = self.forward(x, sigma, cond=cond).to(torch.float64)
    if self.config.sampling.nucleus_p == 1.0:
      return model_output.exp()
    model_output = model_output - model_output.logsumexp(-1, keepdim=True)
    model_output = self._nucleus_sample(model_output.exp())
    return model_output

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, dt, cond=None):
    sigma_t = self._sigma_from_p(self.noise(t)[1])
    sigma_s = self._sigma_from_p(self.noise(t - dt)[1])
    dsigma = sigma_t - sigma_s
    score = self.get_score(x, sigma_t, cond=cond)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)


  def _denoiser_update(self, x, t, cond=None):
    sigma = self._sigma_from_p(self.noise(t)[1])
    score = self.get_score(x, sigma, cond=cond)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples


  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(
      self, batch_dims, device, sampling_eps_min, sampling_eps_max, block_size=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size


    # _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

    # # antithetic sampling along blocks & batches (for uniform sampling)
    # if self.antithetic_sampling:
    #   offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
    #   offset_b = offset_b.view(batch_dims[0], num_blocks)
    #   _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1

    # --- Mechanism C: 当启用结构化 masking 时，answer 内所有 block 共享同一个 t ---
    if self.structured_masking and getattr(self, 'sm_global_t', False):
        _eps_b = torch.rand((batch_dims[0], 1), device=device)
        if self.antithetic_sampling:
            offset_b = torch.arange(batch_dims[0], device=device).float() / batch_dims[0]
            _eps_b = (_eps_b.squeeze(-1) / batch_dims[0] + offset_b) % 1
            _eps_b = _eps_b.unsqueeze(-1)
        _eps_b = _eps_b.expand(batch_dims[0], num_blocks)  # 所有 block 共享同一个 t
    else:
        # --- 原始逻辑：per-block 独立采样 ---
        _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)
        if self.antithetic_sampling:
            offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
            offset_b = offset_b.view(batch_dims[0], num_blocks)
            _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1

    t = _eps_b
    if block_size != self.config.model.length:
      t = t.repeat_interleave(block_size, dim=-1)

    # nll
    if sampling_eps_max >= 1 and sampling_eps_min >= 1:
      return torch.ones_like(t)
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.num_tokens:
      assert seqlen == 2 * self.num_tokens
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.num_tokens)
      end = start + self.num_tokens
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation ppl, since the val
      # examples will all start and end with BOS/EOS
      if self.config.data.insert_train_special == True:
        input_tokens[:, 0] = self.tokenizer.bos_token_id
        output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    
    return input_tokens, output_tokens, new_attention_mask
  def _build_cond_stream(self, x0, token_mask):
    """
    Build condition stream for two-stream mode.
    Keep prefix tokens, wipe answer tokens to prevent leakage.
    token_mask: 1 on answer region (highlights), 0 elsewhere.
    """
    if token_mask is None:
      return x0  # unconditional

    cond = x0.clone()
    # wipe answer (highlights) tokens in cond stream to prevent leakage
    cond[token_mask.bool()] = self.mask_index
    return cond

  def _compute_span_bow_loss(self, model_output, x0, xt, token_mask, block_size):
    """
    Mechanism A: Bag-of-Words span-level loss.
    
    对于被 mask 的 span，将 span 内的 logits 平均池化，
    与真实 span 的词袋分布计算 KL 散度。
    
    只在 answer 区域的 masked span 上计算。
    """
    B, L, V = model_output.shape
    device = model_output.device
    n_blocks = L // block_size
    
    if token_mask is None:
        return torch.tensor(0.0, device=device)
    
    token_mask_bool = token_mask.bool()
    # 与 _structured_mask 保持一致：用 token-level B_max，按 token 切 span
    B_max = int(getattr(self, 'sm_b_max_tokens', 32))
    # 这里 BoW loss 仍按 block 计数（与本函数现有 block-based 实现一致），
    # 把 B_max（token 数）换算成 block 数
    span_blocks = max(1, B_max // block_size)
    
    total_kl = torch.tensor(0.0, device=device)
    n_spans_counted = 0
    
    # 按 span（多个 block）为单位计算 BoW loss
    noised_blk = token_mask_bool.view(B, n_blocks, block_size)
    answer_blk_mask = (noised_blk.sum(-1) > 0)  # [B, n_blocks]
    
    for b in range(B):
        ans_blk_ids = torch.where(answer_blk_mask[b])[0]
        if len(ans_blk_ids) == 0:
            continue
        
        n_ans_blocks = len(ans_blk_ids)
        n_spans = max(1, n_ans_blocks // span_blocks)
        actual_span_size = n_ans_blocks // n_spans
        
        for sp_idx in range(n_spans):
            sp_start_blk = sp_idx * actual_span_size
            sp_end_blk = sp_start_blk + actual_span_size if sp_idx < n_spans - 1 else n_ans_blocks
            span_blk_ids = ans_blk_ids[sp_start_blk:sp_end_blk]
            
            # 收集 span 内的所有 token 位置
            span_positions = []
            for blk_id in span_blk_ids:
                start = blk_id * block_size
                end = start + block_size
                for pos in range(start, min(end, L)):
                    if token_mask_bool[b, pos]:
                        span_positions.append(pos)
            
            if len(span_positions) == 0:
                continue
            
            span_pos = torch.tensor(span_positions, device=device)
            
            # 检查这个 span 是否有被 mask 的 token
            is_masked = (xt[b, span_pos] == self.mask_index)
            if not is_masked.any():
                continue  # span 完全可见，不需要 BoW loss
            
            # 预测词袋：pool span 内 masked 位置的 logits
            masked_pos = span_pos[is_masked]
            span_logits = model_output[b, masked_pos, :]  # [n_masked, V]
            pred_bow = F.softmax(span_logits.mean(dim=0), dim=-1)  # [V]
            
            # 真实词袋：统计 span 内的 token 分布
            true_tokens = x0[b, span_pos]  # [n_span_tokens]
            true_bow = torch.zeros(V, device=device)
            true_bow.scatter_add_(0, true_tokens, torch.ones_like(true_tokens, dtype=torch.float))
            true_bow = true_bow / true_bow.sum().clamp(min=1.0)
            
            # KL(true || pred)
            # 避免 log(0)
            kl = F.kl_div(
                (pred_bow + 1e-10).log(),
                true_bow,
                reduction='sum'
            )
            total_kl += kl
            n_spans_counted += 1
    
    if n_spans_counted > 0:
        # 返回形状与 token_loss 兼容的标量
        avg_kl = total_kl / n_spans_counted
        # 广播到 [B, L] 形状（乘以 token_mask 后只在 answer 区域有值）
        span_loss_map = avg_kl * token_mask.float() / token_mask.float().sum().clamp(min=1.0)
        return span_loss_map
    else:
        return torch.zeros_like(x0, dtype=torch.float)

  def _forward_pass_diffusion(self, x0, t=None, sampling_eps_min=None, sampling_eps_max=None,
    token_mask=None):
    if t is None:
      t = self._sample_t(x0.shape,
                         x0.device,
                         sampling_eps_min,
                         sampling_eps_max)

    loss_scale, p = self.noise(t)
    sigma = self._sigma_from_p(p[:,0].unsqueeze(-1))
    dsigma = - loss_scale * torch.expm1(sigma) # used for sedd

    # below is needed to reproduce mdlm/sedd numbers with models from sahoo et al
    # (numerical imprecision computing probs under loglinear schedule)
    if self.mdlm_loss_scale:
      sigma, dsigma = self.noise.total_noise(t), self.noise.rate_noise(t)
      p = 1 - torch.exp(-sigma)
      loss_scale = - (dsigma / torch.expm1(sigma))

    xt = self.q_xt(x0,
                   p,
                   sampling_eps_min=sampling_eps_min,
                   sampling_eps_max=sampling_eps_max,
                   noised_mask=token_mask,)
    if sampling_eps_min is not None and sampling_eps_min > 0.5:
      loss_scale = - torch.ones_like(loss_scale)
    if self.ignore_bos:
      xt[:, 0] = x0[:, 0]
    
    x_input = xt
    if self.cross_attn:
      cond = self._build_cond_stream(x0, token_mask)
      
      model_output = self.forward(xt, sigma=sigma, cond=cond)

      # x_input = torch.cat((xt, x0), dim=-1)
    else:
      model_output = self.forward(xt, sigma=sigma)

    # model_output = self.forward(x_input, sigma=sigma)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma * self._score_entropy(
        model_output, sigma, xt, x0)

    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    loss = loss_scale * log_p_theta

    # --- Mechanism A: Span-Level BoW Loss ---
    if self.span_loss_enabled and self.training:
        # 计算 s(t) 来决定 span loss 的权重
        if p.dim() == 2:
            r_t_mean = p[:, 0]  # [B]
        else:
            r_t_mean = p.mean()
        
        s_t = torch.clamp(
            (r_t_mean - self.sm_r_low) / (self.sm_r_high - self.sm_r_low + 1e-8),
            min=0.0, max=1.0
        ) if self.structured_masking else r_t_mean  # 如果没有 C，用 r_t 作为权重
        
        alpha_t = s_t.mean()  # scalar，span loss 权重
        
        if alpha_t > 0.01:
            span_loss = self._compute_span_bow_loss(
                model_output, x0, xt, token_mask, block_size)
            combined_loss = token_loss + self.span_loss_lambda * alpha_t * span_loss
            return combined_loss

    return loss

  def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = self.sampling_eps_min
      sampling_eps_max = self.sampling_eps_max
    elif not hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)
    if self.parameterization == 'ar':
      output = self.forward(input_tokens, None)
      loss = - output.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      # print(f"[_LOSS] Before _forward_pass_diffusion, input_tokens shape: {input_tokens.shape}, token_mask shape: {attention_mask.shape}")
      loss = self._forward_pass_diffusion(
        input_tokens,
        sampling_eps_min=sampling_eps_min,
        sampling_eps_max=sampling_eps_max,
        token_mask=attention_mask)
      # print("done loss")
    
    if self.ignore_bos and not self.training:
      attention_mask[:, 0] = 0
      
    nlls = (loss * attention_mask)
    token_nll = nlls.sum() / attention_mask.sum()
    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _clipped_schedule_search(self):
    # collect losses per batch across devices and sum them per interval
    best_var = float('inf')
    for (eps_min, eps_max), var in self.metrics.valid_vars.items():
      all_vars = torch.tensor(0., device=self.device)
      for i in range(len(var)):
        agg_var = var[i].to(self.device)
        agg_var = self.all_gather(agg_var)
        all_vars += agg_var.var()
      if all_vars < best_var:
        best_var = all_vars
        sampling_eps_min_best = eps_min
        sampling_eps_max_best = eps_max
      self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
                all_vars / len(var),
                on_epoch=True,
                on_step=False,
                sync_dist=True)
    if self.config.algo.fix_clipping == False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  def _apply_structured_unmasking(self, x, t, token_mask):
    """
    C-inf: 结构化 unmasking 后处理（修复版）

    核心思路：将零散的 unmask 集中到高置信度的 span 中，
    而不是无差别地撤回所有低占比 span 的 unmask。

    流程：
    1. 计算每个 span 的 unmask 比例（作为置信度代理）
    2. 按置信度排序
    3. 保护排名靠前的 span（不撤回），撤回排名靠后的 span
    4. 撤回力度由 s(t) 控制：高噪声时更激进，低噪声时不干预

    效果：unmask token 从分散变为集中，实现 span-level commitment，
    同时不阻止模型积累有效 unmask。
    """
    B, L = x.shape
    device = x.device
    block_size = self.block_size
    n_blocks = L // block_size

    r_t = t.squeeze(-1)  # [B]

    s_t = torch.clamp(
        (r_t - self.sm_r_low) / (self.sm_r_high - self.sm_r_low + 1e-8),
        min=0.0, max=1.0
    )  # [B]

    # s(t) 太小时不干预
    if s_t.max().item() < 0.05:
        return x

    B_max = int(getattr(self, 'sm_b_max_tokens', 32))
    span_blocks = max(1, B_max // max(1, block_size))
    token_mask_bool = token_mask.bool()
    x_out = x.clone()

    for b in range(B):
        s = s_t[b].item()
        if s < 0.05:
            continue

        # 找到 answer 区域的 block
        noised_blk = token_mask_bool[b].view(n_blocks, block_size)
        answer_blk_mask = (noised_blk.sum(-1) > 0)
        ans_blk_ids = torch.where(answer_blk_mask)[0]
        if len(ans_blk_ids) == 0:
            continue

        n_ans_blocks = len(ans_blk_ids)
        n_spans = max(1, n_ans_blocks // span_blocks)
        actual_span_size = n_ans_blocks // n_spans

        # ---------- 收集每个 span 的信息 ----------
        span_data = []
        for sp_idx in range(n_spans):
            sp_start = sp_idx * actual_span_size
            sp_end = (sp_start + actual_span_size) if sp_idx < n_spans - 1 else n_ans_blocks
            span_blk_ids = ans_blk_ids[sp_start:sp_end]

            positions = []
            for blk_id in span_blk_ids:
                st = blk_id.item() * block_size
                for pos in range(st, st + block_size):
                    if token_mask_bool[b, pos]:
                        positions.append(pos)

            if not positions:
                continue

            pos_t = torch.tensor(positions, device=device)
            is_unmasked = (x[b, pos_t] != self.mask_index)
            ratio = is_unmasked.float().mean().item()
            span_data.append({
                'positions': pos_t,
                'is_unmasked': is_unmasked,
                'ratio': ratio,
            })

        # 不足 2 个 span 时无法做比较排序，跳过
        if len(span_data) < 2:
            continue

        # ---------- 按 unmask 比例降序排列（置信度高的在前） ----------
        span_data.sort(key=lambda d: d['ratio'], reverse=True)

        # ---------- 决定保护多少 span ----------
        # s=1（高噪声）: 保护 60% 的 span，撤回 40%
        # s=0.5:          保护 80%
        # s→0:            保护 100%（不干预）
        protect_frac = 1.0 - 0.4 * s
        n_protect = max(1, int(len(span_data) * protect_frac))

        # ---------- 对未保护的 span 执行软撤回 ----------
        for rank, sd in enumerate(span_data):
            if rank < n_protect:
                # 高置信度 span：保留不动
                continue

            # 低置信度 span：以概率 (s * 0.5) 撤回其 unmask token
            # 乘 0.5 是为了避免过度激进
            unmasked_positions = sd['positions'][sd['is_unmasked']]
            if len(unmasked_positions) == 0:
                continue

            revert_prob = s * 0.5
            revert_mask = torch.rand(len(unmasked_positions), device=device) < revert_prob
            x_out[b, unmasked_positions[revert_mask]] = self.mask_index

    return x_out

  def _aggregate_span_confidence(self, confidences):
    if confidences.numel() == 0:
      return 0.0

    mode = getattr(self, 'cinf_aggregation', 'mean')
    if mode == 'mean':
      return confidences.mean().item()
    if mode == 'max':
      return confidences.max().item()
    if mode == 'top_k_mean':
      k = max(1, math.ceil(confidences.numel() / 2))
      return confidences.topk(k).values.mean().item()
    raise ValueError(f'Unknown C-inf aggregation mode: {mode}')

  def _adaptive_confidence_threshold(self, confidences, s):
    mean = confidences.mean()
    std = confidences.std(unbiased=False)
    return mean + (0.25 * s) * std

  def _select_spans_for_reveal(self, valid_spans, reveal_budget, s):
    if len(valid_spans) == 0:
      return set()

    threshold_mode = getattr(self, 'cinf_threshold', 'fixed_ratio')
    selected = set()

    if threshold_mode == 'fixed_ratio':
      revealed = 0
      for rank, (_, span_data) in enumerate(valid_spans):
        if rank == 0 or revealed < reveal_budget:
          selected.add(rank)
          revealed += span_data['pos_tensor'].numel()
      return selected

    if threshold_mode == 'adaptive':
      conf_tensor = torch.tensor(
          [span_data['confidence'] for _, span_data in valid_spans],
          device=self.device)
      threshold = self._adaptive_confidence_threshold(conf_tensor, s)
      for rank, (_, span_data) in enumerate(valid_spans):
        if span_data['confidence'] >= threshold.item():
          selected.add(rank)
      if not selected:
        selected.add(0)
      return selected

    raise ValueError(f'Unknown C-inf threshold mode: {threshold_mode}')

  def _span_correlated_sample(self, probs, x, t, token_mask, num_steps):
    """
    C-inf: token 级 span-correlated sampling（文档 §3.4 Step 5 对齐）。

    关键点:
    - span 切分完全独立于 block_size，按 token 数切
    - span_size = 1 + s(t) * (B_max - 1)，与训练时 _structured_mask 一致
    - r_t 用 noise(t) 算真实 move_chance（不是 timestep）
    - aggregation=mean, commitment=mixed（未选中 span 保持 MASK）, threshold=fixed_ratio

    Args:
        probs:      [B, L, V]  q_xs from _ddpm_caching_update（含 mask_index 通道）
        x:          [B, L]      current token sequence
        t:          [B, 1]      current timestep
        token_mask: [B, L]      True = answer region, False = prefix
        num_steps:  int         总采样步数（用于 reveal 预算）
    """
    B, L, V = probs.shape
    device = probs.device

    # 独立采样（作为候选值）
    samples = _sample_categorical(probs)

    # === 单调性保护：已 unmask 的 answer token 不能被重采样 ===
    token_mask_bool = token_mask.bool()
    already_unmasked = (x != self.mask_index) & token_mask_bool
    samples = torch.where(already_unmasked, x, samples)

    # === 计算 s(t)：用 noise schedule 算真实 masking ratio ===
    _, move_chance = self.noise(t)
    if move_chance.dim() > 1:
        r_t = move_chance.squeeze(-1)
    else:
        r_t = move_chance

    s_t = torch.clamp(
        (r_t - self.sm_r_low) / (self.sm_r_high - self.sm_r_low + 1e-8),
        min=0.0, max=1.0
    )

    # 极低 s(t)：退化为独立采样（单调性已钉住）
    if s_t.max().item() < 0.05:
        return samples

    B_max = int(getattr(self, 'sm_b_max_tokens', 32))

    for b in range(B):
        s = s_t[b].item()
        if s < 0.05:
            continue

        # === Token 级切 span（与 block_size 完全无关）===
        ans_token_ids = torch.where(token_mask_bool[b])[0]
        n_ans = len(ans_token_ids)
        if n_ans == 0:
            continue

        # 与训练 _structured_mask 一致的 span_size 公式
        span_size = max(1, int(round(1 + s * (B_max - 1))))
        n_spans = max(1, n_ans // span_size)

        # === 收集每个 span 的 masked 位置 + 聚合置信度 ===
        span_info = []
        for sp_idx in range(n_spans):
            sp_start = sp_idx * span_size
            sp_end = min(sp_start + span_size, n_ans) if sp_idx < n_spans - 1 else n_ans
            span_token_ids = ans_token_ids[sp_start:sp_end]

            # 只看这个 span 里还 masked 的位置
            still_masked = (x[b, span_token_ids] == self.mask_index)
            if not still_masked.any():
                span_info.append(None)
                continue
            pos_tensor = span_token_ids[still_masked]

            token_conf = 1.0 - probs[b, pos_tensor, self.mask_index]
            span_conf = self._aggregate_span_confidence(token_conf)
            span_info.append({'pos_tensor': pos_tensor, 'confidence': span_conf})

        valid_spans = [(i, si) for i, si in enumerate(span_info) if si is not None]
        if len(valid_spans) == 0:
            continue
        valid_spans.sort(key=lambda kv: kv[1]['confidence'], reverse=True)

        # === 每步 reveal 预算（token 级，噪声依赖）===
        per_step_budget = max(1, n_ans // max(1, num_steps))
        step_multiplier = 1.5 - s    # s=1(高噪) → 0.5x reveal 少；s=0(低噪) → 1.5x reveal 多
        reveal_budget = max(span_size, int(per_step_budget * step_multiplier))
        selected_ranks = self._select_spans_for_reveal(valid_spans, reveal_budget, s)

        commitment_mode = getattr(self, 'cinf_commitment', 'mixed')
        token_only_samples = None
        if commitment_mode == 'hard':
            token_only_probs = probs[b].clone()
            token_only_probs[:, self.mask_index] = 0
            token_norm = token_only_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            token_only_probs = token_only_probs / token_norm
            token_only_samples = _sample_categorical(token_only_probs)

        # === Gating：只让选中的 span 有机会 reveal，其余本步保持 MASK ===
        for rank, (sp_idx, si) in enumerate(valid_spans):
            if rank not in selected_ranks:
                samples[b, si['pos_tensor']] = self.mask_index
                continue

            if commitment_mode == 'hard':
                samples[b, si['pos_tensor']] = token_only_samples[si['pos_tensor']]
            elif commitment_mode != 'mixed':
                raise ValueError(f'Unknown C-inf commitment mode: {commitment_mode}')

    return samples
    
  @torch.no_grad
  def _analytic_sampler(
    self, n_samples, num_steps, seqlen, eps=1e-5, x_init=None, cond=None,token_mask=None): 
    # x = self._sample_prior(
    #   n_samples,
    #   seqlen).to(self.device)
    if x_init is None:
        x = self._sample_prior(n_samples, seqlen).to(self.device)
    else:
        x = x_init.to(self.device)

    x[:, 0] = self.tokenizer.bos_token_id
    timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)   # ← 这行必须在这里
    dt = (1 - eps) / num_steps

    # 保存 x_init 用于钉住 prefix（放在 for 循环之前）
    x_init_device = x_init.to(self.device) if x_init is not None else None


    for i in tqdm(range(num_steps), desc='step'):
      t = timesteps[i] * torch.ones(
          x.shape[0], 1, device=self.device)

      # --- C-inf: 用 span-correlated sampling 替代标准采样 + 后处理撤回 ---
      if self.structured_inference and token_mask is not None:
          # 分两步：先算概率，再 span 相关采样
          sigma_t = self._sigma_from_p(self.noise(t)[1])
          sigma_s = self._sigma_from_p(self.noise(t - dt)[1])
          dsigma = sigma_t - sigma_s
          score = self.get_score(x, sigma_t, cond=cond)
          stag_score = self._staggered_score(score, dsigma)
          probs = stag_score * self._transp_transition(x, dsigma)
          x = self._span_correlated_sample(probs, x, t, token_mask, num_steps)
      else:
          x = self._analytic_update(x=x, t=t, dt=dt, cond=cond)


    #     x = torch.where(token_mask, x, x_init_device)
    # denoising step
    t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)

    # ===== P0 修复：denoiser 前记录已 unmask 的 answer 位置 =====
    if token_mask is not None:
        answer_already_unmasked = (x != self.mask_index) & token_mask.bool()
        x_before_denoiser = x.clone()

    x = self._denoiser_update(x=x, t=t, cond=cond)

    # ===== P0 修复：denoiser 之后，answer 区已 unmask 的钉回原值 =====
    if token_mask is not None:
        x = torch.where(answer_already_unmasked, x_before_denoiser, x)

    # ===== 钉住 prefix =====
    if x_init_device is not None and token_mask is not None:
        x = torch.where(token_mask, x, x_init_device)

    # ===== 修改：conditional 生成跳过 entropy stop =====
    # conditional 生成的输出可能合理地低熵（如摘要任务），
    # 不应因此被判定为退化
    if token_mask is not None:
        # conditional 模式：不做 stop 检查，直接返回
        return x
    else:
        stop, x = self._check_stop_conds(x)
        if stop:
            return None
        return x

    

  @torch.no_grad()
  def _semi_ar_sampler(
      self,
      n_samples,
      num_steps,
      num_strides,
      seqlen,
      context_size=512,
      cond_stream=None,
      # --- 为了兼容 restore_model_and_sample_conditional 的调用 ---
      x_init=None,
      eps=None,      # semi_ar 不用 eps，但必须接住关键字参数
      cond=None,     # 兼容旧调用：传 cond=... 时映射到 cond_stream
      cinf_token_mask=None,   # 新增
      cinf_num_steps=None,
  ):
      """
      Semi-autoregressive / stride-wise sampler.

      支持 conditional x_init：
        - x_init: 形状 [B, seqlen]，prefix 填真实 token，answer 位置填 mask_index
        - sampler 会逐 stride 生成，每次只更新当前 stride 的最后 block_size token
          (非 mask token 由 copy_flag 保留)

      cond_stream:
        - 若 self.cross_attn=True，则 cond_stream 是对齐到 seqlen 的“条件流”（token ids）
        - 若传入 cond=...，这里会自动映射到 cond_stream
      """
      # 兼容 cond=... 的关键字
      if cond_stream is None and cond is not None:
          cond_stream = cond

      if seqlen is None:
          seqlen = self.config.model.length

      # 保证 stride 数覆盖整个 seqlen（更稳）
      need_strides = (seqlen + self.block_size - 1) // self.block_size
      if num_strides is None:
          num_strides = need_strides
      else:
          num_strides = min(num_strides, need_strides)  # 你也可以改成 max，看你希望生成更长还是严格 seqlen

      sampling_steps = 0

      # reset kvs
      if self.config.sampling.kv_cache:
          self.backbone.reset_kv_cache(eval_batch_size=self.config.loader.eval_batch_size)

      ones = torch.ones((n_samples, 1), dtype=self.dtype, device=self.device)

      # ---------- stride loop ----------
      x_accum = None
      for stride_num in tqdm(range(num_strides)):
          end_idx = (stride_num + 1) * self.block_size

          # 1) 追加下一个 block：来自 x_init（条件生成）或 prior（无条件）
          if stride_num == 0:
              if x_init is None:
                  x_accum = self._sample_prior(n_samples, self.block_size).to(self.device)
              else:
                  # 只取当前已生成长度（第一个 block）
                  x_accum = x_init[:, :end_idx].clone().to(self.device)

              x_accum[:, 0] = self.tokenizer.bos_token_id
          else:
              if x_init is None:
                  x = self._sample_prior(n_samples, self.block_size).to(self.device)
              else:
                  # 取 x_init 的“下一块”（prefix 位置是定值，answer 位置是 mask_index）
                  x = x_init[:, end_idx - self.block_size:end_idx].clone().to(self.device)

              x_accum = torch.cat((x_accum, x), dim=1)

          # 2) sliding window：模型 forward 的上下文不超过 context_size
          start_idx = max(end_idx - context_size, 0)
          fwd_idx = torch.arange(start_idx, end_idx, device=self.device)

          dt = 1.0 / float(num_steps)
          p_x0_cache = None

          # 注意：原版是 1 -> 0；为了完全对齐原行为，这里保留 1->0
          timesteps = torch.linspace(1.0, 0.0, num_steps, device=self.device)
          t = 1.0

          for i in range(num_steps):
              # 只要“当前 stride 的最后 block_size”已经没有 mask，就可以提前结束这个 stride
              x_win = x_accum[:, fwd_idx]
              if not (x_win[:, -self.block_size:] == self.mask_index).any():
                  break

              # first-hitting / 普通时序
              if getattr(self.config.sampling, "first_hitting", False):
                  u = float(np.random.rand())
                  num_masked = (x_win[:, -self.block_size:] == self.mask_index).sum(-1).item()
                  # num_masked=0 时上面已 break；这里安全
                  t *= u ** (1.0 / float(num_masked))
              else:
                  t = float(timesteps[i].item())

              cond_win = None
              if cond_stream is not None:
                  cond_win = cond_stream[:, fwd_idx]
                  cond_win = cond_win.to(self.device)

              # --- C-inf: 按 fwd_idx 切 token_mask ---
              cinf_mask_win = None
              if cinf_token_mask is not None:
                  cinf_mask_win = cinf_token_mask[:, fwd_idx].to(self.device)

              p_x0_cache, x_next = self._ddpm_caching_update(
                  x=x_win,
                  t=(t * ones),
                  dt=dt,
                  p_x0=p_x0_cache,
                  cond=cond_win,
                  cinf_token_mask=cinf_mask_win,      # 新增
                  cinf_num_steps=cinf_num_steps,       # 新增
              )

              if p_x0_cache is None:
                  sampling_steps += 1

              x_accum[:, fwd_idx] = x_next

          # 3) 可选：variable-length / stop 条件（保留原逻辑，但修掉 x 未定义的问题）
          # if x_accum.shape[1] > 256:
          #     stop, x_accum = self._check_stop_conds(x_accum)
          #     if stop and (not getattr(self.config.sampling, "var_length", False)):
          #         return None, None
          #     elif stop:
          #         break
          if x_accum.shape[1] > 256 and x_init is None:
            stop, x_accum = self._check_stop_conds(x_accum)
            if stop and (not getattr(self.config.sampling, "var_length", False)):
                return None, None
            elif stop:
                break

      return x_accum, sampling_steps

  
  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum()
    return entropy
  
  def _check_stop_conds(self, x):
    """Check if sampling should stop based on 1) eos, 2) entropy, or 3) likelihood.
    Entropy/likelihood evaluated on last 256 token-block.
    
    Args:
      x: torch.Tensor, current sample.
    Returns:
      stop: bool, whether to stop sampling.
      x: torch.Tensor, sample (potentially truncated for variable-length sampling).

    """
    """Check if sampling should stop..."""
    # ========== 在这里添加打印 ==========
    # print(f"[DEBUG _check_stop_conds] x shape: {x.shape}")
    # print(f"[DEBUG _check_stop_conds] x sample (first 50): {x[0, :50].tolist()}")
    # print(f"[DEBUG _check_stop_conds] x sample (last 50): {x[0, -50:].tolist()}")
    # print(f"[DEBUG _check_stop_conds] mask_index count in x: {(x == self.mask_index).sum().item()}")
    # ========== 打印结束 ==========


    stop = False # stop sampling?
    truncate_idx = None # truncate sample? (variable-length sampling only)

    # CRITERION 2: always stop sampling if entropy is low
    tail = x[:, -256:]

    # ✅关键：末尾还存在 mask，说明还没生成完，不要用 entropy 判停
    if (tail == self.mask_index).any():
      print(f"[DEBUG _check_stop_conds] Tail still has mask_index, returning False")
      return False, x
    # entropy = self._compute_entropy(x[:, -256:])
    entropy = self._compute_entropy(tail)
    print(f"[DEBUG _check_stop_conds] Entropy: {entropy.item():.4f}, threshold: 4.0")

    if entropy < 4:
      stop = True
      print(f"[DEBUG _check_stop_conds] ⚠️ STOP=True due to low entropy: {entropy.item():.4f} < 4.0")
      print(f"[DEBUG _check_stop_conds] Tail tokens (last 50): {tail[0, -50:].tolist()}")


    # for variable length sampling, check if we should stop
    # sampling, and where to truncate the sample
    if self.config.sampling.var_length:
      # CRITERION 1: stop at sampled EOS token
      if len(torch.where(x == self.tokenizer.eos_token_id)[0]) > 1:
        stop = True
        eos_idx = torch.where(x == self.tokenizer.eos_token_id)
        if len(eos_idx[0]) > 1:
          truncate_idx = min(eos_idx[1][1]+1, x.shape[1])

      # CRITERION 2: stop if entropy/likelihood is low
      if entropy < 4:
        stop = True
        truncate_idx = x.shape[1] - 256

    # truncate sample (variable-length sampling only)
    if truncate_idx is not None:
      x = x[:, :truncate_idx]
      if x.ndim == 1:
        x = x.unsqueeze(0)

    return stop, x
