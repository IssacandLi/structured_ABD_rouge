import copy
from typing import List, Optional

import torch


def _to_float(value) -> Optional[float]:
  if value is None:
    return None
  return float(value)


def _trim_token_ids(token_ids, eos_token_id=None, pad_token_id=None):
  trimmed = []
  for token_id in token_ids:
    if token_id is None:
      continue
    token_id = int(token_id)
    if pad_token_id is not None and token_id == pad_token_id:
      continue
    if eos_token_id is not None and token_id == eos_token_id:
      break
    trimmed.append(token_id)
  return trimmed


class SamplingDiagnosticsRecorder:
  """Collects compact per-sample sampling traces for offline diagnostics."""

  def __init__(
    self,
    tokenizer,
    mask_index: int,
    snapshot_reveal_fractions: List[float],
    early_fraction: float,
  ):
    self.tokenizer = tokenizer
    self.mask_index = int(mask_index)
    self.snapshot_reveal_fractions = sorted(
      float(x) for x in snapshot_reveal_fractions)
    self.early_fraction = float(early_fraction)
    self.records = []
    self._states = []
    self.total_sampling_steps = 0

  def _decode_pieces(self, token_ids):
    pieces = []
    for token_id in token_ids:
      if token_id is None:
        pieces.append(None)
      else:
        pieces.append(self.tokenizer.decode(
          [int(token_id)],
          skip_special_tokens=False,
          clean_up_tokenization_spaces=False))
    return pieces

  def _decode_text(self, token_ids):
    if not token_ids:
      return ''
    return self.tokenizer.decode(token_ids, skip_special_tokens=True)

  def start(
    self,
    x0: torch.Tensor,
    x_init: torch.Tensor,
    sample_token_mask: torch.Tensor,
    num_steps: int,
    num_strides: int = 1,
  ):
    x0_cpu = x0.detach().cpu()
    x_init_cpu = x_init.detach().cpu()
    mask_cpu = sample_token_mask.detach().cpu().bool()

    self.total_sampling_steps = max(1, int(num_steps) * max(1, int(num_strides)))
    self.records = []
    self._states = []

    eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
    pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)

    for batch_idx in range(x0_cpu.shape[0]):
      answer_positions = torch.where(mask_cpu[batch_idx])[0]
      answer_pos_list = answer_positions.tolist()
      answer_token_ids_full = [
        int(x0_cpu[batch_idx, pos].item()) for pos in answer_pos_list]
      gt_answer_token_ids = _trim_token_ids(
        answer_token_ids_full,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id)

      answer_start = int(answer_pos_list[0]) if answer_pos_list else 0
      prefix_token_ids = [
        int(token_id) for token_id in x0_cpu[batch_idx, :answer_start].tolist()
        if pad_token_id is None or int(token_id) != pad_token_id]

      record = {
        'batch_index': int(batch_idx),
        'answer_start': answer_start,
        'generation_token_count': int(len(answer_pos_list)),
        'gt_answer_length': int(len(gt_answer_token_ids)),
        'total_sampling_steps': int(self.total_sampling_steps),
        'snapshot_reveal_fractions': list(self.snapshot_reveal_fractions),
        'early_fraction': float(self.early_fraction),
        'prefix_token_ids': prefix_token_ids,
        'prefix_token_text': self._decode_pieces(prefix_token_ids),
        'prefix_text': self._decode_text(prefix_token_ids),
        'gt_answer_token_ids': gt_answer_token_ids,
        'gt_answer_token_text': self._decode_pieces(gt_answer_token_ids),
        'gt_text': self._decode_text(gt_answer_token_ids),
        'snapshots': [],
      }

      prev_revealed = torch.zeros(len(answer_pos_list), dtype=torch.bool)
      if len(answer_pos_list) > 0:
        prev_revealed = (
          x_init_cpu[batch_idx, answer_positions] != self.mask_index)

      state = {
        'answer_positions': answer_positions,
        'prev_revealed': prev_revealed,
        'first_unmask_step': [-1] * len(answer_pos_list),
        'first_unmask_t': [None] * len(answer_pos_list),
        'first_unmask_token_ids': [None] * len(answer_pos_list),
        'captured_targets': set(),
      }
      self.records.append(record)
      self._states.append(state)

  def record_step(self, x_current: torch.Tensor, step_index: int, t_value=None):
    x_cpu = x_current.detach().cpu()

    for batch_idx, (record, state) in enumerate(zip(self.records, self._states)):
      answer_positions = state['answer_positions']
      if answer_positions.numel() == 0:
        continue

      current_answer = torch.full(
        (answer_positions.numel(),),
        fill_value=self.mask_index,
        dtype=x_cpu.dtype)
      available = answer_positions < x_cpu.shape[1]
      if available.any():
        current_answer[available] = x_cpu[
          batch_idx, answer_positions[available]]
      current_revealed = current_answer != self.mask_index
      newly_revealed = current_revealed & (~ state['prev_revealed'])
      if newly_revealed.any():
        newly_indices = torch.where(newly_revealed)[0].tolist()
        for rel_idx in newly_indices:
          state['first_unmask_step'][rel_idx] = int(step_index)
          state['first_unmask_t'][rel_idx] = _to_float(t_value)
          state['first_unmask_token_ids'][rel_idx] = int(
            current_answer[rel_idx].item())

      revealed_fraction = float(current_revealed.float().mean().item())
      revealed_rel_positions = torch.where(current_revealed)[0].tolist()
      revealed_token_ids = [
        int(current_answer[idx].item()) for idx in revealed_rel_positions]

      for target_fraction in self.snapshot_reveal_fractions:
        if target_fraction in state['captured_targets']:
          continue
        if revealed_fraction + 1e-8 < target_fraction:
          continue

        snapshot = {
          'captured': True,
          'target_reveal_fraction': float(target_fraction),
          'step_index': int(step_index),
          'step_fraction': float(step_index / self.total_sampling_steps),
          't': _to_float(t_value),
          'revealed_fraction': revealed_fraction,
          'revealed_positions': revealed_rel_positions,
          'revealed_token_ids': revealed_token_ids,
          'revealed_token_text': self._decode_pieces(revealed_token_ids),
          'revealed_text': self._decode_text(revealed_token_ids),
        }
        record['snapshots'].append(snapshot)
        state['captured_targets'].add(target_fraction)

      state['prev_revealed'] = current_revealed

  def finalize(self, x_final: torch.Tensor):
    x_cpu = x_final.detach().cpu()
    eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
    pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)

    for batch_idx, (record, state) in enumerate(zip(self.records, self._states)):
      answer_positions = state['answer_positions']
      full_answer_ids = []
      if answer_positions.numel() > 0:
        full_answer_ids = [
          int(x_cpu[batch_idx, pos].item()) for pos in answer_positions.tolist()]

      final_answer_token_ids = _trim_token_ids(
        full_answer_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id)
      first_unmask_step = list(state['first_unmask_step'])
      first_unmask_step_fraction = [
        None if step < 0 else float(step / self.total_sampling_steps)
        for step in first_unmask_step]
      first_unmask_token_ids = list(state['first_unmask_token_ids'])
      first_unmask_token_text = self._decode_pieces(first_unmask_token_ids)

      record['final_answer_token_ids'] = final_answer_token_ids
      record['final_answer_token_text'] = self._decode_pieces(final_answer_token_ids)
      record['final_text'] = self._decode_text(final_answer_token_ids)
      record['final_answer_length'] = int(len(final_answer_token_ids))
      record['first_unmask_step'] = first_unmask_step
      record['first_unmask_step_fraction'] = first_unmask_step_fraction
      record['first_unmask_t'] = list(state['first_unmask_t'])
      record['first_unmask_token_ids'] = first_unmask_token_ids
      record['first_unmask_token_text'] = first_unmask_token_text

      for target_fraction in self.snapshot_reveal_fractions:
        if target_fraction in state['captured_targets']:
          continue
        record['snapshots'].append({
          'captured': False,
          'target_reveal_fraction': float(target_fraction),
          'step_index': None,
          'step_fraction': None,
          't': None,
          'revealed_fraction': None,
          'revealed_positions': [],
          'revealed_token_ids': [],
          'revealed_token_text': [],
          'revealed_text': '',
        })

      record['snapshots'].sort(
        key=lambda item: item['target_reveal_fraction'])

  def get_records(self):
    return copy.deepcopy(self.records)
