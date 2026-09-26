from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional, Sequence
from .backbone import flash_attention, modulate, rope_apply

class MoT(nn.Module):

    def __init__(self, mixtures: Dict[str, nn.Module]):
        super().__init__()
        if not mixtures:
            raise ValueError('`mixtures` cannot be empty.')
        if 'video' not in mixtures or 'action' not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")
        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim
        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(f'All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}')
            if expert.num_heads != self.num_heads:
                raise ValueError(f'All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}')
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(f'All experts must have same attn_head_dim; got {self.attn_head_dim} and {expert.attn_head_dim}')
        for name in self.expert_order:
            expert = self.mixtures[name]

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2), shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2))
        return (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def _mixed_attention(self, q_cat: torch.Tensor, k_cat: torch.Tensor, v_cat: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(block, residual_x: torch.Tensor, mixed_attn_out: torch.Tensor, gate_msa: torch.Tensor, shift_mlp: torch.Tensor, scale_mlp: torch.Tensor, gate_mlp: torch.Tensor, context_payload: Optional[dict]) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
        if context_payload is not None:
            context = context_payload.get('context')
            if context is not None:
                context_mask = context_payload.get('mask')
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(self, expert, block, x: torch.Tensor, freqs: torch.Tensor, t_mod: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)
        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)
        use_gradient_checkpointing = False
        return (q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gradient_checkpointing)

    def forward(self, embeds_all: Dict[str, torch.Tensor], attention_mask: torch.Tensor, freqs_all: Dict[str, torch.Tensor], context_all: Dict[str, Optional[dict]], t_mod_all: Dict[str, torch.Tensor]):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f'Missing expert tokens for {missing}')
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f'Missing expert freqs for {missing}')
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f'Missing expert t_mod for {missing}')
        if attention_mask.ndim != 2:
            raise ValueError(f'`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}')
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f'`attention_mask` must be square, got shape {tuple(attention_mask.shape)}')
        tokens_all = {k: v for k, v in embeds_all.items()}
        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []
            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]
                q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gradient_checkpointing = self._build_expert_attention_io(expert=expert, block=block, x=x, freqs=freqs, t_mod=t_mod)
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {'block': block, 'residual_x': residual_x, 'gate_msa': gate_msa, 'shift_mlp': shift_mlp, 'scale_mlp': scale_mlp, 'gate_mlp': gate_mlp, 'use_gradient_checkpointing': use_gradient_checkpointing}
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)
            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(f'Attention mask seq length mismatch: mask={attention_mask.shape[0]} vs tokens={total_seq}')
            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)
            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert['block']
                context_payload = context_all.get(name)
                updated_tokens = self._apply_post_with_optional_checkpoint(block=block, residual_x=cached_expert['residual_x'], gate_msa=cached_expert['gate_msa'], shift_mlp=cached_expert['shift_mlp'], scale_mlp=cached_expert['scale_mlp'], gate_mlp=cached_expert['gate_mlp'], use_gradient_checkpointing=cached_expert['use_gradient_checkpointing'], mixed_slice=mixed_slice, context_payload=context_payload)
                tokens_all[name] = updated_tokens
                start = end
        return tokens_all

    def _apply_post_with_optional_checkpoint(self, *, block, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gradient_checkpointing, mixed_slice, context_payload):
        return self._apply_expert_post_block(block, residual_x, mixed_slice, gate_msa, shift_mlp, scale_mlp, gate_mlp, context_payload)
