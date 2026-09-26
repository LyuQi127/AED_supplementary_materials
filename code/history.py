from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .layers import HistoryActionVideoTransformerBlock, SerialFeaturePredictionBlock


class ActionTokenizer:
    def __init__(self, bpe_tokenizer, *, scale, min_token, vocab_size, max_tokens):
        if scale <= 0 or not np.isfinite(scale) or min(vocab_size, max_tokens) <= 0:
            raise ValueError("Invalid tokenizer dimensions or scale")
        self._bpe_tokenizer = bpe_tokenizer
        self.scale = float(scale)
        self.min_token = int(min_token)
        self.vocab_size = int(vocab_size)
        self.pad_token_id = self.vocab_size
        self.max_tokens = int(max_tokens)
        self._dct_basis_cache = {}

    def _dct_basis(self, steps):
        basis = self._dct_basis_cache.get(steps)
        if basis is not None:
            return basis
        n = np.arange(steps, dtype=np.float64)
        k = np.arange(steps, dtype=np.float64)[:, None]
        basis = np.cos(np.pi / float(steps) * (n + 0.5) * k)
        basis[0] *= np.sqrt(1.0 / float(steps))
        if steps > 1:
            basis[1:] *= np.sqrt(2.0 / float(steps))
        basis = basis.astype(np.float32)
        self._dct_basis_cache[steps] = basis
        return basis

    def encode(self, actions):
        if actions.ndim != 3 or min(actions.shape) <= 0:
            raise ValueError("actions must be nonempty [B,T,D]")
        if not torch.isfinite(actions).all():
            raise ValueError("actions must be finite")
        action_np = actions.detach().to(device="cpu", dtype=torch.float32).numpy()
        coefficients = np.einsum("kt,btd->bkd", self._dct_basis(action_np.shape[1]), action_np)
        coefficients = np.around(coefficients * self.scale)
        rows, masks = [], []
        for element in coefficients:
            text = "".join(map(chr, np.maximum(element.flatten() - self.min_token, 0).astype(int)))
            ids = list(self._bpe_tokenizer(text)["input_ids"])
            if not ids or any(i < 0 or i >= self.vocab_size for i in ids):
                raise ValueError("Tokenizer returned empty or invalid IDs")
            if len(ids) > self.max_tokens:
                raise ValueError("Encoded sequence exceeds max_tokens")
            padding = self.max_tokens - len(ids)
            masks.append([True] * len(ids) + [False] * padding)
            rows.append(ids + [self.pad_token_id] * padding)
        return (
            torch.tensor(rows, dtype=torch.long, device=actions.device),
            torch.tensor(masks, dtype=torch.bool, device=actions.device),
        )


class HistoryActionVideoAdapter(nn.Module):
    def __init__(self, tokenizer, *, action_hidden_dim, attn_head_dim, num_heads,
                 group_size, gripper_indices, max_groups, block_mlp_ratio, eps):
        super().__init__()
        if min(group_size, max_groups) <= 0:
            raise ValueError("group_size and max_groups must be positive")
        self.fast_tokenizer = tokenizer
        self.group_size = int(group_size)
        self.gripper_indices = tuple(gripper_indices)
        self.action_group_max_positions = int(max_groups)
        self.action_token_embedding = nn.Embedding(
            tokenizer.vocab_size + 1, action_hidden_dim, padding_idx=tokenizer.pad_token_id
        )
        self.action_group_pos = nn.Parameter(torch.zeros(1, max_groups, action_hidden_dim))
        nn.init.trunc_normal_(self.action_group_pos, std=0.02)
        self.transformer_block = HistoryActionVideoTransformerBlock(
            hidden_dim=action_hidden_dim, attn_head_dim=attn_head_dim,
            num_heads=num_heads, eps=eps, ffn_mlp_ratio=block_mlp_ratio,
        )

    def aggregate_actions(self, actions):
        if actions.ndim != 3 or min(actions.shape) <= 0 or actions.shape[1] % self.group_size:
            raise ValueError("actions must be [B,H,D] with H divisible by group_size")
        if any(i < 0 or i >= actions.shape[-1] for i in self.gripper_indices):
            raise ValueError("Invalid gripper dimension")
        batch, horizon, dim = actions.shape
        grouped = actions.reshape(batch, horizon // self.group_size, self.group_size, dim)
        aggregated = grouped.sum(dim=2)
        aggregated[:, :, self.gripper_indices] = grouped[:, :, -1, self.gripper_indices]
        return aggregated

    def encode_action_tokens(self, history_action):
        grouped = self.aggregate_actions(history_action)
        batch, groups, dim = grouped.shape
        if groups > self.action_group_max_positions:
            raise ValueError("History exceeds max_groups")
        token_ids, token_mask = self.fast_tokenizer.encode(grouped.reshape(batch * groups, 1, dim))
        token_ids = token_ids.to(self.action_token_embedding.weight.device)
        token_mask = token_mask.to(token_ids.device)
        token_hidden = self.action_token_embedding(token_ids)
        token_hidden = token_hidden * token_mask.unsqueeze(-1).to(token_hidden.dtype)
        denominator = token_mask.sum(dim=1, keepdim=True).to(token_hidden.dtype)
        if not bool((denominator > 0).all()):
            raise ValueError("Every action interval requires at least one valid token")
        grouped_hidden = (token_hidden.sum(dim=1) / denominator).reshape(batch, groups, -1)
        grouped_hidden = grouped_hidden + self.action_group_pos[:, -groups:].to(grouped_hidden)
        return grouped_hidden

    def forward(self, history_action, history_video_tokens):
        query = self.encode_action_tokens(history_action)
        return self.transformer_block(query, history_video_tokens)


class HistoryVisualMemoryExtractor(nn.Module):
    def __init__(self, *, input_dim, hidden_dim, num_queries, num_layers,
                 num_heads, ffn_dim, max_frames, eps):
        super().__init__()
        if min(input_dim, hidden_dim, num_queries, num_layers, num_heads, max_frames) <= 0:
            raise ValueError("Memory dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_frames = int(max_frames)
        self.input_norm = nn.LayerNorm(input_dim, eps=eps)
        self.input_projector = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.memory_queries = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        self.frame_pos = nn.Parameter(torch.zeros(1, max_frames, 1, hidden_dim))
        nn.init.trunc_normal_(self.memory_queries, std=0.02)
        nn.init.trunc_normal_(self.frame_pos, std=0.02)
        self.blocks = nn.ModuleList([
            SerialFeaturePredictionBlock(hidden_dim=hidden_dim, ffn_dim=ffn_dim, eps=eps, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, frame_patch_tokens):
        if frame_patch_tokens.ndim != 4 or min(frame_patch_tokens.shape) <= 0:
            raise ValueError("frame_patch_tokens must be nonempty [B,T,P,D]")
        batch, frames, patches, dim = frame_patch_tokens.shape
        if frames > self.max_frames or dim != self.input_dim:
            raise ValueError("Visual history dimensions exceed the memory interface")
        context = self.input_projector(self.input_norm(frame_patch_tokens))
        context = context + self.frame_pos[:, -frames:].to(context)
        context = context.reshape(batch, frames * patches, self.hidden_dim)
        memory = self.memory_queries.expand(batch, -1, -1).to(context)
        for block in self.blocks:
            memory = block(memory, context)
        return self.output_norm(memory)
