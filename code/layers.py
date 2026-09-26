from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional, Sequence
from .backbone import RMSNorm, CrossAttention, flash_attention

class SerialFeaturePredictionBlock(nn.Module):

    def __init__(self, *, hidden_dim: int, ffn_dim: int, eps: float, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f'hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}.')
        head_dim = self.hidden_dim // self.num_heads
        self.norm_self = nn.LayerNorm(self.hidden_dim, eps=float(eps))
        self.norm_cross_q = nn.LayerNorm(self.hidden_dim, eps=float(eps))
        self.norm_cross_kv = nn.LayerNorm(self.hidden_dim, eps=float(eps))
        self.norm_ffn = nn.LayerNorm(self.hidden_dim, eps=float(eps))
        self.self_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.self_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.self_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.self_o = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.cross_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.cross_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.cross_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.cross_o = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(self.hidden_dim, int(ffn_dim)), nn.GELU(approximate='tanh'), nn.Linear(int(ffn_dim), self.hidden_dim))
        self._attn_head_dim = head_dim

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if attn_mask is None:
            attn_mask = torch.ones((q.shape[1], k.shape[1]), device=q.device, dtype=torch.bool)
        else:
            if attn_mask.ndim != 2 or attn_mask.shape != (q.shape[0], k.shape[1]):
                raise ValueError(f'Cross-attention mask must be [B,K], got {tuple(attn_mask.shape)} for q={tuple(q.shape)} and k={tuple(k.shape)}.')
            attn_mask = attn_mask.to(device=q.device, dtype=torch.bool)[:, None, None, :]
        return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

    def forward(self, feature_tokens: torch.Tensor, action_context: torch.Tensor, action_context_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        x_norm = self.norm_self(feature_tokens)
        self_out = self._attn(self.self_q(x_norm), self.self_k(x_norm), self.self_v(x_norm))
        x = feature_tokens + self.self_o(self_out)
        q = self.cross_q(self.norm_cross_q(x))
        kv = self.norm_cross_kv(action_context)
        cross_out = self._attn(q, self.cross_k(kv), self.cross_v(kv), attn_mask=action_context_mask)
        x = x + self.cross_o(cross_out)
        x = x + self.ffn(self.norm_ffn(x))
        return x

class HistoryActionVideoTransformerBlock(nn.Module):

    def __init__(self, *, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float, ffn_mlp_ratio: float) -> None:
        super().__init__()
        ffn_hidden = max(hidden_dim, int(round(hidden_dim * float(ffn_mlp_ratio))))
        self.query_norm = RMSNorm(hidden_dim, eps=eps)
        self.memory_norm = RMSNorm(hidden_dim, eps=eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.ffn_norm = RMSNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_hidden), nn.GELU(approximate='tanh'), nn.Linear(ffn_hidden, hidden_dim))
        self.out_norm = RMSNorm(hidden_dim, eps=eps)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, ctx_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        query = query + self.cross_attn(self.query_norm(query), self.memory_norm(memory), ctx_mask=ctx_mask)
        query = query + self.ffn(self.ffn_norm(query))
        return self.out_norm(query)

class HistoryLatentVisualTokenizer(nn.Module):

    def __init__(self, *, in_dim: int, hidden_dim: int, patch_size: Sequence[int], eps: float) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = tuple((int(x) for x in patch_size))
        if len(self.patch_size) != 3:
            raise ValueError(f'`patch_size` must be a 3-tuple, got {self.patch_size}.')
        self.patch_embedding = nn.Conv3d(self.in_dim, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = RMSNorm(self.hidden_dim, eps=eps)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError(f'`latents` must be [B,C,T,H,W], got {tuple(latents.shape)}')
        if latents.shape[1] != self.in_dim:
            raise ValueError(f'`latents` channel dim must be {self.in_dim}, got {latents.shape[1]}')
        for size, patch, name in zip(latents.shape[2:], self.patch_size, ('T', 'H', 'W')):
            if size % patch != 0:
                raise ValueError(f'History latent {name}={size} must be divisible by patch size {patch}.')
        x = self.patch_embedding(latents)
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return self.norm(tokens)

class MotionTransitionPredictor(nn.Module):

    def __init__(self, *, feature_dim: int, action_dim: int, hidden_dim: int, ffn_dim: int, eps: float, num_heads: int, num_layers: int, max_feature_tokens: int, max_action_tokens: int, action_projector_mlp_ratio: float) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.max_feature_tokens = int(max_feature_tokens)
        self.max_action_tokens = int(max_action_tokens)
        self.feature_in = nn.Linear(self.feature_dim, self.hidden_dim)
        action_projector_hidden = max(self.hidden_dim, int(round(self.action_dim * float(action_projector_mlp_ratio))))
        self.action_projector = nn.Sequential(nn.LayerNorm(self.action_dim, eps=float(eps)), nn.Linear(self.action_dim, action_projector_hidden), nn.GELU(approximate='tanh'), nn.Linear(action_projector_hidden, self.hidden_dim), nn.LayerNorm(self.hidden_dim, eps=float(eps)))
        self.blocks = nn.ModuleList([SerialFeaturePredictionBlock(hidden_dim=self.hidden_dim, ffn_dim=int(ffn_dim), eps=float(eps), num_heads=self.num_heads) for _ in range(self.num_layers)])
        self.output_norm = nn.LayerNorm(self.hidden_dim, eps=float(eps))
        self.head = nn.Linear(self.hidden_dim, self.feature_dim)
        self.feature_pos = nn.Parameter(torch.zeros(1, self.max_feature_tokens, self.hidden_dim))
        nn.init.trunc_normal_(self.feature_pos, std=0.02)
        self.action_pos = nn.Parameter(torch.zeros(1, self.max_action_tokens, self.hidden_dim))
        nn.init.trunc_normal_(self.action_pos, std=0.02)

    def forward(self, *, start_feature_tokens: torch.Tensor, action_hidden: torch.Tensor, action_positions: Optional[torch.Tensor]=None) -> torch.Tensor:
        if start_feature_tokens.ndim != 3:
            raise ValueError(f'`start_feature_tokens` must be [B,N,D], got {tuple(start_feature_tokens.shape)}')
        if action_hidden.ndim != 3:
            raise ValueError(f'`action_hidden` must be [B,S,D], got {tuple(action_hidden.shape)}')
        if start_feature_tokens.shape[2] != self.feature_dim:
            raise ValueError(f'`start_feature_tokens` last dim must be {self.feature_dim}, got {start_feature_tokens.shape[2]}')
        if action_hidden.shape[2] != self.action_dim:
            raise ValueError(f'`action_hidden` last dim must be {self.action_dim}, got {action_hidden.shape[2]}')
        if action_hidden.shape[1] <= 0:
            raise ValueError('Serial DINO feature predictor received no action context tokens.')
        feature_len = int(start_feature_tokens.shape[1])
        action_len = int(action_hidden.shape[1])
        if feature_len > self.max_feature_tokens:
            raise ValueError(f'feature_len={feature_len} exceeds max_feature_tokens={self.max_feature_tokens}.')
        if action_positions is not None:
            if action_positions.ndim != 1 or action_positions.numel() != action_len:
                raise ValueError(f'`action_positions` must be [action_len], got {tuple(action_positions.shape)} for action_len={action_len}.')
            if action_positions.dtype != torch.long:
                raise ValueError(f'action_positions must use torch.long, got {action_positions.dtype}.')
        elif action_len > self.max_action_tokens:
            raise ValueError(f'action_len={action_len} exceeds max_action_tokens={self.max_action_tokens}.')
        x = self.feature_in(start_feature_tokens)
        if self.feature_pos is not None:
            x = x + self.feature_pos[:, :feature_len].to(device=x.device, dtype=x.dtype)
        action_context = self.action_projector(action_hidden)
        if self.action_pos is not None:
            if action_positions is None:
                pos = self.action_pos[:, :action_len]
            else:
                pos = self.action_pos[:, action_positions.to(device=self.action_pos.device)]
            action_context = action_context + pos.to(device=action_context.device, dtype=action_context.dtype)
        for block in self.blocks:
            x = block(x, action_context)
        return self.head(self.output_norm(x))
