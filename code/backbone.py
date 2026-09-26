from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional, Sequence

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor]=None, compatibility_mode=True):
    if compatibility_mode:
        q = q.view(q.shape[0], q.shape[1], num_heads, -1).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], num_heads, -1).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], num_heads, -1).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        return x
    else:
        raise NotImplementedError('Only compatibility mode is implemented for flash attention. Please set compatibility_mode=True.')

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

def precompute_freqs_cis_3d(dim: int, end: int=1024, theta: float=10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return (f_freqs_cis, h_freqs_cis, w_freqs_cis)

def precompute_freqs_cis(dim: int, end: int=1024, theta: float=10000.0):
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].double() / dim)
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def rope_apply(x, freqs, num_heads):
    x = x.view(x.shape[0], x.shape[1], num_heads, -1)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == 'npu' else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight

class SelfAttention(nn.Module):

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float=1e-06):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor]=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)

class CrossAttention(nn.Module):

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float=1e-06):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor]=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)

class GateModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float=1e-06):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim ** 0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask: Optional[torch.Tensor]=None):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2), shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2))
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x

class Head(nn.Module):

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x

class VideoDiT(torch.nn.Module):

    def __init__(self, hidden_dim: int, in_dim: int, ffn_dim: int, out_dim: int, text_dim: int, freq_dim: int, eps: float, patch_size: Tuple[int, int, int], num_heads: int, attn_head_dim: int, num_layers: int, video_attention_mask_mode: str):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        if num_heads <= 0:
            raise ValueError(f'`num_heads` must be > 0, got {num_heads}')
        if attn_head_dim <= 0:
            raise ValueError(f'`attn_head_dim` must be > 0, got {attn_head_dim}')
        if attn_head_dim % 2 != 0:
            raise ValueError(f'`attn_head_dim` must be even for RoPE, got {attn_head_dim}')
        self.patch_embedding = nn.Conv3d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.GELU(approximate='tanh'), nn.Linear(hidden_dim, hidden_dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor]=None):
        x = self.patch_embedding(x)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        f, h, w = (int(v) for v in grid_size)
        px, py, pz = self.patch_size
        channels = x.shape[-1] // (px * py * pz)
        x = x.view(x.shape[0], f, h, w, px, py, pz, channels)
        return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            x.shape[0], channels, f * px, h * py, w * pz
        )

    def _validate_forward_inputs(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, context_mask: Optional[torch.Tensor], action: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f'`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}')
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f'`context` must be 3D [B, L, D], got shape {tuple(context.shape)}')
        if timestep.ndim != 1:
            raise ValueError(f'`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}')
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f'`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}')
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(f'`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}')
        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(f'Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}.')
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f'`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}')
        if timestep.shape[0] == 1 and batch_size > 1:
            assert not self.training, 'During training, timestep length must match batch_size.'
            timestep = timestep.expand(batch_size)
        return (x, timestep, context_mask)

    def build_video_to_video_mask(self, video_seq_len: int, video_tokens_per_frame: int, device: torch.device) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f'`video_seq_len` must be positive, got {video_seq_len}')
        if video_tokens_per_frame <= 0:
            raise ValueError(f'`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}')
        if self.video_attention_mask_mode == 'bidirectional':
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        if self.video_attention_mask_mode == 'per_frame_causal':
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(f'`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, got {video_seq_len} and {video_tokens_per_frame}')
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device))
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(video_tokens_per_frame, dim=1)
        if self.video_attention_mask_mode == 'first_frame_causal':
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask
        raise ValueError(f'Unsupported video attention mask mode: {self.video_attention_mask_mode}')

    def pre_dit(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, context_mask: Optional[torch.Tensor]=None, action: Optional[torch.Tensor]=None, control_camera_latents_input: Optional[torch.Tensor]=None) -> Dict[str, Any]:
        x, timestep, context_mask = self._validate_forward_inputs(x=x, timestep=timestep, context=context, context_mask=context_mask, action=action)
        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(f'Latent spatial shape must be divisible by DiT patch size, got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})')
        tokens_per_frame = x.shape[3] // patch_h * (x.shape[4] // patch_w)
        if not hasattr(self, 'patch_size') or len(self.patch_size) < 3:
            raise ValueError(f"Invalid dit.patch_size: {getattr(self, 'patch_size', None)}")
        token_timesteps = torch.ones((batch_size, x.shape[2], tokens_per_frame), dtype=timestep.dtype, device=timestep.device) * timestep.view(batch_size, 1, 1)
        token_timesteps[:, 0, :] = 0
        token_timesteps = token_timesteps.reshape(batch_size, -1)
        token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        f, h, w = x.shape[2:]
        context = self.text_embedding(context)
        context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1)
        x_tokens = x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], f * h * w, x.shape[1]).contiguous()
        freqs = torch.cat([self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1), self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1), self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1).to(x_tokens.device)
        return {'tokens': x_tokens, 'freqs': freqs, 't': t, 't_mod': t_mod, 'context': context, 'context_mask': context_mask, 'meta': {'grid_size': (f, h, w), 'tokens_per_frame': tokens_per_frame, 'batch_size': batch_size}}

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state['meta']['grid_size']
        x = self.head(x_tokens, pre_state['t'])
        x = self.unpatchify(x, (f, h, w))
        return x
