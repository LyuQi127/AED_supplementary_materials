from __future__ import annotations

import torch
from torch import nn

from .backbone import sinusoidal_embedding_1d
from .mot import MoT


class AEDWAM(nn.Module):
    def __init__(self, *, video_expert, action_expert, history_visual_tokenizer,
                 history_visual_memory, history_adapter, transition_predictor, proprio_encoder):
        super().__init__()
        if tuple(video_expert.patch_size)[0] != 1:
            raise ValueError("The conditioned first frame requires temporal patch size one")
        self.mot = MoT({"video": video_expert, "action": action_expert})
        self.history_visual_tokenizer = history_visual_tokenizer
        self.history_visual_memory = history_visual_memory
        self.history_adapter = history_adapter
        self.serial_feature_predictor = transition_predictor
        self.proprio_encoder = proprio_encoder

    @property
    def video_expert(self):
        return self.mot.mixtures["video"]

    @property
    def action_expert(self):
        return self.mot.mixtures["action"]

    def history_prefix(self, history_action, history_latents):
        tokens = self.history_visual_tokenizer(history_latents)
        pt, ph, pw = self.history_visual_tokenizer.patch_size
        batch, _, time, height, width = history_latents.shape
        tokens = tokens.reshape(batch, time // pt, (height // ph) * (width // pw), -1)
        memory = self.history_visual_memory(tokens)
        return self.history_adapter(history_action, memory)

    def _concat_history_action_pre(self, action_pre, prefix):
        prefix_len = prefix.shape[1]
        action_len = action_pre["tokens"].shape[1]
        total_len = prefix_len + action_len
        if total_len > self.action_expert.freqs.shape[0]:
            raise ValueError("Prefix and action length exceed the rotary position cache")
        zeros = prefix.new_zeros(prefix.shape[0])
        time = self.action_expert.time_embedding(sinusoidal_embedding_1d(self.action_expert.freq_dim, zeros))
        prefix_mod = self.action_expert.time_projection(time).unflatten(1, (6, self.action_expert.hidden_dim))
        prefix_mod = prefix_mod.unsqueeze(1).expand(-1, prefix_len, -1, -1)
        action_mod = action_pre["t_mod"].unsqueeze(1).expand(-1, action_len, -1, -1)
        context_mask = action_pre["context_mask"]
        merged = dict(action_pre)
        merged["tokens"] = torch.cat([prefix, action_pre["tokens"]], dim=1)
        merged["t_mod"] = torch.cat([prefix_mod, action_mod], dim=1)
        merged["freqs"] = self.action_expert.freqs[:total_len].view(total_len, 1, -1).to(prefix.device)
        merged["context_mask"] = torch.cat([context_mask[:, :1].expand(-1, prefix_len, -1), context_mask], dim=1)
        return merged

    def _attention_mask(self, video_len, prefix_len, action_len, tokens_per_frame, device):
        mask = torch.zeros(video_len + prefix_len + action_len, video_len + prefix_len + action_len,
                           dtype=torch.bool, device=device)
        mask[:video_len, :video_len] = self.video_expert.build_video_to_video_mask(
            video_len, tokens_per_frame, device
        )
        first_frame = min(tokens_per_frame, video_len)
        end = video_len + prefix_len
        mask[video_len:end, :first_frame] = True
        mask[video_len:end, video_len:end] = True
        mask[end:, :first_frame] = True
        mask[end:, video_len:] = True
        return mask

    def _forward_with_prefix(self, video_latents, noisy_actions, timestep, context, context_mask, proprio, prefix):
        if context_mask.shape != context.shape[:2] or context_mask.dtype != torch.bool:
            raise ValueError("context_mask must be bool [B,L]")
        if not context_mask.any(dim=1).all():
            raise ValueError("Every sample needs a valid text token")
        if proprio.ndim != 2 or proprio.shape[0] != context.shape[0]:
            raise ValueError("proprio must be [B,D]")
        proprio_token = self.proprio_encoder(proprio.to(context).unsqueeze(1)).to(context)
        context = torch.cat([context, proprio_token], dim=1)
        context_mask = torch.cat([context_mask, context_mask.new_ones(context.shape[0], 1)], dim=1)
        video_pre = self.video_expert.pre_dit(video_latents, timestep, context, context_mask)
        action_pre = self.action_expert.pre_dit(noisy_actions, timestep, context, context_mask)
        action_pre = self._concat_history_action_pre(action_pre, prefix)
        states = {"video": video_pre, "action": action_pre}
        mask = self._attention_mask(video_pre["tokens"].shape[1], prefix.shape[1], noisy_actions.shape[1],
                                    video_pre["meta"]["tokens_per_frame"], video_latents.device)
        hidden = self.mot(
            embeds_all={name: state["tokens"] for name, state in states.items()},
            freqs_all={name: state["freqs"] for name, state in states.items()},
            context_all={name: {"context": state["context"], "mask": state["context_mask"]} for name, state in states.items()},
            t_mod_all={name: state["t_mod"] for name, state in states.items()},
            attention_mask=mask,
        )
        action_hidden = hidden["action"][:, prefix.shape[1]:]
        return {
            "video_velocity": self.video_expert.post_dit(hidden["video"], video_pre),
            "action_velocity": self.action_expert.post_dit(action_hidden, action_pre),
            "action_hidden": action_hidden,
        }

    def forward(self, *, video_latents, noisy_actions, timestep, context, context_mask,
                proprio, history_action, history_latents):
        prefix = self.history_prefix(history_action, history_latents)
        return self._forward_with_prefix(video_latents, noisy_actions, timestep, context, context_mask, proprio, prefix)

    def predict_transition(self, *, start_feature_tokens, action_hidden, start_offset, span, visual_intervals):
        horizon = action_hidden.shape[1]
        if visual_intervals <= 0 or horizon % visual_intervals:
            raise ValueError("Action horizon must be divisible by visual_intervals")
        if start_offset < 1 or span < 1 or start_offset + span > visual_intervals:
            raise ValueError("Transition interval must lie between future observations")
        ratio = horizon // visual_intervals
        start, end = start_offset * ratio, (start_offset + span) * ratio
        positions = torch.arange(start, end, device=action_hidden.device)
        return self.serial_feature_predictor(
            start_feature_tokens=start_feature_tokens,
            action_hidden=action_hidden[:, start:end],
            action_positions=positions,
        )

    @torch.no_grad()
    def sample(self, *, video_noise, action_noise, observation_latents, flow_times, model_timesteps,
               context, context_mask, proprio, history_action, history_latents):
        if self.training:
            raise ValueError("Call eval before sampling")
        if flow_times.ndim != 1 or flow_times.numel() < 2 or not torch.isfinite(flow_times).all():
            raise ValueError("flow_times must be a finite one-dimensional schedule")
        if flow_times[0] != 1 or flow_times[-1] != 0 or not (flow_times[1:] < flow_times[:-1]).all():
            raise ValueError("flow_times must decrease strictly from one to zero")
        if model_timesteps.shape != flow_times[:-1].shape or not torch.isfinite(model_timesteps).all():
            raise ValueError("Provide a finite model timestep for each flow interval")
        video, actions = video_noise.clone(), action_noise.clone()
        if observation_latents.shape != video[:, :, :1].shape:
            raise ValueError("observation_latents must match the first latent frame")
        prefix = self.history_prefix(history_action, history_latents)
        video[:, :, :1] = observation_latents
        for index in range(model_timesteps.numel()):
            timestep = model_timesteps[index].to(video).expand(video.shape[0])
            result = self._forward_with_prefix(video, actions, timestep, context, context_mask, proprio, prefix)
            dt = (flow_times[index + 1] - flow_times[index]).to(video)
            video = video + dt * result["video_velocity"]
            actions = actions + dt.to(actions) * result["action_velocity"]
            video[:, :, :1] = observation_latents
        return actions
