# Action Experience Model

The implementation follows the paper architecture: interval-aligned history actions are encoded with a learned action dictionary, history latents are compressed into visual memory, the resulting prefix conditions a joint video-action transformer, and a transition predictor supplies the auxiliary visual-motion objective during training.

## Repository structure

```text
<your path>/
├── action.py
├── backbone.py
├── history.py
├── layers.py
├── model.py
├── mot.py
└── scripts/
    ├── data.py
    ├── factory.py
    ├── infer.py
    ├── infer.sh
    ├── train.py
    └── train.sh
```

`action.py` and `backbone.py` define the action and video experts. `history.py` implements interval aggregation, tokenizer lookup, dictionary pooling, temporal positions, and visual-memory extraction. `layers.py` contains history fusion and transition prediction. `model.py` connects both experts, the mixture-of-transformers path, flow matching, and Euler sampling.

## Data format

Training records are tensor dictionaries stored as `.pt` files in one directory. Every record contains:

| Key | Shape | Description |
| --- | --- | --- |
| `video_target` | `[C, T, H, W]` | Clean latent video target |
| `action_target` | `[H, A]` | Clean action chunk |
| `context` | `[L, D]` | Encoded instruction context |
| `context_mask` | `[L]` | Boolean context mask |
| `proprio` | `[P]` | Proprioceptive state |
| `history_action` | `[H_h, A]` | Interval-aligned action history |
| `history_latents` | `[C_h, T_h, H_h, W_h]` | Visual history latents |
| `transition_features` | `[V+1, N, D_z]` | Optional frozen visual features for the transition loss |

Inference records contain the same conditioning fields with a leading batch dimension and add `observation_latents` with shape `[B, C, 1, H, W]`.

## Training

Run the training entry point from the repository root with the record directory, output file, optimization settings, and model arguments.

```bash
bash <your path>/scripts/train.sh \
  --data-dir data/records/train \
  --output runs/action_experience.pt \
  --batch-size <batch-size> \
  --steps <steps> \
  --learning-rate <learning-rate> \
  --device <device>
```

For each batch, the training loop samples Gaussian endpoints, forms $(1-\tau)x+\tau\epsilon$, and optimizes the video and action velocity targets. When `transition_features` is present, a random visual start interval and span are selected and the predicted feature difference is added with `--transition-weight`.

The model arguments are passed through the same entry point when the release is connected to a model factory.

```bash
python -m <your path>.scripts.train --help
```

The checkpoint payload uses `model`, `optimizer`, and `step` fields.

## Inference

Inference loads a checkpoint and one tensor record, initializes the video and action states with Gaussian noise, and integrates the joint flow from one to zero with descending Euler steps.

```bash
bash <your path>/scripts/infer.sh \
  --record data/records/test/sample.pt \
  --checkpoint runs/action_experience.pt \
  --output runs/sample_actions.pt \
  --horizon <action-horizon> \
  --flow-steps <flow-steps> \
  --video-frames <video-frames> \
  --device <device>
```

The output file is a tensor dictionary with the key `actions` and shape `[B, H, A]`. Sampling uses the history prefix and the observed first latent frame; the transition predictor is used only by the training objective.

Inspect inference arguments with:

```bash
python -m <your path>.scripts.infer --help
```

## Python API

The model can also be assembled from the public modules:

```python
from <your path> import AEDWAM

model = AEDWAM(
    video_expert=video_expert,
    action_expert=action_expert,
    history_visual_tokenizer=history_visual_tokenizer,
    history_visual_memory=history_visual_memory,
    history_adapter=history_adapter,
    transition_predictor=transition_predictor,
    proprio_encoder=proprio_encoder,
)

outputs = model(
    video_latents=video_latents,
    noisy_actions=noisy_actions,
    timestep=timestep,
    context=context,
    context_mask=context_mask,
    proprio=proprio,
    history_action=history_action,
    history_latents=history_latents,
)
```

The forward result contains `video_velocity`, `action_velocity`, and `action_hidden`. `AEDWAM.sample` returns the generated action chunk after Euler integration.

## Verification

```bash
python -m py_compile <your path>/*.py <your path>/scripts/*.py
```
