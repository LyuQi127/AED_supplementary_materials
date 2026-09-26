from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class TensorRecordDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.files = sorted(Path(root).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No tensor records found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = torch.load(self.files[index], map_location="cpu", weights_only=True)
        if not isinstance(record, dict):
            raise TypeError("Each record must be a dictionary of tensors")
        required = {
            "video_target",
            "action_target",
            "context",
            "context_mask",
            "proprio",
            "history_action",
            "history_latents",
        }
        missing = sorted(required - set(record))
        if missing:
            raise KeyError(f"Missing record fields: {missing}")
        return record
