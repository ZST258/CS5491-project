from __future__ import annotations

from pathlib import Path

import torch

from .models import build_model


def load_model_checkpoint(checkpoint_path: str | Path, model_name: str, difficulty: str, device: str = "cpu"):
    payload = torch.load(Path(checkpoint_path), map_location=device)
    metadata = payload.get("metadata", {})
    model_kwargs = metadata.get("model_kwargs", {})
    model = build_model(model_name=model_name, difficulty=difficulty, device=device, **model_kwargs)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata, payload.get("global_step", 0)
