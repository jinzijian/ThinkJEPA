# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

"""OmniJEPA cache dataset utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from cache_train.omnijepa import OmniJepaCacheRecord, normalize_omnijepa_mode


def _decode_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return item
    return value


def _optional_array(npz, key: str):
    return npz[key] if key in npz.files else None


def _to_tensor(value: Any, dtype: torch.dtype | None = None):
    if value is None:
        return None
    tensor = torch.from_numpy(np.asarray(value))
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def write_omnijepa_cache_npz(path: str | Path, **payload: Any) -> Path:
    """Write one OmniJEPA cache sample after schema validation."""

    record = OmniJepaCacheRecord.from_mapping(payload)
    serializable = dict(payload)
    serializable["mode"] = np.asarray(record.mode)
    serializable["episode_id"] = np.asarray(record.episode_id)
    serializable["instruction"] = np.asarray(record.instruction)
    serializable["timestep"] = np.asarray(int(record.timestep), dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    np.savez_compressed(tmp_path, **serializable)
    tmp_npz = tmp_path if tmp_path.suffix == ".npz" else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    tmp_npz.replace(path)
    return path


def load_omnijepa_manifest(path: str | Path) -> list[str]:
    """Load a text manifest or JSON list of OmniJEPA cache files."""

    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"JSON manifest must be a list, got {type(data).__name__}")
        return [str(x) for x in data]
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


class OmniJepaNpzDataset(Dataset):
    """Read unified OmniJEPA .npz cache records.

    Expected keys mirror :class:`OmniJepaCacheRecord`. Tensor fields are returned
    on CPU; the training loop decides device placement.
    """

    def __init__(self, files: Iterable[str | Path], include_obs_frames: bool = False):
        self.files = [str(Path(p)) for p in files]
        if not self.files:
            raise ValueError("OmniJepaNpzDataset requires at least one cache file")
        self.include_obs_frames = bool(include_obs_frames)

    @classmethod
    def from_manifest(cls, manifest: str | Path, include_obs_frames: bool = False):
        return cls(load_omnijepa_manifest(manifest), include_obs_frames=include_obs_frames)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as z:
            mode = normalize_omnijepa_mode(str(_decode_scalar(z["mode"])))
            sample = {
                "path": path,
                "episode_id": str(_decode_scalar(z["episode_id"])),
                "timestep": int(_decode_scalar(z["timestep"])),
                "mode": mode,
                "instruction": str(_decode_scalar(z["instruction"])),
                "current_jepa_latent": _to_tensor(z["current_jepa_latent"], torch.float32),
                "predicted_future_jepa_latent": _to_tensor(
                    z["predicted_future_jepa_latent"], torch.float32
                ),
                "oracle_future_jepa_latent": _to_tensor(
                    z["oracle_future_jepa_latent"], torch.float32
                ),
                "target_action_chunk": _to_tensor(_optional_array(z, "target_action_chunk"), torch.float32),
                "target_traj_tokens": _to_tensor(_optional_array(z, "target_traj_tokens")),
                "target_text": (
                    str(_decode_scalar(z["target_text"])) if "target_text" in z.files else None
                ),
                "future_frames": _to_tensor(_optional_array(z, "future_frames")),
            }
            if self.include_obs_frames:
                sample["obs_frames"] = _to_tensor(z["obs_frames"])
        return sample
