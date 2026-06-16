# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

"""Publication-facing training and cache namespace for the ThinkJEPA release."""

from cache_train.omnijepa import (
    ACTION_MODES,
    MODE_ACT,
    MODE_CHAT,
    MODE_PLAN,
    MODE_QA,
    OmniJepaBridge,
    OmniJepaCacheRecord,
    OmniJepaConfig,
    FlowActionExpert,
    JepaFutureTokenResampler,
    JepaLateFusionAdapter,
    TEXT_MODES,
    VlmToJepaTaskAdapter,
    normalize_omnijepa_mode,
    route_omnijepa_output,
)
from cache_train.omnijepa_data import (
    OmniJepaNpzDataset,
    load_omnijepa_manifest,
    write_omnijepa_cache_npz,
)

__all__ = [
    "ACTION_MODES",
    "MODE_ACT",
    "MODE_CHAT",
    "MODE_PLAN",
    "MODE_QA",
    "OmniJepaBridge",
    "OmniJepaCacheRecord",
    "OmniJepaConfig",
    "FlowActionExpert",
    "JepaFutureTokenResampler",
    "JepaLateFusionAdapter",
    "TEXT_MODES",
    "VlmToJepaTaskAdapter",
    "normalize_omnijepa_mode",
    "route_omnijepa_output",
    "OmniJepaNpzDataset",
    "load_omnijepa_manifest",
    "write_omnijepa_cache_npz",
]
