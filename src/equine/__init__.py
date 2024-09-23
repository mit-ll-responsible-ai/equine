# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING

from .equine import Equine, EquineOutput
from .equine_gp import EquineGP
from .equine_protonet import CovType, EquineProtonet
from .load_equine_model import load_equine_model
from .utils import (
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    generate_episode,
    generate_model_metrics,
    generate_model_summary,
    generate_support,
    generate_train_summary,
    mahalanobis_distance_nosq,
)

if not TYPE_CHECKING:  # pragma: no cover
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "unknown version"
else:  # pragma: no cover
    __version__: str

__all__ = [
    "Equine",
    "EquineOutput",
    "EquineGP",
    "EquineProtonet",
    "CovType",
    "brier_score",
    "brier_skill_score",
    "expected_calibration_error",
    "load_equine_model",
    "generate_support",
    "generate_episode",
    "generate_model_metrics",
    "generate_train_summary",
    "generate_model_summary",
    "mahalanobis_distance_nosq",
]
