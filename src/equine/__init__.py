# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from .equine import Equine, EquineOutput
from .equine_gp import EquineGP
from .equine_protonet import EquineProtonet, CovType

from typing import TYPE_CHECKING

from .utils import (
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    generate_support,
    generate_episode,
    generate_model_metrics,
    generate_train_summary,
    generate_model_summary,
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
    "generate_support",
    "generate_episode",
    "generate_model_metrics",
    "generate_train_summary",
    "generate_model_summary",
]
