# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from .equine import Equine, EquineOutput
from .equine_gp import EquineGP
from .equine_protonet import EquineProtonet, CovType

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

__version__ = "0.1.1rc3"

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
