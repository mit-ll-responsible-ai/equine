# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from torch import Tensor


@dataclass
class EquineOutput:
    classes: Tensor
    ood_scores: Tensor
    embeddings: Tensor
