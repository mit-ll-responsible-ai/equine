# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from torch import Tensor


@dataclass
class EquineOutput:
    """
    Output object containing prediction probabilities, OOD scores, and embeddings, which can be used for visualization.

    Attributes
    ----------
    classes : Tensor
        Tensor of predicted class probabilities.
    ood_scores : Tensor
        Tensor of out-of-distribution (OOD) scores.
    embeddings : Tensor
        Tensor of embeddings produced by the model.
    """

    classes: Tensor
    ood_scores: Tensor
    embeddings: Tensor
