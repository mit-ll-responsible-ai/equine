# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations
import torch
from torch.utils.data import TensorDataset  # type: ignore
from abc import ABC, abstractmethod
from typing import Any

from .equine_output import EquineOutput


class Equine(torch.nn.Module, ABC):
    """EQUINE Abstract Base Class (ABC):
    EQUINE is set up to extend torch's nn.Module to enrich it with
    a method that enables uncertainty quantification and visualization. Most
    importantly, the `.predict()` method must be outfitted to return an
    EquineOutput object that contains both the class logits
    *and* an out-of-distribution (ood) score.
    """

    def __init__(self, embedding_model, head_layers=1) -> None:
        super().__init__()
        self.embedding_model = embedding_model
        self.head_layers = head_layers

    @abstractmethod
    def forward(self, X: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: torch.Tensor) -> EquineOutput:
        raise NotImplementedError

    def train_model(self, dataset: TensorDataset, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls: Equine, path: str) -> Equine:  # noqa: F821 # type: ignore
        raise NotImplementedError
