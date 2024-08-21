# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, TypeVar

import torch
from abc import ABC, abstractmethod
from torch.utils.data import TensorDataset

from .equine_output import EquineOutput

# A type variable for Equine objects
AnyEquine = TypeVar("AnyEquine", bound="Equine")


class Equine(torch.nn.Module, ABC):
    """EQUINE Abstract Base Class (ABC):
    EQUINE is set up to extend torch's nn.Module to enrich it with
    a method that enables uncertainty quantification and visualization. Most
    importantly, the `.predict()` method must be outfitted to return an
    EquineOutput object that contains both the class probabilities
    *and* an out-of-distribution (ood) score.

    Parameters
    ----------
    embedding_model : torch.nn.Module
        The embedding model to use.
    head_layers : int, optional
        The number of layers to use in the model head, by default 1.
    device : str, optional
        The device to train the equine model on (defaults to cpu).

    Attributes
    ----------
    embedding_model : torch.nn.Module
        The neural embedding model to enrich with uncertainty quantification.
    head_layers : int
        The number of linear layers to append to the embedding model (default 1, not always used).
    train_summary : dict[str, Any]
        A dictionary containing information about the model training.

    Raises
    ------
    NotImplementedError
        If any of the abstract methods are not implemented.
    """

    def __init__(self, embedding_model: torch.nn.Module, head_layers: int = 1, device: str = "cpu") -> None:
        super().__init__()
        self.embedding_model = embedding_model
        self.head_layers = head_layers
        self.train_summary: dict[str, Any] = {
            "numTrainExamples": 0,
            "dateTrained": "",
            "modelType": "",
        }
        self.device = device
        self.to(device)
        self.embedding_model.to(device)

        self.support = None
        self.support_embeddings = None
        self.prototypes = None

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. This is to preserve the usual behavior
        of torch.nn.Module.

        Parameters
        ----------
        X : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The output of the model.

        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: torch.Tensor) -> EquineOutput:
        """
        Upon implementation, predicts the class logits and out-of-distribution (ood) scores for the
        given input data.

        Parameters
        ----------
        X : torch.Tensor
            The input data.

        Returns
        -------
        EquineOutput
            An EquineOutput object containing the class probabilities and OOD scores.
        """
        raise NotImplementedError

    @abstractmethod
    def train_model(self, dataset: TensorDataset, **kwargs: Any) -> dict[str, Any]:
        """
        Upon implementation, train the model on the given dataset.

        Parameters
        ----------
        dataset : TensorDataset
            TensorDataset containing the training data.
        **kwargs
            Additional keyword arguments to pass to the training function.

        Returns
        -------
        dict[str, Any]
            Dictionary containing summary training information.
        """
        raise NotImplementedError

    @abstractmethod
    def get_prototypes(self) -> torch.Tensor:
        """
        Upon implementation, returns the prototype embeddings

        Returns
        -------
        torch.Tensor
            A torch tensor of the prototype embeddings
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Upon implementation, save the model to the given file path.

        Parameters
        ----------
        path : str
            File path to save the model to.
        """
        raise NotImplementedError

    @classmethod  # type: ignore
    def load(cls: AnyEquine, path: str) -> AnyEquine:  # noqa: F821 # type: ignore
        """
        Upon implementation, load the model from the given file path.

        Parameters
        ----------
        path : str
            File path to load the model from.

        Returns
        -------
        Equine
            Loaded model object.
        """
        raise NotImplementedError
