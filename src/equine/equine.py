# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, Optional, TypeVar

import icontract
import torch
from abc import ABC, abstractmethod
from collections import OrderedDict
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
    feature_names : list[str], optional
        List of strings of the names of the tabular features (ex ["duration", "fiat_mean", ...])
    label_names : list[str], optional
        List of strings of the names of the labels (ex ["streaming", "voip", ...])

    Attributes
    ----------
    device : str
        The device to train the equine model on (defaults to cpu).
    embedding_model : torch.nn.Module
        The neural embedding model to enrich with uncertainty quantification.
    feature_names : list[str], optional
        List of strings of the names of the tabular features (ex ["duration", "fiat_mean", ...])
    head_layers : int
        The number of linear layers to append to the embedding model (default 1, not always used).
    label_names : list[str], optional
        List of strings of the names of the labels (ex ["streaming", "voip", ...])
    train_summary : dict[str, Any]
        A dictionary containing information about the model training.

    Raises
    ------
    NotImplementedError
        If any of the abstract methods are not implemented.
    """

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        head_layers: int = 1,
        device: str = "cpu",
        feature_names: Optional[list[str]] = None,
        label_names: Optional[list[str]] = None,
    ) -> None:
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
        self.feature_names = feature_names
        self.label_names = label_names

        self.support: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.support_embeddings: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.prototypes: torch.Tensor = torch.Tensor()

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
    def train_model(
        self, dataset: TensorDataset, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
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
            Dictionary containing summary training information and any other data
            Note that at least one key should be 'train_summary'
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

    def get_label_names(self) -> Optional[list[str]]:
        """
        Retrieve the label names used in the model.

        Returns
        -------
        Optional[list[str]]
            A list of label names if available; otherwise, None.
        """
        if hasattr(self, "label_names"):
            return self.label_names
        return None

    def get_feature_names(self) -> Optional[list[str]]:
        """
        Retrieve the feature names used in the model.

        Returns
        -------
        Optional[list[str]]
            A list of feature names if available; otherwise, None.
        """
        if hasattr(self, "feature_names"):
            return self.feature_names
        return None

    @icontract.require(
        lambda num_features, num_classes: num_features > 0 and num_classes > 0
    )
    def validate_feature_label_names(self, num_features: int, num_classes: int) -> None:
        """
        Validate that the feature names and label names, if provided, match the expected counts.

        Parameters
        ----------
        num_features : int
            The expected number of features.
        num_classes : int
            The expected number of classes.

        Raises
        ------
        ValueError
            If the length of `feature_names` does not match `num_features`, or
            if the length of `label_names` does not match `num_classes`.
        """
        feature_names = self.get_feature_names()
        if feature_names is not None and len(feature_names) != num_features:
            raise ValueError(
                f"The length of feature_names ({len(feature_names)}) does not match the number of data features ({num_features}). Update feature_names or set feature_names to None."
            )

        label_names = self.get_label_names()
        if label_names is not None and len(label_names) != num_classes:
            raise ValueError(
                f"The length of label_names ({len(label_names)}) does not match the number of classes ({num_classes}). Update label_names or set label_names to None."
            )
