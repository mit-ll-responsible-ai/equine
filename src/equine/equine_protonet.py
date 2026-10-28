# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import warnings
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, cast

import icontract
import numpy as np
import torch
from beartype import beartype
from scipy.stats import gaussian_kde
from torch.utils.data import TensorDataset
from tqdm import tqdm

from .equine import Equine, EquineOutput, _apply_to_stored_tensors, _eval_mode
from .registry import _describe_error, _positive_int, _truncate
from .utils import (
    _OOD_DISTANCES,
    EQUINE_FORMAT_VERSION,
    _checked_settings,
    _cholesky_with_jitter,
    _embedding_checkpoint,
    _input_to_model,
    _label_text,
    _labels_to_columns,
    _load_checkpoint_for_device,
    _mahalanobis_from_factor,
    _outlier_kde_from_file,
    _plain_names,
    _plain_train_summary,
    _rebuild_embedding,
    _refuse_torchscript,
    _require_device,
    _require_entries,
    _require_examples,
    _require_finite_embeddings,
    _require_matching_weights,
    _require_names,
    _shown,
    _support_from_file,
    generate_episode,
    generate_support,
    generate_train_summary,
    stratified_train_test_split,
)


#####################################
class CovType(Enum):
    """
    Enum class for covariance types used in EQUINE.
    """

    UNIT = "unit"
    DIAGONAL = "diag"
    FULL = "full"


PRED_COV_TYPE = CovType.DIAGONAL
OOD_COV_TYPE = CovType.DIAGONAL
DEFAULT_EPSILON = 1e-5


def _kde_to_state(kde: gaussian_kde) -> dict[str, Any]:
    """Reduce a fitted ``gaussian_kde`` to tensors/scalars for safe saving."""
    return {
        "dataset": torch.as_tensor(kde.dataset).clone(),
        "bw_factor": float(kde.factor),
    }


COV_REG_TYPE = "epsilon"


###############################################


@beartype
class Protonet(torch.nn.Module):
    """
    Private class that implements a prototypical neural network for use in EQUINE.
    """

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        emb_out_dim: int,
        cov_type: CovType,
        cov_reg_type: str,
        epsilon: float,
        device: str = "cpu",
    ) -> None:
        """
        Protonet class constructor.

        Parameters
        ----------
        embedding_model : torch.nn.Module
            The PyTorch embedding model to generate logits with.
        emb_out_dim : int
            Dimension size of given embedding model's output.
        cov_type : CovType
            Type of covariance to use when computing distances [unit, diag, full].
        cov_reg_type : str
            Type of regularization to use when generating the covariance matrix [epsilon, shared].
        epsilon : float
            Epsilon value to use for covariance regularization.
        device : str, optional
            The device to train the protonet model on (defaults to cpu).
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.cov_type = cov_type
        self.cov_reg_type = cov_reg_type
        self.epsilon = epsilon
        self.emb_out_dim = emb_out_dim
        self.to(device)
        self.device = device

        self.support: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.support_embeddings: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.model_head: torch.nn.Module = self.create_model_head(emb_out_dim)
        self.model_head.to(device)

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> Protonet:
        """
        ``nn.Module._apply``, extended to the support, its embeddings, the prototypes and the covariances.

        ``to``, ``cpu``, ``cuda`` and the dtype casts call it. Moving the model moves
        every tensor the Protonet keeps and updates its ``device`` (see
        ``equine._apply_to_stored_tensors``).
        """
        super()._apply(fn, recurse)
        _apply_to_stored_tensors(self, fn, recurse)
        return self

    def create_model_head(self, emb_out_dim: int) -> torch.nn.Identity:
        """
        Method for adding a PyTorch layer on top of the given embedding model. This layer
        is intended to offer extra degrees of freedom for distance learning in the embedding space.

        Parameters
        ----------
        emb_out_dim : int
            Dimension size of the embedding model output.

        Returns
        -------
        torch.nn.Identity
            The model head: an identity layer.
        """
        return torch.nn.Identity()  # (emb_out_dim, emb_out_dim)

    def compute_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method for calculating model embeddings using both the given embedding model and the added model head.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor to compute embeddings on.

        Returns
        -------
        torch.Tensor
            Fully computed embedding tensors for the given X tensor.
        """
        X = _input_to_model(X, self.embedding_model, self.device)
        model_embeddings = self.embedding_model(X)
        head_embeddings = self.model_head(model_embeddings)
        return head_embeddings

    @icontract.require(lambda self: len(self.support_embeddings) > 0)
    def compute_prototypes(self) -> torch.Tensor:
        """
        Method for computing class prototypes based on given support examples.
        ``Prototypes'' in this context are the means of the support embeddings for each class.

        Returns
        -------
        torch.Tensor
            Tensors of prototypes for each of the given classes in the support.
        """
        # Compute prototype for each class
        proto_list = []
        for label in self.support_embeddings:  # look at doing functorch
            class_prototype = torch.mean(self.support_embeddings[label], dim=0)
            proto_list.append(class_prototype)

        prototypes = torch.stack(proto_list)

        return prototypes

    @icontract.require(lambda self: len(self.support_embeddings) > 0)
    def compute_covariance(self, cov_type: CovType) -> torch.Tensor:
        """
        Method for generating the (regularized) support example covariance matrix(es) used for calculating distances.
        Note that this method is only called once per episode, and the resulting tensor is used for all queries.

        Parameters
        ----------
        cov_type : CovType
            Type of covariance to use [unit, diag, full].

        Returns
        -------
        torch.Tensor
            Tensor containing the generated regularized covariance matrix.
        """
        class_cov_dict = OrderedDict().fromkeys(
            self.support_embeddings.keys(), torch.Tensor()
        )
        for label in self.support_embeddings.keys():
            class_covariance = self.compute_covariance_by_type(
                cov_type, self.support_embeddings[label]
            )
            class_cov_dict[label] = class_covariance

        reg_covariance_dict = self.regularize_covariance(
            class_cov_dict, cov_type, self.cov_reg_type
        )
        reg_covariance = torch.stack(list(reg_covariance_dict.values()))

        return reg_covariance  # TODO try putting everything on GPU with .to() and see if faster

    def compute_covariance_by_type(
        self, cov_type: CovType, embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Select the appropriate covariance matrix type based on cov_type.

        Parameters
        ----------
        cov_type : CovType
            Type of covariance to use. Options are ['unit', 'diag', 'full'].
        embedding : torch.Tensor
            Embedding tensor to use when generating the covariance matrix.

        Returns
        -------
        torch.Tensor
            Tensor containing the requested covariance matrix.

        Raises
        ------
        ValueError
            If ``cov_type`` is ``FULL`` or ``DIAGONAL`` and ``embedding`` has
            fewer than 2 rows.
        """
        if cov_type in (CovType.FULL, CovType.DIAGONAL) and embedding.shape[0] < 2:
            raise ValueError(
                f"A {cov_type.value!r} covariance needs at least 2 examples per "
                f"class; got {embedding.shape[0]}."
            )
        if cov_type == CovType.FULL:
            class_covariance = torch.cov(embedding.T)
        elif cov_type == CovType.DIAGONAL:
            class_covariance = torch.var(embedding, dim=0)
        elif cov_type == CovType.UNIT:
            class_covariance = torch.ones(self.emb_out_dim, device=self.device)
        else:
            raise ValueError

        return class_covariance

    def regularize_covariance(
        self,
        class_cov_dict: OrderedDict[int, torch.Tensor],
        cov_type: CovType,
        cov_reg_type: str,
    ) -> OrderedDict[int, torch.Tensor]:
        """
        Method to add regularization to each class covariance matrix based on the selected regularization type.

        Parameters
        ----------
        class_cov_dict : OrderedDict[int, torch.Tensor]
            A dictionary containing each class and the corresponding covariance matrix.
        cov_type : CovType
            Type of covariance to use [unit, diag, full].
        cov_reg_type : str
            Regularization type: ``"epsilon"`` or ``"shared"``.

        Returns
        -------
        OrderedDict[int, torch.Tensor]
            Dictionary containing the regularized class covariance matrices.
        """

        if cov_type == CovType.FULL:
            regularization = torch.diag(self.epsilon * torch.ones(self.emb_out_dim)).to(
                self.device
            )
        elif cov_type == CovType.DIAGONAL:
            regularization = self.epsilon * torch.ones(self.emb_out_dim).to(self.device)
        elif cov_type == CovType.UNIT:
            regularization = torch.zeros(self.emb_out_dim).to(self.device)

        if cov_reg_type == "shared":
            if cov_type != CovType.FULL and cov_type != CovType.DIAGONAL:
                for label in self.support_embeddings:
                    class_cov_dict[label] = class_cov_dict[label] + regularization
                warnings.warn(
                    "Covariance type UNIT is incompatible with shared regularization, \
                    reverting to epsilon regularization"
                )
                return class_cov_dict

            shared_covariance = self.compute_shared_covariance(class_cov_dict, cov_type)

            for label in self.support_embeddings:
                num_class_support = self.support_embeddings[label].shape[0]
                lamb = num_class_support / (num_class_support + 1)

                class_cov_dict[label] = (
                    lamb * class_cov_dict[label]
                    + (1 - lamb) * shared_covariance
                    + regularization
                )

        elif cov_reg_type == "epsilon":
            for label in class_cov_dict.keys():
                class_cov_dict[label] = (
                    class_cov_dict[label].to(self.device) + regularization
                )

        return class_cov_dict

    def compute_shared_covariance(
        self, class_cov_dict: OrderedDict[int, torch.Tensor], cov_type: CovType
    ) -> torch.Tensor:
        """
        Method to calculate a shared covariance matrix.

        The shared covariance matrix is calculated as the weighted average of the class covariance matrices,
        where the weights are the number of support examples for each class. This is useful when the number of
        support examples for each class is small.

        Parameters
        ----------
        class_cov_dict : OrderedDict[int, torch.Tensor]
            A dictionary containing each class and the corresponding covariance matrix.
        cov_type : CovType
            Type of covariance to use [unit, diag, full].

        Returns
        -------
        torch.Tensor
            Tensor containing the shared covariance matrix.
        """
        total_support = sum([x.shape[0] for x in class_cov_dict.values()])

        if cov_type == CovType.FULL:
            shared_covariance = torch.zeros(
                (self.emb_out_dim, self.emb_out_dim), device=self.device
            )
        elif cov_type == CovType.DIAGONAL:
            shared_covariance = torch.zeros(self.emb_out_dim, device=self.device)
        else:
            raise ValueError(
                "Shared covariance can only be used with FULL or DIAGONAL (not UNIT) covariance types"
            )

        for label in class_cov_dict:
            num_class_support = class_cov_dict[label].shape[0]
            shared_covariance = (
                shared_covariance + (num_class_support - 1) * class_cov_dict[label]
            )  # undo N-1 div from cov

        shared_covariance = shared_covariance / (
            total_support - 1
        )  # redo N-1 div for shared cov

        return shared_covariance

    @icontract.require(lambda X_embed, mu: X_embed.shape[-1] == mu.shape[-1])
    @icontract.ensure(lambda result: torch.all(result >= 0))
    def compute_distance(
        self, X_embed: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        """
        Method to compute the distances to class prototypes for the given embeddings.

        Parameters
        ----------
        X_embed : torch.Tensor
            The embeddings of the query examples.
        mu : torch.Tensor
            The class prototypes (means of the support embeddings).
        cov : torch.Tensor
            The support covariance matrix.

        Returns
        -------
        torch.Tensor
            The calculated distances from each of the class prototypes for the given embeddings.
        """
        _queries = torch.unsqueeze(X_embed, 1)  # examples x 1 x dimension
        diff = torch.sub(mu, _queries)

        if len(cov.shape) == 2:  # (diagonal covariance)
            # examples x classes x dimension
            sq_diff = diff**2
            dist = torch.div(sq_diff.to(self.device), cov.to(self.device))
            dist = torch.sum(dist, dim=2)  # examples x classes
            dist = dist.squeeze(dim=1)
            dist = torch.sqrt(dist + self.epsilon)  # examples x classes
        else:  # len(cov.shape) == 3: (full covariance)
            diff = diff.permute(1, 2, 0)  # classes x dimension x examples
            dist = _mahalanobis_from_factor(diff, _cholesky_with_jitter(cov))
            dist = torch.sqrt(dist.permute(1, 0) + self.epsilon)  # examples x classes
            dist = dist.squeeze(dim=1)
        return dist

    def compute_classes(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Method to compute predicted classes from distances via a softmax function.

        Parameters
        ----------
        distances : torch.Tensor
            The distances of embeddings to class prototypes.

        Returns
        -------
        torch.Tensor
            Class probabilities: the softmax of the negated distances.
        """
        softmax = torch.nn.functional.softmax(torch.neg(distances), dim=-1)
        return softmax

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Protonet forward function, generates class probability predictions and distances from prototypes.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of queries for generating predictions.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            tuple containing class probability predictions, and class distances from prototypes.

        Raises
        ------
        ValueError
            If the model has no support, or, in eval mode, the embeddings of
            ``X`` hold a NaN or infinite value.
        """
        classes, distances, _ = self._forward_with_embeddings(X)
        return classes, distances

    def _forward_with_embeddings(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ``forward`` that also returns the embeddings it computed.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Class probability predictions, class distances from prototypes,
            and the embeddings (``compute_embeddings(X)``, single examples
            unsqueezed to a batch of one).

        Raises
        ------
        ValueError
            If the model has no support, or, in eval mode, the embeddings of
            ``X`` hold a NaN or infinite value.
        """
        if len(self.support) == 0 or len(self.support_embeddings) == 0:
            raise ValueError(
                "No support examples found. Protonet Model requires model support to \
                    be set with the 'update_support()' method before calling forward."
            )

        X_embed = self.compute_embeddings(X)
        if X_embed.shape == torch.Size([self.emb_out_dim]):
            X_embed = X_embed.unsqueeze(dim=0)  # handle single examples
        if not self.training:
            _require_finite_embeddings(X_embed)
        distances = self.compute_distance(X_embed, self.prototypes, self.covariance)
        classes = self.compute_classes(distances)

        return classes, distances, X_embed

    def update_support(
        self,
        support: OrderedDict[int, torch.Tensor],
        *,
        _global_moments: bool = True,
    ) -> None:
        """
        Method to update the support examples, and all the calculations that rely on them.

        The global moments of the support embeddings (used for the OOD score)
        are computed whatever mode the caller left the model in (via
        ``train()``/``eval()``). The covariance type depends on the
        mode: ``PRED_COV_TYPE`` in eval mode, ``cov_type`` in training mode.

        Parameters
        ----------
        support : OrderedDict
            Ordered dict containing class labels and their associated support examples.
        _global_moments : bool, optional
            Private, keyword-only. False skips the global moments; only the episode loop
            of ``EquineProtonet.train_model`` passes it.
        """
        # Through the model boundary: on the device and, for floating support, in the
        # embedding's dtype.
        self.support = OrderedDict(
            (label, _input_to_model(x, self.embedding_model, self.device))
            for label, x in support.items()
        )  # TODO torch.nn.ParameterDict(support)

        support_embs = OrderedDict().fromkeys(support.keys(), torch.Tensor())
        for label in self.support:
            support_embs[label] = self.compute_embeddings(self.support[label])

        self.support_embeddings = (
            support_embs  # TODO torch.nn.ParameterDict(support_embs)
        )

        self.prototypes: torch.Tensor = self.compute_prototypes()
        if _global_moments:
            self.compute_global_moments()

        if self.training is False:
            self.covariance: torch.Tensor = self.compute_covariance(
                cov_type=PRED_COV_TYPE
            )
        else:
            self.covariance: torch.Tensor = self.compute_covariance(
                cov_type=self.cov_type
            )

    @icontract.require(lambda self: len(self.support_embeddings) > 0)
    def compute_global_moments(self) -> None:
        """Method to calculate the global moments of the support embeddings for use in OOD score generation"""
        embeddings = torch.cat(list(self.support_embeddings.values()))
        self.global_covariance = torch.unsqueeze(
            self.compute_covariance_by_type(OOD_COV_TYPE, embeddings), dim=0
        )
        global_reg_input = OrderedDict().fromkeys([0], torch.Tensor())
        global_reg_input[0] = self.global_covariance
        self.global_covariance: torch.Tensor = self.regularize_covariance(
            global_reg_input, OOD_COV_TYPE, "epsilon"
        )[0]
        self.global_mean: torch.Tensor = torch.mean(embeddings, dim=0)


def _require_examples_per_class(
    y: torch.Tensor, labels: list[int], part: str, minimum: int
) -> None:
    """
    Refuse a ``calib_frac`` split that leaves a class with too few examples.

    Parameters
    ----------
    y : torch.Tensor
        Labels of one side of the split.
    labels : list[int]
        Every class the model is given.
    part : str
        The side of the split, named in the message.
    minimum : int
        The fewest examples each class needs on this side.

    Raises
    ------
    ValueError
        If a class in ``labels`` has fewer than ``minimum`` examples in ``y``.
    """
    for label in labels:
        count = int((y == label).sum())
        if count < minimum:
            raise ValueError(
                f"Class {_label_text(label)} has {count} {part} example(s) after the "
                f"calib_frac split; it needs at least {minimum}."
            )


###############################################
@beartype
class EquineProtonet(Equine):
    """
    A class representing an EQUINE model that utilizes protonets and (optionally) relative Mahalanobis distances
    to generate OOD and model confidence scores. This wraps any pytorch embedding neural network
    and provides the `forward`, `predict`, `save`, and `load` methods required by Equine.
    """

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        emb_out_dim: int,
        cov_type: CovType = CovType.UNIT,
        relative_mahal: bool = True,
        use_temperature: bool = False,
        init_temperature: float = 1.0,
        device: str = "cpu",
        feature_names: list[str] | None = None,
        label_names: list[str] | None = None,
        ood_distance: str = "rmd_sq",
    ) -> None:
        """
        EquineProtonet class constructor

        Parameters
        ----------
        embedding_model : torch.nn.Module
            Neural Network feature embedding model.
        emb_out_dim : int
            The number of output features from the embedding model.
        cov_type : CovType, optional
            The type of covariance to use when training the protonet [UNIT, DIAG, FULL], by default CovType.UNIT.
        relative_mahal : bool, optional
            Use relative mahalanobis distance for OOD calculations. If false, uses standard mahalanobis distance instead, by default True.
        use_temperature : bool, optional
            Whether to use temperature scaling after training, by default False.
        init_temperature : float, optional
            What to use as the initial temperature (1.0 has no effect), by default 1.0.
        device : str, optional
            The device to train the equine model on (defaults to cpu).
        feature_names : list[str], optional
            List of strings of the names of the tabular features (ex ["duration", "fiat_mean", ...])
        label_names : list[str], optional
            List of strings of the names of the labels (ex ["streaming", "voip", ...])
        ood_distance : str, optional
            The OOD statistic used when ``relative_mahal`` is True, by default
            ``"rmd_sq"``: the squared Mahalanobis distance to the predicted class
            prototype minus the squared Mahalanobis distance to the mean of the
            support embeddings, both computed without the temperature.
            ``"rmd_sqrt"`` subtracts the distances themselves, and with
            ``use_temperature`` divides the class distance by the temperature
            first. A model file without this setting loads as ``"rmd_sqrt"``.
            The outlier-score KDEs are fitted for the statistic in use: after
            changing ``ood_distance``, refit them with ``update_support`` or
            ``train_model``.

        Raises
        ------
        ValueError
            If ``ood_distance`` is not ``"rmd_sq"`` or ``"rmd_sqrt"``.
        """
        if ood_distance not in _OOD_DISTANCES:
            raise ValueError(
                f"ood_distance must be one of {list(_OOD_DISTANCES)}; got "
                f"{_shown(ood_distance)}."
            )
        super().__init__(
            embedding_model,
            device=device,
            feature_names=feature_names,
            label_names=label_names,
        )
        self.cov_type = cov_type
        self.cov_reg_type = COV_REG_TYPE
        self.relative_mahal = relative_mahal
        self.ood_distance = ood_distance
        self.emb_out_dim = emb_out_dim
        self.epsilon = DEFAULT_EPSILON
        self.outlier_score_kde: OrderedDict[int, gaussian_kde] = OrderedDict()
        self.model_summary: dict[str, Any] = dict()
        self.use_temperature = use_temperature
        self.init_temperature = init_temperature
        self.register_buffer(
            "temperature", torch.Tensor(self.init_temperature * torch.ones(1))
        )

        self.model: torch.nn.Module = Protonet(
            embedding_model,
            self.emb_out_dim,
            self.cov_type,
            self.cov_reg_type,
            self.epsilon,
            device=device,
        )
        # Moves the temperature buffer onto the model's device.
        self.to(self.device)

    @property
    def support(self) -> OrderedDict[int, torch.Tensor]:
        """
        The support examples of each class, keyed by label: the dict that
        ``get_support()`` returns, empty before the first support update.
        Read-only.
        """
        return cast(Protonet, self.model).support

    @property
    def support_embeddings(self) -> OrderedDict[int, torch.Tensor]:
        """
        The embeddings of the support examples, keyed by label; empty before the
        first support update. Read-only.
        """
        return cast(Protonet, self.model).support_embeddings

    @property
    def prototypes(self) -> torch.Tensor:
        """
        The class prototypes: the tensor that ``get_prototypes()`` returns, an
        empty tensor before the first support update. Read-only.
        """
        return cast(Protonet, self.model).__dict__.get("prototypes", torch.Tensor())

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore a pickled model; a pickle without ``ood_distance`` uses
        ``"rmd_sqrt"``. ``support``, ``support_embeddings`` and ``prototypes``
        entries in its instance dict are dropped: these names are read-only
        properties.
        """
        state.setdefault("ood_distance", "rmd_sqrt")
        for name in ("support", "support_embeddings", "prototypes"):
            state.pop(name, None)
        super().__setstate__(state)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generates class probabilities based on the input tensor.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor for generating predictions.

        Returns
        -------
        torch.Tensor
            Class probabilities: the softmax of the negated distances to the
            class prototypes.

        Raises
        ------
        ValueError
            If the model has no support, or, in eval mode, the embeddings of
            ``X`` hold a NaN or infinite value.
        """
        preds, _ = self.model(X)
        return preds

    @icontract.require(lambda calib_frac: calib_frac > 0 and calib_frac < 1)
    def train_model(
        self,
        dataset: TensorDataset,
        num_episodes: int,
        calib_frac: float = 0.2,
        support_size: int = 25,
        way: int = 3,
        episode_size: int = 100,
        loss_fn: Callable = torch.nn.functional.cross_entropy,
        opt_class: Callable = torch.optim.Adam,
        num_calibration_epochs: int = 2,
        calibration_lr: float = 0.01,
    ) -> dict[str, Any]:
        """
        Train or fine-tune an EquineProtonet model.

        Parameters
        ----------
        dataset : TensorDataset
            Input pytorch TensorDataset of training data for model.
        num_episodes : int
            The desired number of episodes to use for training.
        calib_frac : float, optional
            Fraction of given training data to reserve for model calibration, by default 0.2.
        support_size : int, optional
            Number of support examples to generate for each class, by default 25.
        way : int, optional
            Number of classes to train on per episode, by default 3.
        episode_size : int, optional
            Number of examples to use per episode, by default 100.
        loss_fn : Callable, optional
            A pytorch loss function, eg., torch.nn.CrossEntropyLoss(), by default torch.nn.functional.cross_entropy.
        opt_class : Callable, optional
            A pytorch optimizer, e.g., torch.optim.Adam, by default torch.optim.Adam.
        num_calibration_epochs : int, optional
            The desired number of epochs to use for temperature scaling, by default 2.
        calibration_lr : float, optional
            Learning rate for temperature scaling, by default 0.01.

        Returns
        -------
        dict[str, Any]
            ``"train_summary"``: the training summary (also kept as
            ``self.train_summary``); ``"calib_x"`` and ``"calib_y"``: the held
            out calibration data and labels as split from ``dataset`` (on its
            device and in its dtypes; the copies moved to the model device are
            internal).

        Raises
        ------
        ValueError
            If ``support_size`` is below 2, or the ``calib_frac`` split leaves a
            class with fewer than 2 calibration examples or too few training
            examples (more than ``support_size`` when ``num_episodes`` is
            positive, ``support_size`` otherwise); the model is then unchanged. Also, after
            training, if a calibration example's embedding or OOD distance holds
            a NaN or infinite value, or a class's calibration examples all have
            the same OOD distance; the model is then trained but its outlier
            scores are not refitted.
        """
        if support_size < 2:
            raise ValueError(
                f"support_size must be at least 2, got {support_size}: the support's "
                "diagonal covariance needs 2 examples per class."
            )
        X, Y = dataset[:]
        labels = torch.unique(Y).tolist()

        self.validate_feature_label_names(X.shape[-1], len(labels))

        train_x, calib_x, train_y, calib_y = stratified_train_test_split(
            X, Y, test_size=calib_frac
        )
        episode_queries = 1 if num_episodes > 0 else 0
        _require_examples_per_class(
            train_y, labels, "training", support_size + episode_queries
        )
        _require_examples_per_class(calib_y, labels, "calibration", 2)

        self.train()

        if self.use_temperature:
            # Rounded through the default dtype, then widened to the buffer's dtype.
            self.temperature: torch.Tensor = torch.full(
                (1,), self.init_temperature, device=self.temperature.device
            ).to(self.temperature.dtype)

        optimizer = opt_class(self.parameters())

        # Episodes and the support are sampled from train_x/train_y on the
        # dataset's device and cross to self.device at the model boundary. The
        # model uses the calibration split on self.device; the caller's is returned.
        model_calib_x = _input_to_model(calib_x, self.embedding_model, self.device)
        model_calib_y = calib_y.to(self.device)

        for i in tqdm(range(num_episodes)):
            optimizer.zero_grad()

            support, episode_x, episode_y = generate_episode(
                train_x, train_y, support_size, way, episode_size
            )
            # Episodes read only the prototypes and the covariance; the
            # global moments are computed by the full-support update below.
            self.model.update_support(support, _global_moments=False)

            _, dists = self.model(episode_x)
            loss_value = loss_fn(
                torch.neg(dists).to(self.device), episode_y.to(self.device)
            )
            loss_value.backward()
            optimizer.step()

        self.eval()
        full_support = generate_support(
            train_x,
            train_y,
            support_size,
            selected_labels=torch.unique(train_y).tolist(),
        )

        # Embedded without autograd: the result carries no graph.
        with torch.no_grad():
            self.model.update_support(
                full_support
            )  # update support with final selected examples

        # The OOD-calibration pass records no autograd graph.
        with torch.no_grad():
            pred_probs, dists, X_embed = self.model._forward_with_embeddings(
                model_calib_x
            )
            ood_dists = self._compute_ood_dist(X_embed, pred_probs, dists)
            self._fit_outlier_scores(ood_dists, model_calib_y)

        if self.use_temperature:
            self.calibrate_temperature(
                model_calib_x, model_calib_y, num_calibration_epochs, calibration_lr
            )

        date_trained = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        self.train_summary: dict[str, Any] = generate_train_summary(
            self, train_y, date_trained
        )
        return_dict: dict[str, Any] = dict()
        return_dict["train_summary"] = self.train_summary
        return_dict["calib_x"] = calib_x
        return_dict["calib_y"] = calib_y
        return return_dict

    def calibrate_temperature(
        self,
        calib_x: torch.Tensor,
        calib_y: torch.Tensor,
        num_calibration_epochs: int = 1,
        calibration_lr: float = 0.01,
    ) -> None:
        """
        Fine-tune the temperature on the given examples. ``train_model`` calls this
        method when ``use_temperature`` is True.

        Parameters
        ----------
        calib_x : torch.Tensor
            Training data to be used for temperature calibration.
        calib_y : torch.Tensor
            Class labels corresponding to `calib_x`, each one of the support's
            labels.
        num_calibration_epochs : int, optional
            Number of epochs to tune temperature, by default 1.
        calibration_lr : float, optional
            Learning rate for temperature optimization, by default 0.01.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the model has no support, `calib_y` holds a label that is not in
            the support, or, in eval mode, the embeddings of ``calib_x`` hold a
            NaN or infinite value.
        """
        if len(cast(Protonet, self.model).support) == 0:
            raise ValueError(
                "No support examples found; call train_model or update_support "
                "before calibrate_temperature."
            )
        targets = _labels_to_columns(calib_y, self._label_order()).to(self.device)
        self.temperature.requires_grad = True
        optimizer = torch.optim.Adam([self.temperature], lr=calibration_lr)
        for t in range(num_calibration_epochs):
            optimizer.zero_grad()
            with torch.no_grad():
                pred_probs, dists = self.model(calib_x)
            dists = dists.to(self.device) / self.temperature.to(self.device)
            loss = torch.nn.functional.cross_entropy(
                torch.neg(dists).to(self.device), targets
            )
            loss.backward()
            optimizer.step()
        self.temperature.requires_grad = False

    @icontract.ensure(lambda self: len(self.model.support_embeddings) > 0)
    def _fit_outlier_scores(
        self, ood_dists: torch.Tensor, calib_y: torch.Tensor
    ) -> None:
        """
        Private function to fit outlier scores with a kernel density estimate (KDE).

        Replaces ``outlier_score_kde`` with one KDE per support label, keyed
        by label, in support order.

        Parameters
        ----------
        ood_dists : torch.Tensor
            Tensor of computed OOD distances.
        calib_y : torch.Tensor
            Tensor of class labels for `ood_dists` examples. Every class needs
            at least 2 examples.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a class's OOD distances include a NaN or infinite value or are all
            equal, or fitting a KDE fails; no KDE is changed then.
        """
        class_ood_dists = OrderedDict(
            (label, ood_dists[calib_y == int(label)])
            for label in self.model.support_embeddings.keys()
        )
        for label, dists in class_ood_dists.items():
            if not bool(torch.isfinite(dists).all()):
                raise ValueError(
                    f"A calibration example of class {label} has a NaN or infinite "
                    "OOD distance; fitting its outlier score needs finite values."
                )
            if bool((dists == dists[0]).all()):
                raise ValueError(
                    f"Every calibration example of class {label} has the same OOD "
                    f"distance ({dists[0].item()}); fitting its outlier score needs "
                    "at least 2 distinct values."
                )
        # TODO convert to torch func
        fitted = OrderedDict(
            (label, gaussian_kde(dists.cpu().detach().numpy()))
            for label, dists in class_ood_dists.items()
        )
        self.outlier_score_kde = fitted

    def _compute_outlier_scores(self, ood_dists, predictions) -> torch.Tensor:
        """
        Private function to compute OOD scores using the calculated kernel density estimate (KDE).

        Each example is scored with the KDE of its predicted label: column
        ``j`` of ``predictions`` is the ``j``-th support label.

        Parameters
        ----------
        ood_dists : torch.Tensor
            Tensor of computed OOD distances.
        predictions : torch.Tensor
            Tensor of model protonet predictions.

        Returns
        -------
        torch.Tensor
            Tensor of OOD scores for the given examples.

        Raises
        ------
        ValueError
            If a support label has no outlier-score KDE, or the KDE of a
            predicted label has more than one dimension.
        """
        label_order = self._label_order()
        missing = [
            label for label in label_order if label not in self.outlier_score_kde
        ]
        if missing:
            raise ValueError(
                f"No outlier-score KDE for support labels {missing}; refit the "
                "outlier scores with EquineProtonet.update_support."
            )
        dists = ood_dists.detach().cpu()
        columns = torch.argmax(predictions, dim=1).cpu()
        ood_scores = torch.zeros(len(dists), dtype=torch.float64)
        for column, label in enumerate(label_order):
            rows = torch.where(columns == column)[0]
            if len(rows) == 0:
                continue
            kde = self.outlier_score_kde[label]
            if kde.d != 1:
                raise ValueError(
                    f"The outlier-score KDE of support label {label} has {kde.d} "
                    "dimensions; it must have one."
                )
            data = torch.from_numpy(kde.dataset[0])
            weights = torch.from_numpy(kde.weights)
            uniform = bool((weights == weights[0]).all())
            bandwidth = float(np.sqrt(kde.covariance[0, 0]))
            # 1 minus the KDE's mass above each distance, for at most 2**22
            # (distance, KDE point) pairs at a time
            for chunk in rows.split(max(1, 2**22 // len(data))):
                z = (dists[chunk, None] - data).double() / bandwidth
                upper = 1.0 - torch.special.ndtr(z)
                if uniform:
                    tail = upper.mean(dim=1)
                else:
                    tail = (upper @ weights).clamp(0.0, 1.0)
                ood_scores[chunk] = 1.0 - tail

        return ood_scores.to(dtype=ood_dists.dtype, device=ood_dists.device)

    @icontract.ensure(lambda result: len(result) > 0)
    def _compute_ood_dist(
        self,
        X_embeddings: torch.Tensor,
        predictions: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Private function to compute OOD distances using a distance function.

        With ``relative_mahal``, the distance to the predicted class minus the
        distance to the global mean (``ood_distance="rmd_sqrt"``) or the
        difference of their squares (``"rmd_sq"``); otherwise the distance to
        the predicted class.

        Parameters
        ----------
        X_embeddings : torch.Tensor
            Tensor of example embeddings.
        predictions : torch.Tensor
            Tensor of model protonet predictions for the given embeddings.
        distances : torch.Tensor
            Tensor of calculated protonet distances for the given embeddings.

        Returns
        -------
        torch.Tensor
            Tensor of OOD distances for the given embeddings.
        """
        preds = torch.argmax(predictions, dim=1)
        preds = preds.unsqueeze(dim=-1)
        # Calculate (Relative) Mahalanobis Distance:
        if self.relative_mahal:
            null_distance = self.model.compute_distance(
                X_embeddings, self.model.global_mean, self.model.global_covariance
            )
            null_distance = null_distance.unsqueeze(dim=-1)
            class_distance = distances.gather(1, preds)
            if self.ood_distance == "rmd_sq":
                ood_dist = class_distance**2 - null_distance**2
            else:
                ood_dist = class_distance - null_distance
        else:
            ood_dist = distances.gather(1, preds)

        ood_dist = torch.reshape(ood_dist, (-1,))
        return ood_dist

    def predict(self, X: torch.Tensor) -> EquineOutput:
        """Predict function for EquineProtonet, inherited and implemented from Equine.

        Computes in eval mode and leaves the model in the mode the caller had it in (via
        ``train()``/``eval()``); a prediction is the same whatever mode the caller left
        the model in. Every submodule's mode is restored; an embedding frozen with
        ``model.embedding_model.eval()`` during fine-tuning stays frozen.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor. It is moved to the model device and cast to the embedding
            model's parameter dtype; a float64 input to a float32 model is accepted.

        Returns
        -------
        EquineOutput
            Output object containing prediction probabilities, OOD scores and
            embeddings. Column ``j`` of the probabilities is the ``j``-th label
            of ``get_support()``; ``train_model`` and ``update_support`` put
            the labels in ascending order. Computed under ``torch.no_grad()``:
            the tensors carry no autograd graph but are ordinary tensors (not
            inference-mode tensors), which a caller can use in autograd.

        Raises
        ------
        ValueError
            If ``X`` is empty, its embeddings hold a NaN or infinite value, or a
            support label has no outlier-score KDE.
        """
        _require_examples(X)
        # One embedding pass, under no_grad.
        with _eval_mode(self), torch.no_grad():
            preds, dists, X_embed = self.model._forward_with_embeddings(X)
            if self.use_temperature:
                scaled = dists / self.temperature
                preds = torch.softmax(torch.negative(scaled), dim=1)
                # The relative "rmd_sq" statistic reads the unscaled distances;
                # the other statistics read the scaled ones.
                if not (self.relative_mahal and self.ood_distance == "rmd_sq"):
                    dists = scaled
            ood_dist = self._compute_ood_dist(X_embed, preds, dists)
            ood_scores = self._compute_outlier_scores(ood_dist, preds)

        self.validate_feature_label_names(X.shape[-1], preds.shape[-1])

        return EquineOutput(classes=preds, ood_scores=ood_scores, embeddings=X_embed)

    @icontract.require(
        lambda calib_frac: (
            not isinstance(calib_frac, (int, float)) or 0 < calib_frac < 1
        )
    )
    def update_support(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        *args: Any,
        calib_frac: float | None = None,
        label_names: list[str] | None = None,
    ) -> None:
        """Function to update protonet support examples with given examples.

        Computes in eval mode and leaves the model in the mode the caller had it in (via
        ``train()``/``eval()``); it works on a freshly constructed model, which is in
        training mode. Every submodule's mode is restored; an embedding frozen with
        ``model.embedding_model.eval()`` during fine-tuning stays frozen.

        Parameters
        ----------
        support_x : torch.Tensor
            Tensor containing support examples for protonet.
        support_y : torch.Tensor
            Tensor containing labels for given support examples.
        *args : Any
            Deprecated: ``calib_frac`` and ``label_names`` passed by position, in
            that order. Emits a ``DeprecationWarning`` and is then handled as the
            keywords.
        calib_frac : float
            Fraction of given support data to use for OOD calibration. Required,
            keyword-only.
        label_names : list[str], optional
            List of strings of the names of the labels (ex ["streaming", "voip", ...]).
            Keyword-only.

        Raises
        ------
        TypeError
            If ``calib_frac`` is missing, an option is passed both by position and
            by keyword, or more than two options are passed by position.
        ValueError
            If the ``calib_frac`` split leaves a class with fewer than 2 support
            or 2 calibration examples, a calibration example's embedding or OOD
            distance holds a NaN or infinite value, or a class's calibration
            examples all have the same OOD distance. A failed update leaves the
            model as it was.
        """
        if len(args) > 2:
            raise TypeError(
                "EquineProtonet.update_support() takes at most 2 options by "
                f"position (calib_frac, label_names) but {len(args)} were given"
            )
        if args:
            if calib_frac is not None or (len(args) == 2 and label_names is not None):
                name = "calib_frac" if calib_frac is not None else "label_names"
                raise TypeError(
                    "EquineProtonet.update_support() got multiple values for "
                    f"argument '{name}'"
                )
            warnings.warn(
                "Passing calib_frac or label_names to EquineProtonet.update_support "
                "by position is deprecated and will stop working in the next "
                "release: pass calib_frac=... and label_names=... by keyword.",
                DeprecationWarning,
                # Past the icontract and beartype wrappers (none under -O), to the
                # caller's line.
                stacklevel=2 + (0 if sys.flags.optimize else 2),
            )
            EquineProtonet.update_support(
                self,
                support_x,
                support_y,
                calib_frac=args[0],
                label_names=args[1] if len(args) == 2 else label_names,
            )
            return
        if calib_frac is None:
            raise TypeError(
                "EquineProtonet.update_support() missing required keyword argument: "
                "'calib_frac'"
            )
        with _eval_mode(self):
            all_labels = torch.unique(support_y).tolist()
            support_x, calib_x, support_y, calib_y = stratified_train_test_split(
                support_x, support_y, test_size=calib_frac
            )
            _require_examples_per_class(support_y, all_labels, "support", 2)
            _require_examples_per_class(calib_y, all_labels, "calibration", 2)
            # Restored if a later step raises.
            state, model_state = dict(vars(self)), dict(vars(self.model))
            try:
                labels, counts = torch.unique(support_y, return_counts=True)
                if label_names is not None:
                    self.label_names = label_names
                self.validate_feature_label_names(support_x.shape[-1], labels.shape[0])

                support = OrderedDict()
                for label, count in list(zip(labels.tolist(), counts.tolist())):
                    class_support = generate_support(
                        support_x,
                        support_y,
                        support_size=count,
                        selected_labels=[label],
                    )
                    support.update(class_support)

                # The support and the calibration pass record no autograd graph.
                with torch.no_grad():
                    self.model.update_support(support)
                    preds, dists, X_embed = self.model._forward_with_embeddings(calib_x)
                    ood_dists = self._compute_ood_dist(X_embed, preds, dists)
                    self._fit_outlier_scores(ood_dists, calib_y)
            except Exception:
                vars(self).clear()
                vars(self).update(state)
                vars(self.model).clear()
                vars(self.model).update(model_state)
                raise

    @icontract.require(lambda self: len(self.model.support) > 0)
    def get_support(self) -> OrderedDict[int, torch.Tensor]:
        """
        Get the support examples for the model.

        Returns
        -------
        OrderedDict[int, torch.Tensor]
            The support examples for the model.
        """
        return self.model.support

    def _label_order(self) -> list[int]:
        """
        The class label of each column of ``predict``'s ``classes``: the
        support's labels, in support order.

        Returns
        -------
        list[int]
            The support's labels.
        """
        return list(cast(Protonet, self.model).support_embeddings)

    @icontract.require(lambda self: len(self.model.prototypes) > 0)
    def get_prototypes(self) -> torch.Tensor:
        """
        Get the prototypes for the model (the class means of the support embeddings).

        Returns
        -------
        torch.Tensor
            The prototpes for the model.
        """
        return self.model.prototypes

    def save(self, path: str) -> None:
        """
        Save all model parameters to a file.

        The embedding model is stored as a *recipe* (registered architecture
        name plus constructor arguments) and a ``state_dict``, so the file holds
        only data.

        Parameters
        ----------
        path : str
            Filename to write the model.

        Raises
        ------
        ValueError
            If the embedding model is not a registered architecture.
        RuntimeError
            If the model has no support examples (neither ``train_model`` nor
            ``update_support`` has run).
        """
        if len(cast(Protonet, self.model).support) == 0:
            raise RuntimeError(
                "The model has no support examples to save; call train_model or "
                "update_support first."
            )
        torch.save(self._to_checkpoint(), path)

    def _to_checkpoint(self) -> dict[str, Any]:
        """Build the checkpoint dictionary that ``save`` writes and ``_from_checkpoint`` reads."""
        model_settings = {
            "cov_type": self.cov_type.value,
            "emb_out_dim": self.emb_out_dim,
            "use_temperature": self.use_temperature,
            "init_temperature": self.temperature.item(),
            "relative_mahal": self.relative_mahal,
            "ood_distance": self.ood_distance,
            "device": self.device,
        }
        # Plain values only, so load_checkpoint can read it with weights_only=True.
        save_data: dict[str, Any] = {
            "equine_format_version": EQUINE_FORMAT_VERSION,
            **_embedding_checkpoint(self.model.embedding_model),
            "feature_names": _plain_names(self.feature_names),
            "label_names": _plain_names(self.label_names),
            "model_head_save": self.model.model_head.state_dict(),
            "outlier_kde": {
                int(label): _kde_to_state(kde)
                for label, kde in self.outlier_score_kde.items()
            },
            "settings": model_settings,
            # Support tensors usually view the training data; store copies.
            "support": {
                int(label): x.detach().clone()
                for label, x in self.model.support.items()
            },
            "train_summary": _plain_train_summary(self.train_summary),
        }
        return save_data

    @classmethod
    def load(
        cls,
        path: str,
        device: str | None = None,
        *,
        embedding_model: torch.nn.Module | None = None,
    ) -> Equine:
        """
        Load a previously saved EquineProtonet model.

        Loading runs no code from the file: it is read with
        ``torch.load(weights_only=True)`` and the embedding model is rebuilt from its
        recipe through the architecture registry.

        Parameters
        ----------
        path : str
            The filename of the saved model.
        device : str | None
            The device to load the model onto. Overrides the device recorded
            in the file; by default the model lands on the saved device.
        embedding_model : torch.nn.Module | None
            Keyword-only. Use this module as the embedding architecture instead
            of rebuilding it from the file's recipe; the file's weights are
            loaded into it, and its floating-point tensors take their stored
            dtypes.

        Returns
        -------
        EquineProtonet
            The reconstituted EquineProtonet object.

        Raises
        ------
        ValueError
            If the file is not in the current EQUINE format or stores its
            embedding model as TorchScript (both are rewritten once with
            ``python -m equine.upgrade``), or names an unregistered
            architecture. If ``device`` is unavailable here, names an index at
            or above its device count, or lacks a dtype the file holds (float64
            on MPS: load such a file with ``device="cpu"``).
        """
        model_save = _load_checkpoint_for_device(path, device)
        _refuse_torchscript(model_save, path)
        return cls._from_checkpoint(model_save, device, embedding_model=embedding_model)

    @classmethod
    def _from_checkpoint(
        cls,
        model_save: dict[str, Any],
        device: str | None = None,
        embedding_model: torch.nn.Module | None = None,
    ) -> Equine:
        """
        Rebuild an EquineProtonet from an already-loaded checkpoint dictionary.

        Parameters
        ----------
        model_save : dict[str, Any]
            The dictionary returned by ``utils.load_checkpoint``.
        device : str | None
            Device override for the reconstituted model.
        embedding_model : torch.nn.Module | None
            Use this module as the embedding architecture instead of the file's
            recipe; the file's weights are loaded into it, and its
            floating-point tensors take their stored dtypes.

        Returns
        -------
        EquineProtonet
            The reconstituted EquineProtonet object.
        """
        # The embedding is rebuilt first.
        embedding = _rebuild_embedding(model_save, embedding_model)
        _require_entries(
            model_save,
            ("settings", "support", "train_summary", "outlier_kde", "model_head_save"),
            cls,
        )
        support = _support_from_file(model_save["support"])
        if len(support) == 0:
            raise ValueError(
                "Model file's support is empty; an EquineProtonet needs at least one "
                "labelled support example."
            )
        for label, rows in support.items():
            if rows.shape[0] < 2:
                raise ValueError(
                    f"Model file's support holds {rows.shape[0]} example(s) of class "
                    f"{label}; an EquineProtonet needs at least 2 per class."
                )
        outlier_kde = _outlier_kde_from_file(model_save["outlier_kde"])
        for key in ("feature_names", "label_names"):
            _require_names(model_save.get(key), key)
        head_save = model_save["model_head_save"]
        if isinstance(head_save, dict) and set(head_save) == {"weight", "bias"}:
            raise ValueError(
                "Model file's model_head_save holds the weights of a Linear head, "
                "which EQUINE <= 0.1.6 saved; the head is now an Identity. Rewrite "
                "the file with `python -m equine.upgrade`, which drops those weights."
            )

        settings = _checked_settings(cls, model_save["settings"])
        # The model is built from the file's settings and then gets the file's
        # ood_distance, or "rmd_sqrt" for a file without it.
        ood_distance = settings.get("ood_distance", "rmd_sqrt")
        # emb_out_dim must be an int >= 1, not a bool.
        _positive_int(
            settings.get("emb_out_dim"), "Model file's settings['emb_out_dim']"
        )
        if "cov_type" in settings:
            cov_type = settings["cov_type"]
            try:
                settings["cov_type"] = CovType(cov_type)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"Model file's settings['cov_type'] ({_shown(cov_type)}) is not "
                    f"one of {[member.value for member in CovType]}."
                ) from err
        # Allow the user to override the saved device state dynamically
        if device is not None:
            settings["device"] = device
        else:
            _require_device(settings, hint=" (pass device= to choose another)")

        # The head is an Identity, which has no weights.
        _require_matching_weights({}, head_save, "model_head_save")
        try:
            eq_model = cls(embedding, **settings)
        except Exception as err:
            raise ValueError(
                f"Could not build {cls.__name__} from the model file's settings "
                f"({_truncate(_describe_error(err), 200)})."
            ) from err
        eq_model.ood_distance = ood_distance

        eq_model.model.model_head.load_state_dict(head_save)
        eq_model.eval()
        support_error = (
            "Model file's support could not be passed through its embedding model "
            "({}); the file is inconsistent or was tampered with."
        )
        # emb_out_dim must be 1 or the embedding's output width.
        try:
            with torch.no_grad():
                probe = cast(Protonet, eq_model.model).compute_embeddings(
                    next(iter(support.values()))[:1]
                )
        except Exception as err:
            raise ValueError(
                support_error.format(_truncate(_describe_error(err), 200))
            ) from err
        width = probe.shape[-1] if probe.dim() > 0 else 1
        if eq_model.emb_out_dim not in (1, width):
            raise ValueError(
                "Model file's settings['emb_out_dim'] is neither 1 nor the "
                f"{width} features its embedding model outputs; the file is "
                "inconsistent or was tampered with."
            )
        try:
            with torch.no_grad():
                eq_model.model.update_support(support)
        except Exception as err:
            raise ValueError(
                support_error.format(_truncate(_describe_error(err), 200))
            ) from err

        eq_model.feature_names = model_save.get("feature_names")
        eq_model.label_names = model_save.get("label_names")
        eq_model.outlier_score_kde = outlier_kde
        eq_model.train_summary = model_save.get("train_summary")

        return eq_model
