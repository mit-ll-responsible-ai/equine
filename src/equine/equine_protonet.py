# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, Self, cast

import icontract
import numpy as np
import torch
from beartype import beartype
from scipy.stats import gaussian_kde
from torch.utils.data import TensorDataset
from tqdm import tqdm

from . import _model_file
from .equine import (
    MIN_TEMPERATURE,
    Equine,
    EquineOutput,
    _apply_to_stored_tensors,
    _eval_mode,
)
from .registry import _describe_error, _positive_int, _truncate
from .utils import (
    _OOD_DISTANCES,
    EQUINE_FORMAT_VERSION,
    _checked_settings,
    _cholesky_with_jitter,
    _embedding_checkpoint,
    _EpisodeSampler,
    _get_shuffle_idxs_by_class,
    _input_to_model,
    _label_text,
    _labels_to_columns,
    _load_checkpoint_for_device,
    _mahalanobis_from_factor,
    _outlier_kde_from_file,
    _plain_names,
    _plain_train_summary,
    _rebuild_embedding,
    _require_calibration_settings,
    _require_device,
    _require_entries,
    _require_examples,
    _require_finite_embeddings,
    _require_fraction,
    _require_matching_weights,
    _require_names,
    _require_rows,
    _require_same_length,
    _require_support,
    _require_whole_number_labels,
    _shown,
    _support_from_file,
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
        "bw_factor": float(cast(float, kde.factor)),
    }


def _upper_tail(z: torch.Tensor, weights: torch.Tensor, uniform: bool) -> torch.Tensor:
    """
    A KDE's mass above each row's distance, from the standardized differences
    ``z`` (one row per distance, one column per KDE point): the mean of
    ``1 - Phi(z)`` over the columns when ``uniform``, otherwise its
    ``weights``-weighted sum clamped to [0, 1].
    """
    upper = 1.0 - torch.special.ndtr(z)
    if uniform:
        return upper.mean(dim=1)
    return (upper @ weights).clamp(0.0, 1.0)


class _OutlierGrid:
    """
    The CDF ``F(d) = sum_i w_i Phi((d - x_i) / h)`` of a one-dimensional KDE with
    points ``x_i``, weights ``w_i`` and bandwidth ``h``, tabulated in float64.

    ``F`` (1 minus ``_upper_tail``), ``F'`` and ``F''`` are tabulated at evenly
    spaced nodes at most ``h / NODES_PER_BANDWIDTH`` apart over
    ``[min x - REACH * h, max x + REACH * h]``. Between two nodes, ``F`` is the
    quintic Hermite interpolant that matches the three values at both nodes,
    clamped to [0, 1]. Below the first node it is 0, above ``max x + REACH * h``
    1, and at NaN it is NaN.
    """

    NODES_PER_BANDWIDTH = 32
    REACH = 9.0
    MAX_NODES = 2**13
    MAX_PAIRS = 2**27
    MAX_NODES_PER_ROW = 8
    WEIGHT_SUM_TOLERANCE = 1e-14

    def __init__(
        self, low: float, step: float, table: torch.Tensor, high: float
    ) -> None:
        self.low = low
        self.step = step
        self.high = high
        # One row per node: F, step * F' and step**2 * F''.
        self.table = table

    @classmethod
    def build(
        cls, points: torch.Tensor, weights: torch.Tensor, bandwidth: float, rows: int
    ) -> _OutlierGrid | None:
        """
        Tabulate the CDF of the KDE with these float64 ``points`` and ``weights``
        and this ``bandwidth``, for a call that scores ``rows`` distances with it.

        Returns
        -------
        _OutlierGrid | None
            The grid; None when the KDE has no points, a point or weight is NaN
            or infinite, a weight is negative, the weights are not all equal and
            their sum differs from 1 by more than ``WEIGHT_SUM_TOLERANCE``, the
            bandwidth is not a positive finite number, the grid would have zero
            width or zero node spacing, or it would need more than ``MAX_NODES``
            nodes, more than ``MAX_PAIRS`` (node, point) pairs or more than
            ``MAX_NODES_PER_ROW`` nodes per row.
        """
        if len(points) == 0 or not bandwidth > 0.0:
            return None
        if not (bool(torch.isfinite(weights).all()) and bool((weights >= 0.0).all())):
            return None
        uniform = bool((weights == weights[0]).all())
        if not uniform and abs(float(weights.sum()) - 1.0) > cls.WEIGHT_SUM_TOLERANCE:
            return None
        low = float(points.min()) - cls.REACH * bandwidth
        high = float(points.max()) + cls.REACH * bandwidth
        intervals = (high - low) / bandwidth * cls.NODES_PER_BANDWIDTH
        if not 0.0 < intervals <= cls.MAX_NODES - 1:
            return None
        nodes = math.ceil(intervals) + 1
        if nodes * len(points) > cls.MAX_PAIRS:
            return None
        if nodes > cls.MAX_NODES_PER_ROW * rows:
            return None
        step = (high - low) / (nodes - 1)
        if not step > 0.0:
            return None
        ratio = step / bandwidth
        offsets = points - low
        table = torch.empty(nodes, 3, dtype=torch.float64)
        # At most 2**22 (node, point) pairs at a time
        size = max(1, 2**22 // len(points))
        for start in range(0, nodes, size):
            stop = min(start + size, nodes)
            node_offsets = torch.arange(start, stop, dtype=torch.float64) * step
            z = (node_offsets[:, None] - offsets) / bandwidth
            density = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            curvature = -z * density
            if uniform:
                density, curvature = density.mean(dim=1), curvature.mean(dim=1)
            else:
                density, curvature = density @ weights, curvature @ weights
            table[start:stop, 0] = 1.0 - _upper_tail(z, weights, uniform)
            table[start:stop, 1] = ratio * density
            table[start:stop, 2] = ratio * ratio * curvature
        return cls(low, step, table, high)

    def __call__(self, distances: torch.Tensor) -> torch.Tensor:
        """``F`` at each of the float64 ``distances``."""
        last = len(self.table) - 1
        position = (distances - self.low) / self.step
        index = position.clamp(0.0, last - 1.0).nan_to_num().floor()
        t = (position - index).clamp(0.0, 1.0)
        start = self.table[index.long()]
        end = self.table[index.long() + 1]
        t2 = t * t
        t3 = t2 * t
        u = 1.0 - t
        value = (
            start[:, 0]
            + t3 * (10.0 - 15.0 * t + 6.0 * t2) * (end[:, 0] - start[:, 0])
            + (t - t3 * (6.0 - 8.0 * t + 3.0 * t2)) * start[:, 1]
            - t3 * (4.0 - 7.0 * t + 3.0 * t2) * end[:, 1]
            + 0.5 * t2 * u * u * u * start[:, 2]
            + 0.5 * t3 * u * u * end[:, 2]
        )
        value = torch.where(position < 0.0, 0.0, value)
        value = torch.where(distances > self.high, 1.0, value)
        return value.clamp(0.0, 1.0)


COV_REG_TYPE = "epsilon"
_COV_REG_TYPES = ("epsilon", "shared")


def _require_cov_reg_type(
    cov_reg_type: object,
    cov_type: object = None,
    pred_cov_type: object = None,
    *,
    name: str = "cov_reg_type",
    unknown_repair: str | None = None,
    unknown_unit_repair: str | None = (
        "pass cov_reg_type='epsilon', or 'shared' with a cov_type or pred_cov_type "
        "of CovType.DIAGONAL or CovType.FULL."
    ),
    unit_repair: str = (
        "pass a cov_type or pred_cov_type of CovType.DIAGONAL or CovType.FULL, or "
        "cov_reg_type='epsilon'."
    ),
) -> None:
    """
    Raise ``ValueError`` unless ``cov_reg_type`` is one of ``_COV_REG_TYPES``, or
    when it is ``"shared"`` and both ``cov_type`` and ``pred_cov_type`` are
    ``CovType.UNIT``. ``name`` names ``cov_reg_type`` in the message. For a value
    not in ``_COV_REG_TYPES`` the message ends with ``unknown_unit_repair`` when
    both types are UNIT and with ``unknown_repair`` otherwise (each when given);
    for ``"shared"`` with both types UNIT it ends with ``unit_repair``. The
    defaults are the constructor's repairs.
    """
    both_unit = cov_type is CovType.UNIT and pred_cov_type is CovType.UNIT
    if not (isinstance(cov_reg_type, str) and cov_reg_type in _COV_REG_TYPES):
        repair = unknown_unit_repair if both_unit else unknown_repair
        ending = "." if repair is None else f"; {repair}"
        raise ValueError(
            f"{name} must be one of {list(_COV_REG_TYPES)}; got "
            f"{_shown(cov_reg_type)}{ending}"
        )
    if cov_reg_type == "shared" and both_unit:
        raise ValueError(
            f"{name} is 'shared', which pools DIAGONAL or FULL covariances, and "
            f"cov_type and pred_cov_type are CovType.UNIT; {unit_repair}"
        )


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
        pred_cov_type: CovType | None = None,
        ood_cov_type: CovType | None = None,
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
            Covariance type of the class distances in training mode [unit, diag, full].
        cov_reg_type : str
            Type of regularization to use when generating the covariance matrix [epsilon, shared].
        epsilon : float
            Epsilon value to use for covariance regularization.
        device : str, optional
            The device to train the protonet model on (defaults to cpu).
        pred_cov_type : CovType, optional
            Covariance of the class distances in eval mode; None selects the module
            default ``PRED_COV_TYPE`` (DIAGONAL).
        ood_cov_type : CovType, optional
            Covariance of the distance to the global mean in the relative
            Mahalanobis OOD score; None selects the module default ``OOD_COV_TYPE``
            (DIAGONAL).
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.cov_type = cov_type
        self.cov_reg_type = cov_reg_type
        self.epsilon = epsilon
        self.emb_out_dim = emb_out_dim
        self.pred_cov_type: CovType = (
            PRED_COV_TYPE if pred_cov_type is None else pred_cov_type
        )
        self.ood_cov_type: CovType = (
            OOD_COV_TYPE if ood_cov_type is None else ood_cov_type
        )
        self.to(device)
        self.device = device

        self.support: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.support_embeddings: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.model_head: torch.nn.Module = self.create_model_head(emb_out_dim)
        self.model_head.to(device)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore a pickled Protonet; a pickle without ``pred_cov_type`` or
        ``ood_cov_type`` uses the module default for it.
        """
        state.setdefault("pred_cov_type", PRED_COV_TYPE)
        state.setdefault("ood_cov_type", OOD_COV_TYPE)
        super().__setstate__(state)

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
        return torch.stack([e.mean(dim=0) for e in self.support_embeddings.values()])

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
            Type of covariance to use: ``CovType.UNIT``, ``CovType.DIAGONAL`` or
            ``CovType.FULL``.
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
        ValueError
            If ``cov_type`` is not a ``CovType`` member; beartype refuses it
            first unless Python runs with ``-O``.
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
        else:  # pragma: no cover
            members = ", ".join(f"CovType.{member.name}" for member in CovType)
            values = ", ".join(repr(member.value) for member in CovType)
            raise ValueError(
                f"cov_type must be one of {members}; got {_shown(cov_type)}. "
                f"CovType(value) converts one of {values}."
            )

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
            Regularization type: ``"epsilon"`` or ``"shared"``. ``"shared"``
            applies to DIAGONAL and FULL covariances; a UNIT covariance is used as
            is.

        Returns
        -------
        OrderedDict[int, torch.Tensor]
            Dictionary containing the regularized class covariance matrices.

        Raises
        ------
        ValueError
            If ``cov_reg_type`` is not ``"epsilon"`` or ``"shared"``; beartype
            refuses a non-string first unless Python runs with ``-O``.
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
        else:
            _require_cov_reg_type(cov_reg_type)

        return class_cov_dict

    def compute_shared_covariance(
        self, class_cov_dict: OrderedDict[int, torch.Tensor], cov_type: CovType
    ) -> torch.Tensor:
        """
        Method to calculate a shared covariance matrix.

        The shared covariance is the within-class covariance pooled over the
        classes, each class weighted by its degrees of freedom: the sum over the
        classes of ``(n_k - 1) * C_k``, divided by ``N - K``. ``C_k`` is the
        covariance of class ``k``, ``n_k`` its number of support embeddings, ``N``
        the number of support embeddings of the classes in ``class_cov_dict``
        and ``K`` the number of those classes.

        Parameters
        ----------
        class_cov_dict : OrderedDict[int, torch.Tensor]
            A dictionary containing each class and the corresponding covariance
            matrix; every class must be in the support.
        cov_type : CovType
            Type of the covariances: ``CovType.DIAGONAL`` or ``CovType.FULL``.

        Returns
        -------
        torch.Tensor
            Tensor containing the shared covariance matrix.

        Raises
        ------
        ValueError
            If ``cov_type`` is ``CovType.UNIT``.
        ValueError
            If ``cov_type`` is not a ``CovType`` member; beartype refuses it
            first unless Python runs with ``-O``.
        ValueError
            If the model has no support embeddings.
        ValueError
            If a label in ``class_cov_dict`` has no support embeddings.
        ValueError
            If the classes in ``class_cov_dict`` have no more support embeddings
            than there are classes (``N - K < 1``), as for an empty
            ``class_cov_dict`` or classes of one support embedding each.
        """
        if cov_type == CovType.FULL:
            shared_covariance = torch.zeros(
                (self.emb_out_dim, self.emb_out_dim), device=self.device
            )
        elif cov_type == CovType.DIAGONAL:
            shared_covariance = torch.zeros(self.emb_out_dim, device=self.device)
        else:
            shown = (
                f"CovType.{cov_type.name}"
                if isinstance(cov_type, CovType)
                else _shown(cov_type)
            )
            raise ValueError(
                "compute_shared_covariance pools DIAGONAL or FULL covariances; got "
                f"{shown}. Pass CovType.DIAGONAL or CovType.FULL."
            )

        if not self.support_embeddings:
            raise ValueError(
                "compute_shared_covariance pools the covariances of the support "
                "classes, and the model has no support; call update_support first."
            )

        missing = [
            label for label in class_cov_dict if label not in self.support_embeddings
        ]
        if missing:
            raise ValueError(
                f"class_cov_dict has labels {missing}, which have no support "
                "embeddings; pass the covariances of the support classes "
                f"{list(self.support_embeddings)}."
            )

        total_support = 0
        for label, class_covariance in class_cov_dict.items():
            num_class_support = self.support_embeddings[label].shape[0]
            total_support += num_class_support
            shared_covariance = (
                shared_covariance + (num_class_support - 1) * class_covariance
            )

        if total_support - len(class_cov_dict) < 1:
            raise ValueError(
                "compute_shared_covariance divides by the number of support examples "
                f"minus the number of classes, {total_support} - {len(class_cov_dict)} "
                "here; pooling needs more support examples than classes; pass the "
                "covariances of classes with at least 2 support examples."
            )

        return shared_covariance / (total_support - len(class_cov_dict))

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
        mode: ``pred_cov_type`` in eval mode, ``cov_type`` in training mode.

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
                cov_type=self.pred_cov_type
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
            self.compute_covariance_by_type(self.ood_cov_type, embeddings), dim=0
        )
        global_reg_input = OrderedDict().fromkeys([0], torch.Tensor())
        global_reg_input[0] = self.global_covariance
        self.global_covariance: torch.Tensor = self.regularize_covariance(
            global_reg_input, self.ood_cov_type, "epsilon"
        )[0]
        self.global_mean: torch.Tensor = torch.mean(embeddings, dim=0)


def _require_examples_per_class(
    y: torch.Tensor,
    labels: list[int],
    part: str,
    minimum: int,
    where: str = "after the calib_frac split",
) -> None:
    """
    Refuse a class with too few examples on one side of a ``calib_frac`` split or
    in an explicit calibration set.

    Parameters
    ----------
    y : torch.Tensor
        Labels of one side.
    labels : list[int]
        Every class the model is given.
    part : str
        The side, named in the message.
    minimum : int
        The fewest examples each class needs on this side.
    where : str, optional
        Where the examples come from, named in the message.

    Raises
    ------
    ValueError
        If a class in ``labels`` has fewer than ``minimum`` examples in ``y``.
    """
    for label in labels:
        count = int((y == label).sum())
        if count < minimum:
            raise ValueError(
                f"Class {_label_text(label)} has {count} {part} example(s) {where}; "
                f"it needs at least {minimum}."
            )


def _require_calibration_set(
    support_x: torch.Tensor,
    labels: list[int],
    calib_x: torch.Tensor,
    calib_y: torch.Tensor,
) -> None:
    """
    Refuse an explicit calibration set that does not match the support.

    Raises
    ------
    ValueError
        If ``calib_y`` is not 1-D or not whole numbers, ``calib_x`` and ``calib_y``
        differ in length, ``calib_x``'s rows are shaped unlike ``support_x``'s, or
        ``calib_y`` holds a label that is not in ``labels``.
    """
    _require_whole_number_labels(calib_y, None, "calib_y")
    if len(calib_x) != len(calib_y):
        raise ValueError(
            f"calib_x has {len(calib_x)} rows and calib_y {len(calib_y)} labels; give "
            "one label per row."
        )
    if calib_x.shape[1:] != support_x.shape[1:]:
        raise ValueError(
            f"calib_x's rows have shape {tuple(calib_x.shape[1:])} and support_x's "
            f"{tuple(support_x.shape[1:])}; give calibration examples shaped like the "
            "support."
        )
    unknown = sorted(set(torch.unique(calib_y).tolist()) - set(labels))
    if unknown:
        raise ValueError(
            f"calib_y holds label {_label_text(unknown[0])}, which support_y does not; "
            "give calibration examples of the support's classes only."
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
        pred_cov_type: CovType | None = None,
        ood_cov_type: CovType | None = None,
        min_kde_samples: int | None = None,
        cov_reg_type: str = COV_REG_TYPE,
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
            The OOD statistic, by default ``"rmd_sq"``. With ``relative_mahal``:
            the squared Mahalanobis distance to the predicted class prototype
            minus the squared Mahalanobis distance to the mean of the support
            embeddings; without it, the Mahalanobis distance to the predicted
            class prototype. Both are computed without the temperature.
            ``"rmd_sqrt"``: with ``relative_mahal`` it subtracts the distances
            themselves, and with ``use_temperature`` it divides the class
            distance by the temperature first, in both modes. A model file
            without this setting loads as ``"rmd_sqrt"``. With
            ``relative_mahal``, the outlier-score KDEs are fitted for the
            statistic in use: after changing ``ood_distance``, refit them with
            ``update_support`` or ``train_model``. Without it, the KDEs are the
            same for both values.
        pred_cov_type : CovType, optional
            Covariance of the class distances at prediction time; None selects the
            module default ``PRED_COV_TYPE`` (DIAGONAL). UNIT uses Euclidean
            distances and accepts classes with a single support example.
        ood_cov_type : CovType, optional
            Covariance of the distance to the global mean in the relative
            Mahalanobis OOD score; None selects the module default ``OOD_COV_TYPE``
            (DIAGONAL).
        min_kde_samples : int, optional
            None (the default) fits one outlier-score KDE per class and refuses a
            class with fewer than 2 calibration examples or equal OOD distances.
            An int pools instead: a class with fewer than ``min_kde_samples``
            calibration examples, or whose distances are all equal (a single
            example always is), shares one KDE fitted on every calibration example.
        cov_reg_type : str, optional
            How the class covariances are regularized, by default ``"epsilon"``.
            ``"epsilon"`` adds ``DEFAULT_EPSILON`` to the variances of each
            DIAGONAL or FULL class covariance. ``"shared"`` also blends each
            DIAGONAL or FULL class covariance with the covariance pooled over the
            classes, and uses a UNIT covariance as is; it needs a DIAGONAL or FULL
            ``cov_type`` or ``pred_cov_type``.

        Raises
        ------
        ValueError
            If ``ood_distance`` is not ``"rmd_sq"`` or ``"rmd_sqrt"``, or
            ``min_kde_samples`` is not None or an int of at least 1 (a bool is
            refused), the rule ``load`` applies to a model file. If
            ``cov_reg_type`` is not ``"epsilon"`` or ``"shared"`` (beartype
            refuses a non-string first unless Python runs with ``-O``), or is
            ``"shared"`` with a UNIT ``cov_type`` and ``pred_cov_type``; ``load``
            refuses a model file with such settings.
        """
        if min_kde_samples is not None:
            _positive_int(min_kde_samples, "min_kde_samples")
        if ood_distance not in _OOD_DISTANCES:
            raise ValueError(
                f"ood_distance must be one of {list(_OOD_DISTANCES)}; got "
                f"{_shown(ood_distance)}."
            )
        _require_cov_reg_type(
            cov_reg_type,
            cov_type,
            PRED_COV_TYPE if pred_cov_type is None else pred_cov_type,
        )
        super().__init__(
            embedding_model,
            device=device,
            feature_names=feature_names,
            label_names=label_names,
        )
        self.cov_type = cov_type
        self.cov_reg_type = cov_reg_type
        self.relative_mahal = relative_mahal
        self.ood_distance = ood_distance
        self.min_kde_samples = min_kde_samples
        self.emb_out_dim = emb_out_dim
        self.epsilon = DEFAULT_EPSILON
        self.outlier_score_kde: OrderedDict[int, gaussian_kde] = OrderedDict()
        # Per label: the KDE's points, weights and bandwidth, and their grid.
        self._outlier_grids: dict[
            int, tuple[tuple[bytes, bytes, float], _OutlierGrid]
        ] = {}
        self.model_summary: dict[str, Any] = dict()
        self.use_temperature = use_temperature
        self.init_temperature = init_temperature
        self.register_buffer(
            "temperature", torch.Tensor(self.init_temperature * torch.ones(1))
        )

        self.model: Protonet = Protonet(
            embedding_model,
            self.emb_out_dim,
            self.cov_type,
            self.cov_reg_type,
            self.epsilon,
            device=device,
            pred_cov_type=pred_cov_type,
            ood_cov_type=ood_cov_type,
        )
        # Moves the temperature buffer onto the model's device.
        self.to(self.device)

    @property
    def pred_cov_type(self) -> CovType:
        """The covariance of the class distances at prediction time. Read-only."""
        return self.model.pred_cov_type

    @property
    def ood_cov_type(self) -> CovType:
        """The covariance of the distance to the global mean in the OOD score. Read-only."""
        return self.model.ood_cov_type

    @property
    def support(self) -> OrderedDict[int, torch.Tensor]:
        """
        The support examples of each class, keyed by label: the dict that
        ``get_support()`` returns, empty before the first support update.
        Read-only.
        """
        return self.model.support

    @property
    def support_embeddings(self) -> OrderedDict[int, torch.Tensor]:
        """
        The embeddings of the support examples, keyed by label; empty before the
        first support update. Read-only.
        """
        return self.model.support_embeddings

    @property
    def prototypes(self) -> torch.Tensor:
        """
        The class prototypes: the tensor that ``get_prototypes()`` returns, an
        empty tensor before the first support update. Read-only.
        """
        return self.model.__dict__.get("prototypes", torch.Tensor())

    def __getstate__(self) -> dict[str, Any]:
        """
        The state that pickling and ``copy.deepcopy`` copy: the instance dict
        without the tabulated outlier scores.
        """
        state = super().__getstate__()
        state.pop("_outlier_grids", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore a pickled model; a pickle without ``ood_distance`` uses
        ``"rmd_sqrt"``, and one without ``min_kde_samples`` uses None.
        ``support``, ``support_embeddings`` and ``prototypes`` entries in its
        instance dict are dropped: these names are read-only properties. The model
        starts with no tabulated outlier scores.
        """
        state.setdefault("ood_distance", "rmd_sqrt")
        state.setdefault("min_kde_samples", None)
        for name in ("support", "support_embeddings", "prototypes"):
            state.pop(name, None)
        state["_outlier_grids"] = {}
        super().__setstate__(state)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generates class probabilities based on the input tensor.

        Computes in the mode the caller left the model in (via
        ``train()``/``eval()``), with autograd.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor for generating predictions.

        Returns
        -------
        torch.Tensor
            Class probabilities: the softmax of the negated distances to the
            class prototypes, with the distances divided by ``temperature``
            first when ``use_temperature`` is True. In eval mode, equal to
            ``predict(X).classes``.

        Raises
        ------
        ValueError
            If the model has no support, or, in eval mode, the embeddings of
            ``X`` hold a NaN or infinite value.
        """
        preds, dists = self.model(X)
        if self.use_temperature:
            preds = torch.softmax(torch.negative(dists / self.temperature), dim=1)
        return preds

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
            Input pytorch TensorDataset of training data for model, with at least
            2 classes, its inputs of shape (examples, features...).
        num_episodes : int
            The desired number of episodes to use for training.
        calib_frac : float, optional
            Fraction of given training data to reserve for model calibration,
            strictly between 0 and 1, by default 0.2.
        support_size : int, optional
            Number of support examples to generate for each class, by default 25.
        way : int, optional
            Number of classes to train on per episode, from 2 to the number of
            classes in ``dataset`` when ``num_episodes`` is positive, by default 3.
        episode_size : int, optional
            Upper bound on the number of query examples per episode, each class
            contributing up to ``episode_size // way``; at least ``way`` when
            ``num_episodes`` is positive, by default 100.
        loss_fn : Callable, optional
            A pytorch loss function, eg., torch.nn.CrossEntropyLoss(), by default torch.nn.functional.cross_entropy.
        opt_class : Callable, optional
            A pytorch optimizer, e.g., torch.optim.Adam, by default torch.optim.Adam.
        num_calibration_epochs : int, optional
            The desired number of epochs to use for temperature scaling, at least 0
            (0 leaves the temperature at ``init_temperature``), by default 2.
        calibration_lr : float, optional
            Learning rate for temperature scaling, a finite float above 0, by
            default 0.01.

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
            If ``support_size`` is below 2; ``calib_frac`` is not strictly
            between 0 and 1; ``num_calibration_epochs`` is not an int of at least
            0 (a bool is refused) or ``calibration_lr`` is not a finite float
            above 0, whether or not ``use_temperature`` is set; the inputs of
            ``dataset`` have fewer than 2 dimensions; ``dataset`` has fewer than 2
            classes, or its labels are not 1-D or not whole numbers; if
            ``num_episodes`` is positive and ``way`` is not between 2 and the
            number of classes, or ``episode_size`` is below ``way``; or if the
            ``calib_frac`` split leaves a class with fewer than 2 calibration
            examples (with ``min_kde_samples`` None; otherwise such a class shares
            the pooled KDE) or too few training examples (more than ``support_size``
            when ``num_episodes`` is positive, ``support_size`` otherwise); the
            model is then unchanged. An ``int`` ``calib_frac``, or an ``int`` or
            bool ``calibration_lr``, raises it under ``python -O`` only. Also, after
            training, if a calibration example's embedding or OOD distance holds a
            NaN or infinite value, or a class's calibration examples all have the
            same OOD distance; the model is then trained but its outlier scores are
            not refitted.
        beartype.roar.BeartypeCallHintParamViolation
            If ``calib_frac`` or ``calibration_lr`` is not a float (an ``int``
            included, and a bool ``calibration_lr``), or ``num_calibration_epochs``
            is not an int, while contracts are on.
        """
        if support_size < 2:
            raise ValueError(
                f"support_size must be at least 2, got {support_size}: the support's "
                "diagonal covariance needs 2 examples per class."
            )
        _require_fraction(calib_frac, "calib_frac")
        _require_calibration_settings(num_calibration_epochs, calibration_lr)
        X, Y = dataset[:]
        _require_rows(X, "dataset inputs")
        labels = torch.unique(Y).tolist()
        if len(labels) < 2:
            raise ValueError(
                f"dataset must hold at least 2 classes, got {len(labels)}."
            )
        _require_whole_number_labels(Y, labels, "dataset labels")
        if num_episodes > 0:
            if not 2 <= way <= len(labels):
                raise ValueError(
                    f"way must be between 2 and the number of classes ({len(labels)}) "
                    f"when num_episodes is positive, got {way}."
                )
            if episode_size < way:
                raise ValueError(
                    f"episode_size must be at least way ({way}) when num_episodes "
                    f"is positive, got {episode_size}."
                )

        self.validate_feature_label_names(X.shape[-1], len(labels))

        train_x, calib_x, train_y, calib_y = stratified_train_test_split(
            X, Y, test_size=calib_frac
        )
        episode_queries = 1 if num_episodes > 0 else 0
        _require_examples_per_class(
            train_y, labels, "training", support_size + episode_queries
        )
        if self.min_kde_samples is None:
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

        sampler = _EpisodeSampler(train_x, train_y) if num_episodes > 0 else None
        for i in tqdm(range(num_episodes)):
            optimizer.zero_grad()

            support, episode_x, episode_y = cast(_EpisodeSampler, sampler).sample(
                support_size, way, episode_size
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
        Fine-tune the temperature on the given examples. After each optimizer step
        the temperature is clamped to at least ``MIN_TEMPERATURE`` (1e-3).
        ``train_model`` calls this method when ``use_temperature`` is True.

        Parameters
        ----------
        calib_x : torch.Tensor
            Training data to be used for temperature calibration.
        calib_y : torch.Tensor
            Class labels corresponding to `calib_x`, each one of the support's
            labels.
        num_calibration_epochs : int, optional
            Number of epochs to tune temperature, at least 0 (0 leaves the
            temperature unchanged), by default 1.
        calibration_lr : float, optional
            Learning rate for temperature optimization, a finite float above 0,
            by default 0.01.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``num_calibration_epochs`` is not an int of at least 0 (a bool is
            refused) or ``calibration_lr`` is not a finite float above 0 (an
            ``int`` or bool ``calibration_lr`` raises it under ``python -O``
            only), or the model has no support, before anything changes; or if
            `calib_y` holds a label that is not in the support, or, in eval mode,
            the embeddings of ``calib_x`` hold a NaN or infinite value.
        beartype.roar.BeartypeCallHintParamViolation
            If ``num_calibration_epochs`` is not an int, or ``calibration_lr`` is
            not a float (an ``int`` or bool included), while contracts are on.
        """
        _require_calibration_settings(num_calibration_epochs, calibration_lr)
        if len(self.model.support) == 0:
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
            with torch.no_grad():
                self.temperature.clamp_(min=MIN_TEMPERATURE)
        self.temperature.requires_grad = False

    @icontract.ensure(lambda self: len(self.model.support_embeddings) > 0)
    def _fit_outlier_scores(
        self, ood_dists: torch.Tensor, calib_y: torch.Tensor
    ) -> None:
        """
        Private function to fit outlier scores with a kernel density estimate (KDE).

        Replaces ``outlier_score_kde`` with one KDE per support label, keyed
        by label, in support order, and drops the tabulated outlier scores.

        Parameters
        ----------
        ood_dists : torch.Tensor
            Tensor of computed OOD distances.
        calib_y : torch.Tensor
            Tensor of class labels for `ood_dists` examples. Every class needs
            at least 2 examples when ``min_kde_samples`` is None.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a class's OOD distances include a NaN or infinite value or are all
            equal (with ``min_kde_samples`` None), or fitting a KDE fails; no KDE is
            changed then.
        """
        if self.min_kde_samples is not None:
            self._fit_pooled_outlier_scores(ood_dists, calib_y)
            return
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
        self._outlier_grids = {}

    def _fit_pooled_outlier_scores(
        self, ood_dists: torch.Tensor, calib_y: torch.Tensor
    ) -> None:
        """
        Fit one KDE per class with enough calibration examples, and share one KDE
        fitted on every calibration example among the rest (``min_kde_samples``).

        Raises
        ------
        ValueError
            If an OOD distance is NaN or infinite, or there are fewer than 2
            calibration examples or all have the same OOD distance; no KDE is
            changed then.
        """
        if not bool(torch.isfinite(ood_dists).all()):
            raise ValueError(
                "A calibration example has a NaN or infinite OOD distance; fitting "
                "the outlier scores needs finite values. Check the embedding model's "
                "output for the calibration examples."
            )
        if len(ood_dists) < 2 or bool((ood_dists == ood_dists[0]).all()):
            raise ValueError(
                "The pooled outlier-score KDE needs at least 2 distinct calibration "
                "OOD distances; give more calibration examples with different OOD "
                "distances (calib_x and calib_y, or a larger calib_frac)."
            )
        assert self.min_kde_samples is not None
        pooled = gaussian_kde(ood_dists.cpu().detach().numpy())
        fitted: OrderedDict[int, gaussian_kde] = OrderedDict()
        for label in self.model.support_embeddings.keys():
            dists = ood_dists[calib_y == int(label)]
            if len(dists) >= self.min_kde_samples and not bool(
                (dists == dists[0]).all()
            ):
                fitted[label] = gaussian_kde(dists.cpu().detach().numpy())
            else:
                fitted[label] = pooled
        self.outlier_score_kde = fitted

    def _compute_outlier_scores(self, ood_dists, predictions) -> torch.Tensor:
        """
        Private function to compute OOD scores using the calculated kernel density estimate (KDE).

        Each example is scored with the KDE of its predicted label: column
        ``j`` of ``predictions`` is the ``j``-th support label. The score is the
        KDE's CDF at the example's distance, computed in float64 on the CPU and
        cast to the dtype and device of ``ood_dists``. A label's KDE is read
        from the grid kept for the label when that grid was built from the KDE's
        current points, weights and bandwidth. Otherwise ``_OutlierGrid.build``
        is given the number of examples predicted as the label: the grid it
        returns is kept and read, and when it returns None the KDE is summed
        directly.

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
        dists = ood_dists.detach().cpu().double()
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
            data = torch.from_numpy(kde.dataset[0]).double()
            weights = torch.from_numpy(kde.weights).double()
            bandwidth = float(np.sqrt(kde.covariance[0, 0]))
            grid = self._outlier_grid(label, data, weights, bandwidth, len(rows))
            if grid is not None:
                ood_scores[rows] = grid(dists[rows])
                continue
            uniform = bool((weights == weights[0]).all())
            # 1 minus the KDE's mass above each distance, for at most 2**22
            # (distance, KDE point) pairs at a time
            for chunk in rows.split(max(1, 2**22 // len(data))):
                z = (dists[chunk, None] - data) / bandwidth
                ood_scores[chunk] = 1.0 - _upper_tail(z, weights, uniform)

        return ood_scores.to(dtype=ood_dists.dtype, device=ood_dists.device)

    def _outlier_grid(
        self,
        label: int,
        points: torch.Tensor,
        weights: torch.Tensor,
        bandwidth: float,
        rows: int,
    ) -> _OutlierGrid | None:
        """
        The tabulated CDF of the outlier-score KDE of ``label``, which has these
        float64 ``points`` and ``weights`` and this ``bandwidth``.

        Returns the grid kept for ``label`` when it was built from the same
        values; otherwise builds one with ``_OutlierGrid.build`` for ``rows``
        distances and keeps it.
        Returns None, and keeps nothing for ``label``, when ``build`` returns
        None.
        """
        grids = self._outlier_grids
        key = (points.numpy().tobytes(), weights.numpy().tobytes(), bandwidth)
        kept = grids.get(label)
        if kept is not None and kept[0] == key:
            return kept[1]
        grid = _OutlierGrid.build(points, weights, bandwidth, rows)
        if grid is None:
            grids.pop(label, None)
        else:
            grids[label] = (key, grid)
        return grid

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

        The OOD scores are read from a table of the CDF of each predicted class's
        outlier-score KDE. Every call reads a class's table while the class's KDE
        keeps the points, weights and bandwidth the table was built from; a table
        is not read after the class's KDE in ``outlier_score_kde`` changes. A
        call that predicts a class with no table for those values builds one when
        it predicts the class for at least one eighth as many examples as the
        table has nodes (1,024 examples always suffice), and that call takes
        longer than later calls; a call with fewer examples of the class sums its
        KDE over its points. An example's score can differ by up to 1e-13
        depending on whether an earlier call built its class's table. A KDE is
        summed over its points on every call instead when its table would need
        more than 8,192 nodes or more than 2**27 (node, point) pairs (more than
        16,384 points at 8,192 nodes), when it has a NaN or infinite point or
        weight, a negative weight, unequal weights that do not sum to 1, or a
        bandwidth that is not a positive finite number, or when its points are
        all equal and its bandwidth is too small to separate two nodes.

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
                # "rmd_sq" reads the unscaled distances; "rmd_sqrt" reads the
                # scaled ones.
                if self.ood_distance == "rmd_sqrt":
                    dists = scaled
            ood_dist = self._compute_ood_dist(X_embed, preds, dists)
            ood_scores = self._compute_outlier_scores(ood_dist, preds)

        self.validate_feature_label_names(X.shape[-1], preds.shape[-1])

        return EquineOutput(classes=preds, ood_scores=ood_scores, embeddings=X_embed)

    def update_support(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        *args: Any,
        calib_frac: float | None = None,
        label_names: list[str] | None = None,
        calib_x: torch.Tensor | None = None,
        calib_y: torch.Tensor | None = None,
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
            Tensor containing labels for given support examples, of at least 2
            classes.
        *args : Any
            Options passed by position are refused with a ``TypeError``; pass them
            by keyword.
        calib_frac : float
            Fraction of given support data to use for OOD calibration, strictly
            between 0 and 1. Required, keyword-only, unless ``calib_x`` is given.
        label_names : list[str], optional
            List of strings of the names of the labels (ex ["streaming", "voip", ...]).
            Keyword-only.
        calib_x : torch.Tensor, optional
            An explicit calibration set for the OOD densities, keyword-only. When
            given, all of ``support_x`` is the support and no ``calib_frac`` split is
            made.
        calib_y : torch.Tensor, optional
            Labels of ``calib_x``; given exactly when ``calib_x`` is.

        Raises
        ------
        TypeError
            If ``calib_frac`` is missing (and no ``calib_x`` is given), or an option
            is passed by position.
        ValueError
            If ``calib_frac`` is given and not strictly between 0 and 1 (an
            ``int`` ``calib_frac`` raises it under ``python -O`` only); if
            ``support_x`` or ``support_y`` has no row dimension, or ``support_y``
            does not hold one label per row of ``support_x``; if ``support_y`` has
            fewer than 2 classes, is not 1-D or holds a label that is not a whole
            number; if only one of ``calib_x`` and ``calib_y`` is given; if
            ``calib_y`` is not 1-D, not whole numbers, not
            one label per ``calib_x`` row or holds a label ``support_y`` does not,
            or ``calib_x``'s rows are shaped unlike ``support_x``'s; if a class has
            fewer than 2 support examples (1 with a UNIT ``pred_cov_type``) or, with
            ``min_kde_samples`` None, fewer than 2 calibration examples; or if a
            calibration example's embedding or OOD distance holds a NaN or infinite
            value, or a class's calibration examples all have the same OOD
            distance. A failed update leaves the model as it was.
        beartype.roar.BeartypeCallHintParamViolation
            If ``calib_frac`` is not a float or None (an ``int`` included), while
            contracts are on.
        """
        if args:
            raise TypeError(
                "EquineProtonet.update_support() takes calib_frac and label_names by "
                f"keyword only, but {len(args)} option(s) were given by position: "
                "call update_support(support_x, support_y, calib_frac=...), adding "
                "label_names=... by keyword."
            )
        if (calib_x is None) != (calib_y is None):
            raise ValueError("Give both calib_x and calib_y, or neither.")
        if calib_frac is None and calib_x is None:
            raise TypeError(
                "EquineProtonet.update_support() missing required keyword argument: "
                "'calib_frac'"
            )
        if calib_frac is not None:
            _require_fraction(calib_frac, "calib_frac")
        _require_same_length(support_x, support_y, "support_x", "support_y")
        with _eval_mode(self):
            all_labels = torch.unique(support_y).tolist()
            if len(all_labels) < 2:
                raise ValueError(
                    f"support_y must hold at least 2 classes, got {len(all_labels)}."
                )
            _require_whole_number_labels(support_y, all_labels, "support_y")
            if calib_x is None or calib_y is None:
                assert calib_frac is not None
                support_x, calib_x, support_y, calib_y = stratified_train_test_split(
                    support_x, support_y, test_size=calib_frac
                )
                where = ("after the calib_frac split",) * 2
            else:
                _require_calibration_set(support_x, all_labels, calib_x, calib_y)
                where = ("in support_y", "in calib_y")
            min_support = 1 if self.model.pred_cov_type == CovType.UNIT else 2
            _require_examples_per_class(
                support_y, all_labels, "support", min_support, where[0]
            )
            if self.min_kde_samples is None:
                _require_examples_per_class(
                    calib_y, all_labels, "calibration", 2, where[1]
                )
            # Restored if a later step raises.
            state, model_state = dict(vars(self)), dict(vars(self.model))
            try:
                labels = torch.unique(support_y)
                if label_names is not None:
                    self.label_names = label_names
                self.validate_feature_label_names(support_x.shape[-1], labels.shape[0])

                shuffled_idxs = _get_shuffle_idxs_by_class(support_y, labels.tolist())
                support = OrderedDict(
                    (int(label), support_x[idxs])
                    for label, idxs in shuffled_idxs.items()
                )

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

    def get_support(self) -> OrderedDict[int, torch.Tensor]:
        """
        Get the support examples for the model.

        Returns
        -------
        OrderedDict[int, torch.Tensor]
            The support examples for the model.

        Raises
        ------
        ValueError
            If the model has no support.
        """
        _require_support(
            self.model.support, "EquineProtonet.get_support", "train_model"
        )
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
        return list(self.model.support_embeddings)

    def get_prototypes(self) -> torch.Tensor:
        """
        Get the prototypes for the model (the class means of the support embeddings).

        Returns
        -------
        torch.Tensor
            The prototpes for the model.

        Raises
        ------
        ValueError
            If the model has no support.
        """
        _require_support(
            self.model.support, "EquineProtonet.get_prototypes", "train_model"
        )
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
            If the embedding model is not a registered architecture, or the model
            holds a value a model file does not store (see ``equine._model_file``).
        ValueError
            If ``model.model.cov_reg_type``, set after construction, is not
            ``"epsilon"`` or ``"shared"``, or is ``"shared"`` while ``cov_type``
            and ``pred_cov_type`` are both UNIT; nothing is written.
        RuntimeError
            If the model has no support examples (neither ``train_model`` nor
            ``update_support`` has run).
        """
        if len(self.model.support) == 0:
            raise RuntimeError(
                "The model has no support examples to save; call train_model or "
                "update_support first."
            )
        _require_cov_reg_type(
            self.model.cov_reg_type,
            self.cov_type,
            self.model.pred_cov_type,
            name="model.model.cov_reg_type",
            unknown_repair="set it to 'epsilon' or 'shared', then save.",
            unknown_unit_repair="set model.model.cov_reg_type to 'epsilon', then save.",
            unit_repair="set model.model.cov_reg_type to 'epsilon', then save.",
        )
        _model_file.write(path, self._to_checkpoint())

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
        # pred_cov_type, ood_cov_type and cov_reg_type are written only when not
        # the default.
        if self.model.pred_cov_type != PRED_COV_TYPE:
            model_settings["pred_cov_type"] = self.model.pred_cov_type.value
        if self.model.ood_cov_type != OOD_COV_TYPE:
            model_settings["ood_cov_type"] = self.model.ood_cov_type.value
        if self.model.cov_reg_type != COV_REG_TYPE:
            model_settings["cov_reg_type"] = self.model.cov_reg_type
        if self.min_kde_samples is not None:
            model_settings["min_kde_samples"] = self.min_kde_samples
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
    ) -> Self:
        """
        Load a previously saved EquineProtonet model.

        Loading runs no code from the file: it is a safetensors file (see
        ``utils.load_checkpoint``) and the embedding model is rebuilt from its recipe
        through the architecture registry.

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
            on MPS: load such a file with ``device="cpu"``). If the file's
            support holds one class. If the file's ``cov_reg_type`` is not
            ``"epsilon"`` or ``"shared"``, or is ``"shared"`` with a UNIT
            ``cov_type`` and ``pred_cov_type`` (a file without ``cov_reg_type``
            loads as ``"epsilon"``).
        """
        model_save = _load_checkpoint_for_device(path, device)
        return cls._from_checkpoint(model_save, device, embedding_model=embedding_model)

    @classmethod
    def _from_checkpoint(
        cls,
        model_save: dict[str, Any],
        device: str | None = None,
        embedding_model: torch.nn.Module | None = None,
    ) -> Self:
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
        if len(support) == 1:
            raise ValueError(
                "Model file's support holds one class; an EquineProtonet needs at "
                "least 2."
            )
        file_settings = model_save["settings"]
        pred_cov_type = (
            file_settings.get("pred_cov_type", PRED_COV_TYPE.value)
            if isinstance(file_settings, dict)
            else PRED_COV_TYPE.value
        )
        min_support = 1 if pred_cov_type == CovType.UNIT.value else 2
        for label, rows in support.items():
            if rows.shape[0] < min_support:
                raise ValueError(
                    f"Model file's support holds {rows.shape[0]} example(s) of class "
                    f"{label}; an EquineProtonet with pred_cov_type "
                    f"{_shown(pred_cov_type)} needs at least {min_support} per class. "
                    "Save the model again, or upgrade the original file again with "
                    "`python -m equine.upgrade`."
                )
        outlier_kde = _outlier_kde_from_file(model_save["outlier_kde"])
        for key in ("feature_names", "label_names"):
            _require_names(model_save.get(key), key)
        head_save = model_save["model_head_save"]

        settings = _checked_settings(cls, model_save["settings"])
        # The model is built from the file's settings and then gets the file's
        # ood_distance, or "rmd_sqrt" for a file without it.
        ood_distance = settings.get("ood_distance", "rmd_sqrt")
        # emb_out_dim must be an int >= 1, not a bool.
        _positive_int(
            settings.get("emb_out_dim"), "Model file's settings['emb_out_dim']"
        )
        for key in ("cov_type", "pred_cov_type", "ood_cov_type"):
            if key not in settings:
                continue
            cov_type = settings[key]
            try:
                settings[key] = CovType(cov_type)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"Model file's settings['{key}'] ({_shown(cov_type)}) is not "
                    f"one of {[member.value for member in CovType]}; the file was not "
                    "written by save() or was changed after. Save the model again, or "
                    "upgrade the original file again with `python -m equine.upgrade`."
                ) from err
        if settings.get("min_kde_samples") is not None:
            try:
                _positive_int(
                    settings["min_kde_samples"],
                    "Model file's settings['min_kde_samples']",
                )
            except ValueError as err:
                raise ValueError(
                    f"{err} Save the model again, or upgrade the original file again "
                    "with `python -m equine.upgrade`."
                ) from err
        if "cov_reg_type" in settings:
            file_repair = (
                "the file was not written by save() or was changed after. Save the "
                "model again, or upgrade the original file again with "
                "`python -m equine.upgrade`."
            )
            _require_cov_reg_type(
                settings["cov_reg_type"],
                settings.get("cov_type", CovType.UNIT),
                settings.get("pred_cov_type", PRED_COV_TYPE),
                name="Model file's settings['cov_reg_type']",
                unknown_repair=file_repair,
                unknown_unit_repair=file_repair,
                unit_repair=file_repair,
            )
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
                probe = eq_model.model.compute_embeddings(
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
        eq_model.train_summary = cast(dict[str, Any], model_save.get("train_summary"))

        return eq_model
