# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import math
import sys
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any, cast

import icontract
import torch
from beartype import beartype
from torch.utils.data import DataLoader, Dataset
from torchmetrics.metric import Metric
from tqdm import tqdm

from .equine import Equine, EquineOutput, _eval_mode
from .registry import _describe_error, _positive_int, _truncate
from .utils import (
    EQUINE_FORMAT_VERSION,
    _adopt_stored_dtypes,
    _checked_settings,
    _embedding_checkpoint,
    _get_shuffle_idxs_by_class,
    _input_to_model,
    _load_checkpoint_for_device,
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
    _support_from_file,
    generate_support,
    generate_train_summary,
)

BatchType = tuple[torch.Tensor, ...]
# -------------------------------------------------------------------------------
# Note that the below code for
# * `_random_ortho`,
# * `_RandomFourierFeatures``, and
# * `_Laplace`
# is copied and modified from https://github.com/y0ast/DUE/blob/main/due/sngp.py
# under its original MIT license, redisplayed here:
# -------------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2021 Joost van Amersfoort
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------
# Following the recommendation of their README at https://github.com/y0ast/DUE
# we encourage anyone using this code in their research to cite the following papers:
#
# @article{van2021on,
#   title={On Feature Collapse and Deep Kernel Learning for Single Forward Pass Uncertainty},
#   author={van Amersfoort, Joost and Smith, Lewis and Jesson, Andrew and Key, Oscar and Gal, Yarin},
#   journal={arXiv preprint arXiv:2102.11409},
#   year={2021}
# }
#
# @article{liu2020simple,
#  title={Simple and principled uncertainty estimation with deterministic deep learning via distance awareness},
#  author={Liu, Jeremiah and Lin, Zi and Padhy, Shreyas and Tran, Dustin and Bedrax Weiss, Tania and Lakshminarayanan, Balaji},
#  journal={Advances in Neural Information Processing Systems},
#  volume={33},
#  pages={7498--7512},
#  year={2020}
# }


@beartype
def _random_ortho(n: int, m: int) -> torch.Tensor:
    """
     Generate a random orthonormal matrix.

     Parameters
     ----------
     n : int
         The number of rows.
    m : int
         The number of columns.

     Returns
     -------
     torch.Tensor
         The random orthonormal matrix.
    """
    q, _ = torch.linalg.qr(torch.randn(n, m))
    return q


@beartype
class _RandomFourierFeatures(torch.nn.Module):
    """
    A private class to generate random Fourier features for the embedding model.
    """

    def __init__(
        self, in_dim: int, num_random_features: int, feature_scale: float | None
    ) -> None:
        """
        Initialize the _RandomFourierFeatures module, which generates random Fourier features
        for the embedding model.

        Parameters
        ----------
        in_dim : int
            The input dimensionality.
        num_random_features : int
            The number of random Fourier features to generate.
        feature_scale : float | None
            The scaling factor for the random Fourier features. If None, defaults to sqrt(num_random_features / 2).
        """
        super().__init__()
        if feature_scale is None:
            feature_scale = math.sqrt(num_random_features / 2)

        self.register_buffer("feature_scale", torch.tensor(feature_scale))

        if num_random_features <= in_dim:
            W: torch.Tensor = _random_ortho(in_dim, num_random_features)
        else:
            # generate blocks of orthonormal rows which are not necessarily orthonormal
            # to each other.
            dim_left = num_random_features
            ws = []
            while dim_left > in_dim:
                ws.append(_random_ortho(in_dim, in_dim))
                dim_left -= in_dim
            ws.append(_random_ortho(in_dim, dim_left))
            W: torch.Tensor = torch.cat(ws, 1)

        feature_norm = torch.randn(W.shape) ** 2

        W = W * feature_norm.sum(0).sqrt()
        self.register_buffer("W", W)

        b: torch.Tensor = torch.empty(num_random_features).uniform_(0, 2 * math.pi)
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the _RandomFourierFeatures module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, in_dim).

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, num_random_features).
        """
        k = torch.cos(x @ self.W + self.b)
        k = k / self.feature_scale

        return k


def _inverse_via_cholesky(a: torch.Tensor) -> torch.Tensor:
    """
    Invert the symmetric positive-definite ``a`` through its Cholesky factor.

    Raises
    ------
    torch.linalg.LinAlgError
        If ``a`` is not positive definite.
    """
    u, info = torch.linalg.cholesky_ex(a)
    if not bool((info == 0).all()):
        raise torch.linalg.LinAlgError("Precision matrix inversion failed!")  # type: ignore
    return torch.cholesky_inverse(u)


def _sync_seen_count(module: "_Laplace", incompatible_keys: Any) -> None:
    """``load_state_dict`` post hook: the Python ``_seen_count`` follows the loaded ``seen_data``."""
    module._seen_count = int(module.seen_data)


class _Laplace(torch.nn.Module):
    """
    A private class to compute a Laplace approximation to a Gaussian Process (GP)
    """

    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        num_deep_features: int,
        num_gp_features: int,
        normalize_gp_features: bool,
        num_random_features: int,
        num_outputs: int,
        feature_scale: float | None,
        mean_field_factor: float | None,  # required for classification problems
        ridge_penalty: float = 1.0,
    ) -> None:
        """
        Initialize the _Laplace module.

        Parameters
        ----------
        feature_extractor : torch.nn.Module
            The feature extractor module.
        num_deep_features : int
            The number of features output by the feature extractor.
        num_gp_features : int
            The number of features to use in the Gaussian process.
        normalize_gp_features : bool
            Whether to normalize the GP features.
        num_random_features : int
            The number of random Fourier features to use.
        num_outputs : int
            The number of outputs of the model.
        feature_scale : float | None
            The scaling factor for the random Fourier features.
        mean_field_factor : float | None
            The mean-field factor for the Gaussian-Softmax approximation.
            Required for classification problems.
        ridge_penalty : float, optional
            The ridge penalty for the Laplace approximation.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.mean_field_factor = mean_field_factor
        self.ridge_penalty = ridge_penalty
        self.train_batch_size = 0  # to be set later

        if num_gp_features > 0:
            self.num_gp_features = num_gp_features
            random_matrix: torch.Tensor = torch.normal(
                0, 0.05, (num_gp_features, num_deep_features)
            )
            self.register_buffer("random_matrix", random_matrix)
        else:
            self.num_gp_features: int = num_deep_features
        self._has_projection: bool = num_gp_features > 0

        self.normalize_gp_features = normalize_gp_features
        if normalize_gp_features:
            self.normalize: torch.nn.LayerNorm = torch.nn.LayerNorm(num_gp_features)

        self.rff: _RandomFourierFeatures = _RandomFourierFeatures(
            num_gp_features, num_random_features, feature_scale
        )
        self.beta: torch.nn.Linear = torch.nn.Linear(num_random_features, num_outputs)

        self.num_data = 0  # to be set later
        self.register_buffer("seen_data", torch.tensor(0))
        # A Python mirror of seen_data, read back from the buffer after load_state_dict.
        self._seen_count: int = 0
        self.register_load_state_dict_post_hook(_sync_seen_count)

        precision = torch.eye(num_random_features) * self.ridge_penalty
        self.register_buffer("precision", precision)

        self.recompute_covariance = True
        self.register_buffer("covariance", torch.eye(num_random_features))
        self.training_parameters_set = False

    def jl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features onto the GP feature space.

        Parameters
        ----------
        x : torch.Tensor
            The feature-extractor output.

        Returns
        -------
        torch.Tensor
            The projected features, or ``x`` unchanged when no
            projection was built.
        """
        if self._has_projection:
            return torch.nn.functional.linear(x, self.random_matrix)
        return x

    def reset_precision_matrix(self) -> None:
        """
        Reset the precision matrix to the identity matrix times the ridge penalty.
        """
        identity = torch.eye(self.precision.shape[0], device=self.precision.device)
        self.precision: torch.Tensor = identity * self.ridge_penalty
        self.seen_data: torch.Tensor = torch.tensor(0, device=self.precision.device)
        self._seen_count = 0
        self.recompute_covariance = True

    @icontract.require(lambda num_data: num_data > 0)
    @icontract.require(
        lambda num_data, batch_size: (0 < batch_size) & (batch_size <= num_data)
    )
    def set_training_params(self, num_data: int, batch_size: int) -> None:
        """
        Set the training parameters for the Laplace approximation.

        Parameters
        ----------
        num_data : int
            The total number of data points.
        batch_size : int
            The batch size to use during training.
        """
        self.num_data: int = num_data
        self.train_batch_size: int = batch_size
        self.training_parameters_set: bool = True

    @icontract.require(lambda mean_field_factor: mean_field_factor is not None)
    def mean_field_logits(
        self, logits: torch.Tensor, pred_cov: torch.Tensor, mean_field_factor: float
    ) -> torch.Tensor:
        """
        Compute the mean-field logits for the Gaussian-Softmax approximation.

        Parameters
        ----------
        logits : torch.Tensor
            The logits tensor of shape (batch_size, num_outputs).
        pred_cov : torch.Tensor
            The predicted covariance matrix of shape (batch_size, batch_size).
        mean_field_factor : float
            Diagonal scaling factor

        Returns
        -------
        torch.Tensor
            The mean-field logits tensor of shape (batch_size, num_outputs).
        """
        # Mean-Field approximation as alternative to MC integration of Gaussian-Softmax
        # Based on: https://arxiv.org/abs/2006.07584

        logits_scale = torch.sqrt(1.0 + torch.diag(pred_cov) * mean_field_factor)
        if mean_field_factor > 0:
            logits = logits / logits_scale.unsqueeze(-1)

        return logits

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the forward pass of the Laplace approximation to the Gaussian Process.

        Requires ``set_training_params`` to have been called.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, num_features).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            The logits of shape (batch_size, num_outputs). In training mode
            they are the raw GP mean; in evaluation mode they are mean-field
            adjusted with the predictive covariance when ``mean_field_factor``
            is set (always, in equine). Only when ``mean_field_factor`` is
            ``None`` does evaluation mode return the tuple ``(mean, pred_cov)``
            instead, with ``pred_cov`` of shape (batch_size, batch_size).
        """
        return self._forward_with_features(x)[0]

    @icontract.require(lambda self: self.training_parameters_set)
    def _forward_with_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        ``forward`` that also returns the random Fourier features it computed.

        The features are the output of ``rff`` (feature_extractor -> jl -> normalize ->
        rff), the same tensor ``EquineGP.compute_embeddings`` returns, except that a
        single unbatched example gets a batch dimension of one here. The
        ``training_parameters_set`` precondition is checked here, before anything is
        computed or the precision/seen_data buffers are touched.

        Returns
        -------
        tuple
            The value ``forward(x)`` would return (the logits, or
            ``(logits, pred_cov)`` when ``mean_field_factor`` is ``None``)
            and the post-``rff`` features.
        """
        f = self.feature_extractor(x)
        f_reduc = self.jl(f)
        if self.normalize_gp_features:
            f_reduc = self.normalize(f_reduc)
        if f_reduc.dim() == 1:
            # A single unbatched example: give it a batch dimension of one.
            f_reduc = f_reduc.unsqueeze(0)

        k = self.rff(f_reduc)

        pred = self.beta(k)

        if self.training:
            # Detached: the stored precision carries no autograd graph.
            k_detached = k.detach()
            precision_minibatch = k_detached.t() @ k_detached
            self.precision += precision_minibatch
            # Marks any covariance cached since the last reset as stale.
            self.recompute_covariance = True
            self.seen_data += k.shape[0]
            self._seen_count += k.shape[0]

            if self._seen_count > self.num_data:
                raise RuntimeError("Did not reset precision matrix at start of epoch")
        else:
            if self._seen_count <= self.num_data - self.train_batch_size:
                raise RuntimeError("Not seen sufficient data for precision matrix")

            if self.recompute_covariance:
                with torch.no_grad():
                    eps = 1e-7
                    jitter = eps * torch.eye(
                        self.precision.shape[1],
                        device=self.precision.device,
                    )
                    covariance = cast(torch.Tensor, self.covariance)
                    covariance.copy_(_inverse_via_cholesky(self.precision + jitter))

                self.recompute_covariance: bool = False

            with torch.no_grad():
                pred_cov = k @ ((self.covariance @ k.t()) * self.ridge_penalty)

            if self.mean_field_factor is None:
                return (pred, pred_cov), k
            else:
                pred = self.mean_field_logits(pred, pred_cov, self.mean_field_factor)

        return pred, k


_DEFAULT_NUM_RANDOM_FEATURES = 1024

_DEVICE_TYPE_DEPRECATION = (
    "EquineGP.device_type is deprecated and will be removed in the next release: "
    "read EquineGP.device (a str) instead; to move the model, call "
    "model.to(value)."
)


def _expected_laplace_state(settings: dict[str, Any]) -> dict[str, torch.Tensor]:
    """
    The ``_Laplace`` state_dict that ``EquineGP(embedding, **settings)`` builds,
    feature extractor excluded, as meta tensors (names and shapes, no data).

    Plain arithmetic mirroring ``EquineGP.__init__`` (which always builds ``_Laplace``
    with ``num_gp_features == emb_out_dim`` and normalized GP features),
    ``_Laplace.__init__`` and ``_RandomFourierFeatures.__init__``.
    """
    e, c, n = (
        _positive_int(settings.get(name, default), f"Model file's settings[{name!r}]")
        for name, default in (
            ("emb_out_dim", None),
            ("num_classes", None),
            ("num_random_features", _DEFAULT_NUM_RANDOM_FEATURES),
        )
    )
    shapes: dict[str, tuple[int, ...]] = {
        "random_matrix": (e, e),
        "seen_data": (),
        "precision": (n, n),
        "covariance": (n, n),
        "normalize.weight": (e,),
        "normalize.bias": (e,),
        "rff.feature_scale": (),
        "rff.W": (e, n),
        "rff.b": (n,),
        "beta.weight": (c, n),
        "beta.bias": (c,),
    }
    try:
        return {key: torch.empty(shape, device="meta") for key, shape in shapes.items()}
    except (RuntimeError, TypeError) as err:  # sizes too large to represent
        raise ValueError(
            "Model file's settings describe a model too large to represent; "
            "refusing to load."
        ) from err


# -------------------------------------------------------------------------------
# EquineGP, below, demonstrates how to adapt that approach in EQUINE
@beartype
class EquineGP(Equine):
    """
    An example of an EQUINE model that builds upon the approach in "Spectral Norm
    Gaussian Processes" (SNGP). This wraps any pytorch embedding neural network and provides
    the `forward`, `predict`, `save`, and `load` methods required by Equine.

    Notes
    -----
    Although this model build upon the approach in SNGP, it does not enforce the spectral normalization
    and ResNet architecture required for SNGP. Instead, it is a simple wrapper around
    any pytorch embedding neural network. Your mileage may vary.
    """

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        emb_out_dim: int,
        num_classes: int,
        num_random_features: int = _DEFAULT_NUM_RANDOM_FEATURES,
        init_temperature: float = 1.0,
        device: str = "cpu",
        feature_names: list[str] | None = None,
        label_names: list[str] | None = None,
    ) -> None:
        """
        Initialize the EquineGP model.

        Parameters
        ----------
        embedding_model : torch.nn.Module
            Neural Network feature embedding.
        emb_out_dim : int
            The number of deep features from the feature embedding.
        num_classes : int
            The number of output classes this model predicts.
        num_random_features : int
            The dimension of the output of the RandomFourierFeatures operation
        init_temperature : float, optional
            What to use as the initial temperature (1.0 has no effect).
        device : str, optional
            The device to train the equine model on ('cpu', 'cuda' or 'mps';
            defaults to cpu).
        feature_names : list[str], optional
            List of strings of the names of the tabular features (ex ["duration", "fiat_mean", ...])
        label_names : list[str], optional
            List of strings of the names of the labels (ex ["streaming", "voip", ...])

        Raises
        ------
        ValueError
            If ``emb_out_dim``, ``num_classes`` or ``num_random_features`` is
            not a positive integer.
        """
        _positive_int(emb_out_dim, "emb_out_dim")
        _positive_int(num_classes, "num_classes")
        _positive_int(num_random_features, "num_random_features")
        super().__init__(
            embedding_model,
            device=device,
            feature_names=feature_names,
            label_names=label_names,
        )
        self.support: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.support_embeddings: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.prototypes: torch.Tensor = torch.Tensor()
        self.num_deep_features = emb_out_dim
        self.num_gp_features = emb_out_dim
        self.normalize_gp_features = True
        self.num_random_features = num_random_features
        self.num_outputs = num_classes
        self.mean_field_factor = 25
        self.ridge_penalty = 1
        self.feature_scale: float = 2.0
        self.init_temperature = init_temperature
        self.register_buffer(
            "temperature", torch.Tensor(self.init_temperature * torch.ones(1))
        )
        self.model: _Laplace = _Laplace(
            self.embedding_model,
            self.num_deep_features,
            self.num_gp_features,
            self.normalize_gp_features,
            self.num_random_features,
            self.num_outputs,
            self.feature_scale,
            self.mean_field_factor,
            self.ridge_penalty,
        )
        # Moves the temperature buffer and the Laplace head onto the model's device.
        self.to(self.device)

    @property
    def device_type(self) -> str:
        """Deprecated alias of ``device``."""
        warnings.warn(
            _DEVICE_TYPE_DEPRECATION,
            DeprecationWarning,
            # Past the beartype wrapper (none under -O), to the caller's line.
            stacklevel=2 + (0 if sys.flags.optimize else 1),
        )
        return self.device

    @device_type.setter
    def device_type(self, value: str) -> None:
        """
        Deprecated: moves the model with ``model.to(value)``, then assigns ``device``.
        """
        warnings.warn(
            _DEVICE_TYPE_DEPRECATION,
            DeprecationWarning,
            # Past nn.Module.__setattr__ and the beartype wrapper (none under -O).
            stacklevel=3 + (0 if sys.flags.optimize else 1),
        )
        # The module moves before ``device`` changes.
        self.to(value)
        self.device = value

    def train_model(
        self,
        dataset: Dataset,
        loss_fn: Callable,
        opt: torch.optim.Optimizer,
        num_epochs: int,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        batch_size: int = 64,
        validation_dataset: Dataset | None = None,
        val_metrics: Iterable[Metric] | None = None,
        vis_support: bool = False,
        support_size: int = 25,
    ) -> dict[str, Any]:
        """
        Train or fine-tune an EquineGP model.

        Parameters
        ----------
        dataset : Dataset
            The training data. ``dataset[:]`` must return the ``(inputs, labels)``
            tensors, as it does for a ``TensorDataset``, a ``Subset`` of one or a
            part returned by ``random_split``. ``vis_support=True`` needs a
            ``TensorDataset`` (``dataset.tensors``).
        loss_fn : Callable
            A pytorch loss function, e.g., torch.nn.CrossEntropyLoss().
        opt : torch.optim.Optimizer
            A pytorch optimizer, e.g., torch.optim.Adam().
        num_epochs : int
            The desired number of epochs to use for training.
        scheduler : torch.optim.lr_scheduler.LRScheduler, optional
            A pytorch scheduler, stepped once after each epoch's training pass.
        validation_dataset: Dataset
            If provided, will compute validation metrics on this dataset after each epoch of training
        val_metrics : Iterable[Metric], optional
            torchmetrics metrics computed on ``validation_dataset`` after each
            epoch. Each one is moved to this model's device in place and reset
            before each epoch's validation pass.
        batch_size : int, optional
            The number of samples to use per batch.
        vis_support : bool, optional
            If True, calls ``update_support`` with the training tensors after the
            last epoch. Requires a ``TensorDataset``.
        support_size : int, optional
            Maximum number of support examples per class that ``update_support``
            keeps when ``vis_support`` is True.

        Returns
        -------
        dict[str, Any]
            ``{"train_summary": ...}`` (the dict ``generate_train_summary``
            builds, also stored on ``self.train_summary``) plus, when a
            ``validation_dataset`` was given, ``"val_metrics"``: one list per
            metric in ``val_metrics`` holding what its ``compute()`` returned
            after each epoch, usually a tensor (``None`` when ``val_metrics``
            was not given). The entry for epoch ``n`` covers that epoch's
            validation pass only.

        Notes
        -----
        Training and eval mode are toggled on this wrapper, which propagates to the
        inner module. The model is left in eval mode afterwards, ready to predict.
        """

        self.validate_feature_label_names(dataset[0][0].shape[-1], self.num_outputs)

        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        val_loader: DataLoader | None = None
        if validation_dataset is not None:
            val_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

        self.model.set_training_params(len(dataset), batch_size)
        val_metrics_outputs: list[list[Any]] | None = None

        if validation_dataset is not None and val_metrics is not None:
            val_metrics = list(val_metrics)
            for metric in val_metrics:
                metric.to(self.device)
            val_metrics_outputs = [[] for _ in val_metrics]

        for _ in tqdm(range(num_epochs)):
            self.train()  # wrapper and inner module together
            self.model.reset_precision_matrix()
            epoch_loss = 0.0
            for i, (xs, labels) in enumerate(train_loader):
                opt.zero_grad()
                xs = self._input_to_model(xs)
                labels = labels.to(self.device)
                yhats = self.model(xs)
                loss = loss_fn(yhats, labels.to(torch.long))
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            if scheduler is not None:
                scheduler.step()
            self.eval()
            # compute the validation metrics
            if (
                validation_dataset is not None
                and val_loader is not None
                and val_metrics is not None
                and val_metrics_outputs is not None
            ):
                for metric in val_metrics:
                    metric.reset()
                for _, (xs_val, labels_val) in enumerate(val_loader):
                    xs_val = self._input_to_model(xs_val)
                    labels_val = labels_val.to(self.device)
                    yhats_val = self.model(xs_val)
                    for metric in val_metrics:
                        metric.update(yhats_val, labels_val)
                for i, metric in enumerate(val_metrics):
                    val_metrics_outputs[i].append(metric.compute())
        if vis_support:
            self.update_support(
                dataset.tensors[0], dataset.tensors[1], support_size=support_size
            )

        _, train_y = dataset[:]
        date_trained = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        self.train_summary: dict[str, Any] = generate_train_summary(
            self, train_y, date_trained
        )

        return_dict: dict[str, Any] = dict()
        return_dict["train_summary"] = self.train_summary
        if validation_dataset is not None:
            return_dict["val_metrics"] = val_metrics_outputs

        return return_dict

    def update_support(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        *args: Any,
        support_size: int | None = None,
    ) -> None:
        """Replace the support with up to ``support_size`` examples of each class,
        and store their embeddings and the class prototypes.

        Computes in eval mode and leaves the model in the mode the caller had it in (via
        ``train()``/``eval()``).

        Parameters
        ----------
        support_x : torch.Tensor
            Tensor containing the examples to take the support from.
        support_y : torch.Tensor
            Tensor containing the label of each example in ``support_x``.
        *args : Any
            Deprecated: ``support_size`` passed by position. Emits a
            ``DeprecationWarning`` and is then handled as the keyword.
        support_size : int
            Maximum number of support examples to keep per class. Required,
            keyword-only.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``support_size`` is missing, is passed both by position and by
            keyword, or more than one option is passed by position.
        ValueError
            If ``support_x`` and ``support_y`` hold different numbers of examples.
        """
        if len(args) > 1:
            raise TypeError(
                "EquineGP.update_support() takes at most 1 option by position "
                f"(support_size) but {len(args)} were given"
            )
        if args:
            if support_size is not None:
                raise TypeError(
                    "EquineGP.update_support() got multiple values for argument "
                    "'support_size'"
                )
            warnings.warn(
                "Passing support_size to EquineGP.update_support by position is "
                "deprecated and will stop working in the next release: pass "
                "support_size=... by keyword.",
                DeprecationWarning,
                # Past the beartype wrapper (none under -O), to the caller's line.
                stacklevel=2 + (0 if sys.flags.optimize else 1),
            )
            EquineGP.update_support(self, support_x, support_y, support_size=args[0])
            return
        if support_size is None:
            raise TypeError(
                "EquineGP.update_support() missing required keyword argument: "
                "'support_size'"
            )
        with _eval_mode(self):
            if support_x.shape[0] != support_y.shape[0]:
                raise ValueError(
                    f"support_x has {support_x.shape[0]} examples but support_y has "
                    f"{support_y.shape[0]} labels; they must match."
                )
            labels, counts = torch.unique(support_y, return_counts=True)
            shuffled_idxs = _get_shuffle_idxs_by_class(support_y, labels.tolist())
            support = OrderedDict()
            for label, count in list(zip(labels.tolist(), counts.tolist())):
                class_support = generate_support(
                    support_x,
                    support_y,
                    support_size=min(count, support_size),
                    selected_labels=[label],
                    shuffled_indexes={label: shuffled_idxs[label]},
                )
                support.update(class_support)

            # Through the model boundary: on the device and, for floating support, in
            # the model's dtype.
            self.support = OrderedDict(
                (label, self._input_to_model(x)) for label, x in support.items()
            )

            # Embedded without autograd: the result carries no graph.
            with torch.no_grad():
                support_embeddings = OrderedDict().fromkeys(
                    self.support.keys(), torch.Tensor()
                )
                for label in self.support:
                    support_embeddings[label] = self.compute_embeddings(
                        self.support[label]
                    )

                self.support_embeddings = support_embeddings
                self.prototypes = self._prototypes_from_embeddings()

    def compute_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method for computing deep embeddings for given input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for generating embeddings.

        Returns
        -------
        torch.Tensor
            Output embeddings .
        """
        x = self._input_to_model(x)
        f = self.model.feature_extractor(x)
        f_reduc = self.model.jl(f)
        if self.model.normalize_gp_features:
            f_reduc = self.model.normalize(f_reduc)

        return self.model.rff(f_reduc)

    @icontract.require(lambda self: len(self.support) > 0)
    def compute_prototypes(self) -> torch.Tensor:
        """
        Method for computing class prototypes based on given support examples.
        ``Prototypes'' in this context are the means of the support embeddings for each class.

        Embeds ``self.support`` with the current model (one pass per class)
        and returns the mean of each class's embeddings, in support order.
        Nothing is stored: ``support_embeddings`` and ``prototypes`` are left
        as they are.

        Returns
        -------
        torch.Tensor
            Tensors of prototypes for each of the given classes in the support.
        """
        embeddings = [self.compute_embeddings(x) for x in self.support.values()]
        return torch.stack([torch.mean(e, dim=0) for e in embeddings])

    @icontract.require(lambda self: len(self.support_embeddings) > 0)
    def _prototypes_from_embeddings(self) -> torch.Tensor:
        """Mean of ``self.support_embeddings`` per class, in support order; embeds nothing."""
        proto_list = []
        for label in self.support_embeddings:  # look at doing functorch
            class_prototype = torch.mean(self.support_embeddings[label], dim=0)
            proto_list.append(class_prototype)

        prototypes = torch.stack(proto_list)

        return prototypes

    @icontract.require(lambda self: len(self.support) > 0)
    def get_support(self) -> OrderedDict[int, torch.Tensor]:
        """
        Method for returning support examples used in training.

        Returns
        -------
        OrderedDict[int, torch.Tensor]
            Dictionary containing support examples for each class.
        """
        return self.support

    @icontract.require(lambda self: len(self.prototypes) > 0)
    def get_prototypes(self) -> torch.Tensor:
        """
        Method for returning class prototypes.

        Returns
        -------
        torch.Tensor
            Tensors of prototypes for each of the given classes in the support.
        """
        return self.prototypes

    @icontract.require(lambda num_calibration_epochs: 0 < num_calibration_epochs)
    @icontract.require(lambda calibration_lr: calibration_lr > 0.0)
    def calibrate_model(
        self,
        dataset: torch.utils.data.Dataset,
        num_calibration_epochs: int = 1,
        calibration_lr: float = 0.01,
        calibration_batch_size: int = 256,
    ) -> None:
        """
        Fine-tune the temperature on ``dataset``.

        ``train_model`` does not call this method; ``temperature`` stays at
        ``init_temperature`` until it is called.

        Parameters
        ----------
        dataset : Dataset
            A map-style pytorch ``Dataset`` (with ``__len__``) of
            ``(input, label)`` pairs.
        num_calibration_epochs : int, optional
            Number of epochs to tune temperature.
        calibration_lr : float, optional
            Learning rate for temperature optimization.
        calibration_batch_size : int, optional
            Batch size of the calibration data loader.
        """

        calibration_loader = DataLoader(
            dataset,
            batch_size=calibration_batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.temperature.requires_grad = True
        loss_fn = torch.nn.functional.cross_entropy
        optimizer = torch.optim.Adam([self.temperature], lr=calibration_lr)
        for _ in range(num_calibration_epochs):
            for xs, labels in calibration_loader:
                optimizer.zero_grad()
                xs = self._input_to_model(xs)
                labels = labels.to(self.device)
                with torch.no_grad():
                    logits = self.model(xs)
                logits = logits / self.temperature
                loss = loss_fn(logits, labels.to(torch.long))
                loss.backward()
                optimizer.step()
        self.temperature.requires_grad = False

    def _input_to_model(self, X: torch.Tensor) -> torch.Tensor:
        """
        Move ``X`` to the model device; a floating input is cast to the model's dtype.

        The reference is the Laplace head (``self.model``, see
        ``utils._input_to_model``): its first parameter is the embedding
        model's when that has one, else the head's own (``normalize.weight`` or
        ``beta``). A parameter-less embedding such as ``nn.Identity`` gets an input in
        the head's dtype.
        """
        return _input_to_model(X, self.model, self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        EquineGP forward function, generates logits for classification.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor for generating predictions. Moved to the model device
            and cast to the embedding model's parameter dtype.

        Returns
        -------
        torch.Tensor
            Logits divided by ``temperature``, shape (examples, classes).
        """
        logits, _ = self._forward_with_embeddings(X)
        return logits

    def _forward_with_embeddings(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ``forward`` that also returns its embeddings.

        The embeddings are ``compute_embeddings(X)``, with a single unbatched
        example given a batch dimension of one.
        """
        X = self._input_to_model(X)
        logits, embeddings = self.model._forward_with_features(X)
        logits = cast(torch.Tensor, logits)  # mean_field_factor is always set
        temperature = cast(torch.Tensor, self.temperature)  # registered buffer
        return logits / temperature, embeddings

    @icontract.ensure(
        lambda result: all((0 <= result.ood_scores) & (result.ood_scores <= 1.0))
    )
    def predict(self, X: torch.Tensor) -> EquineOutput:
        """
        Predict function for EquineGP, inherited and implemented from Equine.

        Computes in eval mode and leaves the model in the mode the caller had it in (via
        ``train()``/``eval()``). Every submodule's mode is restored; an embedding frozen
        with ``model.embedding_model.eval()`` during fine-tuning stays frozen.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor. It is moved to the model device and cast to the embedding
            model's parameter dtype; a float64 input to a float32 model is accepted.

        Returns
        -------
        EquineOutput
            Output object containing prediction probabilities, OOD scores and
            embeddings. Computed under ``torch.no_grad()``: the tensors carry
            no autograd graph but are ordinary tensors (not inference-mode
            tensors), which a caller can use in autograd.

        Raises
        ------
        ValueError
            If ``X`` is empty, or its embeddings hold a NaN or infinite value.
        """
        _require_examples(X)
        # One embedding pass, under no_grad.
        with _eval_mode(self), torch.no_grad():
            logits, embeddings = self._forward_with_embeddings(X)
            _require_finite_embeddings(embeddings)
            preds = torch.softmax(logits, dim=1)
            equiprobable = (
                torch.ones(self.num_outputs, device=logits.device) / self.num_outputs
            )
            max_entropy = torch.sum(torch.special.entr(equiprobable))
            ood_score = torch.sum(torch.special.entr(preds), dim=1) / max_entropy
        eq_out = EquineOutput(
            classes=preds, ood_scores=ood_score, embeddings=embeddings
        )

        self.validate_feature_label_names(X.shape[-1], self.num_outputs)

        return eq_out

    def save(self, path: str) -> None:
        """
        Save all model parameters to a file.

        The embedding model (the feature extractor) is stored as a *recipe*
        (registered architecture name plus constructor arguments) and a
        ``state_dict``, so the file holds only data.

        Parameters
        ----------
        path : str
            Filename to write the model.

        Raises
        ------
        ValueError
            If the embedding model is not a registered architecture.
        RuntimeError
            If the model has not been trained with ``train_model``.
        """
        if self.model.num_data == 0:
            raise RuntimeError(
                "The model has not been trained; call train_model before save."
            )
        torch.save(self._to_checkpoint(), path)

    def _to_checkpoint(self) -> dict[str, Any]:
        """Build the checkpoint dictionary that ``save`` writes and ``_from_checkpoint`` reads."""
        model_settings = {
            "emb_out_dim": self.num_deep_features,
            "num_classes": self.num_outputs,
            "num_random_features": self.num_random_features,
            "init_temperature": self.temperature.item(),
            "device": self.device,
        }
        # The feature extractor is the embedding model, stored separately.
        laplace_sd = {
            key: value
            for key, value in self.model.state_dict().items()
            if "feature_extractor" not in key
        }
        # Plain values only, so load_checkpoint can read it with weights_only=True.
        save_data: dict[str, Any] = {
            "equine_format_version": EQUINE_FORMAT_VERSION,
            **_embedding_checkpoint(self.embedding_model),
            "feature_names": _plain_names(self.feature_names),
            "label_names": _plain_names(self.label_names),
            "laplace_model_save": laplace_sd,
            "num_data": self.model.num_data,
            "settings": model_settings,
            "support": {
                int(label): x.detach().clone() for label, x in self.support.items()
            },
            "train_batch_size": self.model.train_batch_size,
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
        Load a previously saved EquineGP model.

        Loading runs no code from the file: it is read with
        ``torch.load(weights_only=True)`` and the embedding model is rebuilt from its
        recipe through the architecture registry.

        Parameters
        ----------
        path : str
            Input filename.
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
        EquineGP
            The reconstituted EquineGP object.

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
        Rebuild an EquineGP from an already-loaded checkpoint dictionary.

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
        EquineGP
            The reconstituted EquineGP object.
        """
        # The embedding is rebuilt first.
        embedding = _rebuild_embedding(model_save, embedding_model)
        _require_entries(
            model_save,
            (
                "settings",
                "support",
                "train_summary",
                "laplace_model_save",
                "num_data",
                "train_batch_size",
            ),
            cls,
        )
        stored_support = _support_from_file(model_save["support"])
        num_data = _positive_int(model_save["num_data"], "Model file's num_data")
        batch_size = _positive_int(
            model_save["train_batch_size"], "Model file's train_batch_size"
        )
        if batch_size > num_data:
            raise ValueError(
                "Model file's train_batch_size is larger than its num_data; the file "
                "is inconsistent or was tampered with."
            )
        for key in ("feature_names", "label_names"):
            _require_names(model_save.get(key), key)
        settings = _checked_settings(cls, model_save["settings"])
        # Allow the user to override the saved device state dynamically
        if device is not None:
            settings["device"] = device
        else:
            _require_device(settings, hint=" (pass device= to choose another)")
        # The stored Laplace weights are checked against the settings before the model
        # is built.
        _require_matching_weights(
            _expected_laplace_state(settings),
            model_save.get("laplace_model_save"),
            "laplace_model_save",
        )
        try:
            eq_model = cls(embedding, **settings)
        except Exception as err:
            raise ValueError(
                f"Could not build {cls.__name__} from the model file's settings "
                f"({_truncate(_describe_error(err), 200)})."
            ) from err

        eq_model.feature_names = model_save.get("feature_names")
        eq_model.label_names = model_save.get("label_names")
        eq_model.train_summary = model_save.get("train_summary")

        _adopt_stored_dtypes(
            eq_model.model, model_save["laplace_model_save"], "laplace_model_save"
        )
        eq_model.model.load_state_dict(
            model_save.get("laplace_model_save"), strict=False
        )
        # The load_state_dict post hook has already updated _seen_count.
        eq_model.model.seen_data = (
            model_save.get("laplace_model_save")
            .get("seen_data")
            .to(eq_model.model.precision.device)
        )

        eq_model.model.set_training_params(
            model_save.get("num_data"), model_save.get("train_batch_size")
        )
        eq_model.eval()

        # load_checkpoint returns CPU tensors; the support goes to the model's device.
        try:
            support = OrderedDict(
                (label, x.to(eq_model.device)) for label, x in stored_support.items()
            )
            if len(support) > 0:
                eq_model.support = support
                with torch.no_grad():
                    eq_model.support_embeddings = OrderedDict(
                        (label, eq_model.compute_embeddings(x))
                        for label, x in support.items()
                    )
                    eq_model.prototypes = eq_model._prototypes_from_embeddings()
        except Exception as err:
            raise ValueError(
                "Model file's support could not be passed through its embedding "
                f"model ({_truncate(_describe_error(err), 200)}); the "
                "file is inconsistent or was tampered with."
            ) from err

        return eq_model
