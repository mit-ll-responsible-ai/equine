# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import math
import sys
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any, Optional, Union, cast

import icontract
import torch
from beartype import beartype
from torch.utils.data import DataLoader, Dataset
from torchmetrics.metric import Metric
from tqdm import tqdm

from .equine import Equine, EquineOutput
from .registry import _positive_int, _truncate
from .utils import (
    _MIGRATION_HINT,
    EQUINE_FORMAT_VERSION,
    _checked_settings,
    _embedding_checkpoint,
    _input_to_model,
    _plain_names,
    _rebuild_embedding,
    _require_device,
    _require_entries,
    _require_matching_weights,
    _require_names,
    _support_from_file,
    generate_support,
    generate_train_summary,
    load_checkpoint,
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
        self, in_dim: int, num_random_features: int, feature_scale: Optional[float]
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
        feature_scale : Optional[float]
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

    On MPS the factorization and the inverse run on a CPU copy and the result
    returns to MPS: torch 2.6 has no MPS kernel for ``linalg.cholesky_ex``,
    and 2.6 to 2.9 none for ``cholesky_inverse``. On the CPU and CUDA the ops
    run where ``a`` is, as before.
    """
    work = a.cpu() if a.device.type == "mps" else a
    u, info = torch.linalg.cholesky_ex(work)
    assert (info == 0).all(), "Precision matrix inversion failed!"
    return torch.cholesky_inverse(u).to(a.device)


def _entr(x: torch.Tensor) -> torch.Tensor:
    """``torch.special.entr``, computed on a CPU copy for an MPS tensor (no MPS kernel in torch 2.6)."""
    if x.device.type == "mps":
        return torch.special.entr(x.cpu()).to(x.device)
    return torch.special.entr(x)


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
        feature_scale: Optional[float],
        mean_field_factor: Optional[float],  # required for classification problems
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
        feature_scale : Optional[float]
            The scaling factor for the random Fourier features.
        mean_field_factor : Optional[float]
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
            self.jl: Callable = lambda x: torch.nn.functional.linear(
                x, self.random_matrix
            )
        else:
            self.num_gp_features: int = num_deep_features
            self.jl: Callable = lambda x: x  # Identity

        self.normalize_gp_features = normalize_gp_features
        if normalize_gp_features:
            self.normalize: torch.nn.LayerNorm = torch.nn.LayerNorm(num_gp_features)

        self.rff: _RandomFourierFeatures = _RandomFourierFeatures(
            num_gp_features, num_random_features, feature_scale
        )
        self.beta: torch.nn.Linear = torch.nn.Linear(num_random_features, num_outputs)

        self.num_data = 0  # to be set later
        self.register_buffer("seen_data", torch.tensor(0))
        # Python mirror of seen_data for the asserts in the forward pass:
        # reading the buffer would synchronize with the accelerator on every
        # forward. Kept beside the buffer (which stays in the state_dict) and
        # read back from it after any load_state_dict (one sync per load).
        self._seen_count: int = 0
        self.register_load_state_dict_post_hook(_sync_seen_count)

        precision = torch.eye(num_random_features) * self.ridge_penalty
        self.register_buffer("precision", precision)

        self.recompute_covariance = True
        self.register_buffer("covariance", torch.eye(num_random_features))
        self.training_parameters_set = False

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

    @icontract.require(lambda self: self.training_parameters_set)
    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the forward pass of the Laplace approximation to the Gaussian Process.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, num_features).

        Returns
        -------
        Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            If the model is in training mode, returns the predicted mean of shape (batch_size, 1).
            If the model is in evaluation mode, returns a tuple containing the predicted mean of shape (batch_size, 1)
            and the predicted covariance matrix of shape (batch_size, batch_size).
        """
        f = self.feature_extractor(x)
        f_reduc = self.jl(f)
        if self.normalize_gp_features:
            f_reduc = self.normalize(f_reduc)

        k = self.rff(f_reduc)

        pred = self.beta(k)

        if self.training:
            precision_minibatch = k.t() @ k
            self.precision += precision_minibatch
            self.seen_data += x.shape[0]
            self._seen_count += x.shape[0]

            assert self._seen_count <= self.num_data, (
                "Did not reset precision matrix at start of epoch"
            )
        else:
            assert self._seen_count > (self.num_data - self.train_batch_size), (
                "Not seen sufficient data for precision matrix"
            )

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
                return pred, pred_cov
            else:
                pred = self.mean_field_logits(pred, pred_cov, self.mean_field_factor)

        return pred


_DEFAULT_NUM_RANDOM_FEATURES = 1024

_USE_TEMPERATURE_WARNING = (
    "Model file's settings include 'use_temperature' (saved by EQUINE 0.1.5 or "
    "earlier); EquineGP has not accepted this setting since 0.1.6, so it is "
    "dropped. The saved temperature is kept and predictions are unchanged. To "
    f"rewrite the file without it, {_MIGRATION_HINT}."
)

_DEVICE_TYPE_DEPRECATION = (
    "EquineGP.device_type is deprecated and will be removed in the next release: "
    "read EquineGP.device (a str) instead; to move the model, set "
    "model.device = value and call model.to(value)."
)


def _expected_laplace_state(settings: dict[str, Any]) -> dict[str, torch.Tensor]:
    """
    The ``_Laplace`` state_dict that ``EquineGP(embedding, **settings)`` builds,
    feature extractor excluded, as meta tensors (names and shapes, no data).

    Plain arithmetic mirroring ``EquineGP.__init__`` (which always builds
    ``_Laplace`` with ``num_gp_features == emb_out_dim`` and normalized GP
    features), ``_Laplace.__init__`` and ``_RandomFourierFeatures.__init__``,
    so that nothing sized by a file's settings is allocated before those
    settings are checked against the stored weights.
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
        feature_names: Optional[list[str]] = None,
        label_names: Optional[list[str]] = None,
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
        """
        super().__init__(
            embedding_model,
            device=device,
            feature_names=feature_names,
            label_names=label_names,
        )
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
        # Equine.__init__ moved the module before the temperature buffer and
        # the Laplace head existed; move again so everything is on
        # self.device (#170).
        self.to(self.device)

    @property
    def device_type(self) -> str:
        """Deprecated alias of ``device``; removed in the next release."""
        warnings.warn(
            _DEVICE_TYPE_DEPRECATION,
            DeprecationWarning,
            # Skip the beartype wrapper (a no-op under `python -O`), so the
            # warning names the caller's line and default filters show it.
            stacklevel=2 + (0 if sys.flags.optimize else 1),
        )
        return self.device

    @device_type.setter
    def device_type(self, value: str) -> None:
        """
        Deprecated: moves the module to ``value`` and assigns ``device``.

        Before 0.1.9 ``device_type`` was a plain attribute read only by
        ``save()``: assigning it changed the device recorded in the file and
        neither where inputs were placed nor where the module lived. Now
        ``device`` decides where inputs go, so the module moves with it to
        stay consistent. To move a model, set ``model.device = value`` and
        call ``model.to(value)``: either one alone leaves the inputs and the
        module on different devices.
        """
        warnings.warn(
            _DEVICE_TYPE_DEPRECATION,
            DeprecationWarning,
            # Skip nn.Module.__setattr__ and the beartype wrapper (a no-op
            # under `python -O`).
            stacklevel=3 + (0 if sys.flags.optimize else 1),
        )
        # Move first: if the move raises (an unavailable device), ``device``
        # still names where the module is and the model keeps working.
        self.to(value)
        self.device = value

    def train_model(
        self,
        dataset: Dataset,
        loss_fn: Callable,
        opt: torch.optim.Optimizer,
        num_epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        batch_size: int = 64,
        validation_dataset: Optional[Dataset] = None,
        val_metrics: Optional[Iterable[Metric]] = None,
        vis_support: bool = False,
        support_size: int = 25,
    ) -> dict[str, Any]:
        """
        Train or fine-tune an EquineGP model.

        Parameters
        ----------
        dataset : TensorDataset
            An iterable, pytorch TensorDataset.
        loss_fn : Callable
            A pytorch loss function, e.g., torch.nn.CrossEntropyLoss().
        opt : torch.optim.Optimizer
            A pytorch optimizer, e.g., torch.optim.Adam().
        num_epochs : int
            The desired number of epochs to use for training.
        scheduler : torch.optim.LRScheduler
            A pytorch scheduler, if one is desired
        validation_dataset: Dataset
            If provided, will compute validation metrics on this dataset after each epoch of training
        batch_size : int, optional
            The number of samples to use per batch.

        Returns
        -------
        dict[str, Any]
            A dict containing a dict of summary stats and a dataloader for the calibration data.

        """

        self.validate_feature_label_names(dataset[0][0].shape[-1], self.num_outputs)

        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        val_loader: Optional[DataLoader] = None
        if validation_dataset is not None:
            val_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

        self.model.set_training_params(len(dataset), batch_size)
        val_metrics_outputs: Optional[list[list[float]]] = None

        if validation_dataset is not None and val_metrics is not None:
            val_metrics_outputs = [[] for i in range(len(list(val_metrics)))]

        for _ in tqdm(range(num_epochs)):
            self.model.train()
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
            self.model.eval()
            # compute the validation metrics
            if (
                validation_dataset is not None
                and val_loader is not None
                and val_metrics is not None
                and val_metrics_outputs is not None
            ):
                for _, (xs_val, labels_val) in enumerate(val_loader):
                    xs_val = self._input_to_model(xs_val)
                    labels_val = labels_val.to(self.device)
                    yhats_val = self.model(xs_val)
                    for metric in val_metrics:
                        metric.update(yhats_val, labels_val)
                for i, metric in enumerate(val_metrics):
                    val_metrics_outputs[i].append(metric.compute())
        if vis_support:
            self.update_support(dataset.tensors[0], dataset.tensors[1], support_size)

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
        self, support_x: torch.Tensor, support_y: torch.Tensor, support_size: int
    ) -> None:
        """Function to update protonet support examples with given examples.

        Parameters
        ----------
        support_x : torch.Tensor
            Tensor containing support examples for protonet.
        support_y : torch.Tensor
            Tensor containing labels for given support examples.

        Returns
        -------
        None
        """

        labels, counts = torch.unique(support_y, return_counts=True)
        support = OrderedDict()
        for label, count in list(zip(labels.tolist(), counts.tolist())):
            class_support = generate_support(
                support_x,
                support_y,
                support_size=min(count, support_size),
                selected_labels=[label],
            )
            support.update(class_support)

        # Through the model boundary: on the device and, for floating support,
        # in the model's dtype (so a float32 model on MPS accepts float64).
        self.support = OrderedDict(
            (label, self._input_to_model(x)) for label, x in support.items()
        )

        support_embeddings = OrderedDict().fromkeys(self.support.keys(), torch.Tensor())
        for label in self.support:
            support_embeddings[label] = self.compute_embeddings(self.support[label])

        self.support_embeddings = support_embeddings
        self.prototypes: torch.Tensor = self.compute_prototypes()

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

        Returns
        -------
        torch.Tensor
            Tensors of prototypes for each of the given classes in the support.
        """
        # Compute support embeddings
        support_embeddings = OrderedDict().fromkeys(self.support.keys())
        for label in self.support:
            support_embeddings[label] = self.compute_embeddings(self.support[label])

        # Compute prototype for each class
        proto_list = []
        for label in self.support:  # look at doing functorch
            class_prototype = torch.mean(support_embeddings[label], dim=0)  # type: ignore
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
        Fine-tune the temperature after training. Note this function is also run at the conclusion of train_model.

        Parameters
        ----------
        dataset : TensorDataset
            An iterable, pytorch TensorDataset.
        num_calibration_epochs : int, optional
            Number of epochs to tune temperature.
        calibration_lr : float, optional
            Learning rate for temperature optimization.
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
        model's when that has one, else the head's own (``normalize.weight``
        or ``beta``), so a parameter-less embedding such as ``nn.Identity``
        still gets an input the float32 head can take.
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
            Output probabilities computed.
        """
        X = self._input_to_model(X)
        preds = self.model(X)
        return preds / self.temperature.to(self.device)

    @icontract.ensure(
        lambda result: all((0 <= result.ood_scores) & (result.ood_scores <= 1.0))
    )
    def predict(self, X: torch.Tensor) -> EquineOutput:
        """
        Predict function for EquineGP, inherited and implemented from Equine.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor. It is moved to the model device and cast to the
            embedding model's parameter dtype (so a float64 input to a float32
            model is accepted; on MPS, which has no float64, this is required).

        Returns
        -------
        EquineOutput
            Output object containing prediction probabilities and OOD scores.
        """
        X = self._input_to_model(X)
        logits = self(X)
        preds = torch.softmax(logits, dim=1)
        equiprobable = (
            torch.ones(self.num_outputs, device=logits.device) / self.num_outputs
        )
        max_entropy = torch.sum(_entr(equiprobable))
        ood_score = torch.sum(_entr(preds), dim=1) / max_entropy
        embeddings = self.compute_embeddings(X)
        eq_out = EquineOutput(
            classes=preds, ood_scores=ood_score, embeddings=embeddings
        )  # TODO return embeddings

        self.validate_feature_label_names(X.shape[-1], self.num_outputs)

        return eq_out

    def save(self, path: str, *, allow_executable: bool = False) -> None:
        """
        Save all model parameters to a file.

        The embedding model (the feature extractor) is stored as a *recipe*
        (registered architecture name plus constructor arguments) and a
        ``state_dict``, so the file holds only data. If the embedding model is
        not a registered architecture (for example a ``torch.jit.ScriptModule``),
        pass ``allow_executable=True`` to embed a TorchScript copy instead; such
        a file is flagged and can only be opened with
        ``load(..., trust_executable=True)``. This opt-in exists for one release
        to migrate existing files and will then be removed.

        Parameters
        ----------
        path : str
            Filename to write the model.
        allow_executable : bool, optional
            Keyword-only. Permit embedding executable TorchScript when no recipe
            is available; emits a ``FutureWarning``.

        Raises
        ------
        ValueError
            If the embedding model has no recipe and ``allow_executable`` is False.
        """
        checkpoint = self._to_checkpoint(
            allow_executable,
            # _embedding_checkpoint, then _to_checkpoint and save, each behind a
            # beartype wrapper (absent under `python -O`), then the caller.
            _stacklevel=4 + (0 if sys.flags.optimize else 2),
        )
        torch.save(checkpoint, path)

    def _to_checkpoint(
        self, allow_executable: bool = False, *, _stacklevel: int = 2
    ) -> dict[str, Any]:
        """Build the checkpoint dictionary that ``save`` writes and ``_from_checkpoint`` reads."""
        model_settings = {
            "emb_out_dim": self.num_deep_features,
            "num_classes": self.num_outputs,
            "num_random_features": self.num_random_features,
            "init_temperature": self.temperature.item(),
            "device": self.device,
        }
        # The feature extractor is the embedding model, which is stored on its
        # own below, so its weights are left out of the Laplace state_dict.
        laplace_sd = {
            key: value
            for key, value in self.model.state_dict().items()
            if "feature_extractor" not in key
        }
        # Everything stored here must be readable by torch.load(weights_only=True)
        # on every supported torch version: tensors, containers and plain
        # scalars/strings only (issue #168).
        save_data: dict[str, Any] = {
            "equine_format_version": EQUINE_FORMAT_VERSION,
            **_embedding_checkpoint(
                self.embedding_model, allow_executable, _stacklevel
            ),
            "feature_names": _plain_names(self.feature_names),
            "label_names": _plain_names(self.label_names),
            "laplace_model_save": laplace_sd,
            "num_data": self.model.num_data,
            "settings": model_settings,
            # Support tensors are usually views of all their class's training
            # rows; a clone stores just the support rows, each label separately.
            "support": {
                int(label): x.detach().clone() for label, x in self.support.items()
            },
            "train_batch_size": self.model.train_batch_size,
            "train_summary": self.train_summary,
        }
        return save_data

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[str] = None,
        *,
        allow_unsafe_legacy_format: bool = False,
        trust_executable: bool = False,
        embedding_model: Optional[torch.nn.Module] = None,
    ) -> Equine:
        """
        Load a previously saved EquineGP model.

        The file is read with ``torch.load(weights_only=True)`` and the
        embedding model is rebuilt from its recipe through the architecture
        registry, so a default file contains nothing executable.

        Parameters
        ----------
        path : str
            Input filename.
        device : Optional[str]
            The device to load the model onto. Overrides the device recorded
            in the file; by default the model lands on the saved device.
        allow_unsafe_legacy_format : bool, optional
            Keyword-only. Permit loading a file written in the legacy pickle
            format. This uses unrestricted unpickling and can execute code
            embedded in the file, so only enable it for files you trust.
            Implies ``trust_executable``. Defaults to False.
        trust_executable : bool, optional
            Keyword-only. Permit running the TorchScript module embedded in a
            file saved with ``allow_executable=True``; keeping that module as
            the embedding emits a ``FutureWarning``. Defaults to False.
        embedding_model : Optional[torch.nn.Module]
            Keyword-only. Use this module as the embedding architecture instead
            of rebuilding it from the file's recipe; the file's weights are
            loaded into it. With a legacy or executable file (and trust), the
            archive's weights are copied into it, which migrates the file to
            the recipe format on the next ``save``.

        Returns
        -------
        EquineGP
            The reconstituted EquineGP object.

        Raises
        ------
        ValueError
            If the file cannot be loaded safely, names an unregistered
            architecture, or contains executable content without trust. If
            ``device`` is unavailable here, names an index at or above its
            device count, or lacks a dtype the file holds (float64 on MPS:
            load such a file with ``device="cpu"``).
        """
        # map_location so internal tensors map to the correct device
        model_save = load_checkpoint(
            path,
            map_location=device,
            allow_unsafe_legacy_format=allow_unsafe_legacy_format,
            # Skip the beartype wrapper around this classmethod. Under
            # `python -O`, beartype decorators become a no-op (identity)
            # passthrough, so that wrapper frame doesn't exist and the
            # stacklevel must be one shorter to still land on the caller.
            _stacklevel=3 + (0 if sys.flags.optimize else 1),
        )
        return cls._from_checkpoint(
            model_save,
            device,
            trust_executable=trust_executable,
            embedding_model=embedding_model,
            allow_unsafe_legacy_format=allow_unsafe_legacy_format,
            # _from_checkpoint, then load, each behind a beartype wrapper
            # (absent under `python -O`), then the caller.
            _stacklevel=3 + (0 if sys.flags.optimize else 2),
        )

    @classmethod
    def _from_checkpoint(
        cls,
        model_save: dict[str, Any],
        device: Optional[str] = None,
        trust_executable: bool = False,
        embedding_model: Optional[torch.nn.Module] = None,
        allow_unsafe_legacy_format: bool = False,
        _stacklevel: int = 2,
    ) -> Equine:
        """
        Rebuild an EquineGP from an already-loaded checkpoint dictionary.

        Parameters
        ----------
        model_save : dict[str, Any]
            The dictionary returned by ``utils.load_checkpoint``.
        device : Optional[str]
            Device override for the reconstituted model.
        trust_executable : bool, optional
            Permit running an embedded TorchScript module. Defaults to False.
        embedding_model : Optional[torch.nn.Module]
            Use this module as the embedding architecture instead of the file's
            recipe or archive; the file's weights are loaded into it.
        allow_unsafe_legacy_format : bool, optional
            The caller vouched for the file (see ``load``); implies
            ``trust_executable``.
        _stacklevel : int, optional
            Stack depth, counted from this method, at which its warnings are
            reported, so they point at the user's call site rather than at
            EQUINE internals.

        Returns
        -------
        EquineGP
            The reconstituted EquineGP object.
        """
        # The embedding first: it is the single audit point for executable
        # content, so a file that needs trust says so whatever else it lacks.
        embedding = _rebuild_embedding(
            model_save,
            device,
            trust_executable or allow_unsafe_legacy_format,
            embedding_model,
            _stacklevel=_stacklevel + 1,
        )
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
        for key in ("feature_names", "label_names"):
            _require_names(model_save.get(key), key)
        settings = model_save["settings"]
        if isinstance(settings, dict) and "use_temperature" in settings:
            # EquineGP.__init__ dropped use_temperature in 0.1.6. It only chose
            # whether train_model calibrated; the calibrated temperature is
            # stored as init_temperature, so dropping it changes no prediction.
            settings = {k: v for k, v in settings.items() if k != "use_temperature"}
            warnings.warn(_USE_TEMPERATURE_WARNING, UserWarning, stacklevel=_stacklevel)
        settings = _checked_settings(cls, settings)
        # Allow the user to override the saved device state dynamically
        if device is not None:
            settings["device"] = device
        else:
            _require_device(settings, hint=" (pass device= to choose another)")
        # _Laplace allocates buffers sized by the settings (quadratic in
        # num_random_features and emb_out_dim), so the stored head weights must
        # match them before anything is constructed.
        _require_matching_weights(
            _expected_laplace_state(settings),
            model_save.get("laplace_model_save"),
            "laplace_model_save",
        )
        try:
            eq_model = cls(embedding, **settings)
        except Exception as err:  # a backstop for anything the checks above miss
            raise ValueError(
                f"Could not build {cls.__name__} from the model file's settings "
                f"({_truncate(f'{type(err).__name__}: {err}', 200)})."
            ) from err

        eq_model.feature_names = model_save.get("feature_names")
        eq_model.label_names = model_save.get("label_names")
        eq_model.train_summary = model_save.get("train_summary")

        eq_model.model.load_state_dict(
            model_save.get("laplace_model_save"), strict=False
        )
        # (_seen_count already follows the loaded seen_data: load_state_dict
        # post hook.)
        eq_model.model.seen_data = (
            model_save.get("laplace_model_save")
            .get("seen_data")
            .to(eq_model.model.precision.device)
        )

        eq_model.model.set_training_params(
            model_save.get("num_data"), model_save.get("train_batch_size")
        )
        eq_model.eval()

        # Without a device override load_checkpoint returns CPU tensors while
        # the model is on settings["device"]. compute_prototypes embeds the
        # support with the model, so the support goes there, as update_support
        # needed it.
        try:
            support = OrderedDict(
                (label, x.to(eq_model.device)) for label, x in stored_support.items()
            )
            if len(support) > 0:
                eq_model.support = support
                eq_model.prototypes = eq_model.compute_prototypes()
        except Exception as err:  # the embedding is arbitrary code
            raise ValueError(
                "Model file's support could not be passed through its embedding "
                f"model ({_truncate(f'{type(err).__name__}: {err}', 200)}); the "
                "file is inconsistent or was tampered with."
            ) from err

        return eq_model
