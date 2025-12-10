# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, Optional, Union

import icontract
import io
import math
import torch
from beartype import beartype
from collections import OrderedDict
from collections.abc import Callable, Iterable
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torchmetrics.metric import Metric
from tqdm import tqdm

from .equine import Equine, EquineOutput
from .utils import generate_support, generate_train_summary

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
        self.seen_data: torch.Tensor = torch.tensor(0)
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

            assert (
                self.seen_data <= self.num_data
            ), "Did not reset precision matrix at start of epoch"
        else:
            assert self.seen_data > (
                self.num_data - self.train_batch_size
            ), "Not seen sufficient data for precision matrix"

            if self.recompute_covariance:
                with torch.no_grad():
                    eps = 1e-7
                    jitter = eps * torch.eye(
                        self.precision.shape[1],
                        device=self.precision.device,
                    )
                    u, info = torch.linalg.cholesky_ex(self.precision + jitter)
                    assert (info == 0).all(), "Precision matrix inversion failed!"
                    torch.cholesky_inverse(u, out=self.covariance)

                self.recompute_covariance: bool = False

            with torch.no_grad():
                pred_cov = k @ ((self.covariance @ k.t()) * self.ridge_penalty)

            if self.mean_field_factor is None:
                return pred, pred_cov
            else:
                pred = self.mean_field_logits(pred, pred_cov, self.mean_field_factor)

        return pred


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
        num_random_features: int = 1024,
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
            Either 'cuda' or 'cpu'.
        feature_names : list[str], optional
            List of strings of the names of the tabular features (ex ["duration", "fiat_mean", ...])
        label_names : list[str], optional
            List of strings of the names of the labels (ex ["streaming", "voip", ...])
        """
        super().__init__(
            embedding_model, feature_names=feature_names, label_names=label_names
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
        self.device_type = device
        self.device: torch.device = torch.device(self.device_type)
        self.model.to(self.device)

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
                xs = xs.to(self.device)
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
                    xs_val = xs_val.to(self.device)
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

        self.support = support

        support_embeddings = OrderedDict().fromkeys(self.support.keys(), torch.Tensor())
        for label in support:
            support_embeddings[label] = self.compute_embeddings(support[label])

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
                xs = xs.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    logits = self.model(xs)
                logits = logits / self.temperature
                loss = loss_fn(logits, labels.to(torch.long))
                loss.backward()
                optimizer.step()
        self.temperature.requires_grad = False

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        EquineGP forward function, generates logits for classification.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor for generating predictions.

        Returns
        -------
        torch.Tensor
            Output probabilities computed.
        """
        X = X.to(self.device)
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
            Input tensor.

        Returns
        -------
        EquineOutput
            Output object containing prediction probabilities and OOD scores.
        """
        logits = self(X)
        preds = torch.softmax(logits, dim=1)
        equiprobable = torch.ones(self.num_outputs) / self.num_outputs
        max_entropy = torch.sum(torch.special.entr(equiprobable))
        ood_score = torch.sum(torch.special.entr(preds), dim=1) / max_entropy
        embeddings = self.compute_embeddings(X)
        eq_out = EquineOutput(
            classes=preds, ood_scores=ood_score, embeddings=embeddings
        )  # TODO return embeddings

        self.validate_feature_label_names(X.shape[-1], self.num_outputs)

        return eq_out

    def save(self, path: str) -> None:
        """
        Function to save all model parameters to a file.

        Parameters
        ----------
        path : str
            Filename to write the model.
        """
        model_settings = {
            "emb_out_dim": self.num_deep_features,
            "num_classes": self.num_outputs,
            "init_temperature": self.temperature.item(),
            "device": self.device_type,
        }

        jit_model = torch.jit.script(self.model.feature_extractor)
        buffer = io.BytesIO()
        torch.jit.save(jit_model, buffer)
        buffer.seek(0)

        laplace_sd = self.model.state_dict()
        keys_to_delete = []
        for key in laplace_sd:
            if "feature_extractor" in key:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del laplace_sd[key]

        save_data = {
            "embed_jit_save": buffer,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "laplace_model_save": laplace_sd,
            "num_data": self.model.num_data,
            "settings": model_settings,
            "support": self.support,
            "train_batch_size": self.model.train_batch_size,
            "train_summary": self.train_summary,
        }

        torch.save(save_data, path)  # TODO allow model checkpointing

    @classmethod
    def load(cls, path: str) -> Equine:
        """
        Function to load previously saved EquineGP model.

        Parameters
        ----------
        path : str
            Input filename.

        Returns
        -------
        EquineGP
            The reconstituted EquineGP object.
        """
        model_save = torch.load(path, weights_only=False)
        jit_model = torch.jit.load(model_save.get("embed_jit_save"))
        eq_model = cls(jit_model, **model_save.get("settings"))

        eq_model.feature_names = model_save.get("feature_names")
        eq_model.label_names = model_save.get("label_names")
        eq_model.train_summary = model_save.get("train_summary")

        eq_model.model.load_state_dict(
            model_save.get("laplace_model_save"), strict=False
        )
        eq_model.model.seen_data = model_save.get("laplace_model_save").get("seen_data")

        eq_model.model.set_training_params(
            model_save.get("num_data"), model_save.get("train_batch_size")
        )
        eq_model.eval()

        support = model_save.get("support")
        if len(support) > 0:
            eq_model.support = support
            eq_model.prototypes = eq_model.compute_prototypes()

        return eq_model
