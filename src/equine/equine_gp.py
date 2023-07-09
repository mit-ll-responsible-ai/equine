# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import math
import io

import icontract
import torch
from torch.utils.data import TensorDataset, DataLoader  # type: ignore
from typing import Optional, Callable
from tqdm import tqdm
from typeguard import typechecked
from datetime import datetime
from sklearn.model_selection import train_test_split

from .equine import Equine, EquineOutput
from .utils import generate_train_summary


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


@typechecked
def _random_ortho(n: int, m: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(n, m))
    return q


@typechecked
class _RandomFourierFeatures(torch.nn.Module):
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
            W = _random_ortho(in_dim, num_random_features)
        else:
            # generate blocks of orthonormal rows which are not neccesarily orthonormal
            # to each other.
            dim_left = num_random_features
            ws = []
            while dim_left > in_dim:
                ws.append(_random_ortho(in_dim, in_dim))
                dim_left -= in_dim
            ws.append(_random_ortho(in_dim, dim_left))
            W = torch.cat(ws, 1)

        feature_norm = torch.randn(W.shape) ** 2
        W = W * feature_norm.sum(0).sqrt()
        self.register_buffer("W", W)

        b = torch.empty(num_random_features).uniform_(0, 2 * math.pi)
        self.register_buffer("b", b)

    def forward(self, x) -> torch.Tensor:
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
        A private class to compute a Laplace approximation to a Gaussian Process (GP)
        in the .

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
            self.register_buffer(
                "random_matrix",
                torch.normal(0, 0.05, (num_gp_features, num_deep_features)),
            )
            self.jl = lambda x: torch.nn.functional.linear(x, self.random_matrix)  # type: ignore
        else:
            self.num_gp_features = num_deep_features
            self.jl = torch.nn.Identity()

        self.normalize_gp_features = normalize_gp_features
        if normalize_gp_features:
            self.normalize = torch.nn.LayerNorm(num_gp_features)

        self.rff = _RandomFourierFeatures(
            num_gp_features, num_random_features, feature_scale
        )
        self.beta = torch.nn.Linear(num_random_features, num_outputs)

        self.num_data = 0  # to be set later
        self.register_buffer("seen_data", torch.tensor(0))

        precision = torch.eye(num_random_features) * self.ridge_penalty
        self.register_buffer("precision", precision)

        self.recompute_covariance = True
        self.register_buffer("covariance", torch.eye(num_random_features))
        self.training_parameters_set = False

    def reset_precision_matrix(self):
        """
        Reset the precision matrix to the identity matrix times the ridge penalty.
        """
        identity = torch.eye(self.precision.shape[0], device=self.precision.device)
        self.precision = identity * self.ridge_penalty
        self.seen_data = torch.tensor(0)
        self.recompute_covariance = True

    @icontract.require(lambda num_data: num_data > 0)
    @icontract.require(
        lambda num_data, batch_size: (0 < batch_size) & (batch_size <= num_data)
    )
    def set_training_params(self, num_data, batch_size) -> None:
        """
        Set the training parameters for the Laplace approximation.

        Parameters
        ----------
        num_data : int
            The total number of data points.
        batch_size : int
            The batch size to use during training.
        """
        self.num_data = num_data
        self.train_batch_size = batch_size
        self.training_parameters_set = True

    @icontract.require(lambda self: self.mean_field_factor is not None)
    def mean_field_logits(self, logits, pred_cov):
        """
        Compute the mean-field logits for the Gaussian-Softmax approximation.

        Parameters
        ----------
        logits : torch.Tensor
            The logits tensor of shape (batch_size, num_outputs).
        pred_cov : torch.Tensor
            The predicted covariance matrix of shape (batch_size, batch_size).

        Returns
        -------
        torch.Tensor
            The mean-field logits tensor of shape (batch_size, num_outputs).
        """
        # Mean-Field approximation as alternative to MC integration of Gaussian-Softmax
        # Based on: https://arxiv.org/abs/2006.07584

        logits_scale = torch.sqrt(1.0 + torch.diag(pred_cov) * self.mean_field_factor)
        if self.mean_field_factor > 0:  # type: ignore
            logits = logits / logits_scale.unsqueeze(-1)

        return logits

    @icontract.require(lambda self: self.training_parameters_set)
    def forward(self, x):
        """
        Compute the forward pass of the Laplace approximation to the Gaussian Process.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, num_features).

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
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
                    torch.cholesky_inverse(u, out=self.covariance)  # type: ignore

                self.recompute_covariance = False

            with torch.no_grad():
                pred_cov = k @ ((self.covariance @ k.t()) * self.ridge_penalty)

            if self.mean_field_factor is None:
                return pred, pred_cov
            else:
                pred = self.mean_field_logits(pred, pred_cov)

        return pred


# -------------------------------------------------------------------------------
# EquineGP, below, demonstrates how to adapt that approach in EQUINE
@typechecked
class EquineGP(Equine):
    """
    An example of an EQUINE model that builds upon the approach in "Spectral Norm
    Gaussian Processes" (SNGP). This wraps any pytorch embedding neural network and provides
    the `forward`, `predict`, `save`, and `load` methods required by Equine.

    Parameters
    ----------
    embedding_model : torch.nn.Module
        Neural Network feature embedding.
    emb_out_dim : int
        The number of deep features from the feature embedding.
    num_classes : int
        The number of output classes this model predicts.
    use_temperature : bool, optional
        Whether to use temperature scaling after training.
    init_temperature : float, optional
        What to use as the initial temperature (1.0 has no effect).
    device : str, optional
        Either 'cuda' or 'cpu'.

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
        use_temperature: bool = False,
        init_temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(embedding_model)
        self.num_deep_features = emb_out_dim
        self.num_gp_features = emb_out_dim
        self.normalize_gp_features = True
        self.num_random_features = 1024
        self.num_outputs = num_classes
        self.mean_field_factor = 25
        self.ridge_penalty = 1
        self.feature_scale = 2
        self.use_temperature = use_temperature
        self.init_temperature = init_temperature
        self.register_buffer(
            "temperature", torch.Tensor(self.init_temperature * torch.ones(1))
        )
        self.model = _Laplace(
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
        self.device = torch.device(self.device_type)

    def train_model(
        self,
        dataset: TensorDataset,
        loss_fn: Callable,
        opt: torch.optim.Optimizer,
        num_epochs: int,
        batch_size: int = 64,
        calib_frac: float = 0.1,
        num_calibration_epochs: int = 2,
        calibration_lr: float = 0.01,
    ):
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
        batch_size : int, optional
            The number of samples to use per batch.
        calib_frac : float, optional
            Fraction of training data to use in temperature scaling.
        num_calibration_epochs : int, optional
            The desired number of epochs to use for temperature scaling.
        calibration_lr : float, optional
            Learning rate for temperature scaling.

        Returns
        -------
        Tuple[dict[str, Any], DataLoader]
            A tuple containing the training history and a dataloader for the calibration data.

        Notes
        -------
        - If `use_temperature` is True, temperature scaling will be used after training.
        - The calibration data is used to calibrate the temperature scaling.
        """
        if self.use_temperature:
            X, Y = dataset[:]
            train_x, calib_x, train_y, calib_y = train_test_split(
                X, Y, test_size=calib_frac, stratify=Y
            )  # TODO: Replace sklearn with torch call
            dataset = TensorDataset(train_x, train_y)
            self.temperature = torch.Tensor(
                self.init_temperature * torch.ones(1)
            ).type_as(self.temperature)

        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.model.set_training_params(
            len(train_loader.sampler), train_loader.batch_size
        )
        self.model.train()
        for _ in tqdm(range(num_epochs)):
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
        self.model.eval()

        calibration_loader = None
        if self.use_temperature:
            dataset_calibration = TensorDataset(calib_x, calib_y)
            calibration_loader = DataLoader(
                dataset_calibration,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
            )
            self.calibrate_temperature(
                calibration_loader, num_calibration_epochs, calibration_lr
            )

        _, train_y = dataset[:]
        date_trained = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        self.train_summary = generate_train_summary(self, train_y, date_trained)

        return self.train_summary, calibration_loader

    def calibrate_temperature(
        self,
        calibration_loader: DataLoader,
        num_calibration_epochs: int = 1,
        calibration_lr: float = 0.01,
    ) -> None:
        """
        Fine-tune the temperature after training. Note this function is also run at the conclusion of train_model.

        Parameters
        ----------
        calibration_loader : DataLoader
            Data loader returned by train_model.
        num_calibration_epochs : int, optional
            Number of epochs to tune temperature.
        calibration_lr : float, optional
            Learning rate for temperature optimization.
        """
        self.temperature.requires_grad = True
        loss_fn = torch.nn.functional.cross_entropy
        optimizer = torch.optim.Adam([self.temperature], lr=calibration_lr)
        for _ in range(num_calibration_epochs):
            for (xs, labels) in calibration_loader:
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
        with torch.no_grad():
            preds = self.model(X)
        return preds / self.temperature

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
        eq_out = EquineOutput(
            classes=preds, ood_scores=ood_score, embeddings=torch.Tensor([])
        )  # TODO return embeddings
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
            "use_temperature": self.use_temperature,
            "init_temperature": self.temperature.item(),
            "device": self.device_type,
        }

        jit_model = torch.jit.script(self.model.feature_extractor)  # type: ignore
        buffer = io.BytesIO()
        torch.jit.save(jit_model, buffer)  # type: ignore
        buffer.seek(0)

        laplace_sd = self.model.state_dict()
        keys_to_delete = []
        for key in laplace_sd:
            if "feature_extractor" in key:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del laplace_sd[key]

        save_data = {
            "settings": model_settings,
            "num_data": self.model.num_data,
            "train_batch_size": self.model.train_batch_size,
            "laplace_model_save": laplace_sd,
            "embed_jit_save": buffer,
            "train_summary": self.train_summary,
        }

        torch.save(save_data, path)  # TODO allow model checkpointing

    @classmethod
    def load(cls, path: str) -> Equine:  # noqa: F821 # type: ignore
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
        model_save = torch.load(path)
        jit_model = torch.jit.load(model_save["embed_jit_save"])  # type: ignore
        eq_model = cls(jit_model, **model_save["settings"])

        eq_model.train_summary = model_save["train_summary"]
        eq_model.model.load_state_dict(model_save["laplace_model_save"], strict=False)
        eq_model.model.seen_data = model_save["laplace_model_save"]["seen_data"]

        eq_model.model.set_training_params(
            model_save["num_data"], model_save["train_batch_size"]
        )
        eq_model.eval()

        return eq_model
