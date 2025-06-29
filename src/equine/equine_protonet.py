# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any, Optional

import icontract
import io
import numpy as np
import torch
import warnings
from beartype import beartype
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from scipy.stats import gaussian_kde
from torch.utils.data import TensorDataset
from tqdm import tqdm

from .equine import Equine, EquineOutput
from .utils import (
    generate_episode,
    generate_support,
    generate_train_summary,
    mahalanobis_distance_nosq,
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

    def create_model_head(self, emb_out_dim: int) -> torch.nn.Linear:
        """
        Method for adding a PyTorch layer on top of the given embedding model. This layer
        is intended to offer extra degrees of freedom for distance learning in the embedding space.

        Parameters
        ----------
        emb_out_dim : int
            Dimension size of the embedding model output.

        Returns
        -------
        torch.nn.Linear
            The created PyTorch model layer.
        """
        return torch.nn.Linear(emb_out_dim, emb_out_dim)

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
        model_embeddings = self.embedding_model(X.to(self.device))
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
        cov_type : str
            Type of covariance to use. Options are ['unit', 'diag', 'full'].
        embedding : torch.Tensor
            Embedding tensor to use when generating the covariance matrix.

        Returns
        -------
        torch.Tensor
            Tensor containing the requested covariance matrix.
        """
        if cov_type == CovType.FULL:
            class_covariance = torch.cov(embedding.T)
        elif cov_type == CovType.DIAGONAL:
            class_covariance = torch.var(embedding, dim=0)
        elif cov_type == CovType.UNIT:
            class_covariance = torch.ones(self.emb_out_dim)
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

        Returns
        -------
        dict[float, torch.Tensor]
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
            shared_covariance = torch.zeros((self.emb_out_dim, self.emb_out_dim))
        elif cov_type == CovType.DIAGONAL:
            shared_covariance = torch.zeros(self.emb_out_dim)
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
            div = torch.div(sq_diff.to(self.device), cov.to(self.device))
            dist = torch.nan_to_num(div)
            dist = torch.sum(dist, dim=2)  # examples x classes
            dist = dist.squeeze(dim=1)
            dist = torch.sqrt(dist + self.epsilon)  # examples x classes
        else:  # len(cov.shape) == 3: (full covariance)
            diff = diff.permute(1, 2, 0)  # classes x dimension x examples
            dist = mahalanobis_distance_nosq(diff, cov)
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
            Tensor of class predictions.
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
        """
        if len(self.support) == 0 or len(self.support_embeddings) == 0:
            raise ValueError(
                "No support examples found. Protonet Model requires model support to \
                    be set with the 'update_support()' method before calling forward."
            )

        X_embed = self.compute_embeddings(X)
        if X_embed.shape == torch.Size([self.emb_out_dim]):
            X_embed = X_embed.unsqueeze(dim=0)  # handle single examples
        distances = self.compute_distance(X_embed, self.prototypes, self.covariance)
        classes = self.compute_classes(distances)

        return classes, distances

    def update_support(self, support: OrderedDict[int, torch.Tensor]) -> None:
        """
        Method to update the support examples, and all the calculations that rely on them.

        Parameters
        ----------
        support : OrderedDict
            Ordered dict containing class labels and their associated support examples.
        """
        self.support = support  # TODO torch.nn.ParameterDict(support)

        support_embs = OrderedDict().fromkeys(support.keys(), torch.Tensor())
        for label in support:
            support_embs[label] = self.compute_embeddings(support[label])

        self.support_embeddings = (
            support_embs  # TODO torch.nn.ParameterDict(support_embs)
        )

        self.prototypes: torch.Tensor = self.compute_prototypes()

        if self.training is False:
            self.compute_global_moments()
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
        feature_names: Optional[list[str]] = None,
        label_names: Optional[list[str]] = None,
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
        """
        super().__init__(
            embedding_model,
            device=device,
            feature_names=feature_names,
            label_names=label_names,
        )
        self.cov_type = cov_type
        self.cov_reg_type = COV_REG_TYPE
        self.relative_mahal = relative_mahal
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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generates logits for classification based on the input tensor.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor for generating predictions.

        Returns
        -------
        torch.Tensor
            The output class predictions.
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
        tuple[dict[str, Any], torch.Tensor, torch.Tensor]
            A tuple containing the model summary, the held out calibration data, and the calibration labels.
        """
        self.train()

        if self.use_temperature:
            self.temperature: torch.Tensor = torch.Tensor(
                self.init_temperature * torch.ones(1)
            ).type_as(self.temperature)

        X, Y = dataset[:]

        self.validate_feature_label_names(X.shape[-1], torch.unique(Y).shape[0])

        train_x, calib_x, train_y, calib_y = stratified_train_test_split(
            X, Y, test_size=calib_frac
        )
        optimizer = opt_class(self.parameters())

        train_x.to(self.device)
        train_y.to(self.device)
        calib_x.to(self.device)
        calib_y.to(self.device)

        for i in tqdm(range(num_episodes)):
            optimizer.zero_grad()

            support, episode_x, episode_y = generate_episode(
                train_x, train_y, support_size, way, episode_size
            )
            self.model.update_support(support)

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

        self.model.update_support(
            full_support
        )  # update support with final selected examples

        X_embed = self.model.compute_embeddings(calib_x)
        pred_probs, dists = self.model(calib_x)
        ood_dists = self._compute_ood_dist(X_embed, pred_probs, dists)
        self._fit_outlier_scores(ood_dists, calib_y)

        if self.use_temperature:
            self.calibrate_temperature(
                calib_x, calib_y, num_calibration_epochs, calibration_lr
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
        Fine-tune the temperature after training. Note that this function is also run at the conclusion of train_model.

        Parameters
        ----------
        calib_x : torch.Tensor
            Training data to be used for temperature calibration.
        calib_y : torch.Tensor
            Labels corresponding to `calib_x`.
        num_calibration_epochs : int, optional
            Number of epochs to tune temperature, by default 1.
        calibration_lr : float, optional
            Learning rate for temperature optimization, by default 0.01.

        Returns
        -------
        None
        """
        self.temperature.requires_grad = True
        optimizer = torch.optim.Adam([self.temperature], lr=calibration_lr)
        for t in range(num_calibration_epochs):
            optimizer.zero_grad()
            with torch.no_grad():
                pred_probs, dists = self.model(calib_x)
            dists = dists.to(self.device) / self.temperature.to(self.device)
            loss = torch.nn.functional.cross_entropy(
                torch.neg(dists).to(self.device), calib_y.to(torch.long).to(self.device)
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

        Parameters
        ----------
        ood_dists : torch.Tensor
            Tensor of computed OOD distances.
        calib_y : torch.Tensor
            Tensor of class labels for `ood_dists` examples.

        Returns
        -------
        None
        """
        for label in self.model.support_embeddings.keys():
            class_ood_dists = ood_dists[calib_y == int(label)].cpu().detach().numpy()
            class_kde = gaussian_kde(class_ood_dists)  # TODO convert to torch func
            self.outlier_score_kde[label] = class_kde

    def _compute_outlier_scores(self, ood_dists, predictions) -> torch.Tensor:
        """
        Private function to compute OOD scores using the calculated kernel density estimate (KDE).

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
        """
        ood_scores = torch.zeros_like(ood_dists)
        for i in range(len(predictions)):
            # Use KDE and RMD corresponding to the predicted class
            predicted_class = int(torch.argmax(predictions[i, :]))
            p_value = self.outlier_score_kde[int(predicted_class)].integrate_box_1d(
                ood_dists[i].detach().numpy(), np.inf
            )
            ood_scores[i] = 1.0 - np.clip(p_value, 0.0, 1.0)

        return ood_scores

    @icontract.ensure(lambda result: len(result) > 0)
    def _compute_ood_dist(
        self,
        X_embeddings: torch.Tensor,
        predictions: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Private function to compute OOD distances using a distance function.

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
            ood_dist = distances.gather(1, preds) - null_distance
        else:
            ood_dist = distances.gather(1, preds)

        ood_dist = torch.reshape(ood_dist, (-1,))
        return ood_dist

    def predict(self, X: torch.Tensor) -> EquineOutput:
        """Predict function for EquineProtonet, inherited and implemented from Equine.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.

        Returns
        -------
        EquineOutput
            Output object containing prediction probabilities and OOD scores.
        """
        X_embed = self.model.compute_embeddings(X)
        if X_embed.shape == torch.Size([self.model.emb_out_dim]):
            X_embed = X_embed.unsqueeze(dim=0)  # Handle single examples
        preds, dists = self.model(X)
        if self.use_temperature:
            dists = dists / self.temperature
            preds = torch.softmax(torch.negative(dists), dim=1)
        ood_dist = self._compute_ood_dist(X_embed, preds, dists)
        ood_scores = self._compute_outlier_scores(ood_dist, preds)

        self.validate_feature_label_names(X.shape[-1], preds.shape[-1])

        return EquineOutput(classes=preds, ood_scores=ood_scores, embeddings=X_embed)

    @icontract.require(lambda calib_frac: (calib_frac > 0.0) and (calib_frac < 1.0))
    def update_support(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        calib_frac: float,
        label_names: Optional[list[str]] = None,
    ) -> None:
        """Function to update protonet support examples with given examples.

        Parameters
        ----------
        support_x : torch.Tensor
            Tensor containing support examples for protonet.
        support_y : torch.Tensor
            Tensor containing labels for given support examples.
        calib_frac : float
            Fraction of given support data to use for OOD calibration.
        label_names : list[str], optional
            List of strings of the names of the labels (ex ["streaming", "voip", ...])

        Returns
        -------
        None
        """

        support_x, calib_x, support_y, calib_y = stratified_train_test_split(
            support_x, support_y, test_size=calib_frac
        )
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

        self.model.update_support(support)

        X_embed = self.model.compute_embeddings(calib_x)
        preds, dists = self.model(calib_x)
        ood_dists = self._compute_ood_dist(X_embed, preds, dists)

        self._fit_outlier_scores(ood_dists, calib_y)

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

        Parameters
        ----------
        path : str
            Filename to write the model.

        Returns
        -------
        None
        """
        model_settings = {
            "cov_type": self.cov_type,
            "emb_out_dim": self.emb_out_dim,
            "use_temperature": self.use_temperature,
            "init_temperature": self.temperature.item(),
            "relative_mahal": self.relative_mahal,
        }

        jit_model = torch.jit.script(self.model.embedding_model)
        buffer = io.BytesIO()
        torch.jit.save(jit_model, buffer)
        buffer.seek(0)

        save_data = {
            "embed_jit_save": buffer,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "model_head_save": self.model.model_head.state_dict(),
            "outlier_kde": self.outlier_score_kde,
            "settings": model_settings,
            "support": self.model.support,
            "train_summary": self.train_summary,
        }

        torch.save(save_data, path)  # TODO allow model checkpointing

    @classmethod
    def load(cls, path: str) -> Equine:  # noqa: F821
        """
        Load a previously saved EquineProtonet model.

        Parameters
        ----------
        path : str
            The filename of the saved model.

        Returns
        -------
        EquineProtonet
            The reconstituted EquineProtonet object.
        """
        model_save = torch.load(path, weights_only=False)
        support = model_save.get("support")
        jit_model = torch.jit.load(model_save.get("embed_jit_save"))
        eq_model = cls(jit_model, **model_save.get("settings"))

        eq_model.model.model_head.load_state_dict(model_save.get("model_head_save"))
        eq_model.eval()
        eq_model.model.update_support(support)

        eq_model.feature_names = model_save.get("feature_names")
        eq_model.label_names = model_save.get("label_names")
        eq_model.outlier_score_kde = model_save.get("outlier_kde")
        eq_model.train_summary = model_save.get("train_summary")

        return eq_model
