# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Callable, Any
import io
from enum import Enum
from collections import OrderedDict
from datetime import datetime

import icontract
import torch
from torch.utils.data import TensorDataset
from typeguard import typechecked
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from tqdm import tqdm

from .equine import Equine, EquineOutput
from .utils import generate_support, generate_episode, generate_train_summary


#####################################
class CovType(Enum):
    UNIT = "unit"
    DIAGONAL = "diag"
    FULL = "full"


PRED_COV_TYPE = CovType.DIAGONAL
OOD_COV_TYPE = CovType.DIAGONAL
DEFAULT_EPSILON = 1e-5
COV_REG_TYPE = "epsilon"


###############################################
@typechecked
class _Protonet(torch.nn.Module):
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
    ) -> None:
        """Protonet class constructor
        :param embedding_model: The pytorch embedding model to generate logits with
        :param emb_out_dim: Dimention size of given embedding model's output
        :param cov_type: Type of covariance to use when computing distances [unit, diag, full]
        :param cov_reg_type: type of regularization to use when generating the covariance matrix [epsilon, shared]
        :param epsilon: Epsilon value to use for covariance regularization
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.cov_type = cov_type
        self.cov_reg_type = cov_reg_type
        self.epsilon = epsilon
        self.emb_out_dim = emb_out_dim

        self.support = None
        # self.support_embeddings = None
        self.model_head = self.create_model_head(emb_out_dim)

    def create_model_head(self, emb_out_dim: int):
        """Method for adding a pytorch layer on top of given embedding model
        :param emb_out_dim: Dimention size of embedding model output
        :return torch.nn.Linear: Returns the created pytorch model layer
        """
        return torch.nn.Linear(emb_out_dim, emb_out_dim)

    def compute_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        """Method for calculating model embeddings using both the given embedding model and the added model head.
        :param X: Input tensor to compute embeddings on
        :return torch.Tensor: Returns fully computed embedding tensors for given X tensor
        """
        model_embeddings = self.embedding_model(X)
        head_embeddings = self.model_head(model_embeddings)
        return head_embeddings

    @icontract.require(lambda self: self.support_embeddings is not None)
    def compute_prototypes(self) -> torch.Tensor:
        """Method for computing class prototypes based on given support examples
        :return torch.Tensor: Tensors of prototypes for each of the given classes in the support
        """
        # Compute prototype for each class
        proto_list = []
        for label in self.support_embeddings:  # look at doing functorch
            class_prototype = torch.mean(self.support_embeddings[label], dim=0)  # type: ignore
            proto_list.append(class_prototype)

        prototypes = torch.stack(proto_list)

        return prototypes

    @icontract.require(lambda self: len(self.support_embeddings) > 0)
    def compute_covariance(self, cov_type: CovType) -> torch.Tensor:
        """Method for generating the regularized support example covariance matrix used for calculating distances
        :param cov_type: Type of covariance to use [unit, diag, full]
        :return torch.Tensor: Tensor containing the generated regularized covariance matrix
        """
        class_cov_dict = OrderedDict().fromkeys(self.support_embeddings.keys())
        for label in self.support_embeddings.keys():
            class_covariance = self.compute_covariance_by_type(
                cov_type, self.support_embeddings[label]
            )
            class_cov_dict[label] = class_covariance

        reg_covariance_dict = self.regularize_covariance(class_cov_dict)
        reg_covariance = torch.stack(list(reg_covariance_dict.values()))

        return reg_covariance  # TODO try putting everything on GPU with .to() and see if faster

    def compute_covariance_by_type(
        self, cov_type: CovType, embedding: torch.Tensor
    ) -> torch.Tensor:
        """Method to select appropriate covariance matrix type based on cov_type
        :param cov_type: Type of covariance to use [unit, diag, full]
        :param embedding: embedding tensor to use when generating the covariance matrix
        :return torch.Tensor: Tensor containing the requested covariance matrix
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
        self, class_cov_dict: dict[Any, torch.Tensor]
    ) -> dict[Any, torch.Tensor]:
        """Method to add regularization to each class covariance matrix based on the selected regularization type
        :param class_cov_dict: A dictionary containing each class and the corresponding covariance matrix
        :return dict: Dictonary containing the regularized class covariance matrices
        """
        if self.cov_type == CovType.FULL:
            regularization = torch.diag(self.epsilon * torch.ones(self.emb_out_dim))
        elif self.cov_type == CovType.DIAGONAL:
            regularization = self.epsilon * torch.ones(self.emb_out_dim)
        elif self.cov_type == CovType.UNIT:
            regularization = torch.zeros(self.emb_out_dim)
        else:
            raise ValueError("Unknown Covariance Type")

        if self.cov_reg_type == "shared":
            if self.cov_type != CovType.FULL and self.cov_type != CovType.DIAGONAL:
                raise ValueError(
                    "Covariance type FULL and DIAGONAL are incompatible with shared regularization"
                )

            shared_covariance = self.compute_shared_covariance(class_cov_dict)

            for label in self.support_embeddings:
                num_class_support = self.support_embeddings[label].shape[0]
                lamb = num_class_support / (num_class_support + 1)

                class_cov_dict[label] = (
                    lamb * class_cov_dict[label]
                    + (1 - lamb) * shared_covariance
                    + regularization
                )

        elif self.cov_reg_type == "epsilon":
            for label in self.support_embeddings:
                class_cov_dict[label] = class_cov_dict[label] + regularization

        return class_cov_dict

    def compute_shared_covariance(
        self, class_cov_dict: dict[Any, torch.Tensor]
    ) -> torch.Tensor:
        """Method to calculate a shared covariance matrix
        :param class_cov_dict: A dictionary containing each class and the corresponding covariance matrix
        :return torch.Tensor: Tensor containing the shared covariance matrix
        """
        total_support = sum([x.shape[0] for x in class_cov_dict.values()])

        if self.cov_type == CovType.FULL:
            shared_covariance = torch.zeros((self.emb_out_dim, self.emb_out_dim))
        elif self.cov_type == CovType.DIAGONAL:
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

    @typechecked
    @icontract.require(lambda X_embed, mu: X_embed.shape[-1] == mu.shape[-1])
    @icontract.ensure(lambda result: torch.all(result >= 0))
    def compute_distance(
        self, X_embed: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        """Method to compute the distances to class prototypes for the given embeddings
        :param X_embed: The embeddings to compute the distances on
        :param mu: The class prototypes
        :param cov: The support covariance matrix
        :return torch.Tensor: The calculated distances from each of the class prototypes for the given embeddings
        """
        _queries = torch.unsqueeze(X_embed, 1)  # examples x 1 x dimension
        diff = torch.sub(mu, _queries) ** 2  # examples x classes x dimension

        if len(cov.shape) == 2:  # (diagonal covariance)
            # examples x classes x dimension
            dist = torch.nan_to_num(torch.div(diff, cov))
            dist = torch.sum(dist, dim=2)  # examples x classes
            dist = dist.squeeze(dim=1)
            dist = torch.sqrt(dist + self.epsilon)  # examples x classes
        else:  # len(cov.shape) == 3: (full covariance)
            diff = diff.permute(1, 2, 0)  # classes x dimension x examples
            sol = torch.linalg.lstsq(cov, diff, rcond=10 ** (-4))
            sol = sol.solution  # classes x dimension x examples
            dist = torch.sum(diff * sol, dim=1)  # classes x examples
            dist = torch.sqrt(dist.permute(1, 0) + self.epsilon)  # examples x classes
            dist = dist.squeeze(dim=1)

        return dist

    @typechecked
    def compute_classes(self, distances: torch.Tensor) -> torch.Tensor:
        """Method to compute predicted classes from distances via a softmax function
        :param distances: The distances of embeddings to class prototypes
        :return torch.Tensor: Tensor of class predictions
        """
        softmax = torch.nn.functional.softmax(torch.neg(distances), dim=-1)
        return softmax

    @typechecked
    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """_Protonet forward function, generates class probability predictions
        :param X: Input tensor for generating predictions
        :return tuple[torch.Tensor, torch.Tensor]: Tuple containing class probability predictions, and class distances from prototypes
        """
        if self.support is None or self.support_embeddings is None:
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

    def update_support(self, support: OrderedDict) -> None:
        """Method to update the support examples, and all the calculations that rely on them
        :param support: Ordered dict containing class labels and their associated support examples
        """
        self.support = support  # TODO torch.nn.ParameterDict(support)

        support_embs = OrderedDict().fromkeys(support.keys())
        for label in support:
            support_embs[label] = self.compute_embeddings(support[label])

        self.support_embeddings = (
            support_embs  # TODO torch.nn.ParameterDict(support_embs)
        )

        self.prototypes = self.compute_prototypes()

        if self.training is False:
            self.compute_global_moments()
            self.covariance = self.compute_covariance(cov_type=PRED_COV_TYPE)
        else:
            self.covariance = self.compute_covariance(cov_type=self.cov_type)

    @typechecked
    @icontract.require(lambda self: self.support_embeddings is not None)
    def compute_global_moments(self) -> None:
        """Method to calculate the global moments of the support embeddings for use in OOD score generation"""
        embeddings = torch.cat(list(self.support_embeddings.values()))
        self.global_covariance = torch.unsqueeze(
            self.compute_covariance_by_type(OOD_COV_TYPE, embeddings), dim=0
        )
        self.global_mean = torch.mean(embeddings, dim=0)


###############################################
@typechecked
class EquineProtonet(Equine):
    """
    An example of an EQUINE model that utilizes protonets and relative mahalanobis distances
    to generate OOD and model confidence scores. This wraps any pytorch embedding neural network
    and provides the `forward`, `predict`, `save`, and `load` methods required by Equine.
    """

    def __init__(
        self,
        embedding_model,
        emb_out_dim: int,
        cov_type: CovType = CovType.UNIT,
        relative_mahal: bool = True,
    ) -> None:
        """EquineProtonet constructor
        :param embedding_model: Neural Network feature embedding model
        :param emb_out_dim: The number of output features from the embedding model
        :param cov_type: The type of covariance to use when training the protonet [UNIT, DIAG, FULL]
        :param relative_mahal: Use relative mahalanobis distance for OOD calculations. If false, uses standard mahalanobis distance instead
        """
        super().__init__(embedding_model)
        self.cov_type = cov_type
        self.cov_reg_type = COV_REG_TYPE
        self.relative_mahal = relative_mahal
        self.emb_out_dim = emb_out_dim
        self.epsilon = DEFAULT_EPSILON
        self.outlier_score_kde = None
        self.model_summary = None

        self.model = _Protonet(
            embedding_model,
            self.emb_out_dim,
            self.cov_type,
            self.cov_reg_type,
            self.epsilon,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """EquineProtonet forward function, generates logits for classification
        :param X: Input tensor for generating predictions
        :return torch.Tensor: Output class predictions
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
    ) -> dict[str, Any]:
        """Train or fine-tune an EquineProtonet model
        :param dataset: Input pytorch TensorDataset of training data for model
        :param num_episodes: The desired number of episodes to use for training
        :param calib_frac: Fraction of given training data to reserve for model calibration
        :param support_size: Number of support examples to generate for each class
        :param way: Number of classes to train on per episode
        :param episode_size: Number of examples to use per episode
        :param loss_fn: A pytorch loss function, eg., torch.nn.CrossEntropyLoss()
        :param opt_class: A pytorch optimizer, e.g., torch.optim.Adam
        """
        self.train()

        X, Y = dataset[:]

        train_x, calib_x, train_y, calib_y = train_test_split(
            X, Y, test_size=calib_frac, stratify=Y
        )  # TODO: Replace sklearn with torch call
        optimizer = opt_class(self.parameters())

        for i in tqdm(range(num_episodes)):
            optimizer.zero_grad()

            support, episode_x, episode_y = generate_episode(
                train_x, train_y, support_size, way, episode_size
            )
            self.model.update_support(support)

            _, dists = self.model(episode_x)
            loss_value = loss_fn(torch.neg(dists), episode_y)
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

        date_trained = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        self.train_summary = generate_train_summary(self, train_y, date_trained)
        return self.train_summary

    @icontract.ensure(lambda self: self.model.support_embeddings is not None)
    def _fit_outlier_scores(
        self, ood_dists: torch.Tensor, calib_y: torch.Tensor
    ) -> None:
        """Private function to fit outlier scores with a kernel density estimate (KDE)
        :param ood_dists: Tensor of computed OOD distances
        :param calib_y: Tensor of class labels for ood_dists examples
        """
        self.outlier_score_kde = OrderedDict.fromkeys(
            self.model.support_embeddings.keys()
        )

        for label in self.outlier_score_kde:
            class_ood_dists = ood_dists[calib_y == int(label)].detach().numpy()
            class_kde = gaussian_kde(class_ood_dists)  # TODO convert to torch func
            self.outlier_score_kde[label] = class_kde

    def _compute_outlier_scores(self, ood_dists, predictions) -> torch.Tensor:
        """Private function to compute OOD scores using the calculated kernel density estimate (KDE)
        :param ood_dists: Tensor of computed OOD distances
        :param predictions: Tensor of model protonet predictions
        :return torch.Tensor: Tensor of OOD scores for the given examples
        """
        ood_scores = torch.zeros_like(ood_dists)
        for i in range(len(predictions)):
            # Use KDE and RMD corresponding to the predicted class
            predicted_class = int(torch.argmax(predictions[i, :]))
            p_value = self.outlier_score_kde[int(predicted_class)].integrate_box_1d(
                ood_dists[i].detach().numpy(), torch.inf
            )
            ood_scores[i] = 1.0 - p_value

        return ood_scores

    @icontract.ensure(lambda result: len(result) > 0)
    def _compute_ood_dist(
        self,
        X_embeddings: torch.Tensor,
        predictions: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """Private function to compute OOD distances using a distance function
        :param X_embeddings: Tensor of example embeddings
        :param predictions: Tensor of model protonet predictions for the given embeddings
        :param distances: Tensor of calculated protonet distances for the given embeddings
        :return torch.Tensor: Tensor of OOD distances for the given embeddings
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
        """Predict function for EquineProtonet, inherited and implemented from Equine
        :param X: Input tensor
        :return[EquineOutput] : Output object containing prediction probabilities and OOD scores
        """
        X_embed = self.model.compute_embeddings(X)
        if X_embed.shape == torch.Size([self.model.emb_out_dim]):
            X_embed = X_embed.unsqueeze(dim=0)  # Handle single examples
        preds, dists = self.model(X)
        ood_dist = self._compute_ood_dist(X_embed, preds, dists)
        ood_scores = self._compute_outlier_scores(ood_dist, preds)

        return EquineOutput(classes=preds, ood_scores=ood_scores, embeddings=X_embed)

    @icontract.require(lambda calib_frac: (calib_frac > 0.0) and (calib_frac < 1.0))
    def update_support(
        self, support_x: torch.Tensor, support_y: torch.Tensor, calib_frac: float
    ) -> None:
        """Function to update protonet support examples with given examples
        :param support_x: Tensor containing support examples for protonet
        :param support_y: Tensor containing labels for given support examples
        :param calib_frac: Fraction of given support data to use for OOD calibration
        """

        support_x, calib_x, support_y, calib_y = train_test_split(
            support_x, support_y, test_size=calib_frac, stratify=support_y
        )
        labels, counts = torch.unique(support_y, return_counts=True)
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

    def save(self, path: str) -> None:
        """Function to save all model parameters to a file
        :param path: Filename to write the model
        """
        model_settings = {
            "cov_type": self.cov_type,
            "emb_out_dim": self.emb_out_dim,
        }

        jit_model = torch.jit.script(self.model.embedding_model)
        buffer = io.BytesIO()
        torch.jit.save(jit_model, buffer)
        buffer.seek(0)

        save_data = {
            "settings": model_settings,
            "support": self.model.support,
            "outlier_kde": self.outlier_score_kde,
            "model_head_save": self.model.model_head.state_dict(),
            "embed_jit_save": buffer,
            "train_summary": self.train_summary,
        }

        torch.save(save_data, path)  # TODO allow model checkpointing

    @classmethod
    def load(cls, path: str) -> Equine:  # noqa: F821 TODO typehint doesnt want to work?
        """Function to load previously saved EquineProtonet model
        :param path: input filename
        :return[EquineGP] : The reconsituted EquineProtonet object
        """
        model_save = torch.load(path)
        support = model_save["support"]
        jit_model = torch.jit.load(model_save["embed_jit_save"])
        eq_model = cls(jit_model, **model_save["settings"])

        eq_model.model.model_head.load_state_dict(model_save["model_head_save"])
        eq_model.eval()
        eq_model.model.update_support(support)
        eq_model.outlier_score_kde = model_save["outlier_kde"]
        eq_model.train_summary = model_save["train_summary"]

        return eq_model
