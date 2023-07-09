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
        model_embeddings = self.embedding_model(X)
        head_embeddings = self.model_head(model_embeddings)
        return head_embeddings

    @icontract.require(lambda self: self.support_embeddings is not None)
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
            class_prototype = torch.mean(self.support_embeddings[label], dim=0)  # type: ignore
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
        self, class_cov_dict: dict[float, torch.Tensor]
    ) -> dict[float, torch.Tensor]:
        """
        Method to add regularization to each class covariance matrix based on the selected regularization type.

        Parameters
        ----------
        class_cov_dict : dict[float, torch.Tensor]
            A dictionary containing each class and the corresponding covariance matrix.

        Returns
        -------
        dict[float, torch.Tensor]
            Dictionary containing the regularized class covariance matrices.
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
        self, class_cov_dict: dict[float, torch.Tensor]
    ) -> torch.Tensor:
        """
        Method to calculate a shared covariance matrix.

        The shared covariance matrix is calculated as the weighted average of the class covariance matrices,
        where the weights are the number of support examples for each class. This is useful when the number of
        support examples for each class is small.

        Parameters
        ----------
        class_cov_dict : dict[float, torch.Tensor]
            A dictionary containing each class and the corresponding covariance matrix.

        Returns
        -------
        torch.Tensor
            Tensor containing the shared covariance matrix.
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
            dist = torch.sqrt(
                torch.abs(dist.permute(1, 0)) + self.epsilon
            )  # examples x classes
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
            Input tensor of queires for generating predictions.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing class probability predictions, and class distances from prototypes.
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
        """
        Method to update the support examples, and all the calculations that rely on them.

        Parameters
        ----------
        support : OrderedDict
            Ordered dict containing class labels and their associated support examples.
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
    A class representing an EQUINE model that utilizes protonets and (optionally) relative Mahalanobis distances
    to generate OOD and model confidence scores. This wraps any pytorch embedding neural network
    and provides the `forward`, `predict`, `save`, and `load` methods required by Equine.

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
    """

    def __init__(
        self,
        embedding_model,
        emb_out_dim: int,
        cov_type: CovType = CovType.UNIT,
        relative_mahal: bool = True,
        use_temperature: bool = False,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__(embedding_model)
        self.cov_type = cov_type
        self.cov_reg_type = COV_REG_TYPE
        self.relative_mahal = relative_mahal
        self.emb_out_dim = emb_out_dim
        self.epsilon = DEFAULT_EPSILON
        self.outlier_score_kde = None
        self.model_summary = None
        self.use_temperature = use_temperature
        self.init_temperature = init_temperature
        self.register_buffer(
            "temperature", torch.Tensor(self.init_temperature * torch.ones(1))
        )

        self.model = Protonet(
            embedding_model,
            self.emb_out_dim,
            self.cov_type,
            self.cov_reg_type,
            self.epsilon,
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
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
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
            self.temperature = torch.Tensor(
                self.init_temperature * torch.ones(1)
            ).type_as(self.temperature)

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

        if self.use_temperature:
            self.calibrate_temperature(
                calib_x, calib_y, num_calibration_epochs, calibration_lr
            )

        date_trained = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        self.train_summary = generate_train_summary(self, train_y, date_trained)
        return self.train_summary, calib_x, calib_y

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
            dists = dists / self.temperature
            loss = torch.nn.functional.cross_entropy(
                torch.neg(dists), calib_y.to(torch.long)
            )
            loss.backward()
            optimizer.step()
        self.temperature.requires_grad = False

    @icontract.ensure(lambda self: self.model.support_embeddings is not None)
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
        self.outlier_score_kde = OrderedDict.fromkeys(
            self.model.support_embeddings.keys()
        )

        for label in self.outlier_score_kde:
            class_ood_dists = ood_dists[calib_y == int(label)].detach().numpy()
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

        return EquineOutput(classes=preds, ood_scores=ood_scores, embeddings=X_embed)

    @icontract.require(lambda calib_frac: (calib_frac > 0.0) and (calib_frac < 1.0))
    def update_support(
        self, support_x: torch.Tensor, support_y: torch.Tensor, calib_frac: float
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

        Returns
        -------
        None
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
