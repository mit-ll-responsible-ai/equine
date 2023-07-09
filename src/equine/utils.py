# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Any, List, Union, Tuple, Dict
import icontract
import torch
from typeguard import typechecked
from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torchmetrics.classification import MulticlassCalibrationError

from .equine_output import EquineOutput
from .equine import Equine


@icontract.require(lambda y_hat, y_test: y_hat.size(dim=0) == y_test.size(dim=0))
@icontract.ensure(lambda result: result >= 0.0)
@typechecked
def brier_score(y_hat: torch.Tensor, y_test: torch.Tensor) -> float:
    """
    Compute the Brier score for a multiclass problem.

    Parameters
    ----------
    y_hat : torch.Tensor
        Probabilities for each class.
    y_test : torch.Tensor
        Integer argument class labels (ground truth).

    Returns
    -------
    float
        Brier score.
    """
    (_, num_classes) = y_hat.size()
    one_hot_y_test = torch.nn.functional.one_hot(y_test, num_classes=num_classes)
    bs = torch.mean(torch.sum((y_hat - one_hot_y_test) ** 2, dim=1)).item()
    return bs


@icontract.require(lambda y_hat, y_test: y_hat.size(dim=0) == y_test.size(dim=0))
@icontract.ensure(lambda result: result <= 1.0)
@typechecked
def brier_skill_score(y_hat: torch.Tensor, y_test: torch.Tensor) -> float:
    """
    Compute the Brier skill score as compared to randomly guessing.

    Parameters
    ----------
    y_hat : torch.Tensor
        Probabilities for each class.
    y_test : torch.Tensor
        Integer argument class labels (ground truth).

    Returns
    -------
    float
        Brier skill score.
    """
    (_, num_classes) = y_hat.size()
    random_guess = (1.0 / num_classes) * torch.ones(y_hat.size())
    bs0 = brier_score(random_guess, y_test)
    bs1 = brier_score(y_hat, y_test)
    bss = 1.0 - bs1 / bs0
    return bss


@icontract.require(lambda y_hat, y_test: y_hat.size(dim=0) == y_test.size(dim=0))
@icontract.ensure(lambda result: (0.0 <= result) and (result <= 1.0))
@typechecked
def expected_calibration_error(y_hat: torch.Tensor, y_test: torch.Tensor) -> float:
    """
    Compute the expected calibration error (ECE) for a multiclass problem.

    Parameters
    ----------
    y_hat : torch.Tensor
        Probabilities for each class.
    y_test : torch.Tensor
        Class label indices (ground truth).

    Returns
    -------
    float
        Expected calibration error.
    """
    (_, num_classes) = y_hat.size()
    metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=25, norm="l1")
    ece = metric(y_hat, y_test).item()
    return ece


@icontract.require(
    lambda train_y, selected_labels: len(selected_labels) <= len(train_y)
)
@icontract.ensure(
    lambda result, selected_labels: set(result.keys()).issubset(set(selected_labels))
)
@typechecked
def _get_shuffle_idxs_by_class(
    train_y: torch.Tensor, selected_labels: List
) -> dict[Any, torch.Tensor]:
    """
    Internal helper function to randomly select indices of example classes for a given
    set of labels.

    Parameters
    ----------
    train_y : torch.Tensor
        Label data.
    selected_labels : List
        List of unique labels found in the label data.

    Returns
    -------
    Dict[Any, torch.Tensor]
        Tensor of indices corresponding to each label.
    """
    shuffled_idxs_by_class = OrderedDict()
    for label in selected_labels:
        label_idxs = torch.argwhere(train_y == label).squeeze()
        shuffled_idxs_by_class[label] = label_idxs[torch.randperm(label_idxs.shape[0])]

    return shuffled_idxs_by_class


@icontract.require(lambda train_x, train_y: len(train_x) <= len(train_y))
@icontract.require(
    lambda selected_labels, train_x: (0 < len(selected_labels))
    & (len(selected_labels) < len(train_x))
)
@icontract.require(
    lambda support_size, train_x: (0 < support_size) & (support_size < len(train_x))
)
@icontract.require(
    lambda support_size, selected_labels, train_x: support_size * len(selected_labels)
    <= len(train_x)
)
@icontract.ensure(
    lambda result, selected_labels, return_indexes: len(result[0].keys())
    == len(selected_labels)
    if (return_indexes is True)
    else len(result.keys()) == len(selected_labels)
)
@typechecked
def generate_support(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    support_size: int,
    selected_labels: List,
    return_indexes=False,
) -> Union[
    dict[Any, torch.Tensor], Tuple[dict[Any, torch.Tensor], dict[Any, torch.Tensor]]
]:
    """
    Randomly select `support_size` examples of `way` classes from the examples in
    `train_x` with corresponding labels in `train_y` and return them as a dictionary.

    Parameters
    ----------
    train_x : torch.Tensor
        Input training data.
    train_y : torch.Tensor
        Corresponding classification labels.
    support_size : int
        Number of support examples for each class.
    selected_labels : List
        Selected class labels to generate examples from.
    return_indexes : bool, optional
        If True, also return the indices of the support examples.

    Returns
    -------
    Union[Dict[Any, torch.Tensor], Tuple[Dict[Any, torch.Tensor], Dict[Any, torch.Tensor]]]
        Ordered dictionary of class labels with corresponding support examples.
    """
    labels, counts = torch.unique(train_y, return_counts=True)
    for label, count in list(zip(labels, counts)):
        if (label in selected_labels) and (count < support_size):
            raise ValueError(f"Not enough support examples in class {label}")

    shuffled_idxs = _get_shuffle_idxs_by_class(train_y, selected_labels)

    support = OrderedDict()
    for label in selected_labels:
        shuffled_x = train_x[shuffled_idxs[label]]

        assert torch.unique(train_y[shuffled_idxs[label]]).tolist() == [
            label
        ], "Not enough support for label " + str(label)
        selected_support = shuffled_x[:support_size]
        support[label] = selected_support

    if return_indexes:
        return support, shuffled_idxs
    else:
        return support


@typechecked
def generate_episode(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    support_size: int,
    way: int,
    episode_size: int,
) -> Tuple[dict[Any, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Generate a single episode of data for a few-shot learning task.

    Parameters
    ----------
    train_x : torch.Tensor
        Input training data.
    train_y : torch.Tensor
        Corresponding classification labels.
    support_size : int
        Number of support examples for each class.
    way : int
        Number of classes in the episode.
    episode_size : int
        Total number of examples in the episode.

    Returns
    -------
    Tuple[Dict[Any, torch.Tensor], torch.Tensor, torch.Tensor]
        Tuple of support examples, query examples, and query labels.
    """
    labels = torch.unique(train_y)
    selected_labels = sorted(
        labels[torch.randperm(labels.shape[0])][:way].tolist()
    )  # need to be in same order every time

    support, shuffled_idxs = generate_support(
        train_x, train_y, support_size, selected_labels, return_indexes=True
    )

    examples_per_task = int(episode_size / way)

    episode_data_list = []
    episode_label_list = []
    episode_support = OrderedDict()
    for episode_label, label in enumerate(selected_labels):
        shuffled_x = train_x[shuffled_idxs[label]]
        shuffled_y = torch.Tensor(
            [episode_label] * len(shuffled_idxs[label])
        )  # need sequential labels for episode

        num_remaining_examples = shuffled_x.shape[0] - support_size
        assert num_remaining_examples > 0, (
            "Cannot have "
            + str(num_remaining_examples)
            + " left with support_size "
            + str(support_size)
            + " and shape "
            + str(shuffled_x.shape)
            + " from train_x shaped "
            + str(train_x.shape)
        )
        episode_end_idx = support_size + min(num_remaining_examples, examples_per_task)

        episode_data_list.append(shuffled_x[support_size:episode_end_idx])
        episode_label_list.append(shuffled_y[support_size:episode_end_idx])
        episode_support[episode_label] = support[label]

    episode_x = torch.concat(episode_data_list)
    episode_y = torch.concat(episode_label_list)

    return episode_support, episode_x, episode_y.squeeze().to(torch.long)


@icontract.require(
    lambda eq_preds, true_y: eq_preds.classes.size(dim=0) == true_y.size(dim=0)
)
@typechecked
def generate_model_metrics(
    eq_preds: EquineOutput, true_y: torch.Tensor
) -> dict[str, Any]:
    """
    Generate various metrics for evaluating a model's performance.

    Parameters
    ----------
    eq_preds : EquineOutput
        Model predictions.
    true_y : torch.Tensor
        True class labels.

    Returns
    -------
    Dict[str, Any]
        Dictionary of model metrics.
    """
    pred_y = torch.argmax(eq_preds.classes, dim=1)
    metrics = {
        "accuracy": accuracy_score(true_y, pred_y),
        "microF1Score": f1_score(true_y, pred_y, average="micro"),
        "confusionMatrix": confusion_matrix(true_y, pred_y).tolist(),
        "brierScore": brier_score(eq_preds.classes, true_y),
        "brierSkillScore": brier_skill_score(eq_preds.classes, true_y),
        "expectedCalibrationError": expected_calibration_error(
            eq_preds.classes, true_y
        ),
    }
    return metrics


@typechecked
def get_num_examples_per_label(Y: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Get the number of examples per label in the given tensor.

    Parameters
    ----------
    Y : torch.Tensor
        Tensor of class labels.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing label and number of examples.
    """
    tensor_labels, tensor_counts = Y.unique(return_counts=True)

    examples_per_label = []
    for i, label in enumerate(tensor_labels):
        examples_per_label.append(
            {"label": label.item(), "numExamples": tensor_counts[i].item()}
        )

    return examples_per_label


@icontract.require(lambda train_y: train_y.shape[0] > 0)
@typechecked
def generate_train_summary(
    model: Equine, train_y: torch.Tensor, date_trained: str
) -> dict[str, Any]:
    """
    Generate a summary of the training data.

    Parameters
    ----------
    model : Equine
        Model object.
    train_y : torch.Tensor
        Training labels.
    date_trained : str
        Date of training.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing training summary.
    """
    train_summary = {
        "numTrainExamples": get_num_examples_per_label(train_y),
        "dateTrained": date_trained,
        "modelType": model.__class__.__name__,
    }
    return train_summary


@icontract.require(
    lambda eq_preds, test_y: test_y.shape[0] == eq_preds.classes.shape[0]
)
@typechecked
def generate_model_summary(
    model: Equine,
    eq_preds: EquineOutput,
    test_y: torch.Tensor,
):
    """
    Generate a summary of the model's performance.

    Parameters
    ----------
    model : Equine
        Model object.
    eq_preds : EquineOutput
        Model predictions.
    test_y : torch.Tensor
        True class labels.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing model summary.
    """
    summary = generate_model_metrics(eq_preds, test_y)
    summary["numTestExamples"] = get_num_examples_per_label(test_y)
    summary.update(model.train_summary)  # union of train_summary and generated metrics

    return summary
