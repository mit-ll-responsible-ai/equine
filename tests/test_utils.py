# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import hypothesis.extra.numpy as hnp
import icontract
import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

import equine as eq


class _NoAnnotationModel(torch.nn.Module):
    """An nn.Module that declares no class-level annotations.

    On Python 3.14 such a module cannot be passed to ``torch.jit.script``
    without ``prepare_jit_module`` first materializing its instance
    ``__annotations__`` (see PEP 649/749).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@st.composite
def support_dataset(draw):
    support_sz = draw(st.integers(min_value=2, max_value=25))
    task_count = draw(st.integers(min_value=2, max_value=10))
    tasks = [x for x in range(task_count)]
    dataset_row_count = draw(
        st.integers(min_value=(support_sz + 1) * task_count, max_value=1000)
    )
    dataset_col_count = draw(st.integers(min_value=1, max_value=1000))
    shape = (dataset_row_count, dataset_col_count)
    dataset_x = torch.rand(shape)
    dataset_y = []
    for task in tasks:
        dataset_y += [task] * (support_sz + 1)

    if len(dataset_x) > len(dataset_y):
        pad = (
            torch.randint(0, task_count, (len(dataset_x) - len(dataset_y), 1))
            .flatten()
            .tolist()
        )
        dataset_y += pad

    dataset_y = torch.Tensor(dataset_y)
    way = draw(st.integers(min_value=2, max_value=task_count))

    return dataset_x, dataset_y, support_sz, tasks, way


@given(dataset=support_dataset())
@settings(deadline=None)
def test_generate_support(dataset):
    train_x, train_y, support_sz, tasks, _ = dataset
    eq.utils.generate_support(train_x, train_y, support_sz, tasks)


@given(dataset=support_dataset())
@settings(deadline=None)
def test_generate_episode(dataset) -> None:
    train_x, train_y, support_sz, tasks, way = dataset
    episode_size = max(len(tasks), train_x.shape[0] // 4)
    eq.utils.generate_episode(train_x, train_y, support_sz, way, episode_size)
    with pytest.raises(ValueError):
        eq.utils.generate_episode(
            train_x, train_y, support_sz, len(tasks) + 1, episode_size
        )


def test_generate_support_refuses_support_size_larger_than_a_class() -> None:
    # Class 0 has three examples; a support of four cannot be drawn from it.
    # Labels are a float tensor, so the message names the class as 0.0.
    x = torch.rand(8, 2)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]).float()
    with pytest.raises(ValueError, match="Not enough support examples in class 0.0"):
        eq.generate_support(x, y, support_size=4, selected_labels=[0, 1])


def test_generate_episode_refuses_support_size_larger_than_a_class() -> None:
    x = torch.rand(8, 2)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]).float()
    with pytest.raises(ValueError, match="Not enough support examples in class 0.0"):
        eq.generate_episode(x, y, support_size=4, way=2, episode_size=2)


@st.composite
def draw_two_tensors(draw: st.DrawFn) -> tuple[torch.Tensor, torch.Tensor]:
    num_classes = draw(st.integers(min_value=2, max_value=128))
    num_examples = draw(st.integers(min_value=2, max_value=100))
    yh = draw(
        hnp.arrays(
            shape=st.just((num_examples, num_classes)),
            dtype=np.float32,
            elements=st.floats(
                allow_nan=False,
                allow_infinity=False,
                width=32,
                min_value=9.9999998245167e-15,
                max_value=1.0,
            ),
        ).map(lambda arr: torch.tensor(arr, dtype=torch.float32))
    )
    yh = yh / yh.sum(dim=1).unsqueeze(-1)
    true_class = draw(
        st.lists(
            st.integers(min_value=0, max_value=num_classes - 1),
            min_size=num_examples,
            max_size=num_examples,
        ).map(lambda lst: torch.as_tensor(lst, dtype=torch.int64))
    )
    return yh, true_class


@given(draw_two_tensors())
def test_brier_score(two_tensors) -> None:
    yh, yt = two_tensors
    assert eq.utils.brier_score(yh, yt) >= 0.0
    yt = yt.int()  # Regression test to make sure one-hot encoding function doesn't need a LongTensor
    assert eq.utils.brier_score(yh, yt) >= 0.0


@given(draw_two_tensors())
def test_brier_skill_score(two_tensors) -> None:
    yh, yt = two_tensors
    assert eq.utils.brier_skill_score(yh, yt) <= 1.0
    yt = yt.int()  # Regression test to make sure one-hot encoding function doesn't need a LongTensor
    assert eq.utils.brier_skill_score(yh, yt) <= 1.0


@given(draw_two_tensors())
def test_ece(two_tensors) -> None:
    yh, yt = two_tensors
    assert eq.utils.expected_calibration_error(yh, yt) <= 1.0


@given(draw_two_tensors())
def test_metric_summary(two_tensors) -> None:
    yh, yt = two_tensors
    eq_out = eq.EquineOutput(yh, torch.ones(yt.size()), torch.ones(yt.size()))
    metrics = eq.utils.generate_model_metrics(eq_out, yt)
    assert "accuracy" in metrics
    assert "microF1Score" in metrics
    assert "brierScore" in metrics
    assert "brierSkillScore" in metrics
    assert "expectedCalibrationError" in metrics


def test_mahalanobis():
    eps = 10 ** (-4)

    cov = torch.eye(10) * (1 + eps)
    cov = torch.unsqueeze(cov, 0)
    diff = torch.ones((10, 1))

    dist = eq.mahalanobis_distance_nosq(diff, cov)
    target_val = torch.tensor(10 * (1 / (1 + eps)), dtype=dist.dtype)
    assert torch.isclose(dist[0, 0], target_val)

    cov = torch.eye(10) * eps
    cov[0, 0] = cov[0, 0] + 1
    cov = torch.unsqueeze(cov, 0)
    diff = torch.ones((10, 1))

    dist = eq.mahalanobis_distance_nosq(diff, cov)
    target_val = torch.tensor(9 / eps + (1 / (1 + eps)), dtype=dist.dtype)
    assert torch.isclose(dist[0, 0], target_val)

    cov = torch.eye(10) * eps + torch.ones((10, 10))
    cov = torch.unsqueeze(cov, 0)
    diff = torch.ones((10, 1))

    dist = eq.mahalanobis_distance_nosq(diff, cov)
    target_val = torch.tensor(
        (1 / eps) * 10.0 - (100.0) / (eps**2 + eps * 10), dtype=dist.dtype
    )
    assert torch.isclose(dist[0, 0], target_val)


def test_stratified_split_invalid_test_size():
    X = torch.arange(10).float().unsqueeze(1)
    Y = torch.randint(0, 2, (10,))

    with pytest.raises(
        icontract.ViolationError, match="test_size must be between 0 and 1."
    ):
        eq.utils.stratified_train_test_split(X, Y, test_size=1.5)

    with pytest.raises(
        icontract.ViolationError, match="test_size must be between 0 and 1."
    ):
        eq.utils.stratified_train_test_split(X, Y, test_size=0)


def test_stratified_split_mismatched_lengths():
    X = torch.arange(10).float().unsqueeze(1)
    Y = torch.randint(0, 2, (8,))  # Mismatched length

    with pytest.raises(
        icontract.ViolationError, match="X and Y must have the same number of samples."
    ):
        eq.utils.stratified_train_test_split(X, Y, test_size=0.3)


def test_stratified_split_imbalanced_classes():
    X = torch.arange(50).float().unsqueeze(1)
    Y = torch.cat(
        [torch.zeros(45, dtype=torch.long), torch.ones(5, dtype=torch.long)]
    )  # Imbalanced

    _, _, train_y, test_y = eq.utils.stratified_train_test_split(X, Y, test_size=0.2)

    # Check that minority class is present in both splits
    assert (train_y == 1).sum().item() > 0
    assert (test_y == 1).sum().item() > 0

    # Verify counts
    total_counts = torch.tensor([(Y == cls).sum().item() for cls in torch.unique(Y)])
    expected_test_counts = (total_counts.float() * 0.2).round().long()
    expected_train_counts = total_counts - expected_test_counts

    actual_train_counts = torch.tensor(
        [(train_y == cls).sum().item() for cls in torch.unique(Y)]
    )
    actual_test_counts = torch.tensor(
        [(test_y == cls).sum().item() for cls in torch.unique(Y)]
    )

    assert torch.all(actual_train_counts == expected_train_counts)
    assert torch.all(actual_test_counts == expected_test_counts)


@given(
    in_dim=st.integers(min_value=1, max_value=16),
    out_dim=st.integers(min_value=1, max_value=16),
    batch=st.integers(min_value=1, max_value=8),
)
@settings(deadline=None, max_examples=5)
def test_prepare_jit_module_makes_module_scriptable(in_dim, out_dim, batch):
    # Regression test for Python 3.14 (PEP 649/749): a module with no
    # class-level annotations must remain compatible with torch.jit.script
    # after prepare_jit_module, which is what EquineGP/EquineProtonet.save use.
    model = _NoAnnotationModel(in_dim, out_dim)
    returned = eq.utils.prepare_jit_module(model)
    assert returned is model  # returned for convenient inline use

    scripted = torch.jit.script(model)  # must not raise on Python 3.14
    x = torch.rand(batch, in_dim)
    assert torch.allclose(scripted(x), model(x))


def _probs(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_brier_score_known_values() -> None:
    # brier_score = mean over rows of sum_j (f_ij - o_ij)^2, o = one-hot(y)
    y = torch.tensor([0, 1, 2])
    # perfect one-hot predictions -> every term is 0
    assert eq.utils.brier_score(
        _probs([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), y
    ) == pytest.approx(0.0)
    # uniform over K classes: (1 - 1/K)^2 + (K-1)(1/K)^2 = (K-1)/K per row
    k = 3
    assert eq.utils.brier_score(_probs([[1 / k] * k] * 3), y) == pytest.approx(
        (k - 1) / k, abs=1e-6
    )
    # confidently wrong: (0-1)^2 + (1-0)^2 = 2 per row
    assert eq.utils.brier_score(
        _probs([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), y
    ) == pytest.approx(2.0)


def test_brier_skill_score_known_values() -> None:
    # brier_skill_score = 1 - BS / BS_ref, where the reference forecast is the
    # uniform guess 1/K for every class, so BS_ref = (K-1)/K = 2/3 for K=3.
    y = torch.tensor([0, 1, 2])
    # perfect: BS=0 -> 1 - 0 = 1
    assert eq.utils.brier_skill_score(
        _probs([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), y
    ) == pytest.approx(1.0, abs=1e-6)
    # uniform: BS=BS_ref -> 1 - 1 = 0 (no skill over the reference)
    assert eq.utils.brier_skill_score(_probs([[1 / 3] * 3] * 3), y) == pytest.approx(
        0.0, abs=1e-6
    )
    # confidently wrong: BS=2 -> 1 - 2 / (2/3) = 1 - 3 = -2
    assert eq.utils.brier_skill_score(
        _probs([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), y
    ) == pytest.approx(-2.0, abs=1e-6)


def test_expected_calibration_error_known_values() -> None:
    # ECE (torchmetrics MulticlassCalibrationError, n_bins=25, norm="l1") bins
    # rows by max probability and sums |accuracy_bin - confidence_bin| weighted
    # by the fraction of rows in the bin. Every case below puts all rows in a
    # single bin, so ECE = |accuracy - confidence| of that bin.
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    # confidence 1, accuracy 1 -> 0
    right = torch.nn.functional.one_hot(y, 2).float()
    assert eq.utils.expected_calibration_error(right, y) == pytest.approx(0.0, abs=1e-6)
    # confidence 1, accuracy 0 -> 1
    wrong = torch.nn.functional.one_hot(1 - y, 2).float()
    assert eq.utils.expected_calibration_error(wrong, y) == pytest.approx(1.0, abs=1e-6)
    # confidence 0.6 on every row, 6 of 10 right -> |0.6 - 0.6| = 0
    calibrated = _probs([[0.6, 0.4]] * 10)
    y_cal = torch.tensor([0] * 6 + [1] * 4)
    assert eq.utils.expected_calibration_error(calibrated, y_cal) == pytest.approx(
        0.0, abs=1e-6
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#174: generate_model_metrics passes (target, preds) to torchmetrics, "
    "which expects (preds, target)",
)
def test_generate_model_metrics_known_confusion_matrix() -> None:
    y_true = torch.tensor([0, 0, 0, 0, 1, 2])
    probs = torch.nn.functional.one_hot(torch.tensor([0, 0, 0, 1, 1, 1]), 3).float()
    out = eq.EquineOutput(classes=probs, ood_scores=torch.zeros(6), embeddings=probs)
    metrics = eq.utils.generate_model_metrics(out, y_true)
    # MulticlassAccuracy(num_classes=3) defaults to average="macro": the mean of
    # the per-class recalls. class 0 -> 3/4, class 1 -> 1/1, class 2 -> 0/1,
    # so (0.75 + 1 + 0) / 3 = 7/12. With the arguments swapped the "classes"
    # are the predicted labels and the value comes out as 4/9.
    assert metrics["accuracy"] == pytest.approx(7 / 12)
    # rows = true class, columns = predicted class. The swapped call returns the
    # transpose, [[3, 0, 0], [1, 1, 1], [0, 0, 0]].
    assert metrics["confusionMatrix"] == [[3, 1, 0], [0, 1, 0], [0, 1, 0]]
