# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import math
import sys
import warnings

import hypothesis.extra.numpy as hnp
import icontract
import numpy as np
import pytest
import torch
from conftest import script_module
from hypothesis import given, settings, strategies as st

import equine as eq
from equine.utils import _cholesky_with_jitter


class _NoAnnotationModel(torch.nn.Module):
    """An nn.Module whose class declares no annotations."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@st.composite
def support_dataset(draw):
    torch.manual_seed(draw(st.integers(min_value=0, max_value=2**31 - 1)))
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
    support = eq.utils.generate_support(train_x, train_y, support_sz, tasks)
    assert set(support.keys()) == set(tasks)
    for label, examples in support.items():
        assert examples.shape == (support_sz, train_x.shape[1])
        row_idxs = (train_y == label).nonzero().flatten()
        assert all(
            (train_x[row_idxs] == example).all(dim=1).any() for example in examples
        )


@given(dataset=support_dataset())
@settings(deadline=None)
def test_generate_episode(dataset) -> None:
    train_x, train_y, support_sz, tasks, way = dataset
    episode_size = max(len(tasks), train_x.shape[0] // 4)
    support, episode_x, episode_y = eq.utils.generate_episode(
        train_x, train_y, support_sz, way, episode_size
    )
    assert list(support.keys()) == list(range(way))
    assert all(examples.shape[0] == support_sz for examples in support.values())
    assert episode_x.shape[0] == episode_y.shape[0]
    assert episode_x.shape[0] <= episode_size
    assert episode_y.dtype == torch.long
    assert set(episode_y.tolist()) == set(range(way))
    with pytest.raises(ValueError):
        eq.utils.generate_episode(
            train_x, train_y, support_sz, len(tasks) + 1, episode_size
        )


def test_generate_support_refuses_support_size_larger_than_a_class() -> None:
    # Class 0 has three examples; a support of four cannot be drawn from it.
    # Labels are a float tensor; the message names the class as 0.0.
    x = torch.rand(8, 2)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]).float()
    with pytest.raises(ValueError, match="Not enough support examples in class 0.0"):
        eq.generate_support(x, y, support_size=4, selected_labels=[0, 1])


def test_generate_episode_refuses_support_size_larger_than_a_class() -> None:
    x = torch.rand(8, 2)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]).float()
    with pytest.raises(ValueError, match="Class 0 has 3 examples"):
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
@settings(deadline=None)
def test_brier_score(two_tensors) -> None:
    yh, yt = two_tensors
    assert eq.utils.brier_score(yh, yt) >= 0.0
    yt = yt.int()  # Regression test to make sure one-hot encoding function doesn't need a LongTensor
    assert eq.utils.brier_score(yh, yt) >= 0.0


@given(draw_two_tensors())
@settings(deadline=None)
def test_brier_skill_score(two_tensors) -> None:
    yh, yt = two_tensors
    assert eq.utils.brier_skill_score(yh, yt) <= 1.0
    yt = yt.int()  # Regression test to make sure one-hot encoding function doesn't need a LongTensor
    assert eq.utils.brier_skill_score(yh, yt) <= 1.0


@given(draw_two_tensors())
@settings(deadline=None)
def test_ece(two_tensors) -> None:
    yh, yt = two_tensors
    assert eq.utils.expected_calibration_error(yh, yt) <= 1.0


@given(draw_two_tensors())
@settings(deadline=None)
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
def test_script_module_scripts_a_module_without_class_annotations(
    in_dim, out_dim, batch
):
    model = _NoAnnotationModel(in_dim, out_dim)
    scripted = script_module(model)
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
    # imbalanced labels: rows with y=0 score 0.3^2 + 0.2^2 + 0.1^2 = 0.14 and the
    # y=1 row 0.7^2 + 0.8^2 + 0.1^2 = 1.14, so BS = 0.39 -> 1 - 0.39 / (2/3) = 0.415
    # (a class-frequency reference, BS_ref = 0.375, would give -0.04)
    assert eq.utils.brier_skill_score(
        _probs([[0.7, 0.2, 0.1]] * 4), torch.tensor([0, 0, 0, 1])
    ) == pytest.approx(0.415, abs=1e-6)


def test_expected_calibration_error_known_values() -> None:
    # ECE (torchmetrics MulticlassCalibrationError, n_bins=25, norm="l1") bins
    # rows by max probability and sums |accuracy_bin - confidence_bin| weighted
    # by the fraction of rows in the bin. The first three cases put all rows in
    # a single bin, so ECE = |accuracy - confidence| of that bin.
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
    # two bins of 5 rows: confidence 0.9 with accuracy 1, and confidence 0.6
    # with accuracy 1/5 -> 0.5 * |1 - 0.9| + 0.5 * |0.2 - 0.6| = 0.25
    # (one pooled bin would give |0.6 - 0.75| = 0.15)
    two_bins = _probs([[0.9, 0.1]] * 5 + [[0.6, 0.4]] * 5)
    y_two = torch.tensor([0] * 5 + [0, 1, 1, 1, 1])
    assert eq.utils.expected_calibration_error(two_bins, y_two) == pytest.approx(
        0.25, abs=1e-6
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


@pytest.mark.parametrize(
    "platform, version, torch_version, warns",
    [
        ("darwin", (3, 15, 0), "2.14.1", True),
        ("darwin", (3, 16, 0), "2.14.1+cpu", True),
        ("darwin", (3, 15, 0), "2.15.0", False),
        ("darwin", (3, 14, 3), "2.14.1", False),
        ("linux", (3, 15, 0), "2.14.1", False),
    ],
)
def test_warns_where_torch_crashes_with_threads(
    monkeypatch, platform, version, torch_version, warns
):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "version_info", version)
    monkeypatch.setattr(torch, "__version__", torch_version)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        eq.utils._warn_if_torch_crashes_with_threads()
    assert any("several threads" in str(w.message) for w in record) == warns


def _spd(classes: int, dim: int) -> torch.Tensor:
    m = torch.randn(classes, dim, dim, dtype=torch.float64)
    return m @ m.transpose(1, 2) + torch.eye(dim, dtype=torch.float64)


@pytest.mark.parametrize(
    "x_shape, classes, dim",
    [((3, 4, 5), 3, 4), ((4, 5), 3, 4), ((3, 3), 3, 3), ((4,), 3, 4)],
    ids=["classes-dim-examples", "dim-examples", "dim-examples-square", "dim"],
)
def test_mahalanobis_is_x_cinv_x_for_every_input_shape(x_shape, classes, dim):
    """A 2-D ``x`` holds examples as columns and is shared by every class, even when its
    shape equals (classes, dim)."""
    torch.manual_seed(0)
    cov = _spd(classes, dim)
    x = torch.randn(*x_shape, dtype=torch.float64)
    inv = torch.linalg.inv(cov)
    if x.dim() == 1:
        expected = torch.einsum("d,kde,e->k", x, inv, x)
    elif x.dim() == 2:
        expected = torch.einsum("dn,kde,en->kn", x, inv, x)
    else:
        expected = torch.einsum("kdn,kde,ken->kn", x, inv, x)
    dist = eq.mahalanobis_distance_nosq(x, cov)
    assert dist.shape == expected.shape
    torch.testing.assert_close(dist, expected)


@pytest.mark.parametrize(
    "cov",
    [
        torch.eye(4) * 1e-5,
        torch.ones(4, 4) + 1e-5 * torch.eye(4),
    ],
    ids=["all-eigenvalues-equal", "rank-one-plus-epsilon"],
)
def test_mahalanobis_gradient_is_finite_with_repeated_eigenvalues(cov):
    """The distance has finite gradients when the covariance has repeated
    eigenvalues."""
    cov = cov.expand(2, 4, 4).clone().requires_grad_()
    x = torch.rand(2, 4, 3, requires_grad=True)
    eq.mahalanobis_distance_nosq(x, cov).sum().backward()
    assert torch.isfinite(cov.grad).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "cov",
    [torch.ones(1, 3, 3), torch.diag(torch.tensor([1.0, -1.0, 1.0])).unsqueeze(0)],
    ids=["singular", "indefinite"],
)
def test_mahalanobis_raises_on_a_covariance_that_is_not_positive_definite(cov):
    """A covariance whose Cholesky factorization fails raises."""
    with pytest.raises(torch.linalg.LinAlgError, match="positive-definite"):
        eq.mahalanobis_distance_nosq(torch.ones(1, 3, 2), cov)


def test_mahalanobis_rejects_an_unbatched_covariance():
    """A single (dim, dim) covariance is refused."""
    m = torch.randn(4, 4)
    with pytest.raises(icontract.ViolationError, match="classes, dim, dim"):
        eq.mahalanobis_distance_nosq(torch.randn(4, 5), m @ m.T + torch.eye(4))


def test_cholesky_with_jitter_factors_only_the_matrices_that_need_it():
    """A rank-deficient float32 covariance at a large scale is indefinite by
    rounding and gets a small jitter; a well-conditioned one is left unchanged."""
    torch.manual_seed(0)
    emb = (torch.randn(50, 2) @ torch.randn(2, 8)) * 1000
    rounded = torch.cov(emb.T) + 1e-5 * torch.eye(8)
    m = torch.randn(8, 8)
    good = m @ m.T + torch.eye(8)
    cov = torch.stack([good, rounded])
    assert torch.linalg.cholesky_ex(rounded)[1].item() != 0
    factor = _cholesky_with_jitter(cov)
    assert torch.equal(factor[0], torch.linalg.cholesky(good))
    assert torch.isfinite(factor[1]).all()
    largest = rounded.diagonal().max().item()
    torch.testing.assert_close(
        factor[1] @ factor[1].T, rounded, rtol=0, atol=1e-3 * largest
    )


def test_cholesky_with_jitter_raises_when_no_jitter_helps():
    """A NaN covariance does not factor with any jitter."""
    with pytest.raises(torch.linalg.LinAlgError, match="not positive definite"):
        _cholesky_with_jitter(torch.full((1, 3, 3), float("nan")))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("power", [0, 1, 2, 3, "top"])
def test_cholesky_with_jitter_uses_the_smallest_jitter_that_factors(dtype, power):
    """A pivot of -0.5 * eps * 10**power factors with exactly the jitter eps * 10**power."""
    eps = torch.finfo(dtype).eps
    if power == "top":
        power = int(math.log10(1e-3 / eps))
    pivots = torch.tensor([1.0, 1.0, -0.5 * eps * 10**power], dtype=dtype)
    cov = torch.diag(pivots).unsqueeze(0)
    eye = torch.eye(3, dtype=dtype)
    expected = torch.linalg.cholesky(cov + eps * 10**power * eye)
    assert torch.equal(_cholesky_with_jitter(cov), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cholesky_with_jitter_stops_at_its_cap(dtype):
    """A pivot of -5e-4 needs more jitter than the largest rung, which stays at or
    below 1e-3 of the largest variance; the error names that largest rung."""
    eps = torch.finfo(dtype).eps
    top = eps * 10 ** int(math.log10(1e-3 / eps))
    cov = torch.diag(torch.tensor([1.0, 1.0, -5e-4], dtype=dtype)).unsqueeze(0)
    with pytest.raises(torch.linalg.LinAlgError, match=f"jitter of {top:.1e} times"):
        _cholesky_with_jitter(cov)


def test_cholesky_with_jitter_treats_a_non_finite_factor_as_a_failure(monkeypatch):
    """A NaN factor from a factorization that reports success raises."""

    real = torch.linalg.cholesky

    def nan_factor(cov):
        factor = real(cov).clone()
        factor[..., -1, -1] = float("nan")
        return factor, torch.zeros(cov.shape[:-2], dtype=torch.int32)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", nan_factor)
    with pytest.raises(torch.linalg.LinAlgError, match="not positive definite"):
        _cholesky_with_jitter(torch.eye(3).unsqueeze(0))


def test_mahalanobis_works_under_vmap():
    """The distance can be batched with torch.func.vmap."""
    torch.manual_seed(0)
    m = torch.randn(4, 2, 3, 3)
    cov = m @ m.transpose(-1, -2) + torch.eye(3)
    x = torch.randn(4, 2, 3, 5)
    batched = torch.func.vmap(eq.mahalanobis_distance_nosq)(x, cov)
    looped = torch.stack([eq.mahalanobis_distance_nosq(x[i], cov[i]) for i in range(4)])
    torch.testing.assert_close(batched, looped)


def test_require_device_accepts_settings_without_a_device() -> None:
    assert eq.utils._require_device({}) is None


def test_require_device_refuses_a_device_type_this_build_cannot_query() -> None:
    with pytest.raises(ValueError, match="'xla'.* is not available on this machine"):
        eq.utils._require_device({"device": "xla"})


def test_shuffle_idxs_keep_a_class_with_one_example() -> None:
    train_y = torch.tensor([0, 0, 0, 1, 1, 1, 2])
    idxs = eq.utils._get_shuffle_idxs_by_class(train_y, [0, 1, 2])
    assert idxs[2].tolist() == [6]
    assert sorted(idxs[0].tolist()) == [0, 1, 2]


def test_shuffle_idxs_refuse_a_label_without_examples() -> None:
    train_y = torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match="Class 2 has no examples"):
        eq.utils._get_shuffle_idxs_by_class(train_y, [0, 1, 2])


def test_generate_support_refuses_indexes_of_another_label() -> None:
    train_x = torch.rand(6, 2)
    train_y = torch.tensor([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="shuffled indexes for label 0"):
        eq.utils.generate_support(
            train_x, train_y, 2, [0], shuffled_indexes={0: torch.tensor([3, 4])}
        )


def test_generate_episode_names_a_large_integer_label_exactly() -> None:
    train_x = torch.rand(9, 2)
    train_y = torch.tensor([1234567] * 4 + [7] * 5)
    with pytest.raises(ValueError, match="Class 1234567 has 4 examples"):
        eq.utils.generate_episode(
            train_x, train_y, support_size=4, way=2, episode_size=4
        )


def test_generate_episode_needs_a_query_example_per_class() -> None:
    train_x = torch.rand(8, 2)
    train_y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="Class 0 has 4 examples"):
        eq.utils.generate_episode(
            train_x, train_y, support_size=4, way=2, episode_size=4
        )


def _one_hot_output(columns):
    probs = torch.nn.functional.one_hot(torch.tensor(columns), 3).float()
    return eq.EquineOutput(
        classes=probs, ood_scores=torch.zeros(len(columns)), embeddings=probs
    )


@pytest.mark.parametrize("label_order", [[0, 2, 5], [5, 0, 2]])
@pytest.mark.parametrize("true_y", [[0, 0, 2, 2, 5, 5], [0, 0, 2, 2]])
def test_generate_model_metrics_maps_true_labels_through_label_order(
    true_y, label_order
) -> None:
    """Perfect predictions: label ``label_order[j]`` in column ``j``. The second
    case of ``true_y`` leaves out label 5."""
    columns = [label_order.index(label) for label in true_y]
    metrics = eq.generate_model_metrics(
        _one_hot_output(columns), torch.tensor(true_y), label_order=label_order
    )
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["microF1Score"] == pytest.approx(1.0)
    assert metrics["brierScore"] == pytest.approx(0.0)
    matrix = metrics["confusionMatrix"]
    assert sum(matrix[i][j] for i in range(3) for j in range(3) if i != j) == 0


def test_generate_model_metrics_rejects_labels_outside_label_order() -> None:
    out = _one_hot_output([0, 1, 2, 2])
    with pytest.raises(ValueError, match=r"Labels \[3\] are not among"):
        eq.generate_model_metrics(
            out, torch.tensor([0, 2, 3, 5]), label_order=[0, 2, 5]
        )
    for label_order in ([0, 2], [0, 2, 2], [0, 1, 1, 2]):
        with pytest.raises(ValueError, match="must hold 3 distinct labels"):
            eq.generate_model_metrics(
                out, torch.tensor([0, 2, 2, 2]), label_order=label_order
            )
