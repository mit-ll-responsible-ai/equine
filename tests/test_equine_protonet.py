# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import copy
import math
from collections import OrderedDict

import numpy as np
import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation
from conftest import (
    BasicEmbeddingModel,
    assert_valid_prediction,
    generate_random_string_list,
    random_dataset,
    stored_tensors,
    use_basic_embedding_model,
    use_save_load_model_tests,
)
from golden_data import CLASSES, FEATURES, golden_dtype, separable_dataset
from hypothesis import given, settings, strategies as st
from scipy.stats import gaussian_kde

import equine as eq
from equine.equine_protonet import DEFAULT_EPSILON, Protonet


@given(
    data_shape=st.tuples(
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
    ),
    num_classes=st.integers(min_value=2, max_value=256),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(deadline=None)
def test_compute_embeddings(data_shape, num_classes, seed):
    torch.manual_seed(seed)
    queries = torch.rand(data_shape)
    embed_model = BasicEmbeddingModel(data_shape[1], num_classes)
    model = eq.EquineProtonet(embed_model, num_classes)
    embeddings = model.model.compute_embeddings(queries)
    assert embeddings.shape == (data_shape[0], num_classes)


def test_compute_shared_covariance_refuses_unit_covariance() -> None:
    model = eq.EquineProtonet(BasicEmbeddingModel(2, 3), 3)
    class_cov_dict = OrderedDict({0: torch.ones(3), 1: torch.ones(3)})
    with pytest.raises(ValueError, match="not UNIT"):
        model.model.compute_shared_covariance(class_cov_dict, eq.CovType.UNIT)


def _briefly_trained_protonet_with_data():
    """A small float32 model trained just enough for update_support to run
    (train_model sets the statistics update_support relies on)."""
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    return model, x, y.float()


def test_update_support_sets_label_names() -> None:
    model, x, y = _briefly_trained_protonet_with_data()
    assert model.get_label_names() is None
    model.update_support(x, y, calib_frac=0.5, label_names=["a", "b", "c"])
    assert model.get_label_names() == ["a", "b", "c"]


def test_update_support_rejects_wrong_label_names_length() -> None:
    model, x, y = _briefly_trained_protonet_with_data()
    with pytest.raises(
        ValueError,
        match=r"The length of label_names \(2\) does not match the number of classes \(3\)",
    ):
        model.update_support(x, y, calib_frac=0.5, label_names=["only", "two"])


@given(random_dataset=random_dataset())
@settings(deadline=None)
def test_train_episodes(random_dataset):
    dataset, num_classes, train_kwargs = random_dataset
    num_episodes = 10

    X, Y = dataset.tensors
    num_deep_features = 32
    embed_model = BasicEmbeddingModel(X.shape[1], num_deep_features)
    model = eq.EquineProtonet(embed_model, num_deep_features)
    model.train_model(
        dataset,
        num_episodes=num_episodes,
        **train_kwargs,
    )

    assert model.model.training is False, "Model leaves training mode"
    assert len(model.model.support) == num_classes  # type: ignore
    # Test on multiple predictions
    eq_out = model.predict(X)
    assert_valid_prediction(eq_out, len(X), num_classes)
    # Test on single prediction
    pred_out = model(X[0])
    assert len(pred_out) == 1, "Single prediction works"
    eq_out = model.predict(X[0])
    assert_valid_prediction(eq_out, 1, num_classes)

    support = model.get_support()
    assert support is not None and len(support) == num_classes, (
        "Support set is correct size"
    )
    prototypes = model.get_prototypes()
    assert prototypes is not None and len(prototypes) == num_classes, (
        "Prototypes set is correct size"
    )
    assert (
        model.model.support is not None and len(model.model.support) == num_classes
    ), "Support set is correct size"


@given(random_dataset=random_dataset())
@settings(deadline=None)
def test_train_episodes_shared_reg(random_dataset):
    dataset, num_classes, train_kwargs = random_dataset
    num_episodes = 10

    X, Y = dataset.tensors
    num_deep_features = 32
    embed_model = BasicEmbeddingModel(X.shape[1], num_deep_features)
    model = eq.EquineProtonet(
        embed_model, num_deep_features, cov_type=eq.CovType.DIAGONAL
    )
    model.cov_reg_type = "shared"
    model.model.cov_reg_type = "shared"
    model.train_model(
        dataset,
        num_episodes=num_episodes,
        **train_kwargs,
    )

    assert model.model.training is False, "Model leaves training mode"
    assert len(model.model.support) == num_classes  # type: ignore
    # Test on multiple predictions
    eq_out = model.predict(X)
    assert_valid_prediction(eq_out, len(X), num_classes)
    # Test on single prediction
    pred_out = model(X[0])
    assert len(pred_out) == 1, "Single prediction works"
    eq_out = model.predict(X[0])
    assert_valid_prediction(eq_out, 1, num_classes)

    support = model.get_support()
    assert support is not None and len(support) == num_classes, (
        "Support set is correct size"
    )
    model.update_support(X, Y, calib_frac=0.5)
    assert (
        model.model.support is not None and len(model.model.support) == num_classes
    ), "Support set is correct size"


@given(random_dataset=random_dataset())
@settings(deadline=None)
def test_train_episodes_full_cov(random_dataset):
    dataset, num_classes, train_kwargs = random_dataset
    num_episodes = 5

    X, Y = dataset.tensors
    num_deep_features = 4
    embed_model = BasicEmbeddingModel(X.shape[1], num_deep_features)
    model = eq.EquineProtonet(embed_model, num_deep_features, cov_type=eq.CovType.FULL)
    model.cov_reg_type = "epsilon"
    model.model.cov_reg_type = "epsilon"
    model.train_model(
        dataset,
        num_episodes=num_episodes,
        **train_kwargs,
    )

    assert model.model.training is False, "Model leaves training mode"
    assert len(model.model.support) == num_classes  # type: ignore
    # Test on multiple predictions
    eq_out = model.predict(X)
    assert_valid_prediction(eq_out, len(X), num_classes)
    # Test on single prediction
    pred_out = model(X[0])
    assert len(pred_out) == 1, "Single prediction works"
    eq_out = model.predict(X[0])
    assert_valid_prediction(eq_out, 1, num_classes)

    assert (
        model.model.support is not None and len(model.model.support) == num_classes
    ), "Support set is correct size"
    model.update_support(X, Y, calib_frac=0.5)
    assert (
        model.model.support is not None and len(model.model.support) == num_classes
    ), "Support set is correct size"


@given(random_dataset=random_dataset())
@settings(deadline=None)
def test_train_episodes_with_temperature(random_dataset):
    dataset, num_classes, train_kwargs = random_dataset
    num_episodes = 10

    X, Y = dataset.tensors
    num_deep_features = 32
    embed_model = BasicEmbeddingModel(X.shape[1], num_deep_features)
    model = eq.EquineProtonet(embed_model, num_deep_features, use_temperature=True)
    before = model.temperature.item()
    assert before == 1.0, "init_temperature defaults to 1.0"
    train_dict = model.train_model(
        dataset,
        num_episodes=num_episodes,
        **train_kwargs,
    )

    assert "calib_x" in train_dict
    assert "calib_y" in train_dict
    after_train = model.temperature.item()
    assert after_train != before, "train_model(use_temperature=True) must calibrate"
    assert after_train > 0

    model.calibrate_temperature(train_dict["calib_x"], train_dict["calib_y"], 1, 0.01)
    after_calibration = model.temperature.item()
    assert after_calibration != after_train, "calibrate_temperature must move it"
    assert after_calibration > 0

    # Test on multiple predictions
    eq_out = model.predict(X)
    assert_valid_prediction(eq_out, len(X), num_classes)
    # Test on single prediction
    pred_out = model(X[0])
    assert len(pred_out) == 1, "Single prediction works"
    eq_out = model.predict(X[0])
    assert_valid_prediction(eq_out, 1, num_classes)


# The calibrated temperature of ``_float64_protonet_with_temperature(2)``,
# pinned on the CPU with torch 2.9.1.
_BASE_CALIBRATED_TEMPERATURE = 0.8800012564399848


def _float64_protonet_with_temperature(
    num_calibration_epochs: int,
) -> eq.EquineProtonet:
    """A ``.double()`` Protonet trained on a float64 dataset under the float32 default dtype."""
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        use_temperature=True,
        init_temperature=0.9,
    ).double()
    model.train_model(
        torch.utils.data.TensorDataset(x.double(), y.double()),
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
        num_calibration_epochs=num_calibration_epochs,
    )
    return model


def test_temperature_restarts_rounded_through_the_default_dtype() -> None:
    """``train_model`` restarts the temperature at ``init_temperature``.

    The value is rounded to the default dtype (float32 here) and then widened to the
    buffer's: a float64 model starts calibrating from float32(0.9), not from 0.9.
    """
    assert torch.get_default_dtype() == torch.float32
    start = _float64_protonet_with_temperature(num_calibration_epochs=0).temperature
    assert start.dtype == torch.float64
    assert start.item() == float(torch.tensor(0.9, dtype=torch.float32))

    calibrated = _float64_protonet_with_temperature(num_calibration_epochs=2)
    torch.testing.assert_close(
        calibrated.temperature,
        torch.tensor([_BASE_CALIBRATED_TEMPERATURE], dtype=torch.float64),
        atol=0,
        rtol=0,
    )


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_predict_fail_before_training(random_dataset):
    dataset, num_classes, X, embedding_model, _ = use_basic_embedding_model(
        random_dataset
    )

    model = eq.EquineProtonet(embedding_model, num_classes)
    with pytest.raises(ValueError):
        model(X)
    with pytest.raises(ValueError):
        model.predict(X)


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_equine_protonet_save_load(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, train_kwargs = use_basic_embedding_model(
        random_dataset
    )

    model = eq.EquineProtonet(embedding_model, num_classes, relative_mahal=False)
    model.train_model(dataset, num_episodes=2, **train_kwargs)

    use_save_load_model_tests(model, X, tmp_filename="protonet_save_load.eq")


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_equine_protonet_save_load_with_temperature(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, train_kwargs = use_basic_embedding_model(
        random_dataset
    )

    model = eq.EquineProtonet(embedding_model, num_classes, use_temperature=True)
    before = model.temperature.item()
    model.train_model(dataset, num_episodes=2, **train_kwargs)
    calibrated = model.temperature.item()
    assert calibrated != before, "train_model(use_temperature=True) must calibrate"
    assert calibrated > 0

    new_model = use_save_load_model_tests(
        model, X, tmp_filename="protonet_save_load_with_temperature.eq"
    )
    assert new_model.temperature.item() == pytest.approx(calibrated), (
        "temperature changed on reload"
    )


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_equine_protonet_save_load_with_feature_and_label_names(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, train_kwargs = use_basic_embedding_model(
        random_dataset
    )

    # without feature and label names
    model = eq.EquineProtonet(embedding_model, num_classes)
    model.train_model(dataset, num_episodes=2, **train_kwargs)

    new_model = use_save_load_model_tests(
        model, X, tmp_filename="protonet_save_load_no_feature_and_label_names.eq"
    )

    assert new_model.get_feature_names() is None, "feature_names changed on reload"
    assert new_model.get_label_names() is None, "label_names changed on reload"

    # with feature and label names
    feature_names = generate_random_string_list(X.shape[1])
    label_names = generate_random_string_list(num_classes)

    model = eq.EquineProtonet(
        embedding_model,
        num_classes,
        feature_names=feature_names,
        label_names=label_names,
    )
    model.train_model(dataset, num_episodes=2, **train_kwargs)

    new_model = use_save_load_model_tests(
        model, X, tmp_filename="protonet_save_load_with_feature_and_label_names.eq"
    )

    assert new_model.get_feature_names() == feature_names, (
        "feature_names changed on reload"
    )
    assert new_model.get_label_names() == label_names, "label_names changed on reload"


def test_full_covariance_trains_with_an_embedding_wider_than_the_support():
    """64 embedding dimensions from 25 support examples per class: the class
    covariances are rank-deficient, and training must keep the weights finite."""
    torch.manual_seed(0)
    x = torch.rand(300, 8)
    y = torch.arange(3).repeat_interleave(100)
    embedding = torch.nn.Sequential(
        torch.nn.Linear(8, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64)
    )
    model = eq.EquineProtonet(embedding, emb_out_dim=64, cov_type=eq.CovType.FULL)
    model.train_model(torch.utils.data.TensorDataset(x, y), num_episodes=5)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert_valid_prediction(model.predict(x[:10]), 10, 3)


def test_full_covariance_trains_with_large_embeddings():
    """Full-covariance training keeps the weights finite on inputs in the hundreds,
    whose float32 class covariance needs more than the epsilon ridge to stay positive
    definite."""
    torch.manual_seed(0)
    x = torch.rand(300, 8) * 1000
    y = torch.arange(3).repeat_interleave(100)
    embedding = torch.nn.Sequential(
        torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 24)
    )
    model = eq.EquineProtonet(embedding, emb_out_dim=24, cov_type=eq.CovType.FULL)
    model.train_model(torch.utils.data.TensorDataset(x, y), num_episodes=10)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert_valid_prediction(model.predict(x[:10]), 10, 3)


def test_full_covariance_trains_with_a_rank_deficient_embedding():
    """A 2-unit bottleneck makes every 8-dimensional class covariance rank-deficient,
    and with inputs in the thousands float32 rounding leaves it indefinite.
    Full-covariance training keeps the weights finite."""
    torch.manual_seed(0)
    x = torch.rand(300, 8) * 1000
    y = torch.arange(3).repeat_interleave(100)
    embedding = torch.nn.Sequential(
        torch.nn.Linear(8, 2), torch.nn.ReLU(), torch.nn.Linear(2, 8)
    )
    model = eq.EquineProtonet(embedding, emb_out_dim=8, cov_type=eq.CovType.FULL)
    model.train_model(
        torch.utils.data.TensorDataset(x, y), num_episodes=10, support_size=50
    )
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert_valid_prediction(model.predict(x[:10]), 10, 3)


def test_shared_regularization_of_a_unit_covariance_falls_back_to_epsilon() -> None:
    torch.manual_seed(0)
    proto = Protonet(torch.nn.Identity(), 3, eq.CovType.UNIT, "shared", 1e-5)
    support = OrderedDict({0: torch.randn(5, 3), 1: torch.randn(5, 3)})
    proto.train()
    with pytest.warns(UserWarning, match="reverting to epsilon regularization"):
        proto.update_support(support)
    assert torch.equal(proto.covariance, torch.ones(2, 3))


def test_shared_regularization_blends_full_covariances() -> None:
    torch.manual_seed(0)
    support = OrderedDict({0: torch.randn(5, 3), 1: 3 * torch.randn(7, 3)})
    covariances = {}
    for cov_reg_type in ("shared", "epsilon"):
        proto = Protonet(torch.nn.Identity(), 3, eq.CovType.FULL, cov_reg_type, 1e-5)
        proto.train()
        proto.update_support(support)
        covariances[cov_reg_type] = proto.covariance
    shared = covariances["shared"]
    assert shared.shape == (2, 3, 3)
    assert torch.isfinite(shared).all()
    torch.testing.assert_close(shared, shared.transpose(1, 2))
    assert not torch.allclose(shared, covariances["epsilon"])
    assert not torch.allclose(shared[0], shared[1])


def _three_class_data(sizes: tuple[int, int, int]):
    x = torch.cat([torch.randn(n, FEATURES) + 5 * i for i, n in enumerate(sizes)])
    y = torch.cat([torch.full((n,), float(i)) for i, n in enumerate(sizes)])
    return x, y


def test_update_support_refuses_a_class_left_with_one_support_example() -> None:
    model, _, _ = _briefly_trained_protonet_with_data()
    support = model.get_support()
    x, y = _three_class_data((10, 10, 2))
    with pytest.raises(ValueError, match="Class 2 has 1 support example"):
        model.update_support(x, y, calib_frac=0.5, label_names=["a", "b", "c"])
    assert model.get_support() is support
    assert model.get_label_names() is None


def test_update_support_refuses_a_class_left_with_one_calibration_example() -> None:
    model, _, _ = _briefly_trained_protonet_with_data()
    support, kdes = model.get_support(), dict(model.outlier_score_kde)
    x, y = _three_class_data((10, 10, 6))
    with pytest.raises(ValueError, match="Class 2 has 1 calibration example"):
        model.update_support(x, y, calib_frac=0.2)
    assert model.get_support() is support
    assert model.outlier_score_kde == kdes


def test_train_model_refuses_a_class_with_one_calibration_example() -> None:
    torch.manual_seed(0)
    x, y = _three_class_data((30, 30, 7))
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, use_temperature=True
    )
    model.eval()
    with torch.no_grad():
        model.temperature.fill_(0.5)
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(ValueError, match="Class 2 has 1 calibration example"):
        model.train_model(
            torch.utils.data.TensorDataset(x, y),
            num_episodes=5,
            calib_frac=0.2,
            support_size=4,
            way=3,
            episode_size=30,
        )
    assert all(torch.equal(b, p) for b, p in zip(before, model.parameters()))
    assert len(model.model.support) == 0
    assert not model.training
    assert model.temperature.item() == 0.5


@pytest.mark.parametrize(
    "sizes, calib_frac, support_size, num_episodes, left, needed",
    [
        ((30, 30, 3), 0.9, 2, 5, 0, 3),
        ((30, 30, 4), 0.8, 2, 5, 1, 3),
        ((30, 30, 3), 0.9, 2, 0, 0, 2),
        ((20, 20, 19), 0.2, 16, 0, 15, 16),
    ],
)
def test_train_model_refuses_a_class_without_enough_training_examples(
    sizes, calib_frac, support_size, num_episodes, left, needed
) -> None:
    torch.manual_seed(0)
    x, y = _three_class_data(sizes)
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    with pytest.raises(
        ValueError, match=f"Class 2 has {left} training example.*at least {needed}"
    ):
        model.train_model(
            torch.utils.data.TensorDataset(x, y),
            num_episodes=num_episodes,
            calib_frac=calib_frac,
            support_size=support_size,
            way=2,
            episode_size=10,
        )
    assert len(model.model.support) == 0


def test_update_support_leaves_the_model_as_it_was_when_a_later_step_fails(
    monkeypatch,
) -> None:
    model, _, _ = _briefly_trained_protonet_with_data()
    support, kdes = model.get_support(), dict(model.outlier_score_kde)
    tensors = stored_tensors(model)

    def fail(ood_dists, calib_y):
        raise ValueError("fitting failed")

    monkeypatch.setattr(model, "_fit_outlier_scores", fail)
    x, y = _three_class_data((10, 10, 10))
    with pytest.raises(ValueError, match="fitting failed"):
        model.update_support(x, y, calib_frac=0.5, label_names=["a", "b", "c"])
    assert model.get_support() is support
    assert model.outlier_score_kde == kdes
    restored = stored_tensors(model)
    assert restored.keys() == tensors.keys()
    assert all(restored[name] is tensors[name] for name in tensors)
    assert model.get_label_names() is None


def test_outlier_scores_keep_the_kdes_when_a_kde_fit_fails(monkeypatch) -> None:
    model, _, _ = _briefly_trained_protonet_with_data()
    kdes = dict(model.outlier_score_kde)
    fit = eq.equine_protonet.gaussian_kde
    calls = []

    def fail_on_the_second_class(dataset):
        calls.append(dataset)
        if len(calls) == 2:
            raise ValueError("fit failed")
        return fit(dataset)

    monkeypatch.setattr(eq.equine_protonet, "gaussian_kde", fail_on_the_second_class)
    ood_dists = torch.tensor([2.0, 3.0, 1.0, 4.0, 5.0, 6.0])
    calib_y = torch.tensor([0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="fit failed"):
        model._fit_outlier_scores(ood_dists, calib_y)
    assert model.outlier_score_kde == kdes


def test_outlier_scores_refuse_a_non_finite_distance_and_keep_the_kdes() -> None:
    model, _, _ = _briefly_trained_protonet_with_data()
    kdes = dict(model.outlier_score_kde)
    ood_dists = torch.tensor([2.0, 3.0, 1.0, 4.0, 5.0, float("inf")])
    calib_y = torch.tensor([0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="class 2 has a NaN or infinite"):
        model._fit_outlier_scores(ood_dists, calib_y)
    assert model.outlier_score_kde == kdes


def test_train_model_without_episodes_needs_only_support_size_rows() -> None:
    torch.manual_seed(1)
    x, y = _three_class_data((20, 20, 20))
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    model.train_model(
        torch.utils.data.TensorDataset(x, y),
        num_episodes=0,
        calib_frac=0.2,
        support_size=16,
    )
    assert [len(rows) for rows in model.get_support().values()] == [16, 16, 16]


def test_train_model_refuses_a_support_size_below_two() -> None:
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, use_temperature=True
    )
    model.eval()
    with torch.no_grad():
        model.temperature.fill_(0.5)
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(ValueError, match="support_size must be at least 2"):
        model.train_model(
            dataset, num_episodes=5, support_size=1, way=3, episode_size=30
        )
    assert all(torch.equal(b, p) for b, p in zip(before, model.parameters()))
    assert len(model.model.support) == 0
    assert not model.training
    assert model.temperature.item() == 0.5


def test_outlier_scores_refuse_equal_distances_and_keep_the_kdes() -> None:
    model, _, _ = _briefly_trained_protonet_with_data()
    kdes = dict(model.outlier_score_kde)
    ood_dists = torch.tensor([2.0, 3.0, 1.0, 1.0, 4.0, 5.0])
    calib_y = torch.tensor([0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="calibration example of class 1"):
        model._fit_outlier_scores(ood_dists, calib_y)
    assert model.outlier_score_kde == kdes


@pytest.mark.parametrize("cov_type", [eq.CovType.DIAGONAL, eq.CovType.FULL])
def test_covariance_needs_two_examples_per_class(cov_type) -> None:
    proto = Protonet(
        torch.nn.Identity(), 3, cov_type, cov_reg_type="epsilon", epsilon=1e-5
    )
    support = OrderedDict({0: torch.randn(1, 3), 1: torch.randn(5, 3)})
    with pytest.raises(ValueError, match="needs at least 2 examples per class; got 1"):
        proto.update_support(support)


def test_load_refuses_a_class_with_one_support_example(tmp_path) -> None:
    model, _, _ = _briefly_trained_protonet_with_data()
    checkpoint = model._to_checkpoint()
    checkpoint["support"][0] = checkpoint["support"][0][:1]
    path = str(tmp_path / "one_example.eq")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="holds 1 example\\(s\\) of class 0"):
        eq.EquineProtonet.load(path)


def test_save_refuses_an_untrained_model(tmp_path) -> None:
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    with pytest.raises(RuntimeError, match="no support examples to save"):
        model.save(str(tmp_path / "untrained.eq"))
    assert not (tmp_path / "untrained.eq").exists()


def test_predict_refuses_an_empty_batch() -> None:
    model, _, _ = _briefly_trained_protonet_with_data()
    with pytest.raises(ValueError, match="X is empty"):
        model.predict(torch.empty(0, FEATURES))


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_predict_refuses_non_finite_embeddings(value) -> None:
    model, _, _ = _briefly_trained_protonet_with_data()
    with pytest.raises(ValueError, match="embeddings of X hold NaN or infinite"):
        model.predict(torch.full((3, FEATURES), value))


class _ImputingEmbedding(torch.nn.Module):
    """A linear embedding that reads a NaN feature as 0."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(FEATURES, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.nan_to_num(x))


def test_predict_accepts_nan_input_that_the_embedding_handles() -> None:
    torch.manual_seed(0)
    dataset, x, _ = separable_dataset()
    model = eq.EquineProtonet(_ImputingEmbedding(), CLASSES)
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    rows = x[:5].clone()
    rows[:, 0] = float("nan")
    expected = model.predict(torch.nan_to_num(rows))
    out = model.predict(rows)
    assert torch.equal(out.classes, expected.classes)
    assert torch.equal(out.ood_scores, expected.ood_scores)


def test_equine_protonet_deepcopy_after_train_model() -> None:
    """A trained Protonet keeps no autograd history; deepcopy works."""
    model, _, _ = _briefly_trained_protonet_with_data()
    for name, t in stored_tensors(model).items():
        assert t.grad_fn is None, f"{name} still has a grad_fn"
    copy.deepcopy(model)


def test_equine_protonet_loaded_from_a_file_keeps_no_autograd_history(
    tmp_path,
) -> None:
    model, _, _ = _briefly_trained_protonet_with_data()
    path = str(tmp_path / "protonet.eq")
    model.save(path)
    loaded = eq.load_equine_model(path)
    for name, t in stored_tensors(loaded).items():
        assert t.grad_fn is None, f"{name} still has a grad_fn"
    copy.deepcopy(loaded)


def test_equine_protonet_deepcopy_after_update_support() -> None:
    """A Protonet's public ``update_support`` keeps no autograd history."""
    model, x, y = _briefly_trained_protonet_with_data()
    model.update_support(x, y, calib_frac=0.2)
    for name, t in stored_tensors(model).items():
        assert t.grad_fn is None, f"{name} still has a grad_fn"
    copy.deepcopy(model)


def _protonet_trained_on_labels(labels, **kwargs):
    """
    A model trained on ``separable_dataset`` with class ``k`` relabelled
    ``labels[k]``, with the random draws made for labels 0, 1, 2.
    """
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    y = torch.tensor(labels)[y].float()
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, **kwargs)
    model.train_model(
        torch.utils.data.TensorDataset(x, y),
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    return model, x, y


def _ood_scores_from_kde(model, rows, label):
    """The OOD scores of ``rows`` computed with the KDE fitted for ``label``."""
    with torch.no_grad():
        preds, dists, embeddings = model.model._forward_with_embeddings(rows)
        ood_dists = model._compute_ood_dist(embeddings, preds, dists)
    kde = model.outlier_score_kde[label]
    tails = [kde.integrate_box_1d(d, float("inf")) for d in ood_dists.tolist()]
    scores = [1.0 - min(max(float(p), 0.0), 1.0) for p in tails]
    return torch.tensor(scores, dtype=ood_dists.dtype)


@pytest.mark.parametrize("labels", [(1, 2, 3), (0, 2, 5)])
def test_non_contiguous_labels_predict_like_labels_from_zero(labels) -> None:
    reference, x, _ = _protonet_trained_on_labels((0, 1, 2))
    model, _, _ = _protonet_trained_on_labels(labels)
    assert list(model.get_support()) == list(labels)
    expected, out = reference.predict(x), model.predict(x)
    assert torch.equal(out.classes, expected.classes)
    assert torch.equal(out.ood_scores, expected.ood_scores)


def test_ood_score_uses_the_kde_of_the_predicted_label() -> None:
    model, x, y = _protonet_trained_on_labels((0, 2, 5))
    rows = x[y == 5]
    out = model.predict(rows)
    assert out.classes.argmax(dim=1).tolist() == [2] * len(rows)
    expected = _ood_scores_from_kde(model, rows, 5)
    torch.testing.assert_close(out.ood_scores, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("labels", [(1, 2, 3), (0, 2, 5)])
def test_temperature_calibrates_with_non_contiguous_labels(labels) -> None:
    reference, _, _ = _protonet_trained_on_labels((0, 1, 2), use_temperature=True)
    model, _, _ = _protonet_trained_on_labels(labels, use_temperature=True)
    assert torch.equal(model.temperature, reference.temperature)


def test_calibrate_temperature_rejects_a_label_outside_the_support() -> None:
    model, x, _ = _protonet_trained_on_labels((0, 2, 5))
    with pytest.raises(ValueError, match=r"Labels \[1.0\] are not among"):
        model.calibrate_temperature(x[:3], torch.tensor([0.0, 1.0, 2.0]))


def test_update_support_keeps_kdes_only_for_the_new_labels(tmp_path) -> None:
    model, x, y = _protonet_trained_on_labels((0, 1, 2))
    keep = y < 2
    model.update_support(x[keep], y[keep] + 7, calib_frac=0.2)
    assert list(model.outlier_score_kde) == [7, 8]
    path = str(tmp_path / "m.eq")
    model.save(path)
    assert list(torch.load(path, weights_only=True)["outlier_kde"]) == [7, 8]


def test_update_support_scores_ood_with_the_new_labels_kdes() -> None:
    model, x, y = _protonet_trained_on_labels((0, 1, 2))
    keep = y < 2
    model.update_support(x[keep], y[keep] + 7, calib_frac=0.2)
    rows = x[y == 1]
    out = model.predict(rows)
    assert out.classes.argmax(dim=1).tolist() == [1] * len(rows)
    expected = _ood_scores_from_kde(model, rows, 8)
    torch.testing.assert_close(out.ood_scores, expected, atol=1e-6, rtol=0)


def test_generate_model_summary_maps_labels_through_the_support() -> None:
    """Column ``j`` of a Protonet's predictions is its ``j``-th support label."""
    model, _, _ = _protonet_trained_on_labels((0, 2, 5))
    true_y = torch.tensor([0, 0, 2, 2, 5, 5])
    probs = torch.nn.functional.one_hot(torch.tensor([0, 0, 1, 1, 2, 1]), 3).float()
    out = eq.EquineOutput(classes=probs, ood_scores=torch.zeros(6), embeddings=probs)
    summary = eq.generate_model_summary(model, out, true_y)
    assert summary["accuracy"] == pytest.approx(2.5 / 3)
    assert summary["confusionMatrix"] == [[2, 0, 0], [0, 2, 0], [0, 1, 1]]
    assert summary["brierScore"] == pytest.approx(2 / 6)


def test_generate_model_summary_of_labels_from_zero_matches_the_plain_metrics() -> None:
    model, x, y = _protonet_trained_on_labels((0, 1, 2))
    gen = torch.Generator().manual_seed(5)
    out = model.predict(x + 2.0 * torch.randn(x.shape, generator=gen))
    expected = eq.generate_model_metrics(out, y.long())
    for labels in (y, y.long()):
        summary = eq.generate_model_summary(model, out, labels)
        for key, value in expected.items():
            assert summary[key] == value, key


def test_a_reordered_support_maps_columns_in_the_summary_and_temperature() -> None:
    reference, x, _ = _protonet_trained_on_labels((0, 1, 2), use_temperature=True)
    model, _, y = _protonet_trained_on_labels((0, 1, 2), use_temperature=True)
    support = model.get_support()
    with torch.no_grad():
        model.model.update_support(
            OrderedDict((label, support[label]) for label in (2, 0, 1))
        )
    out = model.predict(x)
    predicted_labels = torch.tensor([2, 0, 1])[out.classes.argmax(dim=1)]
    summary = eq.utils.generate_model_summary(model, out, predicted_labels)
    assert summary["accuracy"] == pytest.approx(1.0)
    reference.calibrate_temperature(x, y, num_calibration_epochs=3)
    model.calibrate_temperature(x, y, num_calibration_epochs=3)
    torch.testing.assert_close(model.temperature, reference.temperature)


def test_predict_names_support_labels_without_an_outlier_kde() -> None:
    model, x, y = _protonet_trained_on_labels((0, 1, 2))
    with torch.no_grad():
        model.model.update_support(
            eq.utils.generate_support(x, y + 3, 5, [3.0, 4.0, 5.0])
        )
    with pytest.raises(ValueError, match=r"support labels \[3, 4, 5\]"):
        model.predict(x[:3])
    model.update_support(x, y + 3, calib_frac=0.2)
    assert model.predict(x[:3]).classes.shape == (3, 3)


def test_calibrate_temperature_needs_a_support() -> None:
    _, x, y = separable_dataset()
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, use_temperature=True
    )
    with pytest.raises(ValueError, match="No support examples found"):
        model.calibrate_temperature(x[:10], y[:10].float())


def _hand_ood_distances(**kwargs) -> list[float]:
    """``_compute_ood_dist`` in float64 with unit diagonal covariances: class
    prototypes [0, 0] and [10, 10], global mean [0, 4]; the query [3, 4] is
    predicted as class 0 and [10, 0] as class 1. Their squared Mahalanobis
    distances are 25 and 100 to the predicted class and 9 and 116 to the
    global mean."""
    with golden_dtype():
        model = eq.EquineProtonet(BasicEmbeddingModel(2, 2), 2, **kwargs)
        model.model.global_mean = torch.tensor([0.0, 4.0])
        model.model.global_covariance = torch.ones(1, 2)
        queries = torch.tensor([[3.0, 4.0], [10.0, 0.0]])
        distances = model.model.compute_distance(
            queries, torch.tensor([[0.0, 0.0], [10.0, 10.0]]), torch.ones(2, 2)
        )
        predictions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        return model._compute_ood_dist(queries, predictions, distances).tolist()


def _distance(squared: float) -> float:
    """The distance ``compute_distance`` returns for a squared Mahalanobis distance."""
    return math.sqrt(squared + DEFAULT_EPSILON)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        pytest.param({}, [16.0, -16.0], id="default"),
        pytest.param({"ood_distance": "rmd_sq"}, [16.0, -16.0], id="rmd_sq"),
        pytest.param(
            {"ood_distance": "rmd_sqrt"},
            [_distance(25) - _distance(9), _distance(100) - _distance(116)],
            id="rmd_sqrt",
        ),
        pytest.param(
            {"relative_mahal": False},
            [_distance(25), _distance(100)],
            id="absolute",
        ),
        pytest.param(
            {"relative_mahal": False, "ood_distance": "rmd_sqrt"},
            [_distance(25), _distance(100)],
            id="absolute-rmd_sqrt",
        ),
    ],
)
def test_ood_distance_of_hand_computed_queries(kwargs, expected) -> None:
    assert _hand_ood_distances(**kwargs) == pytest.approx(expected, abs=1e-9, rel=0)


@pytest.mark.parametrize(
    "kwargs, scaled_ood",
    [
        pytest.param({}, False, id="rmd_sq"),
        pytest.param({"ood_distance": "rmd_sqrt"}, True, id="rmd_sqrt"),
        pytest.param({"relative_mahal": False}, True, id="absolute"),
    ],
)
def test_temperature_reaches_the_ood_scores_except_with_rmd_sq(
    kwargs, scaled_ood
) -> None:
    """The temperature scales the class probabilities of every model, and the
    OOD scores of all but the relative ``"rmd_sq"`` statistic."""
    torch.manual_seed(0)
    dataset, x, _ = separable_dataset()
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, use_temperature=True, **kwargs
    )
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    model.temperature.fill_(0.5)
    halved = model.predict(x)
    model.temperature.fill_(1.0)
    plain = model.predict(x)
    assert not torch.equal(halved.classes, plain.classes)
    assert torch.equal(halved.ood_scores, plain.ood_scores) is not scaled_ood


def test_ood_distance_is_checked_at_construction() -> None:
    with pytest.raises(
        ValueError, match=r"ood_distance must be one of \['rmd_sq', 'rmd_sqrt'\]"
    ):
        eq.EquineProtonet(BasicEmbeddingModel(2, 2), 2, ood_distance="rmd")
    with pytest.raises(BeartypeCallHintParamViolation):
        eq.EquineProtonet(BasicEmbeddingModel(2, 2), 2, ood_distance=1)


def _outlier_scores_per_sample(model, ood_dists, predictions):
    """
    The OOD score of each example from ``integrate_box_1d`` of its predicted
    label's KDE, one example at a time.
    """
    label_order = model._label_order()
    ood_scores = torch.zeros_like(ood_dists)
    for i in range(len(predictions)):
        kde = model.outlier_score_kde[label_order[int(torch.argmax(predictions[i]))]]
        p_value = kde.integrate_box_1d(ood_dists[i].cpu().numpy(), np.inf)
        ood_scores[i] = 1.0 - np.clip(p_value, 0.0, 1.0)
    return ood_scores


@pytest.mark.parametrize("bandwidth", ["scott", "silverman", 0.05, 0.3, 3.0])
@pytest.mark.parametrize("n", [2, 3, 8, 100])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
def test_outlier_scores_match_scipy_per_sample(dtype, n, bandwidth) -> None:
    """
    Scores from KDEs of ``n`` points with the given bandwidth, at the points,
    across the bulk, in both tails and at infinity, for examples predicted as
    the first or last label (none as the middle one).
    """
    model, _, _ = _protonet_trained_on_labels((0, 2, 5))
    gen = torch.Generator().manual_seed(n)
    data = OrderedDict(
        [
            (0, 3.0 * torch.randn(n, generator=gen, dtype=torch.float64) - 1.0),
            (2, torch.rand(n, generator=gen, dtype=torch.float64)),
            (5, 1000.0 + torch.rand(n, generator=gen, dtype=torch.float64)),
        ]
    )
    model.outlier_score_kde = OrderedDict()
    for label, points in data.items():
        kde = gaussian_kde(points.to(dtype).numpy())
        kde.set_bandwidth(bandwidth)
        model.outlier_score_kde[label] = kde
    k = torch.tensor([5.0, 8.0, 10.0, 40.0, 1e3], dtype=torch.float64)
    queries, columns = [], []
    for column, label in ((0, 0), (2, 5)):
        points = data[label].to(dtype).double()
        h = float(np.sqrt(model.outlier_score_kde[label].covariance[0, 0]))
        lo, hi = float(points.min()) - 3 * h, float(points.max()) + 3 * h
        mean, std = points.mean(), points.std()
        column_queries = torch.cat(
            [
                points,
                torch.linspace(lo, hi, 50, dtype=torch.float64),
                mean - k * std,
                mean + k * std,
                torch.tensor([float("inf"), float("-inf")], dtype=torch.float64),
            ]
        )
        queries.append(column_queries)
        columns.append(torch.full((len(column_queries),), column))
    order = torch.randperm(sum(len(q) for q in queries), generator=gen)
    ood_dists = torch.cat(queries)[order].to(dtype)
    predictions = torch.nn.functional.one_hot(torch.cat(columns)[order], 3).to(dtype)

    ood_scores = model._compute_outlier_scores(ood_dists, predictions)

    assert ood_scores.dtype == dtype
    assert ood_scores.device == ood_dists.device
    expected = _outlier_scores_per_sample(model, ood_dists, predictions)
    torch.testing.assert_close(ood_scores, expected, atol=1e-12, rtol=0)


def test_outlier_scores_follow_kde_weights() -> None:
    """A KDE with non-uniform weights scores as ``integrate_box_1d`` does."""
    model, _, _ = _protonet_trained_on_labels((0, 2, 5))
    kde = gaussian_kde(np.array([0.0, 1.0, 2.0, 10.0]), weights=[0.7, 0.1, 0.1, 0.1])
    model.outlier_score_kde = OrderedDict((label, kde) for label in (0, 2, 5))
    ood_dists = torch.tensor([-3.0, 1.0, 5.0, 12.0], dtype=torch.float64)
    predictions = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 1]), 3).double()

    ood_scores = model._compute_outlier_scores(ood_dists, predictions)

    expected = _outlier_scores_per_sample(model, ood_dists, predictions)
    torch.testing.assert_close(ood_scores, expected, atol=1e-12, rtol=0)


def test_outlier_scores_refuse_a_kde_with_more_than_one_dimension() -> None:
    model, _, _ = _protonet_trained_on_labels((0, 2, 5))
    model.outlier_score_kde[2] = gaussian_kde(
        np.random.default_rng(0).normal(size=(2, 9))
    )
    predictions = torch.nn.functional.one_hot(torch.tensor([1]), 3).double()
    with pytest.raises(ValueError, match="KDE of support label 2 has 2 dimensions"):
        model._compute_outlier_scores(
            torch.tensor([0.5], dtype=torch.float64), predictions
        )
