# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from collections import OrderedDict

import pytest
import torch
from conftest import (
    BasicEmbeddingModel,
    assert_valid_prediction,
    generate_random_string_list,
    random_dataset,
    use_basic_embedding_model,
    use_save_load_model_tests,
)
from golden_data import CLASSES, FEATURES, separable_dataset
from hypothesis import given, settings, strategies as st

import equine as eq


@given(
    data_shape=st.tuples(
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
    ),
    num_classes=st.integers(min_value=2, max_value=256),
)
@settings(deadline=None)
def test_compute_embeddings(data_shape, num_classes):
    queries = torch.rand(data_shape)
    embed_model = BasicEmbeddingModel(data_shape[1], num_classes)
    model = eq.EquineProtonet(embed_model, num_classes)
    embeddings = model.model.compute_embeddings(queries)
    assert embeddings.shape == (data_shape[0], num_classes)


def test_compute_shared_covariance_refuses_unit_covariance() -> None:
    # regularize_covariance never asks for a shared UNIT covariance (it warns and
    # falls back to epsilon regularization), so the refusal is exercised directly.
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
    model.update_support(x, y, 0.5, label_names=["a", "b", "c"])
    assert model.get_label_names() == ["a", "b", "c"]


def test_update_support_rejects_wrong_label_names_length() -> None:
    model, x, y = _briefly_trained_protonet_with_data()
    with pytest.raises(
        ValueError,
        match=r"The length of label_names \(2\) does not match the number of classes \(3\)",
    ):
        model.update_support(x, y, 0.5, label_names=["only", "two"])


@st.composite
def embeddings(draw):
    dataset_row_count = draw(st.integers(min_value=100, max_value=1000))
    dataset_col_count = draw(st.integers(min_value=3, max_value=10))
    shape = (dataset_row_count, dataset_col_count)
    support_embeddings = torch.rand(shape)
    query_embeddings = torch.rand(shape)
    ptrs = [0]
    curr_idx = 0

    for i in range(dataset_col_count - 1):
        curr_idx += int(dataset_row_count / dataset_col_count)
        ptrs.append(curr_idx)

    ptrs.append(dataset_row_count - 1)
    ptrs = torch.Tensor(ptrs).to(torch.long)

    return support_embeddings, ptrs, query_embeddings


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
    model.update_support(X, Y, 0.5)
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
    # 10 support rows in 4 deep-feature dims keep the full covariance well
    # conditioned, so the derived support_size is sufficient here.
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
    model.update_support(X, Y, 0.5)
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
