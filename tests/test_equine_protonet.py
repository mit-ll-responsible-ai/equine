# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import copy
import functools
import math
import pickle
import re
import threading
import warnings
from collections import OrderedDict

import numpy as np
import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation
from builders import quick_protonet
from conftest import (
    LABEL_KINDS,
    BasicEmbeddingModel,
    assert_model_unchanged,
    assert_valid_prediction,
    generate_random_string_list,
    labels_of_kind,
    model_state,
    random_dataset,
    stored_tensors,
    use_basic_embedding_model,
    use_save_load_model_tests,
)
from golden_data import CLASSES, FEATURES, golden_dtype, separable_dataset
from hypothesis import given, settings, strategies as st
from scipy.stats import gaussian_kde

import equine as eq
from equine import _model_file
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


_SHARED_SUPPORT_SIZES = (3, 30, 12)


def _float64_support(sizes: tuple[int, ...]) -> OrderedDict[int, torch.Tensor]:
    """A float64 support with ``sizes[k]`` rows of width 8 for class ``k``, each
    class with its own spread and mean."""
    gen = torch.Generator().manual_seed(0)
    return OrderedDict(
        (
            label,
            (1 + 2 * label) * torch.randn(n, 8, generator=gen, dtype=torch.float64)
            + label,
        )
        for label, n in enumerate(sizes)
    )


def _float64_shared_protonet(
    cov_type: eq.CovType, sizes: tuple[int, ...] = _SHARED_SUPPORT_SIZES
) -> Protonet:
    """A ``Protonet`` using ``"shared"`` in training mode, given
    ``_float64_support(sizes)`` (by default 3, 30 and 12 rows)."""
    proto = Protonet(torch.nn.Identity(), 8, cov_type, "shared", DEFAULT_EPSILON)
    proto.train()
    proto.update_support(_float64_support(sizes))
    return proto


def _class_covariances(
    proto: Protonet, cov_type: eq.CovType
) -> OrderedDict[int, torch.Tensor]:
    """The ``cov_type`` covariance of each class in ``proto``'s support."""
    return OrderedDict(
        (label, proto.compute_covariance_by_type(cov_type, embeddings))
        for label, embeddings in proto.support_embeddings.items()
    )


def _pooled_covariance(covariances: OrderedDict[int, torch.Tensor]) -> torch.Tensor:
    """The within-class scatter of the 3, 30 and 12 support rows,
    ``2 C_0 + 29 C_1 + 11 C_2``, divided by ``N - K = 45 - 3``."""
    scatter = sum(
        (n - 1) * covariance
        for n, covariance in zip(_SHARED_SUPPORT_SIZES, covariances.values())
    )
    return scatter / (sum(_SHARED_SUPPORT_SIZES) - len(_SHARED_SUPPORT_SIZES))


@pytest.mark.parametrize("cov_type", [eq.CovType.DIAGONAL, eq.CovType.FULL])
def test_shared_covariance_weights_each_class_by_its_degrees_of_freedom(
    cov_type,
) -> None:
    """The shared covariance of classes with 3, 30 and 12 support rows is
    ``(2 C_0 + 29 C_1 + 11 C_2) / 42``, in the support's dtype."""
    proto = _float64_shared_protonet(cov_type)
    covariances = _class_covariances(proto, cov_type)
    shared = proto.compute_shared_covariance(covariances, cov_type)
    assert shared.dtype == torch.float64
    torch.testing.assert_close(
        shared, _pooled_covariance(covariances), atol=1e-6, rtol=0
    )


def test_compute_shared_covariance_refuses_unit_covariance() -> None:
    """A UNIT ``cov_type`` is refused before the support is read; DIAGONAL, as
    the message asks, pools the diagonal class covariances of a support."""
    model = eq.EquineProtonet(BasicEmbeddingModel(2, 3), 3)
    class_cov_dict = OrderedDict({0: torch.ones(3), 1: torch.ones(3)})
    message = (
        "compute_shared_covariance pools DIAGONAL or FULL covariances; got "
        "CovType.UNIT. Pass CovType.DIAGONAL or CovType.FULL."
    )
    with pytest.raises(ValueError) as err:
        model.model.compute_shared_covariance(class_cov_dict, eq.CovType.UNIT)
    assert str(err.value) == message

    proto = _float64_shared_protonet(eq.CovType.DIAGONAL)
    covariances = _class_covariances(proto, eq.CovType.DIAGONAL)
    with pytest.raises(ValueError) as err:
        proto.compute_shared_covariance(covariances, eq.CovType.UNIT)
    assert str(err.value) == message
    shared = proto.compute_shared_covariance(covariances, eq.CovType.DIAGONAL)
    torch.testing.assert_close(
        shared, _pooled_covariance(covariances), atol=1e-6, rtol=0
    )


def test_compute_shared_covariance_refuses_a_label_without_support() -> None:
    """A covariance whose label is not in the support is refused; the
    covariances of the support classes, as the message asks, are pooled."""
    proto = _float64_shared_protonet(eq.CovType.DIAGONAL)
    covariances = _class_covariances(proto, eq.CovType.DIAGONAL)
    with pytest.raises(ValueError) as err:
        proto.compute_shared_covariance(
            OrderedDict({**covariances, 5: covariances[0]}), eq.CovType.DIAGONAL
        )
    assert str(err.value) == (
        "class_cov_dict has labels [5], which have no support embeddings; pass "
        "the covariances of the support classes [0, 1, 2]."
    )
    shared = proto.compute_shared_covariance(covariances, eq.CovType.DIAGONAL)
    torch.testing.assert_close(
        shared, _pooled_covariance(covariances), atol=1e-6, rtol=0
    )


def test_compute_shared_covariance_refuses_a_model_without_support() -> None:
    """A ``Protonet`` that never had ``update_support`` is refused; after
    ``update_support``, as the message asks, the same covariances are pooled."""
    support = _float64_support(_SHARED_SUPPORT_SIZES)
    proto = Protonet(
        torch.nn.Identity(), 8, eq.CovType.DIAGONAL, "shared", DEFAULT_EPSILON
    )
    proto.train()
    covariances = OrderedDict(
        (label, proto.compute_covariance_by_type(eq.CovType.DIAGONAL, rows))
        for label, rows in support.items()
    )
    with pytest.raises(ValueError) as err:
        proto.compute_shared_covariance(covariances, eq.CovType.DIAGONAL)
    assert str(err.value) == (
        "compute_shared_covariance pools the covariances of the support classes, "
        "and the model has no support; call update_support first."
    )
    proto.update_support(support)
    shared = proto.compute_shared_covariance(covariances, eq.CovType.DIAGONAL)
    torch.testing.assert_close(
        shared, _pooled_covariance(covariances), atol=1e-6, rtol=0
    )


def _too_few_support_examples(total: int, classes: int) -> str:
    return (
        "compute_shared_covariance divides by the number of support examples minus "
        f"the number of classes, {total} - {classes} here; pooling needs more support "
        "examples than classes; pass the covariances of classes with at least 2 "
        "support examples."
    )


def test_compute_shared_covariance_refuses_an_empty_class_cov_dict() -> None:
    """An empty ``class_cov_dict`` (0 - 0 in the denominator) is refused; the
    covariances of the support classes, as the message asks, are pooled."""
    proto = _float64_shared_protonet(eq.CovType.DIAGONAL)
    with pytest.raises(ValueError) as err:
        proto.compute_shared_covariance(OrderedDict(), eq.CovType.DIAGONAL)
    assert str(err.value) == _too_few_support_examples(0, 0)
    covariances = _class_covariances(proto, eq.CovType.DIAGONAL)
    shared = proto.compute_shared_covariance(covariances, eq.CovType.DIAGONAL)
    torch.testing.assert_close(
        shared, _pooled_covariance(covariances), atol=1e-6, rtol=0
    )


def test_compute_shared_covariance_refuses_classes_of_one_support_example() -> None:
    """Two classes of one support example each (2 - 2 in the denominator) are
    refused; the covariance of the class with 5 support examples, as the message
    asks, pools to itself."""
    proto = _float64_shared_protonet(eq.CovType.UNIT, sizes=(1, 1, 5))
    zeros = torch.zeros(8, dtype=torch.float64)
    with pytest.raises(ValueError) as err:
        proto.compute_shared_covariance(
            OrderedDict({0: zeros, 1: zeros}), eq.CovType.DIAGONAL
        )
    assert str(err.value) == _too_few_support_examples(2, 2)
    covariance = proto.compute_covariance_by_type(
        eq.CovType.DIAGONAL, proto.support_embeddings[2]
    )
    shared = proto.compute_shared_covariance(
        OrderedDict({2: covariance}), eq.CovType.DIAGONAL
    )
    assert torch.isfinite(shared).all()
    torch.testing.assert_close(shared, covariance, atol=1e-6, rtol=0)


def test_update_support_sets_label_names() -> None:
    model, x, y = quick_protonet()
    assert model.get_label_names() is None
    model.update_support(x, y, calib_frac=0.5, label_names=["a", "b", "c"])
    assert model.get_label_names() == ["a", "b", "c"]


def test_update_support_rejects_wrong_label_names_length() -> None:
    model, x, y = quick_protonet()
    with pytest.raises(
        ValueError,
        match=r"The length of label_names \(2\) does not match the number of classes \(3\)",
    ):
        model.update_support(x, y, calib_frac=0.5, label_names=["only", "two"])


@pytest.mark.parametrize(
    "num_episodes, num_deep_features, kwargs",
    [
        pytest.param(10, 32, {}, id="unit"),
        pytest.param(
            10,
            32,
            {"cov_type": eq.CovType.DIAGONAL, "cov_reg_type": "shared"},
            id="shared_reg",
        ),
        pytest.param(
            5,
            4,
            {"cov_type": eq.CovType.FULL, "cov_reg_type": "epsilon"},
            id="full_cov",
        ),
        pytest.param(10, 32, {"use_temperature": True}, id="with_temperature"),
    ],
)
@given(random_dataset=random_dataset())
@settings(deadline=None)
def test_train_episodes(random_dataset, num_episodes, num_deep_features, kwargs):
    dataset, num_classes, train_kwargs = random_dataset
    X, Y = dataset.tensors
    embed_model = BasicEmbeddingModel(X.shape[1], num_deep_features)
    model = eq.EquineProtonet(embed_model, num_deep_features, **kwargs)
    before = model.temperature.item()
    assert before == 1.0, "init_temperature defaults to 1.0"
    train_dict = model.train_model(dataset, num_episodes=num_episodes, **train_kwargs)

    if model.use_temperature:
        assert "calib_x" in train_dict
        assert "calib_y" in train_dict
        after_train = model.temperature.item()
        assert after_train != before, "train_model(use_temperature=True) must calibrate"
        assert after_train > 0
        model.calibrate_temperature(
            train_dict["calib_x"], train_dict["calib_y"], 1, 0.01
        )
        after_calibration = model.temperature.item()
        assert after_calibration != after_train, "calibrate_temperature must move it"
        assert after_calibration > 0

    assert model.model.training is False, "Model leaves training mode"
    assert len(model.model.support) == num_classes  # type: ignore
    # Test on multiple predictions
    assert_valid_prediction(model.predict(X), len(X), num_classes)
    # Test on single prediction
    assert len(model(X[0])) == 1, "Single prediction works"
    assert_valid_prediction(model.predict(X[0]), 1, num_classes)

    support = model.get_support()
    assert support is not None and len(support) == num_classes, (
        "Support set is correct size"
    )
    prototypes = model.get_prototypes()
    assert prototypes is not None and len(prototypes) == num_classes, (
        "Prototypes set is correct size"
    )
    model.update_support(X, Y, calib_frac=0.5)
    assert (
        model.model.support is not None and len(model.model.support) == num_classes
    ), "Support set is correct size"


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

    use_save_load_model_tests(model, X)


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

    new_model = use_save_load_model_tests(model, X)
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

    new_model = use_save_load_model_tests(model, X)

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

    new_model = use_save_load_model_tests(model, X)

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


def _protonet_warnings(caught) -> list[str]:
    """The messages of the warnings in ``caught`` raised in equine_protonet.py."""
    return [str(w.message) for w in caught if w.filename.endswith("equine_protonet.py")]


def test_shared_regularization_uses_a_unit_covariance_as_is() -> None:
    torch.manual_seed(0)
    proto = Protonet(torch.nn.Identity(), 3, eq.CovType.UNIT, "shared", 1e-5)
    support = OrderedDict({0: torch.randn(5, 3), 1: torch.randn(5, 3)})
    proto.train()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        proto.update_support(support)
    ours = _protonet_warnings(caught)
    assert not ours, f"equine_protonet warned: {ours}"
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


@pytest.mark.parametrize(
    "cov_type, pred_cov_type",
    [
        pytest.param(eq.CovType.DIAGONAL, None, id="diag"),
        pytest.param(eq.CovType.FULL, None, id="full-train-diag-pred"),
        pytest.param(eq.CovType.FULL, eq.CovType.FULL, id="full"),
        pytest.param(eq.CovType.UNIT, None, id="unit-train-diag-pred"),
        pytest.param(eq.CovType.DIAGONAL, eq.CovType.UNIT, id="diag-train-unit-pred"),
    ],
)
def test_a_shared_model_reloads_with_its_classes_and_ood_scores(
    cov_type, pred_cov_type
) -> None:
    model, x, _ = quick_protonet(
        cov_type=cov_type, pred_cov_type=pred_cov_type, cov_reg_type="shared"
    )
    loaded = use_save_load_model_tests(model, x)
    assert loaded.cov_reg_type == "shared"
    assert loaded.model.cov_reg_type == "shared"


@pytest.mark.parametrize(
    "cov_type, pred_cov_type",
    [
        pytest.param(eq.CovType.UNIT, None, id="unit-train-diag-pred"),
        pytest.param(eq.CovType.UNIT, eq.CovType.FULL, id="unit-train-full-pred"),
        pytest.param(eq.CovType.DIAGONAL, eq.CovType.UNIT, id="diag-train-unit-pred"),
        pytest.param(eq.CovType.FULL, eq.CovType.UNIT, id="full-train-unit-pred"),
    ],
)
def test_shared_with_one_unit_covariance_type_trains_without_a_warning(
    cov_type, pred_cov_type
) -> None:
    """``"shared"`` with one UNIT covariance type is accepted; training and
    ``update_support`` use the UNIT covariance as is and do not warn."""
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        cov_type=cov_type,
        pred_cov_type=pred_cov_type,
        cov_reg_type="shared",
    )
    dataset, x, y = separable_dataset()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.train_model(
            dataset, num_episodes=2, support_size=10, way=3, episode_size=30
        )
        model.update_support(x, y, calib_frac=0.2)
    ours = _protonet_warnings(caught)
    assert not ours, f"equine_protonet warned: {ours}"
    assert model.model.cov_reg_type == "shared"
    assert_valid_prediction(model.predict(x), len(x), CLASSES)


def _assert_same_predictions(loaded, out, x) -> None:
    """Fail unless ``loaded`` predicts ``out``'s classes and OOD scores for ``x``."""
    again = loaded.predict(x)
    assert torch.allclose(again.classes, out.classes, atol=1e-6)
    assert torch.allclose(again.ood_scores, out.ood_scores, atol=1e-6)


def test_save_refuses_shared_set_on_a_model_whose_covariance_types_are_unit(
    tmp_path,
) -> None:
    """``"shared"`` set by attribute on a model whose ``cov_type`` and
    ``pred_cov_type`` are UNIT is refused by ``save``, which writes nothing; with
    ``"epsilon"``, as the message asks, the file reloads to the classes and OOD
    scores the model gave with ``"shared"``, which has no effect there."""
    model, x, y = quick_protonet(pred_cov_type=eq.CovType.UNIT)
    model.model.cov_reg_type = "shared"
    torch.manual_seed(0)
    model.update_support(x, y, calib_frac=0.2)
    shared = model.predict(x)
    path = tmp_path / "m.eq"
    with pytest.raises(ValueError) as err:
        model.save(str(path))
    assert str(err.value) == (
        "model.model.cov_reg_type is 'shared', which pools DIAGONAL or FULL "
        "covariances, and cov_type and pred_cov_type are CovType.UNIT; set "
        "model.model.cov_reg_type to 'epsilon', then save."
    )
    assert not path.exists()

    model.model.cov_reg_type = "epsilon"
    model.save(str(path))
    loaded = eq.load_equine_model(str(path))
    assert loaded.model.cov_reg_type == "epsilon"
    _assert_same_predictions(loaded, shared, x)


@pytest.mark.parametrize(
    "pred_cov_type, value, shown, repair",
    [
        pytest.param(
            None, "typo", "'typo'", "set it to 'epsilon' or 'shared'", id="typo"
        ),
        pytest.param(None, 5, "a int", "set it to 'epsilon' or 'shared'", id="int"),
        pytest.param(
            eq.CovType.UNIT,
            "typo",
            "'typo'",
            "set model.model.cov_reg_type to 'epsilon'",
            id="typo-unit-both",
        ),
    ],
)
def test_save_refuses_an_unknown_cov_reg_type_set_by_attribute(
    tmp_path, pred_cov_type, value, shown, repair
) -> None:
    """An unknown ``model.model.cov_reg_type`` set after training is refused by
    ``save``, which writes nothing; on a model whose covariance types are both
    UNIT the message offers only ``"epsilon"``. The value the message asks for
    saves a file that reloads to the same classes and OOD scores."""
    model, x, _ = quick_protonet(pred_cov_type=pred_cov_type)
    out = model.predict(x)
    model.model.cov_reg_type = value
    path = tmp_path / "m.eq"
    with pytest.raises(ValueError) as err:
        model.save(str(path))
    assert str(err.value) == (
        "model.model.cov_reg_type must be one of ['epsilon', 'shared']; got "
        f"{shown}; {repair}, then save."
    )
    assert not path.exists()

    model.model.cov_reg_type = "epsilon"
    model.save(str(path))
    _assert_same_predictions(eq.load_equine_model(str(path)), out, x)


def _unknown_cov_reg_type(shown: str) -> str:
    return f"cov_reg_type must be one of ['epsilon', 'shared']; got {shown}."


_UNKNOWN_WITH_UNIT = (
    "cov_reg_type must be one of ['epsilon', 'shared']; got 'typo'; pass "
    "cov_reg_type='epsilon', or 'shared' with a cov_type or pred_cov_type of "
    "CovType.DIAGONAL or CovType.FULL."
)


_SHARED_WITH_UNIT = (
    "cov_reg_type is 'shared', which pools DIAGONAL or FULL covariances, and "
    "cov_type and pred_cov_type are CovType.UNIT; pass a cov_type or pred_cov_type "
    "of CovType.DIAGONAL or CovType.FULL, or cov_reg_type='epsilon'."
)


@pytest.mark.parametrize(
    ("changes", "error", "message", "repair"),
    [
        pytest.param(
            {"cov_reg_type": "typo"},
            ValueError,
            _unknown_cov_reg_type("'typo'"),
            {"cov_reg_type": "epsilon"},
            id="unknown",
        ),
        *[
            pytest.param(
                {"cov_reg_type": "typo", "pred_cov_type": eq.CovType.UNIT},
                ValueError,
                _UNKNOWN_WITH_UNIT,
                repair,
                id=f"unknown-unit-both-repair-{name}",
            )
            for name, repair in (
                ("epsilon", {"cov_reg_type": "epsilon"}),
                (
                    "shared-cov_type",
                    {"cov_reg_type": "shared", "cov_type": eq.CovType.DIAGONAL},
                ),
                (
                    "shared-pred_cov_type",
                    {"cov_reg_type": "shared", "pred_cov_type": eq.CovType.FULL},
                ),
            )
        ],
        pytest.param(
            {"cov_reg_type": "shared", "pred_cov_type": eq.CovType.UNIT},
            ValueError,
            _SHARED_WITH_UNIT,
            {"cov_type": eq.CovType.DIAGONAL},
            id="shared-unit-both-repair-cov_type",
        ),
        pytest.param(
            {"cov_reg_type": "shared", "pred_cov_type": eq.CovType.UNIT},
            ValueError,
            _SHARED_WITH_UNIT,
            {"pred_cov_type": eq.CovType.FULL},
            id="shared-unit-both-repair-pred_cov_type",
        ),
        pytest.param(
            {
                "cov_reg_type": "shared",
                "cov_type": eq.CovType.UNIT,
                "pred_cov_type": eq.CovType.UNIT,
            },
            ValueError,
            _SHARED_WITH_UNIT,
            {"cov_reg_type": "epsilon"},
            id="shared-unit-both-repair-epsilon",
        ),
        pytest.param(
            {"cov_reg_type": 5},
            BeartypeCallHintParamViolation,
            None,
            {"cov_reg_type": "epsilon"},
            marks=pytest.mark.contracts,
            id="int-contracts-on",
        ),
        pytest.param(
            {"cov_reg_type": 5},
            ValueError,
            _unknown_cov_reg_type("a int"),
            {"cov_reg_type": "epsilon"},
            marks=pytest.mark.skipif(
                __debug__, reason="beartype refuses the wrong type first"
            ),
            id="int-python-O",
        ),
    ],
)
def test_cov_reg_type_is_checked_at_construction(
    changes, error, message, repair
) -> None:
    """An unknown ``cov_reg_type``, or ``"shared"`` with a UNIT ``cov_type`` and
    ``pred_cov_type``, is refused; the settings the message asks for train."""
    with pytest.raises(error) as err:
        eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, **changes)
    if message is not None:
        assert str(err.value) == message

    repaired = {**changes, **repair}
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, **repaired
    )
    dataset, x, _ = separable_dataset()
    model.train_model(dataset, num_episodes=2, support_size=10, way=3, episode_size=30)
    assert model.cov_reg_type == repaired["cov_reg_type"]
    assert model.model.cov_reg_type == repaired["cov_reg_type"]
    assert_valid_prediction(model.predict(x), len(x), CLASSES)


@pytest.mark.parametrize("cov_type", list(eq.CovType))
@pytest.mark.parametrize(
    "cov_reg_type, shown",
    [
        pytest.param("typo", "'typo'", id="typo"),
        pytest.param(
            5,
            "a int",
            marks=pytest.mark.skipif(
                __debug__, reason="beartype refuses the wrong type first"
            ),
            id="int-python-O",
        ),
    ],
)
def test_protonet_refuses_an_unknown_cov_reg_type(
    cov_type, cov_reg_type, shown
) -> None:
    """``regularize_covariance`` refuses an unknown ``cov_reg_type`` given to a
    ``Protonet`` directly; ``"epsilon"`` regularizes the same support."""
    torch.manual_seed(0)
    support = OrderedDict({0: torch.randn(5, 3), 1: torch.randn(5, 3)})
    proto = Protonet(torch.nn.Identity(), 3, cov_type, cov_reg_type, 1e-5)
    proto.train()
    with pytest.raises(ValueError) as err:
        proto.update_support(support)
    assert str(err.value) == _unknown_cov_reg_type(shown)

    repaired = Protonet(torch.nn.Identity(), 3, cov_type, "epsilon", 1e-5)
    repaired.train()
    repaired.update_support(support)
    assert repaired.covariance.shape[0] == 2
    assert torch.isfinite(repaired.covariance).all()


def _three_class_data(sizes: tuple[int, int, int]):
    x = torch.cat([torch.randn(n, FEATURES) + 5 * i for i, n in enumerate(sizes)])
    y = torch.cat([torch.full((n,), float(i)) for i, n in enumerate(sizes)])
    return x, y


def test_update_support_refuses_a_class_left_with_one_support_example() -> None:
    model, _, _ = quick_protonet()
    support = model.get_support()
    x, y = _three_class_data((10, 10, 2))
    with pytest.raises(ValueError, match="Class 2 has 1 support example"):
        model.update_support(x, y, calib_frac=0.5, label_names=["a", "b", "c"])
    assert model.get_support() is support
    assert model.get_label_names() is None


def test_update_support_refuses_a_class_left_with_one_calibration_example() -> None:
    model, _, _ = quick_protonet()
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
    before = model_state(model)
    with pytest.raises(ValueError, match="Class 2 has 1 calibration example"):
        model.train_model(
            torch.utils.data.TensorDataset(x, y),
            num_episodes=5,
            calib_frac=0.2,
            support_size=4,
            way=3,
            episode_size=30,
        )
    assert_model_unchanged(model, before)


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
    before = model_state(model)
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
    assert_model_unchanged(model, before)


def test_update_support_leaves_the_model_as_it_was_when_a_later_step_fails(
    monkeypatch,
) -> None:
    model, _, _ = quick_protonet()
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
    model, _, _ = quick_protonet()
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
    model, _, _ = quick_protonet()
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


def test_train_model_accepts_one_query_example_per_class() -> None:
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    model.train_model(
        dataset, num_episodes=2, support_size=10, way=CLASSES, episode_size=CLASSES
    )
    assert [len(rows) for rows in model.get_support().values()] == [10, 10, 10]


@pytest.mark.parametrize("way, episode_size", [(1, 100), (3, 0)])
def test_train_model_without_episodes_accepts_any_way_and_episode_size(
    way, episode_size
) -> None:
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    model.train_model(
        dataset, num_episodes=0, support_size=10, way=way, episode_size=episode_size
    )
    assert [len(rows) for rows in model.get_support().values()] == [10, 10, 10]


class _FlatRows(torch.nn.Module):
    """A linear embedding of each example flattened to a row; a 1-D input is one
    feature per example."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.reshape(len(x), -1))


def _one_feature_dataset(rows: bool) -> torch.utils.data.TensorDataset:
    """The first feature of ``separable_dataset``: 1-D, or one column per row."""
    _, x, y = separable_dataset()
    feature = x[:, 0] + 3 * y
    return torch.utils.data.TensorDataset(
        feature.unsqueeze(1) if rows else feature, y.float()
    )


def _one_class(dataset):
    x, y = dataset[:]
    keep = y == 0
    return torch.utils.data.TensorDataset(x[keep], y[keep])


def _with_temperature() -> tuple[eq.EquineProtonet, torch.utils.data.TensorDataset]:
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, use_temperature=True
    )
    return model, separable_dataset()[0]


def _without_temperature() -> tuple[eq.EquineProtonet, torch.utils.data.TensorDataset]:
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    return model, separable_dataset()[0]


def _one_feature_model() -> tuple[eq.EquineProtonet, torch.utils.data.TensorDataset]:
    model = eq.EquineProtonet(_FlatRows(), CLASSES, use_temperature=True)
    return model, _one_feature_dataset(rows=False)


def _calib_frac_message(value: object) -> str:
    return f"calib_frac must be a number strictly between 0 and 1; got {value!r}."


def _epochs_message(shown: str) -> str:
    return (
        "num_calibration_epochs must be a non-negative integer (an int >= 0, not a "
        f"bool; 0 leaves the temperature unchanged); got {shown}."
    )


def _lr_message(value: object) -> str:
    return (
        "calibration_lr must be a finite float above 0 (not an int or a bool), such as "
        f"0.01 or 1.0; got {value!r}."
    )


_ONE_FEATURE_MESSAGE = (
    "dataset inputs must have a row dimension and at least one feature dimension; "
    "got shape (120,)."
)


@pytest.mark.parametrize(
    ("build", "changes", "message", "repair"),
    [
        *[
            pytest.param(
                _with_temperature,
                {"calib_frac": value},
                _calib_frac_message(value),
                {"calib_frac": 0.2},
                id=f"calib_frac={value}",
            )
            for value in (1.5, 1.0, 0.0, -0.25, math.nan)
        ],
        *[
            pytest.param(
                _with_temperature,
                {"num_calibration_epochs": value},
                _epochs_message(shown),
                {"num_calibration_epochs": 1},
                id=f"num_calibration_epochs={value}",
            )
            for value, shown in ((-1, "int -1"), (True, "bool"))
        ],
        *[
            pytest.param(
                _with_temperature,
                {"calibration_lr": value},
                _lr_message(value),
                {"calibration_lr": 0.01},
                id=f"calibration_lr={value}",
            )
            for value in (0.0, -0.01, math.nan, math.inf)
        ],
        pytest.param(
            _without_temperature,
            {"calibration_lr": -0.01},
            _lr_message(-0.01),
            {"calibration_lr": 0.01},
            id="calibration_lr=-0.01-without-use_temperature",
        ),
        *[
            pytest.param(
                _one_feature_model,
                {"num_episodes": episodes},
                _ONE_FEATURE_MESSAGE,
                {"dataset": _one_feature_dataset(rows=True)},
                id=f"1-d-inputs-num_episodes={episodes}",
            )
            for episodes in (3, 0)
        ],
        pytest.param(
            _with_temperature,
            {"support_size": 1},
            "support_size must be at least 2",
            {"support_size": 5},
            id="support_size=1",
        ),
        *[
            pytest.param(
                _with_temperature,
                {"way": way, "episode_size": episode_size},
                message,
                {"way": 3, "episode_size": 30},
                id=f"way={way}-episode_size={episode_size}",
            )
            for way, episode_size, message in (
                (1, 1, "way must be between 2 and the number of classes"),
                (0, 30, "way must be between 2 and the number of classes"),
                (4, 30, "way must be between 2 and the number of classes"),
                (3, 2, "episode_size must be at least way"),
            )
        ],
        *[
            pytest.param(
                _with_temperature,
                {"dataset": _one_class(separable_dataset()[0]), "num_episodes": n},
                "dataset must hold at least 2 classes, got 1",
                {"dataset": separable_dataset()[0]},
                id=f"one-class-num_episodes={n}",
            )
            for n in (0, 1)
        ],
    ],
)
def test_train_model_refuses_bad_input(build, changes, message, repair) -> None:
    """Each refusal names the argument, draws no random numbers and leaves the
    model as it was; the call its message asks for trains it."""
    torch.manual_seed(0)
    model, dataset = build()
    model.eval()
    with torch.no_grad():
        model.temperature.fill_(0.5)
    args = {
        "dataset": dataset,
        "num_episodes": 3,
        "calib_frac": 0.2,
        "support_size": 5,
        "way": 3,
        "episode_size": 30,
        "num_calibration_epochs": 1,
        "calibration_lr": 0.01,
        **changes,
    }
    before = model_state(model)
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match=re.escape(message)):
        model.train_model(**args)
    assert_model_unchanged(model, before)
    assert torch.equal(torch.get_rng_state(), rng), "the refusal drew random numbers"

    args.update(repair)
    model.train_model(**args)
    assert list(model.get_support()) == list(range(CLASSES))


_WRONG_NUMBER_TYPES = [
    ({"calib_frac": 0}, "int-calib_frac-0", _calib_frac_message(0)),
    ({"calib_frac": 1}, "int-calib_frac-1", _calib_frac_message(1)),
    ({"calibration_lr": 0}, "int-calibration_lr-0", _lr_message(0)),
    ({"calibration_lr": 1}, "int-calibration_lr-1", _lr_message(1)),
    ({"calibration_lr": True}, "bool-calibration_lr", _lr_message(True)),
    (
        {"num_calibration_epochs": 1.0},
        "float-num_calibration_epochs",
        _epochs_message("float"),
    ),
]


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        *[
            pytest.param(
                changes,
                BeartypeCallHintParamViolation,
                None,
                marks=pytest.mark.contracts,
                id=f"{name}-contracts-on",
            )
            for changes, name, _ in _WRONG_NUMBER_TYPES
        ],
        *[
            pytest.param(
                changes,
                ValueError,
                message,
                marks=pytest.mark.skipif(
                    __debug__, reason="beartype refuses the wrong type first"
                ),
                id=f"{name}-python-O",
            )
            for changes, name, message in _WRONG_NUMBER_TYPES
        ],
    ],
)
def test_train_model_refuses_a_wrong_number_type(changes, error, message) -> None:
    """An int ``calib_frac``, an int or bool ``calibration_lr``, or a float
    ``num_calibration_epochs``, is a type error while contracts are on and a
    ``ValueError`` under python -O; the model is left as it was, and the values the
    message asks for train it."""
    torch.manual_seed(0)
    model, dataset = _with_temperature()
    args = {
        "dataset": dataset,
        "num_episodes": 3,
        "calib_frac": 0.2,
        "support_size": 5,
        "way": 3,
        "episode_size": 30,
        "num_calibration_epochs": 1,
        "calibration_lr": 0.01,
    }
    before = model_state(model)
    rng = torch.get_rng_state()
    with pytest.raises(error, match=None if message is None else re.escape(message)):
        model.train_model(**{**args, **changes})
    assert_model_unchanged(model, before)
    assert torch.equal(torch.get_rng_state(), rng), "the refusal drew random numbers"

    model.train_model(**args)
    assert list(model.get_support()) == list(range(CLASSES))


@pytest.mark.parametrize(
    ("changes", "message", "repair"),
    [
        *[
            pytest.param(
                {"num_calibration_epochs": value},
                _epochs_message(shown),
                {"num_calibration_epochs": 1},
                id=f"num_calibration_epochs={value}",
            )
            for value, shown in ((-1, "int -1"), (True, "bool"))
        ],
        *[
            pytest.param(
                {"calibration_lr": value},
                _lr_message(value),
                {"calibration_lr": 0.01},
                id=f"calibration_lr={value}",
            )
            for value in (0.0, -0.01, math.nan, math.inf)
        ],
    ],
)
def test_calibrate_temperature_refuses_bad_input(changes, message, repair) -> None:
    """Each refusal names the argument and leaves the model as it was, the
    temperature and its ``requires_grad`` included; the call its message asks for
    calibrates the temperature."""
    model, x, y = quick_protonet()
    with torch.no_grad():
        model.temperature.fill_(0.5)
    args = {"num_calibration_epochs": 1, "calibration_lr": 0.01, **changes}
    before = model_state(model)
    with pytest.raises(ValueError, match=re.escape(message)):
        model.calibrate_temperature(x, y, **args)
    assert_model_unchanged(model, before)

    args.update(repair)
    model.calibrate_temperature(x, y, **args)
    assert model.temperature.item() != 0.5
    assert not model.temperature.requires_grad


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        *[
            pytest.param(
                changes,
                BeartypeCallHintParamViolation,
                None,
                marks=pytest.mark.contracts,
                id=f"{name}-contracts-on",
            )
            for changes, name, _ in _WRONG_NUMBER_TYPES[2:]
        ],
        *[
            pytest.param(
                changes,
                ValueError,
                message,
                marks=pytest.mark.skipif(
                    __debug__, reason="beartype refuses the wrong type first"
                ),
                id=f"{name}-python-O",
            )
            for changes, name, message in _WRONG_NUMBER_TYPES[2:]
        ],
    ],
)
def test_calibrate_temperature_refuses_a_wrong_number_type(
    changes, error, message
) -> None:
    """An int or bool ``calibration_lr`` or a float ``num_calibration_epochs`` is a
    type error while contracts are on and a ``ValueError`` under python -O; the model
    is left as it was, and the values the message asks for calibrate the
    temperature."""
    model, x, y = quick_protonet()
    with torch.no_grad():
        model.temperature.fill_(0.5)
    before = model_state(model)
    with pytest.raises(error, match=None if message is None else re.escape(message)):
        model.calibrate_temperature(x, y, **changes)
    assert_model_unchanged(model, before)

    model.calibrate_temperature(x, y, num_calibration_epochs=1, calibration_lr=0.01)
    assert model.temperature.item() != 0.5


def test_outlier_scores_refuse_equal_distances_and_keep_the_kdes() -> None:
    model, _, _ = quick_protonet()
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
    model, _, _ = quick_protonet()
    checkpoint = model._to_checkpoint()
    checkpoint["support"][0] = checkpoint["support"][0][:1]
    path = str(tmp_path / "one_example.eq")
    _model_file.write(path, checkpoint)
    with pytest.raises(ValueError, match="holds 1 example\\(s\\) of class 0"):
        eq.EquineProtonet.load(path)


def test_load_refuses_a_one_class_support(tmp_path) -> None:
    model, _, _ = quick_protonet()
    checkpoint = model._to_checkpoint()
    checkpoint["support"] = {0: checkpoint["support"][0]}
    checkpoint["outlier_kde"] = {0: checkpoint["outlier_kde"][0]}
    path = str(tmp_path / "one_class.eq")
    _model_file.write(path, checkpoint)
    for loader in (eq.EquineProtonet.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="support holds one class"):
            loader(path)


def test_save_refuses_an_untrained_model(tmp_path) -> None:
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    with pytest.raises(RuntimeError, match="no support examples to save"):
        model.save(str(tmp_path / "untrained.eq"))
    assert not (tmp_path / "untrained.eq").exists()


def test_equine_protonet_deepcopy_after_update_support() -> None:
    """A Protonet's public ``update_support`` keeps no autograd history."""
    model, x, y = quick_protonet()
    model.update_support(x, y, calib_frac=0.2)
    for name, t in stored_tensors(model).items():
        assert t.grad_fn is None, f"{name} still has a grad_fn"
    copy.deepcopy(model)


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
    reference, x, _ = quick_protonet(labels=(0, 1, 2))
    model, _, _ = quick_protonet(labels=labels)
    assert list(model.get_support()) == list(labels)
    expected, out = reference.predict(x), model.predict(x)
    assert torch.equal(out.classes, expected.classes)
    assert torch.equal(out.ood_scores, expected.ood_scores)


def test_ood_score_uses_the_kde_of_the_predicted_label() -> None:
    model, x, y = quick_protonet(labels=(0, 2, 5))
    rows = x[y == 5]
    out = model.predict(rows)
    assert out.classes.argmax(dim=1).tolist() == [2] * len(rows)
    expected = _ood_scores_from_kde(model, rows, 5)
    torch.testing.assert_close(out.ood_scores, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("labels", [(1, 2, 3), (0, 2, 5)])
def test_temperature_calibrates_with_non_contiguous_labels(labels) -> None:
    reference, _, _ = quick_protonet(labels=(0, 1, 2), use_temperature=True)
    model, _, _ = quick_protonet(labels=labels, use_temperature=True)
    assert torch.equal(model.temperature, reference.temperature)


def test_calibrate_temperature_rejects_a_label_outside_the_support() -> None:
    model, x, _ = quick_protonet(labels=(0, 2, 5))
    with pytest.raises(ValueError, match=r"Labels \[1.0\] are not among"):
        model.calibrate_temperature(x[:3], torch.tensor([0.0, 1.0, 2.0]))


def test_update_support_keeps_kdes_only_for_the_new_labels(tmp_path) -> None:
    model, x, y = quick_protonet(labels=(0, 1, 2))
    keep = y < 2
    model.update_support(x[keep], y[keep] + 7, calib_frac=0.2)
    assert list(model.outlier_score_kde) == [7, 8]
    path = str(tmp_path / "m.eq")
    model.save(path)
    assert list(_model_file.read(path)["outlier_kde"]) == [7, 8]


def test_update_support_scores_ood_with_the_new_labels_kdes() -> None:
    model, x, y = quick_protonet(labels=(0, 1, 2))
    keep = y < 2
    model.update_support(x[keep], y[keep] + 7, calib_frac=0.2)
    rows = x[y == 1]
    out = model.predict(rows)
    assert out.classes.argmax(dim=1).tolist() == [1] * len(rows)
    expected = _ood_scores_from_kde(model, rows, 8)
    torch.testing.assert_close(out.ood_scores, expected, atol=1e-6, rtol=0)


def test_generate_model_summary_maps_labels_through_the_support() -> None:
    """Column ``j`` of a Protonet's predictions is its ``j``-th support label."""
    model, _, _ = quick_protonet(labels=(0, 2, 5))
    true_y = torch.tensor([0, 0, 2, 2, 5, 5])
    probs = torch.nn.functional.one_hot(torch.tensor([0, 0, 1, 1, 2, 1]), 3).float()
    out = eq.EquineOutput(classes=probs, ood_scores=torch.zeros(6), embeddings=probs)
    summary = eq.generate_model_summary(model, out, true_y)
    assert summary["accuracy"] == pytest.approx(2.5 / 3)
    assert summary["confusionMatrix"] == [[2, 0, 0], [0, 2, 0], [0, 1, 1]]
    assert summary["brierScore"] == pytest.approx(2 / 6)


def test_generate_model_summary_of_labels_from_zero_matches_the_plain_metrics() -> None:
    model, x, y = quick_protonet(labels=(0, 1, 2))
    gen = torch.Generator().manual_seed(5)
    out = model.predict(x + 2.0 * torch.randn(x.shape, generator=gen))
    expected = eq.generate_model_metrics(out, y.long())
    for labels in (y, y.long()):
        summary = eq.generate_model_summary(model, out, labels)
        for key, value in expected.items():
            assert summary[key] == value, key


def test_a_reordered_support_maps_columns_in_the_summary_and_temperature() -> None:
    reference, x, _ = quick_protonet(labels=(0, 1, 2), use_temperature=True)
    model, _, y = quick_protonet(labels=(0, 1, 2), use_temperature=True)
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
    model, x, y = quick_protonet(labels=(0, 1, 2))
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
        pytest.param({"relative_mahal": False}, False, id="absolute"),
        pytest.param(
            {"relative_mahal": False, "ood_distance": "rmd_sqrt"},
            True,
            id="absolute-rmd_sqrt",
        ),
    ],
)
def test_temperature_reaches_the_ood_scores_except_with_rmd_sq(
    kwargs, scaled_ood
) -> None:
    """The temperature scales the class probabilities of every model, and the
    OOD scores only of the ``"rmd_sqrt"`` statistic (relative or not), which a
    model file without ``ood_distance`` loads with. The OOD scores are those of
    the distances the statistic reads."""
    model, x, _ = quick_protonet(use_temperature=True, **kwargs)
    model.temperature.fill_(0.5)
    halved = model.predict(x)
    with torch.no_grad():
        _, dists, embeddings = model.model._forward_with_embeddings(x)
        probs = torch.softmax(torch.negative(dists / 0.5), dim=1)
        read = dists / 0.5 if scaled_ood else dists
        expected = model._compute_outlier_scores(
            model._compute_ood_dist(embeddings, probs, read), probs
        )
    torch.testing.assert_close(halved.ood_scores, expected)
    model.temperature.fill_(1.0)
    plain = model.predict(x)
    assert not torch.equal(halved.classes, plain.classes)
    assert torch.equal(halved.ood_scores, plain.ood_scores) is not scaled_ood


def test_ood_distance_is_checked_at_construction() -> None:
    with pytest.raises(
        ValueError, match=r"ood_distance must be one of \['rmd_sq', 'rmd_sqrt'\]"
    ):
        eq.EquineProtonet(BasicEmbeddingModel(2, 2), 2, ood_distance="rmd")


@pytest.mark.contracts
def test_an_ood_distance_of_the_wrong_type_is_refused_by_beartype() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        eq.EquineProtonet(BasicEmbeddingModel(2, 2), 2, ood_distance=1)


def _outlier_scores_per_sample(model, ood_dists, predictions):
    """
    The OOD score of each example from ``integrate_box_1d`` of its predicted
    label's KDE at its distance in float64, one example at a time.
    """
    label_order = model._label_order()
    ood_scores = torch.zeros_like(ood_dists)
    for i in range(len(predictions)):
        kde = model.outlier_score_kde[label_order[int(torch.argmax(predictions[i]))]]
        p_value = kde.integrate_box_1d(ood_dists[i].double().cpu().numpy(), np.inf)
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
    model, _, _ = quick_protonet(labels=(0, 2, 5))
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
    model, _, _ = quick_protonet(labels=(0, 2, 5))
    kde = gaussian_kde(np.array([0.0, 1.0, 2.0, 10.0]), weights=[0.7, 0.1, 0.1, 0.1])
    model.outlier_score_kde = OrderedDict((label, kde) for label in (0, 2, 5))
    ood_dists = torch.tensor([-3.0, 1.0, 5.0, 12.0], dtype=torch.float64)
    predictions = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 1]), 3).double()

    ood_scores = model._compute_outlier_scores(ood_dists, predictions)

    expected = _outlier_scores_per_sample(model, ood_dists, predictions)
    torch.testing.assert_close(ood_scores, expected, atol=1e-12, rtol=0)


def test_outlier_scores_refuse_a_kde_with_more_than_one_dimension() -> None:
    model, _, _ = quick_protonet(labels=(0, 2, 5))
    model.outlier_score_kde[2] = gaussian_kde(
        np.random.default_rng(0).normal(size=(2, 9))
    )
    predictions = torch.nn.functional.one_hot(torch.tensor([1]), 3).double()
    with pytest.raises(ValueError, match="KDE of support label 2 has 2 dimensions"):
        model._compute_outlier_scores(
            torch.tensor([0.5], dtype=torch.float64), predictions
        )


def _direct_outlier_cdf(kde, dists: torch.Tensor) -> torch.Tensor:
    """The CDF of a one-dimensional ``kde`` at ``dists``, summed over its points in
    float64."""
    points = torch.from_numpy(kde.dataset[0]).double()
    weights = torch.from_numpy(kde.weights).double()
    z = (dists.double()[:, None] - points) / float(np.sqrt(kde.covariance[0, 0]))
    upper = 1.0 - torch.special.ndtr(z)
    if bool((weights == weights[0]).all()):
        return 1.0 - upper.mean(dim=1)
    return 1.0 - (upper @ weights).clamp(0.0, 1.0)


@functools.cache
def _model_with_labels_0_2_5():
    """A model trained on labels 0, 2 and 5 that has not predicted; tests that
    change it take a copy."""
    model, _, _ = quick_protonet(labels=(0, 2, 5))
    return model


def _first_label_predictions(rows: int) -> torch.Tensor:
    """Predictions of the first support label for ``rows`` examples."""
    return torch.nn.functional.one_hot(torch.zeros(rows, dtype=torch.long), 3).double()


def _build(points, weights, bandwidth):
    """``_OutlierGrid.build`` for as many rows as a grid can have nodes."""
    grid_class = eq.equine_protonet._OutlierGrid
    return grid_class.build(points, weights, bandwidth, grid_class.MAX_NODES)


def _heavy_tail(n: int, gen: np.random.Generator) -> np.ndarray:
    points = np.abs(gen.standard_normal(n))
    points[: max(1, n // 100)] *= 50.0
    return points


# ``n`` points of each kind of KDE, drawn from ``gen``.
KDE_POINTS = {
    "halfnormal": lambda n, gen: np.abs(gen.standard_normal(n)),
    "lognormal": lambda n, gen: gen.lognormal(0.0, 1.5, n),
    "heavy_tail": _heavy_tail,
    "two_clusters": lambda n, gen: np.where(
        gen.random(n) < 0.5, gen.normal(1.0, 0.1, n), gen.normal(10.0, 0.3, n)
    ),
    "ties": lambda n, gen: np.round(gen.lognormal(0.0, 1.0, n) * 3.0),
    "around_zero": lambda n, gen: 5.0 * gen.standard_normal(n),
    "far_from_zero": lambda n, gen: 1000.0 + gen.random(n),
}


def _assert_grid_matches_direct_sum(kde, extra=None):
    """
    Score ``kde`` with a model whose labels all have it, at its points, across
    their range and 12 bandwidths beyond it, at minus and plus infinity and at
    the float64 distances ``extra``. Assert that the scores come from a grid, are
    within 1e-13 of the CDF summed over the KDE's points and lie in [0, 1].
    Returns the grid.
    """
    model = copy.deepcopy(_model_with_labels_0_2_5())
    model.outlier_score_kde = OrderedDict((label, kde) for label in (0, 2, 5))
    points = torch.from_numpy(kde.dataset[0]).double()
    h = float(np.sqrt(kde.covariance[0, 0]))
    lo, hi = float(points.min()) - 12 * h, float(points.max()) + 12 * h
    parts = [
        points,
        torch.linspace(lo, hi, 4001, dtype=torch.float64),
        torch.tensor([-math.inf, math.inf], dtype=torch.float64),
    ]
    if extra is not None:
        parts.append(extra)
    dists = torch.cat(parts)

    scores = model._compute_outlier_scores(dists, _first_label_predictions(len(dists)))

    assert list(model._outlier_grids) == [0]
    expected = _direct_outlier_cdf(kde, dists)
    torch.testing.assert_close(scores, expected, atol=1e-13, rtol=0)
    assert bool(((scores >= 0.0) & (scores <= 1.0)).all())
    return model._outlier_grids[0][1]


@pytest.mark.parametrize("weighted", [False, True], ids=["uniform", "weighted"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("n", [10, 1000])
@pytest.mark.parametrize("kind", KDE_POINTS)
def test_outlier_grid_scores_match_the_direct_sum(kind, n, dtype, weighted) -> None:
    """
    Scores read from the grid match the direct sum (see
    ``_assert_grid_matches_direct_sum``). The ``far_from_zero`` KDE has points in
    [1000, 1001] and a bandwidth factor of 0.05.
    """
    gen = np.random.default_rng(n)
    weights = gen.dirichlet(np.ones(n)) if weighted else None
    kde = gaussian_kde(KDE_POINTS[kind](n, gen).astype(dtype), weights=weights)
    if kind == "far_from_zero":
        kde.set_bandwidth(0.05)
    _assert_grid_matches_direct_sum(kde)


def test_outlier_grid_built_in_several_chunks_matches_the_direct_sum() -> None:
    """A grid of a 5,000-point KDE has more nodes than one chunk of the build
    holds (2**22 (node, point) pairs) and matches the direct sum."""
    points = np.abs(np.random.default_rng(5000).standard_normal(5000))
    grid = _assert_grid_matches_direct_sum(gaussian_kde(points))
    assert len(grid.table) > 2**22 // 5000


def test_outlier_grid_with_a_dominant_cluster_matches_the_direct_sum() -> None:
    """A KDE of one point at 0 and 1,999 points at 1000 (one of them 1e-9 higher)
    gets a grid of more than 7,000 nodes that matches the direct sum, across
    [990, 1010] and at 1002.9813258381799."""
    kde = gaussian_kde(np.r_[0.0, 1000.0 + 1e-9, np.full(1998, 1000.0)])
    extra = torch.cat(
        [
            torch.linspace(990.0, 1010.0, 4001, dtype=torch.float64),
            torch.tensor([1002.9813258381799], dtype=torch.float64),
        ]
    )
    grid = _assert_grid_matches_direct_sum(kde, extra)
    assert len(grid.table) > 7000


def test_outlier_grid_of_unnormalized_equal_weights_matches_the_direct_sum() -> None:
    """A KDE whose weights are all 0.5 keeps its grid, which matches the direct
    sum."""
    kde = gaussian_kde(np.abs(np.random.default_rng(0).standard_normal(50)))
    kde.weights[:] = 0.5
    _assert_grid_matches_direct_sum(kde)


def test_outlier_grid_scores_the_top_point_when_its_reach_rounds_away() -> None:
    """A KDE of 3.0 and the next float64 above it, with a bandwidth below their
    spacing, gets a grid that scores both points as the direct sum does."""
    top = float(np.nextafter(3.0, np.inf))
    kde = gaussian_kde(np.array([3.0, top]), bw_method=0.017)
    _assert_grid_matches_direct_sum(kde)


def test_outlier_grid_takes_unequal_weights_within_1e_14_of_summing_to_1() -> None:
    """``build`` returns None for unequal weights whose sum is 2e-14 from 1 and a
    grid for unequal weights whose sum is 5e-15 from 1."""
    build = _build
    points = torch.tensor([0.0, 1.0], dtype=torch.float64)
    off = torch.tensor([0.5, 0.5 + 2e-14], dtype=torch.float64)
    near = torch.tensor([0.5, 0.5 + 5e-15], dtype=torch.float64)
    assert build(points, off, 0.3) is None
    assert build(points, near, 0.3) is not None


def test_outlier_grid_is_0_below_1_above_nan_at_nan_and_clamped() -> None:
    """Below its first node the grid gives 0, above its upper end 1, at NaN NaN,
    and between nodes its interpolant clamped to [0, 1]."""
    # Tables whose interpolant rises above 1 or falls below 0 between the nodes.
    above = torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    below = torch.tensor([[0.0, -0.5, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    dists = torch.tensor(
        [-math.inf, -1.0, 0.5, 2.0, math.inf, math.nan], dtype=torch.float64
    )
    rises = eq.equine_protonet._OutlierGrid(0.0, 1.0, above, 1.0)(dists)
    falls = eq.equine_protonet._OutlierGrid(0.0, 1.0, below, 1.0)(dists)
    assert rises[:5].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert falls[:5].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert math.isnan(rises[5]) and math.isnan(falls[5])


# Writes (attribute, index, value) that change a KDE in place into one that
# ``_OutlierGrid.build`` does not tabulate.
OFF_THE_GRID = {
    "nan_point": [("dataset", (0, 0), np.nan)],
    "infinite_point": [("dataset", (0, 0), np.inf)],
    "nan_weight": [("weights", 0, np.nan)],
    "infinite_weights": [("weights", slice(None), np.inf)],
    "negative_weight": [("weights", 0, -0.01), ("weights", 1, 0.05)],
    "weights_summing_below_1": [("weights", 0, 0.0)],
    "weights_summing_above_1": [("weights", 0, 0.5)],
    "tied_points_below_an_ulp": [
        ("dataset", (0, slice(None)), 3.0),
        ("covariance", (0, 0), 1e-34),
    ],
    "zero_bandwidth": [("covariance", (0, 0), 0.0)],
    "nan_bandwidth": [("covariance", (0, 0), np.nan)],
    "infinite_bandwidth": [("covariance", (0, 0), np.inf)],
}


@pytest.mark.parametrize("change", OFF_THE_GRID)
def test_outlier_scores_of_a_kde_off_the_grid_are_summed_directly(change) -> None:
    """A KDE that ``_OutlierGrid.build`` does not tabulate is scored by the
    direct sum, and the grid kept for its label is dropped."""
    model = copy.deepcopy(_model_with_labels_0_2_5())
    kde = gaussian_kde(np.abs(np.random.default_rng(0).standard_normal(50)))
    model.outlier_score_kde = OrderedDict((label, kde) for label in (0, 2, 5))
    dists = torch.linspace(-1.0, 4.0, 1001, dtype=torch.float64)
    predictions = _first_label_predictions(len(dists))
    model._compute_outlier_scores(dists, predictions)
    assert list(model._outlier_grids) == [0]
    for attribute, index, value in OFF_THE_GRID[change]:
        getattr(kde, attribute)[index] = value

    scores = model._compute_outlier_scores(dists, predictions)

    assert model._outlier_grids == {}
    expected = _direct_outlier_cdf(kde, dists)
    torch.testing.assert_close(scores, expected, atol=0, rtol=0, equal_nan=True)


def test_outlier_scores_of_a_kde_without_points_raise_zero_division() -> None:
    model = copy.deepcopy(_model_with_labels_0_2_5())
    kde = gaussian_kde(np.abs(np.random.default_rng(0).standard_normal(50)))
    kde.dataset = np.empty((1, 0))
    model.outlier_score_kde = OrderedDict((label, kde) for label in (0, 2, 5))
    with pytest.raises(ZeroDivisionError):
        model._compute_outlier_scores(
            torch.tensor([0.5], dtype=torch.float64), _first_label_predictions(1)
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cap", ["MAX_NODES", "MAX_PAIRS"])
def test_a_kde_over_a_cap_is_summed_directly(monkeypatch, cap, dtype) -> None:
    """A KDE whose grid would need more than ``MAX_NODES`` nodes or more than
    ``MAX_PAIRS`` (node, point) pairs is scored by the direct sum in float64; one
    that needs exactly the cap gets a grid. The KDE's points and the distances
    have ``dtype``."""
    gen = torch.Generator().manual_seed(0)
    kde = gaussian_kde(torch.randn(50, generator=gen).abs().to(dtype).numpy())
    points = torch.from_numpy(kde.dataset[0]).double()
    weights = torch.from_numpy(kde.weights).double()
    bandwidth = float(np.sqrt(kde.covariance[0, 0]))
    grid = _build(points, weights, bandwidth)
    assert grid is not None
    nodes = len(grid.table)
    limit = nodes if cap == "MAX_NODES" else nodes * len(points)
    dists = torch.linspace(-1.0, 4.0, 2001, dtype=dtype)
    predictions = _first_label_predictions(len(dists))
    model = copy.deepcopy(_model_with_labels_0_2_5())
    model.outlier_score_kde = OrderedDict((label, kde) for label in (0, 2, 5))

    monkeypatch.setattr(eq.equine_protonet._OutlierGrid, cap, limit - 1)
    direct = model._compute_outlier_scores(dists.double(), predictions)
    assert model._outlier_grids == {}
    assert torch.equal(direct, _direct_outlier_cdf(kde, dists))
    assert torch.equal(
        model._compute_outlier_scores(dists, predictions), direct.to(dtype)
    )

    monkeypatch.setattr(eq.equine_protonet._OutlierGrid, cap, limit)
    tabulated = model._compute_outlier_scores(dists.double(), predictions)
    assert len(model._outlier_grids[0][1].table) == nodes
    torch.testing.assert_close(tabulated, direct, atol=1e-13, rtol=0)


def test_outlier_grid_is_not_built_with_a_zero_node_spacing() -> None:
    """``build`` returns None when the spacing of the nodes rounds to 0."""
    grid = _build(
        torch.tensor([0.0, 2e-322], dtype=torch.float64),
        torch.full((2,), 0.5, dtype=torch.float64),
        5e-324,
    )
    assert grid is None


def test_outlier_grid_is_built_for_at_least_an_eighth_as_many_rows_as_nodes() -> None:
    """``build`` tabulates a KDE whose grid has 800 nodes for 100 rows and returns
    None for 99."""
    grid_class = eq.equine_protonet._OutlierGrid
    points = torch.tensor([0.0, 6.96875], dtype=torch.float64)
    weights = torch.full((2,), 0.5, dtype=torch.float64)
    grid = grid_class.build(points, weights, 1.0, 100)
    assert grid is not None
    assert len(grid.table) == 800
    assert grid_class.build(points, weights, 1.0, 99) is None


def _model_with_one_kde():
    """A copy of ``_model_with_labels_0_2_5()`` whose labels share one KDE of 50
    points, the KDE, and the fewest rows for which ``build`` tabulates it."""
    model = copy.deepcopy(_model_with_labels_0_2_5())
    kde = gaussian_kde(np.abs(np.random.default_rng(0).standard_normal(50)))
    model.outlier_score_kde = OrderedDict((label, kde) for label in (0, 2, 5))
    points = torch.from_numpy(kde.dataset[0]).double()
    weights = torch.from_numpy(kde.weights).double()
    nodes = len(_build(points, weights, float(np.sqrt(kde.covariance[0, 0]))).table)
    per_row = eq.equine_protonet._OutlierGrid.MAX_NODES_PER_ROW
    return model, kde, math.ceil(nodes / per_row)


def test_a_call_builds_the_grid_of_each_label_it_predicts_often_enough() -> None:
    """A call that predicts the first label for the fewest examples ``build``
    tabulates its KDE for, and the second label for one fewer, builds and reads
    the first label's grid and sums the second label's KDE directly."""
    model, kde, rows = _model_with_one_kde()
    dists = torch.linspace(-1.0, 4.0, 2 * rows - 1, dtype=torch.float64)
    columns = torch.arange(len(dists)) % 2
    first, second = columns == 0, columns == 1
    assert int(first.sum()) == rows
    assert int(second.sum()) == rows - 1
    predictions = torch.nn.functional.one_hot(columns, 3).double()

    scores = model._compute_outlier_scores(dists, predictions)

    assert list(model._outlier_grids) == [0]
    grid = model._outlier_grids[0][1]
    assert torch.equal(scores[first], grid(dists[first]))
    assert torch.equal(scores[second], _direct_outlier_cdf(kde, dists[second]))


def test_a_kept_grid_scores_a_single_example(monkeypatch) -> None:
    """After a call builds the first label's grid, a call that predicts the first
    label for one example reads that grid without summing the KDE or building
    another grid."""
    model, _, rows = _model_with_one_kde()
    dists = torch.linspace(-1.0, 4.0, rows, dtype=torch.float64)
    model._compute_outlier_scores(dists, _first_label_predictions(rows))
    grid = model._outlier_grids[0][1]

    def fail(*args):
        raise AssertionError("summed or built")

    monkeypatch.setattr(eq.equine_protonet, "_upper_tail", fail)
    monkeypatch.setattr(eq.equine_protonet._OutlierGrid, "build", staticmethod(fail))
    one = torch.tensor([1.234], dtype=torch.float64)

    score = model._compute_outlier_scores(one, _first_label_predictions(1))

    assert torch.equal(score, grid(one))
    assert model._outlier_grids[0][1] is grid


def test_a_call_with_too_few_examples_drops_a_stale_grid_and_sums_directly() -> None:
    """After the KDE of a label with a grid changes, a call that predicts the label
    for one example drops the grid and sums the KDE directly."""
    model, kde, rows = _model_with_one_kde()
    dists = torch.linspace(-1.0, 4.0, rows, dtype=torch.float64)
    model._compute_outlier_scores(dists, _first_label_predictions(rows))
    assert list(model._outlier_grids) == [0]
    kde.set_bandwidth(kde.factor * 2.0)
    one = torch.tensor([1.234], dtype=torch.float64)

    score = model._compute_outlier_scores(one, _first_label_predictions(1))

    assert model._outlier_grids == {}
    assert torch.equal(score, _direct_outlier_cdf(kde, one))


def test_predict_of_one_example_builds_no_grid() -> None:
    """A ``predict`` of one example on a model without grids keeps no grid."""
    model = copy.deepcopy(_model_with_labels_0_2_5())
    _, x, _ = separable_dataset()
    model.predict(x[:1])
    assert model._outlier_grids == {}


def _model_with_grids():
    """A copy of ``_model_with_labels_0_2_5()``, its training rows and labels, the
    rows repeated 4 times, and the OOD scores of its first ``predict`` of those,
    which builds a grid for each label."""
    model = copy.deepcopy(_model_with_labels_0_2_5())
    _, x, y = separable_dataset()
    y = torch.tensor([0.0, 2.0, 5.0])[y]
    queries = x.repeat(4, 1)
    scores = model.predict(queries).ood_scores
    assert sorted(model._outlier_grids) == [0, 2, 5]
    return model, x, y, queries, scores


def _scores_from_new_grids(model, x) -> torch.Tensor:
    """``predict``'s OOD scores of ``x`` with grids built from the model's KDEs as
    they are now; the grids the model keeps are left as they were."""
    kept = model._outlier_grids
    model._outlier_grids = {}
    try:
        return model.predict(x).ood_scores
    finally:
        model._outlier_grids = kept


def _change_kde_of_label_2(model, change: str) -> None:
    kde = model.outlier_score_kde[2]
    if change == "set_bandwidth":
        kde.set_bandwidth(kde.factor * 3.0)
    elif change == "dataset_in_place":
        kde.dataset[0] += 0.5
    elif change == "weights_in_place":
        kde.weights[:] = np.random.default_rng(0).dirichlet(np.ones(kde.n))
    elif change == "assign_entry":
        model.outlier_score_kde[2] = gaussian_kde(kde.dataset[0] * 2.0)
    else:
        model.outlier_score_kde = OrderedDict(
            (label, gaussian_kde(other.dataset[0] * 2.0))
            for label, other in model.outlier_score_kde.items()
        )


@pytest.mark.parametrize(
    "change",
    [
        "set_bandwidth",
        "dataset_in_place",
        "weights_in_place",
        "assign_entry",
        "assign_dict",
    ],
)
def test_predict_rebuilds_the_grid_of_a_changed_kde(change) -> None:
    model, _, _, queries, before = _model_with_grids()
    _change_kde_of_label_2(model, change)
    expected = _scores_from_new_grids(model, queries)
    assert not torch.equal(expected, before)
    assert torch.equal(model.predict(queries).ood_scores, expected)


def test_predict_keeps_the_grid_of_a_kde_replaced_by_an_equal_one() -> None:
    model, _, _, queries, before = _model_with_grids()
    grid = model._outlier_grids[2][1]
    model.outlier_score_kde[2] = copy.deepcopy(model.outlier_score_kde[2])
    assert torch.equal(model.predict(queries).ood_scores, before)
    assert model._outlier_grids[2][1] is grid


def test_update_support_and_train_model_drop_the_grids() -> None:
    model, x, y, queries, before = _model_with_grids()
    model.update_support(x, y, calib_frac=0.5)
    assert model._outlier_grids == {}
    expected = _scores_from_new_grids(model, queries)
    assert not torch.equal(expected, before)
    assert torch.equal(model.predict(queries).ood_scores, expected)
    assert sorted(model._outlier_grids) == [0, 2, 5]
    before = expected
    model.train_model(
        torch.utils.data.TensorDataset(x, y),
        num_episodes=2,
        calib_frac=0.3,
        support_size=10,
        way=3,
        episode_size=30,
    )
    assert model._outlier_grids == {}
    expected = _scores_from_new_grids(model, queries)
    assert not torch.equal(expected, before)
    assert torch.equal(model.predict(queries).ood_scores, expected)


def test_a_failed_update_support_keeps_the_grids(monkeypatch) -> None:
    """``update_support`` that fails after refitting the KDEs restores the KDEs and
    the grids built from them."""
    model, x, y, queries, before = _model_with_grids()
    grids = model._outlier_grids
    kept = dict(grids)
    fit = model._fit_outlier_scores

    def fit_then_fail(ood_dists, calib_y):
        fit(ood_dists, calib_y)
        raise ValueError("a later step failed")

    monkeypatch.setattr(model, "_fit_outlier_scores", fit_then_fail)
    with pytest.raises(ValueError, match="a later step failed"):
        model.update_support(x, y, calib_frac=0.5)
    assert model._outlier_grids is grids
    assert grids == kept
    assert torch.equal(model.predict(queries).ood_scores, before)
    assert all(grids[label][1] is kept[label][1] for label in kept)


def test_load_unpickling_and_deepcopy_start_without_grids(tmp_path) -> None:
    model, _, _, queries, before = _model_with_grids()
    assert "_outlier_grids" not in model.__getstate__()
    assert sorted(model._outlier_grids) == [0, 2, 5]
    path = str(tmp_path / "protonet.eq")
    model.save(path)
    for other in (
        eq.load_equine_model(path),
        pickle.loads(pickle.dumps(model)),
        copy.deepcopy(model),
    ):
        assert other._outlier_grids == {}
        assert torch.equal(other.predict(queries).ood_scores, before)
        assert sorted(other._outlier_grids) == [0, 2, 5]
    assert sorted(model._outlier_grids) == [0, 2, 5]


def test_two_threads_predicting_at_once_get_the_scores_of_one(monkeypatch) -> None:
    """Two threads that build the grids of a model at the same time both get the
    scores of a single-threaded ``predict``, and the model keeps a grid per
    label."""
    model = copy.deepcopy(_model_with_labels_0_2_5())
    queries = separable_dataset()[1].repeat(4, 1)
    expected = _scores_from_new_grids(model, queries)
    build = eq.equine_protonet._OutlierGrid.build
    together = threading.Barrier(2, timeout=30)

    def build_together(points, weights, bandwidth, rows):
        together.wait()
        return build(points, weights, bandwidth, rows)

    monkeypatch.setattr(
        eq.equine_protonet._OutlierGrid, "build", staticmethod(build_together)
    )
    results: list = [None, None]

    def predict(i: int) -> None:
        try:
            results[i] = model.predict(queries).ood_scores
        except Exception as err:
            results[i] = err

    threads = [threading.Thread(target=predict, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for result in results:
        assert isinstance(result, torch.Tensor), result
        assert torch.equal(result, expected)
    assert sorted(model._outlier_grids) == [0, 2, 5]
    monkeypatch.undo()
    assert torch.equal(model.predict(queries).ood_scores, expected)


@pytest.mark.parametrize("label_dtype", [torch.float32, torch.long])
@pytest.mark.parametrize("shift", [0, 7])
def test_update_support_draws_one_permutation_per_class(label_dtype, shift) -> None:
    """After the calibration split, the support and the RNG state after the call
    equal a replay of one ``randperm`` over each class's support rows in ascending
    label order."""
    support_y = torch.tensor([0] * 8 + [1] * 10 + [2] * 12) + shift
    order = torch.randperm(len(support_y), generator=torch.Generator().manual_seed(1))
    support_y = support_y[order].to(label_dtype)
    support_x = torch.rand(
        len(support_y), FEATURES, generator=torch.Generator().manual_seed(2)
    )
    support_x[:, 0] = torch.arange(len(support_y))
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    for seed in range(5):
        torch.manual_seed(seed)
        model.update_support(support_x, support_y, calib_frac=0.25)
        state = torch.get_rng_state()

        torch.manual_seed(seed)
        split_x, _, split_y, _ = eq.utils.stratified_train_test_split(
            support_x, support_y, test_size=0.25
        )
        expected = []
        for label in torch.unique(split_y).tolist():
            rows = torch.argwhere(split_y == label).reshape(-1)
            expected.append((label, split_x[rows[torch.randperm(len(rows))]]))
        support = model.get_support()
        assert list(support) == [label for label, _ in expected]
        for label, rows in expected:
            assert torch.equal(support[label], rows)
        assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("labels", LABEL_KINDS)
@pytest.mark.parametrize("num_episodes", [0, 1])
def test_train_model_refuses_labels_that_are_not_1d_whole_numbers(
    labels, num_episodes
) -> None:
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    y, message = labels_of_kind(y, labels)
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, use_temperature=True
    )
    model.eval()
    with torch.no_grad():
        model.temperature.fill_(0.5)
    before = model_state(model)
    with pytest.raises(ValueError, match=f"dataset labels {message}"):
        model.train_model(
            torch.utils.data.TensorDataset(x, y),
            num_episodes=num_episodes,
            support_size=10,
            way=2,
            episode_size=30,
        )
    assert_model_unchanged(model, before)


@pytest.mark.parametrize("trained", [False, True])
def test_update_support_refuses_one_class(trained) -> None:
    if trained:
        model, x, y = quick_protonet()
    else:
        torch.manual_seed(0)
        _, x, y = separable_dataset()
        model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    support = {k: v.clone() for k, v in model.model.support.items()}
    kde_labels = list(model.outlier_score_kde)
    training = model.training
    keep = y == 0
    with pytest.raises(
        ValueError, match="support_y must hold at least 2 classes, got 1"
    ):
        model.update_support(x[keep], y[keep].float(), calib_frac=0.2)
    assert list(model.model.support) == list(support)
    assert all(torch.equal(model.model.support[k], v) for k, v in support.items())
    assert list(model.outlier_score_kde) == kde_labels
    assert model.training == training


@pytest.mark.parametrize("labels", LABEL_KINDS)
@pytest.mark.parametrize("trained", [False, True])
def test_update_support_refuses_labels_that_are_not_1d_whole_numbers(
    labels, trained
) -> None:
    if trained:
        model, x, y = quick_protonet()
    else:
        torch.manual_seed(0)
        _, x, y = separable_dataset()
        model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    y, message = labels_of_kind(y, labels)
    support = {k: v.clone() for k, v in model.model.support.items()}
    kde_labels = list(model.outlier_score_kde)
    training = model.training
    with pytest.raises(ValueError, match=f"support_y {message}"):
        model.update_support(x, y, calib_frac=0.2)
    assert list(model.model.support) == list(support)
    assert all(torch.equal(model.model.support[k], v) for k, v in support.items())
    assert list(model.outlier_score_kde) == kde_labels
    assert model.training == training


def test_two_classes_train_and_update_support() -> None:
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    keep = y < 2
    two = torch.utils.data.TensorDataset(x[keep], y[keep].float())
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    model.train_model(two, num_episodes=1, support_size=10, way=2, episode_size=30)
    assert list(model.get_support()) == [0, 1]
    model.update_support(x[keep], y[keep].float(), calib_frac=0.2)
    assert list(model.get_support()) == [0, 1]


@pytest.mark.skipif(
    __debug__, reason="runs only under python -O, where beartype is off"
)
def test_cov_type_that_is_not_a_member_is_refused_with_its_conversion_under_python_O() -> (
    None
):
    dataset, _, _ = separable_dataset()
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, cov_type="full"
    )
    message = (
        "cov_type must be one of CovType.UNIT, CovType.DIAGONAL, CovType.FULL; "
        "got 'full'. CovType(value) converts one of 'unit', 'diag', 'full'."
    )
    with pytest.raises(ValueError) as err:
        model.train_model(
            dataset, num_episodes=2, support_size=10, way=3, episode_size=30
        )
    assert str(err.value) == message

    repaired = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, cov_type=eq.CovType("full")
    )
    repaired.train_model(
        dataset, num_episodes=2, support_size=10, way=3, episode_size=30
    )
    assert repaired.cov_type is eq.CovType.FULL
