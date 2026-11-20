# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
import copy
import functools

import pytest
import torch
from builders import quick_gp, quick_protonet
from conftest import BasicEmbeddingModel, stored_tensors
from golden_data import CLASSES, FEATURES

import equine as eq
from equine.equine import MIN_TEMPERATURE


def test_get_feature_names():
    num_features = 3
    num_classes = 5
    embed_model = BasicEmbeddingModel(num_features, num_classes)

    model = eq.EquineProtonet(embed_model, num_classes)
    assert eq.Equine.get_feature_names(model) is None

    del model.feature_names
    assert eq.Equine.get_feature_names(model) is None

    model = eq.EquineProtonet(embed_model, num_classes, feature_names=["123", "456"])
    assert eq.Equine.get_feature_names(model) == ["123", "456"]


def test_get_label_names():
    num_features = 3
    num_classes = 5
    embed_model = BasicEmbeddingModel(num_features, num_classes)

    model = eq.EquineProtonet(embed_model, num_classes)
    assert eq.Equine.get_label_names(model) is None

    del model.label_names
    assert eq.Equine.get_label_names(model) is None

    model = eq.EquineProtonet(embed_model, num_classes, label_names=["123", "456"])
    assert eq.Equine.get_label_names(model) == ["123", "456"]


def test_validate_feature_label_names():
    num_features = 3
    num_classes = 5
    embed_model = BasicEmbeddingModel(num_features, num_classes)

    # no feature or label names
    model = eq.EquineProtonet(embed_model, num_classes)
    eq.Equine.validate_feature_label_names(
        model, num_features=num_features, num_classes=num_classes
    )

    # with feature names, no label names
    model = eq.EquineProtonet(
        embed_model, num_classes, feature_names=["correct", "num", "features"]
    )
    eq.Equine.validate_feature_label_names(
        model, num_features=num_features, num_classes=num_classes
    )

    # no feature names, with label names
    model = eq.EquineProtonet(
        embed_model, num_classes, label_names=["correct", "num", "of", "label", "names"]
    )
    eq.Equine.validate_feature_label_names(
        model, num_features=num_features, num_classes=num_classes
    )

    # with feature and label names
    model = eq.EquineProtonet(
        embed_model,
        num_classes,
        feature_names=["correct", "num", "features"],
        label_names=["correct", "num", "of", "label", "names"],
    )
    eq.Equine.validate_feature_label_names(
        model, num_features=num_features, num_classes=num_classes
    )


def test_validate_feature_label_names_bad_feature_names():
    num_features = 3
    num_classes = 5
    embed_model = BasicEmbeddingModel(num_features, num_classes)

    model = eq.EquineProtonet(
        embed_model, num_classes, feature_names=["only one feature name"]
    )
    with pytest.raises(ValueError) as exc_info:
        eq.Equine.validate_feature_label_names(
            model, num_features=num_features, num_classes=num_classes
        )
    assert (
        str(exc_info.value)
        == "The length of feature_names (1) does not match the number of data features (3). Update feature_names or set feature_names to None."
    )

    model = eq.EquineProtonet(
        embed_model, num_classes, feature_names=["too", "many", "feature", "names"]
    )
    with pytest.raises(ValueError) as exc_info:
        eq.Equine.validate_feature_label_names(
            model, num_features=num_features, num_classes=num_classes
        )
    assert (
        str(exc_info.value)
        == "The length of feature_names (4) does not match the number of data features (3). Update feature_names or set feature_names to None."
    )


def test_validate_feature_label_names_bad_label_names():
    num_features = 3
    num_classes = 5
    embed_model = BasicEmbeddingModel(num_features, num_classes)

    model = eq.EquineProtonet(
        embed_model, num_classes, label_names=["not", "enough", "label", "names"]
    )
    with pytest.raises(ValueError) as exc_info:
        eq.Equine.validate_feature_label_names(
            model, num_features=num_features, num_classes=num_classes
        )
    assert (
        str(exc_info.value)
        == "The length of label_names (4) does not match the number of classes (5). Update label_names or set label_names to None."
    )

    model = eq.EquineProtonet(
        embed_model,
        num_classes,
        label_names=["too", "many", "label", "names", "to", "pass", "test"],
    )
    with pytest.raises(ValueError) as exc_info:
        eq.Equine.validate_feature_label_names(
            model, num_features=num_features, num_classes=num_classes
        )
    assert (
        str(exc_info.value)
        == "The length of label_names (7) does not match the number of classes (5). Update label_names or set label_names to None."
    )


BUILDERS = pytest.mark.parametrize(
    "build", [quick_protonet, quick_gp], ids=["protonet", "gp"]
)


@BUILDERS
def test_predict_refuses_an_empty_batch(build) -> None:
    model, _, _ = build()
    with pytest.raises(ValueError, match="X is empty"):
        model.predict(torch.empty(0, FEATURES))


@BUILDERS
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
)
def test_predict_refuses_non_finite_embeddings(build, value) -> None:
    model, _, _ = build()
    with pytest.raises(ValueError, match="embeddings of X hold NaN or infinite"):
        model.predict(torch.full((3, FEATURES), value))


class _ImputingEmbedding(torch.nn.Module):
    """A linear embedding that reads a NaN feature as 0."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(FEATURES, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.nan_to_num(x))


@BUILDERS
def test_predict_accepts_nan_input_that_the_embedding_handles(build) -> None:
    model, x, _ = build(_ImputingEmbedding)
    rows = x[:5].clone()
    rows[:, 0] = float("nan")
    expected = model.predict(torch.nan_to_num(rows))
    out = model.predict(rows)
    assert torch.equal(out.classes, expected.classes)
    assert torch.equal(out.ood_scores, expected.ood_scores)


def _assert_no_autograd_history(model) -> None:
    """Every tensor ``stored_tensors`` finds on ``model`` has no grad_fn."""
    for name, t in stored_tensors(model).items():
        assert t.grad_fn is None, f"{name} still has a grad_fn"


@BUILDERS
def test_a_trained_model_keeps_no_autograd_history(build) -> None:
    """A trained model keeps no autograd history; ``copy.deepcopy`` works."""
    model, _, _ = build()
    _assert_no_autograd_history(model)
    copy.deepcopy(model)


@pytest.mark.parametrize(
    "build",
    [quick_protonet, functools.partial(quick_gp, vis_support=True, support_size=10)],
    ids=["protonet", "gp"],
)
def test_a_model_loaded_from_a_file_keeps_no_autograd_history(build, tmp_path) -> None:
    model, _, _ = build()
    path = str(tmp_path / "model.eq")
    model.save(path)
    loaded = eq.load_equine_model(path)
    _assert_no_autograd_history(loaded)
    copy.deepcopy(loaded)


@pytest.mark.parametrize(
    "build, calibrate",
    [
        (quick_protonet, lambda m, x, y, **kw: m.calibrate_temperature(x, y, **kw)),
        (
            quick_gp,
            lambda m, x, y, **kw: m.calibrate_model(
                torch.utils.data.TensorDataset(x, y), **kw
            ),
        ),
    ],
    ids=["protonet", "gp"],
)
def test_calibration_keeps_the_temperature_at_or_above_the_floor(
    monkeypatch, build, calibrate
) -> None:
    model, x, y = build()
    used = []
    cross_entropy = torch.nn.functional.cross_entropy

    def record(*args, **kwargs):
        used.append(model.temperature.item())
        return cross_entropy(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", record)
    calibrate(model, x, y, num_calibration_epochs=200, calibration_lr=5.0)
    assert len(used) == 200
    assert min(used) >= MIN_TEMPERATURE
    assert model.temperature.item() >= MIN_TEMPERATURE
