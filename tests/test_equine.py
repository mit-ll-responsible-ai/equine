# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
import pytest
from conftest import BasicEmbeddingModel

import equine as eq


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
