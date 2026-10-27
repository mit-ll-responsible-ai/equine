# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytest
import torch
from conftest import BasicEmbeddingModel, assert_valid_prediction, random_dataset
from hypothesis import given, settings

import equine as eq


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=5)
def test_equine_protonet_instantiation(random_dataset) -> None:
    dataset, num_classes, train_kwargs = random_dataset
    X, Y = dataset.tensors
    embedding_model = BasicEmbeddingModel(X.shape[1], num_classes)

    model = eq.EquineProtonet(embedding_model, num_classes)
    model.train_model(
        torch.utils.data.TensorDataset(X, Y),  # type: ignore
        num_episodes=0,
        **train_kwargs,
    )


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=5)
def test_equine_gp_instantiation(random_dataset) -> None:
    dataset, num_classes, _ = random_dataset
    X, _ = dataset.tensors
    embedding_model = BasicEmbeddingModel(X.shape[1], num_classes)

    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
    model.train_model(dataset, loss_fn, optimizer, 10)


def test_unimplemented_errors():
    em = BasicEmbeddingModel(2, 4)
    t = torch.Tensor([1, 2])
    td = torch.utils.data.TensorDataset(t, t)  # type: ignore
    with pytest.raises(TypeError):
        eq.Equine(em)  # type: ignore
    with pytest.raises(NotImplementedError):
        eq.Equine.predict(None, torch.Tensor([1, 2]))  # type: ignore
    with pytest.raises(NotImplementedError):
        eq.Equine.forward(None, torch.Tensor([1, 2]))  # type: ignore
    with pytest.raises(NotImplementedError):
        eq.Equine.train_model(None, td)  # type: ignore
    with pytest.raises(NotImplementedError):
        eq.Equine.save(None, "tmp")  # type: ignore
    with pytest.raises(NotImplementedError):
        eq.Equine.load("tmp")  # type: ignore


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=5)
def test_model_summary(random_dataset) -> None:
    dataset, num_classes, _ = random_dataset
    X, Y = dataset.tensors
    embedding_model = BasicEmbeddingModel(X.shape[1], num_classes)

    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
    model.train_model(dataset, loss_fn, optimizer, 2)
    eq_out = model.predict(X)
    assert_valid_prediction(eq_out, len(X), num_classes)

    eq.utils.generate_model_summary(model, eq_out, Y)


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=20)
def test_random_dataset_shape_and_training_args(random_dataset) -> None:
    dataset, num_classes, train_kwargs = random_dataset
    X, Y = dataset.tensors
    rows = X.shape[0]
    assert 120 <= rows <= 200
    assert 2 <= num_classes <= 5
    labels, counts = torch.unique(Y, return_counts=True)
    assert len(labels) == num_classes
    assert counts.min() >= 30

    # The real property: the derived arguments survive the split and episode
    # generation that train_model performs on this data.
    train_x, _, train_y, _ = eq.utils.stratified_train_test_split(
        X, Y, train_kwargs["calib_frac"]
    )
    support, _, episode_y = eq.utils.generate_episode(
        train_x,
        train_y,
        train_kwargs["support_size"],
        train_kwargs["way"],
        train_kwargs["episode_size"],
    )
    assert len(support) == train_kwargs["way"]
    assert episode_y.unique().numel() == train_kwargs["way"]
    full = eq.utils.generate_support(
        train_x, train_y, train_kwargs["support_size"], torch.unique(train_y).tolist()
    )
    assert len(full) == num_classes
    assert all(len(s) == train_kwargs["support_size"] for s in full.values())
