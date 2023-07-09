# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import equine as eq
import torch
from hypothesis import given, settings

import pytest
from conftest import BasicEmbeddingModel, random_dataset


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=5)
def test_equine_protonet_instantiation(random_dataset) -> None:
    dataset, num_classes, _ = random_dataset
    X, Y = dataset.tensors
    embedding_model = BasicEmbeddingModel(X.shape[1], num_classes)

    model = eq.EquineProtonet(embedding_model, num_classes)
    model.train_model(torch.utils.data.TensorDataset(X, Y), num_episodes=0)


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
    with pytest.raises(TypeError):
        eq.Equine(em)  # type: ignore
    with pytest.raises(NotImplementedError):
        eq.Equine.predict(None, torch.Tensor([1, 2]))  # type: ignore
    with pytest.raises(NotImplementedError):
        eq.Equine.forward(None, torch.Tensor([1, 2]))  # type: ignore
    with pytest.raises(NotImplementedError):
        eq.Equine.train_model(None, torch.Tensor([1, 2]))  # type: ignore
    with pytest.raises(NotImplementedError):
        eq.Equine.save(None, "tmp")  # type: ignore
    with pytest.raises(NotImplementedError):
        eq.Equine.load("tmp")  # type: ignore
