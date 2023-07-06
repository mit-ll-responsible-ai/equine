# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import equine as eq
import torch
import os
from hypothesis import given, settings

from conftest import BasicEmbeddingModel, random_dataset


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=10)
def test_equine_gp_train_from_scratch(random_dataset) -> None:
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

    model.predict(X[1:10])  # Contracts should fire asserts on errors


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=10)
def test_equine_gp_train_from_scratch_with_temperature(random_dataset) -> None:
    dataset, num_classes, _ = random_dataset
    X, _ = dataset.tensors
    embedding_model = BasicEmbeddingModel(X.shape[1], num_classes)

    model = eq.EquineGP(embedding_model, num_classes, num_classes, use_temperature=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
    _, cal_loader = model.train_model(dataset, loss_fn, optimizer, 10)

    model.predict(X[1:10])  # Contracts should fire asserts on errors

    model.calibrate_temperature(cal_loader, 1, 0.01)


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=2)
def test_equine_gp_save_load(random_dataset) -> None:
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

    old_output = model.predict(X[1:10])
    tmp_filename = "tmp_eq_gp.pt"
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)
    model.save(tmp_filename)
    new_model = eq.EquineGP.load(tmp_filename)
    new_output = new_model.predict(X[1:10])
    assert (
        torch.nn.functional.mse_loss(old_output.classes, new_output.classes) <= 1e-7
    ), "Predictions changed on reload"
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)  # Cleanup


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=2)
def test_equine_gp_save_load_with_temperature(random_dataset) -> None:
    dataset, num_classes, _ = random_dataset
    X, _ = dataset.tensors
    embedding_model = BasicEmbeddingModel(X.shape[1], num_classes)

    model = eq.EquineGP(embedding_model, num_classes, num_classes, use_temperature=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
    model.train_model(dataset, loss_fn, optimizer, 10)

    old_output = model.predict(X[1:10])
    tmp_filename = "tmp_eq_gp.pt"
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)
    model.save(tmp_filename)
    new_model = eq.EquineGP.load(tmp_filename)
    new_output = new_model.predict(X[1:10])
    assert (
        torch.nn.functional.mse_loss(old_output.classes, new_output.classes) <= 1e-7
    ), "Predictions changed on reload"
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)  # Cleanup
