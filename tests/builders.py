# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Small float32 models trained on ``separable_dataset``.

Each builder seeds torch with 0, then takes the data, then builds the embedding (from
a factory) and the model, then trains.
"""

import torch
from conftest import BasicEmbeddingModel
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq


def _embedding(make_embedding) -> torch.nn.Module:
    if make_embedding is None:
        return BasicEmbeddingModel(FEATURES, CLASSES)
    return make_embedding()


def quick_protonet(make_embedding=None, *, labels=None, **kwargs):
    """``(model, x, y)``: an ``EquineProtonet`` over ``make_embedding()`` (by default
    a ``BasicEmbeddingModel``) built with ``kwargs`` and trained for 5 episodes on
    ``separable_dataset()``, with class ``k`` labelled ``labels[k]`` when given.
    ``y`` holds the float labels it trained on."""
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    y = (y if labels is None else torch.tensor(labels)[y]).float()
    model = eq.EquineProtonet(_embedding(make_embedding), CLASSES, **kwargs)
    model.train_model(
        torch.utils.data.TensorDataset(x, y),
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    return model, x, y


def quick_gp(
    make_embedding=None, *, num_epochs=1, vis_support=False, support_size=25, **kwargs
):
    """``(model, x, y)``: an ``EquineGP`` with 16 random features over
    ``make_embedding()`` (by default a ``BasicEmbeddingModel``) built with ``kwargs``
    and trained with SGD (lr 0.05) in batches of 32 for ``num_epochs`` on
    ``separable_dataset()``. ``y`` holds the int64 class indices."""
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineGP(
        _embedding(make_embedding), CLASSES, CLASSES, num_random_features=16, **kwargs
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=num_epochs,
        batch_size=32,
        vis_support=vis_support,
        support_size=support_size,
    )
    return model, x, y
