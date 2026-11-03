# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Deterministic data and models for golden-value and cross-version tests.

Everything here is seeded. The golden helpers build, train and query in float64 inside
``golden_dtype()``. tests/test_golden.py pins their outputs.

The ``fixture_*`` functions build the float32 models stored in tests/fixtures/
from their own data (``fixture_dataset``) and share no code with the golden
helpers. tests/test_fixtures.py checks the stored files against
tests/fixtures/expected.json on ``fixture_query_batch()``.
"""

import contextlib

import torch
from conftest import BasicEmbeddingModel

import equine as eq

ROWS, FEATURES, CLASSES = 120, 6, 3
SEED = 0
# Query seed for the golden batch.
QUERY_SEED = SEED + 1


@contextlib.contextmanager
def golden_dtype():
    """Run the golden path in float64 (see the module docstring). Re-entrant."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def separable_dataset(seed: int = SEED):
    """Uniform noise plus a class-dependent shift of 3 on the first `CLASSES` features.

    Features take the default dtype (float32 normally, float64 inside
    ``golden_dtype()``). Labels are float, which is what
    ``EquineProtonet.train_model`` expects; ``EquineGP.train_model`` casts them
    to long itself.
    """
    gen = torch.Generator().manual_seed(seed)
    y = torch.arange(CLASSES).repeat_interleave(ROWS // CLASSES)
    x = torch.rand(ROWS, FEATURES, generator=gen)
    x[:, :CLASSES] += 3 * torch.nn.functional.one_hot(y, CLASSES).float()
    return torch.utils.data.TensorDataset(x, y.float()), x, y


def query_batch(seed: int = QUERY_SEED) -> torch.Tensor:
    """Fixed in-distribution queries: one per class, same construction as the training data.

    Takes the default dtype, so call it inside ``golden_dtype()`` when querying
    a golden (float64) model.
    """
    gen = torch.Generator().manual_seed(seed)
    y = torch.arange(CLASSES)
    x = torch.rand(CLASSES, FEATURES, generator=gen)
    x[:, :CLASSES] += 3 * torch.nn.functional.one_hot(y, CLASSES).float()
    return x


def far_ood_batch() -> torch.Tensor:
    """Queries shifted by +6 on every feature: far from every class."""
    return query_batch() + 6.0


def trained_protonet(seed: int = SEED) -> eq.EquineProtonet:
    """Small Protonet built and trained in float64."""
    with golden_dtype():
        torch.manual_seed(seed)
        dataset, _, _ = separable_dataset(seed)
        model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
        model.train_model(
            dataset,
            num_episodes=20,
            calib_frac=0.2,
            support_size=10,
            way=3,
            episode_size=30,
        )
    return model


def trained_gp(seed: int = SEED) -> eq.EquineGP:
    """Small GP built and trained in float64."""
    with golden_dtype():
        torch.manual_seed(seed)
        dataset, _, _ = separable_dataset(seed)
        model = eq.EquineGP(
            BasicEmbeddingModel(FEATURES, CLASSES),
            CLASSES,
            CLASSES,
            num_random_features=16,
        )
        model.train_model(
            dataset,
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.05),
            num_epochs=40,
            batch_size=32,
            vis_support=True,
            support_size=10,
        )
    return model


# Names persisted by the *_nondefault fixtures.
FEATURE_NAMES = [f"feature_{i}" for i in range(6)]
LABEL_NAMES = ["alpha", "beta", "gamma"]


def fixture_dataset() -> torch.utils.data.TensorDataset:
    """Float32 training data of every fixture model: 120 rows, 6 features, 3 classes."""
    gen = torch.Generator().manual_seed(0)
    y = torch.arange(3).repeat_interleave(40)
    x = torch.rand(120, 6, generator=gen, dtype=torch.float32)
    x[:, :3] += 3 * torch.nn.functional.one_hot(y, 3).float()
    return torch.utils.data.TensorDataset(x, y.float())


def fixture_query_batch() -> torch.Tensor:
    """The float32 queries whose predictions tests/fixtures/expected.json records."""
    gen = torch.Generator().manual_seed(1)
    y = torch.arange(3)
    x = torch.rand(3, 6, generator=gen, dtype=torch.float32)
    x[:, :3] += 3 * torch.nn.functional.one_hot(y, 3).float()
    return x


def fixture_protonet() -> eq.EquineProtonet:
    """Builds the float32 model stored in tests/fixtures/protonet_v2.eq."""
    torch.manual_seed(0)
    model = eq.EquineProtonet(BasicEmbeddingModel(6, 3), 3)
    model.train_model(
        fixture_dataset(),
        num_episodes=20,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    return model


def fixture_gp() -> eq.EquineGP:
    """Builds the float32 model stored in tests/fixtures/gp_v2.eq."""
    torch.manual_seed(0)
    model = eq.EquineGP(BasicEmbeddingModel(6, 3), 3, 3, num_random_features=16)
    model.train_model(
        fixture_dataset(),
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=40,
        batch_size=32,
        vis_support=True,
        support_size=10,
    )
    return model


def fixture_protonet_nondefault() -> eq.EquineProtonet:
    """Builds the float32 model stored in tests/fixtures/protonet_v2_nondefault.eq.

    ``fixture_protonet`` with every non-default persisted setting: full
    covariance, absolute Mahalanobis distance, temperature scaling, feature and
    label names.
    """
    torch.manual_seed(0)
    model = eq.EquineProtonet(
        BasicEmbeddingModel(6, 3),
        3,
        cov_type=eq.CovType.FULL,
        relative_mahal=False,
        use_temperature=True,
        feature_names=FEATURE_NAMES,
        label_names=LABEL_NAMES,
    )
    model.train_model(
        fixture_dataset(),
        num_episodes=20,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    return model


def fixture_gp_nondefault() -> eq.EquineGP:
    """Builds the float32 model stored in tests/fixtures/gp_v2_nondefault.eq.

    ``fixture_gp`` with feature and label names, then ``calibrate_model``; the persisted
    temperature differs from 1.0.
    """
    torch.manual_seed(0)
    model = eq.EquineGP(
        BasicEmbeddingModel(6, 3),
        3,
        3,
        num_random_features=16,
        feature_names=FEATURE_NAMES,
        label_names=LABEL_NAMES,
    )
    dataset = fixture_dataset()
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=40,
        batch_size=32,
        vis_support=True,
        support_size=10,
    )
    model.calibrate_model(dataset, num_calibration_epochs=5)
    return model
