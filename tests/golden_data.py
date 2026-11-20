# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Deterministic data and models for golden-value and cross-version tests.

Everything here is seeded. The golden helpers build, train and query in float64 inside
``golden_dtype()``. tests/test_golden.py pins their outputs.

The ``fixture_*`` functions are the recipes (architecture, data, training) of the
float32 models stored in tests/fixtures/. They use their own data (``fixture_dataset``)
and share no code with the golden helpers. ``FIXTURES_0_1_8_SCRIPT`` holds the script
that wrote the fixtures saved by EQUINE 0.1.8. tests/test_fixtures.py checks the
stored files as stored, against tests/fixtures/expected.json on
``fixture_query_batch()``.
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
    """A float32 ``EquineProtonet`` over ``BasicEmbeddingModel(6, 3)`` with
    ``ood_distance="rmd_sqrt"``, trained with seed 0 for 20 episodes on
    ``fixture_dataset()``: the recipe of tests/fixtures/protonet_v4.eq. The stored
    file is checked as stored, against expected.json."""
    torch.manual_seed(0)
    model = eq.EquineProtonet(BasicEmbeddingModel(6, 3), 3, ood_distance="rmd_sqrt")
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
    """The recipe of the float32 model in tests/fixtures/gp_v4.eq: an ``EquineGP``
    with 16 random features over ``BasicEmbeddingModel(6, 3)``, trained with SGD (lr
    0.05) for 40 epochs in batches of 32 on ``fixture_dataset()``, with a support of
    10 examples per class. The stored file is checked as stored, against
    tests/fixtures/expected.json."""
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
    """The recipe of tests/fixtures/protonet_v4_nondefault.eq, which is checked as
    stored, against expected.json.

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
        ood_distance="rmd_sqrt",
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
    """The recipe of the float32 model in tests/fixtures/gp_v4_nondefault.eq.

    ``fixture_gp`` with feature and label names, then ``calibrate_model``; the persisted
    temperature differs from 1.0. The stored file is checked as stored, against
    tests/fixtures/expected.json.
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


# The script that wrote tests/fixtures/protonet_0_1_8.eq and gp_0_1_8.eq with EQUINE
# 0.1.8's own save(), and printed their expected.json entries. It runs with EQUINE
# v0.1.8's src/ on PYTHONPATH and imports ``fixture_dataset`` and
# ``fixture_query_batch`` from this module.
FIXTURES_0_1_8_SCRIPT = r'''"""
Write tests/fixtures/protonet_0_1_8.eq and gp_0_1_8.eq with EQUINE 0.1.8's own save(),
and print their expected.json entries: each file's SHA-256, and the classes and OOD
scores the saved model predicts for golden_data's fixture_query_batch().

Run with EQUINE v0.1.8's src/ on PYTHONPATH:
    python build_fixtures_018.py TESTS_DIR OUT_DIR
"""

import hashlib
import json
import os
import sys
import types
import warnings

import torch

import equine as eq

tests_dir, out_dir = sys.argv[1], sys.argv[2]
print("equine imported from", eq.__file__, file=sys.stderr)
sys.path.insert(0, tests_dir)
# Stands in for the tests' conftest, from which golden_data imports BasicEmbeddingModel.
sys.modules["conftest"] = types.SimpleNamespace(BasicEmbeddingModel=None)
from golden_data import fixture_dataset, fixture_query_batch  # noqa: E402


class Net(torch.nn.Module):
    """The layers and state-dict keys of equine.MLP(6, [16], 3)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


warnings.simplefilter("ignore")
dataset = fixture_dataset()

torch.manual_seed(0)
protonet = eq.EquineProtonet(Net(), 3)
protonet.train_model(
    dataset, num_episodes=20, calib_frac=0.2, support_size=10, way=3, episode_size=30
)

torch.manual_seed(0)
gp = eq.EquineGP(Net(), 3, 3, num_random_features=16)
gp.train_model(
    dataset,
    torch.nn.CrossEntropyLoss(),
    torch.optim.SGD(gp.parameters(), lr=0.05),
    num_epochs=40,
    batch_size=32,
    vis_support=True,
    support_size=10,
)

entries = {}
for name, model in (("protonet_0_1_8", protonet), ("gp_0_1_8", gp)):
    path = os.path.join(out_dir, f"{name}.eq")
    model.save(path)
    out = model.predict(fixture_query_batch())
    with open(path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    entries[name] = {
        "classes": out.classes.detach().tolist(),
        "ood_scores": out.ood_scores.detach().tolist(),
        "sha256": sha256,
    }
print(json.dumps(entries, indent=1))
'''
