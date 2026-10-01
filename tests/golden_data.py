# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Deterministic data and models for golden-value and cross-version tests.

Everything here is seeded so that the same code produces the same numbers on
CPU across runs, platforms and BLAS backends. The golden literals in
tests/test_golden.py are produced by these functions inside ``golden_dtype()``.
The fixture files in tests/fixtures/ are float32 and never regenerated:
the default ones were written by the float32 forerunners of these builders
when the fixtures were added, so the current (float64) builders do NOT
reproduce them; the ``*_nondefault`` ones by ``fixture_protonet_nondefault``
and ``fixture_gp_nondefault``.
tests/fixtures/expected.json pins their predictions and SHA-256.

The golden path (``trained_protonet``, ``trained_gp`` and the query batches,
all used inside ``golden_dtype()``) runs in float64 precisely for
cross-platform reproducibility: in float32 the same seeded training run
drifted between macOS arm64 and Linux x86_64 by up to 5.0e-4 (Protonet OOD)
and 5.0e-5 (GP classes) because BLAS summation order accumulates over the
training steps; in float64 the deviation is below 1e-15, so tests/test_golden.py
can hold atol=1e-6 and still catch subtle numeric changes.
"""

import contextlib

import torch
from conftest import BasicEmbeddingModel

import equine as eq

ROWS, FEATURES, CLASSES = 120, 6, 3
SEED = 0
# Query seed for the golden batch. Kept at SEED + 1 after checking seeds SEED + 1
# .. SEED + 5 in float64: it gives far-OOD minus in-distribution mean OOD margins
# of +0.38 (Protonet, in-dist [0.93, 0.18, 0.74] vs far 1.0) and +0.20 (GP).
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
    """Small Protonet built and trained in float64.

    With calib_frac=0.2 each class keeps 32 training rows, so support_size=10
    leaves 22 query rows per class and episode_size=30 fits within way=3 * 22.
    """
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
    """Small GP built and trained in float64, long enough to be confident in-distribution.

    Five epochs at lr=0.01 left every class probability near 1/3 and every
    OOD score near 1 (degenerate golden values); 40 epochs at lr=0.05 gives
    argmax-correct predictions and a clear in-distribution vs far-OOD gap.
    """
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


# Names for the non-default fixtures (tests/fixtures/*_v2_nondefault.eq).
FEATURE_NAMES = [f"feature_{i}" for i in range(FEATURES)]
LABEL_NAMES = ["alpha", "beta", "gamma"]


def fixture_protonet_nondefault(seed: int = SEED) -> eq.EquineProtonet:
    """How tests/fixtures/protonet_v2_nondefault.eq was produced (float32).

    Documents the fixture's provenance only: every non-default persisted
    setting (full covariance, absolute Mahalanobis, temperature scaling,
    feature and label names) on top of ``trained_protonet``'s training. Runs
    in the default dtype, NOT inside ``golden_dtype()``, like the fixture.
    Must not be re-run to regenerate the file; expected.json pins its SHA-256.
    Deliberately shares no code with ``trained_protonet``: this is a frozen
    record, and retuning the golden builder must not silently change it.
    """
    torch.manual_seed(seed)
    dataset, _, _ = separable_dataset(seed)
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        cov_type=eq.CovType.FULL,
        relative_mahal=False,
        use_temperature=True,
        feature_names=FEATURE_NAMES,
        label_names=LABEL_NAMES,
    )
    model.train_model(
        dataset,
        num_episodes=20,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    return model


def fixture_gp_nondefault(seed: int = SEED) -> eq.EquineGP:
    """How tests/fixtures/gp_v2_nondefault.eq was produced (float32).

    Documents the fixture's provenance only: ``trained_gp``'s training with
    feature and label names, then ``calibrate_model`` so the persisted
    temperature differs from 1.0 (``EquineGP.__init__`` has no use_temperature
    since 0.1.6). Runs in the default dtype, NOT inside ``golden_dtype()``,
    like the fixture. Must not be re-run to regenerate the file; expected.json
    pins its SHA-256. Deliberately shares no code with ``trained_gp``: this is
    a frozen record, and retuning the golden builder must not silently change it.
    """
    torch.manual_seed(seed)
    dataset, _, _ = separable_dataset(seed)
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        CLASSES,
        num_random_features=16,
        feature_names=FEATURE_NAMES,
        label_names=LABEL_NAMES,
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
    model.calibrate_model(dataset, num_calibration_epochs=5)
    return model
