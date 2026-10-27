# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Cross-version compatibility: files saved by an earlier code version keep
loading and keep producing the same predictions.

tests/fixtures/*_v2*.eq are fixed format-version-2, float32 files, built by
the ``fixture_*`` functions in golden_data.

expected.json records each file's SHA-256, asserted before loading. It
also records the persisted metadata (names, temperature, model type), which
is asserted after load.
"""

import hashlib
import json
import os

import pytest
import torch
from golden_data import fixture_query_batch

import equine as eq

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
with open(os.path.join(FIXTURES, "expected.json")) as f:
    EXPECTED = json.load(f)

# How each key of expected.json's "metadata" is read back from a loaded model.
METADATA_ACCESSORS = {
    "feature_names": lambda model: model.get_feature_names(),
    "label_names": lambda model: model.get_label_names(),
    "temperature": lambda model: float(model.temperature),
    "modelType": lambda model: model.train_summary["modelType"],
    "cov_type": lambda model: model.cov_type.value,
    "relative_mahal": lambda model: model.relative_mahal,
}

# expected.json key -> (fixture file, class). The *_nondefault fixtures carry
# every non-default persisted setting (names, temperature, and for the
# Protonet a full covariance and absolute Mahalanobis distance).
FIXTURE_FILES = {
    "protonet": ("protonet_v2.eq", eq.EquineProtonet),
    "gp": ("gp_v2.eq", eq.EquineGP),
    "protonet_nondefault": ("protonet_v2_nondefault.eq", eq.EquineProtonet),
    "gp_nondefault": ("gp_v2_nondefault.eq", eq.EquineGP),
}


@pytest.mark.parametrize("name", list(FIXTURE_FILES))
def test_v2_fixture_loads_and_predicts_the_same(name) -> None:
    filename, cls = FIXTURE_FILES[name]
    path = os.path.join(FIXTURES, filename)
    with open(path, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED[name]["sha256"]
    model = eq.load_equine_model(path)
    assert isinstance(model, cls)
    for key, expected in EXPECTED[name]["metadata"].items():
        assert key in METADATA_ACCESSORS, f"no accessor for metadata key {key!r}"
        if isinstance(expected, float):
            expected = pytest.approx(expected)
        assert METADATA_ACCESSORS[key](model) == expected, key
    out = model.predict(fixture_query_batch())
    # rtol=0: atol is the whole bound.
    assert torch.allclose(
        out.classes, torch.tensor(EXPECTED[name]["classes"]), atol=1e-5, rtol=0
    )
    assert torch.allclose(
        out.ood_scores, torch.tensor(EXPECTED[name]["ood_scores"]), atol=1e-5, rtol=0
    )
