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
    "ood_distance": lambda model: model.ood_distance,
}

# expected.json key -> (fixture file, class). The *_nondefault fixtures carry
# every non-default setting a version-2 file persists (names, temperature, and
# for the Protonet a full covariance and absolute Mahalanobis distance). The
# Protonet fixtures hold no ood_distance setting and load as "rmd_sqrt".
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


class _ProtonetWithoutOodDistance(eq.EquineProtonet):
    """An ``EquineProtonet`` subclass whose ``__init__`` does not take
    ``ood_distance``."""

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        emb_out_dim: int,
        cov_type: eq.CovType = eq.CovType.UNIT,
        relative_mahal: bool = True,
        use_temperature: bool = False,
        init_temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            embedding_model,
            emb_out_dim,
            cov_type=cov_type,
            relative_mahal=relative_mahal,
            use_temperature=use_temperature,
            init_temperature=init_temperature,
            device=device,
        )


def test_v2_fixture_loads_into_a_subclass_without_ood_distance() -> None:
    """A subclass whose ``__init__`` does not take ``ood_distance`` loads a file
    without the setting, which scores with ``"rmd_sqrt"``."""
    path = os.path.join(FIXTURES, "protonet_v2.eq")
    model = _ProtonetWithoutOodDistance.load(path)
    assert model.ood_distance == "rmd_sqrt"
    out = model.predict(fixture_query_batch())
    assert torch.allclose(
        out.ood_scores,
        torch.tensor(EXPECTED["protonet"]["ood_scores"]),
        atol=1e-5,
        rtol=0,
    )


def test_subclass_without_ood_distance_refuses_a_file_with_the_setting(
    tmp_path,
) -> None:
    """A file that holds ``ood_distance`` needs a class whose ``__init__`` takes
    it, as for every saved setting."""
    model = _ProtonetWithoutOodDistance.load(os.path.join(FIXTURES, "protonet_v2.eq"))
    path = str(tmp_path / "saved.eq")
    model.save(path)
    assert torch.load(path, weights_only=True)["settings"]["ood_distance"] == "rmd_sqrt"
    with pytest.raises(ValueError, match="does not accept, e.g. 'ood_distance'"):
        _ProtonetWithoutOodDistance.load(path)


class _ProtonetRequiringOodDistance(eq.EquineProtonet):
    """An ``EquineProtonet`` subclass whose ``__init__`` requires ``ood_distance``
    and records the value it receives."""

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        emb_out_dim: int,
        ood_distance: str,
        **kwargs,
    ) -> None:
        super().__init__(
            embedding_model, emb_out_dim, ood_distance=ood_distance, **kwargs
        )
        self.received_ood_distance = ood_distance


@pytest.mark.parametrize("ood_distance", ["rmd_sq", "rmd_sqrt"])
def test_subclass_requiring_ood_distance_receives_the_files_value(
    tmp_path, ood_distance
) -> None:
    """A subclass whose ``__init__`` requires ``ood_distance`` receives the value a
    file holds."""
    model = eq.EquineProtonet.load(os.path.join(FIXTURES, "protonet_v2.eq"))
    model.ood_distance = ood_distance
    path = str(tmp_path / "saved.eq")
    model.save(path)
    loaded = _ProtonetRequiringOodDistance.load(path)
    assert loaded.received_ood_distance == ood_distance
    assert loaded.ood_distance == ood_distance


class _ProtonetSwallowingKwargs(eq.EquineProtonet):
    """An ``EquineProtonet`` subclass whose ``__init__`` takes ``**kwargs`` and
    does not pass them on."""

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        emb_out_dim: int,
        cov_type: eq.CovType = eq.CovType.UNIT,
        relative_mahal: bool = True,
        use_temperature: bool = False,
        init_temperature: float = 1.0,
        device: str = "cpu",
        **extra,
    ) -> None:
        super().__init__(
            embedding_model,
            emb_out_dim,
            cov_type=cov_type,
            relative_mahal=relative_mahal,
            use_temperature=use_temperature,
            init_temperature=init_temperature,
            device=device,
        )


def test_subclass_swallowing_kwargs_scores_with_the_files_statistic() -> None:
    """A subclass whose ``**kwargs`` drops ``ood_distance`` still scores a file
    without the setting with ``"rmd_sqrt"``."""
    path = os.path.join(FIXTURES, "protonet_v2.eq")
    model = _ProtonetSwallowingKwargs.load(path)
    assert model.ood_distance == "rmd_sqrt"
    queries = fixture_query_batch()
    assert torch.equal(
        model.predict(queries).ood_scores,
        eq.EquineProtonet.load(path).predict(queries).ood_scores,
    )


class _PassThroughProtonet(_ProtonetWithoutOodDistance):
    """A subclass that passes ``*args`` and ``**kwargs`` to a parent whose
    ``__init__`` does not take ``ood_distance``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


def test_pass_through_subclass_of_a_narrow_parent_loads_a_file_without_the_setting() -> (
    None
):
    """A ``*args, **kwargs`` subclass of a class whose ``__init__`` does not take
    ``ood_distance`` loads a file without the setting."""
    path = os.path.join(FIXTURES, "protonet_v2.eq")
    model = _PassThroughProtonet.load(path)
    assert model.ood_distance == "rmd_sqrt"
    queries = fixture_query_batch()
    assert torch.equal(
        model.predict(queries).ood_scores,
        eq.EquineProtonet.load(path).predict(queries).ood_scores,
    )
