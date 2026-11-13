# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Cross-version compatibility: files saved by an earlier code version load, or are
upgraded with ``equine.upgrade``, and keep producing the same predictions.

tests/fixtures/*_v4*.eq are fixed format-4, float32 files holding the models that
the ``fixture_*`` functions in golden_data build. tests/fixtures/*_0_1_8.eq are files
saved by EQUINE 0.1.8's own save(), with the script in golden_data's
``FIXTURES_0_1_8_SCRIPT``; their test upgrades a copy.

expected.json records each file's SHA-256, asserted before the file is read, and
its predictions on ``fixture_query_batch()``. For the format-4 files it also
records the persisted metadata (names, temperature, model type), which is asserted
after load.
"""

import hashlib
import json
import os
import shutil

import pytest
import torch
from golden_data import fixture_query_batch

import equine as eq
from equine import _model_file
from equine.upgrade import upgrade_model_file

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
# non-default settings (names, temperature, and for the Protonet a full covariance
# and absolute Mahalanobis distance). The Protonet fixtures hold no ood_distance
# setting and load as "rmd_sqrt".
FIXTURE_FILES = {
    "protonet": ("protonet_v4.eq", eq.EquineProtonet),
    "gp": ("gp_v4.eq", eq.EquineGP),
    "protonet_nondefault": ("protonet_v4_nondefault.eq", eq.EquineProtonet),
    "gp_nondefault": ("gp_v4_nondefault.eq", eq.EquineGP),
}

# expected.json key -> (file saved by EQUINE 0.1.8, class). Each file's TorchScript
# embedding has the layers of equine.MLP(6, [16], 3).
FIXTURES_0_1_8 = {
    "protonet_0_1_8": ("protonet_0_1_8.eq", eq.EquineProtonet),
    "gp_0_1_8": ("gp_0_1_8.eq", eq.EquineGP),
}


def _fixture(filename: str) -> str:
    """The path of the committed fixture ``filename``."""
    return os.path.join(FIXTURES, filename)


@pytest.mark.parametrize("name", list(FIXTURE_FILES))
def test_v4_fixture_loads_and_predicts_the_same(name) -> None:
    filename, cls = FIXTURE_FILES[name]
    with open(_fixture(filename), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED[name]["sha256"]
    model = eq.load_equine_model(_fixture(filename))
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


@pytest.mark.parametrize("name", list(FIXTURES_0_1_8))
def test_0_1_8_fixture_upgrades_and_predicts_the_same(name, tmp_path) -> None:
    """A file saved by EQUINE 0.1.8, upgraded with the architecture of its embedding,
    predicts what the 0.1.8 model predicted."""
    filename, cls = FIXTURES_0_1_8[name]
    with open(_fixture(filename), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED[name]["sha256"]
    path = str(tmp_path / filename)
    shutil.copy(_fixture(filename), path)
    with pytest.warns(FutureWarning, match="deprecated"):
        assert upgrade_model_file(path, eq.MLP(6, [16], 3)) == path
    model = eq.load_equine_model(path)
    assert isinstance(model, cls)
    out = model.predict(fixture_query_batch())
    # rtol=0: atol is the whole bound.
    assert torch.allclose(
        out.classes, torch.tensor(EXPECTED[name]["classes"]), atol=1e-5, rtol=0
    )
    assert torch.allclose(
        out.ood_scores, torch.tensor(EXPECTED[name]["ood_scores"]), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("name", list(FIXTURES_0_1_8))
def test_0_1_8_fixture_is_refused_with_legacy_format_error(name) -> None:
    """The model loaders refuse a file saved by EQUINE 0.1.8 with
    ``equine.LegacyFormatError``, naming the upgrade."""
    filename, cls = FIXTURES_0_1_8[name]
    for load in (eq.load_equine_model, cls.load):
        with pytest.raises(eq.LegacyFormatError, match="python -m equine.upgrade"):
            load(_fixture(filename))


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


def test_subclass_without_ood_distance_refuses_a_file_with_the_setting(
    tmp_path,
) -> None:
    """A file that holds ``ood_distance`` needs a class whose ``__init__`` takes
    it, as for every saved setting."""
    model = _ProtonetWithoutOodDistance.load(_fixture("protonet_v4.eq"))
    path = str(tmp_path / "saved.eq")
    model.save(path)
    assert _model_file.read(path)["settings"]["ood_distance"] == "rmd_sqrt"
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
    model = eq.EquineProtonet.load(_fixture("protonet_v4.eq"))
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


class _PassThroughProtonet(_ProtonetWithoutOodDistance):
    """A subclass that passes ``*args`` and ``**kwargs`` to a parent whose
    ``__init__`` does not take ``ood_distance``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


@pytest.mark.parametrize(
    "cls",
    [_ProtonetWithoutOodDistance, _ProtonetSwallowingKwargs, _PassThroughProtonet],
    ids=["without-ood-distance", "swallowing-kwargs", "pass-through"],
)
def test_subclass_loads_a_file_without_ood_distance(cls) -> None:
    """A subclass whose ``__init__`` does not take ``ood_distance``, whose
    ``**kwargs`` drops it, or that passes ``*args, **kwargs`` to a parent that does
    not take it loads a file without the setting and scores with ``"rmd_sqrt"``."""
    path = _fixture("protonet_v4.eq")
    model = cls.load(path)
    assert model.ood_distance == "rmd_sqrt"
    queries = fixture_query_batch()
    assert torch.equal(
        model.predict(queries).ood_scores,
        eq.EquineProtonet.load(path).predict(queries).ood_scores,
    )
