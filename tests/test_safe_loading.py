# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
The file-reading layer (``equine.utils.load_checkpoint``) and the safe format:
model files are read without unrestricted unpickling, so opening an untrusted
file cannot run pickle payloads, and a crafted file cannot make the reader
allocate more than the file holds. Legacy files are refused and rewritten by
``python -m equine.upgrade``. The model layer on top (recipes, save/load) is
tested in test_persistence.py.
"""

import io
import os
import re
import shutil
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import BasicEmbeddingModel, _rewrite_zip, assert_on_device

import equine as eq
import equine.utils
from equine.upgrade import main, upgrade_model_file
from equine.utils import (
    EQUINE_FORMAT_VERSION,
    LegacyFormatError,
    load_checkpoint,
    prepare_jit_module,
)


class _MakesDirectoryWhenUnpickled:
    """Stand-in for a malicious pickle payload: unpickling it creates a directory."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (os.mkdir, (self.marker,))


def _tiny_dataset(seed: int = 0):
    torch.manual_seed(seed)
    X = torch.rand(120, 6)
    Y = torch.tensor([0] * 40 + [1] * 40 + [2] * 40)
    return torch.utils.data.TensorDataset(X, Y), X


def _trained_protonet(cov_type=eq.CovType.UNIT, use_temperature=False):
    dataset, X = _tiny_dataset()
    model = eq.EquineProtonet(
        BasicEmbeddingModel(6, 3), 3, cov_type=cov_type, use_temperature=use_temperature
    )
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    return model, X


def _trained_gp():
    dataset, X = _tiny_dataset()
    model = eq.EquineGP(BasicEmbeddingModel(6, 3), 3, 3, num_random_features=16)
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.001),
        num_epochs=1,
        batch_size=32,
        vis_support=True,
    )
    return model, X


def _assert_same_predictions(a, b, X):
    out_a, out_b = a.predict(X[1:10]), b.predict(X[1:10])
    assert torch.allclose(out_a.classes, out_b.classes, atol=1e-6)
    assert torch.allclose(out_a.ood_scores, out_b.ood_scores, atol=1e-6)


def _jit_buffer(module: torch.nn.Module) -> io.BytesIO:
    buffer = io.BytesIO()
    torch.jit.save(torch.jit.script(prepare_jit_module(module)), buffer)
    buffer.seek(0)
    return buffer


def _write_legacy_protonet_file(model, path: str) -> None:
    """Write a Protonet file in the legacy layout: BytesIO, Enum and scipy objects."""
    torch.save(
        {
            "embed_jit_save": _jit_buffer(model.model.embedding_model),
            "feature_names": model.feature_names,
            "label_names": model.label_names,
            "model_head_save": model.model.model_head.state_dict(),
            "outlier_kde": model.outlier_score_kde,
            "settings": {
                "cov_type": model.cov_type,
                "emb_out_dim": model.emb_out_dim,
                "use_temperature": model.use_temperature,
                "init_temperature": model.temperature.item(),
                "relative_mahal": model.relative_mahal,
                "device": model.device,
            },
            "support": model.model.support,
            "train_summary": model.train_summary,
        },
        path,
    )


def _write_legacy_gp_file(model, path: str) -> None:
    """Write a GP file in the legacy layout: BytesIO buffer and raw OrderedDicts."""
    laplace_sd = {
        k: v
        for k, v in model.model.state_dict().items()
        if "feature_extractor" not in k
    }
    torch.save(
        {
            "embed_jit_save": _jit_buffer(model.model.feature_extractor),
            "feature_names": model.feature_names,
            "label_names": model.label_names,
            "laplace_model_save": laplace_sd,
            "num_data": model.model.num_data,
            "settings": {
                "emb_out_dim": model.num_deep_features,
                "num_classes": model.num_outputs,
                "num_random_features": model.num_random_features,
                "init_temperature": model.temperature.item(),
                "device": model.device,
            },
            "support": model.support,
            "train_batch_size": model.model.train_batch_size,
            "train_summary": model.train_summary,
        },
        path,
    )


LEGACY_CASES = [
    pytest.param(_trained_protonet, _write_legacy_protonet_file, id="protonet"),
    pytest.param(_trained_gp, _write_legacy_gp_file, id="gp"),
]


# ---------------------------------------------------------------------------
# Untrusted files must not run pickle payloads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loader",
    [
        lambda p: eq.load_equine_model(p),
        lambda p: eq.EquineProtonet.load(p),
        lambda p: eq.EquineGP.load(p),
    ],
    ids=["load_equine_model", "EquineProtonet.load", "EquineGP.load"],
)
def test_loading_untrusted_file_does_not_execute_payload(tmp_path, loader) -> None:
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.eq"
    torch.save(
        {
            "train_summary": {"modelType": "EquineProtonet"},
            "payload": _MakesDirectoryWhenUnpickled(str(marker)),
        },
        path,
    )

    with pytest.raises(ValueError, match="not in the current EQUINE model format"):
        loader(str(path))

    assert not marker.exists(), "unpickling the model file executed its payload"


# ---------------------------------------------------------------------------
# Files written by save() must be loadable with torch.load(weights_only=True)
# ---------------------------------------------------------------------------


def _assert_weights_only_safe_layout(path: str) -> dict:
    checkpoint = torch.load(path, weights_only=True)  # must not raise
    assert checkpoint["equine_format_version"] == 2
    if checkpoint["contains_executable"]:
        # a flagged file holds its TorchScript archive as a uint8 tensor
        archive = checkpoint["embed_jit_save"]
        assert isinstance(archive, torch.Tensor) and archive.dtype == torch.uint8
    else:
        assert isinstance(checkpoint["embedding_recipe"]["builder"], str)
        assert "embedding_state_dict" in checkpoint
        assert "embed_jit_save" not in checkpoint
    return checkpoint


def test_protonet_save_format_is_weights_only_safe(tmp_path) -> None:
    model, X = _trained_protonet(cov_type=eq.CovType.DIAGONAL, use_temperature=True)
    # A non-default bandwidth catches a factor dropped or misapplied on reload.
    model.outlier_score_kde[1].set_bandwidth(0.3)
    path = str(tmp_path / "protonet.eq")
    model.save(path)

    checkpoint = _assert_weights_only_safe_layout(path)
    assert checkpoint["settings"]["cov_type"] == "diag"

    reloaded = eq.load_equine_model(path)

    assert isinstance(reloaded, eq.EquineProtonet)
    assert reloaded.cov_type is eq.CovType.DIAGONAL
    assert reloaded.use_temperature is True
    assert list(reloaded.outlier_score_kde.keys()) == list(
        model.outlier_score_kde.keys()
    )
    for label, kde in model.outlier_score_kde.items():
        rebuilt = reloaded.outlier_score_kde[label]
        assert rebuilt.factor == pytest.approx(kde.factor)
        np.testing.assert_allclose(rebuilt.covariance, kde.covariance)
        np.testing.assert_array_equal(rebuilt.dataset, kde.dataset)
    assert reloaded.outlier_score_kde[1].factor == pytest.approx(0.3)
    _assert_same_predictions(model, reloaded, X)


def test_gp_save_format_is_weights_only_safe(tmp_path) -> None:
    model, X = _trained_gp()
    path = str(tmp_path / "gp.eq")
    model.save(path)

    _assert_weights_only_safe_layout(path)

    reloaded = eq.load_equine_model(path)

    assert isinstance(reloaded, eq.EquineGP)
    assert list(reloaded.support.keys()) == list(model.support.keys())
    _assert_same_predictions(model, reloaded, X)


# ---------------------------------------------------------------------------
# Files from earlier releases are refused; `python -m equine.upgrade` rewrites them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
def test_legacy_file_is_refused_and_names_the_upgrade(
    tmp_path, train, write_legacy
) -> None:
    model, _ = train()
    path = str(tmp_path / "legacy.eq")
    write_legacy(model, path)

    with pytest.raises(ValueError, match="python -m equine.upgrade"):
        eq.load_equine_model(path)
    with pytest.raises(ValueError, match="python -m equine.upgrade"):
        type(model).load(path)


def test_file_without_format_version_is_refused(tmp_path, monkeypatch) -> None:
    """A file torch can read with weights_only=True but that never declared a
    format version is not a current EQUINE file either."""
    model, _ = _trained_protonet()
    path = tmp_path / "unversioned.eq"
    model.save(str(path))
    checkpoint = torch.load(path, weights_only=True)
    del checkpoint["equine_format_version"]
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="python -m equine.upgrade"):
        eq.load_equine_model(str(path))

    copies: list = []
    monkeypatch.setattr(
        equine.utils, "_copy_off_the_file", lambda tensors, device: copies.append(1)
    )
    with pytest.raises(ValueError, match="python -m equine.upgrade"):
        load_checkpoint(str(path))
    assert copies == [], "a refused unversioned file must not be copied"


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
# The filter keeps pytest from collecting the FutureWarnings the trusted loads
# emit; the upgrade's own FutureWarning is asserted below with pytest.warns.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_rewrites_legacy_file_as_a_flagged_executable_file(
    tmp_path, train, write_legacy
) -> None:
    """A legacy file's embedding is TorchScript, so the upgraded file is flagged
    and needs trust; loading its weights into a registered class and saving
    again yields a data-only file."""
    model, X = train()
    legacy_path = str(tmp_path / "legacy.eq")
    write_legacy(model, legacy_path)
    out_path = str(tmp_path / "upgraded.eq")

    with pytest.warns(FutureWarning, match="deprecated"):
        assert upgrade_model_file(legacy_path, out_path) == out_path

    checkpoint = _assert_weights_only_safe_layout(out_path)
    assert checkpoint["contains_executable"] is True
    with pytest.raises(ValueError, match="trust_executable=True"):
        eq.load_equine_model(out_path)
    reloaded = eq.load_equine_model(out_path, trust_executable=True)
    assert type(reloaded) is type(model)
    _assert_same_predictions(model, reloaded, X)

    migrated = eq.load_equine_model(
        out_path, trust_executable=True, embedding_model=BasicEmbeddingModel(6, 3)
    )
    data_only = str(tmp_path / "migrated.eq")
    migrated.save(data_only)
    assert _assert_weights_only_safe_layout(data_only)["contains_executable"] is False
    _assert_same_predictions(model, eq.load_equine_model(data_only), X)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_replaces_the_file_in_place_by_default(tmp_path) -> None:
    model, X = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    assert upgrade_model_file(path) == path

    _assert_weights_only_safe_layout(path)
    _assert_same_predictions(
        model, eq.load_equine_model(path, trust_executable=True), X
    )


def test_upgrade_leaves_a_current_file_alone(tmp_path) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "current.eq")
    model.save(path)
    before = Path(path).read_bytes()

    with pytest.warns(FutureWarning):
        assert upgrade_model_file(path, str(tmp_path / "copy.eq")) is None

    assert Path(path).read_bytes() == before
    assert not (tmp_path / "copy.eq").exists()


def test_legacy_refusal_is_its_own_error_type(tmp_path) -> None:
    """The upgrade keys off this type, so it must not cover other refusals."""
    model, _ = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    with pytest.raises(LegacyFormatError):
        load_checkpoint(path)

    current = tmp_path / "newer.eq"
    model.save(str(current))
    checkpoint = torch.load(current, weights_only=True)
    checkpoint["equine_format_version"] = EQUINE_FORMAT_VERSION + 1
    torch.save(checkpoint, current)
    with pytest.raises(ValueError, match="Upgrade EQUINE") as err:
        load_checkpoint(str(current))
    assert not isinstance(err.value, LegacyFormatError)


def test_upgrade_does_not_downgrade_a_newer_format(tmp_path) -> None:
    """A file written by a newer EQUINE is refused by the loader with its own
    message; the upgrade must not unpickle it and re-stamp it as version 2."""
    model, _ = _trained_protonet()
    path = tmp_path / "newer.eq"
    model.save(str(path))
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["equine_format_version"] = EQUINE_FORMAT_VERSION + 1
    torch.save(checkpoint, path)
    before = path.read_bytes()

    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="Upgrade EQUINE"):
        upgrade_model_file(str(path))

    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["newer.eq"]


def test_upgrade_refuses_to_run_on_old_torch(tmp_path, monkeypatch) -> None:
    """On torch < 2.6 the unrestricted unpickler must not run either."""
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    before = path.read_bytes()
    monkeypatch.setattr(equine.utils, "_torch_version", lambda: (2, 5))
    loads: list = []
    real_load = torch.load
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: loads.append(k) or real_load(*a, **k)
    )

    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="CVE-2025-32434"):
        upgrade_model_file(str(path))

    assert path.read_bytes() == before
    assert loads == []  # load_checkpoint refuses before torch.load is ever called


def test_upgrade_leaves_the_original_when_the_result_is_refused(tmp_path) -> None:
    """A legacy file holding a pickled object the converter does not know
    cannot be rewritten; the original stays, and the error says why. The
    directory name holds ". ", which the message must not be cut at."""
    model, _ = _trained_protonet()
    tmp_path = tmp_path / "my. models"
    tmp_path.mkdir()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["train_summary"]["blob"] = io.BytesIO(b"opaque")
    torch.save(checkpoint, path)
    before = path.read_bytes()

    with (
        pytest.warns(FutureWarning),
        pytest.raises(
            ValueError, match="safe loader refuses the rewritten file"
        ) as err,
    ):
        upgrade_model_file(str(path))

    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.eq"]
    assert ".upgrading" not in str(err.value)
    assert "python -m equine.upgrade" not in str(err.value)
    assert "is not in the current EQUINE model format" in str(err.value)


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
@pytest.mark.parametrize(
    "container", [list, tuple, np.array], ids=["list", "tuple", "ndarray"]
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_turns_names_into_a_list_of_plain_strings(
    tmp_path, train, write_legacy, container
) -> None:
    """Names from LabelEncoder are numpy strings or arrays, which the safe loader
    refuses; the loaders also require a list, whatever the file held."""
    model, _ = train()
    model.feature_names = container([np.str_(f"f{i}") for i in range(6)])
    model.label_names = container([np.str_(n) for n in ("a", "b", "c")])
    path = str(tmp_path / "legacy.eq")
    write_legacy(model, path)

    upgrade_model_file(path)

    checkpoint = torch.load(path, weights_only=True)
    features = [f"f{i}" for i in range(6)]
    for key, expected in (
        ("label_names", ["a", "b", "c"]),
        ("feature_names", features),
    ):
        names = checkpoint[key]
        assert type(names) is list and all(type(n) is str for n in names)
        assert names == expected
    reloaded = eq.load_equine_model(path, trust_executable=True)
    assert reloaded.label_names == ["a", "b", "c"]
    assert reloaded.feature_names == features


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_leaves_malformed_names_for_the_loader_to_report(tmp_path) -> None:
    """A value that is not a sequence of names is written as it is, so the loader
    gives its own diagnostic instead of the upgrade guessing or crashing."""
    model, _ = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["label_names"] = "ab"
    torch.save(checkpoint, path)

    upgrade_model_file(path)

    assert torch.load(path, weights_only=True)["label_names"] == "ab"
    with pytest.raises(ValueError, match="label_names must be None or a list"):
        eq.load_equine_model(path, trust_executable=True)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_in_place_keeps_a_relative_path(tmp_path, monkeypatch, capsys) -> None:
    """The documented invocation is `python -m equine.upgrade old.eq` from the
    file's directory; the result must be reported as an in-place upgrade."""
    model, X = _trained_protonet()
    _write_legacy_protonet_file(model, str(tmp_path / "old.eq"))
    _write_legacy_protonet_file(model, str(tmp_path / "old2.eq"))
    monkeypatch.chdir(tmp_path)

    assert upgrade_model_file("old.eq") == "old.eq"
    assert main(["old2.eq"]) == 0
    assert "Upgraded old2.eq in place." in capsys.readouterr().out
    _assert_same_predictions(
        model, eq.load_equine_model("old2.eq", trust_executable=True), X
    )


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_turns_numpy_scalars_into_python_values(tmp_path) -> None:
    """A numpy integer anywhere in the settings or the training summary would
    make torch.load(weights_only=True) refuse the rewritten file."""
    model, X = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["settings"]["emb_out_dim"] = np.int64(3)
    checkpoint["train_summary"]["numTrainExamples"][0]["numExamples"] = np.int64(40)
    torch.save(checkpoint, path)

    upgrade_model_file(path)

    upgraded = torch.load(path, weights_only=True)
    assert type(upgraded["settings"]["emb_out_dim"]) is int
    assert type(upgraded["train_summary"]["numTrainExamples"][0]["numExamples"]) is int
    _assert_same_predictions(
        model, eq.load_equine_model(path, trust_executable=True), X
    )


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_warns_when_the_mode_cannot_be_copied(tmp_path, monkeypatch) -> None:
    """A filesystem that rejects mode changes still gets the upgraded file."""
    model, X = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    def refuse(src, dst, **kwargs):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(shutil, "copymode", refuse)
    with pytest.warns(UserWarning, match="permission bits of '.*legacy.eq'"):
        upgrade_model_file(path)
    _assert_same_predictions(
        model, eq.load_equine_model(path, trust_executable=True), X
    )

    # The warning names the file whose bits were read: the one being replaced.
    other = str(tmp_path / "other.eq")
    _write_legacy_protonet_file(model, other)
    existing = str(tmp_path / "existing.eq")
    _write_legacy_protonet_file(model, existing)
    with pytest.warns(UserWarning, match="permission bits of '.*existing.eq'"):
        upgrade_model_file(other, existing)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_follows_a_symlink_and_keeps_the_mode(tmp_path) -> None:
    model, X = _trained_protonet()
    target = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(target))
    os.chmod(target, 0o600)
    link = tmp_path / "link.eq"
    link.symlink_to(target)

    assert upgrade_model_file(str(link)) == str(target.resolve())

    assert link.is_symlink()
    assert os.stat(target).st_mode & 0o777 == 0o600
    _assert_same_predictions(
        model, eq.load_equine_model(str(link), trust_executable=True), X
    )


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_keeps_use_temperature_for_protonet_files(tmp_path) -> None:
    """EquineProtonet still uses use_temperature; only EquineGP files lose the key."""
    model, X = _trained_protonet(use_temperature=True)
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    upgrade_model_file(path)

    assert torch.load(path, weights_only=True)["settings"]["use_temperature"] is True
    reloaded = eq.load_equine_model(path, trust_executable=True)
    assert reloaded.use_temperature is True
    _assert_same_predictions(model, reloaded, X)


@pytest.mark.parametrize(
    "names",
    [np.array("abc"), np.array([["a", "b"], ["c", "d"]])],
    ids=["0-d", "2-d"],
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_leaves_a_name_array_of_the_wrong_rank_alone(tmp_path, names) -> None:
    """Only one-dimensional arrays are names; any other rank is written as it
    is, which the safe loader then refuses, so the upgrade reports that and
    keeps the original."""
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["label_names"] = names
    torch.save(checkpoint, path)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="safe loader refuses the rewritten file"):
        upgrade_model_file(str(path))

    assert path.read_bytes() == before


def test_upgrade_refuses_a_file_that_is_not_an_equine_model(tmp_path) -> None:
    path = tmp_path / "other.pt"
    torch.save({"weights": torch.zeros(2)}, path)

    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="not an EQUINE"):
        upgrade_model_file(str(path))


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_command_line(tmp_path, capsys) -> None:
    model, X = _trained_gp()
    legacy_path = str(tmp_path / "legacy.eq")
    _write_legacy_gp_file(model, legacy_path)
    out_path = str(tmp_path / "upgraded.eq")

    assert main([legacy_path, "-o", out_path]) == 0
    captured = capsys.readouterr()
    assert f"Upgraded {legacy_path} -> {out_path}" in captured.out
    assert "deprecated" in captured.err
    _assert_same_predictions(
        model, eq.load_equine_model(out_path, trust_executable=True), X
    )

    assert main([str(tmp_path / "missing.eq")]) == 1
    assert "error:" in capsys.readouterr().err

    assert main([out_path]) == 0  # already current
    captured = capsys.readouterr()
    assert "already in the current format; nothing written" in captured.out
    assert "Upgraded" not in captured.out

    garbage = tmp_path / "garbage.eq"
    garbage.write_bytes(b"hello world, not a pickle")
    assert main([str(garbage)]) == 1
    assert "error:" in capsys.readouterr().err

    assert main([str(tmp_path)]) == 1  # a directory
    assert "error:" in capsys.readouterr().err

    legacy3 = str(tmp_path / "legacy3.eq")
    _write_legacy_gp_file(model, legacy3)
    assert main([legacy3, "-o", str(tmp_path / "missing_dir" / "out.eq")]) == 1
    assert "error: Could not write" in capsys.readouterr().err

    # In place, with the note about dropped Linear-head weights.
    noted = str(tmp_path / "head.eq")
    protonet, _ = _trained_protonet()
    _write_legacy_protonet_file(protonet, noted)
    checkpoint = torch.load(noted, weights_only=False)
    checkpoint["model_head_save"] = dict(_LINEAR_HEAD)
    torch.save(checkpoint, noted)
    assert main([noted]) == 0
    captured = capsys.readouterr()
    assert f"Upgraded {noted} in place." in captured.out
    assert "note:" in captured.err and "Linear head" in captured.err


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_converts_non_zip_files(tmp_path) -> None:
    """torch's pre-1.6 format sizes storages from the pickle alone, so the safe
    loader refuses it; the upgrade rewrites it as a zip archive."""
    model, X = _trained_protonet()
    checkpoint_path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, checkpoint_path)
    old_format = str(tmp_path / "old_format.eq")
    torch.save(
        torch.load(checkpoint_path, weights_only=False),
        old_format,
        _use_new_zipfile_serialization=False,
    )

    with pytest.raises(ValueError, match="python -m equine.upgrade"):
        load_checkpoint(old_format)
    upgrade_model_file(old_format)
    _assert_same_predictions(
        model, eq.load_equine_model(old_format, trust_executable=True), X
    )


_LINEAR_HEAD = {"weight": torch.eye(3), "bias": torch.zeros(3)}


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_drops_legacy_linear_head_weights_with_a_note(tmp_path) -> None:
    """A legacy Linear head's weights are dropped, with a note."""
    model, X = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["model_head_save"] = dict(_LINEAR_HEAD)
    torch.save(checkpoint, path)

    with pytest.warns(UserWarning, match="Linear head"):
        upgrade_model_file(path)

    assert torch.load(path, weights_only=True)["model_head_save"] == {}
    _assert_same_predictions(
        model, eq.load_equine_model(path, trust_executable=True), X
    )


def test_linear_head_weights_in_a_current_file_name_the_upgrade(tmp_path) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "m.eq")
    model.save(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["model_head_save"] = dict(_LINEAR_HEAD)
    torch.save(checkpoint, path)

    for loader in (eq.EquineProtonet.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="0.1.6") as err:
            loader(path)
        assert "python -m equine.upgrade" in str(err.value)
        assert "tampered" not in str(err.value)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_drops_use_temperature_from_gp_settings(tmp_path) -> None:
    """Legacy EquineGP files may store use_temperature, which EquineGP does not
    accept; the upgrade drops it and keeps the calibrated temperature."""
    model, X = _trained_gp()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_gp_file(model, path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["settings"]["use_temperature"] = True
    torch.save(checkpoint, path)

    upgrade_model_file(path)

    assert "use_temperature" not in torch.load(path, weights_only=True)["settings"]
    _assert_same_predictions(
        model, eq.load_equine_model(path, trust_executable=True), X
    )


def _slices_of_one_tensor(support):
    """The same support rows, re-stored as views of a single tensor."""
    base = torch.cat(list(support.values()))
    out, start = type(support)(), 0
    for label, rows in support.items():
        out[label] = base[start : start + len(rows)]
        start += len(rows)
    return out


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_copies_aliased_support_apart(tmp_path, train, write_legacy) -> None:
    """Legacy files may store support tensors that share one storage. The safe
    loader refuses that (see _aliasing_checkpoint); the upgrade copies the
    tensors apart."""
    model, X = train()
    support = model.model.support if isinstance(model, eq.EquineProtonet) else None
    if support is not None:
        model.model.support = _slices_of_one_tensor(support)
    else:
        model.support = _slices_of_one_tensor(model.support)
    legacy_path = str(tmp_path / "legacy.eq")
    write_legacy(model, legacy_path)
    stored = torch.load(legacy_path, weights_only=False)["support"]
    storages = {t.untyped_storage().data_ptr() for t in stored.values()}
    assert len(storages) == 1, "the legacy file must hold aliased support"

    upgrade_model_file(legacy_path)

    upgraded = torch.load(legacy_path, weights_only=True)["support"]
    assert len({t.untyped_storage().data_ptr() for t in upgraded.values()}) == len(
        upgraded
    )
    _assert_same_predictions(
        model, eq.load_equine_model(legacy_path, trust_executable=True), X
    )


# The transition's FutureWarning is asserted in its own test.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_relabelled_executable_file_is_refused_without_trust(
    tmp_path, monkeypatch
) -> None:
    """Every model type the generic loader dispatches to enforces the executable
    check, so relabelling a flagged file cannot route it past the check."""
    dataset, _ = _tiny_dataset()
    model = eq.EquineProtonet(
        torch.jit.script(prepare_jit_module(BasicEmbeddingModel(6, 3))), 3
    )
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    flagged = str(tmp_path / "flagged.eq")
    model.save(flagged, allow_executable=True)
    checkpoint = torch.load(flagged, weights_only=True)
    assert checkpoint["contains_executable"] is True
    checkpoint["train_summary"]["modelType"] = "EquineGP"
    relabelled = str(tmp_path / "relabelled.eq")
    torch.save(checkpoint, relabelled)

    jit_loads: list = []
    real_jit_load = torch.jit.load

    def spy(*args, **kwargs):
        jit_loads.append(args)
        return real_jit_load(*args, **kwargs)

    monkeypatch.setattr(torch.jit, "load", spy)
    with pytest.raises(ValueError, match="trust_executable=True"):
        eq.load_equine_model(relabelled)
    assert jit_loads == [], "the embedded TorchScript ran without trust"

    eq.load_equine_model(flagged, trust_executable=True)
    assert len(jit_loads) == 1, "the spy must see the archive being loaded"


def test_safe_layout_with_payload_is_refused_without_running_it(tmp_path) -> None:
    """A file that declares the current format but smuggles a payload is not an
    EQUINE model; the default load never reaches the payload."""
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.eq"
    torch.save(
        {
            "equine_format_version": EQUINE_FORMAT_VERSION,
            "train_summary": {"modelType": "EquineProtonet"},
            "payload": _MakesDirectoryWhenUnpickled(str(marker)),
        },
        path,
    )
    with pytest.raises(ValueError, match="not in the current EQUINE model format"):
        eq.load_equine_model(str(path))
    assert not marker.exists()


# ---------------------------------------------------------------------------
# Every tensor a checkpoint yields must be backed by its own storage, so a
# tensor's shape never claims more memory than the file actually supplied.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_tensor, message",
    [
        (lambda: torch.zeros(1).expand(1000, 6), "storage is smaller than its shape"),
        (
            lambda: torch.arange(100.0).as_strided((10, 3), (1, 1)),
            "storage is smaller than its shape",
        ),
        (lambda: torch.empty(1000, 1000, device="meta"), "sparse or meta tensor"),
        (
            lambda: torch.sparse_coo_tensor(
                torch.tensor([[0], [0]]), torch.tensor([1.0]), (100_000, 6)
            ),
            "sparse or meta tensor",
        ),
    ],
    ids=["expanded", "overlapping", "meta", "sparse"],
)
def test_load_checkpoint_refuses_tensors_without_honest_storage(
    tmp_path, make_tensor, message
) -> None:
    path = tmp_path / "trick.pt"
    torch.save({"nested": [{"deep": (make_tensor(),)}]}, path)
    with pytest.raises(ValueError, match=message):
        load_checkpoint(str(path))


def test_load_checkpoint_accepts_honest_views_and_cycles(tmp_path) -> None:
    base = torch.arange(24.0).reshape(4, 6)
    cyclic: list = []
    cyclic.append(cyclic)
    data = {
        "transposed": base.t(),
        "columns": base[:, :3],
        "rows": base[1:3],
        "empty": torch.zeros(0, 5),
        "shared": [base, base],
        "cyclic": cyclic,
        "equine_format_version": EQUINE_FORMAT_VERSION,
    }
    path = tmp_path / "honest.pt"
    torch.save(data, path)
    loaded = load_checkpoint(str(path))
    for key in ("transposed", "columns", "rows", "empty"):
        assert torch.equal(loaded[key], data[key])


def test_load_checkpoint_refuses_a_non_dictionary(tmp_path) -> None:
    path = tmp_path / "list.pt"
    torch.save([torch.zeros(2)], path)
    with pytest.raises(ValueError, match="not an EQUINE model file"):
        load_checkpoint(str(path))


# ---------------------------------------------------------------------------
# torch.save writes an uncompressed zip, so an archive's entries can never
# hold more than the file itself. A compressed archive (a zip bomb) or one
# whose entries claim more bytes than the file is refused before torch.load
# can inflate it.
# ---------------------------------------------------------------------------

_NOT_PLAIN_ARCHIVE = r"not a plain \(uncompressed\) PyTorch archive"


def _rewrite_zip_deflated(src: str, dst: str) -> None:
    """Copy a torch archive entry by entry into a DEFLATE-compressed zip."""
    with (
        zipfile.ZipFile(src) as zin,
        zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout,
    ):
        for info in zin.infolist():
            zout.writestr(info.filename, zin.read(info.filename))


@pytest.fixture
def torch_load_calls(monkeypatch) -> list:
    """Record every call to torch.load."""
    calls: list = []
    real_load = torch.load

    def spy(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("f"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    return calls


def test_compressed_archive_is_refused_before_loading(
    tmp_path, torch_load_calls
) -> None:
    model, X = _trained_protonet()
    genuine = str(tmp_path / "genuine.eq")
    model.save(genuine)
    deflated = str(tmp_path / "deflated.eq")
    _rewrite_zip_deflated(genuine, deflated)

    torch_load_calls.clear()
    for loader in (eq.load_equine_model, eq.EquineProtonet.load, load_checkpoint):
        with pytest.raises(ValueError, match=_NOT_PLAIN_ARCHIVE):
            loader(deflated)
    assert torch_load_calls == [], "torch.load ran on a compressed archive"

    _assert_same_predictions(model, eq.load_equine_model(genuine), X)


def test_archive_whose_entries_claim_more_than_the_file_is_refused(tmp_path) -> None:
    """Overlapping entries (several names over the same bytes) show up the same
    way: the entries add up to more than the file."""
    path = tmp_path / "sizes.pt"
    torch.save({"x": torch.zeros(16)}, path)
    data = bytearray(path.read_bytes())
    header = data.find(b"PK\x01\x02")  # central-directory entries
    while header != -1:
        struct.pack_into("<I", data, header + 24, 2**31)  # uncompressed size
        header = data.find(b"PK\x01\x02", header + 4)
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match=_NOT_PLAIN_ARCHIVE):
        load_checkpoint(str(path))


def test_unreadable_archive_is_refused(tmp_path) -> None:
    """torch treats any file starting with the zip signature as a zip; one that
    the zipfile module cannot read is refused rather than left to torch."""
    path = tmp_path / "broken.pt"
    path.write_bytes(b"PK\x03\x04" + bytes(64))
    with pytest.raises(ValueError, match=_NOT_PLAIN_ARCHIVE):
        load_checkpoint(str(path))


def test_stored_archive_without_a_pickle_is_refused(tmp_path) -> None:
    """A well-formed, uncompressed zip that is not a PyTorch archive passes the
    zipfile pre-check; torch's own failure to read it surfaces as a bounded
    ValueError rather than a RuntimeError."""
    path = tmp_path / "no_pickle.pt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("no_pickle/version", "3\n")
    with pytest.raises(ValueError, match="could not be read as a PyTorch archive"):
        load_checkpoint(str(path))


# ---------------------------------------------------------------------------
# A storage's size comes from the pickle, not from the bytes in the file, so
# the safe path maps the file (mmap), refuses storages that claim more than
# their record holds, and then copies every storage off the mapping.
# ---------------------------------------------------------------------------

_STORAGE_LIE = "declares tensor storage larger than the data it contains"


def _first_record_shortened(name: str, data: bytes) -> bytes:
    return data[:8] if name.endswith("/data/0") else data


def test_storage_claim_past_the_end_of_the_file_is_refused(tmp_path) -> None:
    src, dst = str(tmp_path / "src.pt"), str(tmp_path / "dst.pt")
    torch.save({"a": torch.zeros(1_000_000, dtype=torch.uint8)}, src)
    _rewrite_zip(src, dst, _first_record_shortened)
    assert os.path.getsize(dst) < 10_000
    with pytest.raises(ValueError, match="refusing to load"):
        load_checkpoint(dst)


def test_storage_claim_into_the_next_record_is_refused(tmp_path) -> None:
    """A short record followed by more data: the claim stays inside the file but
    reads the following record, so the mapped storages overlap."""
    src, dst = str(tmp_path / "src.pt"), str(tmp_path / "dst.pt")
    torch.save(
        {
            "a": torch.full((1024,), 7, dtype=torch.uint8),
            "b": torch.full((200_000,), 9, dtype=torch.uint8),
        },
        src,
    )
    _rewrite_zip(src, dst, _first_record_shortened)
    with pytest.raises(ValueError, match=_STORAGE_LIE):
        load_checkpoint(dst)


def test_loaded_tensors_are_copied_off_the_file(tmp_path) -> None:
    base = torch.arange(1000.0)
    path = tmp_path / "t.pt"
    torch.save(
        {
            "a": base,
            "view": base[10:20],
            "b": torch.ones(5),
            "equine_format_version": EQUINE_FORMAT_VERSION,
        },
        path,
    )
    loaded = load_checkpoint(str(path))
    # A storage read from a file is never resizable; a fresh copy is.
    tensors = [v for v in loaded.values() if isinstance(v, torch.Tensor)]
    assert all(t.untyped_storage().resizable() for t in tensors)
    # Copied once per storage, so views still share it.
    assert (
        loaded["view"].untyped_storage().data_ptr()
        == loaded["a"].untyped_storage().data_ptr()
    )
    os.truncate(path, 0)  # would fault a tensor still backed by the file
    assert torch.equal(loaded["view"], torch.arange(10.0, 20.0))


def _aliasing_checkpoint(kind: str) -> dict:
    shared = torch.rand(50, 6)
    kde = {"dataset": shared, "bw_factor": 0.5}
    return {
        "same_tensor": {"support": {0: shared, 1: shared}},
        "same_kde": {"outlier_kde": {0: kde, 1: kde}},
        "support_and_kde": {"support": {0: shared}, "outlier_kde": {0: kde}},
        "views": {"support": {0: shared[:25], 1: shared[25:]}},
    }[kind]


@pytest.mark.parametrize(
    "kind", ["same_tensor", "same_kde", "support_and_kde", "views"]
)
def test_aliased_support_or_kde_tensors_are_refused(tmp_path, kind) -> None:
    """One stored tensor reused across labels multiplies per-label work and memory."""
    path = str(tmp_path / "alias.pt")
    expected = _aliasing_checkpoint(kind)
    torch.save(expected, path)
    with pytest.raises(ValueError, match="alias one another") as err:
        load_checkpoint(path)
    assert "python -m equine.upgrade" in str(err.value)

    # The upgrade gives each label its own copy, which the loader then accepts.
    equine.utils._unalias_support(expected)
    assert not equine.utils._support_is_aliased(expected)


@pytest.mark.parametrize("container", [set, frozenset])
def test_load_checkpoint_refuses_sets(tmp_path, container) -> None:
    """Tensors inside a set would skip the storage checks and the copy; EQUINE
    never writes sets."""
    path = str(tmp_path / "set.pt")
    torch.save({"s": container([torch.zeros(1).expand(1000, 6)])}, path)
    if container is set:
        with pytest.raises(ValueError, match="contains a set"):
            load_checkpoint(path)
    else:  # restricted unpickling refuses frozensets by itself
        with pytest.raises(ValueError, match="not in the current EQUINE model format"):
            load_checkpoint(path)


_OTHER_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else None
)


@pytest.mark.accelerator
@pytest.mark.skipif(_OTHER_DEVICE is None, reason="needs a CUDA or MPS device")
def test_load_checkpoint_maps_tensors_to_the_requested_device(tmp_path) -> None:
    base = torch.arange(100.0)
    path = str(tmp_path / "t.pt")
    torch.save(
        {
            "a": base,
            "view": base[10:20],
            "equine_format_version": EQUINE_FORMAT_VERSION,
        },
        path,
    )
    loaded = load_checkpoint(path, map_location=_OTHER_DEVICE)
    tensors = [v for v in loaded.values() if isinstance(v, torch.Tensor)]
    assert {t.device.type for t in tensors} == {_OTHER_DEVICE}
    assert torch.equal(loaded["view"].cpu(), torch.arange(10.0, 20.0))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
# The transition's FutureWarning is asserted in its own test.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgraded_legacy_file_loads_onto_mps_and_predicts(
    tmp_path, train, write_legacy
) -> None:
    """A legacy file reaches an accelerator through the upgrade:
    every tensor lands on the device, and the model predicts there."""
    model, X = train()  # float32, trained on the CPU
    path = str(tmp_path / "legacy.eq")
    write_legacy(model, path)
    upgrade_model_file(path)

    checkpoint = load_checkpoint(path, map_location="mps")
    tensors = equine.utils._unique_tensors(checkpoint)
    assert tensors and {t.device.type for t in tensors} == {"mps"}

    expected = model.predict(X[1:10])
    for loader in (type(model).load, eq.load_equine_model):
        loaded = loader(path, device="mps", trust_executable=True)
        assert loaded.device == "mps"
        assert_on_device(loaded, "mps")
        actual = loaded.predict(X[1:10])
        torch.testing.assert_close(
            actual.classes.cpu(), expected.classes, atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            actual.ood_scores.cpu(), expected.ood_scores, atol=1e-4, rtol=0
        )


def test_jit_archive_tensor_round_trips() -> None:
    """`_jit_archive_to_tensor` followed by `_load_jit_archive` must reconstruct
    a scripted module whose outputs match the module that was saved."""
    from equine.utils import _jit_archive_to_tensor, _load_jit_archive

    module = BasicEmbeddingModel(6, 3)
    buffer = _jit_buffer(module)
    x = torch.rand(4, 6)
    expected = module(x)

    tensor = _jit_archive_to_tensor(buffer)
    rebuilt = _load_jit_archive(tensor)
    assert torch.allclose(rebuilt(x), expected, atol=1e-6)


def test_archive_bytes_handles_offset_view() -> None:
    """A uint8 tensor that is an offset, non-full view of a larger storage
    must convert to exactly its own bytes, not the whole backing storage."""
    from equine.utils import _archive_bytes

    padded = torch.arange(20, dtype=torch.uint8)
    view = padded[5:10]
    assert view.storage_offset() != 0

    assert _archive_bytes(view) == bytes(range(5, 10))


def test_archive_bytes_is_fast_for_large_tensors() -> None:
    """`_archive_bytes` must convert a multi-megabyte archive quickly: a loop
    over the storage one Python int at a time takes ~1 s/MB, far above the
    0.5 s bound for this 4 MB tensor."""
    import time

    from equine.utils import _archive_bytes

    tensor = torch.randint(0, 256, (4_000_000,), dtype=torch.uint8)

    start = time.perf_counter()
    result = _archive_bytes(tensor)
    elapsed = time.perf_counter() - start

    assert result == tensor.numpy().tobytes()
    assert elapsed < 0.5


def test_load_jit_archive_accepts_all_stored_forms() -> None:
    """The archive loader reads the uint8 tensor that ``save`` writes as well as
    the bytes and BytesIO forms found in legacy files."""
    from equine.utils import _jit_archive_to_tensor, _load_jit_archive

    module = BasicEmbeddingModel(6, 3)
    buffer = _jit_buffer(module)
    x = torch.rand(4, 6)
    expected = module(x)

    for archive in (_jit_archive_to_tensor(buffer), buffer.getvalue(), buffer):
        rebuilt = _load_jit_archive(archive)
        assert torch.allclose(rebuilt(x), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# The torch floor (weights_only=True is bypassable before torch 2.6)
# ---------------------------------------------------------------------------


def test_declared_torch_floor_is_at_least_2_6() -> None:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            pytest.skip("tomllib/tomli not available")

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    dependencies = data["project"]["dependencies"]
    torch_deps = [dep for dep in dependencies if re.match(r"^torch\b", dep)]

    assert len(torch_deps) == 1, (
        f"expected exactly one torch dependency, got {torch_deps}"
    )
    torch_dep = torch_deps[0]
    assert ";" not in torch_dep, (
        f"torch dependency must not be platform-conditional: {torch_dep!r}"
    )

    match = re.search(r">=\s*(\d+)\.(\d+)", torch_dep)
    assert match, f"could not find a >= floor in {torch_dep!r}"
    floor = (int(match.group(1)), int(match.group(2)))
    assert floor >= (2, 6), f"declared torch floor {floor} is below the required (2, 6)"


def test_load_checkpoint_refuses_old_torch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(equine.utils, "_torch_version", lambda: (2, 5))

    path = tmp_path / "trivial.pt"
    torch.save({"a": 1}, path)

    with pytest.raises(ValueError, match="CVE-2025-32434"):
        load_checkpoint(str(path))


def test_load_checkpoint_accepts_supported_torch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(equine.utils, "_torch_version", lambda: (2, 6))

    path = tmp_path / "trivial.pt"
    torch.save({"a": 1, "equine_format_version": EQUINE_FORMAT_VERSION}, path)

    assert load_checkpoint(str(path))["a"] == 1


@pytest.mark.parametrize("version", ["3", True], ids=["str", "bool"])
def test_load_rejects_a_malformed_format_version(tmp_path, version) -> None:
    path = str(tmp_path / "m.eq")
    torch.save({"equine_format_version": version}, path)
    with pytest.raises(ValueError, match="equine_format_version is malformed") as err:
        load_checkpoint(path)
    assert len(str(err.value)) < 300


def test_huge_format_version_gives_a_bounded_error() -> None:
    """The weights-only unpickler refuses such an int; the check still bounds its
    message for any caller that reaches it another way."""
    with pytest.raises(ValueError, match="Unsupported EQUINE") as err:
        equine.utils._validate_format_version({"equine_format_version": 10**4000})
    assert len(str(err.value)) < 300


def test_load_rejects_newer_format_version(tmp_path) -> None:
    """A file declaring a format version newer than this EQUINE build
    understands must be rejected rather than silently misread."""
    model, _ = _trained_protonet()
    path = tmp_path / "future.eq"
    model.save(str(path))

    checkpoint = torch.load(path, weights_only=True)
    checkpoint["equine_format_version"] = EQUINE_FORMAT_VERSION + 1
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="Unsupported EQUINE model format version"):
        eq.load_equine_model(str(path))
