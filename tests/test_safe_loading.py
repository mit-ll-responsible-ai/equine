# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
The file-reading layer (``equine.utils.load_checkpoint``) and ``python -m
equine.upgrade``: files from earlier releases (PyTorch archives and pickles) are
refused without being unpickled, and the upgrade rewrites them in the current format.
The file format itself is tested in test_model_file.py, the model layer on top
(recipes, save/load) in test_persistence.py.
"""

import collections
import contextlib
import io
import itertools
import json
import os
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import (
    BasicEmbeddingModel,
    archive_tensor,
    assert_on_device,
    script_module,
)
from safetensors.torch import save_file

import equine as eq
import equine.upgrade
import equine.utils
from equine.upgrade import main, upgrade_model_file
from equine.utils import (
    EQUINE_FORMAT_VERSION,
    LegacyFormatError,
    load_checkpoint,
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
    """A Protonet trained on ``_tiny_dataset`` with ``ood_distance="rmd_sqrt"``."""
    dataset, X = _tiny_dataset()
    model = eq.EquineProtonet(
        BasicEmbeddingModel(6, 3),
        3,
        cov_type=cov_type,
        use_temperature=use_temperature,
        ood_distance="rmd_sqrt",
    )
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    return model, X


def _trained_gp(embedding=None, vis_support=True):
    dataset, X = _tiny_dataset()
    embedding = BasicEmbeddingModel(6, 3) if embedding is None else embedding
    model = eq.EquineGP(embedding, 3, 3, num_random_features=16)
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.001),
        num_epochs=1,
        batch_size=32,
        vis_support=vis_support,
    )
    return model, X


def _assert_same_predictions(a, b, X):
    out_a, out_b = a.predict(X[1:10]), b.predict(X[1:10])
    assert torch.allclose(out_a.classes, out_b.classes, atol=1e-6)
    assert torch.allclose(out_a.ood_scores, out_b.ood_scores, atol=1e-6)


_RECIPE = (
    '{"builder": "equine.tests.basic", "kwargs": {"tensor_dim": 6, "num_classes": 3}}'
)


def _upgrade(path, out_path=None, **kwargs):
    """``upgrade_model_file`` with the test architecture the files were built from."""
    return upgrade_model_file(path, BasicEmbeddingModel(6, 3), out_path, **kwargs)


@contextlib.contextmanager
def _default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _jit_buffer(module: torch.nn.Module) -> io.BytesIO:
    buffer = io.BytesIO()
    torch.jit.save(script_module(module), buffer)
    buffer.seek(0)
    return buffer


def _write_legacy_protonet_file(model, path: str) -> None:
    """Write a Protonet file in the legacy layout: BytesIO, Enum and scipy objects.

    The settings hold no ``ood_distance``; ``model`` must use ``"rmd_sqrt"``.
    """
    assert model.ood_distance == "rmd_sqrt"
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
# Files written by save() hold the embedding as a recipe and weights
# ---------------------------------------------------------------------------


def _assert_data_only_layout(path: str) -> dict:
    checkpoint = load_checkpoint(path)
    assert checkpoint["equine_format_version"] == EQUINE_FORMAT_VERSION
    assert isinstance(checkpoint["embedding_recipe"]["builder"], str)
    assert "embedding_state_dict" in checkpoint
    assert "embed_jit_save" not in checkpoint
    assert "contains_executable" not in checkpoint
    return checkpoint


def test_protonet_save_writes_a_data_only_file(tmp_path) -> None:
    model, X = _trained_protonet(cov_type=eq.CovType.DIAGONAL, use_temperature=True)
    # A non-default bandwidth catches a factor dropped or misapplied on reload.
    model.outlier_score_kde[1].set_bandwidth(0.3)
    path = str(tmp_path / "protonet.eq")
    model.save(path)

    checkpoint = _assert_data_only_layout(path)
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


def test_gp_save_writes_a_data_only_file(tmp_path) -> None:
    model, X = _trained_gp()
    path = str(tmp_path / "gp.eq")
    model.save(path)

    _assert_data_only_layout(path)

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


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_rewrites_legacy_file_as_a_data_only_file(
    tmp_path, train, write_legacy
) -> None:
    """A legacy file's TorchScript embedding becomes the supplied architecture's
    recipe and weights."""
    model, X = train()
    legacy_path = str(tmp_path / "legacy.eq")
    write_legacy(model, legacy_path)
    out_path = str(tmp_path / "upgraded.eq")

    with pytest.warns(FutureWarning, match="deprecated"):
        assert _upgrade(legacy_path, out_path) == out_path

    _assert_data_only_layout(out_path)
    reloaded = eq.load_equine_model(out_path)
    assert type(reloaded) is type(model)
    _assert_same_predictions(model, reloaded, X)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_reads_a_legacy_file_when_torch_allows_its_globals(tmp_path) -> None:
    model, _ = _trained_gp()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_gp_file(model, path)
    with torch.serialization.safe_globals([io.BytesIO]):
        with pytest.raises(LegacyFormatError):
            load_checkpoint(path)
        assert _upgrade(path) == path


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_replaces_the_file_in_place_by_default(tmp_path) -> None:
    model, X = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    assert _upgrade(path) == path

    _assert_data_only_layout(path)
    _assert_same_predictions(model, eq.load_equine_model(path), X)


def test_upgrade_leaves_a_current_file_alone(tmp_path) -> None:
    """Nothing is written and no deprecation is issued."""
    model, _ = _trained_protonet()
    path = str(tmp_path / "current.eq")
    model.save(path)
    before = Path(path).read_bytes()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _upgrade(path, str(tmp_path / "copy.eq")) is None

    assert Path(path).read_bytes() == before
    assert not (tmp_path / "copy.eq").exists()


def _newer_model_file(path) -> None:
    newer = EQUINE_FORMAT_VERSION + 1
    save_file(
        {},
        str(path),
        metadata={
            "equine_format_version": str(newer),
            "equine": json.dumps({"equine_format_version": newer}),
        },
    )


@pytest.mark.parametrize(
    "write, message",
    [
        (_newer_model_file, "Upgrade EQUINE"),
        (
            lambda path: torch.save({"equine_format_version": 5}, path),
            "is a PyTorch archive with equine_format_version 5, which no EQUINE "
            "version wrote",
        ),
    ],
    ids=["newer-model-file", "archive"],
)
def test_upgrade_does_not_downgrade_a_newer_format(tmp_path, write, message) -> None:
    """A model file written by a newer EQUINE is refused by the loader with its own
    message, and a PyTorch archive declaring a format other than 2 or 3 is refused;
    the upgrade must not unpickle either and re-stamp it as the current version."""
    path = tmp_path / "newer.eq"
    write(path)
    before = path.read_bytes()

    with pytest.raises(ValueError, match=message):
        _upgrade(str(path))

    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["newer.eq"]


def test_upgrade_refuses_to_run_on_old_torch(tmp_path, monkeypatch) -> None:
    """On torch < 2.6 the unrestricted unpickler must not run either."""
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    before = path.read_bytes()
    monkeypatch.setattr(equine.upgrade, "_torch_version", lambda: (2, 5))
    loads: list = []
    real_load = torch.load
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: loads.append(k) or real_load(*a, **k)
    )

    with pytest.raises(ValueError, match="CVE-2025-32434"):
        _upgrade(str(path))

    assert path.read_bytes() == before
    assert loads == []  # the upgrade refuses before torch.load is ever called


@pytest.mark.parametrize(
    "version, key, value, reason",
    [
        (None, "blob", io.BytesIO(b"opaque"), "['blob'] is a BytesIO"),
        (None, "blob", b"opaque", "['blob'] is a bytes"),
        (None, "blob", {1, 2}, "['blob'] is a set"),
        (None, "label_names", np.array("abc"), "['label_names'] is a ndarray"),
        (
            None,
            "label_names",
            np.array([["a", "b"], ["c", "d"]]),
            "['label_names'] is a ndarray",
        ),
        (2, "blob", torch.ones(2).to(torch.complex128), "is a complex128 tensor"),
        (3, "blob", torch.ones(2).to(torch.complex128), "is a complex128 tensor"),
    ],
    ids=["bytesio", "bytes", "set", "names-0-d", "names-2-d", "format-2", "format-3"],
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_leaves_the_original_when_the_result_is_refused(
    tmp_path, version, key, value, reason
) -> None:
    """A file holding a value a model file does not store cannot be rewritten: a
    legacy file (``version`` None) or a format-2 or format-3 file. Names are converted
    only from a list, a tuple or a one-dimensional array. The original stays, and the
    error says why. The directory name holds ". ", which the message must not be cut
    at."""
    model, _ = _trained_protonet()
    tmp_path = tmp_path / "my. models"
    tmp_path.mkdir()
    path = tmp_path / "legacy.eq"
    if version is None:
        _write_legacy_protonet_file(model, str(path))
        checkpoint = torch.load(path, weights_only=False)
    else:
        checkpoint = _in_format(model, str(path), version)
    if key == "blob":
        checkpoint["train_summary"]["blob"] = value
    else:
        checkpoint[key] = value
    torch.save(checkpoint, path)
    before = path.read_bytes()

    with pytest.raises(ValueError) as err:
        _upgrade(str(path))

    assert re.fullmatch(
        f"'{re.escape(str(path))}' holds a value the upgrade does not convert "
        rf"\(Cannot save: checkpoint.*{re.escape(reason)}, which a model file does "
        r"not store; .*\); it was left unchanged\. To convert it, read it with "
        r"torch\.load \(weights_only=True for a format 2 or 3 file; weights_only=False "
        r"for a file from EQUINE 0\.1\.x, which runs code from the file, so only if you "
        r"trust it\), change that value, write it back with torch\.save, and upgrade it "
        r"again\.",
        str(err.value),
    ), str(err.value)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.eq"]


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
@pytest.mark.parametrize(
    "container", [list, tuple, np.array], ids=["list", "tuple", "ndarray"]
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_turns_names_into_a_list_of_plain_strings(
    tmp_path, train, write_legacy, container
) -> None:
    """Names from LabelEncoder are numpy strings or arrays, which a model file does
    not store; the loaders also require a list, whatever the file held."""
    model, _ = train()
    model.feature_names = container([np.str_(f"f{i}") for i in range(6)])
    model.label_names = container([np.str_(n) for n in ("a", "b", "c")])
    path = str(tmp_path / "legacy.eq")
    write_legacy(model, path)

    _upgrade(path)

    checkpoint = load_checkpoint(path)
    features = [f"f{i}" for i in range(6)]
    for key, expected in (
        ("label_names", ["a", "b", "c"]),
        ("feature_names", features),
    ):
        names = checkpoint[key]
        assert type(names) is list and all(type(n) is str for n in names)
        assert names == expected
    reloaded = eq.load_equine_model(path)
    assert reloaded.label_names == ["a", "b", "c"]
    assert reloaded.feature_names == features


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_refuses_names_the_loader_would_refuse(tmp_path) -> None:
    """A value the loader refuses fails the upgrade's reload check; the original
    stays."""
    model, _ = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["label_names"] = "ab"
    torch.save(checkpoint, path)
    before = Path(path).read_bytes()

    with pytest.raises(
        ValueError,
        match=r"rewritten file does not load: Model file's label_names must be None "
        r"or a list of strings; got 'ab'\. '",
    ):
        _upgrade(path)
    assert Path(path).read_bytes() == before
    assert not list(tmp_path.glob(".*.upgrading"))


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_in_place_keeps_a_relative_path(tmp_path, monkeypatch, capsys) -> None:
    """The documented invocation is `python -m equine.upgrade old.eq --recipe ...`
    from the file's directory; the result must be reported as an in-place upgrade."""
    model, X = _trained_protonet()
    _write_legacy_protonet_file(model, str(tmp_path / "old.eq"))
    _write_legacy_protonet_file(model, str(tmp_path / "old2.eq"))
    monkeypatch.chdir(tmp_path)

    assert _upgrade("old.eq") == "old.eq"
    assert main(["old2.eq", "--recipe", _RECIPE]) == 0
    assert "Upgraded old2.eq in place." in capsys.readouterr().out
    _assert_same_predictions(model, eq.load_equine_model("old2.eq"), X)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_turns_numpy_scalars_into_python_values(tmp_path) -> None:
    """A numpy integer anywhere in the settings or the training summary is written
    as a Python int."""
    model, X = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["settings"]["emb_out_dim"] = np.int64(3)
    checkpoint["train_summary"]["numTrainExamples"][0]["numExamples"] = np.int64(40)
    torch.save(checkpoint, path)

    _upgrade(path)

    upgraded = load_checkpoint(path)
    assert type(upgraded["settings"]["emb_out_dim"]) is int
    assert type(upgraded["train_summary"]["numTrainExamples"][0]["numExamples"]) is int
    _assert_same_predictions(model, eq.load_equine_model(path), X)


def _refuse_read_only_targets(monkeypatch) -> None:
    """Make os.replace and os.remove refuse a read-only file, as Windows does."""
    replace, remove = os.replace, os.remove

    def read_only(path) -> bool:
        return os.path.exists(path) and not os.stat(path).st_mode & 0o200

    def windows_replace(src, dst):
        if read_only(dst):
            raise PermissionError(13, "Access is denied", dst)
        replace(src, dst)

    def windows_remove(path):
        if read_only(path):
            raise PermissionError(13, "Access is denied", path)
        remove(path)

    monkeypatch.setattr(os, "replace", windows_replace)
    monkeypatch.setattr(os, "remove", windows_remove)
    monkeypatch.setattr(equine.upgrade, "_RENAME_NEEDS_A_WRITABLE_TARGET", True)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("fchmod", [True, False], ids=["fchmod", "no-fchmod"])
def test_upgrade_replaces_a_read_only_file_the_os_refuses_to_replace(
    tmp_path, monkeypatch, fchmod
) -> None:
    model, X = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    os.chmod(path, 0o444)
    _refuse_read_only_targets(monkeypatch)
    if not fchmod:
        monkeypatch.delattr(os, "fchmod", raising=False)

    _upgrade(str(path))

    assert os.stat(path).st_mode & 0o777 == 0o444
    assert not list(tmp_path.glob(".*.upgrading"))
    _assert_same_predictions(model, eq.load_equine_model(str(path)), X)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("fchmod", [True, False], ids=["fchmod", "no-fchmod"])
def test_upgrade_restores_a_read_only_file_when_the_move_fails(
    tmp_path, monkeypatch, fchmod
) -> None:
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    os.chmod(path, 0o444)
    before = path.read_bytes()
    _refuse_read_only_targets(monkeypatch)
    if not fchmod:
        monkeypatch.delattr(os, "fchmod", raising=False)

    def refuse(src, dst):
        raise PermissionError(13, "Access is denied", dst)

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError):
        _upgrade(str(path))

    assert os.stat(path).st_mode & 0o777 == 0o444
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*.upgrading"))


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.skipif(os.name == "nt", reason="needs symbolic links")
@pytest.mark.parametrize("windows", [False, True], ids=["posix", "windows-link"])
def test_upgrade_leaves_files_alone_when_a_retry_cannot_help(
    tmp_path, monkeypatch, windows
) -> None:
    """A refused rename is retried only where a read-only target blocks it
    (Windows), and not when the target has become a link in the meantime: no
    file's mode is changed."""
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    os.chmod(victim, 0o600)
    output = tmp_path / "out.eq"
    output.write_bytes(b"theirs")
    os.chmod(output, 0o444)
    monkeypatch.setattr(equine.upgrade, "_RENAME_NEEDS_A_WRITABLE_TARGET", windows)
    chmodded = []
    chmod = os.chmod

    def spy(target, mode, **kwargs):
        chmodded.append(os.fspath(target))
        chmod(target, mode, **kwargs)

    def refuse(src, dst):
        if windows:  # a link swapped in while the rename was refused
            os.remove(dst)
            os.symlink(victim, dst)
        raise PermissionError(1, "Operation not permitted", dst)

    monkeypatch.setattr(os, "chmod", spy)
    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError):
        _upgrade(str(path), str(output))

    assert str(output) not in chmodded
    assert os.stat(victim).st_mode & 0o777 == 0o600
    assert victim.read_text() == "keep me"


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.skipif(os.name == "nt", reason="group and other bits are POSIX")
@pytest.mark.parametrize("fchmod", [True, False], ids=["fchmod", "no-fchmod"])
@pytest.mark.parametrize("mode", [0o640, 0o444])
def test_upgrade_moves_a_file_that_already_has_the_targets_bits(
    tmp_path, monkeypatch, mode, fchmod
) -> None:
    model, X = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    os.chmod(path, mode)
    if not fchmod:
        monkeypatch.delattr(os, "fchmod", raising=False)
    modes_at_move = []
    chmodded = []
    replace, chmod = os.replace, os.chmod

    def replace_spy(src, dst):
        modes_at_move.append(os.stat(src).st_mode & 0o777)
        replace(src, dst)

    def chmod_spy(target, mode, **kwargs):
        chmodded.append(os.fspath(target))
        chmod(target, mode, **kwargs)

    monkeypatch.setattr(os, "replace", replace_spy)
    monkeypatch.setattr(os, "chmod", chmod_spy)
    old_umask = os.umask(0o077)
    try:
        _upgrade(str(path))
    finally:
        os.umask(old_umask)

    assert modes_at_move == [mode]
    assert [os.path.basename(p).endswith(".upgrading") for p in chmodded] == (
        [] if fchmod else [True]
    )
    assert os.stat(path).st_mode & 0o777 == mode
    _assert_same_predictions(model, eq.load_equine_model(str(path)), X)


def _swap_during_the_checks(monkeypatch, swap) -> None:
    """Run ``swap`` once the rewritten file is written and read back, before
    its embedding is compared and it is moved into place."""
    compare = equine.upgrade._require_same_embedding
    swapped = []

    def spy(*args):
        if not swapped:
            swapped.append(True)
            swap()
        return compare(*args)

    monkeypatch.setattr(equine.upgrade, "_require_same_embedding", spy)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.skipif(os.name == "nt", reason="needs POSIX links and permission bits")
@pytest.mark.parametrize("existing", [True, False], ids=["existing", "new"])
@pytest.mark.parametrize(
    "replacement",
    ["symlink", "symlink-to-written", "dangling-symlink", "hard-link", "removed"],
)
def test_upgrade_refuses_a_rewritten_file_swapped_before_the_move(
    tmp_path, monkeypatch, replacement, existing
) -> None:
    """Another file put at the temp name is neither given the target's bits
    nor moved into place, even when the target's bits were changed too."""
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    output = tmp_path / "out.eq"
    if existing:
        output.write_bytes(b"theirs")
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    os.chmod(victim, 0o600)

    def swap():
        (partial,) = tmp_path.glob(".out.eq.*.upgrading")
        if existing:
            os.chmod(output, 0o666)
        if replacement == "symlink-to-written":
            written = tmp_path / "written.eq"
            partial.rename(written)
            partial.symlink_to(written)
            return
        partial.unlink()
        if replacement == "symlink":
            partial.symlink_to(victim)
        elif replacement == "dangling-symlink":
            partial.symlink_to(tmp_path / "missing")
        elif replacement == "hard-link":
            os.link(victim, partial)

    _swap_during_the_checks(monkeypatch, swap)
    with pytest.raises(OSError, match="replaced or removed before it could be moved"):
        _upgrade(str(path), str(output))

    assert os.stat(victim).st_mode & 0o777 == 0o600
    assert victim.read_text() == "keep me"
    if existing:
        assert not output.is_symlink()
        assert output.read_bytes() == b"theirs"
    else:
        assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".*.upgrading"))


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.skipif(os.name == "nt", reason="the temp file is checked on POSIX")
def test_upgrade_reports_the_refusal_when_a_directory_is_swapped_in(
    tmp_path, monkeypatch
) -> None:
    """A directory put at the temp name, which the clean-up cannot remove,
    does not hide the refusal."""
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    output = tmp_path / "out.eq"
    output.write_bytes(b"theirs")

    def swap():
        (partial,) = tmp_path.glob(".out.eq.*.upgrading")
        partial.unlink()
        partial.mkdir()
        (partial / "theirs.txt").write_text("")

    _swap_during_the_checks(monkeypatch, swap)
    with pytest.raises(OSError, match="replaced or removed before it could be moved"):
        _upgrade(str(path), str(output))

    assert output.read_bytes() == b"theirs"
    (left,) = tmp_path.glob(".out.eq.*.upgrading")
    assert left.is_dir()


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("failing", ["flush", "close"])
def test_upgrade_reports_a_write_that_fails_when_flushed(
    tmp_path, monkeypatch, failing
) -> None:
    """A rewritten file that cannot be flushed, when it is written or when it
    is closed before the move, is reported with its name and removed."""
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    before = path.read_bytes()

    class FullDisk(io.BufferedWriter):
        closing = False

        def flush(self):
            if failing == "flush" or self.closing:
                raise OSError(28, "No space left on device")
            super().flush()

        def close(self):
            self.closing = True
            super().close()

    monkeypatch.setattr(
        os, "fdopen", lambda descriptor, *args: FullDisk(io.FileIO(descriptor, "wb"))
    )
    with pytest.raises(
        OSError, match=r"Could not write '.*\.upgrading'.*No space left on device"
    ):
        _upgrade(str(path))

    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.eq"]


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_reports_a_failed_check_when_the_file_cannot_be_closed(
    tmp_path, monkeypatch
) -> None:
    """An error from closing the rewritten file does not hide the error that
    stopped the upgrade."""
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    before = path.read_bytes()

    class QuotaAtClose(io.FileIO):
        def close(self):
            super().close()
            raise OSError(122, "Disk quota exceeded")

    def refuse(*args):
        raise ValueError("does not reproduce the archive's outputs")

    monkeypatch.setattr(
        os,
        "fdopen",
        lambda descriptor, *args: io.BufferedWriter(QuotaAtClose(descriptor, "wb")),
    )
    monkeypatch.setattr(equine.upgrade, "_require_same_embedding", refuse)
    with pytest.raises(ValueError, match="does not reproduce"):
        _upgrade(str(path))

    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.eq"]


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_closes_the_rewritten_file_before_moving_it(
    tmp_path, monkeypatch
) -> None:
    """The rewritten file is closed before it is renamed."""
    model, X = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    opened = []
    closed_at_move = []
    fdopen, replace = os.fdopen, os.replace

    def fdopen_spy(*args, **kwargs):
        opened.append(fdopen(*args, **kwargs))
        return opened[-1]

    def replace_spy(src, dst):
        closed_at_move.append([file.closed for file in opened])
        replace(src, dst)

    monkeypatch.setattr(os, "fdopen", fdopen_spy)
    monkeypatch.setattr(os, "replace", replace_spy)
    _upgrade(str(path))

    assert closed_at_move == [[True]]
    _assert_same_predictions(model, eq.load_equine_model(str(path)), X)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.skipif(os.name == "nt", reason="needs symbolic links")
def test_upgrade_refuses_an_output_link_that_cannot_be_followed(
    tmp_path, capsys
) -> None:
    """Refused, like the other arguments, before the file is unpickled."""
    path, marker = _legacy_file_with_payload(tmp_path)
    loop = tmp_path / "loop.eq"
    loop.symlink_to(loop)
    capsys.readouterr()  # training progress

    with pytest.raises(ValueError, match="is a symbolic link that cannot be followed"):
        _upgrade(path, str(loop))
    assert main([path, "--recipe", _RECIPE, "-o", str(loop)]) == 1
    assert "cannot be followed" in capsys.readouterr().err

    assert not marker.exists()
    assert loop.is_symlink()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.eq", "loop.eq"]


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.skipif(os.name == "nt", reason="needs symbolic links")
def test_upgrade_refuses_a_target_that_became_a_link(tmp_path, monkeypatch) -> None:
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    output = tmp_path / "out.eq"
    output.write_bytes(b"theirs")
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    os.chmod(victim, 0o600)

    def swap():
        output.unlink()
        output.symlink_to(victim)

    _swap_during_the_checks(monkeypatch, swap)
    with pytest.raises(OSError, match="became a symbolic link during the upgrade"):
        _upgrade(str(path), str(output))

    assert output.is_symlink()
    assert os.stat(victim).st_mode & 0o777 == 0o600
    assert victim.read_text() == "keep me"
    assert not list(tmp_path.glob(".*.upgrading"))


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "folder",
    [
        "models",
        pytest.param(
            "back\\slash",
            marks=pytest.mark.skipif(
                os.name == "nt", reason="backslash is a path separator"
            ),
        ),
    ],
)
@pytest.mark.parametrize("fchmod", [True, False], ids=["fchmod", "no-fchmod"])
def test_upgrade_warns_when_the_mode_cannot_be_copied(
    tmp_path, monkeypatch, folder, fchmod
) -> None:
    """A filesystem that rejects mode changes still gets the upgraded file."""
    model, X = _trained_protonet()
    (tmp_path / folder).mkdir()
    path = str(tmp_path / folder / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    def refuse(target, mode, **kwargs):
        raise PermissionError(1, "Operation not permitted", os.fspath(target))

    def refuse_descriptor(descriptor, mode):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "chmod", refuse)
    if fchmod:
        monkeypatch.setattr(os, "fchmod", refuse_descriptor, raising=False)
    else:
        monkeypatch.delattr(os, "fchmod", raising=False)
    with pytest.warns(UserWarning, match="permission bits of '.*legacy.eq'") as caught:
        _upgrade(path)
    assert not any(".upgrading" in str(w.message) for w in caught)
    _assert_same_predictions(model, eq.load_equine_model(path), X)

    # The warning names the file whose bits were read: the one being replaced.
    other = str(tmp_path / "other.eq")
    _write_legacy_protonet_file(model, other)
    existing = str(tmp_path / "existing.eq")
    _write_legacy_protonet_file(model, existing)
    with pytest.warns(UserWarning, match="permission bits of '.*existing.eq'"):
        _upgrade(other, existing)


def test_a_converted_file_that_reads_back_different_is_not_kept(
    tmp_path, monkeypatch
) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "m.eq")
    torch.save(_in_format(model, path, 3), path)
    before = Path(path).read_bytes()

    def altered(file: str) -> dict:
        checkpoint = load_checkpoint(file)
        checkpoint["support"][0] = checkpoint["support"][0] + 1
        return checkpoint

    monkeypatch.setattr(equine.upgrade, "load_checkpoint", altered)
    with pytest.raises(ValueError, match="does not read back as written; '.*m.eq' was"):
        upgrade_model_file(path)
    assert Path(path).read_bytes() == before
    assert os.listdir(tmp_path) == ["m.eq"]


def test_the_repair_a_refused_upgrade_names_lets_the_file_convert(tmp_path) -> None:
    model, X = _trained_protonet()
    path = str(tmp_path / "m.eq")
    checkpoint = _in_format(model, path, 3)
    checkpoint["train_summary"]["blob"] = torch.ones(2, dtype=torch.complex128)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="weights_only=True for a format 2 or 3 file"):
        upgrade_model_file(path)

    repaired = torch.load(path, weights_only=True)
    repaired["train_summary"]["blob"] = repaired["train_summary"]["blob"].real
    torch.save(repaired, path)
    assert upgrade_model_file(path) == path
    _assert_same_predictions(model, eq.load_equine_model(path), X)


def test_converting_a_data_only_file_imports_the_modules_named_on_the_command_line(
    tmp_path, capsys
) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "m.eq")
    torch.save(_in_format(model, path, 3), path)
    capsys.readouterr()  # training progress
    assert main([path, "--import", "no_such_module_for_this_test"]) == 1
    assert "Could not import 'no_such_module_for_this_test'" in capsys.readouterr().err


@pytest.mark.parametrize("via", ["api", "cli"])
def test_converting_a_data_only_file_refuses_an_unusable_output(
    tmp_path, capsys, via
) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "m.eq")
    torch.save(_in_format(model, path, 3), path)
    before = Path(path).read_bytes()
    outputs = {
        "is not an existing directory": str(tmp_path / "no_such_dir" / "out.eq"),
        "names a directory": str(tmp_path),
    }
    names = ["m.eq"]
    if os.name != "nt":
        loop = tmp_path / "loop.eq"
        loop.symlink_to(loop)
        outputs["is a symbolic link that cannot be followed"] = str(loop)
        names.append("loop.eq")
    capsys.readouterr()  # training progress
    for message, output in outputs.items():
        if via == "api":
            with pytest.raises(ValueError, match=message):
                upgrade_model_file(path, out_path=output)
        else:
            assert main([path, "-o", output]) == 1
            assert message in capsys.readouterr().err
    assert Path(path).read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(names)


@pytest.mark.parametrize("via", ["api", "cli"])
def test_converting_a_data_only_file_warns_when_the_mode_cannot_be_copied(
    tmp_path, monkeypatch, capsys, via
) -> None:
    model, X = _trained_protonet()
    path = str(tmp_path / "m.eq")
    torch.save(_in_format(model, path, 3), path)

    def refuse(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "chmod", refuse)
    monkeypatch.setattr(os, "fchmod", refuse, raising=False)
    capsys.readouterr()  # training progress
    if via == "api":
        with pytest.warns(UserWarning, match="permission bits of '.*m.eq'"):
            assert upgrade_model_file(path) == path
    else:
        assert main([path]) == 0
        out, err = capsys.readouterr()
        assert out == f"Upgraded {path} in place.\n"
        assert err.startswith(f"note: Could not copy the permission bits of '{path}'")
    _assert_same_predictions(model, eq.load_equine_model(path), X)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_follows_a_symlink_and_keeps_the_mode(tmp_path) -> None:
    model, X = _trained_protonet()
    target = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(target))
    os.chmod(target, 0o600)
    link = tmp_path / "link.eq"
    link.symlink_to(target)

    assert _upgrade(str(link)) == str(target.resolve())

    assert link.is_symlink()
    assert os.stat(target).st_mode & 0o777 == 0o600
    _assert_same_predictions(model, eq.load_equine_model(str(link)), X)


@contextlib.contextmanager
def _umask(mask):
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_writes_a_private_file_for_a_private_original(
    tmp_path, monkeypatch
) -> None:
    """The written file has the original's permission bits, not the umask's,
    while it is written and while it is checked; a new output gets the
    umask's, as save() does."""
    model, X = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    os.chmod(path, 0o600)
    write_modes: list = []

    class Recording(io.BufferedWriter):
        def write(self, data):
            write_modes.append(os.fstat(self.fileno()).st_mode & 0o777)
            return super().write(data)

    modes: list = []
    compare = equine.upgrade._require_same_embedding

    def spy(*args):
        modes.extend(p.stat().st_mode & 0o777 for p in tmp_path.glob(".*.upgrading"))
        return compare(*args)

    monkeypatch.setattr(
        os, "fdopen", lambda descriptor, *args: Recording(io.FileIO(descriptor, "wb"))
    )
    monkeypatch.setattr(equine.upgrade, "_require_same_embedding", spy)
    with _umask(0o022):
        _upgrade(str(path))
    assert write_modes and set(write_modes) == {0o600}
    assert modes and set(modes) == {0o600}
    assert path.stat().st_mode & 0o777 == 0o600
    _assert_same_predictions(model, eq.load_equine_model(str(path)), X)

    _write_legacy_protonet_file(model, str(path))
    out = tmp_path / "new.eq"
    write_modes.clear()
    with _umask(0o027):
        _upgrade(str(path), str(out))
    assert write_modes and set(write_modes) == {0o640}
    assert out.stat().st_mode & 0o777 == 0o640


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX permission bits, which root bypasses",
)
@pytest.mark.parametrize("mode", [0o200, 0o000], ids=["write-only", "no-access"])
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_replaces_an_output_its_owner_cannot_read(tmp_path, mode) -> None:
    """The written file can be read back while it is checked, and the result
    gets the exact permission bits of the file it replaces."""
    model, X = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    out = tmp_path / "out.eq"
    out.write_bytes(b"replaced")
    os.chmod(out, mode)

    assert _upgrade(str(path), str(out)) == str(out)

    assert out.stat().st_mode & 0o777 == mode
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.eq", "out.eq"]
    os.chmod(out, 0o600)
    _assert_same_predictions(model, eq.load_equine_model(str(out)), X)


@pytest.mark.skipif(os.name == "nt", reason="needs symbolic links")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_does_not_follow_a_link_planted_at_the_temp_name(
    tmp_path, monkeypatch
) -> None:
    """The written file is created exclusively under a random name: links at
    a predictable name, or at the name drawn, are left alone."""
    model, X = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    names = itertools.cycle(["planted", "fresh"])
    monkeypatch.setattr(equine.upgrade, "token_hex", lambda nbytes: next(names))
    links = [
        tmp_path / ".legacy.eq.upgrading",
        tmp_path / ".legacy.eq.planted.upgrading",
    ]
    for link in links:
        link.symlink_to(victim)

    _upgrade(str(path))

    assert victim.read_text() == "keep me"
    assert not path.is_symlink()
    for link in links:
        assert link.is_symlink() and link.resolve() == victim.resolve()
    assert not (tmp_path / ".legacy.eq.fresh.upgrading").exists()
    _assert_same_predictions(model, eq.load_equine_model(str(path)), X)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_removes_its_file_when_the_write_fails(tmp_path, monkeypatch) -> None:
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    before = path.read_bytes()

    class FailingWrite(io.BufferedWriter):
        def write(self, data):
            super().write(b"partial")
            raise RuntimeError("disk full")

    monkeypatch.setattr(
        os,
        "fdopen",
        lambda descriptor, *args: FailingWrite(io.FileIO(descriptor, "wb")),
    )
    with pytest.raises(OSError, match=r"Could not write '.*\.upgrading'.*disk full"):
        _upgrade(str(path))

    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.eq"]


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_gives_up_when_every_temp_name_is_taken(tmp_path, monkeypatch) -> None:
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    before = path.read_bytes()
    taken = tmp_path / ".legacy.eq.taken.upgrading"
    taken.write_text("someone else's")
    monkeypatch.setattr(equine.upgrade, "token_hex", lambda nbytes: "taken")

    with pytest.raises(ValueError, match="each of 100 random names already exists"):
        _upgrade(str(path))

    assert path.read_bytes() == before
    assert taken.read_text() == "someone else's"


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_keeps_use_temperature_for_protonet_files(tmp_path) -> None:
    """EquineProtonet still uses use_temperature; only EquineGP files lose the key."""
    model, X = _trained_protonet(use_temperature=True)
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    _upgrade(path)

    assert load_checkpoint(path)["settings"]["use_temperature"] is True
    reloaded = eq.load_equine_model(path)
    assert reloaded.use_temperature is True
    _assert_same_predictions(model, reloaded, X)


def test_upgrade_refuses_a_file_that_is_not_an_equine_model(tmp_path) -> None:
    path = tmp_path / "other.pt"
    torch.save({"weights": torch.zeros(2)}, path)

    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="not an EQUINE"):
        _upgrade(str(path))


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_command_line(tmp_path, capsys) -> None:
    model, X = _trained_gp()
    legacy_path = str(tmp_path / "legacy.eq")
    _write_legacy_gp_file(model, legacy_path)
    out_path = str(tmp_path / "upgraded.eq")

    assert main([legacy_path, "-o", out_path, "--recipe", _RECIPE]) == 0
    captured = capsys.readouterr()
    assert f"Upgraded {legacy_path} -> {out_path}" in captured.out
    assert "deprecated" in captured.err
    _assert_same_predictions(model, eq.load_equine_model(out_path), X)

    assert main([str(tmp_path / "missing.eq")]) == 1
    assert "error:" in capsys.readouterr().err

    assert main([out_path]) == 0  # already current
    captured = capsys.readouterr()
    assert "already in the current format; nothing written" in captured.out
    assert "Upgraded" not in captured.out

    garbage = tmp_path / "garbage.eq"
    garbage.write_bytes(b"hello world, not a pickle")
    assert main([str(garbage), "--recipe", _RECIPE]) == 1
    assert f"error: '{garbage}' is not an EQUINE model file" in capsys.readouterr().err
    garbage.write_bytes(b"PK\x03\x04" + bytes(64))
    assert main([str(garbage), "--recipe", _RECIPE]) == 1
    assert f"error: '{garbage}' could not be read as a PyTorch file" in (
        capsys.readouterr().err
    )
    garbage.write_bytes(b"\x80\x02cnonexistent_module_for_this_test\nthing\nq\x00.")
    assert main([str(garbage), "--recipe", _RECIPE]) == 1
    assert (
        f"error: '{garbage}' could not be read as a PyTorch file (ModuleNotFoundError"
        in capsys.readouterr().err
    )

    assert main([str(tmp_path)]) == 1  # a directory
    assert "error:" in capsys.readouterr().err

    legacy3 = str(tmp_path / "legacy3.eq")
    _write_legacy_gp_file(model, legacy3)
    assert (
        main(
            [
                legacy3,
                "-o",
                str(tmp_path / "missing_dir" / "out.eq"),
                "--recipe",
                _RECIPE,
            ]
        )
        == 1
    )
    assert "missing_dir' is not an existing directory" in capsys.readouterr().err

    # In place, with the note about dropped Linear-head weights.
    noted = str(tmp_path / "head.eq")
    protonet, _ = _trained_protonet()
    _write_legacy_protonet_file(protonet, noted)
    checkpoint = torch.load(noted, weights_only=False)
    checkpoint["model_head_save"] = dict(_LINEAR_HEAD)
    torch.save(checkpoint, noted)
    assert main([noted, "--recipe", _RECIPE]) == 0
    captured = capsys.readouterr()
    assert f"Upgraded {noted} in place." in captured.out
    assert "note:" in captured.err and "Linear head" in captured.err


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_converts_non_zip_files(tmp_path) -> None:
    """The loader refuses torch's pre-1.6 format; the upgrade rewrites it."""
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
    _upgrade(old_format)
    _assert_same_predictions(model, eq.load_equine_model(old_format), X)


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
        _upgrade(path)

    assert load_checkpoint(path)["model_head_save"] == {}
    _assert_same_predictions(model, eq.load_equine_model(path), X)


def test_linear_head_weights_in_a_current_file_are_refused(tmp_path) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "m.eq")
    model.save(path)
    checkpoint = load_checkpoint(path)
    checkpoint["model_head_save"] = dict(_LINEAR_HEAD)
    equine._model_file.write(path, checkpoint)

    for loader in (eq.EquineProtonet.load, eq.load_equine_model):
        with pytest.raises(
            ValueError, match="model_head_save does not match the model its settings"
        ):
            loader(path)


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

    _upgrade(path)

    assert "use_temperature" not in load_checkpoint(path)["settings"]
    _assert_same_predictions(model, eq.load_equine_model(path), X)


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
    """Legacy files may store support tensors that share one storage; the upgrade
    writes each label's rows."""
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

    _upgrade(legacy_path)

    _assert_same_predictions(model, eq.load_equine_model(legacy_path), X)


@eq.embedding_architecture("equine.tests.normalized")
class _Normalized(torch.nn.Module):
    """The registered form of an embedding that normalizes with stored statistics."""

    def __init__(self, tensor_dim: int, num_classes: int) -> None:
        super().__init__()
        self.register_buffer("train_means", torch.zeros(tensor_dim))
        self.register_buffer("train_deviations", torch.ones(tensor_dim))
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(tensor_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.stack((x - self.train_means) / self.train_deviations)


class _NormalizedWithAttributes(torch.nn.Module):
    """The same embedding with the statistics as plain tensor attributes."""

    def __init__(self, tensor_dim: int, num_classes: int) -> None:
        super().__init__()
        self.train_means = torch.randn(tensor_dim)
        self.train_deviations = torch.rand(tensor_dim) + 0.5
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(tensor_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.stack((x - self.train_means) / self.train_deviations)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_takes_tensors_the_archive_keeps_as_attributes(tmp_path) -> None:
    dataset, X = _tiny_dataset()
    model = eq.EquineProtonet(
        _NormalizedWithAttributes(6, 3), 3, ood_distance="rmd_sqrt"
    )
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    upgrade_model_file(path, _Normalized(6, 3))

    _assert_same_predictions(model, eq.load_equine_model(path), X)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_refuses_an_architecture_that_computes_differently(tmp_path) -> None:
    """Same weight names and shapes, different forward: the check catches it."""

    @eq.embedding_architecture("equine.tests.tanh_variant")
    class TanhVariant(BasicEmbeddingModel):
        def forward(self, x):
            return torch.tanh(super().forward(x))

    model, _ = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    before = Path(path).read_bytes()

    with pytest.raises(ValueError, match="does not reproduce the file's embedding"):
        upgrade_model_file(path, TanhVariant(6, 3))
    assert Path(path).read_bytes() == before
    assert not list(tmp_path.glob(".*.upgrading"))


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_checks_the_written_file_not_the_callers_instance(tmp_path) -> None:
    """Statistics set on the instance are not stored. The check runs on the written
    file and fails."""

    @eq.embedding_architecture("equine.tests.instance_stats")
    class InstanceStats(torch.nn.Module):
        def __init__(self, tensor_dim: int, num_classes: int) -> None:
            super().__init__()
            self.train_means = torch.zeros(tensor_dim)
            self.train_deviations = torch.ones(tensor_dim)
            self.stack = torch.nn.Sequential(
                torch.nn.Linear(tensor_dim, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, num_classes),
            )

        def forward(self, x):
            return self.stack((x - self.train_means) / self.train_deviations)

    dataset, _ = _tiny_dataset()
    legacy = eq.EquineProtonet(
        _NormalizedWithAttributes(6, 3), 3, ood_distance="rmd_sqrt"
    )
    legacy.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(legacy, path)
    target = InstanceStats(6, 3)
    target.train_means = legacy.model.embedding_model.train_means.clone()
    target.train_deviations = legacy.model.embedding_model.train_deviations.clone()

    with pytest.raises(ValueError, match="does not match|does not reproduce"):
        upgrade_model_file(path, target)


class _LegacyWithBuffer(torch.nn.Module):
    """A legacy embedding with a non-persistent buffer (listed in its scripted
    state_dict)."""

    def __init__(self, tensor_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(tensor_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes),
        )
        self.register_buffer("scale", torch.full((1,), 2.0), persistent=False)

    def forward(self, x):
        return self.linear_relu_stack(x) * self.scale


@eq.embedding_architecture("equine.tests.with_extras")
class _WithExtras(torch.nn.Module):
    """The registered form, with the same buffer and an extra state."""

    def __init__(self, tensor_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(tensor_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes),
        )
        self.register_buffer("scale", torch.full((1,), 2.0), persistent=False)

    def forward(self, x):
        return self.linear_relu_stack(x) * self.scale

    def get_extra_state(self):
        return {"version": 1}

    def set_extra_state(self, state):
        if state != {"version": 1}:
            raise ValueError(f"unexpected extra state {state!r}")


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_keeps_extra_state_and_non_persistent_buffers(tmp_path) -> None:
    dataset, X = _tiny_dataset()
    model = eq.EquineProtonet(_LegacyWithBuffer(6, 3), 3, ood_distance="rmd_sqrt")
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    upgrade_model_file(path, _WithExtras(6, 3))

    _assert_same_predictions(model, eq.load_equine_model(path), X)


@eq.embedding_architecture("equine.tests.complex_extra_state")
class _ComplexExtraState(torch.nn.Module):
    """``_WithExtras`` with an extra state that a model file does not store."""

    def __init__(self, tensor_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(tensor_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes),
        )
        self.register_buffer("scale", torch.full((1,), 2.0), persistent=False)

    def forward(self, x):
        return self.linear_relu_stack(x) * self.scale

    def get_extra_state(self):
        return torch.ones(2, dtype=torch.complex64)

    def set_extra_state(self, state):
        pass


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_names_the_embedding_model_for_a_value_a_file_does_not_store(
    tmp_path,
) -> None:
    dataset, _ = _tiny_dataset()
    model = eq.EquineProtonet(_LegacyWithBuffer(6, 3), 3, ood_distance="rmd_sqrt")
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    before = Path(path).read_bytes()

    with pytest.raises(
        ValueError,
        match=r"^The supplied embedding_model \(_ComplexExtraState\) holds a value a "
        r"model file does not store \(Cannot save: "
        r"checkpoint\['embedding_state_dict'\]\['_extra_state'\] is a complex64 tensor",
    ):
        upgrade_model_file(path, _ComplexExtraState(6, 3))
    assert Path(path).read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.eq"]


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_reports_names_for_a_mismatched_architecture(tmp_path) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    with pytest.raises(ValueError, match="does not match the file's TorchScript"):
        upgrade_model_file(path, eq.MLP(6, [7], 3))


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_refuses_an_unregistered_module(tmp_path) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    class Plain(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="embedding_architecture"):
        upgrade_model_file(path, Plain())


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_of_a_gpu_trained_file_on_the_cpu(tmp_path, monkeypatch) -> None:
    """The settings name cuda and the archive is read onto the CPU."""
    model, X = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["settings"]["device"] = "cuda"
    torch.save(checkpoint, path)
    locations = []
    real_load = torch.jit.load

    def spy(buffer, map_location=None, **kwargs):
        locations.append(map_location)
        return real_load(buffer, map_location=map_location, **kwargs)

    monkeypatch.setattr(torch.jit, "load", spy)

    _upgrade(path)

    assert locations == ["cpu"]

    assert load_checkpoint(path)["settings"]["device"] == "cuda"
    _assert_same_predictions(model, eq.load_equine_model(path, device="cpu"), X)


def _gp_file_without_support(tmp_path, support=None, embedding=None):
    """A legacy GP file whose support is None, missing or empty
    (``vis_support=False``, the default)."""
    model, X = _trained_gp(embedding, vis_support=False)
    path = str(tmp_path / "legacy.eq")
    _write_legacy_gp_file(model, path)
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["support"] is None or len(checkpoint["support"]) == 0
    if support == "missing":
        del checkpoint["support"]
    else:
        checkpoint["support"] = support
    torch.save(checkpoint, path)
    return model, X, path


_NO_SUPPORT = pytest.mark.parametrize(
    "support", [{}, "missing", None], ids=["empty", "0.1.2", "0.1.5"]
)


@_NO_SUPPORT
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_refuses_a_file_without_rows_to_compare_on(
    tmp_path, monkeypatch, support
) -> None:
    """Without support rows or check_inputs the rebuilt embedding cannot be
    checked: nothing is written, before the archive is even loaded."""
    _, _, path = _gp_file_without_support(tmp_path, support)
    before = Path(path).read_bytes()
    loads: list = []
    monkeypatch.setattr(torch.jit, "load", lambda *a, **k: loads.append(1))

    with pytest.raises(ValueError, match="stores no support rows") as err:
        _upgrade(path)

    for hint in ("check_inputs=", "--check-inputs FILE", "allow_unchecked=True"):
        assert hint in str(err.value)
    assert "--no-check" in str(err.value)
    assert loads == []
    assert Path(path).read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.eq"]


@_NO_SUPPORT
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_of_a_gp_file_without_support_with_allow_unchecked(
    tmp_path, support
) -> None:
    model, X, path = _gp_file_without_support(tmp_path, support)

    with pytest.warns(UserWarning, match="not compared"):
        assert _upgrade(path, allow_unchecked=True) == path

    reloaded = eq.load_equine_model(path)
    assert len(reloaded.support) == 0
    assert torch.allclose(
        reloaded.predict(X[1:10]).classes, model.predict(X[1:10]).classes, atol=1e-6
    )


_MLP_RECIPE = (
    '{"builder": "equine.mlp", "kwargs": {"in_features": 6, "hidden_sizes": [8], '
    '"out_features": 3%s}}'
)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_command_line_without_support_rows(tmp_path, capsys) -> None:
    """A tanh MLP upgraded with the default (relu) recipe: refused without rows
    to compare on, caught with --check-inputs, written unchecked with --no-check."""
    model, X, path = _gp_file_without_support(
        tmp_path, {}, eq.MLP(6, [8], 3, activation="tanh")
    )
    before = Path(path).read_bytes()
    relu, tanh = _MLP_RECIPE % "", _MLP_RECIPE % ', "activation": "tanh"'
    rows = str(tmp_path / "rows.pt")
    torch.save(X[:16], rows)
    capsys.readouterr()  # training progress

    assert main([path, "--recipe", relu]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "stores no support rows" in err
    assert "--check-inputs FILE" in err and "--no-check" in err
    assert main([path, "--recipe", relu, "--check-inputs", rows]) == 1
    assert "does not reproduce the file's embedding" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([path, "--recipe", tanh, "--check-inputs", rows, "--no-check"])
    capsys.readouterr()
    assert Path(path).read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.eq", "rows.pt"]

    checked = str(tmp_path / "checked.eq")
    assert main([path, "-o", checked, "--recipe", tanh, "--check-inputs", rows]) == 0
    assert "note:" not in capsys.readouterr().err
    _assert_same_predictions(model, eq.load_equine_model(checked), X)

    assert main([path, "--recipe", relu, "--no-check"]) == 0
    captured = capsys.readouterr()
    assert f"Upgraded {path} in place." in captured.out
    assert "note:" in captured.err and "not compared" in captured.err


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_without_the_check_still_compares_support_rows(
    tmp_path, capsys
) -> None:
    """allow_unchecked (--no-check) waives only a comparison there are no rows
    for: a tanh MLP's support rows still refuse the default (relu) recipe."""
    dataset, _ = _tiny_dataset()
    model = eq.EquineProtonet(
        eq.MLP(6, [8], 3, activation="tanh"), 3, ood_distance="rmd_sqrt"
    )
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    before = path.read_bytes()
    capsys.readouterr()  # training progress

    with pytest.raises(ValueError, match="does not reproduce"):
        upgrade_model_file(str(path), eq.MLP(6, [8], 3), allow_unchecked=True)
    assert path.read_bytes() == before
    assert main([str(path), "--recipe", _MLP_RECIPE % "", "--no-check"]) == 1
    assert "does not reproduce the file's embedding" in capsys.readouterr().err
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy.eq"]


@pytest.mark.parametrize(
    "rows, message",
    [
        ({"x": torch.zeros(2, 6)}, "holds a dict"),
        (torch.tensor(1.0), r"holds a tensor of shape \(\)"),
        (torch.zeros(0, 6), r"holds a tensor of shape \(0, 6\)"),
        (b"not a torch file", "could not be read"),
        (None, "could not be read"),
    ],
    ids=["dict", "0-d", "no-rows", "not-torch", "missing"],
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_command_line_refuses_check_inputs_that_are_not_rows(
    tmp_path, capsys, rows, message
) -> None:
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    before = path.read_bytes()
    rows_path = tmp_path / "rows.pt"
    if isinstance(rows, bytes):
        rows_path.write_bytes(rows)
    elif rows is not None:
        torch.save(rows, rows_path)
    capsys.readouterr()  # training progress

    assert main([str(path), "--recipe", _RECIPE, "--check-inputs", str(rows_path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"error: --check-inputs '{rows_path}'")
    assert re.search(message, err)
    assert path.read_bytes() == before


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_compares_on_check_inputs_when_given(tmp_path) -> None:
    model, X = _trained_gp()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_gp_file(model, path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["support"] = None
    torch.save(checkpoint, path)

    @eq.embedding_architecture("equine.tests.tanh_variant_gp")
    class TanhVariant(BasicEmbeddingModel):
        def forward(self, x):
            return torch.tanh(super().forward(x))

    with pytest.raises(ValueError, match="does not reproduce"):
        upgrade_model_file(path, TanhVariant(6, 3), check_inputs=X[:16])
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as record:
        _warnings.simplefilter("always")
        _upgrade(path, check_inputs=X[:16])
    assert not any("not compared" in str(w.message) for w in record)


@pytest.mark.parametrize("via", ["api", "cli"])
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_of_a_float64_file(tmp_path, via) -> None:
    """A float64 legacy model upgrades under the default float32 dtype: the
    check casts the rows to each module's own dtype."""
    with _default_dtype(torch.float64):
        model, X = _trained_protonet()
        path = str(tmp_path / "legacy.eq")
        _write_legacy_protonet_file(model, path)
    if via == "api":
        _upgrade(path)
    else:
        assert main([path, "--recipe", _RECIPE]) == 0
    reloaded = eq.load_equine_model(path)
    a, b = model.predict(X[1:10]), reloaded.predict(X[1:10].float())
    assert torch.allclose(a.classes, b.classes.to(a.classes.dtype), atol=1e-5)
    assert torch.allclose(a.ood_scores, b.ood_scores.to(a.ood_scores.dtype), atol=1e-5)


class _Tokens(torch.nn.Module):
    """A legacy embedding over integer token ids (scriptable)."""

    def __init__(self, vocab: int, dim: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.out = torch.nn.Linear(dim, 3)

    def forward(self, x):
        return self.out(self.embed(x).mean(dim=1))


@eq.embedding_architecture("equine.tests.tokens")
class _RegisteredTokens(_Tokens):
    """The registered form of ``_Tokens``."""


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_of_a_model_over_integer_inputs(tmp_path) -> None:
    """Integer support rows reach both modules uncast."""
    torch.manual_seed(0)
    x = torch.randint(0, 50, (120, 5))
    y = torch.arange(3).repeat_interleave(40)
    dataset = torch.utils.data.TensorDataset(x, y)
    model = eq.EquineProtonet(_Tokens(50, 8), 3, ood_distance="rmd_sqrt")
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    assert all(
        not s.is_floating_point()
        for s in torch.load(path, weights_only=False)["support"].values()
    )

    upgrade_model_file(
        path, _RegisteredTokens(50, 8), check_inputs=torch.randint(0, 50, (4, 9))
    )

    _assert_same_predictions(model, eq.load_equine_model(path), x)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgraded_file_loads_without_torchscript(tmp_path, monkeypatch) -> None:
    model, X = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    _upgrade(path)

    def refuse(*args, **kwargs):
        raise AssertionError("torch.jit.load called")

    monkeypatch.setattr(torch.jit, "load", refuse)
    _assert_same_predictions(model, eq.load_equine_model(path), X)


@pytest.mark.parametrize("flag", [True, False, None], ids=["flag", "false", "no-flag"])
@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_format_3_file_with_an_archive_is_upgraded(
    tmp_path, train, write_legacy, flag
) -> None:
    """A format-3 file holding a TorchScript archive is upgraded only with the
    architecture, and the upgrade warns that reading it is deprecated."""
    model, X = train()
    path = str(tmp_path / "m.eq")
    checkpoint = _with_archive(model, path, 3)
    if flag is not None:
        checkpoint["contains_executable"] = flag
    torch.save(checkpoint, path)
    before = Path(path).read_bytes()
    copy = str(tmp_path / "copy.eq")
    shutil.copyfile(path, copy)

    with (
        pytest.warns(FutureWarning, match="deprecated"),
        pytest.raises(
            ValueError,
            match="embedding is TorchScript, so the upgrade needs the registered "
            r"architecture to rebuild it: pass embedding_model\.$",
        ),
    ):
        upgrade_model_file(path)
    assert Path(path).read_bytes() == before

    with pytest.warns(FutureWarning, match="deprecated"):
        assert _upgrade(path) == path
    assert main([copy, "--recipe", _RECIPE]) == 0

    for upgraded in (path, copy):
        checkpoint = load_checkpoint(upgraded)
        assert (
            "embed_jit_save" not in checkpoint
            and "contains_executable" not in checkpoint
        )
        _assert_same_predictions(model, eq.load_equine_model(upgraded), X)


def test_upgrade_command_line_recipe_and_import(tmp_path, capsys, monkeypatch) -> None:
    model, X = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    assert main([path]) == 1
    assert "--recipe is required" in capsys.readouterr().err
    assert main([path, "--recipe", "{not json"]) == 1
    assert "not valid JSON" in capsys.readouterr().err
    assert main([path, "--recipe", '{"builder": "nobody.registered"}']) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "--import" in err and "embedding_model=" not in err
    assert main([path, "--recipe", "[1, 2]"]) == 1
    assert '"builder" string' in capsys.readouterr().err
    assert main([path, "--recipe", _RECIPE, "--import", "no_such_module_xyz"]) == 1
    assert "Could not import" in capsys.readouterr().err

    (tmp_path / "my_arch.py").write_text(
        "import torch, equine\n"
        "@equine.embedding_architecture('my_arch.net')\n"
        "class Net(torch.nn.Module):\n"
        "    def __init__(self, tensor_dim: int, num_classes: int):\n"
        "        super().__init__()\n"
        "        self.linear_relu_stack = torch.nn.Sequential(\n"
        "            torch.nn.Linear(tensor_dim, 32), torch.nn.ReLU(),\n"
        "            torch.nn.Linear(32, 32), torch.nn.ReLU(),\n"
        "            torch.nn.Linear(32, num_classes))\n"
        "    def forward(self, x):\n"
        "        return self.linear_relu_stack(x)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    recipe = '{"builder": "my_arch.net", "kwargs": {"tensor_dim": 6, "num_classes": 3}}'
    assert main([path, "--recipe", recipe, "--import", "my_arch"]) == 0
    assert f"Upgraded {path} in place." in capsys.readouterr().out
    import my_arch  # noqa: F401  (registered by the CLI's import)

    _assert_same_predictions(model, eq.load_equine_model(path), X)


def _legacy_file_with_payload(tmp_path):
    """A legacy file whose unpickling creates a directory: its path and the
    directory's."""
    model, _ = _trained_protonet()
    path = tmp_path / "legacy.eq"
    _write_legacy_protonet_file(model, str(path))
    checkpoint = torch.load(path, weights_only=False)
    marker = tmp_path / "pwned"
    checkpoint["train_summary"]["payload"] = _MakesDirectoryWhenUnpickled(str(marker))
    torch.save(checkpoint, path)
    return str(path), marker


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_command_line_checks_its_arguments_before_unpickling(
    tmp_path, capsys
) -> None:
    path, marker = _legacy_file_with_payload(tmp_path)
    rows_marker = tmp_path / "pwned_by_rows"
    payload_rows = tmp_path / "payload_rows.pt"
    torch.save(_MakesDirectoryWhenUnpickled(str(rows_marker)), payload_rows)
    unbuildable = (
        '{"builder": "equine.mlp", "kwargs": {"in_features": 0, '
        '"hidden_sizes": [], "out_features": 3}}'
    )
    capsys.readouterr()  # training progress
    for args in (
        [],
        ["--recipe", "{bad"],
        ["--recipe", '{"builder": "nobody.registered"}'],
        ["--recipe", _RECIPE, "--import", "no_such_module_xyz"],
        ["--recipe", unbuildable],
        ["--recipe", _RECIPE, "--check-inputs", str(tmp_path / "missing.pt")],
        ["--recipe", _RECIPE, "--check-inputs", str(payload_rows)],
        ["--recipe", _RECIPE, "-o", str(tmp_path / "no_such_dir" / "out.eq")],
        ["--recipe", _RECIPE, "-o", str(tmp_path)],
        ["--recipe", _RECIPE, "-o", str(tmp_path / "no_such_dir") + os.sep],
        ["--recipe", _RECIPE, "-o", os.path.join(tmp_path, "no_such_dir", os.curdir)],
        ["--recipe", _RECIPE, "-o", os.path.join(tmp_path, "no_such_dir", os.pardir)],
    ):
        assert main([path, *args]) == 1
        assert capsys.readouterr().err.startswith("error: ")
        assert not marker.exists(), f"the file was unpickled for {args}"
    assert not rows_marker.exists()

    # A valid invocation unpickles the file: the payload is live.
    assert main([path, "--recipe", _RECIPE]) == 0
    assert marker.exists()


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod does not stop this user from writing to a directory",
)
@pytest.mark.parametrize("mode", [0o500, 0o600], ids=["read-only", "no-search"])
def test_upgrade_refuses_an_unwritable_output_directory_before_unpickling(
    tmp_path, capsys, mode
) -> None:
    path, marker = _legacy_file_with_payload(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, mode)
    try:
        capsys.readouterr()  # training progress
        assert main([path, "--recipe", _RECIPE, "-o", str(locked / "out.eq")]) == 1
        assert "cannot be written" in capsys.readouterr().err
        with pytest.raises(ValueError, match="cannot be written"):
            _upgrade(path, str(locked / "out.eq"))
    finally:
        os.chmod(locked, 0o700)
    assert not marker.exists()
    assert not list(locked.iterdir())


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.skipif(
    os.name == "nt", reason="Windows removes '..' from a path without looking it up"
)
def test_upgrade_refuses_an_output_through_a_missing_directory(
    tmp_path, capsys
) -> None:
    path, marker = _legacy_file_with_payload(tmp_path)
    output = os.path.join(tmp_path, "no_such_dir", os.pardir, "out.eq")
    capsys.readouterr()  # training progress
    assert main([path, "--recipe", _RECIPE, "-o", output]) == 1
    assert "is not an existing directory" in capsys.readouterr().err
    with pytest.raises(ValueError, match="is not an existing directory"):
        _upgrade(path, output)
    assert not marker.exists()


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_checks_its_arguments_before_unpickling(tmp_path) -> None:
    path, marker = _legacy_file_with_payload(tmp_path)

    class Plain(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="pass embedding_model"):
        upgrade_model_file(path)
    with pytest.raises(ValueError, match="embedding_architecture"):
        upgrade_model_file(path, Plain())
    with pytest.raises(ValueError, match="check_inputs must be a tensor"):
        _upgrade(path, check_inputs=torch.zeros(0, 6))
    with pytest.raises(ValueError, match="no_such_dir' is not an existing directory"):
        _upgrade(path, str(tmp_path / "no_such_dir" / "out.eq"))
    for output in (
        str(tmp_path),
        str(tmp_path / "no_such_dir") + os.sep,
        os.path.join(tmp_path, "no_such_dir", os.curdir),
        os.path.join(tmp_path, "no_such_dir", os.pardir),
    ):
        with pytest.raises(ValueError, match="names a directory"):
            _upgrade(path, output)
    assert not marker.exists()

    _upgrade(path)
    assert marker.exists()


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_of_a_current_file_needs_no_recipe(tmp_path, capsys) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "current.eq")
    model.save(path)
    before = Path(path).read_bytes()

    assert main([path, "--recipe", "{bad", "--check-inputs", "missing.pt"]) == 0
    assert "nothing written" in capsys.readouterr().out
    assert Path(path).read_bytes() == before


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_names_shape_and_layer_mismatches(tmp_path, capsys) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    with pytest.raises(
        ValueError, match="different shapes: linear_relu_stack.4.weight"
    ):
        upgrade_model_file(path, BasicEmbeddingModel(6, 4))
    wrong = '{"builder": "equine.tests.basic", "kwargs": {"tensor_dim": 6, "num_classes": 4}}'
    assert main([path, "--recipe", wrong]) == 1
    assert "error:" in capsys.readouterr().err

    @eq.embedding_architecture("equine.tests.shorter")
    class Shorter(torch.nn.Module):
        def __init__(self, tensor_dim: int, num_classes: int) -> None:
            super().__init__()
            self.linear_relu_stack = torch.nn.Sequential(
                torch.nn.Linear(tensor_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
            )

        def forward(self, x):
            return self.linear_relu_stack(x)

    with pytest.raises(ValueError, match="not in the supplied architecture"):
        upgrade_model_file(path, Shorter(6, 3))


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_reports_a_comparison_that_cannot_run(tmp_path) -> None:
    _, X, path = _gp_file_without_support(tmp_path)
    before = Path(path).read_bytes()
    with pytest.raises(ValueError, match="could not compare"):
        _upgrade(path, check_inputs=X[:16, :5])  # wrong width for the embedding
    assert Path(path).read_bytes() == before


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_reports_an_output_of_another_shape(tmp_path) -> None:
    @eq.embedding_architecture("equine.tests.narrow")
    class Narrow(BasicEmbeddingModel):
        def forward(self, x):
            return super().forward(x)[:, :2]

    _, X, path = _gp_file_without_support(tmp_path)
    with pytest.raises(ValueError, match="output shape"):
        upgrade_model_file(path, Narrow(6, 3), check_inputs=X[:16])


@pytest.mark.parametrize(
    "bad",
    [torch.tensor(1.0), torch.zeros(0, 6), [[0.0] * 6]],
    ids=["0-d", "no-rows", "list"],
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_refuses_check_inputs_that_are_not_rows(tmp_path, bad) -> None:
    _, _, path = _gp_file_without_support(tmp_path)
    with pytest.raises(ValueError, match="check_inputs must be a tensor"):
        _upgrade(path, check_inputs=bad)


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
    equine._model_file.write(
        path,
        {
            "a": base,
            "view": base[10:20],
            "equine_format_version": EQUINE_FORMAT_VERSION,
        },
    )
    loaded = load_checkpoint(path, map_location=_OTHER_DEVICE)
    tensors = [v for v in loaded.values() if isinstance(v, torch.Tensor)]
    assert {t.device.type for t in tensors} == {_OTHER_DEVICE}
    assert torch.equal(loaded["view"].cpu(), torch.arange(10.0, 20.0))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgraded_legacy_file_loads_onto_mps_and_predicts(
    tmp_path, train, write_legacy
) -> None:
    """A legacy file reaches an accelerator through the upgrade:
    every tensor lands on the device, and the model predicts there."""
    model, X = train()  # float32, trained on the CPU
    path = str(tmp_path / "legacy.eq")
    write_legacy(model, path)
    _upgrade(path)

    checkpoint = load_checkpoint(path, map_location="mps")
    tensors = equine.utils._unique_tensors(checkpoint)
    assert tensors and {t.device.type for t in tensors} == {"mps"}

    expected = model.predict(X[1:10])
    for loader in (type(model).load, eq.load_equine_model):
        loaded = loader(path, device="mps")
        assert loaded.device == "mps"
        assert_on_device(loaded, "mps")
        actual = loaded.predict(X[1:10])
        torch.testing.assert_close(
            actual.classes.cpu(), expected.classes, atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            actual.ood_scores.cpu(), expected.ood_scores, atol=1e-4, rtol=0
        )


def test_archive_bytes_handles_offset_view() -> None:
    """A uint8 tensor that is an offset, non-full view of a larger storage
    must convert to exactly its own bytes, not the whole backing storage."""
    from equine.upgrade import _archive_bytes

    padded = torch.arange(20, dtype=torch.uint8)
    view = padded[5:10]
    assert view.storage_offset() != 0

    assert _archive_bytes(view) == bytes(range(5, 10))


def test_archive_bytes_is_fast_for_large_tensors() -> None:
    """`_archive_bytes` must convert a multi-megabyte archive quickly: a loop
    over the storage one Python int at a time takes ~1 s/MB, far above the
    0.5 s bound for this 4 MB tensor."""
    import time

    from equine.upgrade import _archive_bytes

    tensor = torch.randint(0, 256, (4_000_000,), dtype=torch.uint8)

    start = time.perf_counter()
    result = _archive_bytes(tensor)
    elapsed = time.perf_counter() - start

    assert result == tensor.numpy().tobytes()
    assert elapsed < 0.5


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_refuses_a_flagged_file_without_an_archive(
    tmp_path, monkeypatch
) -> None:
    """A format-3 file flagged as holding TorchScript, without an archive, is
    refused from the safe read."""
    model, _ = _trained_protonet()
    path = str(tmp_path / "m.eq")
    checkpoint = _in_format(model, path, 3)
    checkpoint["contains_executable"] = True
    torch.save(checkpoint, path)
    before = Path(path).read_bytes()
    monkeypatch.setattr(torch, "load", _weights_only_load(torch.load))
    with pytest.raises(ValueError, match="holds no TorchScript archive") as err:
        _upgrade(path)
    assert "was not written by an EQUINE version the upgrade converts" in str(err.value)
    assert Path(path).read_bytes() == before


@pytest.mark.parametrize(
    "archive, message",
    [
        (io.BytesIO(b"not a TorchScript archive"), "archive could not be loaded"),
        (42, "is not a TorchScript archive"),
    ],
    ids=["unreadable", "not-an-archive"],
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_reports_an_archive_it_cannot_load(tmp_path, archive, message) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["embed_jit_save"] = archive
    torch.save(checkpoint, path)
    before = Path(path).read_bytes()
    with pytest.raises(ValueError, match=message):
        _upgrade(path)
    assert Path(path).read_bytes() == before


class _DropoutEmbedding(torch.nn.Module):
    """A legacy embedding whose output depends on the mode (scriptable)."""

    def __init__(self, tensor_dim: int, num_classes: int) -> None:
        super().__init__()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(tensor_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.stack(x)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_compares_in_eval_mode(tmp_path) -> None:
    """An archive saved in training mode still compares equal in eval mode."""

    @eq.embedding_architecture("equine.tests.dropout")
    class Registered(_DropoutEmbedding):
        pass

    dataset, X = _tiny_dataset()
    model = eq.EquineProtonet(_DropoutEmbedding(6, 3), 3, ood_distance="rmd_sqrt")
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    model.train()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    upgrade_model_file(path, Registered(6, 3))

    model.eval()
    _assert_same_predictions(model, eq.load_equine_model(path), X)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_without_torchscript_in_torch(tmp_path, monkeypatch) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    before = Path(path).read_bytes()
    monkeypatch.delattr(torch.jit, "load")
    with pytest.raises(ValueError, match="has no torch.jit.load"):
        _upgrade(path)
    assert Path(path).read_bytes() == before


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_refuses_support_rows_without_a_row_dimension(tmp_path) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["support"][0] = torch.tensor(1.0)
    torch.save(checkpoint, path)
    before = Path(path).read_bytes()
    with pytest.raises(ValueError):
        _upgrade(path)
    assert Path(path).read_bytes() == before


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_reports_an_architecture_the_model_cannot_use(tmp_path) -> None:
    """An embedding whose output the model cannot use fails the reload as a
    ValueError, not as a traceback."""

    @eq.embedding_architecture("equine.tests.integer_output")
    class IntegerOutput(BasicEmbeddingModel):
        def forward(self, x):
            return super().forward(x).long()

    model, _ = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    before = Path(path).read_bytes()
    with pytest.raises(
        ValueError,
        match=r"rewritten file does not load: .*through its embedding model "
        r"\([A-Za-z]+Error: .*\)\. '",
    ) as err:
        upgrade_model_file(path, IntegerOutput(6, 3))
    assert "tampered" not in str(err.value)
    assert Path(path).read_bytes() == before


def _train_subclass_protonet():
    class MyProtonet(eq.EquineProtonet):
        pass

    dataset, X = _tiny_dataset()
    model = MyProtonet(BasicEmbeddingModel(6, 3), 3, ood_distance="rmd_sqrt")
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    return model, X


def _train_subclass_gp():
    class MyGP(eq.EquineGP):
        pass

    dataset, X = _tiny_dataset()
    model = MyGP(BasicEmbeddingModel(6, 3), 3, 3, num_random_features=16)
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.001),
        num_epochs=1,
        batch_size=32,
        vis_support=True,
    )
    return model, X


@pytest.mark.parametrize(
    "train, write_legacy",
    [
        (_train_subclass_protonet, _write_legacy_protonet_file),
        (_train_subclass_gp, _write_legacy_gp_file),
    ],
    ids=["protonet", "gp"],
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_of_a_file_saved_from_a_subclass(tmp_path, train, write_legacy) -> None:
    model, X = train()
    assert model.train_summary["modelType"] == type(model).__name__
    path = str(tmp_path / "legacy.eq")
    write_legacy(model, path)

    _upgrade(path)

    reloaded = type(model).load(path)
    assert type(reloaded) is type(model)
    _assert_same_predictions(model, reloaded, X)


@pytest.mark.parametrize(
    "err, expected",
    [
        (RuntimeError(), "RuntimeError."),
        (RuntimeError("bad shape"), "RuntimeError: bad shape."),
        (ValueError("bad shape at '/tmp/.m.upgrading'"), "bad shape at 'm.eq'."),
        (ValueError("Bad file. Rewrite it once with `python -m ...`"), "Bad file."),
        (
            ValueError(
                "Bad support (x); the file is inconsistent or was tampered with."
            ),
            "Bad support (x).",
        ),
        (
            ValueError(
                "Model file's header is malformed (x); the file was not written by "
                "save() or was changed after. Save the model again, or upgrade the "
                "original file again with `python -m equine.upgrade`."
            ),
            "Model file's header is malformed (x).",
        ),
    ],
    ids=[
        "no-message",
        "other-type",
        "temp-path",
        "upgrade-hint",
        "tampering",
        "malformed",
    ],
)
def test_upgrade_error_reasons_read_as_one_sentence(err, expected) -> None:
    from equine.upgrade import _reason

    assert _reason(err, "/tmp/.m.upgrading", "m.eq") == expected


def test_upgrade_error_reasons_name_a_quoted_temp_path_as_the_target() -> None:
    from equine.upgrade import _reason

    partial = "C:\\models\\.m.eq.0123.upgrading"
    path = "C:\\models\\m.eq"
    err = FileNotFoundError(2, "No such file or directory", partial)

    assert _reason(err, partial, path) == (
        f"FileNotFoundError: [Errno 2] No such file or directory: {path!r}."
    )


def test_upgrade_error_reasons_name_a_plainly_quoted_temp_path_as_the_target() -> None:
    from equine.upgrade import _reason

    partial = "/out/.o.eq.0123.upgrading"
    path = "/src/back\\slash/m.eq"
    err = ValueError(f"'{partial}' is not an EQUINE model file")

    assert _reason(err, partial, path) == f"'{path}' is not an EQUINE model file."


def test_upgrade_error_reasons_name_an_escaped_temp_path_both_ways_quoted() -> None:
    from equine.upgrade import _reason

    partial = "C:\\models\\.m.eq.0123.upgrading"
    path = "C:\\models\\m.eq"
    err = ValueError(f"'{partial}' holds {partial!r}")

    assert _reason(err, partial, path) == f"'{path}' holds {path!r}."


def test_upgrade_reads_every_stored_archive_form() -> None:
    """The archive loader reads the uint8 tensor that format-2 files store, the
    BytesIO that legacy files store, and bytes."""
    from equine.upgrade import _load_archive

    module = BasicEmbeddingModel(6, 3)
    buffer = _jit_buffer(module)
    x = torch.randn(4, 6)
    tensor = torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)
    for archive in (tensor, buffer.getvalue(), buffer):
        rebuilt = _load_archive(archive)
        assert torch.allclose(rebuilt(x), module(x))


# ---------------------------------------------------------------------------
# The torch floor (weights_only=True is bypassable before torch 2.6)
# ---------------------------------------------------------------------------


def test_declared_torch_floor_is_at_least_2_6() -> None:
    import tomllib

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


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_keeps_the_support_order_of_a_legacy_file(tmp_path) -> None:
    model, X = _trained_protonet()
    support = model.get_support()
    model.model.update_support(
        collections.OrderedDict((label, support[label]) for label in (2, 0, 1))
    )
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    assert _upgrade(path) == path

    loaded = eq.load_equine_model(path)
    assert list(loaded.get_support()) == [2, 0, 1]
    _assert_same_predictions(model, loaded, X)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgraded_protonet_file_has_no_ood_distance_and_loads_as_rmd_sqrt(
    tmp_path,
) -> None:
    model, X = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    _upgrade(path)

    checkpoint = load_checkpoint(path)
    assert checkpoint["equine_format_version"] == EQUINE_FORMAT_VERSION
    assert "ood_distance" not in checkpoint["settings"]
    upgraded = eq.load_equine_model(path)
    assert upgraded.ood_distance == "rmd_sqrt"
    _assert_same_predictions(model, upgraded, X)


# ---------------------------------------------------------------------------
# Files from development builds: PyTorch archives of format 2 or 3
# ---------------------------------------------------------------------------


def _weights_only_load(real_load):
    """``torch.load`` that fails the test when called with ``weights_only=False``."""

    def load(*args, **kwargs):
        if kwargs.get("weights_only") is False:
            pytest.fail("torch.load ran with weights_only=False")
        return real_load(*args, **kwargs)

    return load


def _in_format(model, path: str, version: int | None) -> dict:
    """``model``'s saved checkpoint declaring format ``version`` (no version for
    None), for the caller to write with ``torch.save``."""
    model.save(path)
    checkpoint = load_checkpoint(path)
    if version is None:
        del checkpoint["equine_format_version"]
    else:
        checkpoint["equine_format_version"] = version
    return checkpoint


def _with_archive(model, path: str, version: int | None) -> dict:
    """``_in_format``'s checkpoint with its embedding as a TorchScript archive."""
    checkpoint = _in_format(model, path, version)
    del checkpoint["embedding_recipe"], checkpoint["embedding_state_dict"]
    checkpoint["embed_jit_save"] = archive_tensor(
        model.model.embedding_model
        if isinstance(model, eq.EquineProtonet)
        else model.model.feature_extractor
    )
    return checkpoint


def _kde_datasets_of_one_tensor(kdes: dict) -> dict:
    """Each label's KDE state with its ``dataset`` a view of one shared tensor."""
    base = torch.cat([kde["dataset"] for kde in kdes.values()], dim=1)
    out, start = {}, 0
    for label, kde in kdes.items():
        width = kde["dataset"].shape[1]
        out[label] = {**kde, "dataset": base[:, start : start + width]}
        start += width
    return out


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_converts_a_file_without_a_version_whose_support_is_aliased(
    tmp_path, train, write_legacy
) -> None:
    model, X = train()
    path = str(tmp_path / "m.eq")
    checkpoint = _with_archive(model, path, None)
    checkpoint["support"] = _slices_of_one_tensor(checkpoint["support"])
    torch.save(checkpoint, path)
    with pytest.raises(LegacyFormatError):
        load_checkpoint(path)

    assert _upgrade(path) == path

    _assert_same_predictions(model, eq.load_equine_model(path), X)


_FORMAT_2_ALIASING = [
    pytest.param(_trained_protonet, "support", id="protonet-support"),
    pytest.param(_trained_gp, "support", id="gp-support"),
    pytest.param(_trained_protonet, "outlier_kde", id="protonet-kde"),
]


@pytest.mark.parametrize("train, aliased", _FORMAT_2_ALIASING)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_format_2_file_with_aliased_data_is_upgraded_from_the_safe_read(
    tmp_path, monkeypatch, train, aliased
) -> None:
    """A format-2 file holding a TorchScript archive whose support or KDE data are
    views of one tensor: the upgrade reads the file without unrestricted unpickling
    and writes each label's data."""
    model, X = train()
    path = str(tmp_path / "m.eq")
    checkpoint = _with_archive(model, path, 2)
    if aliased == "support":
        original = dict(checkpoint["support"])
        checkpoint["support"] = _slices_of_one_tensor(checkpoint["support"])
    else:
        original = {k: v["dataset"] for k, v in checkpoint["outlier_kde"].items()}
        checkpoint["outlier_kde"] = _kde_datasets_of_one_tensor(
            checkpoint["outlier_kde"]
        )
    torch.save(checkpoint, path)

    monkeypatch.setattr(torch, "load", _weights_only_load(torch.load))
    assert _upgrade(path) == path

    stored = load_checkpoint(path)[aliased]
    rows = {k: v if aliased == "support" else v["dataset"] for k, v in stored.items()}
    assert all(torch.equal(rows[k], original[k]) for k in original)
    _assert_same_predictions(model, eq.load_equine_model(path), X)


@pytest.mark.parametrize("via", ["api", "cli"])
@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize(
    "train", [_trained_protonet, _trained_gp], ids=["protonet", "gp"]
)
def test_data_only_file_of_format_2_or_3_is_converted_without_the_architecture(
    tmp_path, capsys, train, version, via
) -> None:
    """A file whose embedding is a recipe and weights is rewritten with the same
    contents, without embedding_model or --recipe and without a deprecation warning:
    in place through the API, to -o through the command line. A format-2 file's
    ``contains_executable`` entry is dropped."""
    model, X = train()
    path = str(tmp_path / "m.eq")
    checkpoint = _in_format(model, path, version)
    if version == 2:
        checkpoint["contains_executable"] = False
    torch.save(checkpoint, path)
    before = Path(path).read_bytes()
    out = str(tmp_path / "out.eq")
    capsys.readouterr()  # training progress

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if via == "api":
            assert upgrade_model_file(path) == path
            out = path
        else:
            assert main([path, "-o", out]) == 0
            assert Path(path).read_bytes() == before
    assert capsys.readouterr() == (
        "" if via == "api" else f"Upgraded {path} -> {out}\n",
        "",
    )

    expected = {k: v for k, v in checkpoint.items() if k != "contains_executable"}
    expected["equine_format_version"] = EQUINE_FORMAT_VERSION
    tensors, header = equine._model_file.encode(expected)
    written_tensors, written_header = equine._model_file.encode(load_checkpoint(out))
    assert written_header == header
    assert written_tensors.keys() == tensors.keys()
    for name, tensor in tensors.items():
        assert written_tensors[name].dtype == tensor.dtype
        assert torch.equal(written_tensors[name], tensor)
    expected_output = model.predict(X[1:10])
    output = eq.load_equine_model(out).predict(X[1:10])
    assert torch.equal(output.classes, expected_output.classes)
    assert torch.equal(output.ood_scores, expected_output.ood_scores)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_upgrade_names_what_a_file_without_a_train_summary_lacks(tmp_path) -> None:
    path = str(tmp_path / "x.pt")
    torch.save({"embed_jit_save": torch.zeros(3, dtype=torch.uint8)}, path)
    with pytest.raises(
        ValueError,
        match="is not an EQUINE model file: every EQUINE model file is a dictionary "
        "with a train_summary",
    ):
        _upgrade(path)
