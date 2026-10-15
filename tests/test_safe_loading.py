# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Tests that EQUINE model files are read without unrestricted unpickling, so
that opening an untrusted file cannot run pickle payloads.
"""

import io
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import BasicEmbeddingModel

import equine as eq
import equine.utils
from equine.upgrade import _current_layout, main, upgrade_model_file
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
                "device": model.device_type,
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
    archive = checkpoint["embed_jit_save"]
    assert isinstance(archive, torch.Tensor) and archive.dtype == torch.uint8
    assert checkpoint["equine_format_version"] == 2
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


def test_file_without_format_version_is_refused(tmp_path) -> None:
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


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
def test_upgrade_rewrites_legacy_file_in_the_current_format(
    tmp_path, train, write_legacy
) -> None:
    model, X = train()
    legacy_path = str(tmp_path / "legacy.eq")
    write_legacy(model, legacy_path)
    out_path = str(tmp_path / "upgraded.eq")

    with pytest.warns(FutureWarning, match="deprecated"):
        written = upgrade_model_file(legacy_path, out_path)

    assert written == out_path
    _assert_weights_only_safe_layout(out_path)
    reloaded = eq.load_equine_model(out_path)
    assert type(reloaded) is type(model)
    _assert_same_predictions(model, reloaded, X)


def test_upgrade_replaces_the_file_in_place_by_default(tmp_path) -> None:
    model, X = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)

    with pytest.warns(FutureWarning):
        assert upgrade_model_file(path) == path

    _assert_weights_only_safe_layout(path)
    _assert_same_predictions(model, eq.load_equine_model(path), X)


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
    reloaded = eq.load_equine_model(path)
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
    _assert_same_predictions(model, eq.load_equine_model("old2.eq"), X)


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
    _assert_same_predictions(model, eq.load_equine_model(path), X)


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
    _assert_same_predictions(model, eq.load_equine_model(str(link)), X)


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
    _assert_same_predictions(model, eq.load_equine_model(path), X)

    # The warning names the file whose bits were read: the one being replaced.
    other = str(tmp_path / "other.eq")
    _write_legacy_protonet_file(model, other)
    existing = str(tmp_path / "existing.eq")
    _write_legacy_protonet_file(model, existing)
    with pytest.warns(UserWarning, match="permission bits of '.*existing.eq'"):
        upgrade_model_file(other, existing)


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


def test_upgrade_command_line(tmp_path, capsys) -> None:
    model, X = _trained_gp()
    legacy_path = str(tmp_path / "legacy.eq")
    _write_legacy_gp_file(model, legacy_path)
    out_path = str(tmp_path / "upgraded.eq")

    assert main([legacy_path, "-o", out_path]) == 0
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

    in_place = str(tmp_path / "legacy2.eq")
    _write_legacy_gp_file(model, in_place)
    assert main([in_place]) == 0
    assert f"Upgraded {in_place} in place." in capsys.readouterr().out
    _assert_same_predictions(model, eq.load_equine_model(in_place), X)

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


def test_current_layout_converts_every_legacy_value() -> None:
    """The converter turns the objects legacy files hold into plain values:
    the archive (BytesIO or bytes) into a uint8 tensor, the CovType into its
    value, KDE objects into their state, and names into plain strings."""
    module = BasicEmbeddingModel(6, 3)
    buffer = _jit_buffer(module)
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(np.random.default_rng(0).normal(size=20))
    for archive in (buffer, buffer.getvalue()):
        out = _current_layout(
            {
                "embed_jit_save": archive,
                "settings": {"cov_type": eq.CovType.DIAGONAL},
                "support": {torch.tensor(1): torch.zeros(2, 6)},
                "outlier_kde": {1: kde},
                "label_names": [np.str_("a"), np.str_("b")],
                "train_summary": {},
            }
        )
        assert out["equine_format_version"] == EQUINE_FORMAT_VERSION
        assert out["embed_jit_save"].dtype == torch.uint8
        assert out["settings"]["cov_type"] == "diag"
        assert list(out["support"]) == [1]
        assert out["outlier_kde"][1]["bw_factor"] == pytest.approx(kde.factor)
        assert out["label_names"] == ["a", "b"] and all(
            type(n) is str for n in out["label_names"]
        )
        torch.save(out, io.BytesIO())  # weights_only-safe values only


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


def test_jit_archive_tensor_round_trips() -> None:
    """`jit_archive_to_tensor` followed by `load_jit_archive` must reconstruct
    a scripted module whose outputs match the module that was saved."""
    from equine.utils import jit_archive_to_tensor, load_jit_archive

    module = BasicEmbeddingModel(6, 3)
    buffer = _jit_buffer(module)
    x = torch.rand(4, 6)
    expected = module(x)

    tensor = jit_archive_to_tensor(buffer)
    rebuilt = load_jit_archive(tensor)
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

    result = load_checkpoint(str(path))
    assert result["a"] == 1


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
