# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
The file-reading layer (``equine.utils.load_checkpoint``) and the safe
format (issue #168): model files are read without unrestricted unpickling, so
opening an untrusted file cannot run pickle payloads, and a crafted file cannot
make the reader allocate more than the file holds. Legacy files need an explicit
opt-in. The model layer on top (recipes, save/load) is tested in
test_persistence.py.
"""

import io
import os
import re
import struct
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import BasicEmbeddingModel, _rewrite_zip

import equine as eq
import equine.utils
from equine.utils import load_checkpoint, prepare_jit_module


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


def _load_without_unsafe_warning(fn):
    """Run `fn()` and assert EQUINE's unsafe-load warning was not raised.

    Uses record-and-assert rather than `warnings.simplefilter("error")` so
    that unrelated warnings torch itself may emit on other torch versions
    (e.g. TypedStorage deprecation on torch 2.0, a `torch.jit.load`
    FutureWarning on torch 2.14) don't fail the test.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = fn()
    unsafe = [
        w
        for w in record
        if issubclass(w.category, UserWarning) and "unsafe" in str(w.message)
    ]
    assert not unsafe, "safe load path must not raise EQUINE's unsafe-load warning"
    return result


def _write_legacy_protonet_file(model, path: str) -> None:
    """Reproduce the pre-#168 on-disk layout: BytesIO, Enum and scipy objects."""
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
    """Reproduce the pre-#168 EquineGP layout: BytesIO buffer and raw OrderedDicts."""
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

    with pytest.raises(ValueError, match="safely"):
        loader(str(path))

    assert not marker.exists(), "unpickling the model file executed its payload"


# ---------------------------------------------------------------------------
# Files written by save() must be loadable with torch.load(weights_only=True)
# ---------------------------------------------------------------------------


def _assert_weights_only_safe_layout(path: str) -> dict:
    checkpoint = torch.load(path, weights_only=True)  # must not raise
    assert checkpoint["equine_format_version"] == 2
    if checkpoint["contains_executable"]:
        # transition layout: the TorchScript archive travels as a uint8 tensor
        archive = checkpoint["embed_jit_save"]
        assert isinstance(archive, torch.Tensor) and archive.dtype == torch.uint8
    else:
        assert isinstance(checkpoint["embedding_recipe"]["builder"], str)
        assert "embedding_state_dict" in checkpoint
        assert "embed_jit_save" not in checkpoint
    return checkpoint


def test_protonet_save_format_is_weights_only_safe(tmp_path) -> None:
    model, X = _trained_protonet(cov_type=eq.CovType.DIAGONAL, use_temperature=True)
    # A non-default bandwidth on one class makes the test sensitive to the
    # bandwidth factor being dropped or misapplied on reload.
    model.outlier_score_kde[1].set_bandwidth(0.3)
    path = str(tmp_path / "protonet.eq")
    model.save(path)

    checkpoint = _assert_weights_only_safe_layout(path)
    assert checkpoint["settings"]["cov_type"] == "diag"

    reloaded = _load_without_unsafe_warning(lambda: eq.load_equine_model(path))

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

    reloaded = _load_without_unsafe_warning(lambda: eq.load_equine_model(path))

    assert isinstance(reloaded, eq.EquineGP)
    assert list(reloaded.support.keys()) == list(model.support.keys())
    _assert_same_predictions(model, reloaded, X)


# ---------------------------------------------------------------------------
# Legacy files (pre-safe format) need an explicit opt-in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
def test_legacy_file_is_rejected_by_default(tmp_path, train, write_legacy) -> None:
    model, _ = train()
    path = str(tmp_path / "legacy.eq")
    write_legacy(model, path)

    with pytest.raises(ValueError, match="allow_unsafe_legacy_format"):
        eq.load_equine_model(path)
    with pytest.raises(ValueError, match="allow_unsafe_legacy_format"):
        type(model).load(path)


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
# The transition's FutureWarning is asserted in its own test.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_legacy_file_loads_with_opt_in_and_resaves_safely(
    tmp_path, train, write_legacy
) -> None:
    model, X = train()
    legacy_path = str(tmp_path / "legacy.eq")
    write_legacy(model, legacy_path)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        reloaded = eq.load_equine_model(legacy_path, allow_unsafe_legacy_format=True)
    unsafe = [w for w in record if issubclass(w.category, UserWarning)]
    assert len(unsafe) == 1, "exactly one unsafe-load warning per load"
    assert "unsafe" in str(unsafe[0].message)
    assert unsafe[0].filename == __file__, "warning must point at the caller"
    _assert_same_predictions(model, reloaded, X)

    # A legacy file carries a TorchScript embedding. Re-saving keeps it
    # executable (flagged, trust required); migrating through a registered
    # class yields a data-only file.
    flagged = str(tmp_path / "resaved.eq")
    reloaded.save(flagged, allow_executable=True)
    assert _assert_weights_only_safe_layout(flagged)["contains_executable"] is True
    _assert_same_predictions(
        model, eq.load_equine_model(flagged, trust_executable=True), X
    )

    with pytest.warns(UserWarning, match="unsafe"):
        migrated = eq.load_equine_model(
            legacy_path,
            allow_unsafe_legacy_format=True,
            embedding_model=BasicEmbeddingModel(6, 3),
        )
    data_only = str(tmp_path / "migrated.eq")
    migrated.save(data_only)
    assert _assert_weights_only_safe_layout(data_only)["contains_executable"] is False
    _assert_same_predictions(model, eq.load_equine_model(data_only), X)


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
def test_class_load_opt_in_warning_points_at_caller(
    tmp_path, train, write_legacy
) -> None:
    model, _ = train()
    path = str(tmp_path / "legacy.eq")
    write_legacy(model, path)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        type(model).load(path, allow_unsafe_legacy_format=True)
    unsafe = [w for w in record if issubclass(w.category, UserWarning)]
    assert len(unsafe) == 1
    assert unsafe[0].filename == __file__


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
# The transition's FutureWarning is asserted in its own test.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_bare_legacy_load_points_at_the_embedding_model_migration(
    tmp_path, train, write_legacy
) -> None:
    """Without embedding_model= a legacy file's embedding is TorchScript, which
    save() refuses; the warning and the refusal both name the supported route."""
    model, _ = train()
    path = str(tmp_path / "legacy.eq")
    write_legacy(model, path)

    with pytest.warns(UserWarning, match="unsafe") as record:
        reloaded = type(model).load(path, allow_unsafe_legacy_format=True)
    unsafe = [
        w
        for w in record
        if issubclass(w.category, UserWarning)
        and "unrestricted unpickling" in str(w.message)
    ]
    assert len(unsafe) == 1
    assert "embedding_model=" in str(unsafe[0].message)

    with pytest.raises(ValueError, match="embedding_model=") as err:
        reloaded.save(str(tmp_path / "resaved.eq"))
    assert "allow_unsafe_legacy_format=True" in str(err.value)
    assert not (tmp_path / "resaved.eq").exists()


_LINEAR_HEAD = {"weight": torch.eye(3), "bias": torch.zeros(3)}


def test_legacy_linear_head_weights_are_dropped_with_a_warning(tmp_path) -> None:
    """EQUINE <= 0.1.6 saved a Linear head; the head has been Identity since #146,
    so a trusted legacy file loads with those weights ignored."""
    model, X = _trained_protonet()
    path = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["model_head_save"] = dict(_LINEAR_HEAD)
    torch.save(checkpoint, path)

    for loader in (eq.EquineProtonet.load, eq.load_equine_model):
        with pytest.warns(UserWarning) as record:
            reloaded = loader(
                path,
                allow_unsafe_legacy_format=True,
                embedding_model=BasicEmbeddingModel(6, 3),
            )
        messages = [str(w.message) for w in record]
        head = [m for m in messages if "head weights" in m]
        assert len(head) == 1
        assert "EQUINE <= 0.1.6 are ignored" in head[0]
        assert "#146" in head[0]
        _assert_same_predictions(model, reloaded, X)


def test_linear_head_weights_without_the_opt_in_name_the_old_version(
    tmp_path,
) -> None:
    model, _ = _trained_protonet()
    path = str(tmp_path / "m.eq")
    model.save(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["model_head_save"] = dict(_LINEAR_HEAD)
    torch.save(checkpoint, path)
    for loader in (eq.EquineProtonet.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="0.1.6") as err:
            loader(path)
        assert "allow_unsafe_legacy_format=True" in str(err.value)
        assert "tampered" not in str(err.value)


def _slices_of_one_tensor(support):
    """The same support rows, re-stored as views of a single tensor."""
    base = torch.cat(list(support.values()))
    out, start = type(support)(), 0
    for label, rows in support.items():
        out[label] = base[start : start + len(rows)]
        start += len(rows)
    return out


@pytest.mark.parametrize("train, write_legacy", LEGACY_CASES)
def test_legacy_file_with_aliased_support_migrates(
    tmp_path, train, write_legacy
) -> None:
    """Earlier releases saved support tensors as they were in memory, which may
    share one storage. The trusted legacy path copies them apart instead of
    refusing; the safe path still refuses (see _aliasing_checkpoint)."""
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

    with pytest.warns(UserWarning, match="unsafe"):
        migrated = eq.load_equine_model(
            legacy_path,
            allow_unsafe_legacy_format=True,
            embedding_model=BasicEmbeddingModel(6, 3),
        )
    _assert_same_predictions(model, migrated, X)
    data_only = str(tmp_path / "migrated.eq")
    migrated.save(data_only)
    assert _assert_weights_only_safe_layout(data_only)["contains_executable"] is False
    _assert_same_predictions(
        model, _load_without_unsafe_warning(lambda: eq.load_equine_model(data_only)), X
    )


def test_legacy_path_loads_onto_the_cpu_unless_mapped(tmp_path, monkeypatch) -> None:
    """Both paths honour "CPU unless map_location": the legacy torch.load must
    not restore the devices the tensors were saved from."""
    path = str(tmp_path / "old_format.pt")
    torch.save({"a": torch.arange(4.0)}, path, _use_new_zipfile_serialization=False)
    locations: list = []
    real_load = torch.load

    def spy(*args, **kwargs):
        locations.append(kwargs.get("map_location"))
        return real_load(*args, **{**kwargs, "map_location": "cpu"})

    monkeypatch.setattr(torch, "load", spy)
    with pytest.warns(UserWarning, match="unsafe"):
        load_checkpoint(path, allow_unsafe_legacy_format=True)
    with pytest.warns(UserWarning, match="unsafe"):
        load_checkpoint(path, map_location="cuda:1", allow_unsafe_legacy_format=True)
    assert locations == ["cpu", "cuda:1"]


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


def test_opt_in_still_uses_safe_path_for_safe_files(tmp_path) -> None:
    """The flag is a fallback, not a switch: a safe-format file is never unpickled
    without restrictions, so it must load silently even when the flag is set."""
    model, X = _trained_protonet()
    path = str(tmp_path / "safe.eq")
    model.save(path)

    reloaded = _load_without_unsafe_warning(
        lambda: eq.load_equine_model(path, allow_unsafe_legacy_format=True)
    )
    _assert_same_predictions(model, reloaded, X)


def test_opt_in_does_not_bypass_safe_path_for_safe_layout_with_payload(
    tmp_path,
) -> None:
    """A file that declares the safe format but smuggles a payload is not an
    EQUINE model; with the flag set it falls back to unrestricted unpickling by
    design, so callers must only set the flag for files they trust. This test
    pins that the *default* (flag off) never reaches the payload."""
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.eq"
    torch.save(
        {
            "equine_format_version": 2,
            "train_summary": {"modelType": "EquineProtonet"},
            "payload": _MakesDirectoryWhenUnpickled(str(marker)),
        },
        path,
    )
    with pytest.raises(ValueError, match="safely"):
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
    }
    path = tmp_path / "honest.pt"
    torch.save(data, path)
    loaded = _load_without_unsafe_warning(lambda: load_checkpoint(str(path)))
    for key in ("transposed", "columns", "rows", "empty"):
        assert torch.equal(loaded[key], data[key])


def test_load_checkpoint_checks_legacy_results_too(tmp_path) -> None:
    path = tmp_path / "legacy_trick.pt"
    torch.save({"buffer": io.BytesIO(b"x"), "t": torch.zeros(1).expand(1000, 6)}, path)
    with (
        pytest.warns(UserWarning, match="unsafe"),
        pytest.raises(ValueError, match="storage is smaller than its shape"),
    ):
        load_checkpoint(str(path), allow_unsafe_legacy_format=True)


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


def test_legacy_path_applies_the_archive_check(tmp_path, torch_load_calls) -> None:
    model, _ = _trained_protonet()
    legacy = str(tmp_path / "legacy.eq")
    _write_legacy_protonet_file(model, legacy)
    deflated = str(tmp_path / "legacy_deflated.eq")
    _rewrite_zip_deflated(legacy, deflated)

    torch_load_calls.clear()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=_NOT_PLAIN_ARCHIVE):
            eq.load_equine_model(deflated, allow_unsafe_legacy_format=True)
    assert torch_load_calls == [], "torch.load ran on a compressed archive"
    assert not [w for w in record if "unsafe" in str(w.message)], (
        "refused before the unsafe fallback, so no unsafe-load warning"
    )


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


def test_non_zip_file_needs_the_legacy_opt_in(tmp_path) -> None:
    """torch's pre-1.6 format sizes storages from the pickle alone."""
    path = str(tmp_path / "old_format.pt")
    torch.save({"a": torch.arange(4.0)}, path, _use_new_zipfile_serialization=False)
    with pytest.raises(ValueError, match="not a PyTorch zip archive"):
        load_checkpoint(path)
    with pytest.warns(UserWarning, match="unsafe"):
        loaded = load_checkpoint(path, allow_unsafe_legacy_format=True)
    assert torch.equal(loaded["a"], torch.arange(4.0))


def test_loaded_tensors_are_copied_off_the_file(tmp_path) -> None:
    base = torch.arange(1000.0)
    path = tmp_path / "t.pt"
    torch.save({"a": base, "view": base[10:20], "b": torch.ones(5)}, path)
    loaded = _load_without_unsafe_warning(lambda: load_checkpoint(str(path)))
    # A storage read from a file is never resizable; a fresh copy is.
    assert all(t.untyped_storage().resizable() for t in loaded.values())
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
    assert "embedding_model=" in str(err.value)

    # A caller who vouches for the file gets each label's data copied apart.
    loaded = _load_without_unsafe_warning(
        lambda: load_checkpoint(path, allow_unsafe_legacy_format=True)
    )
    assert not equine.utils._support_is_aliased(loaded)
    for key in ("support", "outlier_kde"):
        for label, value in expected.get(key, {}).items():
            if key == "support":
                assert torch.equal(loaded[key][label], value)
            else:
                assert torch.equal(loaded[key][label]["dataset"], value["dataset"])
                assert loaded[key][label]["bw_factor"] == value["bw_factor"]


@pytest.mark.parametrize("container", [set, frozenset])
def test_load_checkpoint_refuses_sets(tmp_path, container) -> None:
    """Tensors inside a set would skip the storage checks and the copy; EQUINE
    never writes sets."""
    path = str(tmp_path / "set.pt")
    torch.save({"s": container([torch.zeros(1).expand(1000, 6)])}, path)
    if container is set:  # restricted unpickling refuses frozensets by itself
        with pytest.raises(ValueError, match="contains a set"):
            load_checkpoint(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the legacy fallback warns
        with pytest.raises(ValueError, match="contains a set"):
            load_checkpoint(path, allow_unsafe_legacy_format=True)


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
    torch.save({"a": base, "view": base[10:20]}, path)
    loaded = load_checkpoint(path, map_location=_OTHER_DEVICE)
    assert {t.device.type for t in loaded.values()} == {_OTHER_DEVICE}
    assert torch.equal(loaded["view"].cpu(), torch.arange(10.0, 20.0))


@pytest.mark.accelerator
@pytest.mark.skipif(_OTHER_DEVICE is None, reason="needs a CUDA or MPS device")
def test_legacy_path_does_not_restore_the_saved_device(tmp_path) -> None:
    path = str(tmp_path / "old_format.pt")
    torch.save(
        {"a": torch.arange(4.0, device=_OTHER_DEVICE)},
        path,
        _use_new_zipfile_serialization=False,
    )
    with pytest.warns(UserWarning, match="unsafe"):
        loaded = load_checkpoint(path, allow_unsafe_legacy_format=True)
    assert loaded["a"].device.type == "cpu"


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
    """`_archive_bytes` must convert a multi-megabyte archive quickly. The
    conversion this replaced iterated the storage one Python int at a time
    (~1s/MB), so a 4 MB tensor must comfortably clear a generous bound."""
    import time

    from equine.utils import _archive_bytes

    tensor = torch.randint(0, 256, (4_000_000,), dtype=torch.uint8)

    start = time.perf_counter()
    result = _archive_bytes(tensor)
    elapsed = time.perf_counter() - start

    assert result == tensor.numpy().tobytes()
    assert elapsed < 0.5


def test_load_jit_archive_accepts_all_stored_forms() -> None:
    """The archive loader reads the uint8 tensor written today as well as the
    bytes and BytesIO forms found in files written by earlier versions."""
    from equine.utils import _jit_archive_to_tensor, _load_jit_archive

    module = BasicEmbeddingModel(6, 3)
    buffer = _jit_buffer(module)
    x = torch.rand(4, 6)
    expected = module(x)

    for archive in (_jit_archive_to_tensor(buffer), buffer.getvalue(), buffer):
        rebuilt = _load_jit_archive(archive)
        assert torch.allclose(rebuilt(x), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# torch.load(weights_only=True) has a known bypass before torch 2.6
# (CVE-2025-32434); EQUINE must require and enforce that floor.
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

    with pytest.warns(UserWarning, match="unsafe"):
        result = load_checkpoint(str(path), allow_unsafe_legacy_format=True)
    assert result == {"a": 1}


def test_load_checkpoint_accepts_supported_torch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(equine.utils, "_torch_version", lambda: (2, 6))

    path = tmp_path / "trivial.pt"
    torch.save({"a": 1}, path)

    result = _load_without_unsafe_warning(lambda: load_checkpoint(str(path)))
    assert result == {"a": 1}


@pytest.mark.parametrize("version", ["3", True], ids=["str", "bool"])
def test_load_rejects_a_malformed_format_version(tmp_path, version) -> None:
    path = str(tmp_path / "m.eq")
    torch.save({"equine_format_version": version}, path)
    with pytest.raises(ValueError, match="equine_format_version is malformed") as err:
        load_checkpoint(path)
    assert len(str(err.value)) < 300


def test_huge_format_version_gives_a_bounded_error() -> None:
    """The weights-only unpickler refuses such an int, but the legacy path
    (unrestricted unpickling) does not."""
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
    checkpoint["equine_format_version"] = 3
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="Unsupported EQUINE model format version"):
        eq.load_equine_model(str(path))
