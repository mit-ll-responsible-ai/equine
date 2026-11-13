# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Rewrite model files from earlier EQUINE versions in the current format (format 4).

    python -m equine.upgrade MODEL.eq [-o OUT.eq] [--recipe JSON] [--import MODULE]
                             [--check-inputs FILE | --no-check]

A PyTorch archive whose embedding is stored as a recipe and weights (formats 2 and 3,
written by development builds) is read with ``torch.load(weights_only=True)`` and
rewritten with the same contents. The written file must load back to an equal
checkpoint before it replaces the original.

Files saved by EQUINE 0.1.x hold pickled Python objects, which
``torch.load(weights_only=True)`` refuses, and store the embedding model as
TorchScript, which EQUINE does not load. The upgrade reads such a file with
unrestricted unpickling and loads its TorchScript archive; both run code
embedded in the file, so only upgrade files you trust. A file is unpickled
without restrictions only after ``torch.load(weights_only=True)`` refused it and
every argument that does not need the file has been checked.

For a TorchScript embedding the caller supplies the architecture: an instance of a
registered architecture (``@equine.embedding_architecture``), or on the command line
its recipe. The archive's weights are copied into it and the model is written as a
data-only file. Before anything replaces the original, the written file is loaded back
and its embedding must reproduce the archive's outputs on the file's support rows and
on ``check_inputs`` (``--check-inputs``). A file with neither is refused before its
archive is loaded, unless the comparison is waived with ``allow_unchecked=True``
(``--no-check``).

Reading files whose embedding is TorchScript is deprecated.
"""

import argparse
import contextlib
import importlib
import io
import itertools
import json
import os
import pickle
import stat
import sys
import warnings
from collections.abc import Iterator
from enum import Enum
from secrets import token_hex
from typing import Any, BinaryIO

import numpy as np
import torch
from scipy.stats import gaussian_kde

from . import _model_file
from .equine_gp import EquineGP
from .equine_protonet import EquineProtonet, _kde_to_state
from .registry import (
    _describe_error,
    _truncate,
    build_from_recipe,
    embedding_recipe,
    registered_architectures,
)
from .utils import (
    EQUINE_FORMAT_VERSION,
    LegacyFormatError,
    _embedding_checkpoint,
    _holds_torchscript,
    _is_extra_state_key,
    _torch_version,
    load_checkpoint,
)

_DEPRECATION = (
    "Reading model files that store the embedding model as TorchScript is "
    "deprecated and will be removed in a later release; upgrade your files now."
)

_HEAD_NOTE = (
    "The file held the weights of a Linear head, which EQUINE <= 0.1.6 saved; the "
    "head is now an Identity, so those weights were dropped and "
    "predictions can differ from those of the EQUINE version that saved the file."
)

_NOTHING_TO_COMPARE_ERROR = (
    "'{path}' stores no support rows, so there are no inputs on which to check "
    "that the rebuilt embedding reproduces the file's TorchScript one; nothing "
    "was written. Pass inputs to compare on, one per row: check_inputs= from "
    "Python, or --check-inputs FILE (a tensor saved with torch.save) on the "
    "command line. To upgrade without the comparison, pass allow_unchecked=True, "
    "or --no-check."
)

_NOT_COMPARED_NOTE = (
    "The file stores no support rows and no check inputs were given, so the "
    "rebuilt embedding was not compared with the file's TorchScript one."
)

# Windows does not replace a read-only file; POSIX renames ignore the target's mode.
_RENAME_NEEDS_A_WRITABLE_TARGET = os.name == "nt"

_RTOL = 1e-4
_ATOL = 1e-5

# torch < 2.6 has a known bypass of weights_only=True (CVE-2025-32434).
_MIN_SAFE_TORCH = (2, 6)

_OLD_TORCH_ERROR = (
    "The upgrade needs torch >= 2.6 to read model files (installed: {version}): "
    "torch.load(weights_only=True) has a known bypass on earlier releases "
    "(CVE-2025-32434)."
)

# Random names tried for the file written before it replaces the original.
_PARTIAL_ATTEMPTS = 100


def _plain(value: Any) -> Any:
    """``value`` with numpy scalars and strings, at any depth, as Python values."""
    if isinstance(value, dict):
        return {_plain(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _archive_bytes(archive: torch.Tensor) -> bytes:
    """
    Return the exact bytes of a ``uint8`` archive tensor.

    A non-full or offset view yields exactly that view's bytes, not the whole backing
    storage: ``.numpy()`` operates on the tensor's own data (respecting its shape,
    stride and storage offset).
    """
    return archive.detach().cpu().contiguous().numpy().tobytes()


def _load_archive(archive: Any) -> torch.nn.Module:
    """
    The TorchScript module stored in a checkpoint, loaded on the CPU.

    Accepts a ``uint8`` tensor (as format-2 files store it), an ``io.BytesIO`` (as files
    from EQUINE 0.1.8 and earlier store it), and ``bytes`` or ``bytearray``.
    """
    load = getattr(getattr(torch, "jit", None), "load", None)
    if load is None:
        raise ValueError(
            "This torch build has no torch.jit.load, which the upgrade needs to "
            "read the file's TorchScript archive; run the upgrade with a torch "
            "release that still ships TorchScript."
        )
    if isinstance(archive, torch.Tensor):
        buffer = io.BytesIO(_archive_bytes(archive))
    elif isinstance(archive, (bytes, bytearray)):
        buffer = io.BytesIO(archive)
    elif isinstance(archive, io.BytesIO):
        buffer = archive
    else:
        raise ValueError(
            "The file's embed_jit_save is not a TorchScript archive "
            f"({type(archive).__name__})."
        )
    buffer.seek(0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # torch's own notice
            return load(buffer, map_location="cpu")
    except Exception as err:
        raise ValueError(
            "The file's TorchScript archive could not be loaded "
            f"({_describe_error(err)})."
        ) from err


def _current_layout(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The checkpoint rewritten with plain values only, minus its embedding, plus notes."""
    out = dict(checkpoint)
    notes: list[str] = []
    out.pop("embed_jit_save", None)
    out.pop("contains_executable", None)
    out["equine_format_version"] = EQUINE_FORMAT_VERSION
    settings = dict(out.get("settings", {}))
    if isinstance(settings.get("cov_type"), Enum):
        settings["cov_type"] = settings["cov_type"].value
    if "laplace_model_save" in out:
        # EquineGP takes no use_temperature; init_temperature holds the value.
        settings.pop("use_temperature", None)
        if out.get("support") is None:  # absent or None when no support was set
            out["support"] = {}
    out["settings"] = _plain(settings)
    if isinstance(out.get("support"), dict):
        out["support"] = {int(label): x for label, x in out["support"].items()}
    if isinstance(out.get("outlier_kde"), dict):
        out["outlier_kde"] = {
            int(label): _kde_to_state(kde) if isinstance(kde, gaussian_kde) else kde
            for label, kde in out["outlier_kde"].items()
        }
    head = out.get("model_head_save")
    if isinstance(head, dict) and set(head) == {"weight", "bias"}:
        out["model_head_save"] = {}
        notes.append(_HEAD_NOTE)
    for key in ("feature_names", "label_names"):
        names = out.get(key)
        if isinstance(names, (list, tuple)) or (
            isinstance(names, np.ndarray) and names.ndim == 1
        ):  # anything else is left for the loader to report
            out[key] = [str(name) for name in names]
    if "train_summary" in out:
        out["train_summary"] = _plain(out["train_summary"])
    return out, notes


def _attribute_tensor(module: Any, key: str) -> torch.Tensor | None:
    """The tensor reached by attribute access along dotted ``key``, if any."""
    value = module
    for part in key.split("."):
        try:
            value = getattr(value, part)
        except (AttributeError, RuntimeError):
            return None
    return value if isinstance(value, torch.Tensor) else None


def _weights_from_archive(
    scripted: torch.nn.Module, target: torch.nn.Module
) -> dict[str, Any]:
    """
    A ``state_dict`` for ``target`` taken from the TorchScript module.

    Each entry comes from the archive's ``state_dict`` or, failing that, from the tensor
    attribute of the same dotted name. Extra-state entries keep the target's values.
    Archive entries the target does not store are allowed only for the target's
    non-persistent buffers.
    """
    source = scripted.state_dict()
    expected = target.state_dict()
    weights: dict[str, Any] = {}
    missing: list[str] = []
    wrong_shape: list[str] = []
    for key, reference in expected.items():
        if _is_extra_state_key(key):
            weights[key] = reference
            continue
        value = source[key] if key in source else _attribute_tensor(scripted, key)
        if value is None:
            missing.append(key)
        elif tuple(value.shape) != tuple(reference.shape):
            wrong_shape.append(
                f"{key} {tuple(value.shape)} vs {tuple(reference.shape)}"
            )
        else:
            weights[key] = value
    non_persistent = {name for name, _ in target.named_buffers()} - set(expected)
    extra = sorted(set(source) - set(expected) - non_persistent)
    problems = []
    if missing:
        problems.append(f"not in the archive: {', '.join(missing[:5])}")
    if wrong_shape:
        problems.append(f"different shapes: {'; '.join(wrong_shape[:5])}")
    if extra:
        problems.append(f"not in the supplied architecture: {', '.join(extra[:5])}")
    if problems:
        raise ValueError(
            f"The supplied embedding_model ({type(target).__name__}) does not match "
            f"the file's TorchScript embedding ({'; '.join(problems)})."
        )
    return weights


def _first_float_dtype(module: torch.nn.Module) -> torch.dtype | None:
    for tensor in itertools.chain(module.parameters(), module.buffers()):
        if tensor.is_floating_point():
            return tensor.dtype
    return None


def _embed(module: torch.nn.Module, rows: torch.Tensor) -> torch.Tensor:
    """``module``'s eval-mode output for ``rows`` on the CPU, in its own dtype."""
    dtype = _first_float_dtype(module)
    x = rows.detach().cpu()
    if dtype is not None and x.is_floating_point():
        x = x.to(dtype)
    module.eval()
    with torch.no_grad():
        out = module(x)
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"the module returned a {type(out).__name__}, not a tensor")
    return out


def _require_same_embedding(
    scripted: torch.nn.Module, rebuilt: torch.nn.Module, rows: torch.Tensor
) -> None:
    """Raise unless the rebuilt embedding reproduces the archive's outputs."""
    try:
        expected = _embed(scripted, rows)
        actual = _embed(rebuilt, rows)
    except Exception as err:
        raise ValueError(
            "The upgrade could not compare the supplied architecture's embedding "
            f"with the file's ({_describe_error(err)})."
        ) from err
    if actual.shape != expected.shape:
        raise ValueError(
            "The supplied architecture does not reproduce the file's embedding: "
            f"its output shape {tuple(actual.shape)} differs from "
            f"{tuple(expected.shape)}."
        )
    actual = actual.to(expected.dtype)
    if not torch.allclose(actual, expected, rtol=_RTOL, atol=_ATOL):
        largest = (actual - expected).abs().max().item()
        raise ValueError(
            "The supplied architecture does not reproduce the file's embedding: "
            f"its outputs differ from the TorchScript module's by up to {largest:.3g}."
        )


def _has_rows(value: Any) -> bool:
    """Whether ``value`` is a tensor with at least one row."""
    return isinstance(value, torch.Tensor) and value.dim() > 0 and len(value) > 0


def _comparison_blocks(
    checkpoint: dict[str, Any], check_inputs: torch.Tensor | None
) -> list[torch.Tensor]:
    """The inputs to compare the embeddings on: each label's support rows and
    ``check_inputs``, each kept as its own block (dtypes and shapes may differ)."""
    support = checkpoint.get("support")
    blocks = list(support.values()) if isinstance(support, dict) else []
    if check_inputs is not None:
        blocks.append(check_inputs)
    return [b for b in blocks if _has_rows(b)]


def _replaced_file(path: str, out_path: str | None) -> str:
    """
    The path the upgrade writes its result to: ``out_path``, or ``path`` when
    it is not given; for a symbolic link, the file the link points to.
    """
    target = out_path or path
    return os.path.realpath(target) if os.path.islink(target) else target


def _parent_directory(path: str) -> str:
    """
    The directory that holds ``path``, spelled as ``path`` spells it.
    """
    return os.path.dirname(path) or os.curdir


def _require_output(target: str) -> None:
    """Raise unless the upgrade can write ``target``; reads nothing from the file."""
    if os.path.islink(target):
        raise ValueError(
            f"The output '{target}' is a symbolic link that cannot be followed."
        )
    if os.path.isdir(target) or os.path.basename(target) in ("", os.curdir, os.pardir):
        raise ValueError(
            f"The output '{target}' names a directory; give the path of a file."
        )
    directory = _parent_directory(target)
    if not os.path.isdir(directory):
        raise ValueError(
            f"The output '{target}' cannot be written: '{directory}' is not an "
            "existing directory."
        )
    try:
        descriptor, probe = _create_partial(target)
    except OSError as err:
        raise ValueError(f"The output '{target}' cannot be written: {err}") from err
    os.close(descriptor)
    os.remove(probe)


def _require_arguments(
    embedding_model: torch.nn.Module | None,
    check_inputs: torch.Tensor | None,
    target: str,
) -> None:
    """
    Raise unless the arguments can drive an upgrade of a TorchScript embedding that
    writes ``target``; reads nothing from the file.
    """
    if embedding_model is None:
        raise ValueError(
            "The file's embedding is TorchScript, so the upgrade needs the registered "
            "architecture to rebuild it: pass embedding_model."
        )
    _require_output(target)
    if embedding_recipe(embedding_model) is None:
        raise ValueError(
            f"The supplied embedding_model ({type(embedding_model).__name__}) is "
            "not a registered architecture; register its class with "
            "@equine.embedding_architecture and pass an instance."
        )
    if check_inputs is not None and not _has_rows(check_inputs):
        raise ValueError(
            "check_inputs must be a tensor with at least one row (one row per input)."
        )


def _read_safely(path: str) -> dict[str, Any] | None:
    """
    ``path`` read without running code from it: a current-format file with
    ``load_checkpoint``, a PyTorch archive with ``torch.load(weights_only=True)``. None
    for a file that read refuses, or that is not a dictionary with an
    ``equine_format_version``; only the unrestricted read in ``_read_legacy`` opens it.
    """
    try:
        return load_checkpoint(path)
    except LegacyFormatError:
        pass
    if _torch_version() < _MIN_SAFE_TORCH:
        raise ValueError(_OLD_TORCH_ERROR.format(version=torch.__version__))
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        return None
    except Exception as err:
        raise ValueError(
            f"'{path}' could not be read as a PyTorch file "
            f"({_truncate(_describe_error(err), 200)}). Upgrade a file that EQUINE "
            "saved."
        ) from err
    if not isinstance(checkpoint, dict) or "equine_format_version" not in checkpoint:
        return None
    if checkpoint["equine_format_version"] not in (2, 3):
        raise ValueError(
            f"'{path}' is a PyTorch archive with equine_format_version "
            f"{_truncate(repr(checkpoint['equine_format_version']), 50)}, which no "
            "EQUINE version wrote; only formats 2 and 3 were PyTorch archives. "
            "Upgrade a file that EQUINE saved."
        )
    return checkpoint


def _read_legacy(path: str) -> dict[str, Any]:
    """A file from an earlier release, read with unrestricted unpickling."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as err:
        raise ValueError(
            f"'{path}' could not be read as a PyTorch file ({_describe_error(err)})."
        ) from err
    if not isinstance(checkpoint, dict) or "train_summary" not in checkpoint:
        raise ValueError(
            f"'{path}' is not an EQUINE model file: every EQUINE model file is a "
            "dictionary with a train_summary."
        )
    return checkpoint


def _temp_named_as(message: str, partial: str, path: str) -> str:
    """
    ``message`` with the temp file ``partial`` named as ``path``. Where repr
    escapes ``partial`` (a Windows separator does), its repr, which is how
    ``OSError`` quotes file names, is replaced with ``path``'s repr first.
    """
    quoted = repr(partial)
    if quoted[1:-1] != partial:
        message = message.replace(quoted, repr(path))
    return message.replace(partial, path)


def _reason(err: Exception, partial: str, path: str) -> str:
    """
    ``err`` as one sentence for an upgrade error: the temp file named as
    ``path``, the loader's upgrade hint, its advice to save or upgrade again, and
    its tampering suspicion cut off, the type named unless it is a ValueError (or
    the message is empty), and a final period.
    """
    message = _temp_named_as(str(err), partial, path)
    for tail in (
        ". Rewrite it once with",
        "; the file is inconsistent or was tampered",
        "; the file was not written by save()",
    ):
        message = message.split(tail)[0]
    message = message.strip()
    if not message:
        message = type(err).__name__
    elif not isinstance(err, ValueError):
        message = f"{type(err).__name__}: {message}"
    return message if message.endswith(".") else f"{message}."


def _create_partial(out_path: str) -> tuple[int, str]:
    """
    Create the file the upgrade is written to before it replaces ``out_path``, and
    return its descriptor (open for writing) and path. It is a new file with a random
    name next to ``out_path``, created exclusively, and with the permission bits of the
    file it replaces plus owner read and write (those of a new file otherwise), less the
    umask.
    """
    directory = _parent_directory(out_path)
    mode = (
        (os.stat(out_path).st_mode & 0o777) | 0o600
        if os.path.exists(out_path)
        else 0o666
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(_PARTIAL_ATTEMPTS):
        name = f".{os.path.basename(out_path)}.{token_hex(8)}.upgrading"
        partial = os.path.join(directory, name)
        try:
            return os.open(partial, flags, mode), partial
        except FileExistsError:
            continue
        except OSError as err:
            raise OSError(
                f"Could not write '{partial}' ({_describe_error(err)})."
            ) from err
    raise OSError(
        f"Could not create a file for the upgrade in '{directory}': each of "
        f"{_PARTIAL_ATTEMPTS} random names already exists."
    )


@contextlib.contextmanager
def _closed_quietly_on_error(file: BinaryIO) -> Iterator[None]:
    """Close ``file`` when the block raises, ignoring an error from the close."""
    try:
        yield
    except BaseException:
        with contextlib.suppress(OSError):
            file.close()
        raise


def _require_written_file(file: BinaryIO, partial: str, out_path: str) -> None:
    """Raise unless ``partial`` still names ``file``, the file written there."""
    written = os.fstat(file.fileno())
    named: os.stat_result | None
    try:
        named = os.lstat(partial)
    except FileNotFoundError:
        named = None
    if named is None or not os.path.samestat(named, written):
        raise OSError(
            "The upgraded file was replaced or removed before it could be moved "
            f"into place; '{out_path}' was left unchanged."
        )


def _move_into_place(
    file: BinaryIO, partial: str, out_path: str, notes: list[str]
) -> None:
    """
    Close ``file``, the upgraded file written at ``partial``, and move it to
    ``out_path``. It first gets the permission bits ``out_path`` had, read without
    following a link and set through ``file`` (through ``partial`` where ``os.fchmod``
    is missing). The move is refused if ``out_path`` has become a symbolic link or, on
    POSIX, if ``partial`` no longer names ``file``. Where a rename needs a writable
    target (Windows), the file also gets owner write, and a read-only ``out_path`` gets
    its bits back once the move is done; a read-only ``out_path`` that is not a link is
    made writable for a second attempt and gets its bits back if that attempt fails too.
    """

    def note(err: OSError) -> None:
        notes.append(
            f"Could not copy the permission bits of '{out_path}' onto the "
            f"upgraded file ({_temp_named_as(str(err), partial, out_path)})."
        )

    target: os.stat_result | None
    try:
        target = os.lstat(out_path)
    except FileNotFoundError:
        target = None
    if target is not None and stat.S_ISLNK(target.st_mode):
        raise OSError(
            f"'{out_path}' became a symbolic link during the upgrade; it was left "
            "unchanged."
        )
    mode: int | None = None if target is None else stat.S_IMODE(target.st_mode)
    moved_mode: int | None = None
    if mode is not None:
        moved_mode = mode | stat.S_IWUSR if _RENAME_NEEDS_A_WRITABLE_TARGET else mode
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(file.fileno(), moved_mode)
            else:
                os.chmod(partial, moved_mode)
        except OSError as err:  # a filesystem that rejects mode changes
            note(err)
            mode = None
    if os.name == "posix":
        _require_written_file(file, partial, out_path)
    try:
        file.close()
    except OSError as err:
        raise OSError(f"Could not write '{partial}' ({_describe_error(err)}).") from err
    try:
        os.replace(partial, out_path)
    except PermissionError:
        if (
            not _RENAME_NEEDS_A_WRITABLE_TARGET
            or mode is None
            or mode & stat.S_IWUSR
            or os.path.islink(out_path)
        ):
            raise
        os.chmod(out_path, mode | stat.S_IWUSR)
        try:
            os.replace(partial, out_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.chmod(out_path, mode)
            raise
    if mode is not None and mode != moved_mode:
        try:
            os.chmod(out_path, mode)
        except OSError as err:
            note(err)


def _same_encoding(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether ``a`` and ``b`` encode to the same header metadata and tensor bytes."""
    (tensors_a, metadata_a), (tensors_b, metadata_b) = map(_model_file.encode, (a, b))
    return (
        metadata_a == metadata_b
        and tensors_a.keys() == tensors_b.keys()
        and all(
            tensors_a[name].dtype == tensors_b[name].dtype
            and tensors_a[name].shape == tensors_b[name].shape
            and torch.equal(
                tensors_a[name].reshape(-1).view(torch.uint8),
                tensors_b[name].reshape(-1).view(torch.uint8),
            )
            for name in tensors_a
        )
    )


def _write_checked(
    path: str,
    current: dict[str, Any],
    out_path: str | None,
    notes: list[str],
    check: Any = None,
) -> str:
    """
    Write ``current`` as a model file that replaces ``out_path`` (``path`` when it is
    None), once it loads back, the checkpoint read back encodes like ``current``
    (``_same_encoding``), and ``check(rebuilt)``, when given, passes for the model
    built from it. Returns where it was written.
    """
    try:
        data = _model_file.to_bytes(current)
    except ValueError as err:
        raise ValueError(
            f"'{path}' holds a value the upgrade does not convert "
            f"({_truncate(str(err), 300)}); it was left unchanged. To convert it, read "
            "it with torch.load (weights_only=True for a format 2 or 3 file; "
            "weights_only=False for a file from EQUINE 0.1.x, which runs code from the "
            "file, so only if you trust it), change that value, write it back with "
            "torch.save, and upgrade it again."
        ) from err
    out_path = _replaced_file(path, out_path)
    descriptor, partial = _create_partial(out_path)
    try:
        with os.fdopen(descriptor, "wb") as file, _closed_quietly_on_error(file):
            try:
                file.write(data)
                file.flush()
            except Exception as err:
                raise OSError(
                    f"Could not write '{partial}' ({_describe_error(err)})."
                ) from err
            # The class follows the file's layout.
            model_class = (
                EquineGP if "laplace_model_save" in current else EquineProtonet
            )
            try:
                read = load_checkpoint(partial)
                rebuilt = model_class.load(partial, device="cpu")
            except Exception as err:
                raise ValueError(
                    "The rewritten file does not load: "
                    f"{_reason(err, partial, path)} '{path}' was left unchanged."
                ) from err
            if not _same_encoding(current, read):
                raise ValueError(
                    f"The rewritten file does not read back as written; '{path}' was "
                    "left unchanged. This is a bug in EQUINE: please report it, with "
                    "the file, at https://github.com/mit-ll-responsible-ai/equine/issues."
                )
            if check is not None:
                check(rebuilt)
            _move_into_place(file, partial, out_path, notes)
    except BaseException:
        with contextlib.suppress(OSError):  # the upgrade's error is reported
            if os.path.lexists(partial):
                os.remove(partial)
        raise
    return out_path


def _convert(
    path: str, checkpoint: dict[str, Any], out_path: str | None
) -> tuple[str, list[str]]:
    """
    Rewrite a format-2 or format-3 file with no TorchScript in the current format,
    without its ``contains_executable`` entry; return where, plus the notes to report.
    """
    current = {k: v for k, v in checkpoint.items() if k != "contains_executable"}
    current["equine_format_version"] = EQUINE_FORMAT_VERSION
    notes: list[str] = []
    return _write_checked(path, current, out_path, notes), notes


def _upgrade(
    path: str,
    checkpoint: dict[str, Any],
    embedding_model: torch.nn.Module,
    out_path: str | None,
    check_inputs: torch.Tensor | None,
    allow_unchecked: bool,
) -> tuple[str, list[str]]:
    """Write the upgraded file and return where, plus the notes to report."""
    if "embed_jit_save" not in checkpoint:
        raise ValueError(
            f"'{path}' holds no TorchScript archive to upgrade from: files from EQUINE "
            "0.1.8 and earlier, and format-2 files, store one in embed_jit_save, so "
            "this file was not written by an EQUINE version the upgrade converts. If "
            "it holds its embedding as a recipe and weights, remove its "
            "contains_executable entry: read it with torch.load(path, "
            "map_location='cpu', weights_only=True), delete the entry, and write it "
            "back with torch.save."
        )
    current, notes = _current_layout(checkpoint)
    blocks = _comparison_blocks(current, check_inputs)
    if not blocks:
        if not allow_unchecked:
            raise ValueError(_NOTHING_TO_COMPARE_ERROR.format(path=path))
        notes.append(_NOT_COMPARED_NOTE)
    scripted = _load_archive(checkpoint["embed_jit_save"])
    embedding_model.load_state_dict(_weights_from_archive(scripted, embedding_model))
    current.update(_embedding_checkpoint(embedding_model))
    try:
        _model_file.encode(
            {key: current[key] for key in ("embedding_recipe", "embedding_state_dict")}
        )
    except ValueError as err:
        raise ValueError(
            f"The supplied embedding_model ({type(embedding_model).__name__}) holds a "
            f"value a model file does not store ({_truncate(str(err), 300)}); pass an "
            "architecture whose constructor arguments and state hold only values a "
            f"model file stores. '{path}' was left unchanged."
        ) from err

    def check(rebuilt: Any) -> None:
        for rows in blocks:
            _require_same_embedding(scripted, rebuilt.embedding_model, rows)

    return _write_checked(path, current, out_path, notes, check), notes


def upgrade_model_file(
    path: str,
    embedding_model: torch.nn.Module | None = None,
    out_path: str | None = None,
    *,
    check_inputs: torch.Tensor | None = None,
    allow_unchecked: bool = False,
) -> str | None:
    """
    Rewrite a model file from an earlier EQUINE version in the current format.

    Parameters
    ----------
    path : str
        The file to upgrade. A file that ``load_checkpoint`` reads is current, and
        nothing is written unless it holds a TorchScript archive. A PyTorch archive
        is read with ``torch.load(weights_only=True)``; a format-2 or format-3 file
        with no TorchScript archive is rewritten with the same contents. A file that
        read refuses (a file saved by EQUINE 0.1.x) is then read with unrestricted
        unpickling, once the other arguments have been checked: only upgrade such
        files if you trust them. A TorchScript archive is loaded.
    embedding_model : torch.nn.Module | None
        An instance of the registered architecture the file's embedding was
        built from, needed only for a file whose embedding is TorchScript. The
        archive's weights are copied into it.
    out_path : str | None
        Where to write the result; defaults to replacing ``path``. Its directory must
        exist and allow creating a file. The result is written to a new file with a
        random name next to it, created with the permission bits of the file it replaces
        plus owner read and write, and moved into place only after it has been loaded
        back and checked. A failed upgrade leaves the original as it was. A symbolic
        link is followed and its target is replaced; the replaced file keeps its
        permission bits where the filesystem allows it. A link that cannot be followed,
        such as one that loops, is refused; a link to a missing file creates that file.
    check_inputs : torch.Tensor | None
        Keyword-only. Inputs, one per row, on which the new embedding must
        reproduce the archive's outputs, in addition to the file's support
        rows. A file that stores no support rows (most ``EquineGP`` files)
        needs them unless ``allow_unchecked`` is True.
    allow_unchecked : bool
        Keyword-only. Upgrade a file that stores no support rows without
        ``check_inputs``; the new embedding is then not compared with the
        archive's. By default such a file is refused and nothing is written.

    Returns
    -------
    str | None
        The path of the written file, or None if ``path`` is already a
        data-only current-format file and nothing was written.

    Raises
    ------
    ValueError
        If the file is not an EQUINE model file, if ``load_checkpoint`` refuses
        it for a reason other than its age, if ``out_path`` names a directory,
        is a symbolic link that cannot be followed, or its directory does not
        exist or no file can be created in it, if the file's embedding is
        TorchScript and ``embedding_model`` is None, is not a registered
        architecture or does not match the archive's weights, if ``check_inputs``
        is not a tensor with at least one row, if the file stores no support rows
        and neither ``check_inputs`` nor ``allow_unchecked`` is given, if the file
        holds a value the current format does not store, if the rewritten file
        does not load or does not read back equal, or if its embedding does not
        reproduce the archive's outputs.
    OSError
        If writing the result or moving it into place fails.

    Warns
    -----
    FutureWarning
        When the file's embedding is TorchScript: reading these files is deprecated.
    UserWarning
        When Linear-head weights were dropped, when ``allow_unchecked`` let
        the upgrade write a file whose embedding was not compared, or when
        the permission bits could not be copied.
    """
    checkpoint = _read_safely(path)
    if checkpoint is not None and not _holds_torchscript(checkpoint):
        if checkpoint["equine_format_version"] == EQUINE_FORMAT_VERSION:
            return None
        _require_output(_replaced_file(path, out_path))
        written, notes = _convert(path, checkpoint, out_path)
    else:
        warnings.warn(_DEPRECATION, FutureWarning, stacklevel=2)
        target = _replaced_file(path, out_path)
        _require_arguments(embedding_model, check_inputs, target)
        assert embedding_model is not None
        if checkpoint is None:
            checkpoint = _read_legacy(path)
        written, notes = _upgrade(
            path, checkpoint, embedding_model, out_path, check_inputs, allow_unchecked
        )
    for note in notes:
        warnings.warn(note, UserWarning, stacklevel=2)
    return written


def _import_modules(imports: list[str]) -> None:
    """Import each module in ``imports``."""
    for name in imports:
        try:
            importlib.import_module(name)
        except Exception as err:
            raise ValueError(
                f"Could not import {name!r} ({_describe_error(err)})."
            ) from err


def _module_from_recipe(recipe_json: str, imports: list[str]) -> torch.nn.Module:
    """The architecture named by a JSON recipe, after importing ``imports``."""
    _import_modules(imports)
    try:
        recipe = json.loads(recipe_json)
    except json.JSONDecodeError as err:
        raise ValueError(f"--recipe is not valid JSON ({err}).") from err
    builder = recipe.get("builder") if isinstance(recipe, dict) else None
    if not isinstance(builder, str):
        raise ValueError(
            '--recipe must be a JSON object with a "builder" string and optional '
            '"kwargs".'
        )
    if builder not in registered_architectures():
        known = ", ".join(registered_architectures()) or "(none)"
        raise ValueError(
            f"--recipe names architecture {builder[:200]!r}, which is not "
            "registered; pass --import MODULE for the module that registers it "
            f"(registered: {known})."
        )
    try:
        return build_from_recipe(recipe)
    except ValueError as err:
        raise ValueError(f"--recipe could not be built: {err}") from err


def _check_inputs_from_file(path: str) -> torch.Tensor:
    """The tensor of input rows in ``path``, read with ``weights_only=True``."""
    try:
        rows = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as err:
        raise ValueError(
            f"--check-inputs '{path}' could not be read "
            f"({_truncate(_describe_error(err), 200)})."
        ) from err
    if not _has_rows(rows):
        held = (
            f"a tensor of shape {tuple(rows.shape)}"
            if isinstance(rows, torch.Tensor)
            else f"a {type(rows).__name__}"
        )
        raise ValueError(
            f"--check-inputs '{path}' must hold one tensor with at least one row "
            f"(one row per input); it holds {held}."
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m equine.upgrade",
        description="Rewrite a model file from an earlier EQUINE version in the "
        "current format. A file whose embedding is TorchScript (every file saved by "
        "EQUINE 0.1.x) needs --recipe. Only upgrade such files if you trust them: "
        "the file is unpickled without restrictions and its TorchScript archive is "
        "loaded, either of which can run code embedded in it.",
    )
    parser.add_argument("path", help="the model file to upgrade")
    parser.add_argument(
        "-o", "--output", help="write the result here instead of replacing PATH"
    )
    parser.add_argument(
        "--recipe",
        help="the embedding architecture as a recipe, for example "
        '\'{"builder": "equine.mlp", "kwargs": {"in_features": 6, '
        '"hidden_sizes": [32], "out_features": 3}}\'',
    )
    parser.add_argument(
        "--import",
        dest="imports",
        action="append",
        default=[],
        metavar="MODULE",
        help="import MODULE first, so the architectures it registers are known "
        "(repeatable); needed when the file's recipe or --recipe names an "
        "architecture that EQUINE does not ship",
    )
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--check-inputs",
        metavar="FILE",
        help="a file holding one tensor of input rows (written with torch.save) "
        "on which the rebuilt embedding must reproduce the file's TorchScript "
        "one, in addition to the file's support rows; a file that stores no "
        "support rows (most EquineGP files) needs this or --no-check",
    )
    comparison.add_argument(
        "--no-check",
        action="store_true",
        help="upgrade a file that stores no support rows without comparing the "
        "rebuilt embedding with the file's TorchScript one (the support rows of a "
        "file that has them are still compared)",
    )
    args = parser.parse_args(argv)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = _read_safely(args.path)
            if checkpoint is not None and not _holds_torchscript(checkpoint):
                if checkpoint["equine_format_version"] == EQUINE_FORMAT_VERSION:
                    print(
                        f"{args.path} is already in the current format; nothing "
                        "written."
                    )
                    return 0
                _import_modules(args.imports)
                _require_output(_replaced_file(args.path, args.output))
                written, notes = _convert(args.path, checkpoint, args.output)
                for note in notes:
                    print(f"note: {note}", file=sys.stderr)
                print(_upgraded(args.path, written))
                return 0
            if args.recipe is None:
                raise ValueError(
                    "--recipe is required: the file's embedding is TorchScript, so "
                    "the upgrade needs the registered architecture to rebuild it."
                )
            module = _module_from_recipe(args.recipe, args.imports)
            check_inputs = (
                None
                if args.check_inputs is None
                else _check_inputs_from_file(args.check_inputs)
            )
            _require_arguments(
                module, check_inputs, _replaced_file(args.path, args.output)
            )
            if checkpoint is None:
                checkpoint = _read_legacy(args.path)
            written, notes = _upgrade(
                args.path, checkpoint, module, args.output, check_inputs, args.no_check
            )
    except (ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    print(_upgraded(args.path, written))
    print(_DEPRECATION, file=sys.stderr)
    return 0


def _upgraded(path: str, written: str) -> str:
    return (
        f"Upgraded {path} in place."
        if written == path
        else f"Upgraded {path} -> {written}"
    )


if __name__ == "__main__":
    sys.exit(main())
