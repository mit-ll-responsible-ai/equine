# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import copy
import inspect
import io
import itertools
import math
import os
import pickle
import sys
import warnings
import zipfile
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Optional, Union

import icontract
import torch
from beartype import beartype
from scipy.stats import gaussian_kde
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCalibrationError,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from .equine import Equine
from .equine_output import EquineOutput
from .registry import (
    _check_plain,
    _truncate,
    build_from_recipe,
    embedding_recipe,
)

# Version of the on-disk layout written by ``EquineProtonet.save`` and
# ``EquineGP.save``. Version 2 contains only tensors, containers and plain
# Python scalars/strings, so it can be read with ``torch.load(weights_only=True)``
# on every supported torch version. The embedding model is a recipe plus a
# state_dict (see ``equine.registry``); a TorchScript archive appears only in
# files saved with ``allow_executable=True`` during the transition release.
# Files without this key are "legacy" files that pickled arbitrary Python
# objects and need unrestricted unpickling.
EQUINE_FORMAT_VERSION = 2

# How to turn a trusted file whose embedding is TorchScript (a legacy file, or
# one saved with allow_executable=True) into a data-only file: save() refuses a
# TorchScript embedding, so the migration loads the weights into a registered
# module through embedding_model=.
_MIGRATION_HINT = (
    "load it with embedding_model=<an instance of your registered architecture> "
    "(plus allow_unsafe_legacy_format=True for a file written by an earlier EQUINE "
    "release, or trust_executable=True for one saved with allow_executable=True), "
    "then call save() to rewrite it in the data-only format"
)

_EXECUTABLE_ERROR = (
    "This model file embeds an executable TorchScript module (saved with "
    "allow_executable=True or by an earlier EQUINE version). Opening it runs that "
    "code. If you trust the file, load it with trust_executable=True. To migrate "
    f"it, {_MIGRATION_HINT}."
)

_EXECUTABLE_SAVE_WARNING = (
    "save(..., allow_executable=True) stored the embedding model as executable "
    "TorchScript. This option, and loading such files, is removed in the next "
    "EQUINE release. Register the embedding's architecture with "
    "@equine.embedding_architecture and save a model built from it instead; for a "
    f"model loaded from a TorchScript file, {_MIGRATION_HINT}."
)

_EXECUTABLE_LOAD_WARNING = (
    "The embedding model was loaded from the file's executable TorchScript archive "
    "(trust_executable=True). Such files stop loading in the next EQUINE release. "
    f"To migrate the file, {_MIGRATION_HINT}."
)

_UNSAFE_LOAD_WARNING = (
    "Loading '{path}' with allow_unsafe_legacy_format=True fell back to "
    "unrestricted unpickling (torch.load(weights_only=False)), which can execute "
    "arbitrary code embedded in the file. Only do this for files you trust. To "
    f"migrate the file, {_MIGRATION_HINT}."
)

_UNSAFE_LOAD_ERROR = (
    "Could not safely load '{path}'. EQUINE reads model files with "
    "torch.load(weights_only=True), which refuses pickled Python objects that "
    "could run code when the file is opened, on torch >= 2.6. This file either "
    "was written by an EQUINE version that used the legacy pickle format, or is "
    "not an EQUINE model. If you created the file yourself and trust it, "
    f"{_MIGRATION_HINT}."
)

# torch.load(weights_only=True) has a known unpickling bypass on every torch
# release before 2.6.0 (CVE-2025-32434 / GHSA-53q9-r3pm-6pq6), so the safe load
# path below is only trustworthy on torch >= 2.6.
_MIN_SAFE_TORCH = (2, 6)

_OLD_TORCH_ERROR = (
    "Restricted unpickling (torch.load(weights_only=True)) has a known bypass "
    "on torch < 2.6 (CVE-2025-32434); installed torch is {version}. Upgrade "
    "torch, or pass allow_unsafe_legacy_format=True only for files you trust."
)


_STORAGE_TRICK_ERROR = (
    "Model file contains a tensor whose storage is smaller than its shape "
    "(expanded or overlapping view); refusing to load."
)

_NOT_PLAIN_ARCHIVE_ERROR = (
    "Model file is not a plain (uncompressed) PyTorch archive; refusing to load."
)

# torch.load reads a file as a zip archive exactly when it starts with this
# local-file-header signature (torch.serialization._is_zipfile).
_ZIP_SIGNATURE = b"PK\x03\x04"

_NO_STORAGE_ERROR = (
    "Model file contains a sparse or meta tensor, which has no dense storage "
    "backing its shape; refusing to load."
)

_NOT_ZIP_ERROR = (
    "Model file is not a PyTorch zip archive (files saved by torch >= 1.6 are); "
    "legacy files need allow_unsafe_legacy_format=True."
)

_STORAGE_SIZE_ERROR = (
    "Model file declares tensor storage larger than the data it contains; "
    "refusing to load."
)

_ALIASED_SUPPORT_ERROR = (
    "Model file's support/KDE tensors alias one another, which lets a small file "
    "multiply the memory and work of loading it; refusing to load. The current "
    "format stores each label's data separately. If you created the file yourself "
    "and trust it, load it with allow_unsafe_legacy_format=True, which copies the "
    "tensors apart (a file from an earlier EQUINE release also needs "
    "embedding_model=<an instance of your registered architecture>), then call "
    "save() to rewrite it."
)

_SET_ERROR = "Model file contains a set; EQUINE files never do; refusing to load."


def _torch_version() -> tuple[int, int]:
    """Return the installed torch version as an ``(major, minor)`` tuple."""
    major, minor = (int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    return (major, minor)


def load_checkpoint(
    path: str,
    map_location: Optional[str] = None,
    allow_unsafe_legacy_format: bool = False,
    _stacklevel: int = 3,
) -> dict[str, Any]:
    """
    Read a saved EQUINE model file with restricted unpickling.

    The file is read with ``torch.load(weights_only=True)``, which only
    reconstructs tensors and plain Python containers, so a crafted pickle
    payload cannot run when the file is opened, provided torch >= 2.6 (the
    minimum EQUINE requires) is installed. Files written by EQUINE versions
    before the safe format (see ``EQUINE_FORMAT_VERSION``) stored Python
    objects that require unrestricted unpickling; they are rejected unless
    ``allow_unsafe_legacy_format`` is set, in which case the unrestricted
    load is used only after the safe load has failed.

    Memory is bounded by the bytes actually present in the file. The file must
    be a plain zip archive (every entry uncompressed, entries no larger than
    the file; see ``_check_zip_archive``). The safe path maps it with
    ``mmap=True``, so a storage is a slice of the file rather than a buffer
    sized by the pickle's claim: a claim running past the end of the file is
    refused by torch, and one running into the following records is refused
    here because honest records never overlap. Expanded or overlapping views,
    meta and sparse tensors, and support/KDE tensors that alias one another
    are refused too. Finally every storage is copied once off the mapping
    (onto ``map_location``), so the result does not depend on the file.
    With ``allow_unsafe_legacy_format`` the caller vouches for the file, so
    aliased support/KDE tensors (which files from earlier releases may hold)
    are copied apart instead of refused.

    A file in the current format stores its embedding as a recipe and weights
    and contains no executable code, unless it was saved with
    ``allow_executable=True``; such a file carries a TorchScript archive,
    is flagged with ``contains_executable``, and only loads with
    ``trust_executable=True``. Legacy files always embed TorchScript.

    Parameters
    ----------
    path : str
        Filename of the saved model.
    map_location : Optional[str]
        Device to move the loaded tensors to. Tensors are loaded on the CPU
        (memory-mapped on the safe path) and then moved to ``map_location`` if
        it is given; with None they stay on the CPU, whatever device they were
        saved from. Both the safe and the legacy path follow this.
    allow_unsafe_legacy_format : bool, optional
        If the safe load fails, fall back to ``weights_only=False`` with a
        ``UserWarning``. This can execute arbitrary code embedded in the file,
        so only enable it for files you created or fully trust. Such a file
        stores its embedding as TorchScript, which ``save()`` refuses to write
        without ``allow_executable=True``; to rewrite it in the data-only
        format, load it with this flag and ``embedding_model=<an instance of
        your registered architecture>`` (``EquineProtonet.load``,
        ``EquineGP.load`` or ``load_equine_model``), then call ``save()``.
        Defaults to False.
    _stacklevel : int, optional
        Stack depth at which the fallback warning is reported, so it points at
        the user's call site rather than at EQUINE internals.

    Returns
    -------
    dict[str, Any]
        The saved checkpoint dictionary.

    Raises
    ------
    ValueError
        If the file cannot be loaded safely and ``allow_unsafe_legacy_format``
        is False (restricted unpickling refuses it, torch < 2.6 is installed,
        it is not a zip archive, or its support/KDE tensors alias one
        another). Whatever the flag: if it is a zip archive with compressed
        or oversized entries, if it does not hold a dictionary, if it
        contains a set, or if a tensor's storage is smaller than its shape
        or claims more data than the file holds.
    """
    old_torch = _torch_version() < _MIN_SAFE_TORCH
    if old_torch and not allow_unsafe_legacy_format:
        raise ValueError(_OLD_TORCH_ERROR.format(version=torch.__version__))
    is_zip = _check_zip_archive(path)
    if not is_zip and not allow_unsafe_legacy_format:
        raise ValueError(_NOT_ZIP_ERROR)

    if is_zip and not old_torch:
        try:
            # mmap needs a path, so torch reopens the file checked above; a file
            # swapped in between skips only that pre-check, not the ones below.
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except pickle.UnpicklingError as err:
            if not allow_unsafe_legacy_format:
                raise ValueError(_UNSAFE_LOAD_ERROR.format(path=path)) from err
        except Exception as err:  # malformed archive: report it as a ValueError
            if "resiz" in str(err):  # torch refusing a slice past the end of the file
                raise ValueError(_STORAGE_SIZE_ERROR) from err
            raise ValueError(
                f"Model file could not be read as a PyTorch archive "
                f"({_truncate(str(err), 200)}); refusing to load."
            ) from err
        else:
            _require_dict(checkpoint, path)
            _validate_format_version(checkpoint)
            tensors = _unique_tensors(checkpoint)
            _reject_storage_tricks(tensors)
            _reject_overlapping_storages(tensors)
            # Checked before the copy below, so a refused file is never copied.
            aliased = _support_is_aliased(checkpoint)
            if aliased and not allow_unsafe_legacy_format:
                raise ValueError(_ALIASED_SUPPORT_ERROR)
            _copy_off_the_file(tensors, map_location)
            if aliased:  # the caller vouched for the file
                _unalias_support(checkpoint)
            return checkpoint

    # Legacy path: old torch, a non-zip file, or restricted unpickling refused
    # the file; the caller opted in to unrestricted unpickling.
    warnings.warn(
        _UNSAFE_LOAD_WARNING.format(path=path), UserWarning, stacklevel=_stacklevel
    )
    checkpoint = torch.load(
        path,
        map_location=map_location if map_location is not None else "cpu",
        weights_only=False,
    )
    _require_dict(checkpoint, path)
    _validate_format_version(checkpoint)
    _reject_storage_tricks(_unique_tensors(checkpoint))
    # Earlier releases saved support tensors as they were in memory, possibly
    # views of one storage; the file is trusted here, so copy them apart.
    if _support_is_aliased(checkpoint):
        _unalias_support(checkpoint)
    return checkpoint


def _validate_format_version(checkpoint: Any) -> None:
    """
    Raise if a loaded checkpoint declares a format version newer than this
    EQUINE build understands.

    Files without an ``equine_format_version`` key are legacy files written
    before the key existed and are left alone here. A present key must be an
    ``int`` (not a ``bool``); a huge one is not formatted into the message.
    """
    if isinstance(checkpoint, dict) and "equine_format_version" in checkpoint:
        version = checkpoint["equine_format_version"]
        if type(version) is not int:
            raise ValueError(
                "Model file's equine_format_version is malformed (expected an int; "
                f"got {_shown(version)})."
            )
        if version > EQUINE_FORMAT_VERSION:
            shown = version if version < 10**6 else "a much newer one"
            raise ValueError(
                f"Unsupported EQUINE model format version {shown}; this "
                f"EQUINE supports up to {EQUINE_FORMAT_VERSION}. Upgrade EQUINE."
            )


def _plain_names(names: Optional[list[str]]) -> Optional[list[str]]:
    """
    ``feature_names`` / ``label_names`` as plain ``str`` for a checkpoint.

    Names often come from numpy (``LabelEncoder.inverse_transform`` returns
    ``numpy.str_``), which ``torch.load(weights_only=True)`` refuses to unpickle.
    """
    return None if names is None else [str(name) for name in names]


def _require_dict(checkpoint: Any, path: str) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"'{path}' is not an EQUINE model file (expected a dictionary at the "
            "top level)."
        )


def _check_zip_archive(path: str) -> bool:
    """
    Refuse a zip archive that ``torch.load`` could inflate beyond the file's size.

    ``torch.save`` writes every entry uncompressed, so a genuine archive's
    entries add up to no more than the file. ``torch.load`` would nevertheless
    decompress DEFLATE entries (a zip bomb) and read overlapping entries
    several times over, so both are refused here, before ``torch.load`` runs.
    A file that torch would read as a zip but the ``zipfile`` module cannot
    parse is refused too, so the two readers cannot disagree about it.

    Returns whether the file is a zip archive at all. A non-zip file uses
    torch's pre-1.6 format, which sizes storages from the pickle alone, so the
    caller only accepts it on the legacy (opted-in) path.
    """
    with open(path, "rb") as file:
        if file.read(len(_ZIP_SIGNATURE)) != _ZIP_SIGNATURE:
            return False
        try:
            with zipfile.ZipFile(file) as archive:
                entries = archive.infolist()
        except Exception as err:  # any parse failure of a hostile file
            raise ValueError(_NOT_PLAIN_ARCHIVE_ERROR) from err
        file_size = os.fstat(file.fileno()).st_size
    compressed = any(info.compress_type != zipfile.ZIP_STORED for info in entries)
    if compressed or sum(info.file_size for info in entries) > file_size:
        raise ValueError(_NOT_PLAIN_ARCHIVE_ERROR)
    return True


def _has_honest_storage(tensor: torch.Tensor) -> bool:
    """Whether every element of a dense tensor has its own bytes in its storage."""
    if tensor.numel() == 0:
        return True
    if tensor.untyped_storage().nbytes() < tensor.numel() * tensor.element_size():
        return False
    # No two elements may share an address: with the dimensions sorted by
    # stride, each stride must step past everything the smaller strides can
    # reach. Size-1 dimensions never step, so their stride is irrelevant.
    reach = 0
    steps = sorted(
        (stride, size)
        for size, stride in zip(tensor.shape, tensor.stride())
        if size > 1
    )
    for stride, size in steps:
        if stride <= reach:
            return False
        reach += stride * (size - 1)
    return True


def _unique_tensors(obj: Any) -> list[torch.Tensor]:
    """
    Every distinct tensor reachable from ``obj`` through dicts, lists and tuples.

    Walks without recursion and tolerates cycles; other objects (only present
    in legacy files) are not inspected. A set is refused: EQUINE never writes
    one, and tensors inside it would escape the checks that use this walk.
    """
    tensors: list[torch.Tensor] = []
    pending = [obj]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif isinstance(item, (set, frozenset)):
            raise ValueError(_SET_ERROR)
    return tensors


def _reject_storage_tricks(tensors: list[torch.Tensor]) -> None:
    """
    Refuse any tensor whose shape is not backed by its own storage.

    ``torch.load`` restores an expanded or overlapping view (a huge shape over
    a few bytes) or a meta tensor (a shape with no data) as readily as an
    ordinary tensor. Such a tensor passes a name-and-shape comparison, so
    without this check a tiny file could make a loader allocate for a network
    of any size.
    """
    for tensor in tensors:
        try:
            dense = tensor.layout == torch.strided and not tensor.is_meta
            honest = dense and _has_honest_storage(tensor)
        except (RuntimeError, NotImplementedError):
            dense = honest = False
        if not dense:
            raise ValueError(_NO_STORAGE_ERROR)
        if not honest:
            raise ValueError(_STORAGE_TRICK_ERROR)


def _reject_overlapping_storages(tensors: list[torch.Tensor]) -> None:
    """
    Refuse storages that claim more bytes than their record in the file holds.

    With ``mmap=True`` every storage is a slice of one mapping of the file, as
    long as the pickle claims. torch refuses a slice that runs past the end of
    the file but not one that runs on into the following records; honest
    records never overlap, so overlapping slices mean some claim exceeded its
    record. (A claim that runs only into the archive's trailing metadata is
    still bounded by the file's size.)
    """
    spans = sorted(
        {(s.data_ptr(), s.nbytes()) for s in (t.untyped_storage() for t in tensors)}
    )
    end = 0
    for start, nbytes in spans:
        if nbytes == 0:
            continue
        if start < end:
            raise ValueError(_STORAGE_SIZE_ERROR)
        end = max(end, start + nbytes)


def _support_is_aliased(checkpoint: dict[str, Any]) -> bool:
    """
    Whether support or KDE data is stored once and referenced several times.

    Each label's support set and KDE dataset is processed on its own, so one
    stored tensor referenced by many labels multiplies memory and time without
    growing the file. An untrusted file must therefore keep every container
    and tensor under ``support`` and ``outlier_kde`` distinct, with no two of
    those tensors sharing a storage. Runs before ``_copy_off_the_file``, so a
    refused file is never copied.
    """
    seen_objects: set[int] = set()
    seen_storages: set[tuple[int, int]] = set()
    pending = [checkpoint.get("support"), checkpoint.get("outlier_kde")]
    while pending:
        item = pending.pop()
        if not isinstance(item, (torch.Tensor, dict, list, tuple)):
            continue
        if id(item) in seen_objects:
            return True
        seen_objects.add(id(item))
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if storage.nbytes() and key in seen_storages:
                return True
            seen_storages.add(key)
        elif isinstance(item, dict):
            pending.extend(item.values())
        else:
            pending.extend(item)
    return False


def _unalias_support(checkpoint: dict[str, Any]) -> None:
    """
    Give each label of a trusted file its own copy of its support and KDE data.

    Covers the layouts EQUINE has written: ``support`` maps labels to tensors,
    and ``outlier_kde`` maps labels to ``{"dataset": tensor, ...}`` states (or,
    in legacy files, to ``gaussian_kde`` objects, which hold no tensors). The
    per-label containers are rebuilt too, so none is shared between labels.
    """
    for key in ("support", "outlier_kde"):
        per_label = checkpoint.get(key)
        if not isinstance(per_label, dict):
            continue
        copied = copy.copy(per_label)  # keeps an OrderedDict an OrderedDict
        for label, value in per_label.items():
            if isinstance(value, torch.Tensor):
                copied[label] = value.clone()
            elif isinstance(value, dict):
                copied[label] = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
        checkpoint[key] = copied


def _copy_off_the_file(
    tensors: list[torch.Tensor], map_location: Optional[str]
) -> None:
    """
    Give every tensor its own copy of its storage, on ``map_location``.

    The safe path loads with ``mmap=True``, so tensors first view a mapping of
    the file; copying keeps them valid if the file later changes or shrinks
    (which would otherwise fault on access). Each storage is copied once and
    its tensors are re-pointed in place, so views and tied tensors keep
    sharing, containers are untouched, and the copies add up to no more than
    the storages already checked.
    """
    device = torch.device(map_location if map_location is not None else "cpu")
    copies: dict[tuple[int, int], torch.UntypedStorage] = {}
    with torch.no_grad():
        for tensor in tensors:
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in copies:
                # Copy through a uint8 view: UntypedStorage.to() does not
                # support every device type (e.g. MPS on torch 2.9).
                raw = torch.empty(0, dtype=torch.uint8).set_(storage)
                copies[key] = (
                    raw.clone() if device.type == "cpu" else raw.to(device)
                ).untyped_storage()
            layout = (
                copies[key],
                tensor.storage_offset(),
                tensor.size(),
                tensor.stride(),
            )
            if device.type == "cpu":
                tensor.set_(*layout)
            else:
                tensor.data = torch.empty(0, dtype=tensor.dtype, device=device).set_(
                    *layout
                )


def _jit_archive_to_tensor(buffer: io.BytesIO) -> torch.Tensor:
    """
    Convert a serialized TorchScript archive into a ``uint8`` tensor for saving.

    As a tensor, the archive is read like every other tensor in the file
    (memory-mapped, checked against the file's size, and copied off the file)
    rather than as a ``bytes`` object inside the pickle.
    """
    return torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)


def _archive_bytes(archive: torch.Tensor) -> bytes:
    """
    Return the exact bytes of a ``uint8`` archive tensor.

    Uses ``.numpy().tobytes()`` rather than iterating ``untyped_storage()``
    element-by-element, which is dramatically faster for large archives.
    ``.numpy()`` operates on the tensor's own data (respecting its shape,
    stride and storage offset), so a non-full or offset view still yields
    exactly that view's bytes, not the whole backing storage.
    """
    return archive.detach().cpu().contiguous().numpy().tobytes()


def _load_jit_archive(
    archive: Union[torch.Tensor, bytes, bytearray, io.BytesIO],
    map_location: Optional[str] = None,
) -> torch.jit.ScriptModule:
    """
    Reconstruct a TorchScript module from an archive stored in a checkpoint.

    Accepts the ``uint8`` tensor written by the current ``save``, as well as the
    ``bytes`` and ``io.BytesIO`` forms found in older files.
    """
    if isinstance(archive, torch.Tensor):
        buffer = io.BytesIO(_archive_bytes(archive))
    elif isinstance(archive, (bytes, bytearray)):
        buffer = io.BytesIO(archive)
    else:
        buffer = archive
    buffer.seek(0)
    return torch.jit.load(buffer, map_location=map_location)


@icontract.ensure(lambda result, module: result is module)
@beartype
def prepare_jit_module(module: torch.nn.Module) -> torch.nn.Module:
    """
    Make an ``nn.Module`` safe to pass to ``torch.jit.script`` on Python 3.14+.

    Starting with Python 3.14 (PEP 649/749) a class's ``__annotations__`` is
    exposed through a ``type`` getset descriptor instead of being stored in the
    class ``__dict__``. ``torch.jit``'s scripting checker reads
    ``__annotations__`` from the module *instance*, where attribute lookup only
    consults the ``__dict__`` of each class in the MRO. For any ``nn.Module``
    that declares no class-level annotations this lookup misses and
    ``nn.Module.__getattr__`` raises ``AttributeError``, breaking
    ``torch.jit.script``. Materializing each submodule's own class annotations
    onto its instance ``__dict__`` lets that lookup succeed. This is a no-op on
    Python < 3.14.

    Parameters
    ----------
    module : torch.nn.Module
        The module that is about to be scripted with ``torch.jit.script``.

    Returns
    -------
    torch.nn.Module
        The same module, returned for convenient inline use.
    """
    if sys.version_info >= (3, 14):
        for submodule in module.modules():
            if "__annotations__" not in submodule.__dict__:
                submodule.__dict__["__annotations__"] = dict(
                    type(submodule).__annotations__
                )
    return module


def _is_extra_state_key(key: str) -> bool:
    """Whether ``key`` is a module's ``get_extra_state`` entry in a state_dict."""
    return key == "_extra_state" or key.endswith("._extra_state")


def _state_dict_mismatch(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> Optional[str]:
    """
    Check that ``actual`` (weights from a file) can fill ``expected`` (a module).

    ``expected`` should come from ``state_dict(keep_vars=True)`` so that tied
    parameters are one object and counted once. Parameter names and shapes
    must agree, and the distinct storages behind ``actual`` must hold at least
    one byte per element of ``expected``: every tensor in a checkpoint is
    individually honest (see ``load_checkpoint``), but many distinct tensors
    could still view one small storage and so claim a much larger network.
    One byte per element is the loosest honest ratio (``bool``/``int8``
    weights). Extra-state entries are skipped and left to ``load_state_dict``.
    Returns None when the weights fit, otherwise a short description whose
    length does not depend on the (possibly file-supplied) key names.
    """
    expected_shapes = {
        k: tuple(v.shape) for k, v in expected.items() if not _is_extra_state_key(k)
    }
    actual_shapes = {
        k: tuple(v.shape) for k, v in actual.items() if not _is_extra_state_key(k)
    }
    differing = sorted(
        k
        for k in expected_shapes.keys() | actual_shapes.keys()
        if expected_shapes.get(k) != actual_shapes.get(k)
    )
    if differing:
        examples = ", ".join(_truncate(repr(k), 80) for k in differing[:3])
        return f"{len(differing)} parameter(s) differ in name or shape, e.g. {examples}"

    unique_expected = {
        id(v): v for k, v in expected.items() if not _is_extra_state_key(k)
    }
    needed = sum(tensor.numel() for tensor in unique_expected.values())
    storages = {
        (storage.data_ptr(), storage.nbytes())
        for storage in (
            v.untyped_storage() for k, v in actual.items() if not _is_extra_state_key(k)
        )
    }
    stored = sum(nbytes for _, nbytes in storages)
    if stored < needed:
        return (
            f"the weights occupy {stored} bytes of storage, too small for the "
            f"architecture they claim to describe ({needed} elements)"
        )
    return None


def _require_weights_dict(
    value: Any, key: str, allow_extra_state: bool = False
) -> None:
    """
    Refuse ``model_save[key]`` unless it maps parameter names to tensors.

    With ``allow_extra_state`` (an embedding's ``state_dict``), a module's
    ``get_extra_state`` entries may hold other plain values; ``load_state_dict``
    validates those.
    """
    if not (
        isinstance(value, dict)
        and all(
            isinstance(k, str)
            and (
                isinstance(v, torch.Tensor)
                or (allow_extra_state and _is_extra_state_key(k))
            )
            for k, v in value.items()
        )
    ):
        raise ValueError(
            f"Model file's {key} is malformed (expected a dictionary of parameter "
            "names to tensors)."
        )


def _require_matching_weights(
    expected: Mapping[str, torch.Tensor], stored: Any, key: str
) -> None:
    """
    Refuse stored weights ``model_save[key]`` that cannot fill ``expected``.

    ``expected`` holds meta tensors with the names and shapes that the model's
    ``settings`` would build. Checking them before the model is constructed
    means a file cannot make the constructor allocate for dimensions its
    stored weights do not back (see ``_state_dict_mismatch``).
    """
    _require_weights_dict(stored, key)
    mismatch = _state_dict_mismatch(expected, stored)
    if mismatch is not None:
        raise ValueError(
            f"Model file's {key} does not match the model its settings describe "
            f"({mismatch}); the file is inconsistent or was tampered with."
        )


def _require_entries(
    model_save: dict[str, Any], keys: tuple[str, ...], cls: type
) -> None:
    """Refuse a checkpoint that lacks any of the entries ``cls`` rebuilds itself from."""
    missing = [key for key in keys if key not in model_save]
    if missing:
        raise ValueError(
            f"Model file is missing {', '.join(repr(key) for key in missing)}, which "
            f"{cls.__name__} needs; the file is incomplete or was not written by "
            "EQUINE."
        )


def _shown(value: Any) -> str:
    """A bounded description of a file-supplied value for an error message."""
    if isinstance(value, str):
        return _truncate(repr(value), 80)
    return f"a {type(value).__name__}"


def _require_device(settings: dict[str, Any], hint: str = "") -> None:
    """
    Refuse a file's ``settings["device"]`` unless this machine can build on it.

    It must name a torch device that is the CPU or an accelerator available
    here; ``meta`` (no data) is refused. ``hint`` is appended to the message
    for an unavailable device.
    """
    if "device" not in settings:
        return
    value = settings["device"]
    try:
        if not isinstance(value, str):
            raise TypeError("not a string")
        device = torch.device(value)
    except (TypeError, ValueError, RuntimeError) as err:
        raise ValueError(
            f"Model file's settings['device'] is not a torch device name "
            f"({_shown(value)})."
        ) from err
    if device.type == "cpu":
        return
    if device.type == "meta":
        raise ValueError(
            "Model file's settings['device'] is 'meta', which holds no data; "
            "refusing to load."
        )
    try:
        is_available = getattr(torch.get_device_module(device.type), "is_available")
        available = bool(is_available())
    except Exception:  # an unknown or unregistered device type
        available = False
    if not available:
        raise ValueError(
            f"Model file's settings['device'] ({_shown(value)}) is not available on "
            f"this machine{hint}."
        )


def _int_label(label: Any) -> Optional[int]:
    """
    ``label`` as an ``int`` if it is a whole number that ``int()`` converts exactly.

    Legacy files may store a label as a 0-dim tensor or a numpy integer. A
    ``bool`` (or bool tensor) or ``str`` is refused, as is anything ``int()``
    changes (``1.5``) or cannot convert (``inf``, a tuple); returns None then.
    """
    if isinstance(label, (bool, str, bytes)) or (
        isinstance(label, torch.Tensor) and label.dtype == torch.bool
    ):
        return None
    try:
        converted = int(label)
        exact = bool(converted == label)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return None
    return converted if exact else None


def _support_from_file(support: Any) -> OrderedDict[int, torch.Tensor]:
    """
    A checkpoint's ``support`` as ``{label: tensor}`` with distinct ``int`` labels.

    Anything else is refused with a bounded ValueError (see ``_int_label``).
    """
    error = ValueError(
        "Model file's support is malformed (expected a dictionary of distinct "
        "integer labels to tensors)."
    )
    if not isinstance(support, dict):
        raise error
    converted: OrderedDict[int, torch.Tensor] = OrderedDict()
    for label, rows in support.items():
        key = _int_label(label)
        if key is None or key in converted or not isinstance(rows, torch.Tensor):
            raise error
        converted[key] = rows
    return converted


def _kde_from_state(state: Any) -> gaussian_kde:
    """Rebuild one outlier-score KDE from the state that ``save`` writes."""
    if not (isinstance(state, dict) and set(state) == {"dataset", "bw_factor"}):
        raise ValueError(
            "Model file's outlier_kde is malformed (each entry must be a dictionary "
            "with exactly 'dataset' and 'bw_factor')."
        )
    dataset, bw_factor = state["dataset"], state["bw_factor"]
    if not (
        isinstance(dataset, torch.Tensor)
        and (dataset.dim() == 1 or (dataset.dim() == 2 and dataset.shape[0] == 1))
    ):
        raise ValueError(
            "Model file's outlier_kde is malformed ('dataset' must be a 1-D or "
            "(1, n) tensor)."
        )
    if not (
        type(bw_factor) in (int, float) and math.isfinite(bw_factor) and bw_factor > 0
    ):
        raise ValueError(
            "Model file's outlier_kde is malformed ('bw_factor' must be a positive "
            f"finite number; got {_shown(bw_factor)})."
        )
    try:
        return gaussian_kde(dataset.detach().cpu().numpy(), bw_method=float(bw_factor))
    except Exception as err:  # e.g. too few points for a covariance
        raise ValueError(
            "Model file's outlier_kde could not be rebuilt "
            f"({_truncate(f'{type(err).__name__}: {err}', 200)})."
        ) from err


def _outlier_kde_from_file(value: Any, legacy: bool) -> OrderedDict[int, gaussian_kde]:
    """
    A checkpoint's ``outlier_kde`` as ``{label: gaussian_kde}`` with distinct labels.

    The current format stores each KDE as ``{"dataset": tensor, "bw_factor":
    number}`` (see ``_kde_from_state``). A pickled ``gaussian_kde`` object is
    what legacy files hold; it is accepted only when the caller opted in with
    ``allow_unsafe_legacy_format`` (``legacy``).
    """
    if not isinstance(value, dict):
        raise ValueError(
            "Model file's outlier_kde is malformed (expected a dictionary of labels "
            "to KDE states)."
        )
    kdes: OrderedDict[int, gaussian_kde] = OrderedDict()
    for label, state in value.items():
        key = _int_label(label)
        if key is None or key in kdes:
            raise ValueError(
                "Model file's outlier_kde is malformed (its labels must be distinct "
                "integers)."
            )
        if isinstance(state, gaussian_kde):
            if not legacy:
                raise ValueError(
                    "Model file's outlier_kde holds a pickled gaussian_kde object, "
                    "which only files loaded with allow_unsafe_legacy_format=True "
                    "may contain."
                )
            kdes[key] = state
        else:
            kdes[key] = _kde_from_state(state)
    return kdes


def _require_names(value: Any, key: str) -> None:
    """
    Refuse ``feature_names`` / ``label_names`` that are not None or a list of str.

    ``str`` subclasses pass: legacy files may hold numpy strings.
    """
    if value is None or (
        isinstance(value, list) and all(isinstance(name, str) for name in value)
    ):
        return
    raise ValueError(
        f"Model file's {key} must be None or a list of strings; got {_shown(value)}."
    )


def _checked_settings(cls: type, settings: Any) -> dict[str, Any]:
    """
    Return a copy of a file's ``settings`` once ``cls`` is known to accept them.

    The settings are passed as ``cls(embedding, **settings)``, where an unknown
    key would surface as a ``TypeError`` quoting the file-supplied key in full.
    Keys must be strings naming a parameter of ``cls.__init__`` (other than
    ``self`` and ``embedding_model``); a constructor taking ``**kwargs``
    accepts any string key. The plain values common to the model classes are
    type-checked here (an ``int`` temperature becomes a ``float``); sizes,
    ``cov_type`` and ``device`` are checked by the caller.
    """
    if not (isinstance(settings, dict) and all(isinstance(k, str) for k in settings)):
        raise ValueError(
            "Model file's settings are malformed (expected a dictionary with "
            "string keys)."
        )
    parameters = inspect.signature(cls.__init__).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        accepted = set(parameters) - {"self", "embedding_model"}
        unknown = sorted(k for k in settings if k not in accepted)
        if unknown:
            examples = ", ".join(_truncate(repr(k), 80) for k in unknown[:3])
            raise ValueError(
                f"Model file's settings contain {len(unknown)} key(s) that "
                f"{cls.__name__} does not accept, e.g. {examples}; the file was "
                "written by an incompatible EQUINE version or was tampered with."
            )
    checked = dict(settings)
    if "init_temperature" in checked:
        temperature = checked["init_temperature"]
        if type(temperature) not in (int, float):
            raise ValueError(
                "Model file's settings['init_temperature'] must be a number; got "
                f"{_shown(temperature)}."
            )
        checked["init_temperature"] = float(temperature)
    for key in ("relative_mahal", "use_temperature"):
        if key in checked and type(checked[key]) is not bool:
            raise ValueError(
                f"Model file's settings[{key!r}] must be True or False; got "
                f"{_shown(checked[key])}."
            )
    for key in ("feature_names", "label_names"):
        if key in checked:
            _require_names(checked[key], f"settings[{key!r}]")
    return checked


def _off_meta_tensors(module: torch.nn.Module) -> list[str]:
    """Names of the parameters and buffers a meta-device build left off the meta device."""
    owned = itertools.chain(module.named_parameters(), module.named_buffers())
    return [name for name, tensor in owned if not tensor.is_meta]


def _submodule_layout(module: torch.nn.Module) -> list[tuple[str, type, str]]:
    """
    Name, class and hyper-parameters of every submodule of ``module``.

    Hyper-parameters come from ``extra_repr`` for torch's own layers only:
    those describe sizes and options, never tensor values or the device, so a
    meta-device rebuild describes itself exactly like the live module. A
    user-defined module's ``extra_repr`` may show values, so only its class
    is compared (its children are listed separately).
    """
    return [
        (
            name,
            type(sub),
            sub.extra_repr() if type(sub).__module__.startswith("torch.") else "",
        )
        for name, sub in module.named_modules()
    ]


def _submodule_difference(
    rebuilt: torch.nn.Module, live: torch.nn.Module
) -> Optional[str]:
    """Where ``live``'s submodules differ from those its recipe rebuilds, if anywhere."""

    def describe(entry: Optional[tuple[str, type, str]]) -> str:
        if entry is None:
            return "nothing"
        return _truncate(f"{entry[1].__name__}({entry[2]})", 100)

    expected, actual = _submodule_layout(rebuilt), _submodule_layout(live)
    for want, have in itertools.zip_longest(expected, actual):
        if want != have:
            name = (have or want)[0]
            return (
                f"its submodules differ from those its recipe builds (at "
                f"{_truncate(repr(name), 80)}: {describe(have)} in the model, "
                f"{describe(want)} from the recipe); every architectural choice must "
                "be a constructor argument, not set after construction"
            )
    return None


def _recipe_problem(
    recipe: Any,
    state_dict: Mapping[str, Any],
    live: Optional[torch.nn.Module] = None,
) -> Optional[str]:
    """
    Why ``recipe`` cannot be trusted to rebuild a module holding ``state_dict``.

    Returns None when it can. The recipe is built on the meta device (no
    memory), so a recipe read from an untrusted file is checked before
    anything is allocated: the constructor must leave every tensor on the meta
    device, ``state_dict()`` (including ``get_extra_state``) must run there,
    and the stored weights must match it in names and shapes with enough
    storage behind them (see ``_state_dict_mismatch``). With ``live``, the
    module being saved, its submodules must also match the rebuild, which
    catches a module changed after construction. A recipe that cannot be
    built at all raises ``build_from_recipe``'s ValueError.
    """
    with torch.device("meta"):
        skeleton = build_from_recipe(recipe)
    off_meta = _off_meta_tensors(skeleton)
    if off_meta:
        return (
            f"its constructor allocated {len(off_meta)} tensor(s) off the meta "
            f"device, e.g. {_truncate(repr(off_meta[0]), 80)}, so its size cannot be "
            "checked before it is built"
        )
    try:
        expected = skeleton.state_dict(keep_vars=True)
    except Exception as err:  # get_extra_state is arbitrary code
        return (
            "its state_dict()/get_extra_state could not run on the meta device "
            f"({_truncate(f'{type(err).__name__}: {err}', 300)})"
        )
    if live is not None:
        difference = _submodule_difference(skeleton, live)
        if difference is not None:
            return difference
    mismatch = _state_dict_mismatch(expected, state_dict)
    if mismatch is not None:
        return f"the recipe does not match the weights ({mismatch})"
    return None


def _require_rebuildable(
    embedding: torch.nn.Module, recipe: dict[str, Any], state_dict: Mapping[str, Any]
) -> None:
    """
    Refuse to save an embedding that the loader could not rebuild from its recipe.

    The loader runs the same ``_recipe_problem`` check; running it here turns
    a file that could never be loaded into an error at save time, and also
    compares the rebuild's submodules with the live module. Likewise extra
    state (``get_extra_state``) holding anything but plain values and tensors,
    which ``torch.load(weights_only=True)`` or ``load_checkpoint`` would refuse.
    """
    for key, value in state_dict.items():
        if _is_extra_state_key(key):
            try:
                _check_plain(value, key, allow_tensors=True, what="Extra state")
            except TypeError as err:
                raise ValueError(
                    f"Cannot save the embedding model ({type(embedding).__name__}): "
                    f"{err}"
                ) from err
    try:
        problem = _recipe_problem(recipe, state_dict, live=embedding)
    except ValueError as err:
        problem = f"building it on the meta device failed ({_truncate(str(err), 300)})"
    if problem is not None:
        raise ValueError(
            f"Cannot save: the embedding model ({type(embedding).__name__}, registered "
            f"as {recipe['builder']!r}) cannot be rebuilt from its recipe: {problem}. "
            "It may have been modified after construction, or its constructor may not "
            "run on the meta device; see the constructor requirements in the "
            "equine.embedding_architecture docstring."
        )


def _embedding_checkpoint(
    embedding: torch.nn.Module, allow_executable: bool, _stacklevel: int = 2
) -> dict[str, Any]:
    """
    The checkpoint entries describing an embedding model.

    A registered architecture is stored as its recipe and ``state_dict``,
    after checking that the recipe rebuilds the same module (see
    ``_require_rebuildable``). An unregistered one is refused unless
    ``allow_executable`` is set, in which case a flagged TorchScript archive
    is stored instead (transition only), with a ``FutureWarning`` reported
    ``_stacklevel`` frames up.
    """
    recipe = embedding_recipe(embedding)
    if recipe is not None:
        state_dict = embedding.state_dict()
        # The check below calls the constructor with the recipe's arguments; keep
        # what is written independent of anything that constructor mutates.
        stored_recipe = copy.deepcopy(recipe)
        _require_rebuildable(embedding, recipe, state_dict)
        return {
            "embedding_recipe": stored_recipe,
            "embedding_state_dict": state_dict,
            "contains_executable": False,
        }
    if allow_executable:
        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(prepare_jit_module(embedding)), buffer)
        warnings.warn(_EXECUTABLE_SAVE_WARNING, FutureWarning, stacklevel=_stacklevel)
        return {
            "embed_jit_save": _jit_archive_to_tensor(buffer),
            "contains_executable": True,
        }
    raise ValueError(
        f"Cannot save: the embedding model ({type(embedding).__name__}) is not a "
        "registered architecture, so it has no recipe to store. Register it with "
        "@equine.embedding_architecture('yourproject.name') and construct it with "
        "plain-valued arguments, or pass save(path, allow_executable=True) to embed a "
        "TorchScript copy (executable content, removed in the next release; the file "
        "will then require load(..., trust_executable=True)). If this model was "
        "loaded from a file that stores its embedding as TorchScript, "
        f"{_MIGRATION_HINT}."
    )


def _load_embedding_weights(
    module: torch.nn.Module, state_dict: Mapping[str, Any]
) -> None:
    """``load_state_dict`` with any failure reported as a bounded ValueError."""
    try:
        module.load_state_dict(state_dict)
    except Exception as err:
        raise ValueError(
            "Could not load the embedding weights stored in the file: "
            f"{_truncate(str(err))}"
        ) from err


def _rebuild_embedding(
    model_save: dict[str, Any],
    device: Optional[str],
    trust_executable: bool,
    embedding_model: Optional[torch.nn.Module],
    _stacklevel: int = 2,
) -> torch.nn.Module:
    """
    Reconstitute an embedding model from a checkpoint.

    Order: a caller-supplied module (weights loaded into it), then the recipe
    (rebuilt through the registry), then a TorchScript archive. Executable
    content requires ``trust_executable`` whichever branch is taken: a file
    with an archive, or with ``contains_executable`` set, is refused without
    it, so a hand-edited file cannot bypass the check by dropping the flag or
    by adding a recipe beside the archive. Returning the archive's module
    emits a ``FutureWarning`` reported ``_stacklevel`` frames up.
    """
    state_dict = model_save.get("embedding_state_dict")
    archive = model_save.get("embed_jit_save")
    # Single audit point for executable content (spec section 5).
    executable = archive is not None or model_save.get("contains_executable") is True
    if executable and not trust_executable:
        raise ValueError(_EXECUTABLE_ERROR)
    if state_dict is not None:
        _require_weights_dict(
            state_dict, "embedding_state_dict", allow_extra_state=True
        )

    if embedding_model is not None:
        if state_dict is None:
            if archive is None:
                raise ValueError(
                    "Model file contains no embedding weights to load into the "
                    "supplied embedding_model (no embedding_state_dict and no archive)."
                )
            state_dict = _load_jit_archive(archive, device).state_dict()
        mismatch = _state_dict_mismatch(
            embedding_model.state_dict(keep_vars=True), state_dict
        )
        if mismatch is not None:
            raise ValueError(
                f"The supplied embedding_model ({type(embedding_model).__name__}) does "
                f"not match the weights stored in the file ({mismatch}); pass a module "
                "of the architecture the file was saved with."
            )
        _load_embedding_weights(embedding_model, state_dict)
        return embedding_model

    if "embedding_recipe" in model_save:
        recipe = model_save["embedding_recipe"]
        # Memory guard: a recipe from an untrusted file could describe an
        # enormous network, so it is checked on the meta device against the
        # file's weights before being built for real (see _recipe_problem).
        # load_checkpoint sized every storage from the bytes actually present
        # in the file's records (mmap, no overlaps) and copied each once, so
        # the real build allocates at most the element size (a small
        # constant) times the weight bytes in the file, on top of that copy.
        problem = _recipe_problem(recipe, state_dict or {})
        if problem is not None:
            raise ValueError(
                f"Refusing to build the embedding recipe {recipe.get('builder')!r} "
                f"stored in the file: {problem}."
            )
        embedding = build_from_recipe(recipe)
        _load_embedding_weights(embedding, state_dict or {})
        return embedding

    if archive is not None:
        warnings.warn(_EXECUTABLE_LOAD_WARNING, FutureWarning, stacklevel=_stacklevel)
        return _load_jit_archive(archive, device)
    raise ValueError(
        "Model file contains no embedding model (no recipe and no archive)."
    )


@icontract.require(lambda y_hat, y_test: y_hat.size(dim=0) == y_test.size(dim=0))
@icontract.ensure(lambda result: result >= 0.0)
@beartype
def brier_score(y_hat: torch.Tensor, y_test: torch.Tensor) -> float:
    """
    Compute the Brier score for a multiclass problem:
    $$ \\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j=1}^{M} (f_{ij} - o_{ij})^2 , $$
    where $f_{ij}$ is the predicted probability of class $j$ for inference sample $i$
    and $o_{ij}$ is the one-hot encoded ground truth label.

    Parameters
    ----------
    y_hat : torch.Tensor
        Probabilities for each class.
    y_test : torch.Tensor
        Integer argument class labels (ground truth).

    Returns
    -------
    float
        Brier score.
    """
    _, num_classes = y_hat.size()
    one_hot_y_test = torch.nn.functional.one_hot(y_test.long(), num_classes=num_classes)
    bs = torch.mean(torch.sum((y_hat - one_hot_y_test) ** 2, dim=1)).item()
    return bs


@icontract.require(lambda y_hat, y_test: y_hat.size(dim=0) == y_test.size(dim=0))
@icontract.ensure(lambda result: result <= 1.0)
@beartype
def brier_skill_score(y_hat: torch.Tensor, y_test: torch.Tensor) -> float:
    """
    Compute the Brier skill score as compared to randomly guessing.

    Parameters
    ----------
    y_hat : torch.Tensor
        Probabilities for each class.
    y_test : torch.Tensor
        Integer argument class labels (ground truth).

    Returns
    -------
    float
        Brier skill score.
    """
    _, num_classes = y_hat.size()
    random_guess = (1.0 / num_classes) * torch.ones(y_hat.size())
    bs0 = brier_score(random_guess, y_test)
    bs1 = brier_score(y_hat, y_test)
    bss = 1.0 - bs1 / bs0
    return bss


@icontract.require(lambda y_hat, y_test: y_hat.size(dim=0) == y_test.size(dim=0))
@icontract.ensure(lambda result: (0.0 <= result) and (result <= 1.0))
@beartype
def expected_calibration_error(y_hat: torch.Tensor, y_test: torch.Tensor) -> float:
    """
    Compute the expected calibration error (ECE) for a multiclass problem.

    Parameters
    ----------
    y_hat : torch.Tensor
        Probabilities for each class.
    y_test : torch.Tensor
        Class label indices (ground truth).

    Returns
    -------
    float
        Expected calibration error.
    """
    _, num_classes = y_hat.size()
    metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=25, norm="l1")
    ece = metric(y_hat, y_test).item()
    return ece


@icontract.require(
    lambda train_y, selected_labels: len(selected_labels) <= len(train_y)
)
@icontract.ensure(
    lambda result, selected_labels: set(result.keys()).issubset(set(selected_labels))
)
@beartype
def _get_shuffle_idxs_by_class(
    train_y: torch.Tensor, selected_labels: list
) -> dict[Any, torch.Tensor]:
    """
    Internal helper function to randomly select indices of example classes for a given
    set of labels.

    Parameters
    ----------
    train_y : torch.Tensor
        Label data.
    selected_labels : list
        list of unique labels found in the label data.

    Returns
    -------
    dict[Any, torch.Tensor]
        Tensor of indices corresponding to each label.
    """
    shuffled_idxs_by_class = OrderedDict()
    for label in selected_labels:
        label_idxs = torch.argwhere(train_y == label).squeeze()
        shuffled_idxs_by_class[label] = label_idxs[torch.randperm(label_idxs.shape[0])]

    return shuffled_idxs_by_class


@icontract.require(lambda train_x, train_y: len(train_x) <= len(train_y))
@icontract.require(
    lambda selected_labels, train_x: (
        (0 < len(selected_labels)) & (len(selected_labels) < len(train_x))
    )
)
@icontract.require(
    lambda support_size, train_x: (0 < support_size) & (support_size < len(train_x))
)
@icontract.require(
    lambda support_size, selected_labels, train_x: (
        support_size * len(selected_labels) <= len(train_x)
    )
)
@icontract.require(
    lambda selected_labels, shuffled_indexes: (
        (len(shuffled_indexes.keys()) == len(selected_labels))
        if shuffled_indexes is not None
        else True
    )
)
@icontract.ensure(
    lambda result, selected_labels: len(result.keys()) == len(selected_labels)
)
@beartype
def generate_support(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    support_size: int,
    selected_labels: list[Any],
    shuffled_indexes: Union[None, dict[Any, torch.Tensor]] = None,
) -> OrderedDict[int, torch.Tensor]:
    """
    Randomly select `support_size` examples of `way` classes from the examples in
    `train_x` with corresponding labels in `train_y` and return them as a dictionary.

    Parameters
    ----------
    train_x : torch.Tensor
        Input training data.
    train_y : torch.Tensor
        Corresponding classification labels.
    support_size : int
        Number of support examples for each class.
    selected_labels : list
        Selected class labels to generate examples from.
    shuffled_indexes: Union[None, dict[Any, torch.Tensor]], optional
        Simply use the precomputed indexes if they are available

    Returns
    -------
    OrderedDict[int, torch.Tensor]
        Ordered dictionary of class labels with corresponding support examples.
    """
    labels, counts = torch.unique(train_y, return_counts=True)
    if shuffled_indexes is None:
        for label, count in list(zip(labels, counts)):
            if (label in selected_labels) and (count < support_size):
                raise ValueError(f"Not enough support examples in class {label}")
        shuffled_idxs = _get_shuffle_idxs_by_class(train_y, selected_labels)
    else:
        shuffled_idxs = shuffled_indexes

    support = OrderedDict[int, torch.Tensor]()
    for label in selected_labels:
        shuffled_x = train_x[shuffled_idxs[label]]

        assert torch.unique(train_y[shuffled_idxs[label]]).tolist() == [label], (
            "Not enough support for label " + str(label)
        )
        selected_support = shuffled_x[:support_size]
        support[int(label)] = selected_support

    return support


@icontract.require(lambda train_x: len(train_x.shape) >= 2)
@icontract.require(lambda train_y: len(train_y.shape) == 1)
@icontract.require(lambda support_size: support_size > 1)
@icontract.require(lambda way: way > 0)
@icontract.require(lambda episode_size: episode_size > 0)
@icontract.ensure(lambda result: len(result) == 3)
@icontract.ensure(lambda result: result[1].shape[0] == result[2].shape[0])
@icontract.ensure(lambda way, result: len(result[0]) == way)
@icontract.ensure(
    lambda support_size, result: all(
        len(support) == support_size for support in result[0].values()
    )
)
@beartype
def generate_episode(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    support_size: int,
    way: int,
    episode_size: int,
) -> tuple[OrderedDict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Generate a single episode of data for a few-shot learning task.

    Parameters
    ----------
    train_x : torch.Tensor
        Input training data.
    train_y : torch.Tensor
        Corresponding classification labels.
    support_size : int
        Number of support examples for each class.
    way : int
        Number of classes in the episode.
    episode_size : int
        Total number of examples in the episode.

    Returns
    -------
    tuple[dict[Any, torch.Tensor], torch.Tensor, torch.Tensor]
        tuple of support examples, query examples, and query labels.
    """
    labels, counts = torch.unique(train_y, return_counts=True)
    if way > len(labels):
        raise ValueError(
            f"The way (#classes in each episode), {way}, must be <= number of labels, {len(labels)}"
        )

    selected_labels = sorted(
        labels[torch.randperm(labels.shape[0])][:way].tolist()
    )  # need to be in same order every time

    for label, count in list(zip(labels, counts)):
        if (label in selected_labels) and (count < support_size):
            raise ValueError(f"Not enough support examples in class {label}")
    shuffled_idxs = _get_shuffle_idxs_by_class(train_y, selected_labels)

    support = generate_support(
        train_x, train_y, support_size, selected_labels, shuffled_idxs
    )

    examples_per_task = episode_size // way

    episode_data_list = []
    episode_label_list = []
    episode_support = OrderedDict()
    for episode_label, label in enumerate(selected_labels):
        shuffled_x = train_x[shuffled_idxs[label]]
        shuffled_y = torch.Tensor(
            [episode_label] * len(shuffled_idxs[label])
        )  # need sequential labels for episode

        num_remaining_examples = shuffled_x.shape[0] - support_size
        assert num_remaining_examples > 0, (
            "Cannot have "
            + str(num_remaining_examples)
            + " left with support_size "
            + str(support_size)
            + " and shape "
            + str(shuffled_x.shape)
            + " from train_x shaped "
            + str(train_x.shape)
        )
        episode_end_idx = support_size + min(num_remaining_examples, examples_per_task)

        episode_data_list.append(shuffled_x[support_size:episode_end_idx])
        episode_label_list.append(shuffled_y[support_size:episode_end_idx])
        episode_support[episode_label] = support[label]

    episode_x = torch.concat(episode_data_list)
    episode_y = torch.concat(episode_label_list)

    return episode_support, episode_x, episode_y.squeeze().to(torch.long)


@icontract.require(
    lambda eq_preds, true_y: eq_preds.classes.size(dim=0) == true_y.size(dim=0)
)
@beartype
def generate_model_metrics(
    eq_preds: EquineOutput, true_y: torch.Tensor
) -> dict[str, Any]:
    """
    Generate various metrics for evaluating a model's performance.

    Parameters
    ----------
    eq_preds : EquineOutput
        Model predictions.
    true_y : torch.Tensor
        True class labels.

    Returns
    -------
    dict[str, Any]
        Dictionary of model metrics.
    """
    pred_y = torch.argmax(eq_preds.classes, dim=1)
    accuracy = MulticlassAccuracy(num_classes=eq_preds.classes.shape[1])
    f1_score = MulticlassF1Score(num_classes=eq_preds.classes.shape[1], average="micro")
    confusion_matrix = MulticlassConfusionMatrix(num_classes=eq_preds.classes.shape[1])
    metrics = {
        "accuracy": accuracy(true_y, pred_y),
        "microF1Score": f1_score(true_y, pred_y),
        "confusionMatrix": confusion_matrix(true_y, pred_y).tolist(),
        "brierScore": brier_score(eq_preds.classes, true_y),
        "brierSkillScore": brier_skill_score(eq_preds.classes, true_y),
        "expectedCalibrationError": expected_calibration_error(
            eq_preds.classes, true_y
        ),
    }
    return metrics


@icontract.require(lambda Y: len(Y.shape) == 1)
@icontract.ensure(
    lambda result: all("label" in d and "numExamples" in d for d in result)
)
@icontract.ensure(lambda result: all(d["numExamples"] >= 0 for d in result))
@beartype
def get_num_examples_per_label(Y: torch.Tensor) -> list[dict[str, Any]]:
    """
    Get the number of examples per label in the given tensor.

    Parameters
    ----------
    Y : torch.Tensor
        Tensor of class labels.

    Returns
    -------
    list[dict[str, Any]]
        list of dictionaries containing label and number of examples.
    """
    tensor_labels, tensor_counts = Y.unique(return_counts=True)

    examples_per_label = []
    for i, label in enumerate(tensor_labels):
        examples_per_label.append(
            {"label": label.item(), "numExamples": tensor_counts[i].item()}
        )

    return examples_per_label


@icontract.require(lambda train_y: train_y.shape[0] > 0)
@beartype
def generate_train_summary(
    model: Equine, train_y: torch.Tensor, date_trained: str
) -> dict[str, Any]:
    """
    Generate a summary of the training data.

    Parameters
    ----------
    model : Equine
        Model object.
    train_y : torch.Tensor
        Training labels.
    date_trained : str
        Date of training.

    Returns
    -------
    dict[str, Any]
        Dictionary containing training summary.
    """
    train_summary = {
        "numTrainExamples": get_num_examples_per_label(train_y),
        "dateTrained": date_trained,
        "modelType": model.__class__.__name__,
    }
    return train_summary


@icontract.require(
    lambda eq_preds, test_y: test_y.shape[0] == eq_preds.classes.shape[0]
)
@beartype
def generate_model_summary(
    model: Equine,
    eq_preds: EquineOutput,
    test_y: torch.Tensor,
) -> dict[str, Any]:
    """
    Generate a summary of the model's performance.

    Parameters
    ----------
    model : Equine
        Model object.
    eq_preds : EquineOutput
        Model predictions.
    test_y : torch.Tensor
        True class labels.

    Returns
    -------
    dict[str, Any]
        Dictionary containing model summary.
    """
    summary = generate_model_metrics(eq_preds, test_y)
    summary["numTestExamples"] = get_num_examples_per_label(test_y)
    summary.update(model.train_summary)  # union of train_summary and generated metrics

    return summary


@icontract.require(lambda cov: cov.shape[-2] == cov.shape[-1])
def mahalanobis_distance_nosq(x: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """
    Compute Mahalanobis distance $x^T C x$ (without square root), assume cov is symmetric positive definite

    Parameters
    ----------
    x : torch.Tensor
        vectors to compute distances for
    cov : torch.Tensor
        covariance matrix, assumes first dimension is number of classes
    """
    U, S, _ = torch.linalg.svd(cov)
    S_inv_sqrt = torch.stack(
        [torch.diag(torch.sqrt(1.0 / S[i])) for i in range(S.shape[0])], dim=0
    )
    prod = torch.matmul(S_inv_sqrt, torch.transpose(U, 1, 2))
    dist = torch.sum(torch.square(torch.matmul(prod, x)), dim=1)
    return dist


@icontract.require(
    lambda X, Y: X.shape[0] == Y.shape[0],
    "X and Y must have the same number of samples.",
)
@icontract.require(
    lambda test_size: 0.0 < test_size < 1.0, "test_size must be between 0 and 1."
)
@icontract.ensure(
    lambda result: len(result) == 4, "Function must return four elements."
)
@icontract.ensure(
    lambda X, result: result[0].shape[0] + result[1].shape[0] == X.shape[0],
    "Total samples must be preserved.",
)
@icontract.ensure(
    lambda Y, result: result[2].shape[0] + result[3].shape[0] == Y.shape[0],
    "Total labels must be preserved.",
)
@icontract.ensure(
    lambda result: result[0].shape[0] == result[2].shape[0],
    "Train features and labels must match in size.",
)
@icontract.ensure(
    lambda result: result[1].shape[0] == result[3].shape[0],
    "Test features and labels must match in size.",
)
@beartype
def stratified_train_test_split(
    X: torch.Tensor, Y: torch.Tensor, test_size: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A pytorch-ified version of sklearn's train_test_split with data stratification

    Parameters
    ----------
    X : torch.Tensor
        Input features tensor of shape (n_samples, n_features).
    Y : torch.Tensor
        Labels tensor of shape (n_samples,).
    test_size : float
        Proportion of the dataset to include in the test split (between 0.0 and 1.0).

    Returns
    -------
    train_x : torch.Tensor
        Training set features.
    calib_x : torch.Tensor
        Test set features.
    train_y : torch.Tensor
        Training set labels.
    calib_y : torch.Tensor
        Test set labels.
    """
    unique_classes, class_counts = torch.unique(Y, return_counts=True)
    test_counts = (class_counts.float() * test_size).round().long()
    train_indices = []
    test_indices = []

    for cls, test_count in zip(unique_classes, test_counts):
        cls_indices = torch.where(Y == cls)[0]
        cls_indices = cls_indices[torch.randperm(len(cls_indices))]
        test_idx = cls_indices[:test_count]
        train_idx = cls_indices[test_count:]
        train_indices.append(train_idx)
        test_indices.append(test_idx)

    train_indices = torch.cat(train_indices)
    test_indices = torch.cat(test_indices)

    train_x = X[train_indices]
    train_y = Y[train_indices]
    calib_x = X[test_indices]
    calib_y = Y[test_indices]

    return train_x, calib_x, train_y, calib_y
