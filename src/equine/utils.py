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
from typing import Any

import icontract
import torch
from beartype import beartype
from scipy.stats import gaussian_kde
from torch import _weights_only_unpickler
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
    _describe_error,
    _message_or_type,
    _truncate,
    build_from_recipe,
    embedding_recipe,
)

# Layout version written by ``save``. Version 2 holds only tensors, containers
# and plain scalars/strings, so ``torch.load(weights_only=True)`` reads it.
# The embedding is a recipe plus a state_dict (see ``equine.registry``).
EQUINE_FORMAT_VERSION = 2

_UPGRADE_HINT = "python -m equine.upgrade"

_UPGRADE_USAGE = (
    f"`{_UPGRADE_HINT} {{path}} --recipe '<recipe JSON>' [--import MODULE]`, or "
    "`from equine.upgrade import upgrade_model_file; upgrade_model_file({path!r}, "
    "<an instance of your registered architecture>)`"
)

_TORCHSCRIPT_FILE_ERROR = (
    "'{path}' stores its embedding model as TorchScript, which EQUINE does not "
    "load. Rewrite it once with "
    + _UPGRADE_USAGE
    + ". Only upgrade files you trust: loading TorchScript runs code from the file."
)

_TORCHSCRIPT_CHECKPOINT_ERROR = (
    "Model file stores its embedding model as TorchScript, which EQUINE does not "
    f"load. Rewrite it once with `{_UPGRADE_HINT}`. Only upgrade files you trust: "
    "loading TorchScript runs code from the file."
)

_LEGACY_FILE_ERROR = (
    "'{path}' is not in the current EQUINE model format: it has no "
    "equine_format_version, or it holds pickled Python objects, as files saved "
    "by earlier EQUINE releases do. Rewrite it once with "
    + _UPGRADE_USAGE
    + ". Only upgrade files you trust: the upgrade unpickles the file "
    "without restrictions, which can run code embedded in it."
)


class LegacyFormatError(ValueError):
    """The file predates the current format; ``python -m equine.upgrade`` rewrites it."""


# torch < 2.6 has a known bypass of weights_only=True (CVE-2025-32434).
_MIN_SAFE_TORCH = (2, 6)

_OLD_TORCH_ERROR = (
    "EQUINE needs torch >= 2.6 to read model files (installed: {version}): "
    "torch.load(weights_only=True) has a known bypass on earlier releases "
    "(CVE-2025-32434)."
)


_STORAGE_TRICK_ERROR = (
    "Model file contains a tensor whose storage is smaller than its shape "
    "(expanded or overlapping view); refusing to load."
)

_NOT_PLAIN_ARCHIVE_ERROR = (
    "Model file is not a plain (uncompressed) PyTorch archive; refusing to load."
)

_PICKLE_CALL_ERROR = (
    "Model file's pickle uses {name}, which EQUINE model files do not contain; "
    "refusing to load."
)

_PICKLE_SIZE_ERROR = (
    "Model file's pickle copies more items than it has bytes; refusing to load."
)

_PICKLE_STORAGE_CALL_ERROR = (
    "Model file's pickle calls {name}, which EQUINE model files only reference as a "
    "tensor's storage; refusing to load."
)

# Functions, storage classes and dtypes a model file's pickle may name.
_PICKLE_FUNCTIONS = frozenset(
    {
        "torch._utils._rebuild_tensor_v2",
        "torch._utils._rebuild_tensor_v3",
        "torch._utils._rebuild_parameter",
    }
)
_PICKLE_STORAGE_CLASSES = frozenset(
    f"torch.{kind}Storage"
    for kind in (
        "Float",
        "Double",
        "Half",
        "BFloat16",
        "Long",
        "Int",
        "Short",
        "Char",
        "Byte",
        "Bool",
        "ComplexFloat",
        "ComplexDouble",
    )
) | {"torch.storage.UntypedStorage"}
_PICKLE_DTYPES = {
    f"torch.{name}": value
    for name, value in vars(torch).items()
    if isinstance(value, torch.dtype)
}

# torch.load reads a file as a zip archive exactly when it starts with this
# local-file-header signature (torch.serialization._is_zipfile).
_ZIP_SIGNATURE = b"PK\x03\x04"

_NO_STORAGE_ERROR = (
    "Model file contains a sparse or meta tensor, which has no dense storage "
    "backing its shape; refusing to load."
)

_STORAGE_SIZE_ERROR = (
    "Model file declares tensor storage larger than the data it contains; "
    "refusing to load."
)

_ALIASED_SUPPORT_ERROR = (
    "Model file's support/KDE tensors alias one another, which lets a small file "
    "multiply the memory and work of loading it; refusing to load. The current "
    f"format stores each label's data separately; `{_UPGRADE_HINT}` rewrites a "
    "trusted file from an earlier release with the tensors copied apart."
)

_SET_ERROR = "Model file contains a set; EQUINE files never do; refusing to load."


def _torch_version() -> tuple[int, int]:
    """Return the installed torch version as an ``(major, minor)`` tuple."""
    major, minor = (int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    return (major, minor)


def _warn_if_torch_crashes_with_threads() -> None:
    """
    Warn on macOS with Python 3.15 or later and torch older than 2.15, where
    torch can crash when it runs on several threads at once.
    """
    if (
        sys.platform == "darwin"
        and sys.version_info >= (3, 15)
        and _torch_version() < (2, 15)
    ):
        warnings.warn(
            f"torch {torch.__version__} can crash on macOS with Python 3.15 when "
            "torch runs on several threads at once (for example, two threads "
            "calling predict). Install torch >= 2.15; until it is on PyPI: "
            "pip install --upgrade 'torch>=2.15' "
            "--index-url https://download.pytorch.org/whl/test/cpu",
            RuntimeWarning,
            stacklevel=3,
        )


def load_checkpoint(path: str, map_location: str | None = None) -> dict[str, Any]:
    """
    Read a saved EQUINE model file with ``torch.load(weights_only=True)``.

    Files from earlier releases hold pickled Python objects and are refused;
    ``python -m equine.upgrade`` rewrites them in the current format.

    Memory is bounded by the bytes present in the file. The file must be a plain zip
    archive (see ``_check_zip_archive``) whose pickle names only what ``save`` writes
    (see ``_check_pickle_globals``), and is mapped with ``mmap=True``; each storage is a
    slice of the file. Expanded or overlapping views, meta and sparse tensors, and
    support/KDE tensors that alias one another are refused. Every storage is then
    copied once off the mapping, onto ``map_location``.

    The file contains no executable code: the embedding model is stored as a recipe and
    weights.

    Parameters
    ----------
    path : str
        Filename of the saved model.
    map_location : str | None
        Device to move the loaded tensors to; with None they stay on the CPU,
        whatever device they were saved from. It must name the CPU or an
        accelerator available on this machine, with an index below its device count;
        anything else, ``meta`` included, is refused before the file is opened.

    Returns
    -------
    dict[str, Any]
        The saved checkpoint dictionary.

    Raises
    ------
    ValueError
        If ``map_location`` names a device this machine cannot build on, or an
        accelerator without a dtype the file holds (float64 on MPS). If torch
        is older than 2.6, if the file is not in the current format (not a zip
        archive, pickled objects, or no ``equine_format_version``), if it
        declares a format version this EQUINE does not know, or if its layout
        could exhaust memory (compressed or oversized entries, a pickle global
        ``save`` does not write, no dictionary at the top level, a set, a tensor
        whose storage is smaller than its shape, storages that overlap in the
        file, aliased support/KDE tensors).
    """
    if map_location is not None:
        # Checked before the file is opened.
        _require_device({"device": map_location}, what="The requested device")
    if _torch_version() < _MIN_SAFE_TORCH:
        raise ValueError(_OLD_TORCH_ERROR.format(version=torch.__version__))
    if not _check_zip_archive(path):
        raise LegacyFormatError(_LEGACY_FILE_ERROR.format(path=path))
    _check_pickle_globals(path)
    try:
        # torch reopens the file; a swapped file skips the zip and pickle pre-checks.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except pickle.UnpicklingError as err:
        raise LegacyFormatError(_LEGACY_FILE_ERROR.format(path=path)) from err
    except Exception as err:  # malformed archive: report it as a ValueError
        if "resiz" in str(err):  # torch refusing a tensor larger than its storage
            raise ValueError(_STORAGE_SIZE_ERROR) from err
        raise ValueError(
            f"Model file could not be read as a PyTorch archive "
            f"({_truncate(str(err), 200)}); refusing to load."
        ) from err
    _require_dict(checkpoint, path)
    _validate_format_version(checkpoint)
    tensors = _unique_tensors(checkpoint)
    _reject_storage_tricks(tensors)
    _reject_overlapping_storages(tensors)
    # Checked before the copy below.
    if _support_is_aliased(checkpoint):
        raise ValueError(_ALIASED_SUPPORT_ERROR)
    if "equine_format_version" not in checkpoint:
        raise LegacyFormatError(_LEGACY_FILE_ERROR.format(path=path))
    if map_location is not None:
        _require_supported_dtypes(tensors, torch.device(map_location))
    _copy_off_the_file(tensors, map_location)
    return checkpoint


def _load_checkpoint_for_device(path: str, device: str | None) -> dict[str, Any]:
    """
    ``load_checkpoint(path)`` for a model loader that builds on ``device``.

    The tensors stay on the CPU; the loader builds the model from them and
    moves it to ``device``. An explicit ``device`` is checked as
    ``map_location`` is: before the file is opened, and against the dtypes
    of the tensors the file holds.
    """
    if device is not None:
        _require_device({"device": device}, what="The requested device")
    checkpoint = load_checkpoint(path)
    if device is not None:
        _require_supported_dtypes(_unique_tensors(checkpoint), torch.device(device))
    return checkpoint


def _validate_format_version(checkpoint: Any) -> None:
    """
    Raise if a loaded checkpoint declares a format version newer than this
    EQUINE build understands.

    A missing key is reported by the caller. A present key must be an ``int``
    (not a ``bool``); a huge one is not formatted into the message.
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


def _plain_names(names: list[str] | None) -> list[str] | None:
    """
    ``feature_names`` / ``label_names`` as plain ``str`` for a checkpoint.

    Names often come from numpy (``LabelEncoder.inverse_transform`` returns
    ``numpy.str_``), which ``torch.load(weights_only=True)`` refuses to unpickle.
    """
    return None if names is None else [str(name) for name in names]


def _plain_train_summary(summary: Any) -> Any:
    """
    Return ``summary`` after the plain-value check, accepting tensors (not quantized,
    without Python attributes), ``torch.Size`` and ``OrderedDict`` too, and str, int,
    float, bool and None dict keys. A refused entry raises ``ValueError``.
    """
    try:
        _check_plain(
            {"train_summary": summary},
            allow_tensors=True,
            what="Training summary entry",
            owner="the model",
            key_types=(str, int, float, bool, type(None)),
        )
    except TypeError as err:
        raise ValueError(f"Cannot save: {err}") from err
    return summary


def _require_dict(checkpoint: Any, path: str) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"'{path}' is not an EQUINE model file (expected a dictionary at the "
            "top level)."
        )


def _check_zip_archive(path: str) -> bool:
    """
    Refuse a zip archive that ``torch.load`` could inflate beyond the file's size.

    DEFLATE entries are refused here, before ``torch.load`` runs, and so is an archive
    whose entries add up to more than the file. A file that torch would read as a zip
    but the ``zipfile`` module cannot parse is refused too.

    Returns whether the file is a zip archive at all. The caller refuses a non-zip file
    (torch's pre-1.6 format); ``python -m equine.upgrade`` rewrites such files.
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


class _PickleRefusal(ValueError):
    """A global or call the dry run of a model file's pickle refuses."""


class _Placeholder:
    """What the dry run of a model file's pickle builds for a tensor or storage."""


class _StorageClass:
    """A storage class named by the pickle; only a storage reference may hold it."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: Any) -> None:
        raise _PickleRefusal(_PICKLE_STORAGE_CALL_ERROR.format(name=self.name))


def _sizes(args: tuple[Any, ...]) -> int:
    """The number of arguments plus the lengths of those that have one."""
    return len(args) + sum(len(arg) for arg in args if hasattr(arg, "__len__"))


class _DryRunUnpickler(pickle._Unpickler):
    """
    Unpickle with placeholders for tensors and storages, refusing other globals.

    The calls it allows and the state ``BUILD`` sets are charged their number of
    arguments plus the lengths of those arguments; together they may use at most
    ``budget`` items. ``NEWOBJ`` is refused.
    """

    dispatch = dict(pickle._Unpickler.dispatch)
    stack: list[Any]

    def __init__(self, file: Any, budget: int) -> None:
        super().__init__(file)
        self.budget = budget

    def _spend(self, items: int) -> None:
        self.budget -= items
        if self.budget < 0:
            raise _PickleRefusal(_PICKLE_SIZE_ERROR)

    def _counted(self, build: Any) -> Any:
        def call(*args: Any) -> Any:
            self._spend(_sizes(args))
            return build(*args)

        return call

    def _rebuild(self, source: Any, *args: Any) -> _Placeholder:
        if not isinstance(source, _Placeholder):
            raise _PickleRefusal(
                _PICKLE_CALL_ERROR.format(name="a tensor without a storage")
            )
        self._spend(_sizes((source, *args)))
        return _Placeholder()

    def load_build(self) -> None:
        state = self.stack[-1]
        self._spend(_sizes(state if isinstance(state, tuple) else (state,)))
        pickle._Unpickler.load_build(self)  # type: ignore

    dispatch[pickle.BUILD[0]] = load_build  # type: ignore

    def load_newobj(self) -> None:
        raise _PickleRefusal(_PICKLE_CALL_ERROR.format(name="NEWOBJ"))

    dispatch[pickle.NEWOBJ[0]] = load_newobj  # type: ignore

    def find_class(self, module: str, name: str) -> Any:
        full_name = f"{module}.{name}"
        if full_name == "collections.OrderedDict":
            return self._counted(OrderedDict)
        if full_name == "torch.Size":
            return self._counted(tuple)
        if full_name in ("__builtin__.set", "builtins.set"):
            return self._counted(set)
        if full_name in _PICKLE_DTYPES:
            return _PICKLE_DTYPES[full_name]
        if full_name in _PICKLE_FUNCTIONS:
            return self._rebuild
        if full_name in _PICKLE_STORAGE_CLASSES:
            return _StorageClass(full_name)
        raise _PickleRefusal(_PICKLE_CALL_ERROR.format(name=_truncate(full_name, 100)))

    def persistent_load(self, pid: Any) -> _Placeholder:
        if not (
            type(pid) is tuple
            and len(pid) == 5
            and pid[0] == "storage"
            and isinstance(pid[1], _StorageClass)
        ):
            raise _PickleRefusal(
                _PICKLE_CALL_ERROR.format(name="a malformed storage reference")
            )
        return _Placeholder()


def _check_pickle_globals(path: str) -> None:
    """
    Refuse a pickle that names a global ``save`` does not write, before ``torch.load``.

    The pickle checked is the record ``torch.load`` reads. A global outside torch's
    default ``weights_only`` allowlist marks a file from an earlier release
    (``LegacyFormatError``). Of the globals torch allows, a model file may name only
    ``OrderedDict``, ``torch.Size``, ``set``, torch dtypes, the tensor and parameter
    rebuild functions, and the storage classes, which only a storage reference may
    hold. The pickle is then unpickled with placeholders for tensors and storages; that
    dry run calls no torch function, and the arguments of the calls it allows, counted
    with their lengths, add up to at most as many items as the pickle has bytes.
    """
    try:
        with (
            open(path, "rb") as file,
            torch.serialization._open_zipfile_reader(file) as archive,
        ):
            pickled = archive.get_record("data.pkl")
        names = _weights_only_unpickler.get_globals_in_pkl(io.BytesIO(pickled))
    except Exception as err:  # malformed pickle or archive
        raise ValueError(
            f"Model file could not be read as a PyTorch archive "
            f"({_truncate(str(err), 200)}); refusing to load."
        ) from err
    if not names <= _weights_only_unpickler._get_allowed_globals().keys():
        raise LegacyFormatError(_LEGACY_FILE_ERROR.format(path=path))
    try:
        _DryRunUnpickler(io.BytesIO(pickled), budget=len(pickled)).load()
    except _PickleRefusal:
        raise
    except Exception as err:  # malformed pickle
        raise ValueError(
            f"Model file could not be read as a PyTorch archive "
            f"({_truncate(str(err), 200)}); refusing to load."
        ) from err


def _has_honest_storage(tensor: torch.Tensor) -> bool:
    """Whether every element of a dense tensor has its own bytes in its storage."""
    if tensor.numel() == 0:
        return True
    if tensor.untyped_storage().nbytes() < tensor.numel() * tensor.element_size():
        return False
    # No two elements may share an address: with the dimensions sorted by stride, each
    # stride must step past everything the smaller strides can reach. Size-1 dimensions
    # are skipped.
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

    Walks without recursion and tolerates cycles; other objects are not inspected. A set
    or frozenset is refused.
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

    Meta tensors (a shape with no data), non-strided tensors, and expanded or
    overlapping views (a huge shape over a few bytes) are refused.
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
    Refuse tensors whose storages overlap in the file without being the same slice.

    With ``mmap=True`` every storage is a slice of one mapping of the file: as long as
    the pickle claims, cut at the end of the file. torch refuses a tensor whose shape
    needs bytes past that cut. Overlapping slices are refused here, except identical
    ones: storages with the same slice count as one, and ``_copy_off_the_file`` copies
    it once. A slice that runs past its own record into bytes no other storage of
    ``tensors`` uses (zip headers, padding, other records, trailing metadata) is not
    detected; it is bounded by the file's size.
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

    An untrusted file must keep every container and tensor under ``support`` and
    ``outlier_kde`` distinct, with no two of those tensors sharing a storage. Runs
    before ``_copy_off_the_file``.
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
    Give each label its own copy of its support and KDE data (used by
    ``equine.upgrade``).

    ``support`` maps labels to tensors, and ``outlier_kde`` maps labels to ``{"dataset":
    tensor, ...}`` states. The per-label containers are rebuilt too. No container is
    shared between labels.
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


def _require_supported_dtypes(
    tensors: list[torch.Tensor], device: torch.device
) -> None:
    """
    Refuse ``tensors`` if ``device`` lacks one of their dtypes (MPS has no float64).

    A file saved from a float64 model, or holding float64 support, loads onto
    MPS only with ``device="cpu"``.
    """
    for dtype in dict.fromkeys(tensor.dtype for tensor in tensors):
        try:
            torch.empty(0, dtype=dtype, device=device)
        except (TypeError, RuntimeError) as err:
            raise ValueError(
                f"Model file holds a {str(dtype).removeprefix('torch.')} tensor, "
                f"which {device.type} does not support; load it with device='cpu'"
            ) from err


def _copy_off_the_file(tensors: list[torch.Tensor], map_location: str | None) -> None:
    """
    Give every tensor its own copy of its storage, on ``map_location``.

    Tensors loaded by the safe path (``mmap=True``) first view a mapping of the file.
    After the copy they stay valid if the file later changes or shrinks. Each storage is
    copied once and its tensors are re-pointed in place. Views and tied tensors keep
    sharing, containers are untouched, and the copies add up to no more than the
    storages already checked.
    """
    device = torch.device(map_location if map_location is not None else "cpu")
    copies: dict[tuple[int, int], torch.UntypedStorage] = {}
    with torch.no_grad():
        for tensor in tensors:
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in copies:
                # Copied through a uint8 view of the storage.
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


def _is_extra_state_key(key: str) -> bool:
    """Whether ``key`` is a module's ``get_extra_state`` entry in a state_dict."""
    return key == "_extra_state" or key.endswith("._extra_state")


def _state_dict_mismatch(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> str | None:
    """
    Check that ``actual`` (weights from a file) can fill ``expected`` (a module).

    ``expected`` should come from ``state_dict(keep_vars=True)``. Parameter names and
    shapes must agree, and the distinct storages behind ``actual`` must hold at least
    one byte per element of ``expected``. Extra-state entries are skipped and left to
    ``load_state_dict``.
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


def _dtype_kind(dtype: torch.dtype) -> str:
    """Whether ``dtype`` is floating point, complex, or neither (integer or bool)."""
    if dtype.is_floating_point:
        return "floating-point"
    if dtype.is_complex:
        return "complex"
    return "integer or bool"


def _dtype_kind_mismatch(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> str | None:
    """
    Describe the first tensor whose dtype kind differs between ``expected``
    (a module's tensors) and ``actual`` (weights from a file), if any.

    ``load_state_dict`` converts each stored tensor to the dtype of the
    module's tensor without complaint, even floating-point weights into an
    integer buffer. Extra-state entries, which it hands to ``set_extra_state``
    unchanged, are skipped. The description's length does not depend on the
    key names.
    """
    for key, tensor in expected.items():
        stored = actual.get(key)
        if _is_extra_state_key(key) or not (
            isinstance(tensor, torch.Tensor) and isinstance(stored, torch.Tensor)
        ):
            continue
        have, want = _dtype_kind(stored.dtype), _dtype_kind(tensor.dtype)
        if have != want:
            return (
                f"{_truncate(repr(key), 80)} is {have} in the file but {want} in "
                "the model"
            )
    return None


def _adopt_stored_dtypes(
    module: torch.nn.Module, stored: Mapping[str, Any], key: str
) -> None:
    """
    Give each floating-point and complex tensor of ``module`` the dtype of the
    tensor stored under its name in ``model_save[key]``.

    A module is built at the dtypes its constructor chooses (usually torch's default
    dtype). Converting each parameter and buffer in place first keeps every stored
    precision, tensor by tensor, and keeps tied parameters tied. A stored tensor of
    another kind is refused (see ``_dtype_kind_mismatch``); names the module lacks are
    left to ``load_state_dict``.
    """
    owned = dict(
        itertools.chain(
            module.named_parameters(remove_duplicate=False),
            module.named_buffers(remove_duplicate=False),
        )
    )
    mismatch = _dtype_kind_mismatch(owned, stored)
    if mismatch is not None:
        raise ValueError(
            f"Model file's {key} does not match the model it describes "
            f"({mismatch}); the file is inconsistent or was tampered with."
        )
    for name, tensor in owned.items():
        value = stored.get(name)
        if (
            isinstance(value, torch.Tensor)
            and tensor.dtype != value.dtype
            and (tensor.is_floating_point() or tensor.is_complex())
        ):
            tensor.data = tensor.data.to(value.dtype)


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
    ``settings`` would build; the check runs before the model is constructed (see
    ``_state_dict_mismatch``).
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


def _input_to_model(
    X: torch.Tensor, module: torch.nn.Module, device: str
) -> torch.Tensor:
    """
    Move ``X`` to ``device`` for ``module``, casting a floating input to its dtype.

    A floating-point input is cast to the dtype of the module's first registered
    parameter (a mixed-dtype module follows whichever parameter is registered first). A
    non-floating input (integer indices for an ``nn.Embedding``, booleans) keeps its
    dtype and is only moved, as is any input to a module without parameters.
    """
    param = next(module.parameters(), None)
    if param is None or not X.is_floating_point():
        return X.to(device=device)
    return X.to(device=device, dtype=param.dtype)


def _require_examples(X: torch.Tensor) -> None:
    """
    Refuse an empty input batch.

    Raises
    ------
    ValueError
        If ``X`` has no elements.
    """
    if X.numel() == 0:
        raise ValueError("X is empty; predict needs at least one example.")


def _require_finite_embeddings(embeddings: torch.Tensor) -> None:
    """
    Refuse embeddings that hold a NaN or infinite value.

    Raises
    ------
    ValueError
        If ``embeddings`` holds a NaN or infinite value.
    """
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError(
            "The embeddings of X hold NaN or infinite values; check X and the "
            "embedding model."
        )


def _label_text(label: Any) -> str:
    """``label`` as text, with an integral float shown as an integer."""
    if isinstance(label, float) and label.is_integer():
        return str(int(label))
    return str(label)


def _require_device(
    settings: dict[str, Any],
    hint: str = "",
    what: str = "Model file's settings['device']",
) -> None:
    """
    Refuse a file's ``settings["device"]`` unless this machine can build on it.

    It must name a torch device that is the CPU or an accelerator available here, with
    an index below that accelerator's device count when it has one (``cuda:7`` on a one-
    GPU machine, or ``mps:1``, is refused); ``meta`` is refused. ``hint`` is appended to
    the message for an unavailable device or index. ``what`` opens every message and
    names the value being checked (``load_checkpoint`` and
    ``_load_checkpoint_for_device`` pass "The requested device" for a caller's
    ``map_location`` or ``device``).
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
            f"{what} is not a torch device name ({_shown(value)})."
        ) from err
    if device.type == "cpu":
        return
    if device.type == "meta":
        raise ValueError(f"{what} is 'meta', which holds no data; refusing to load.")
    try:
        is_available = getattr(torch.get_device_module(device.type), "is_available")
        available = bool(is_available())
    except Exception:  # an unknown or unregistered device type
        available = False
    if not available:
        raise ValueError(
            f"{what} ({_shown(value)}) is not available on this machine{hint}."
        )
    if device.index is not None:
        device_count = getattr(
            torch.get_device_module(device.type), "device_count", None
        )
        count = device_count() if device_count is not None else None
        if count is not None and device.index >= count:
            raise ValueError(
                f"{what} ({_shown(value)}) names device index {device.index}, but "
                f"this machine has {count} {device.type} device(s){hint}."
            )


def _int_label(label: Any) -> int | None:
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

    Each tensor holds rows of examples and has at least one dimension; every label's
    examples have the same shape. Anything else is refused with a bounded ValueError
    (see ``_int_label``).
    """
    error = ValueError(
        "Model file's support is malformed (expected a dictionary of distinct "
        "integer labels to tensors of rows that share one example shape)."
    )
    if not isinstance(support, dict):
        raise error
    converted: OrderedDict[int, torch.Tensor] = OrderedDict()
    example_shape: torch.Size | None = None
    for label, rows in support.items():
        key = _int_label(label)
        if key is None or key in converted or not isinstance(rows, torch.Tensor):
            raise error
        if rows.dim() == 0 or (
            example_shape is not None and rows.shape[1:] != example_shape
        ):
            raise error
        example_shape = rows.shape[1:]
        converted[key] = rows
    return converted


def _finite_float(value: Any) -> float | None:
    """
    ``value`` as a finite ``float`` if it is exactly an ``int`` or ``float``.

    Returns None for anything else, including an ``int`` too large for a
    ``float`` and an infinite or NaN ``float``. A ``bool`` is refused.
    """
    if type(value) not in (int, float):
        return None
    try:
        converted = float(value)
    except OverflowError:
        return None
    return converted if math.isfinite(converted) else None


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
    bandwidth = _finite_float(bw_factor)
    if bandwidth is None or bandwidth <= 0:
        raise ValueError(
            "Model file's outlier_kde is malformed ('bw_factor' must be a positive "
            f"finite number; got {_shown(bw_factor)})."
        )
    try:
        return gaussian_kde(dataset.detach().cpu().numpy(), bw_method=bandwidth)
    except Exception as err:  # e.g. too few points for a covariance
        raise ValueError(
            "Model file's outlier_kde could not be rebuilt "
            f"({_truncate(_describe_error(err), 200)})."
        ) from err


def _outlier_kde_from_file(value: Any) -> OrderedDict[int, gaussian_kde]:
    """
    A checkpoint's ``outlier_kde`` as ``{label: gaussian_kde}`` with distinct labels.

    Each KDE is stored as ``{"dataset": tensor, "bw_factor": number}`` (see
    ``_kde_from_state``).
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
            raise ValueError(
                "Model file's outlier_kde holds a pickled gaussian_kde object, as "
                f"files from earlier EQUINE releases do; rewrite the file with "
                f"`{_UPGRADE_HINT}`."
            )
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

    The settings are passed as ``cls(embedding, **settings)``.
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
        try:
            checked["init_temperature"] = float(temperature)
        except OverflowError as err:
            raise ValueError(
                "Model file's settings['init_temperature'] is an int too large for "
                "a float."
            ) from err
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


def _submodule_layout(module: torch.nn.Module) -> list[tuple[str, str, str, str]]:
    """
    Name, class (module and qualified name) and hyper-parameters of every
    submodule of ``module``.

    A class is identified by its module and qualified name. A class created anew for
    each instance (as torch's parametrizations do) or defined again by re-running its
    definition matches. Hyper-parameters come from ``extra_repr`` for torch's own layers
    only. For a user-defined module only its class is compared (its children are listed
    separately).
    """
    return [
        (
            name,
            type(sub).__module__,
            type(sub).__qualname__,
            sub.extra_repr() if type(sub).__module__.startswith("torch.") else "",
        )
        for name, sub in module.named_modules()
    ]


def _submodule_difference(
    rebuilt: torch.nn.Module, live: torch.nn.Module
) -> str | None:
    """Where ``live``'s submodules differ from those its recipe rebuilds, if anywhere."""

    def describe(
        entry: tuple[str, str, str, str] | None,
        other: tuple[str, str, str, str] | None,
    ) -> str:
        if entry is None:
            return "nothing"
        _, module, qualname, hyper = entry
        shown = qualname.rpartition(".")[2]
        # Two classes with the same name are told apart by where they are defined.
        if other is not None and other[2].rpartition(".")[2] == shown:
            if other[1] != module:
                shown = f"{module}.{qualname}"
            elif other[2] != qualname:
                shown = qualname
        return _truncate(f"{shown}({hyper})", 100)

    expected, actual = _submodule_layout(rebuilt), _submodule_layout(live)
    for want, have in itertools.zip_longest(expected, actual):
        if want != have:
            name = (have or want)[0]
            return (
                f"its submodules differ from those its recipe builds (at "
                f"{_truncate(repr(name), 80)}: {describe(have, want)} in the model, "
                f"{describe(want, have)} from the recipe); every architectural choice "
                "must be a constructor argument, not set after construction"
            )
    return None


def _recipe_problem(
    recipe: Any,
    state_dict: Mapping[str, Any],
    live: torch.nn.Module | None = None,
) -> str | None:
    """
    Why ``recipe`` cannot be trusted to rebuild a module holding ``state_dict``.

    Returns None when it can. The recipe is built on the meta device (no memory) and
    checked before anything is allocated: the constructor must leave every tensor on the
    meta device, ``state_dict()`` (including ``get_extra_state``) must run there,
    and the stored weights must match it in names, shapes and dtype kinds
    with enough storage behind them (see ``_state_dict_mismatch`` and
    ``_dtype_kind_mismatch``). With ``live``, the module being saved, its submodules
    must also match the rebuild. A recipe that cannot be built at all raises
    ``build_from_recipe``'s ValueError.
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
    except Exception as err:
        return (
            "its state_dict()/get_extra_state could not run on the meta device "
            f"({_truncate(_describe_error(err), 300)})"
        )
    if live is not None:
        difference = _submodule_difference(skeleton, live)
        if difference is not None:
            return difference
    mismatch = _state_dict_mismatch(expected, state_dict) or _dtype_kind_mismatch(
        expected, state_dict
    )
    if mismatch is not None:
        return f"the recipe does not match the weights ({mismatch})"
    return None


def _require_rebuildable(
    embedding: torch.nn.Module, recipe: dict[str, Any], state_dict: Mapping[str, Any]
) -> None:
    """
    Refuse to save an embedding that the loader could not rebuild from its recipe.

    Runs the loader's ``_recipe_problem`` check, and also compares the rebuild's
    submodules with the live module. Extra state (``get_extra_state``) holding anything
    but plain values and tensors (not quantized, without Python attributes) is refused
    too.
    """
    extra_state = {k: v for k, v in state_dict.items() if _is_extra_state_key(k)}
    try:
        _check_plain(extra_state, allow_tensors=True, what="Extra state")
    except TypeError as err:
        raise ValueError(
            f"Cannot save the embedding model ({type(embedding).__name__}): {err}"
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


def _embedding_checkpoint(embedding: torch.nn.Module) -> dict[str, Any]:
    """
    The checkpoint entries describing an embedding model: its recipe and its
    ``state_dict``, after checking that the recipe rebuilds the same module
    (see ``_require_rebuildable``).
    """
    recipe = embedding_recipe(embedding)
    if recipe is None:
        raise ValueError(
            f"Cannot save: the embedding model ({type(embedding).__name__}) is not a "
            "registered architecture, so it has no recipe to store. Register it with "
            "@equine.embedding_architecture('yourproject.name') and construct it "
            "with plain-valued arguments."
        )
    state_dict = embedding.state_dict()
    # The check below runs the constructor, which could mutate the recipe.
    stored_recipe = copy.deepcopy(recipe)
    _require_rebuildable(embedding, recipe, state_dict)
    return {"embedding_recipe": stored_recipe, "embedding_state_dict": state_dict}


def _load_embedding_weights(
    module: torch.nn.Module, state_dict: Mapping[str, Any]
) -> None:
    """``load_state_dict`` with any failure reported as a bounded ValueError."""
    try:
        module.load_state_dict(state_dict)
    except Exception as err:
        raise ValueError(
            "Could not load the embedding weights stored in the file: "
            f"{_truncate(_message_or_type(err))}"
        ) from err


def _holds_torchscript(checkpoint: Mapping[str, Any]) -> bool:
    """Whether a checkpoint stores its embedding model as TorchScript."""
    return (
        "embed_jit_save" in checkpoint or checkpoint.get("contains_executable") is True
    )


def _refuse_torchscript(checkpoint: Mapping[str, Any], path: str) -> None:
    """Raise if the checkpoint read from ``path`` stores its embedding as TorchScript."""
    if _holds_torchscript(checkpoint):
        raise ValueError(_TORCHSCRIPT_FILE_ERROR.format(path=path))


def _rebuild_embedding(
    model_save: dict[str, Any], embedding_model: torch.nn.Module | None
) -> torch.nn.Module:
    """
    Reconstitute an embedding model from a checkpoint: a caller-supplied
    module (the file's stored dtypes and weights loaded into it), else the
    file's recipe rebuilt through the registry. A file that stores its
    embedding as TorchScript is refused first, whatever else it holds.
    """
    if _holds_torchscript(model_save):
        raise ValueError(_TORCHSCRIPT_CHECKPOINT_ERROR)
    state_dict = model_save.get("embedding_state_dict")
    if state_dict is not None:
        _require_weights_dict(
            state_dict, "embedding_state_dict", allow_extra_state=True
        )

    if embedding_model is not None:
        if state_dict is None:
            raise ValueError(
                "Model file contains no embedding weights to load into the "
                "supplied embedding_model (no embedding_state_dict)."
            )
        owned = embedding_model.state_dict(keep_vars=True)
        mismatch = _state_dict_mismatch(owned, state_dict) or _dtype_kind_mismatch(
            owned, state_dict
        )
        if mismatch is not None:
            raise ValueError(
                f"The supplied embedding_model ({type(embedding_model).__name__}) does "
                f"not match the weights stored in the file ({mismatch}); pass a module "
                "of the architecture the file was saved with."
            )
        # Converted on the CPU; the model's constructor then moves the module to the
        # model's device.
        embedding_model.to("cpu")
        _adopt_stored_dtypes(embedding_model, state_dict, "embedding_state_dict")
        _load_embedding_weights(embedding_model, state_dict)
        return embedding_model

    if "embedding_recipe" in model_save:
        recipe = model_save["embedding_recipe"]
        # Checked on the meta device against the file's weights first.
        problem = _recipe_problem(recipe, state_dict or {})
        if problem is not None:
            raise ValueError(
                f"Refusing to build the embedding recipe {recipe.get('builder')!r} "
                f"stored in the file: {problem}."
            )
        embedding = build_from_recipe(recipe)
        _adopt_stored_dtypes(embedding, state_dict or {}, "embedding_state_dict")
        _load_embedding_weights(embedding, state_dict or {})
        return embedding

    raise ValueError("Model file contains no embedding model (no recipe).")


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
        Integer argument class labels (ground truth), moved to ``y_hat``'s device.

    Returns
    -------
    float
        Brier score.
    """
    _, num_classes = y_hat.size()
    one_hot_y_test = torch.nn.functional.one_hot(
        y_test.to(y_hat.device).long(), num_classes=num_classes
    )
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
        Integer argument class labels (ground truth), moved to ``y_hat``'s device.

    Returns
    -------
    float
        Brier skill score.
    """
    _, num_classes = y_hat.size()
    random_guess = (1.0 / num_classes) * torch.ones(y_hat.size(), device=y_hat.device)
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
        Class label indices (ground truth), moved to ``y_hat``'s device.

    Returns
    -------
    float
        Expected calibration error.
    """
    _, num_classes = y_hat.size()
    metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=25, norm="l1")
    ece = metric.to(y_hat.device)(y_hat, y_test.to(y_hat.device)).item()
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

    Raises
    ------
    ValueError
        If a label in ``selected_labels`` has no examples in ``train_y``.
    """
    shuffled_idxs_by_class = OrderedDict()
    for label in selected_labels:
        label_idxs = torch.argwhere(train_y == label).reshape(-1)
        if label_idxs.shape[0] == 0:
            raise ValueError(f"Class {label} has no examples.")
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
    shuffled_indexes: dict[Any, torch.Tensor] | None = None,
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
    shuffled_indexes: dict[Any, torch.Tensor] | None, optional
        Simply use the precomputed indexes if they are available

    Returns
    -------
    OrderedDict[int, torch.Tensor]
        Ordered dictionary of class labels with corresponding support examples.

    Raises
    ------
    ValueError
        If ``shuffled_indexes`` is None and a selected class has fewer than
        ``support_size`` examples, or ``shuffled_indexes`` for a label selects
        no example or an example of another label.
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

        if torch.unique(train_y[shuffled_idxs[label]]).tolist() != [label]:
            raise ValueError(
                f"The shuffled indexes for label {label} must select at least one "
                "example, and only examples of that label."
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

    Raises
    ------
    ValueError
        If ``way`` exceeds the number of labels, or a selected class has no
        more than ``support_size`` examples.
    """
    labels, counts = torch.unique(train_y, return_counts=True)
    if way > len(labels):
        raise ValueError(
            f"The way (#classes in each episode), {way}, must be <= number of labels, {len(labels)}"
        )

    selected_labels = sorted(
        labels[torch.randperm(labels.shape[0])][:way].tolist()
    )  # need to be in same order every time

    for label, count in list(zip(labels.tolist(), counts.tolist())):
        if (label in selected_labels) and (count <= support_size):
            raise ValueError(
                f"Class {_label_text(label)} has {count} examples; an episode needs "
                f"more than support_size ({support_size})."
            )
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
        True class labels, moved to the device of ``eq_preds.classes``, where
        the metrics are computed.

    Returns
    -------
    dict[str, Any]
        Dictionary of model metrics. ``"confusionMatrix"`` has one row per true
        class and one column per predicted class.
    """
    device = eq_preds.classes.device
    true_y = true_y.to(device)
    pred_y = torch.argmax(eq_preds.classes, dim=1)
    accuracy = MulticlassAccuracy(num_classes=eq_preds.classes.shape[1]).to(device)
    f1_score = MulticlassF1Score(
        num_classes=eq_preds.classes.shape[1], average="micro"
    ).to(device)
    confusion_matrix = MulticlassConfusionMatrix(
        num_classes=eq_preds.classes.shape[1]
    ).to(device)
    metrics = {
        "accuracy": accuracy(pred_y, true_y),
        "microF1Score": f1_score(pred_y, true_y),
        "confusionMatrix": confusion_matrix(pred_y, true_y).tolist(),
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


def _mahalanobis_from_factor(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """``x^T C^{-1} x`` per column of ``x`` (classes, dim, examples), from ``C = L L^T``."""
    return torch.sum(torch.linalg.solve_triangular(factor, x, upper=False) ** 2, dim=1)


def _cholesky_with_jitter(cov: torch.Tensor) -> torch.Tensor:
    """
    Cholesky factors of a batch of covariance matrices, shape (classes, dim, dim).

    A matrix that factors as it is gets no jitter. A matrix whose factorization
    fails gets a diagonal jitter of machine epsilon times its largest variance,
    multiplied by 10 until the factorization succeeds, as long as the jitter
    stays at or below 1e-3 of that variance.

    Raises
    ------
    torch.linalg.LinAlgError
        If a matrix does not factor with the largest jitter (for example, one
        that holds NaN).
    """
    factor, info = torch.linalg.cholesky_ex(cov)
    failed = (info != 0) | ~torch.isfinite(factor).flatten(-2).all(-1)
    if not bool(failed.any()):
        return factor
    eps = torch.finfo(cov.dtype).eps
    largest = cov.detach().diagonal(dim1=-2, dim2=-1).amax(dim=-1)
    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    steps = int(math.log10(1e-3 / eps)) + 1
    jitter = torch.zeros_like(largest)
    for power in range(steps):
        jitter = torch.where(failed, eps * 10.0**power * largest, jitter)
        factor, info = torch.linalg.cholesky_ex(cov + jitter[:, None, None] * eye)
        failed = (info != 0) | ~torch.isfinite(factor).flatten(-2).all(-1)
        if not bool(failed.any()):
            return factor
    raise torch.linalg.LinAlgError(
        "A covariance matrix is not positive definite even with a diagonal "
        f"jitter of {eps * 10.0 ** (steps - 1):.1e} times its largest variance."
    )


@icontract.require(
    lambda cov: cov.dim() == 3, "cov must have shape (classes, dim, dim)"
)
@icontract.require(lambda cov: cov.shape[-2] == cov.shape[-1])
def mahalanobis_distance_nosq(x: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared Mahalanobis distance $x^T C^{-1} x$ (no square root).

    Computed as the squared norm of $L^{-1} x$, where $C = L L^T$ is the Cholesky
    factorization. The result is never negative, and the gradient is finite whenever $C$
    is positive definite, including when it has repeated eigenvalues.

    Parameters
    ----------
    x : torch.Tensor
        Vectors, shape ``(classes, dim, examples)``. A ``(dim, examples)`` or
        ``(dim,)`` tensor is used for every class.
    cov : torch.Tensor
        Symmetric positive-definite covariance matrices, shape
        ``(classes, dim, dim)``.

    Returns
    -------
    torch.Tensor
        Distances, shape ``(classes, examples)``, or ``(classes,)`` for a
        ``(dim,)`` input.

    Raises
    ------
    torch.linalg.LinAlgError
        If the Cholesky factorization of a covariance matrix fails. A singular
        matrix that rounding leaves positive definite gives very large
        distances instead, and a matrix that holds NaN gives NaN distances
        where the LAPACK build reports success for it (OpenBLAS does).
    """
    if x.dim() == 1:
        return mahalanobis_distance_nosq(x.unsqueeze(-1), cov).squeeze(-1)
    if x.dim() == 2:
        x = x.expand(cov.shape[0], -1, -1)
    return _mahalanobis_from_factor(x, torch.linalg.cholesky(cov))


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
