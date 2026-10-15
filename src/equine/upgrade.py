# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Rewrite model files whose embedding model is stored as TorchScript.

    python -m equine.upgrade MODEL.eq [-o OUT.eq] --recipe JSON [--import MODULE]

Files saved by earlier EQUINE releases hold pickled Python objects, which
``torch.load(weights_only=True)`` refuses, and store the embedding model as
TorchScript, which EQUINE does not load. The upgrade reads such a file with
unrestricted unpickling and loads its TorchScript archive; both run code
embedded in the file, so only upgrade files you trust. A current-format file
that holds a TorchScript archive is upgraded the same way.

The archive is the file's only description of the embedding architecture, so
the caller supplies it: an instance of a registered architecture
(``@equine.embedding_architecture``), or on the command line its recipe. The
archive's weights are copied into it and the model is written as a data-only
file. Before anything replaces the original, the written file is loaded back
and its embedding must reproduce the archive's outputs on the file's support
rows and on ``check_inputs``; with neither, the comparison is skipped with a
warning.

Reading these files is deprecated: this module is removed in a later release,
and files not upgraded by then cannot be opened.
"""

import argparse
import importlib
import io
import itertools
import json
import os
import shutil
import sys
import warnings
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
from scipy.stats import gaussian_kde

from .equine_gp import EquineGP
from .equine_protonet import EquineProtonet, _kde_to_state
from .registry import build_from_recipe, embedding_recipe, registered_architectures
from .utils import (
    EQUINE_FORMAT_VERSION,
    LegacyFormatError,
    _embedding_checkpoint,
    _is_extra_state_key,
    _unalias_support,
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

_NOT_COMPARED_NOTE = (
    "The file stores no support rows, so the new embedding was not compared with "
    "the file's TorchScript one. Compare the two models' predictions, or pass "
    "check_inputs to upgrade_model_file."
)

_RTOL = 1e-4
_ATOL = 1e-5


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

    ``.numpy()`` works on the tensor's own data (shape, stride and storage
    offset), so a partial or offset view yields that view's bytes only.
    """
    return archive.detach().cpu().contiguous().numpy().tobytes()


def _load_archive(archive: Any) -> torch.nn.Module:
    """
    The TorchScript module stored in a checkpoint, loaded on the CPU.

    Accepts the ``uint8`` tensor of a current-format file and the ``bytes`` and
    ``io.BytesIO`` forms of files from earlier releases.
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
            f"({type(err).__name__}: {err})."
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
    _unalias_support(out)  # legacy support rows may share one storage
    return out, notes


def _attribute_tensor(module: Any, key: str) -> Optional[torch.Tensor]:
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

    Each entry comes from the archive's ``state_dict`` or, failing that, from
    the tensor attribute of the same dotted name (TorchScript keeps plain
    tensor attributes outside its ``state_dict``). Extra-state entries keep
    the target's values. Archive entries the target does not store are
    allowed only for the target's non-persistent buffers.
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


def _first_float_dtype(module: torch.nn.Module) -> Optional[torch.dtype]:
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
            f"with the file's ({type(err).__name__}: {err})."
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


def _comparison_blocks(
    checkpoint: dict[str, Any], check_inputs: Optional[torch.Tensor]
) -> list[torch.Tensor]:
    """The inputs to compare the embeddings on: each label's support rows and
    ``check_inputs``, each kept as its own block (dtypes and shapes may differ)."""
    support = checkpoint.get("support")
    blocks = list(support.values()) if isinstance(support, dict) else []
    if check_inputs is not None:
        if not isinstance(check_inputs, torch.Tensor) or check_inputs.dim() == 0:
            raise ValueError("check_inputs must be a tensor with one row per input.")
        blocks.append(check_inputs)
    return [
        b for b in blocks if isinstance(b, torch.Tensor) and b.dim() > 0 and len(b) > 0
    ]


def _read(path: str) -> Optional[dict[str, Any]]:
    """The checkpoint to upgrade, or None if ``path`` is already data-only."""
    try:
        checkpoint = load_checkpoint(path)
    except LegacyFormatError:
        pass
    else:
        if (
            "embed_jit_save" in checkpoint
            or checkpoint.get("contains_executable") is True
        ):
            return checkpoint
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as err:
        raise ValueError(
            f"'{path}' could not be read as a PyTorch file "
            f"({type(err).__name__}: {err})."
        ) from err
    if not isinstance(checkpoint, dict) or "train_summary" not in checkpoint:
        raise ValueError(f"'{path}' is not an EQUINE model file.")
    return checkpoint


def _reason(err: Exception, partial: str, path: str) -> str:
    """
    ``err`` as one sentence for an upgrade error: the temp file named as
    ``path``, the loader's upgrade hint cut off, the type named unless it is
    a ValueError (or the message is empty), and a final period.
    """
    message = str(err).replace(partial, path).split(". Rewrite it once with")[0]
    message = message.strip()
    if not message:
        message = type(err).__name__
    elif not isinstance(err, ValueError):
        message = f"{type(err).__name__}: {message}"
    return message if message.endswith(".") else f"{message}."


def _upgrade(
    path: str,
    checkpoint: dict[str, Any],
    embedding_model: torch.nn.Module,
    out_path: Optional[str],
    check_inputs: Optional[torch.Tensor],
) -> tuple[str, list[str]]:
    """Write the upgraded file and return where, plus the notes to report."""
    if embedding_recipe(embedding_model) is None:
        raise ValueError(
            f"The supplied embedding_model ({type(embedding_model).__name__}) is "
            "not a registered architecture; register its class with "
            "@equine.embedding_architecture and pass an instance."
        )
    if "embed_jit_save" not in checkpoint:
        raise ValueError(f"'{path}' holds no TorchScript archive to upgrade from.")
    scripted = _load_archive(checkpoint["embed_jit_save"])
    current, notes = _current_layout(checkpoint)
    embedding_model.load_state_dict(_weights_from_archive(scripted, embedding_model))
    current.update(_embedding_checkpoint(embedding_model))
    blocks = _comparison_blocks(current, check_inputs)

    out_path = out_path or path
    if os.path.islink(out_path):
        out_path = os.path.realpath(out_path)  # replace the link's target, not the link
    directory = os.path.dirname(os.path.abspath(out_path))
    partial = os.path.join(directory, f".{os.path.basename(out_path)}.upgrading")
    try:
        try:
            torch.save(current, partial)
        except Exception as err:
            raise OSError(
                f"Could not write '{partial}' ({type(err).__name__}: {err})."
            ) from err
        try:
            load_checkpoint(partial)
        except ValueError as err:
            raise ValueError(
                "The safe loader refuses the rewritten file: "
                f"{_reason(err, partial, path)} '{path}' holds a value the upgrade "
                "does not convert; it was left unchanged."
            ) from err
        # The class follows the layout, so files saved from a subclass load too.
        model_class = EquineGP if "laplace_model_save" in current else EquineProtonet
        try:
            rebuilt = model_class.load(partial, device="cpu")
        except Exception as err:  # the supplied architecture is arbitrary code
            raise ValueError(
                f"The rewritten file does not load: {_reason(err, partial, path)} "
                f"'{path}' was left unchanged."
            ) from err
        if not blocks:
            notes.append(_NOT_COMPARED_NOTE)
        for rows in blocks:
            _require_same_embedding(scripted, rebuilt.embedding_model, rows)
        if os.path.exists(out_path):
            try:
                shutil.copymode(out_path, partial)
            except OSError as err:  # a filesystem that rejects mode changes
                notes.append(
                    f"Could not copy the permission bits of '{out_path}' onto the "
                    f"upgraded file ({err})."
                )
        os.replace(partial, out_path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return out_path, notes


def upgrade_model_file(
    path: str,
    embedding_model: torch.nn.Module,
    out_path: Optional[str] = None,
    *,
    check_inputs: Optional[torch.Tensor] = None,
) -> Optional[str]:
    """
    Rewrite a trusted model file whose embedding model is TorchScript as a
    data-only file.

    Parameters
    ----------
    path : str
        The file to upgrade: a file saved by an earlier EQUINE release, or a
        current-format file holding a TorchScript archive. It is read with
        unrestricted unpickling and its archive is loaded.
    embedding_model : torch.nn.Module
        An instance of the registered architecture the file's embedding was
        built from. The archive's weights are copied into it.
    out_path : Optional[str]
        Where to write the result; defaults to replacing ``path``. The result
        is written next to it and moved into place only after it has been
        loaded back and checked, so a failed upgrade leaves the original as it
        was. A symbolic link is followed, so its target is replaced, and the
        replaced file keeps its permission bits where the filesystem allows
        it.
    check_inputs : Optional[torch.Tensor]
        Keyword-only. Extra inputs on which the new embedding must reproduce
        the archive's outputs, in addition to the file's support rows.

    Returns
    -------
    Optional[str]
        The path of the written file, or None if ``path`` is already a
        data-only current-format file and nothing was written.

    Raises
    ------
    ValueError
        If the file is not an EQUINE model file, if the safe loader refuses it
        for a reason other than its age, if ``embedding_model`` is not a
        registered architecture or does not match the archive's weights, if
        the rewritten file does not load, or if its embedding does not
        reproduce the archive's outputs.
    OSError
        If the result cannot be written next to ``out_path``.

    Warns
    -----
    FutureWarning
        Always: reading these files is deprecated.
    UserWarning
        When Linear-head weights were dropped, when no rows were available to
        compare the embeddings, or when the permission bits could not be
        copied.
    """
    warnings.warn(_DEPRECATION, FutureWarning, stacklevel=2)
    checkpoint = _read(path)
    if checkpoint is None:
        return None
    written, notes = _upgrade(path, checkpoint, embedding_model, out_path, check_inputs)
    for note in notes:
        warnings.warn(note, UserWarning, stacklevel=2)
    return written


def _module_from_recipe(recipe_json: str, imports: list[str]) -> torch.nn.Module:
    """The architecture named by a JSON recipe, after importing ``imports``."""
    for name in imports:
        try:
            importlib.import_module(name)
        except Exception as err:
            raise ValueError(
                f"Could not import {name!r} ({type(err).__name__}: {err})."
            ) from err
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m equine.upgrade",
        description="Rewrite a model file whose embedding model is TorchScript "
        "(every file saved by an earlier EQUINE release) as a data-only file. "
        "Only upgrade files you trust: the file is unpickled without "
        "restrictions and its TorchScript archive is loaded, either of which can "
        "run code embedded in it.",
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
        "(repeatable)",
    )
    args = parser.parse_args(argv)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = _read(args.path)
            if checkpoint is None:
                print(f"{args.path} is already in the current format; nothing written.")
                return 0
            if args.recipe is None:
                raise ValueError(
                    "--recipe is required: the file's embedding is TorchScript, so "
                    "the upgrade needs the registered architecture to rebuild it."
                )
            module = _module_from_recipe(args.recipe, args.imports)
            written, notes = _upgrade(args.path, checkpoint, module, args.output, None)
    except (ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    if written == args.path:
        print(f"Upgraded {args.path} in place.")
    else:
        print(f"Upgraded {args.path} -> {written}")
    print(_DEPRECATION, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
