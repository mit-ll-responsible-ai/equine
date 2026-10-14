# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Rewrite model files from earlier EQUINE releases in the current format.

    python -m equine.upgrade MODEL.eq [-o OUT.eq]

Earlier releases saved pickled Python objects (an ``io.BytesIO`` holding the
TorchScript archive, ``CovType`` members, scipy KDE objects, numpy strings),
which ``torch.load(weights_only=True)`` refuses, so EQUINE no longer opens
those files. The upgrade reads the file with unrestricted unpickling, which
runs any code embedded in it, so only upgrade files you trust. It then writes
the same model in the current layout and checks that the safe loader reads it.

Reading the old layout is deprecated: this module is removed in a later
release, and files not upgraded by then cannot be opened.
"""

import argparse
import io
import os
import shutil
import sys
import warnings
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
from scipy.stats import gaussian_kde

from .equine_protonet import _kde_to_state
from .utils import (
    EQUINE_FORMAT_VERSION,
    LegacyFormatError,
    jit_archive_to_tensor,
    load_checkpoint,
)

_DEPRECATION = (
    "Reading model files from earlier EQUINE releases is deprecated and will be "
    "removed in a later release; upgrade your files now."
)


def _plain(value: Any) -> Any:
    """``value`` with numpy scalars and strings, at any depth, as Python values."""
    if isinstance(value, dict):
        return {_plain(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _current_layout(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The legacy checkpoint dictionary rewritten with plain values only."""
    out = dict(checkpoint)
    out["equine_format_version"] = EQUINE_FORMAT_VERSION
    archive = out.get("embed_jit_save")
    if isinstance(archive, (bytes, bytearray)):
        archive = io.BytesIO(archive)
    if isinstance(archive, io.BytesIO):
        out["embed_jit_save"] = jit_archive_to_tensor(archive)
    settings = dict(out.get("settings", {}))
    if isinstance(settings.get("cov_type"), Enum):
        settings["cov_type"] = settings["cov_type"].value
    out["settings"] = _plain(settings)
    if isinstance(out.get("support"), dict):
        out["support"] = {int(label): x for label, x in out["support"].items()}
    if isinstance(out.get("outlier_kde"), dict):
        out["outlier_kde"] = {
            int(label): _kde_to_state(kde) if isinstance(kde, gaussian_kde) else kde
            for label, kde in out["outlier_kde"].items()
        }
    for key in ("feature_names", "label_names"):
        names = out.get(key)
        if isinstance(names, (list, tuple)) or (
            isinstance(names, np.ndarray) and names.ndim == 1
        ):  # anything else is left for the loader to report
            out[key] = [str(name) for name in names]
    if "train_summary" in out:
        out["train_summary"] = _plain(out["train_summary"])
    return out


def upgrade_model_file(path: str, out_path: Optional[str] = None) -> Optional[str]:
    """
    Rewrite a trusted model file from an earlier EQUINE release in the current
    format.

    Parameters
    ----------
    path : str
        The file to upgrade. It is read with unrestricted unpickling.
    out_path : Optional[str]
        Where to write the result; defaults to replacing ``path``. The result
        is written next to it and moved into place only after the safe loader
        has read it, so a failed upgrade leaves the original as it was. A
        symbolic link is followed, so its target is replaced, and the
        replaced file keeps its permission bits where the filesystem allows
        it (a ``UserWarning`` is issued otherwise).

    Returns
    -------
    Optional[str]
        The path of the written file, or None if ``path`` is already in the
        current format and nothing was written.

    Raises
    ------
    ValueError
        If the file is not an EQUINE model file, if the safe loader refuses it
        for a reason other than its age (for example a format version newer
        than this EQUINE knows), or if the upgraded file is still refused.
    OSError
        If the result cannot be written next to ``out_path``.
    """
    warnings.warn(_DEPRECATION, FutureWarning, stacklevel=2)
    try:
        load_checkpoint(path)
    except LegacyFormatError:
        pass
    else:
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
    current = _current_layout(checkpoint)
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
            reason = str(err).replace(partial, path).split(". Rewrite it once with")[0]
            raise ValueError(
                f"The safe loader refuses the rewritten file: {reason}. '{path}' "
                "holds a value the upgrade does not convert; it was left unchanged."
            ) from err
        if os.path.exists(out_path):
            try:
                shutil.copymode(out_path, partial)
            except OSError as err:  # a filesystem that rejects mode changes
                warnings.warn(
                    f"Could not copy the permission bits of '{out_path}' onto the "
                    f"upgraded file ({err}).",
                    UserWarning,
                    stacklevel=2,
                )
        os.replace(partial, out_path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m equine.upgrade",
        description="Rewrite a model file from an earlier EQUINE release in the "
        "current format. Only upgrade files you trust: the file is unpickled "
        "without restrictions, which can run code embedded in it.",
    )
    parser.add_argument("path", help="the model file to upgrade")
    parser.add_argument(
        "-o", "--output", help="write the result here instead of replacing PATH"
    )
    args = parser.parse_args(argv)
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            written = upgrade_model_file(args.path, args.output)
    except (ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    for w in record:
        if issubclass(w.category, UserWarning):
            print(f"note: {w.message}", file=sys.stderr)
    if written is None:
        print(f"{args.path} is already in the current format; nothing written.")
        return 0
    if written == args.path:
        print(f"Upgraded {args.path} in place.")
    else:
        print(f"Upgraded {args.path} -> {written}")
    print(_DEPRECATION, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
