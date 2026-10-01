# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import sys
from typing import Optional

import torch

from .equine import Equine
from .equine_gp import EquineGP
from .equine_protonet import EquineProtonet
from .registry import _truncate
from .utils import load_checkpoint


def load_equine_model(
    model_path: str,
    *,
    allow_unsafe_legacy_format: bool = False,
    trust_executable: bool = False,
    embedding_model: Optional[torch.nn.Module] = None,
) -> Equine:
    """
    Attempt to load an EQUINE model from a file

    Parameters
    ----------
    model_path : str
        The path to the model file
    allow_unsafe_legacy_format : bool, optional
        Keyword-only. Permit loading a file written in the legacy pickle
        format. This uses unrestricted unpickling and can execute code embedded
        in the file, so only enable it for files you trust. Implies
        ``trust_executable``. Defaults to False.
    trust_executable : bool, optional
        Keyword-only. Permit running the TorchScript module embedded in a file
        saved with ``allow_executable=True``; keeping that module as the
        embedding emits a ``FutureWarning``. Defaults to False.
    embedding_model : Optional[torch.nn.Module]
        Keyword-only. Use this module as the embedding architecture instead of
        rebuilding it from the file's recipe; the file's weights are loaded
        into it.

    Returns
    -------
    Equine
        The loaded EQUINE model

    Raises
    ------
    ValueError
        If the model type is missing or unknown, the file cannot be loaded
        safely, its recipe names an unregistered architecture, or it contains
        executable content without ``trust_executable``.

    Notes
    -----
    The file is read with restricted unpickling (see ``utils.load_checkpoint``).
    A recipe file (the default ``save()`` output) contains no executable
    content: its embedding model is rebuilt through the architecture registry.
    Files saved with ``allow_executable=True`` embed a TorchScript module, which
    is executable code, and need ``trust_executable=True``.
    """
    model_save = load_checkpoint(
        model_path, allow_unsafe_legacy_format=allow_unsafe_legacy_format
    )
    summary = model_save.get("train_summary")
    model_type = summary.get("modelType") if isinstance(summary, dict) else None
    if not isinstance(model_type, str):
        raise ValueError(
            "Model file has a malformed train_summary (expected a dictionary with a "
            "string 'modelType')."
        )
    classes: dict[str, type[EquineProtonet] | type[EquineGP]] = {
        "EquineProtonet": EquineProtonet,
        "EquineGP": EquineGP,
    }
    if model_type not in classes:
        raise ValueError(f"Unknown model type {_truncate(repr(model_type), 80)}")
    return classes[model_type]._from_checkpoint(
        model_save,
        trust_executable=trust_executable,
        embedding_model=embedding_model,
        allow_unsafe_legacy_format=allow_unsafe_legacy_format,
        # _from_checkpoint behind its beartype wrapper (absent under
        # `python -O`), then this function, then the caller.
        _stacklevel=3 + (0 if sys.flags.optimize else 1),
    )
