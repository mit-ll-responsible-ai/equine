# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Optional

import torch

from .equine import Equine
from .equine_gp import EquineGP
from .equine_protonet import EquineProtonet
from .registry import _truncate
from .utils import _load_checkpoint_for_device, _refuse_torchscript


def load_equine_model(
    model_path: str,
    device: Optional[str] = None,
    *,
    embedding_model: Optional[torch.nn.Module] = None,
) -> Equine:
    """
    Attempt to load an EQUINE model from a file

    Parameters
    ----------
    model_path : str
        The path to the model file
    device : Optional[str]
        The device to load the model onto. Overrides the device recorded in
        the file; by default the model lands on the saved device.
    embedding_model : Optional[torch.nn.Module]
        Keyword-only. Use this module as the embedding architecture instead of
        rebuilding it from the file's recipe; the file's weights are loaded
        into it, and its floating-point tensors take their stored dtypes.

    Returns
    -------
    Equine
        The loaded EQUINE model

    Raises
    ------
    ValueError
        If the model type is missing or unknown, the file is not in the
        current EQUINE format (files from earlier releases must first be
        rewritten with ``python -m equine.upgrade``), it stores its embedding
        model as TorchScript, or its recipe names an unregistered architecture.
        If ``device`` is unavailable here, names an
        index at or above its device count, or lacks a dtype the file holds
        (float64 on MPS: load such a file with ``device="cpu"``).

    Notes
    -----
    Loading runs no code from the file: it is read with restricted unpickling (see
    ``utils.load_checkpoint``) and its embedding model is rebuilt through the
    architecture registry.
    """
    model_save = _load_checkpoint_for_device(model_path, device)
    _refuse_torchscript(model_save, model_path)
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
        model_save, device, embedding_model=embedding_model
    )
