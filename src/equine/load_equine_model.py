# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from .equine import Equine
from .equine_gp import EquineGP
from .equine_protonet import EquineProtonet
from .utils import load_checkpoint


def load_equine_model(
    model_path: str, allow_unsafe_legacy_format: bool = False
) -> Equine:
    """
    Attempt to load an EQUINE model from a file

    Parameters
    ----------
    model_path : str
        The path to the model file
    allow_unsafe_legacy_format : bool, optional
        Permit loading a file written in the legacy pickle format. This uses
        unrestricted unpickling and can execute code embedded in the file, so
        only enable it for files you trust. Defaults to False.

    Returns
    -------
    Equine
        The loaded EQUINE model

    Raises
    ------
    ValueError
        If the model type is unknown, or if the file cannot be loaded safely
        and ``allow_unsafe_legacy_format`` is False.

    Notes
    -----
    The file is read with restricted unpickling (see ``utils.load_checkpoint``),
    but the embedded TorchScript module is still executable code. Only load
    files you trust.
    """
    model_save = load_checkpoint(
        model_path, allow_unsafe_legacy_format=allow_unsafe_legacy_format
    )
    model_type = model_save["train_summary"]["modelType"]

    if model_type == "EquineProtonet":
        model = EquineProtonet._from_checkpoint(model_save)
    elif model_type == "EquineGP":
        model = EquineGP._from_checkpoint(model_save)
    else:
        raise ValueError(f"Unknown model type '{model_type}'")
    return model
