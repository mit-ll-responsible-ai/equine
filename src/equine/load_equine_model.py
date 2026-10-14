# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from .equine import Equine
from .equine_gp import EquineGP
from .equine_protonet import EquineProtonet
from .utils import load_checkpoint


def load_equine_model(model_path: str) -> Equine:
    """
    Attempt to load an EQUINE model from a file

    Parameters
    ----------
    model_path : str
        The path to the model file

    Returns
    -------
    Equine
        The loaded EQUINE model

    Raises
    ------
    ValueError
        If the model type is unknown, or if the file is not in the current
        EQUINE format (files from earlier releases must first be rewritten with
        ``python -m equine.upgrade``).
    """
    model_save = load_checkpoint(model_path)
    model_type = model_save["train_summary"]["modelType"]

    if model_type == "EquineProtonet":
        model = EquineProtonet._from_checkpoint(model_save)
    elif model_type == "EquineGP":
        model = EquineGP._from_checkpoint(model_save)
    else:
        raise ValueError(f"Unknown model type '{model_type}'")
    return model
