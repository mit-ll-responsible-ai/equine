# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import torch

from .equine import Equine
from .equine_gp import EquineGP
from .equine_protonet import EquineProtonet


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
        If the model type is unknown
    """
    model_type = torch.load(model_path, weights_only=False)["train_summary"][
        "modelType"
    ]

    if model_type == "EquineProtonet":
        model = EquineProtonet.load(model_path)
    elif model_type == "EquineGP":
        model = EquineGP.load(model_path)
    else:
        raise ValueError(f"Unknown model type '{model_type}'")
    return model
