# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import torch

from .equine_gp import EquineGP
from .equine_protonet import EquineProtonet


def load_equine_model(model_path: str):
    model_type = torch.load(model_path)["train_summary"]["modelType"]

    if model_type == "EquineProtonet":
        model = EquineProtonet.load(model_path)
    elif model_type == "EquineGP":
        model = EquineGP.load(model_path)
    else:
        raise ValueError(f"Unknown model type '{model_type}'")
    return model
