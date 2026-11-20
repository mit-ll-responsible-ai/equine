# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Embedding architectures that ship with EQUINE.

They are registered on import. A model file that names one loads anywhere EQUINE is
installed, without the user defining a class. Their constructor signatures are part of
the model-file format: change them only additively, with defaults.

``MLP`` enforces ``MAX_HIDDEN_LAYERS`` and ``MAX_PARAMETERS`` caps on its constructor
arguments. Raising either cap later is a compatible change; lowering one is not.

``MLP``'s ``state_dict()`` layout is also part of the model-file format: the
parameter names and their order (``net.0.weight``, ``net.0.bias``, ...) must
not change, and any new constructor option must not shift the indices of the
existing ``torch.nn.Sequential`` layers.

``SpectralResidualMLP`` follows the embedding SNGP pairs with a Gaussian-process
head: residual blocks under spectral normalization, each of which, once torch's
power-iteration estimate has converged, keeps the distance between two hidden states
within fixed bounds. Its caps and its ``state_dict()``
layout (``inp.*``, ``blocks.<i>.*``, ``out.*``) are part of the format in the same
way.
"""

from __future__ import annotations

import reprlib
from collections.abc import Sequence

import torch

from .registry import _positive_int, embedding_architecture

__all__ = ["MLP", "SpectralResidualMLP"]

_ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}

#: Most hidden layers a recipe may request.
MAX_HIDDEN_LAYERS = 64

#: Most parameters a recipe may request.
MAX_PARAMETERS = 10**9


def _mlp_parameter_count(
    in_features: int, hidden_sizes: Sequence[int], out_features: int
) -> int:
    """Count ``MLP``'s parameters (weights + biases) from plain sizes, no allocation."""
    count = 0
    width = in_features
    for hidden in hidden_sizes:
        count += width * hidden + hidden
        width = hidden
    count += width * out_features + out_features
    return count


@embedding_architecture("equine.mlp")
class MLP(torch.nn.Module):
    """
    A plain multilayer perceptron for tabular features.

    Parameters
    ----------
    in_features : int
        Number of input features. Must be >= 1.
    hidden_sizes : list[int] or tuple[int, ...]
        Width of each hidden layer, in order. An empty list gives a single
        linear map. Must have at most ``MAX_HIDDEN_LAYERS`` entries, each
        >= 1.
    out_features : int
        Embedding dimension (``emb_out_dim`` for the EQUINE model). Must be
        >= 1.
    activation : str, optional
        One of ``"relu"``, ``"gelu"``, ``"tanh"``, ``"sigmoid"``, by default
        ``"relu"``.

    Raises
    ------
    TypeError
        If an argument is not a plain value (for example a numpy integer or
        a set), which a recipe could not store; raised when the instance
        records its recipe, before the checks below.
    ValueError
        If ``in_features``, ``out_features``, or any entry of
        ``hidden_sizes`` is not exactly an ``int`` (``bool`` is rejected) or
        is < 1; if ``hidden_sizes`` is not exactly a ``list`` or ``tuple``;
        if ``hidden_sizes`` has more than ``MAX_HIDDEN_LAYERS`` entries; if
        the resulting parameter count exceeds ``MAX_PARAMETERS``; or if
        ``activation`` is not exactly a ``str`` naming a known activation.
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: list[int] | tuple[int, ...],
        out_features: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        _positive_int(in_features, "in_features")
        _positive_int(out_features, "out_features")

        if type(hidden_sizes) not in (list, tuple):
            raise ValueError(
                "hidden_sizes must be a list or tuple (not "
                f"{type(hidden_sizes).__name__}); got {reprlib.repr(hidden_sizes)}."
            )
        for index, hidden in enumerate(hidden_sizes):
            _positive_int(hidden, f"hidden_sizes[{index}]")
        if len(hidden_sizes) > MAX_HIDDEN_LAYERS:
            raise ValueError(
                f"hidden_sizes has {len(hidden_sizes)} entries, exceeding the "
                f"MAX_HIDDEN_LAYERS cap of {MAX_HIDDEN_LAYERS}."
            )

        if type(activation) is not str:
            raise ValueError(
                f"activation must be a str (not {type(activation).__name__}); "
                f"got {reprlib.repr(activation)}."
            )
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {reprlib.repr(activation)}; choose from "
                f"{sorted(_ACTIVATIONS)}."
            )

        parameter_count = _mlp_parameter_count(in_features, hidden_sizes, out_features)
        if parameter_count > MAX_PARAMETERS:
            raise ValueError(
                f"This MLP would have {reprlib.repr(parameter_count)} parameters, "
                f"exceeding the MAX_PARAMETERS cap of {MAX_PARAMETERS}."
            )

        layers: list[torch.nn.Module] = []
        width = in_features
        for hidden in hidden_sizes:
            layers += [torch.nn.Linear(width, hidden), _ACTIVATIONS[activation]()]
            width = hidden
        layers.append(torch.nn.Linear(width, out_features))
        self.net: torch.nn.Sequential = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed ``x`` by applying the linear/activation stack in ``self.net``."""
        return self.net(x)


def _spectral_residual_mlp_parameter_count(
    in_features: int, hidden_size: int, out_features: int, num_blocks: int
) -> int:
    """Count ``SpectralResidualMLP``'s weights and biases from plain sizes."""
    return (
        in_features * hidden_size
        + hidden_size
        + num_blocks * (hidden_size * hidden_size + hidden_size)
        + hidden_size * out_features
        + out_features
    )


@embedding_architecture("equine.spectral_residual_mlp")
class SpectralResidualMLP(torch.nn.Module):
    """
    A residual multilayer perceptron under spectral normalization, the embedding
    that Spectral-normalized Neural Gaussian Processes (SNGP; Liu et al., 2020)
    pair with a Gaussian-process head.

    A linear projection to ``hidden_size`` and a ReLU are followed by
    ``num_blocks`` residual blocks, ``h + coefficient * relu(W h + b)``, and a
    linear map to ``out_features``. The blocks' ``W`` and the final map's weight
    are spectrally normalized: torch divides each by its power-iteration
    estimate of the largest singular value. The estimate is at most the true
    value and approaches it as training runs the iteration, and it is saved
    with the model. Once it has converged, each block changes the distance
    between two hidden states by a factor between ``1 - coefficient`` and
    ``1 + coefficient``. The input projection, its ReLU and the final map have
    no such lower bound: with ``out_features`` below ``in_features``, some
    distinct inputs map to nearly the same embedding. For the embedding as a
    whole, and for ``EquineGP``'s variance, the class gives no distance
    guarantee. Its spectrally normalized layers cannot be pickled: torch
    serializes parametrized modules only through ``state_dict()``, which
    ``save()`` uses; ``copy.deepcopy`` works.

    Parameters
    ----------
    in_features : int
        Number of input features. Must be >= 1.
    hidden_size : int
        Width of the projection and of every residual block. Must be >= 1.
    out_features : int
        Embedding dimension (``emb_out_dim`` for the EQUINE model). Must be
        >= 1.
    num_blocks : int, optional
        Number of residual blocks, by default 1. Must be >= 1 and at most
        ``MAX_HIDDEN_LAYERS``.
    coefficient : float, optional
        Bound on each residual branch's Lipschitz constant, by default 0.95
        (SNGP's value). Must be a float strictly between 0 and 1.

    Raises
    ------
    TypeError
        If an argument is not a plain value, which a recipe could not store;
        raised when the instance records its recipe, before the checks below.
    ValueError
        If ``in_features``, ``hidden_size``, ``out_features`` or
        ``num_blocks`` is not exactly an ``int`` (``bool`` is rejected) or is
        < 1; if ``num_blocks`` exceeds ``MAX_HIDDEN_LAYERS``; if
        ``coefficient`` is not exactly a ``float`` strictly between 0 and 1; or
        if the resulting parameter count exceeds ``MAX_PARAMETERS``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
        num_blocks: int = 1,
        coefficient: float = 0.95,
    ) -> None:
        super().__init__()

        _positive_int(in_features, "in_features")
        _positive_int(hidden_size, "hidden_size")
        _positive_int(out_features, "out_features")
        _positive_int(num_blocks, "num_blocks")
        if num_blocks > MAX_HIDDEN_LAYERS:
            raise ValueError(
                f"num_blocks is {num_blocks}, exceeding the MAX_HIDDEN_LAYERS cap "
                f"of {MAX_HIDDEN_LAYERS}."
            )
        if type(coefficient) is not float or not 0.0 < coefficient < 1.0:
            raise ValueError(
                "coefficient must be a float strictly between 0 and 1; got "
                f"{type(coefficient).__name__} {reprlib.repr(coefficient)}."
            )

        parameter_count = _spectral_residual_mlp_parameter_count(
            in_features, hidden_size, out_features, num_blocks
        )
        if parameter_count > MAX_PARAMETERS:
            raise ValueError(
                "This SpectralResidualMLP would have "
                f"{reprlib.repr(parameter_count)} parameters, exceeding the "
                f"MAX_PARAMETERS cap of {MAX_PARAMETERS}."
            )

        spectral_norm = torch.nn.utils.parametrizations.spectral_norm
        self.inp: torch.nn.Linear = torch.nn.Linear(in_features, hidden_size)
        self.blocks: torch.nn.ModuleList = torch.nn.ModuleList(
            [
                spectral_norm(torch.nn.Linear(hidden_size, hidden_size))
                for _ in range(num_blocks)
            ]
        )
        self.out: torch.nn.Module = spectral_norm(
            torch.nn.Linear(hidden_size, out_features)
        )
        self.coefficient = coefficient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x``, apply the residual blocks, and project to the embedding."""
        h = torch.relu(self.inp(x))
        for block in self.blocks:
            h = h + self.coefficient * torch.relu(block(h))
        return self.out(h)
