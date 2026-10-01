# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Embedding architectures that ship with EQUINE.

They are registered on import, so a model file that names one loads anywhere
EQUINE is installed without the user defining a class. Their constructor
signatures are part of the model-file format: change them only additively,
with defaults.

``MLP`` enforces ``MAX_HIDDEN_LAYERS`` and ``MAX_PARAMETERS`` caps on its
constructor arguments. These are deliberate limits against oversized recipes
read from an untrusted model file (a recipe controls ``hidden_sizes``, which
would otherwise let a crafted file force an unbounded allocation). Raising
either cap later is a compatible change; lowering one is not.

``MLP``'s ``state_dict()`` layout is also part of the model-file format: the
parameter names and their order (``net.0.weight``, ``net.0.bias``, ...) must
not change, and any new constructor option must not shift the indices of the
existing ``torch.nn.Sequential`` layers.
"""

from __future__ import annotations

import reprlib
from collections.abc import Sequence

import torch

from .registry import _positive_int, embedding_architecture

__all__ = ["MLP"]

_ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}

#: Deliberate cap on the number of hidden layers a recipe may request, so a
#: crafted model file cannot force an unbounded ``Sequential``.
MAX_HIDDEN_LAYERS = 64

#: Deliberate cap on the total parameter count a recipe may request, so a
#: crafted model file cannot force an unbounded allocation before training
#: or inference even runs.
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


# Intentionally not @beartype-decorated: beartype checks only one randomly
# chosen element of a sequence per call, so it could not guarantee every
# hidden_sizes entry is an int (and not a bool, which beartype would accept
# where an int is expected). The explicit checks below raise ValueError for a
# plain but invalid value whether MLP(...) is called directly or built through
# build_from_recipe; a non-plain value is refused earlier, with TypeError, by
# the registry's recipe recorder (build_from_recipe reports it as ValueError).
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
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed ``x`` by applying the linear/activation stack in ``self.net``."""
        return self.net(x)
