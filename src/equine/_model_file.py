# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
EQUINE model files: safetensors files whose header metadata holds the checkpoint.

The metadata is exactly ``{"equine_format_version": "4", "equine": <JSON>}``. The JSON
is the checkpoint dictionary with each tensor replaced by ``{"__tensor__": "<name>"}``,
a tensor stored in the file under that name. Values JSON does not hold are written as
one-key objects (tags): ``{"__float__": "NaN" | "Infinity" | "-Infinity"}``,
``{"__tuple__": [...]}``, ``{"__items__": [[key, value], ...]}`` for a dict with a key
that is not a string (or whose only key is a tag name), ``{"__odict__": {"items":
[...], "metadata": ...}}`` for an ``OrderedDict`` and its ``_metadata``, and
``{"__size__": [...]}`` for a ``torch.Size``.
"""

import json
import math
import re
from collections import Counter, OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save

from .registry import _MAX_PLAIN_DEPTH, _truncate

EQUINE_FORMAT_VERSION = 4

_VERSION_KEY = "equine_format_version"
_JSON_KEY = "equine"

# Largest ``equine`` JSON, in bytes, that ``encode`` writes and ``read`` parses.
MAX_JSON_BYTES = 4 * 2**20

# Deepest nesting of values in a checkpoint: the checkpoint, the recipe and its
# kwargs, plain values nested at most ``_MAX_PLAIN_DEPTH`` levels, and a tag (a
# tensor, a torch.Size, NaN or infinity) as the innermost value.
_MAX_DEPTH = _MAX_PLAIN_DEPTH + 4

_TAGS = frozenset(
    {"__float__", "__tuple__", "__items__", "__odict__", "__size__", "__tensor__"}
)
_FLOATS = frozenset({"NaN", "Infinity", "-Infinity"})
_KEY_TYPES = (type(None), bool, int, float, str)

# Most keys of one dict that may share a hash with another of its keys.
_MAX_HASH_REPEATS = 8

# Longest safetensors header, in bytes, that ``to_bytes`` writes and ``read`` opens:
# the JSON, which the header escapes, and an entry for each tensor.
_MAX_HEADER_BYTES = 3 * MAX_JSON_BYTES

# Dtypes a model file stores.
_DTYPES = frozenset(
    {
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    }
)

_MALFORMED = (
    "Model file's header is malformed ({}); the file was not written by save() or was "
    "changed after. Save the model again, or upgrade the original file again with "
    "`python -m equine.upgrade`."
)


class NotSafetensorsError(ValueError):
    """The file is not a safetensors file."""


def _cannot_save(where: str, what: str) -> ValueError:
    return ValueError(f"Cannot save: {where} {what}.")


def _too_much_json() -> ValueError:
    return _cannot_save(
        "the model's settings, recipe, feature and label names and train summary",
        f"need more than {MAX_JSON_BYTES} bytes of JSON; store less in them (a list "
        "passed in several places counts once for each place)",
    )


def _stored(tensor: torch.Tensor, where: str) -> torch.Tensor:
    """A contiguous CPU copy of ``tensor`` with its own storage."""
    if tensor.is_quantized:
        raise _cannot_save(where, "is a quantized tensor; store tensor.dequantize()")
    if tensor.is_meta:
        raise _cannot_save(
            where,
            "is a meta tensor, which holds no values; give the module its values "
            "(for example with module.to_empty(device='cpu') and load_state_dict) "
            "before saving",
        )
    if tensor.layout != torch.strided:
        raise _cannot_save(
            where, f"is a {tensor.layout} tensor; store tensor.to_dense()"
        )
    if tensor.dtype not in _DTYPES:
        raise _cannot_save(
            where,
            f"is a {str(tensor.dtype).removeprefix('torch.')} tensor, which a model "
            "file does not store; convert it to a supported dtype "
            f"({', '.join(sorted(str(d).removeprefix('torch.') for d in _DTYPES))})",
        )
    return tensor.detach().to("cpu").clone(memory_format=torch.contiguous_format)


def encode(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """
    The tensors and the header metadata of the model file for ``checkpoint``.

    Each place a tensor appears gets its own stored copy. A container reached along
    several paths is written once per path, and counts toward ``MAX_JSON_BYTES`` each
    time.

    Raises
    ------
    ValueError
        If ``checkpoint`` holds a value of a type the file does not store, a tensor
        that is not dense or whose dtype the file does not store, or more than
        ``MAX_JSON_BYTES`` of JSON.
    """
    tensors: dict[str, torch.Tensor] = {}
    budget = [MAX_JSON_BYTES]
    # The encoded form and size of each container without tensors already walked,
    # by the container's id and depth.
    done: dict[tuple[int, int], tuple[Any, int]] = {}

    def charge(nodes: int) -> None:
        budget[0] -= nodes
        if budget[0] < 0:
            raise _too_much_json()

    def walk(value: Any, where: str, depth: int) -> Any:
        kind = type(value)
        if value is None or kind in (bool, int, str):
            charge(1)
            return value
        if kind is float and math.isfinite(value):
            charge(1)
            return value
        if depth >= _MAX_DEPTH:
            raise _cannot_save(
                where,
                f"nests values more than {_MAX_DEPTH} levels deep (a tensor, "
                "torch.Size, NaN or infinity counts as a level); store a flatter value",
            )
        if kind in (list, tuple, dict, OrderedDict):
            if (id(value), depth) in done:
                encoded, size = done[id(value), depth]
                charge(size)
                return encoded
            before, stored = budget[0], len(tensors)
            charge(1)
            encoded = container(value, kind, where, depth + 1)
            if len(tensors) == stored:
                done[id(value), depth] = (encoded, before - budget[0])
            return encoded
        charge(1)
        if kind is float:
            name = (
                "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
            )
            return {"__float__": name}
        if kind is torch.Size:
            return {"__size__": list(value)}
        if kind in (torch.Tensor, torch.nn.Parameter):
            name = str(len(tensors))
            tensors[name] = _stored(value, where)
            return {"__tensor__": name}
        raise _cannot_save(
            where,
            f"is a {kind.__name__}, which a model file does not store; use None, "
            "bool, int, float, str, list, tuple, dict, OrderedDict, torch.Size or "
            "torch.Tensor",
        )

    def container(value: Any, kind: type, where: str, depth: int) -> Any:
        if kind is list:
            return [walk(item, f"{where}[{i}]", depth) for i, item in enumerate(value)]
        if kind is tuple:
            return {
                "__tuple__": [
                    walk(item, f"{where}[{i}]", depth) for i, item in enumerate(value)
                ]
            }
        if (
            kind is dict
            and all(type(key) is str for key in value)
            and not (len(value) == 1 and next(iter(value)) in _TAGS)
        ):
            return {
                key: walk(item, f"{where}[{key!r}]", depth)
                for key, item in value.items()
            }
        bad = next((key for key in value if type(key) not in _KEY_TYPES), None)
        if bad is not None:
            raise _cannot_save(
                where,
                f"has a dict key of type {type(bad).__name__}; use str, int, float, "
                "bool or None keys",
            )
        if len(value) - len({hash(key) for key in value}) > _MAX_HASH_REPEATS:
            raise _cannot_save(
                where,
                f"has more than {_MAX_HASH_REPEATS} dict keys whose hashes repeat; "
                "use other keys, for example distinct small integers or strings",
            )
        items = [
            [walk(key, f"{where} key", depth), walk(item, f"{where}[{key!r}]", depth)]
            for key, item in value.items()
        ]
        if kind is dict:
            return {"__items__": items}
        meta = getattr(value, "_metadata", None)
        return {
            "__odict__": {
                "items": items,
                "metadata": None
                if meta is None
                else walk(meta, f"{where} metadata", depth),
            }
        }

    text = json.dumps(walk(checkpoint, "checkpoint", 0), allow_nan=False)
    if len(text.encode()) > MAX_JSON_BYTES:
        raise _too_much_json()
    return tensors, {_VERSION_KEY: str(EQUINE_FORMAT_VERSION), _JSON_KEY: text}


def write(path: str, checkpoint: dict[str, Any]) -> None:
    """Write ``checkpoint`` to ``path`` as a model file, opening ``path`` for writing
    only once the file's bytes are built."""
    data = to_bytes(checkpoint)
    with open(path, "wb") as file:
        file.write(data)


def to_bytes(checkpoint: dict[str, Any]) -> bytes:
    """
    The bytes of the model file for ``checkpoint``.

    Raises
    ------
    ValueError
        As ``encode`` does, and if the file's header would be longer than
        ``_MAX_HEADER_BYTES``.
    """
    tensors, metadata = encode(checkpoint)
    data = save(tensors, metadata=metadata)
    header = int.from_bytes(data[:8], "little")
    if header > _MAX_HEADER_BYTES:
        raise _cannot_save(
            "the model file's header",
            f"would be {header} bytes, more than the {_MAX_HEADER_BYTES} a model file "
            "may have; store fewer tensors, or less in the settings, recipe, names and "
            "train summary",
        )
    return data


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"JSON constant {name} is not valid JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"JSON number {_truncate(text, 50)} is not a finite float")
    return value


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("an object repeats a key")
    return result


def _tensor_names(node: Any) -> list[Any]:
    """The names of every ``{"__tensor__": name}`` in the parsed JSON ``node``."""
    names: list[Any] = []
    pending = [node]
    while pending:
        item = pending.pop()
        if type(item) is list:
            pending.extend(item)
        elif type(item) is dict:
            if len(item) == 1 and "__tensor__" in item:
                names.append(item["__tensor__"])
            else:
                pending.extend(item.values())
    return names


def _decode(node: Any, tensors: Mapping[str, torch.Tensor], depth: int) -> Any:
    """The value the parsed JSON ``node`` encodes, at container depth ``depth``."""
    if node is None or type(node) in (bool, int, float, str):
        return node
    if depth >= _MAX_DEPTH:
        raise ValueError(f"values nested more than {_MAX_DEPTH} levels deep")
    if type(node) is list:
        return [_decode(item, tensors, depth + 1) for item in node]
    if len(node) == 1 and next(iter(node)) in _TAGS:
        ((tag, body),) = node.items()
        return _decode_tag(tag, body, tensors, depth)
    return {key: _decode(item, tensors, depth + 1) for key, item in node.items()}


def _decode_items(body: Any, tensors: Mapping[str, torch.Tensor], depth: int) -> list:
    if type(body) is not list or any(
        type(pair) is not list or len(pair) != 2 for pair in body
    ):
        raise ValueError("an __items__ tag that is not a list of [key, value] pairs")
    items = []
    for key, value in body:
        key = _decode(key, tensors, depth + 1)
        if type(key) not in _KEY_TYPES:
            raise ValueError(f"a dict key of type {type(key).__name__}")
        items.append((key, _decode(value, tensors, depth + 1)))
    if len(items) - len({hash(key) for key, _ in items}) > _MAX_HASH_REPEATS:
        raise ValueError("dict keys whose hashes repeat")
    return items


def _decode_tag(
    tag: str, body: Any, tensors: Mapping[str, torch.Tensor], depth: int
) -> Any:
    if tag == "__float__":
        if type(body) is not str or body not in _FLOATS:
            raise ValueError("a __float__ tag that is not NaN, Infinity or -Infinity")
        return float(body)
    if tag == "__tuple__":
        if type(body) is not list:
            raise ValueError("a __tuple__ tag that is not a list")
        return tuple(_decode(item, tensors, depth + 1) for item in body)
    if tag == "__items__":
        items = _decode_items(body, tensors, depth)
        result = dict(items)
        if len(result) != len(items):
            raise ValueError("an __items__ tag that repeats a key")
        return result
    if tag == "__odict__":
        if type(body) is not dict or set(body) != {"items", "metadata"}:
            raise ValueError("an __odict__ tag without exactly items and metadata")
        items = _decode_items(body["items"], tensors, depth)
        odict = OrderedDict(items)
        if len(odict) != len(items):
            raise ValueError("an __odict__ tag that repeats a key")
        if body["metadata"] is not None:
            meta = _decode(body["metadata"], tensors, depth + 1)
            if not isinstance(meta, dict):
                raise ValueError("an __odict__ tag whose metadata is not a dict")
            odict._metadata = meta  # type: ignore[attr-defined]
        return odict
    if tag == "__size__":
        if type(body) is not list or any(type(n) is not int or n < 0 for n in body):
            raise ValueError("a __size__ tag that is not a list of sizes")
        return torch.Size(body)
    return tensors[body]


def read(path: str) -> dict[str, Any]:
    """
    The checkpoint stored in the model file ``path``, its tensors on the CPU.

    The header may be at most ``_MAX_HEADER_BYTES`` long. Before any tensor is read,
    the metadata must be exactly the format version and the JSON; the JSON at most ``MAX_JSON_BYTES`` and strict (no NaN constants,
    numbers too large for a float or repeated keys); and every tensor in the file
    referenced exactly once. While the values are rebuilt, the tags must be well
    formed, containers nested at most ``_MAX_DEPTH`` deep, and the keys of a dict
    may share a hash at most ``_MAX_HASH_REPEATS`` times. Each tensor is read once and
    copied off the file.

    Raises
    ------
    NotSafetensorsError
        If the file is not a safetensors file.
    ValueError
        If the file declares a newer format version, or its header is malformed.
    """
    with open(path, "rb") as start:
        header = int.from_bytes(start.read(8), "little")
    if header > _MAX_HEADER_BYTES:
        raise NotSafetensorsError(
            f"'{path}' is not an EQUINE model file (its header is {header} bytes; "
            f"save() writes at most {_MAX_HEADER_BYTES}); load a file written by save()."
        )
    try:
        handle = safe_open(path, framework="pt", device="cpu")
    except Exception as err:
        raise NotSafetensorsError(
            f"'{path}' is not an EQUINE model file ({_truncate(str(err), 200)}); "
            "load a file written by save()."
        ) from err
    with handle as file:
        metadata = file.metadata()
        if not isinstance(metadata, dict) or set(metadata) != {_VERSION_KEY, _JSON_KEY}:
            raise ValueError(
                _MALFORMED.format(
                    f"its metadata is not exactly {_VERSION_KEY} and {_JSON_KEY}"
                )
            )
        version = metadata[_VERSION_KEY]
        if not re.fullmatch("0|[1-9][0-9]{0,5}", version):
            raise ValueError(_MALFORMED.format(f"{_VERSION_KEY} is not a number"))
        if int(version) > EQUINE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported EQUINE model format version {int(version)}; this EQUINE "
                f"supports up to {EQUINE_FORMAT_VERSION}. Upgrade EQUINE."
            )
        if int(version) != EQUINE_FORMAT_VERSION:
            raise ValueError(_MALFORMED.format(f"{_VERSION_KEY} {version}"))
        text = metadata[_JSON_KEY]
        try:
            if len(text.encode()) > MAX_JSON_BYTES:
                raise ValueError(f"its JSON is larger than {MAX_JSON_BYTES} bytes")
            node = json.loads(
                text,
                parse_constant=_refuse_constant,
                parse_float=_finite_float,
                object_pairs_hook=_unique_keys,
            )
            names = _tensor_names(node)
            if any(type(name) is not str for name in names):
                raise ValueError("a __tensor__ tag whose name is not a string")
            counts = Counter(names)
            if any(count > 1 for count in counts.values()):
                raise ValueError("a tensor referenced more than once")
            if set(counts) != set(file.keys()):
                raise ValueError("tensors that are not referenced exactly once")
            checkpoint = _decode(
                node, {name: file.get_tensor(name).clone() for name in counts}, 0
            )
        except Exception as err:
            raise ValueError(_MALFORMED.format(_truncate(str(err), 200))) from err
    stored = checkpoint.get(_VERSION_KEY) if type(checkpoint) is dict else None
    if type(stored) is not int or stored != int(version):
        raise ValueError(
            _MALFORMED.format(f"its JSON is not a dict with {_VERSION_KEY} {version}")
        )
    return checkpoint
