# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""The model file format: what ``_model_file.encode`` writes and ``read`` accepts."""

import json
import math
import pickle
import time
from collections import OrderedDict
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

import equine as eq
import equine.utils
from equine import _model_file
from equine.architectures import MLP
from equine.registry import _MAX_PLAIN_DEPTH

VERSION = _model_file.EQUINE_FORMAT_VERSION


def _same(expected: Any, actual: Any) -> bool:
    """Equal value for value, with the same types, dtypes and ``_metadata``."""
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, torch.Tensor):
        return expected.dtype == actual.dtype and torch.equal(expected, actual)
    if isinstance(expected, float):
        return expected == actual or (math.isnan(expected) and math.isnan(actual))
    if isinstance(expected, dict):
        return (
            list(expected) == list(actual)
            and all(_same(expected[k], actual[k]) for k in expected)
            and _same(
                getattr(expected, "_metadata", None), getattr(actual, "_metadata", None)
            )
        )
    if isinstance(expected, (list, tuple)):
        return len(expected) == len(actual) and all(map(_same, expected, actual))
    return expected == actual


def _write_raw(path, header: Any, tensors=None, version: str = str(VERSION)) -> None:
    """A safetensors file whose ``equine`` metadata is ``header`` (JSON-encoded
    unless it is already a string)."""
    text = header if isinstance(header, str) else json.dumps(header)
    metadata = {"equine_format_version": version, "equine": text}
    save_file(tensors or {}, str(path), metadata=metadata)


def test_every_value_kind_reads_back_exactly(tmp_path) -> None:
    ordered = OrderedDict(
        [("w", torch.arange(6.0).reshape(2, 3)), ("b", torch.ones(2))]
    )
    ordered._metadata = OrderedDict([("", {"version": 1})])  # type: ignore[attr-defined]
    tied = torch.randn(3)
    checkpoint = {
        "equine_format_version": VERSION,
        "scalars": [None, True, False, 0, -7, 2**70, 0.1, -0.0, 1e308, 5e-324, "é"],
        "special": [math.nan, math.inf, -math.inf],
        "tuple": (1, (2.5, "x"), []),
        "keys": {7: "int", 2.5: "float", True: "bool", None: "none", "s": "str"},
        "tag_named": {"__tuple__": [1, 2]},
        "with_tag_key": {"__tuple__": 1, "other": 2},
        "ordered": ordered,
        "size": torch.Size([3, 0, 2]),
        "tensors": [
            torch.tensor(3.0, dtype=torch.float64),
            torch.empty(0, 4, dtype=torch.int8),
            torch.arange(12, dtype=torch.int64).reshape(3, 4).t(),
            torch.tensor([True, False]),
            torch.ones(2, dtype=torch.bfloat16),
        ],
        "tied": [tied, tied],
    }
    path = tmp_path / "m.eq"
    _model_file.write(str(path), checkpoint)
    loaded = _model_file.read(str(path))
    assert _same(checkpoint, loaded)
    assert loaded["tied"][0] is not loaded["tied"][1]
    assert loaded["tensors"][2].is_contiguous()


def test_a_saved_model_keeps_its_state_dict_metadata(tmp_path) -> None:
    torch.manual_seed(0)
    X, y = torch.randn(60, 4), torch.arange(60) % 3
    model = eq.EquineProtonet(MLP(4, (8,), 3), emb_out_dim=3)
    model.train_model(
        torch.utils.data.TensorDataset(X, y), 1, support_size=5, episode_size=30
    )
    path = str(tmp_path / "m.eq")
    model.save(path)
    stored = equine.utils.load_checkpoint(path)
    embedding = model.model.embedding_model.state_dict()
    assert stored["embedding_state_dict"]._metadata == embedding._metadata
    assert stored["embedding_recipe"]["kwargs"]["hidden_sizes"] == (8,)


def test_a_file_whose_first_byte_is_a_pickle_opcode_loads(tmp_path) -> None:
    path = tmp_path / "m.eq"
    for pad in range(256):
        header = {"equine_format_version": VERSION, "pad": "a" * pad}
        _write_raw(path, header)
        if path.read_bytes()[:1] == b"\x80":
            break
    assert path.read_bytes()[:1] == b"\x80"
    assert equine.utils.load_checkpoint(str(path))["pad"] == "a" * pad


@pytest.mark.parametrize(
    "content, error, match",
    [
        (
            "zip",
            equine.utils.LegacyFormatError,
            "not in the current EQUINE model format",
        ),
        ("pickle", equine.utils.LegacyFormatError, "not in the current EQUINE model"),
        (b"", ValueError, "is not an EQUINE model file"),
        (b"not a model file at all", ValueError, "is not an EQUINE model file"),
    ],
)
def test_files_that_are_not_safetensors_files_are_refused(
    tmp_path, content, error, match
) -> None:
    path = tmp_path / "m.eq"
    if content == "zip":
        torch.save({"equine_format_version": 3}, path)
    elif content == "pickle":
        path.write_bytes(pickle.dumps({"equine_format_version": 3}, protocol=2))
    else:
        path.write_bytes(content)
    with pytest.raises(error, match=match):
        equine.utils.load_checkpoint(str(path))


@pytest.mark.parametrize(
    "metadata, match",
    [
        (None, "metadata is not exactly"),
        ({"equine_format_version": "4"}, "metadata is not exactly"),
        (
            {"equine_format_version": "4", "equine": "{}", "format": "pt"},
            "metadata is not exactly",
        ),
        ({"equine_format_version": "x", "equine": "{}"}, "is not a number"),
        ({"equine_format_version": "04", "equine": "{}"}, "is not a number"),
        (
            {"equine_format_version": "3", "equine": '{"equine_format_version": 3}'},
            "malformed \\(equine_format_version 3\\)",
        ),
        ({"equine_format_version": "5", "equine": "{}"}, "version 5; this EQUINE"),
    ],
)
def test_metadata_other_than_the_version_and_the_json_is_refused(
    tmp_path, metadata, match
) -> None:
    path = str(tmp_path / "m.eq")
    save_file({}, path, metadata=metadata)
    with pytest.raises(ValueError, match=match):
        _model_file.read(path)


@pytest.mark.parametrize(
    "header, tensors, match",
    [
        ("{", {}, "Expecting property name"),
        ('{"equine_format_version": 4, "x": NaN}', {}, "constant NaN"),
        ('{"equine_format_version": 4, "x": 1e999}', {}, "1e999 is not a finite"),
        ('{"equine_format_version": 4, "x": 1, "x": 2}', {}, "repeats a key"),
        ("[4]", {}, "not a dict with equine_format_version 4"),
        ({"equine_format_version": 3}, {}, "not a dict with equine_format_version 4"),
        ({"equine_format_version": 4.0}, {}, "not a dict with equine_format_version 4"),
        ({"x": {"__float__": "nan"}}, {}, "__float__ tag"),
        ({"x": {"__tuple__": 1}}, {}, "__tuple__ tag that is not a list"),
        ({"x": {"__items__": [[1]]}}, {}, "list of \\[key, value\\] pairs"),
        ({"x": {"__items__": [[[1], 2]]}}, {}, "dict key of type list"),
        ({"x": {"__items__": [[1, 2], [1, 3]]}}, {}, "__items__ tag that repeats"),
        ({"x": {"__odict__": {"items": []}}}, {}, "without exactly items and metadata"),
        (
            {"x": {"__odict__": {"items": [[1, 2], [1, 3]], "metadata": None}}},
            {},
            "__odict__ tag that repeats",
        ),
        (
            {"x": {"__odict__": {"items": [], "metadata": [1]}}},
            {},
            "metadata is not a dict",
        ),
        ({"x": {"__size__": [-1]}}, {}, "__size__ tag"),
        ({"x": {"__size__": [True]}}, {}, "__size__ tag"),
        ({"x": {"__tensor__": 0}}, {"0": torch.ones(1)}, "name is not a string"),
        ({"x": {"__tensor__": "1"}}, {"0": torch.ones(1)}, "not referenced exactly"),
        (
            {"x": [{"__tensor__": "0"}, {"__tensor__": "0"}]},
            {"0": torch.ones(1)},
            "referenced more than once",
        ),
        ({"x": 1}, {"0": torch.ones(1)}, "not referenced exactly once"),
    ],
)
def test_a_malformed_header_is_refused(tmp_path, header, tensors, match) -> None:
    if isinstance(header, dict) and "equine_format_version" not in header:
        header = {"equine_format_version": VERSION, **header}
    path = tmp_path / "m.eq"
    _write_raw(path, header, tensors)
    with pytest.raises(ValueError, match="header is malformed") as err:
        _model_file.read(str(path))
    assert err.match(match)


def test_json_larger_than_the_cap_is_refused_at_save_and_at_read(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(_model_file, "MAX_JSON_BYTES", 100)
    checkpoint = {"equine_format_version": VERSION, "x": "a" * 100}
    with pytest.raises(ValueError, match="need more than 100 bytes of JSON"):
        _model_file.encode(checkpoint)
    path = tmp_path / "m.eq"
    _write_raw(path, checkpoint)
    with pytest.raises(ValueError, match="larger than 100 bytes"):
        _model_file.read(str(path))


def test_a_container_reached_along_many_paths_is_refused_quickly_at_save() -> None:
    node: Any = [1]
    for _ in range(24):
        node = [node, node]
    start = time.perf_counter()
    with pytest.raises(
        ValueError, match=f"need more than {_model_file.MAX_JSON_BYTES} bytes of JSON"
    ):
        _model_file.encode({"equine_format_version": VERSION, "x": node})
    assert time.perf_counter() - start < 1.0


def _nested(depth: int, leaf: Any = 1) -> Any:
    value: Any = leaf
    for _ in range(depth):
        value = [value]
    return value


def test_nesting_is_bounded_where_save_stops(tmp_path) -> None:
    """A recipe argument nested as deep as the registry allows, NaN innermost, saves
    and loads; one more level is refused at save and at read."""
    path = tmp_path / "m.eq"
    recipe = {"builder": "b", "kwargs": {"x": _nested(_MAX_PLAIN_DEPTH, math.nan)}}
    _model_file.write(
        str(path), {"equine_format_version": VERSION, "embedding_recipe": recipe}
    )
    innermost = _model_file.read(str(path))["embedding_recipe"]["kwargs"]["x"]
    for _ in range(_MAX_PLAIN_DEPTH):
        innermost = innermost[0]
    assert math.isnan(innermost)
    recipe["kwargs"]["x"] = _nested(_MAX_PLAIN_DEPTH + 1, math.nan)
    with pytest.raises(
        ValueError, match=f"nests values more than {_model_file._MAX_DEPTH} levels"
    ):
        _model_file.encode(
            {"equine_format_version": VERSION, "embedding_recipe": recipe}
        )
    recipe["kwargs"]["x"] = _nested(_MAX_PLAIN_DEPTH + 1, {"__float__": "NaN"})
    _write_raw(path, {"equine_format_version": VERSION, "embedding_recipe": recipe})
    with pytest.raises(
        ValueError, match=f"nested more than {_model_file._MAX_DEPTH} levels deep"
    ):
        _model_file.read(str(path))


def test_json_too_deep_for_the_parser_is_refused(tmp_path) -> None:
    path = tmp_path / "m.eq"
    depth = 100_000
    _write_raw(
        path, '{"equine_format_version": 4, "x": ' + "[" * depth + "]" * depth + "}"
    )
    with pytest.raises(ValueError, match="header is malformed") as err:
        _model_file.read(str(path))
    assert len(str(err.value)) < 1000


@pytest.mark.parametrize(
    "value, match",
    [
        (torch.ones(2, dtype=torch.complex128), "is a complex128 tensor"),
        (
            torch.ones(2).to_sparse(),
            "is a torch.sparse_coo tensor; store tensor.to_dense",
        ),
        (
            torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.quint8),
            "is a quantized tensor; store tensor.dequantize",
        ),
        (torch.ones(2, device="meta"), "is a meta tensor, which holds no values"),
        ({1, 2}, "is a set, which a model file does not store"),
        (b"bytes", "is a bytes, which a model file does not store"),
        ({(1, 2): 3}, "has a dict key of type tuple"),
    ],
)
def test_values_a_model_file_does_not_store_are_refused_at_save(value, match) -> None:
    with pytest.raises(
        ValueError, match=f"Cannot save: checkpoint\\['x'\\]\\[0\\] {match}"
    ):
        _model_file.encode({"equine_format_version": VERSION, "x": [value]})


@pytest.mark.parametrize(
    "tensor, repair",
    [
        (torch.ones(2).to_sparse(), lambda t: t.to_dense()),
        (
            torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.quint8),
            lambda t: t.dequantize(),
        ),
    ],
    ids=["sparse", "quantized"],
)
def test_the_repair_a_tensor_refusal_names_saves(tensor, repair) -> None:
    with pytest.raises(ValueError, match="Cannot save"):
        _model_file.encode({"equine_format_version": VERSION, "x": tensor})
    tensors, _ = _model_file.encode(
        {"equine_format_version": VERSION, "x": repair(tensor)}
    )
    assert torch.equal(tensors["0"], repair(tensor))


def test_two_nan_keys_read_back_as_two_keys(tmp_path) -> None:
    path = str(tmp_path / "m.eq")
    _model_file.write(
        path,
        {"equine_format_version": VERSION, "x": {float("nan"): 1, float("nan"): 2}},
    )
    keys = list(_model_file.read(path)["x"])
    assert len(keys) == 2 and all(math.isnan(key) for key in keys)
    assert list(_model_file.read(path)["x"].values()) == [1, 2]


def test_dict_keys_whose_hashes_repeat_are_refused_quickly(tmp_path) -> None:
    path = tmp_path / "m.eq"
    colliding = [[k * (2**61 - 1), 0] for k in range(1, 40_001)]
    _write_raw(path, {"equine_format_version": VERSION, "x": {"__items__": colliding}})
    start = time.perf_counter()
    with pytest.raises(ValueError, match="dict keys whose hashes repeat"):
        _model_file.read(str(path))
    assert time.perf_counter() - start < 1.0
    labels = {k: k for k in range(60_000)}
    _model_file.write(str(path), {"equine_format_version": VERSION, "x": labels})
    assert _model_file.read(str(path))["x"] == labels


def test_a_truncated_model_file_is_not_taken_for_an_old_format(tmp_path) -> None:
    path = tmp_path / "m.eq"
    for pad in range(256):
        _write_raw(path, {"equine_format_version": VERSION, "pad": "a" * pad})
        if path.read_bytes()[:1] == b"\x80":
            break
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(ValueError, match="is not an EQUINE model file") as err:
        equine.utils.load_checkpoint(str(path))
    assert not isinstance(err.value, equine.utils.LegacyFormatError)


def test_write_follows_a_link_and_keeps_the_mode(tmp_path) -> None:
    checkpoint = {"equine_format_version": VERSION, "x": torch.ones(2)}
    target = tmp_path / "target.eq"
    target.write_bytes(b"old")
    target.chmod(0o640)
    link = tmp_path / "link.eq"
    link.symlink_to(target)
    _model_file.write(str(link), checkpoint)
    assert link.is_symlink()
    assert (target.stat().st_mode & 0o777) == 0o640
    assert torch.equal(_model_file.read(str(target))["x"], torch.ones(2))


def test_a_save_refused_while_encoding_leaves_the_file_as_it_was(tmp_path) -> None:
    path = tmp_path / "m.eq"
    path.write_bytes(b"old")
    with pytest.raises(ValueError, match="Cannot save"):
        _model_file.write(str(path), {"equine_format_version": VERSION, "x": {1, 2}})
    assert path.read_bytes() == b"old"


def test_a_header_longer_than_save_writes_is_refused_before_it_is_parsed(
    tmp_path,
) -> None:
    path = tmp_path / "m.eq"
    length = _model_file._MAX_HEADER_BYTES + 1
    path.write_bytes(length.to_bytes(8, "little") + b"{}")
    with pytest.raises(
        _model_file.NotSafetensorsError, match=f"its header is {length}"
    ):
        _model_file.read(str(path))
    with pytest.raises(ValueError, match="is not an EQUINE model file") as err:
        equine.utils.load_checkpoint(str(path))
    assert not isinstance(err.value, equine.utils.LegacyFormatError)


def test_a_header_longer_than_a_model_file_may_have_is_refused_at_save(
    monkeypatch,
) -> None:
    monkeypatch.setattr(_model_file, "_MAX_HEADER_BYTES", 200)
    checkpoint = {"equine_format_version": VERSION, "x": [torch.ones(1)] * 5}
    with pytest.raises(
        ValueError, match="header would be [0-9]+ bytes, more than the 200"
    ):
        _model_file.to_bytes(checkpoint)


def test_dict_keys_whose_hashes_repeat_are_refused_at_save() -> None:
    keys = {k * (2**61 - 1): k for k in range(1, 11)}
    with pytest.raises(
        ValueError, match="has more than 8 dict keys whose hashes repeat"
    ):
        _model_file.encode({"equine_format_version": VERSION, "x": keys})
    assert _model_file.encode(
        {"equine_format_version": VERSION, "x": dict(list(keys.items())[:9])}
    )


def test_values_nested_deeper_than_read_allows_are_refused_at_save() -> None:
    deep: Any = OrderedDict()
    deep._metadata = _nested(_model_file._MAX_DEPTH)  # type: ignore[attr-defined]
    looped: list = []
    looped.append(looped)
    for value in (deep, looped):
        with pytest.raises(
            ValueError, match=f"nests values more than {_model_file._MAX_DEPTH} levels"
        ):
            _model_file.encode({"equine_format_version": VERSION, "x": value})


def test_a_container_reused_deeper_is_checked_again() -> None:
    shared = _nested(90)
    deeper: Any = shared
    for _ in range(20):
        deeper = [deeper]
    with pytest.raises(
        ValueError, match=f"nests values more than {_model_file._MAX_DEPTH} levels"
    ):
        _model_file.encode({"equine_format_version": VERSION, "a": shared, "b": deeper})
