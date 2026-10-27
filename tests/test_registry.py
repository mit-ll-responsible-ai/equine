# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Registry of embedding architectures: recipes recorded at construction, rebuilt by name."""

import collections
import enum
import re
import sys
import threading
import time
import tracemalloc
import warnings
from typing import Any

import numpy
import pytest
import torch
from conftest import BasicEmbeddingModel

import equine as eq
import equine.utils
from equine import architectures, registry

_TORCH_CRASHES_WITH_THREADS = pytest.mark.skipif(
    sys.platform == "darwin"
    and sys.version_info >= (3, 15)
    and equine.utils._torch_version() < (2, 15),
    reason="torch < 2.15 crashes on macOS with Python 3.15 when threads use it",
)


def _fresh_name(suffix: str) -> str:
    return f"equine.tests.registry.{suffix}"


# --------------------------------------------------------------------------- #
# Baseline behavior
# --------------------------------------------------------------------------- #


def test_decorated_class_records_recipe_with_defaults_applied() -> None:
    @registry.embedding_architecture(_fresh_name("defaults"))
    class Net(torch.nn.Module):
        def __init__(self, width: int, depth: int = 2, act: str = "relu") -> None:
            super().__init__()
            self.lin = torch.nn.Linear(width, width)

    net = Net(8)
    assert registry.embedding_recipe(net) == {
        "builder": _fresh_name("defaults"),
        "kwargs": {"width": 8, "depth": 2, "act": "relu"},
    }
    assert _fresh_name("defaults") in registry.registered_architectures()


def test_build_from_recipe_calls_registered_constructor() -> None:
    @registry.embedding_architecture(_fresh_name("build"))
    class Net(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(width, 1)

    built = registry.build_from_recipe(
        {"builder": _fresh_name("build"), "kwargs": {"width": 5}}
    )
    assert isinstance(built, Net)
    assert built.lin.in_features == 5


def test_non_plain_constructor_argument_is_rejected_at_construction() -> None:
    @registry.embedding_architecture(_fresh_name("nonplain"))
    class Net(torch.nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight)

    with pytest.raises(TypeError, match="'weight'"):
        Net(torch.zeros(2))


def test_var_positional_constructor_is_rejected_at_registration() -> None:
    class Net(torch.nn.Module):
        def __init__(self, *sizes: int) -> None:
            super().__init__()

    with pytest.raises(TypeError, match=r"\*args"):
        registry.register_embedding_architecture(_fresh_name("varargs"), Net)


def test_var_keyword_arguments_are_flattened_into_recipe() -> None:
    @registry.embedding_architecture(_fresh_name("varkw"))
    class Net(torch.nn.Module):
        def __init__(self, width: int, **extra: int) -> None:
            super().__init__()

    assert registry.embedding_recipe(Net(3, depth=4))["kwargs"] == {
        "width": 3,
        "depth": 4,
    }


def test_duplicate_name_for_different_class_is_rejected_and_same_class_is_noop() -> (
    None
):
    name = _fresh_name("dup")

    @registry.embedding_architecture(name)
    class A(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class B(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

    registry.register_embedding_architecture(name, A)  # idempotent
    with pytest.raises(ValueError, match="already registered"):
        registry.register_embedding_architecture(name, B)


def test_unknown_recipe_error_lists_registered_names_and_override() -> None:
    with pytest.raises(ValueError) as err:
        registry.build_from_recipe({"builder": "nobody.home", "kwargs": {}})
    message = str(err.value)
    assert "'nobody.home'" in message
    assert "not registered" in message
    assert "embedding_model=" in message
    assert "embedding_architecture" in message


def test_unregistered_module_has_no_recipe() -> None:
    assert registry.embedding_recipe(torch.nn.Linear(2, 2)) is None


def test_public_names_are_declared() -> None:
    assert sorted(registry.__all__) == [
        "embedding_architecture",
        "embedding_recipe",
        "register_embedding_architecture",
        "registered_architectures",
    ]
    assert architectures.__all__ == ["MLP"]
    assert all(
        getattr(eq, name) is getattr(registry, name) for name in registry.__all__
    )


# --------------------------------------------------------------------------- #
# Exact-class recording
# --------------------------------------------------------------------------- #


def test_unregistered_subclass_of_registered_class_has_no_recipe() -> None:
    @registry.embedding_architecture(_fresh_name("exact_parent"))
    class Parent(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.width = width

    class Child(Parent):
        pass

    child = Child(4)
    assert registry.embedding_recipe(child) is None


def test_registered_subclass_records_its_own_name() -> None:
    @registry.embedding_architecture(_fresh_name("exact_parent2"))
    class Parent(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()

    @registry.embedding_architecture(_fresh_name("exact_child2"))
    class Child(Parent):
        def __init__(self, width: int, depth: int) -> None:
            super().__init__(width)
            self.depth = depth

    child = Child(4, 3)
    assert registry.embedding_recipe(child) == {
        "builder": _fresh_name("exact_child2"),
        "kwargs": {"width": 4, "depth": 3},
    }
    # The parent's own recipe machinery must not have clobbered anything.
    assert registry.embedding_recipe(Parent(1))["builder"] == _fresh_name(
        "exact_parent2"
    )


# --------------------------------------------------------------------------- #
# Strict plain-value check
# --------------------------------------------------------------------------- #

_StrEnum = enum.Enum("_StrEnum", {"A": "a"}, type=str)
_Point = collections.namedtuple("_Point", ["x", "y"])

ACCEPTED_PLAIN_VALUES = [
    1,
    1.5,
    "s",
    True,
    None,
    [1, "a", None, [2, 3]],
    (1, 2, 3),
    {"a": 1, "b": {"c": [1, 2]}},
    {1: "x", 2: "y"},
]

REJECTED_PLAIN_VALUES = [
    numpy.float64(1.0),
    _StrEnum.A,
    _Point(1, 2),
    collections.defaultdict(int),
    torch.zeros(1),
]


@pytest.mark.parametrize("value", ACCEPTED_PLAIN_VALUES)
def test_accepted_plain_values_survive_weights_only_round_trip(value, tmp_path) -> None:
    @registry.embedding_architecture(_fresh_name("plain_ok"))
    class Net(torch.nn.Module):
        def __init__(self, value) -> None:
            super().__init__()

    recipe = registry.embedding_recipe(Net(value))
    path = tmp_path / "recipe.pt"
    torch.save(recipe, path)
    loaded = torch.load(path, weights_only=True)
    assert loaded == recipe


@pytest.mark.parametrize("value", REJECTED_PLAIN_VALUES)
def test_rejected_non_plain_values_raise_type_error(value) -> None:
    @registry.embedding_architecture(_fresh_name("plain_bad"))
    class Net(torch.nn.Module):
        def __init__(self, value) -> None:
            super().__init__()

    with pytest.raises(TypeError, match="'value'"):
        Net(value)


def test_dict_with_non_plain_key_raises_type_error_naming_allowed_key_types() -> None:
    @registry.embedding_architecture(_fresh_name("plain_bad_dictkey"))
    class Net(torch.nn.Module):
        def __init__(self, value) -> None:
            super().__init__()

    with pytest.raises(TypeError, match="'value") as err:
        Net({(1, 2): "x"})
    message = str(err.value)
    assert "str" in message
    assert "int" in message


# --------------------------------------------------------------------------- #
# Shared, deeply nested and self-referencing values
# --------------------------------------------------------------------------- #


def _architecture_taking(suffix: str) -> type[torch.nn.Module]:
    """A registered architecture whose only constructor argument is ``options``."""

    @registry.embedding_architecture(_fresh_name(suffix))
    class Net(torch.nn.Module):
        def __init__(self, options: Any = None) -> None:
            super().__init__()

    return Net


def _nested(depth: int, leaf: Any = 1) -> Any:
    """``leaf`` inside ``depth`` nested lists."""
    for _ in range(depth):
        leaf = [leaf]
    return leaf


def test_containers_shared_many_times_are_checked_once(tmp_path) -> None:
    """Each level holds the level below twice, so a walk of every path would
    visit 2**25 lists; a file stores each list once."""
    Net = _architecture_taking("shared_dag")
    node: Any = [1]
    for _ in range(24):
        node = [node, node]

    start = time.perf_counter()
    recipe = registry.embedding_recipe(Net(node))
    path = tmp_path / "recipe.pt"
    torch.save(recipe, path)
    rebuilt = registry.build_from_recipe(torch.load(path, weights_only=True))
    assert time.perf_counter() - start < 2.0

    assert path.stat().st_size < 10_000
    options = registry.embedding_recipe(rebuilt)["kwargs"]["options"]
    assert options[0] is options[1]  # the recipe's copies keep the sharing


def test_a_list_passed_twice_is_accepted_and_round_trips(tmp_path) -> None:
    @registry.embedding_architecture(_fresh_name("shared_list"))
    class Net(torch.nn.Module):
        def __init__(self, sizes: list, other_sizes: list, nested: dict) -> None:
            super().__init__()

    sizes = [4, 8]
    recipe = registry.embedding_recipe(Net(sizes, sizes, {"a": sizes, "b": [sizes]}))
    assert recipe["kwargs"] == {
        "sizes": [4, 8],
        "other_sizes": [4, 8],
        "nested": {"a": [4, 8], "b": [[4, 8]]},
    }
    path = tmp_path / "recipe.pt"
    torch.save(recipe, path)
    rebuilt = registry.build_from_recipe(torch.load(path, weights_only=True))
    assert registry.embedding_recipe(rebuilt) == recipe


def test_a_container_shared_by_many_arguments_is_checked_once() -> None:
    """Many keyword arguments can hold one large list, which a file stores
    once; walking it once per argument would visit 40 million items."""

    @registry.embedding_architecture(_fresh_name("shared_kwargs"))
    class Net(torch.nn.Module):
        def __init__(self, **options: Any) -> None:
            super().__init__()

    shared = [0] * 20_000
    kwargs: dict[str, Any] = {f"k{i}": shared for i in range(2_000)}
    start = time.perf_counter()
    Net(**kwargs)
    with pytest.raises(TypeError, match="'last' .* got set"):
        Net(**kwargs, last={1})
    with pytest.raises(ValueError, match="'last' .* got set"):
        registry.build_from_recipe(
            {"builder": _fresh_name("shared_kwargs"), "kwargs": {**kwargs, "last": {1}}}
        )
    assert time.perf_counter() - start < 10.0


def _reached_again(depth: int) -> list:
    """A chain of 50 lists, then ``depth`` lists ending in that same chain: the
    deepest nesting (1 + ``depth`` + 50 lists) is only reached the second time
    the chain is."""
    inner = _nested(50)
    return [inner, _nested(depth, inner)]


def test_nesting_up_to_the_limit_is_accepted() -> None:
    Net = _architecture_taking("deep_ok")
    limit = registry._MAX_PLAIN_DEPTH
    assert limit == 100
    Net(_nested(limit))
    Net({"a": _nested(limit - 1)})
    Net(_reached_again(49))
    registry.build_from_recipe(
        {"builder": _fresh_name("deep_ok"), "kwargs": {"options": _nested(limit)}}
    )


def _shallow_then_deep() -> list:
    """Two chains of 60 lists; the second ends in the first, so the deepest
    nesting (121 lists) is only reached through the second."""
    shallow = _nested(60)
    return [shallow, _nested(60, shallow)]


@pytest.mark.parametrize(
    "value",
    [
        _nested(101),
        {"a": {"b": _nested(99)}},
        _reached_again(50),
        _shallow_then_deep(),
        _shallow_then_deep()[::-1],
    ],
    ids=[
        "lists",
        "dicts",
        "reached-again",
        "shared-shallow-first",
        "shared-deep-first",
    ],
)
def test_nesting_beyond_the_limit_is_refused(value) -> None:
    Net = _architecture_taking("deep_bad")
    with pytest.raises(TypeError, match="more than 100 levels deep") as err:
        Net(value)
    assert "options[" in str(err.value)  # names where the nesting goes too deep
    assert len(str(err.value)) < 1_000


def test_depth_counts_from_each_argument_that_reaches_a_shared_container() -> None:
    @registry.embedding_architecture(_fresh_name("deep_shared"))
    class Net(torch.nn.Module):
        def __init__(self, first: Any, second: Any) -> None:
            super().__init__()

    inner = _nested(50)
    Net(inner, _nested(50, inner))
    Net(_nested(50, inner), inner)
    with pytest.raises(TypeError, match=r"'second\[0\].* more than 100 levels deep"):
        Net(inner, _nested(51, inner))


def test_deep_recipe_from_a_file_fails_fast_without_recursing() -> None:
    """A recipe from a file nested far deeper than the limit is refused quickly, before
    anything recurses through it."""
    _architecture_taking("deep_file")
    start = time.perf_counter()
    with pytest.raises(ValueError, match="more than 100 levels deep") as err:
        registry.build_from_recipe(
            {"builder": _fresh_name("deep_file"), "kwargs": {"options": _nested(10**5)}}
        )
    assert time.perf_counter() - start < 2.0
    assert len(str(err.value)) < 1_000


def test_a_value_that_contains_itself_is_refused() -> None:
    Net = _architecture_taking("cycle")
    loop: list = [1]
    loop.append({"back": loop})
    with pytest.raises(TypeError, match="contains itself"):
        Net(loop)
    with pytest.raises(ValueError, match="contains itself"):
        registry.build_from_recipe(
            {"builder": _fresh_name("cycle"), "kwargs": {"options": loop}}
        )


def test_error_location_stays_bounded_for_huge_keys() -> None:
    Net = _architecture_taking("huge_keys")
    with pytest.raises(TypeError, match="got set") as err:
        Net({"k" * 100_000: {10**5000: [set()]}})
    assert len(str(err.value)) < 1_000


@pytest.mark.parametrize("depth", [100, 150], ids=["at-limit", "over-limit"])
def test_long_keys_along_a_chain_take_bounded_memory(depth) -> None:
    """A file can repeat one long key at every level of a chain of dicts while
    storing it once; the check must not build a location string per level."""
    Net = _architecture_taking("long_keys")
    key = "k" * 10_000
    value: Any = 1
    for _ in range(depth):
        value = {key: value}
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        baseline = tracemalloc.get_traced_memory()[0]
        if depth <= registry._MAX_PLAIN_DEPTH:
            Net(value)
        else:
            with pytest.raises(TypeError, match="levels deep") as err:
                Net(value)
            assert len(str(err.value)) < 1_000
        peak = tracemalloc.get_traced_memory()[1] - baseline
    finally:
        if not was_tracing:
            tracemalloc.stop()
    assert peak < 5_000_000


# --------------------------------------------------------------------------- #
# Copies, not references
# --------------------------------------------------------------------------- #


def test_constructor_mutating_its_list_argument_does_not_affect_recorded_recipe() -> (
    None
):
    @registry.embedding_architecture(_fresh_name("copies_ctor_mutate"))
    class Net(torch.nn.Module):
        def __init__(self, sizes: list) -> None:
            super().__init__()
            sizes.append(999)

    original = [1, 2, 3]
    net = Net(original)
    assert registry.embedding_recipe(net)["kwargs"]["sizes"] == [1, 2, 3]
    assert original == [1, 2, 3, 999]


def test_mutating_returned_recipe_does_not_affect_instance_recipe() -> None:
    @registry.embedding_architecture(_fresh_name("copies_caller_mutate"))
    class Net(torch.nn.Module):
        def __init__(self, sizes: list) -> None:
            super().__init__()

    net = Net([1, 2, 3])
    recipe = registry.embedding_recipe(net)
    recipe["kwargs"]["sizes"].append(999)
    assert registry.embedding_recipe(net)["kwargs"]["sizes"] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Positional-only, *args-first, and other unrecordable constructor shapes
# --------------------------------------------------------------------------- #


def test_positional_only_constructor_parameter_is_rejected_at_registration() -> None:
    class Net(torch.nn.Module):
        def __init__(self, w, /) -> None:
            super().__init__()

    with pytest.raises(TypeError, match="positional-only"):
        registry.register_embedding_architecture(_fresh_name("posonly"), Net)


def test_first_parameter_is_skipped_by_position_not_name() -> None:
    @registry.embedding_architecture(_fresh_name("notself"))
    class Net(torch.nn.Module):
        def __init__(this, width: int) -> None:
            torch.nn.Module.__init__(this)

    net = Net(7)
    assert registry.embedding_recipe(net) == {
        "builder": _fresh_name("notself"),
        "kwargs": {"width": 7},
    }


def test_first_parameter_absorbed_into_var_positional_is_rejected() -> None:
    class Net(torch.nn.Module):
        def __init__(*args, **kwargs) -> None:
            torch.nn.Module.__init__(args[0])

    with pytest.raises(TypeError, match="first parameter"):
        registry.register_embedding_architecture(_fresh_name("firstvarpos"), Net)


def test_first_parameter_keyword_only_is_rejected() -> None:
    class Net(torch.nn.Module):
        def __init__(*, width: int) -> None:  # no usable instance parameter
            pass

    with pytest.raises(TypeError, match="first parameter"):
        registry.register_embedding_architecture(_fresh_name("firstkwonly"), Net)


def test_first_parameter_var_keyword_is_rejected() -> None:
    class Net(torch.nn.Module):
        def __init__(**kwargs) -> None:
            pass

    with pytest.raises(TypeError, match="first parameter"):
        registry.register_embedding_architecture(_fresh_name("firstvarkw"), Net)


# --------------------------------------------------------------------------- #
# Redefinition is allowed
# --------------------------------------------------------------------------- #


def test_redefining_same_class_replaces_registration_with_warning() -> None:
    name = _fresh_name("redefine")

    class Net(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()

    registry.register_embedding_architecture(name, Net)
    first_cls = Net

    class Net(torch.nn.Module):  # noqa: F811 - intentional redefinition, simulates a reload
        def __init__(self, width: int, depth: int = 1) -> None:
            super().__init__()

    assert Net is not first_cls
    assert Net.__qualname__ == first_cls.__qualname__
    assert Net.__module__ == first_cls.__module__

    with pytest.warns(UserWarning, match="re-registered"):
        registry.register_embedding_architecture(name, Net)

    built = registry.build_from_recipe(
        {"builder": name, "kwargs": {"width": 2, "depth": 5}}
    )
    assert isinstance(built, Net)


def test_registering_class_under_second_name_raises_value_error() -> None:
    @registry.embedding_architecture(_fresh_name("firstname"))
    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

    with pytest.raises(ValueError, match="already registered"):
        registry.register_embedding_architecture(_fresh_name("secondname"), Net)


@pytest.mark.parametrize(
    "name",
    [("equine", "tests"), 1, None, b"equine.tests.bytes", numpy.str_("equine.tests")],
    ids=["tuple", "int", "none", "bytes", "numpy-str"],
)
def test_architecture_names_must_be_strings(name) -> None:
    class Net(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()

    with pytest.raises(TypeError, match="must be a str"):
        registry.register_embedding_architecture(name, Net)
    with pytest.raises(TypeError, match="must be a str"):
        registry.embedding_architecture(name)
    assert "equine.mlp" in registry.registered_architectures()
    with pytest.raises(ValueError, match="not registered"):
        registry.build_from_recipe({"builder": "equine.tests.missing"})


def test_architecture_names_must_not_be_empty() -> None:
    class Net(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()

    with pytest.raises(ValueError, match="must not be empty"):
        registry.register_embedding_architecture("", Net)
    with pytest.raises(ValueError, match="must not be empty"):
        registry.embedding_architecture("")


def test_invalid_redefinition_raises_without_warning_and_keeps_old_entry() -> None:
    name = _fresh_name("redefine_invalid")

    class Net(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()

    registry.register_embedding_architecture(name, Net)
    original_cls = Net

    class Net(torch.nn.Module):  # noqa: F811 - intentional redefinition, now invalid
        def __init__(self, *sizes: int) -> None:
            super().__init__()

    assert Net.__qualname__ == original_cls.__qualname__
    assert Net.__module__ == original_cls.__module__

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        with pytest.raises(TypeError, match=r"\*args"):
            registry.register_embedding_architecture(name, Net)

    built = registry.build_from_recipe({"builder": name, "kwargs": {"width": 9}})
    assert type(built) is original_cls


# --------------------------------------------------------------------------- #
# Warnings point at the caller, not at this module
# --------------------------------------------------------------------------- #


def test_register_embedding_architecture_warning_points_at_caller_line() -> None:
    name = _fresh_name("stacklevel_direct")

    class Net(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()

    registry.register_embedding_architecture(name, Net)

    class Net(torch.nn.Module):  # noqa: F811 - intentional redefinition
        def __init__(self, width: int, depth: int = 1) -> None:
            super().__init__()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        registry.register_embedding_architecture(name, Net)

    assert len(caught) == 1
    assert caught[0].filename == __file__


def test_embedding_architecture_decorator_warning_points_at_caller_line() -> None:
    name = _fresh_name("stacklevel_decorator")

    @registry.embedding_architecture(name)
    class Net(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @registry.embedding_architecture(name)
        class Net(torch.nn.Module):  # noqa: F811 - intentional redefinition
            def __init__(self, width: int, depth: int = 1) -> None:
                super().__init__()

    assert len(caught) == 1
    assert caught[0].filename == __file__


# --------------------------------------------------------------------------- #
# Thread safety
# --------------------------------------------------------------------------- #


@pytest.fixture
def slow_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make each registration spend 10 ms inside the registry lock."""
    validate = registry._validate_constructor_signature

    def slow_validate(*args, **kwargs):
        time.sleep(0.01)
        return validate(*args, **kwargs)

    monkeypatch.setattr(registry, "_validate_constructor_signature", slow_validate)


@_TORCH_CRASHES_WITH_THREADS
@pytest.mark.usefixtures("slow_registration")
def test_concurrent_registration_of_distinct_names_does_not_crash() -> None:
    errors: list[Exception] = []

    def register_one(i: int) -> None:
        try:

            class Net(torch.nn.Module):
                def __init__(self, width: int) -> None:
                    super().__init__()

            registry.register_embedding_architecture(_fresh_name(f"thread_{i}"), Net)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=register_one, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for i in range(16):
        assert _fresh_name(f"thread_{i}") in registry.registered_architectures()


@_TORCH_CRASHES_WITH_THREADS
@pytest.mark.usefixtures("slow_registration")
def test_concurrent_registration_of_same_name_different_classes_exactly_one_wins() -> (
    None
):
    name = _fresh_name("thread_same_name_race")
    winners: list[type] = []
    failures: list[Exception] = []
    result_lock = threading.Lock()

    def register_one(i: int) -> None:
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)

        # Each thread registers a genuinely different class (a distinct __qualname__),
        # not a "redefinition" of the same source location.
        cls = type(f"RaceNet_{i}", (torch.nn.Module,), {"__init__": __init__})
        try:
            registry.register_embedding_architecture(name, cls)
        except ValueError as exc:
            with result_lock:
                failures.append(exc)
        else:
            with result_lock:
                winners.append(cls)

    threads = [threading.Thread(target=register_one, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1
    assert len(failures) == 15
    assert all("already registered" in str(exc) for exc in failures)
    assert registry._REGISTRY[name] is winners[0]


@_TORCH_CRASHES_WITH_THREADS
@pytest.mark.usefixtures("slow_registration")
def test_concurrent_registration_of_one_class_under_many_names_exactly_one_wins() -> (
    None
):
    class Net(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()

    names = [_fresh_name(f"thread_one_class_{i}") for i in range(16)]
    winners: list[str] = []
    failures: list[Exception] = []
    result_lock = threading.Lock()

    def register_one(name: str) -> None:
        try:
            registry.register_embedding_architecture(name, Net)
        except ValueError as exc:
            with result_lock:
                failures.append(exc)
        else:
            with result_lock:
                winners.append(name)

    threads = [threading.Thread(target=register_one, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1
    assert len(failures) == 15
    assert all("already registered as" in str(exc) for exc in failures)
    assert [n for n in names if n in registry.registered_architectures()] == winners
    assert registry.embedding_recipe(Net(3)) == {
        "builder": winners[0],
        "kwargs": {"width": 3},
    }


# --------------------------------------------------------------------------- #
# Inherited __init__
# --------------------------------------------------------------------------- #


def test_class_without_own_init_is_rejected_at_registration() -> None:
    class Net(torch.nn.Module):
        pass

    with pytest.raises(TypeError, match="own __init__"):
        registry.register_embedding_architecture(_fresh_name("noinit"), Net)


# --------------------------------------------------------------------------- #
# Validate recipes read from files
# --------------------------------------------------------------------------- #


def test_build_from_recipe_rejects_non_dict_recipe() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        registry.build_from_recipe(["not", "a", "dict"])


def test_build_from_recipe_rejects_non_string_builder() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        registry.build_from_recipe({"builder": 123, "kwargs": {}})


def test_build_from_recipe_rejects_non_string_kwargs_keys() -> None:
    with pytest.raises(ValueError, match="Malformed embedding recipe"):
        registry.build_from_recipe({"builder": "equine.mlp", "kwargs": {1: 6}})


def test_build_from_recipe_rejects_non_dict_kwargs() -> None:
    with pytest.raises(ValueError, match="'kwargs' must be a dict; got list"):
        registry.build_from_recipe({"builder": "equine.tests.basic", "kwargs": [1]})


def test_build_from_recipe_missing_kwargs_defaults_to_empty() -> None:
    @registry.embedding_architecture(_fresh_name("missingkwargs"))
    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

    built = registry.build_from_recipe({"builder": _fresh_name("missingkwargs")})
    assert isinstance(built, Net)


def test_build_from_recipe_wraps_constructor_type_error_as_value_error() -> None:
    name = _fresh_name("unknownkw")

    @registry.embedding_architecture(name)
    class Net(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()

    with pytest.raises(ValueError) as err:
        registry.build_from_recipe(
            {"builder": name, "kwargs": {"width": 3, "depth": 9}}
        )
    assert name in str(err.value)


def test_build_from_recipe_reports_type_name_not_full_value_for_non_dict_recipe() -> (
    None
):
    with pytest.raises(ValueError) as err:
        registry.build_from_recipe(list(range(10_000)))
    message = str(err.value)
    assert "list" in message
    assert "9999" not in message  # the raw untrusted content must not be embedded
    assert len(message) < 500


def test_build_from_recipe_reports_truncated_repr_for_non_string_builder() -> None:
    with pytest.raises(ValueError) as err:
        registry.build_from_recipe({"builder": list(range(10_000)), "kwargs": {}})
    message = str(err.value)
    assert "Malformed" in message
    assert "9999" not in message
    assert len(message) < 500


# --------------------------------------------------------------------------- #
# Shipped equine.mlp and public exports
# --------------------------------------------------------------------------- #


def test_shipped_mlp_is_registered_and_records_recipe() -> None:
    mlp = eq.MLP(6, [16, 8], 3)
    assert eq.embedding_recipe(mlp) == {
        "builder": "equine.mlp",
        "kwargs": {
            "in_features": 6,
            "hidden_sizes": [16, 8],
            "out_features": 3,
            "activation": "relu",
        },
    }
    assert "equine.mlp" in eq.registered_architectures()
    assert mlp(torch.rand(4, 6)).shape == (4, 3)


def test_shipped_mlp_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="activation"):
        eq.MLP(6, [16], 3, activation="swish")


def test_public_api_exports() -> None:
    for name in (
        "MLP",
        "embedding_architecture",
        "register_embedding_architecture",
        "registered_architectures",
        "embedding_recipe",
    ):
        assert name in eq.__all__, name
    for name in (
        "embedding_architecture",
        "register_embedding_architecture",
        "registered_architectures",
        "embedding_recipe",
    ):
        assert getattr(eq, name) is getattr(registry, name)
    assert eq.MLP is architectures.MLP


def test_mlp_state_dict_layout_is_pinned() -> None:
    mlp = eq.MLP(6, [16, 8], 3)
    assert list(mlp.state_dict()) == [
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
        "net.4.weight",
        "net.4.bias",
    ]
    state_dict = mlp.state_dict()
    assert tuple(state_dict["net.0.weight"].shape) == (16, 6)
    assert tuple(state_dict["net.0.bias"].shape) == (16,)
    assert tuple(state_dict["net.2.weight"].shape) == (8, 16)
    assert tuple(state_dict["net.2.bias"].shape) == (8,)
    assert tuple(state_dict["net.4.weight"].shape) == (3, 8)
    assert tuple(state_dict["net.4.bias"].shape) == (3,)


def test_mlp_recipe_round_trips_through_build_from_recipe() -> None:
    mlp = eq.MLP(6, [16, 8], 3, activation="gelu")
    rebuilt = registry.build_from_recipe(eq.embedding_recipe(mlp))
    rebuilt.load_state_dict(mlp.state_dict())
    batch = torch.rand(5, 6)
    assert torch.allclose(mlp(batch), rebuilt(batch))


def test_mlp_empty_hidden_sizes_gives_a_single_linear() -> None:
    mlp = eq.MLP(6, [], 3)
    assert list(mlp.state_dict()) == ["net.0.weight", "net.0.bias"]
    assert len(mlp.net) == 1
    assert isinstance(mlp.net[0], torch.nn.Linear)


@pytest.mark.parametrize(
    ("activation", "expected_type"),
    [
        ("relu", torch.nn.ReLU),
        ("gelu", torch.nn.GELU),
        ("tanh", torch.nn.Tanh),
        ("sigmoid", torch.nn.Sigmoid),
    ],
)
def test_mlp_activation_choices(
    activation: str, expected_type: type[torch.nn.Module]
) -> None:
    mlp = eq.MLP(6, [16], 3, activation=activation)
    assert type(mlp.net[1]) is expected_type


def test_mlp_accepts_tuple_hidden_sizes_and_records_a_tuple_recipe() -> None:
    mlp = eq.MLP(6, (16, 8), 3)
    recipe = eq.embedding_recipe(mlp)
    assert recipe["kwargs"]["hidden_sizes"] == (16, 8)
    assert isinstance(recipe["kwargs"]["hidden_sizes"], tuple)
    assert mlp(torch.rand(2, 6)).shape == (2, 3)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"in_features": -1, "hidden_sizes": [16], "out_features": 3},
            "in_features must be a positive integer",
        ),
        (
            {"in_features": 0, "hidden_sizes": [16], "out_features": 3},
            "in_features must be a positive integer",
        ),
        (
            {"in_features": True, "hidden_sizes": [16], "out_features": 3},
            "in_features must be a positive integer",
        ),
        (
            {"in_features": 6, "hidden_sizes": [-1], "out_features": 3},
            r"hidden_sizes\[0\] must be a positive integer",
        ),
        (
            {"in_features": 6, "hidden_sizes": [0], "out_features": 3},
            r"hidden_sizes\[0\] must be a positive integer",
        ),
        (
            {"in_features": 6, "hidden_sizes": [True], "out_features": 3},
            r"hidden_sizes\[0\] must be a positive integer",
        ),
        (
            {"in_features": 6, "hidden_sizes": "16", "out_features": 3},
            "hidden_sizes must be a list or tuple",
        ),
        (
            {"in_features": 6, "hidden_sizes": [16], "out_features": -1},
            "out_features must be a positive integer",
        ),
        (
            {"in_features": 6, "hidden_sizes": [16], "out_features": 0},
            "out_features must be a positive integer",
        ),
        (
            {
                "in_features": 6,
                "hidden_sizes": [16],
                "out_features": 3,
                "activation": ["relu"],
            },
            "activation must be a str",
        ),
    ],
    ids=[
        "negative-in_features",
        "zero-in_features",
        "bool-in_features",
        "negative-hidden_sizes-entry",
        "zero-hidden_sizes-entry",
        "bool-hidden_sizes-entry",
        "str-hidden_sizes",
        "negative-out_features",
        "zero-out_features",
        "list-activation",
    ],
)
def test_build_from_recipe_rejects_invalid_mlp_kwargs(
    kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        registry.build_from_recipe({"builder": "equine.mlp", "kwargs": kwargs})


def test_build_from_recipe_rejects_too_many_hidden_layers() -> None:
    with pytest.raises(ValueError, match="MAX_HIDDEN_LAYERS"):
        registry.build_from_recipe(
            {
                "builder": "equine.mlp",
                "kwargs": {
                    "in_features": 1,
                    "hidden_sizes": [1] * 65,
                    "out_features": 1,
                },
            }
        )


def test_build_from_recipe_rejects_parameter_count_over_cap(monkeypatch) -> None:
    built: list[tuple] = []
    real_linear = torch.nn.Linear

    def recording_linear(*args, **kwargs):
        built.append(args)
        return real_linear(*args, **kwargs)

    monkeypatch.setattr(torch.nn, "Linear", recording_linear)
    # The meta device allocates nothing, whatever size the recipe asks for.
    with torch.device("meta"), pytest.raises(ValueError, match="MAX_PARAMETERS"):
        registry.build_from_recipe(
            {
                "builder": "equine.mlp",
                "kwargs": {
                    "in_features": 100_000,
                    "hidden_sizes": [100_000],
                    "out_features": 1,
                },
            }
        )
    assert built == []  # refused before any layer was built


def test_build_from_recipe_wraps_arbitrary_constructor_exception_as_value_error() -> (
    None
):
    name = _fresh_name("runtimeerror")

    @registry.embedding_architecture(name)
    class Net(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            raise RuntimeError("boom")

    with pytest.raises(ValueError) as err:
        registry.build_from_recipe({"builder": name, "kwargs": {"width": 3}})
    assert name in str(err.value)


# --------------------------------------------------------------------------- #
# conftest fixture registration
# --------------------------------------------------------------------------- #


def test_basic_embedding_model_fixture_is_registered() -> None:
    assert eq.embedding_recipe(BasicEmbeddingModel(6, 3)) == {
        "builder": "equine.tests.basic",
        "kwargs": {"tensor_dim": 6, "num_classes": 3},
    }


# --------------------------------------------------------------------------- #
# Error-message size is bounded
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "in_features": 6,
            "hidden_sizes": [16],
            "out_features": 3,
            "activation": "x" * 100_000,
        },
        {
            "in_features": 6,
            "hidden_sizes": "x" * 100_000,
            "out_features": 3,
        },
        {
            "in_features": 6,
            "hidden_sizes": [16],
            "out_features": 3,
            "activation": ["x"] * 100_000,
        },
        {
            "in_features": 6,
            "hidden_sizes": [16, "x" * 100_000],
            "out_features": 3,
        },
    ],
    ids=[
        "unknown-activation",
        "hidden-sizes-not-a-list-or-tuple",
        "non-string-activation",
        "non-int-hidden-sizes-element",
    ],
)
def test_build_from_recipe_bounds_error_message_for_oversized_mlp_value(
    kwargs: dict,
) -> None:
    with pytest.raises(ValueError, match=re.escape("'equine.mlp'")) as err:
        registry.build_from_recipe({"builder": "equine.mlp", "kwargs": kwargs})
    assert len(str(err.value)) < 1_000


def test_build_from_recipe_bounds_error_message_for_huge_unknown_builder_name() -> None:
    huge_name = "x" * 100_000
    with pytest.raises(ValueError) as err:
        registry.build_from_recipe({"builder": huge_name, "kwargs": {}})
    message = str(err.value)
    assert len(message) < 1_000
    assert "not registered" in message
    assert "x" in message  # a truncated builder reference is still present


def test_build_from_recipe_bounds_error_message_for_huge_unexpected_keyword() -> None:
    name = _fresh_name("hugekw")

    @registry.embedding_architecture(name)
    class Net(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()

    with pytest.raises(ValueError) as err:
        registry.build_from_recipe(
            {"builder": name, "kwargs": {"width": 3, "x" * 100_000: 1}}
        )
    message = str(err.value)
    assert len(message) < 1_000
    assert name in message


def test_describe_error_names_the_type_and_any_message() -> None:
    assert registry._describe_error(RuntimeError()) == "RuntimeError"
    assert registry._describe_error(ValueError("bad width")) == "ValueError: bad width"


def test_a_constructor_error_without_a_message_is_named_by_its_type() -> None:
    @registry.embedding_architecture("equine.tests.asserting")
    class Asserting(torch.nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            if n <= 0:
                raise AssertionError()

    with pytest.raises(ValueError, match="recorded recipe: AssertionError$"):
        registry.build_from_recipe(
            {"builder": "equine.tests.asserting", "kwargs": {"n": 0}}
        )
