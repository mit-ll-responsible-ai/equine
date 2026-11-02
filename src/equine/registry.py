# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
Registry of embedding-model architectures.

An EQUINE model file describes its embedding model as a *recipe*: the name of a
registered architecture plus the plain-valued constructor arguments it was built with.
``load()`` rebuilds the module by calling only a registered constructor. A model file
never names an import path, and carries executable content only when the saver
explicitly opts in.

Recipes are recorded per exact class (a subclass must register itself to get its own
recipe) and are deep-copied both when recorded and when handed back. Neither the
constructor nor the caller can mutate a module's stored recipe. Registration is
thread-safe.
"""

from __future__ import annotations

import copy
import functools
import inspect
import reprlib
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Optional, TypeVar

import torch

__all__ = [
    "embedding_architecture",
    "embedding_recipe",
    "register_embedding_architecture",
    "registered_architectures",
]

_M = TypeVar("_M", bound=torch.nn.Module)

_REGISTRY: dict[str, type[torch.nn.Module]] = {}
_REGISTRY_LOCK = threading.Lock()
_RECIPE_ATTR = "_equine_recipe"
_REGISTERED_AS_ATTR = "_equine_registered_as"

_PLAIN_SCALAR_TYPES = (int, float, str, bool, type(None))
_PLAIN_CONTAINER_TYPES = (list, tuple)
_PLAIN_KEY_TYPES = (str, int)
# Deepest nesting of lists, tuples and dicts accepted in a plain value.
_MAX_PLAIN_DEPTH = 100

# Shows real builder names in full and cuts a long name from a file short.
_ERROR_REPR = reprlib.Repr()
_ERROR_REPR.maxstring = 200
_ERROR_REPR.maxother = 200


def _message_or_type(err: BaseException) -> str:
    """``err``'s message, or its type name when it has no message."""
    return str(err) or type(err).__name__


def _describe_error(err: BaseException) -> str:
    """``err`` as ``TypeName: message``, or as ``TypeName`` when it has no message."""
    message = str(err)
    return f"{type(err).__name__}: {message}" if message else type(err).__name__


def _truncate(text: str, limit: int = 500) -> str:
    """Bound ``text`` to at most ``limit`` characters plus a truncation marker."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(truncated)"


def _positive_int(value: object, name: str) -> int:
    """
    Return ``value`` if it is exactly an ``int`` (``bool`` is refused) >= 1.

    Otherwise raise a ValueError naming ``name``. The value is shown only when it is a
    small int.
    """
    if type(value) is int and value >= 1:
        return value
    shown = f" {value}" if type(value) is int and value > -(10**6) else ""
    raise ValueError(
        f"{name} must be a positive integer (an int >= 1, not a bool); got "
        f"{type(value).__name__}{shown}."
    )


def _shown_key(key: Any) -> str:
    """A bounded repr of a dict key or sequence index for an error message."""
    try:
        return _ERROR_REPR.repr(key)
    except ValueError:  # an int with too many digits to convert to str
        return f"<{type(key).__name__}>"


def _check_plain(
    values: Mapping[str, Any],
    *,
    allow_tensors: bool = False,
    what: str = "Constructor argument",
    owner: str = "a registered embedding architecture",
    key_types: tuple[type, ...] = _PLAIN_KEY_TYPES,
) -> None:
    """
    Require each of ``values`` to be built only from exact plain types, with
    lists, tuples and dicts nested at most ``_MAX_PLAIN_DEPTH`` levels deep.

    With ``allow_tensors`` (for a module's extra state, which lives in its
    state_dict), tensors that are not quantized and carry no Python attributes,
    ``torch.Size`` and ``OrderedDict`` are accepted too: the safe loader reads all
    three. ``what`` and ``owner`` name the kind of value and what it belongs to in the
    error message, which locates an offending value starting from its key in
    ``values``. Dict keys must be exactly one of ``key_types``.

    Values read from a model file can share one container between many places, within
    one value or across several, or contain themselves. The walk is iterative and checks
    each distinct container once for all of ``values``. The depth is the longest chain
    of nested containers, whichever way a shared container is reached, and a container
    that contains itself is refused. The location of an offending value is formatted
    only for the error message, and is bounded.
    """
    # The key in ``values`` being checked, then the key or index leading to
    # each container on the current chain, then to the item being checked.
    keys: list[Any] = []
    key_names = ["None" if kind is type(None) else kind.__name__ for kind in key_types]

    def where() -> str:
        path = keys[0] + "".join(f"[{_shown_key(key)}]" for key in keys[1:])
        return _truncate(path, 200)

    def items_of(item: Any) -> Optional[Iterator[tuple[Any, Any]]]:
        """The ``(key, item)`` pairs of a plain container; None for a plain leaf."""
        item_type = type(item)
        if item_type in _PLAIN_SCALAR_TYPES:
            return None
        if allow_tensors and item_type in (
            torch.Tensor,
            torch.nn.Parameter,
            torch.Size,
        ):
            if item_type is torch.Size or not (item.is_quantized or vars(item)):
                return None
        if item_type in _PLAIN_CONTAINER_TYPES:
            return enumerate(item)
        if item_type is dict or (allow_tensors and item_type is OrderedDict):
            return iter(item.items())
        allowed = (
            "int, float, str, bool, None, list, tuple, dict or OrderedDict (with "
            f"{'/'.join(key_names)} keys), torch.Tensor, or torch.Size"
            if allow_tensors
            else "int, float, str, bool, None, list, tuple, or dict (with "
            f"{'/'.join(key_names)} keys)"
        )
        if item_type in (set, frozenset):
            hint = "use a sorted list, e.g. sorted(x)"
        elif isinstance(item, torch.Tensor) and item.is_quantized:
            hint = "a quantized tensor; store x.dequantize()"
        elif isinstance(item, torch.Tensor):
            hint = "a tensor with Python attributes; store x.detach().clone()"
        else:
            hint = "e.g. convert with float(x) or int(x)"
        raise TypeError(
            f"{what} {where()!r} of {owner} must be a "
            f"plain value of exactly one of these types: {allowed}, or a nested "
            "combination of those, so it can be stored in a model file; got "
            f"{item_type.__name__} ({hint})."
        )

    def too_deep() -> TypeError:
        return TypeError(
            f"{what} {where()!r} of {owner} nests "
            f"lists, tuples and dicts more than {_MAX_PLAIN_DEPTH} levels deep; "
            f"plain values may be nested at most {_MAX_PLAIN_DEPTH} levels deep."
        )

    # For each container checked so far, how many levels of containers it
    # holds, itself included; 0 while it is still on the chain being walked.
    heights: dict[int, int] = {}
    for path, value in values.items():
        if id(value) in heights:  # a container checked under an earlier key
            continue
        keys.append(path)
        root_items = items_of(value)
        if root_items is None:
            keys.pop()
            continue
        heights[id(value)] = 0
        # The containers from ``value`` down to the one being walked: its id,
        # its remaining items, and whether it is a dict.
        chain = [(id(value), root_items, type(value) not in _PLAIN_CONTAINER_TYPES)]
        # The deepest nesting found so far in each container on the chain.
        deepest = [1]
        while chain:
            ident, items, is_dict = chain[-1]
            entry = next(items, None)
            if entry is None:
                chain.pop()
                heights[ident] = deepest.pop()
                keys.pop()
                if chain:
                    deepest[-1] = max(deepest[-1], heights[ident] + 1)
                continue
            key, item = entry
            if is_dict and type(key) not in key_types:
                raise TypeError(
                    f"{what} {where()!r} has a dict key {_shown_key(key)} of type "
                    f"{type(key).__name__}; dict keys must be exactly one of "
                    f"{', '.join(key_names[:-1])} or {key_names[-1]} (e.g. convert "
                    "with str(key))."
                )
            keys.append(key)
            height = heights.get(id(item))
            if height is None:
                nested = items_of(item)
                if nested is None:
                    keys.pop()
                    continue
                if len(chain) >= _MAX_PLAIN_DEPTH:
                    raise too_deep()
                heights[id(item)] = 0
                chain.append(
                    (id(item), nested, type(item) not in _PLAIN_CONTAINER_TYPES)
                )
                deepest.append(1)
            elif height == 0:
                raise TypeError(
                    f"{what} {where()!r} of {owner} contains itself; plain values "
                    "must not refer back to a list, tuple or dict that encloses them."
                )
            elif len(chain) + height > _MAX_PLAIN_DEPTH:
                raise too_deep()
            else:
                keys.pop()
                deepest[-1] = max(deepest[-1], height + 1)


def _validate_constructor_signature(
    cls: type[torch.nn.Module], signature: inspect.Signature
) -> str:
    """Ensure ``signature`` can be recorded as a recipe; return the first parameter's name."""
    if cls.__init__ is torch.nn.Module.__init__:
        raise TypeError(
            f"{cls.__qualname__} does not define its own __init__ (it inherits "
            "torch.nn.Module.__init__ unchanged); a registered architecture must "
            "define its own __init__ so its constructor arguments can be recorded "
            "in a recipe."
        )
    param_names = list(signature.parameters)
    first_param_name = param_names[0]
    first_kind = signature.parameters[first_param_name].kind
    if first_kind in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise TypeError(
            f"{cls.__qualname__}.__init__'s first parameter {first_param_name!r} is "
            "*args, **kwargs, or keyword-only, so its instance argument would be "
            "silently absorbed; registered architectures need a plain first "
            "parameter for the instance so later constructor arguments can be "
            "recorded in a recipe."
        )
    for pname in param_names[1:]:
        param = signature.parameters[pname]
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(
                f"{cls.__qualname__}.__init__ accepts *args; registered architectures "
                "need named constructor arguments so they can be recorded in a recipe."
            )
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{cls.__qualname__}.__init__ has positional-only parameter {pname!r}; "
                "registered architectures need named (keyword-capable) constructor "
                "arguments so they can be recorded in a recipe."
            )
    return first_param_name


def _require_name(name: Any) -> None:
    """Refuse an architecture name that is not a non-empty ``str``."""
    if type(name) is not str:
        raise TypeError(
            "An embedding architecture name must be a str (it is stored in model "
            f"files); got {type(name).__name__}."
        )
    if not name:
        raise ValueError("An embedding architecture name must not be empty.")


def _register(name: str, cls: type[_M], stacklevel: int) -> None:
    """
    Shared implementation for ``register_embedding_architecture`` and the
    ``embedding_architecture`` decorator.

    ``stacklevel`` is set per entry point and attributes a re-registration
    ``UserWarning`` to the caller's source line.
    """
    with _REGISTRY_LOCK:
        already_as = cls.__dict__.get(_REGISTERED_AS_ATTR)
        if already_as is not None and already_as != name:
            raise ValueError(
                f"{cls.__module__}.{cls.__qualname__} is already registered as "
                f"{already_as!r}; register each class under exactly one name "
                f"(cannot also register it as {name!r})."
            )

        existing = _REGISTRY.get(name)
        if existing is cls:
            return  # Already fully registered under this exact name; no-op.

        is_redefinition = False
        if existing is not None:
            same_definition = (
                existing.__module__ == cls.__module__
                and existing.__qualname__ == cls.__qualname__
            )
            if not same_definition:
                raise ValueError(
                    f"Embedding architecture name {name!r} is already registered to "
                    f"{existing.__module__}.{existing.__qualname__}; cannot register "
                    f"{cls.__module__}.{cls.__qualname__} under it."
                )
            is_redefinition = True

        original_init: Callable[..., None] = cls.__init__
        signature = inspect.signature(original_init)
        first_param_name = _validate_constructor_signature(cls, signature)

        if is_redefinition:
            warnings.warn(
                f"Embedding architecture {name!r} was re-registered: "
                f"{cls.__module__}.{cls.__qualname__} replaces the previous "
                "definition registered under that name (expected after a reload "
                "or re-running a notebook cell).",
                UserWarning,
                stacklevel=stacklevel,
            )

        @functools.wraps(original_init)
        def recording_init(self: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
            if type(self) is not cls:
                original_init(self, *args, **kwargs)
                return
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            recipe_kwargs: dict[str, Any] = {}
            for key, value in bound.arguments.items():
                if key == first_param_name:
                    continue
                if signature.parameters[key].kind is inspect.Parameter.VAR_KEYWORD:
                    recipe_kwargs.update(value)
                else:
                    recipe_kwargs[key] = value
            _check_plain(recipe_kwargs)
            snapshot = copy.deepcopy(recipe_kwargs)
            original_init(self, *args, **kwargs)
            setattr(self, _RECIPE_ATTR, {"builder": name, "kwargs": snapshot})

        setattr(cls, "__init__", recording_init)
        setattr(cls, _REGISTERED_AS_ATTR, name)
        _REGISTRY[name] = cls


def register_embedding_architecture(
    name: str, cls: type[_M], *, _stacklevel: int = 3
) -> None:
    """
    Register ``cls`` under ``name`` and make its instances carry a recipe.

    Every instance of exactly ``cls`` (not an unregistered subclass) records the
    arguments it was constructed with (defaults applied, ``**kwargs`` flattened in),
    through a wrapper around the class's ``__init__``. Those
    arguments must be plain values, with lists, tuples and dicts nested at
    most 100 levels deep. Registering the exact same class object
    under the same name again is a no-op. Re-registering under the same name
    a *different* class object with the same ``__module__`` and
    ``__qualname__`` (as happens after ``importlib.reload`` or re-running a
    notebook cell that redefines the class) replaces the registration and
    emits a ``UserWarning``; a genuinely different class raises
    ``ValueError``. Registering one class object under two different names
    also raises ``ValueError``. Concurrent registrations resolve deterministically: the
    whole check-then-register sequence, including the two-names check, is serialized on
    a module-level lock.

    Parameters
    ----------
    name : str
        Stable identifier stored in model files. Namespace it by project
        (``"myproject.encoder"``). Never an import path.
    cls : type[_M]
        The architecture class. It must not inherit ``torch.nn.Module.__init__``
        unchanged (an ``__init__`` inherited from a user-defined base class is
        recorded with that base's signature). Its ``__init__`` needs a plain
        first (instance) parameter and named, non-variadic-positional,
        non-positional-only constructor arguments after it.

    Raises
    ------
    TypeError
        If ``name`` is not a ``str``, ``cls`` inherits
        ``torch.nn.Module.__init__`` unchanged, or its ``__init__`` has an
        unusable first parameter (absorbed into ``*args``/``**kwargs`` or
        keyword-only) or a later ``*args``/positional-only parameter.
    ValueError
        If ``name`` is already registered to an unrelated class, or ``cls``
        is already registered under a different name, or ``name`` is empty.
    """
    _require_name(name)
    _register(name, cls, stacklevel=_stacklevel)


def embedding_architecture(name: str) -> Callable[[type[_M]], type[_M]]:
    """
    Class-decorator form of ``register_embedding_architecture``.

    Trust boundary: a saved model file replays this registration by calling the named
    constructor with whatever plain-valued ``kwargs`` it recorded. Any registered
    constructor must be safe to call with arbitrary plain input from an untrusted file —
    it must not open a path, unpickle a blob, or fetch a URL from a constructor
    argument, and should treat any size-like argument as attacker-controlled. The same
    holds for ``forward``: the size of its output must be determined by the module's
    weights or bounded in the constructor, not set by a constructor argument alone (for
    example a ``repeat`` count, or a one-hot encoding over ``k`` classes).

    Constructor requirements: the loader first builds the architecture on the meta
    device (no memory) and checks its parameter names and shapes against the weights in
    the file, and only then builds it for real. The constructor must be
    buildable on the meta device: allocate tensors only through standard
    factory functions and layers (``torch.zeros``, ``torch.nn.Linear``, ...)
    without an explicit ``device=`` (the legacy ``torch.Tensor(n, m)``
    constructor ignores the meta device), and do not read tensor values
    (``.item()``) or move the module (``.to(...)``) while constructing. The
    check calls ``state_dict()`` on that meta-device module, so a
    ``get_extra_state`` it defines must not read tensor values either. Every tensor the
    module owns must also be in its ``state_dict``; a non-persistent buffer must not be
    sized from a constructor argument.

    Every architectural choice must be a constructor argument. ``save()`` rebuilds the
    recipe the same way and refuses a module it cannot rebuild, comparing the rebuild's
    submodules (their names, classes and, for torch's own layers, hyper-parameters such
    as an activation's options) and weights with the live module; a submodule replaced
    or reconfigured after construction is refused.

    If a registered class is also decorated with ``@beartype``, put
    ``@embedding_architecture`` ABOVE ``@beartype`` (closer to the class statement runs
    first).

    Parameters
    ----------
    name : str
        Stable identifier stored in model files, passed through to
        ``register_embedding_architecture``.

    Returns
    -------
    Callable[[type[_M]], type[_M]]
        A decorator that registers and returns the class unchanged.

    Raises
    ------
    TypeError
        If ``name`` is not a ``str``. Applying the decorator raises the
        errors of ``register_embedding_architecture``.
    ValueError
        If ``name`` is empty.

    Examples
    --------
    >>> @equine.embedding_architecture("myproject.encoder")
    ... class Encoder(torch.nn.Module):
    ...     def __init__(self, in_features: int, out_features: int) -> None: ...

    >>> @equine.embedding_architecture("myproject.checked_encoder")
    ... @beartype
    ... class CheckedEncoder(torch.nn.Module):
    ...     def __init__(self, in_features: int, out_features: int) -> None: ...
    """

    _require_name(name)

    def decorator(cls: type[_M]) -> type[_M]:
        register_embedding_architecture(name, cls, _stacklevel=4)
        return cls

    return decorator


def registered_architectures() -> list[str]:
    """
    List the embedding architecture names registered in this process.

    Returns
    -------
    list[str]
        Registered names, sorted.
    """
    with _REGISTRY_LOCK:
        return sorted(_REGISTRY)


def embedding_recipe(module: torch.nn.Module) -> Optional[dict[str, Any]]:
    """
    Return the recipe recorded on ``module``, if any.

    Parameters
    ----------
    module : torch.nn.Module
        A module instance, possibly built by a registered architecture.

    Returns
    -------
    dict[str, Any] or None
        A deep copy of the recorded ``{"builder": str, "kwargs": dict}``
        recipe, safe for the caller to mutate; ``None`` if ``module`` was not
        built by a registered architecture.
    """
    recipe = getattr(module, _RECIPE_ATTR, None)
    if recipe is None:
        return None
    return copy.deepcopy(recipe)


def build_from_recipe(recipe: dict[str, Any]) -> torch.nn.Module:
    """
    Instantiate a registered architecture from a recipe read out of a model file.

    Parameters
    ----------
    recipe : dict[str, Any]
        A ``{"builder": str, "kwargs": dict}`` mapping, typically produced by
        ``embedding_recipe`` and round-tripped through a model file.
        ``"kwargs"`` may be omitted, defaulting to ``{}``.

    Returns
    -------
    torch.nn.Module
        A freshly constructed, untrained instance of the registered architecture.

    Raises
    ------
    ValueError
        If ``recipe`` is malformed, names an architecture not registered in
        this process, holds ``kwargs`` that are not plain values, or its
        constructor rejects them (for example because the class's signature
        changed since the file was saved).
    """
    if type(recipe) is not dict:
        raise ValueError(
            "Malformed embedding recipe: expected a dict with a string 'builder' "
            f"key; got a {type(recipe).__name__}."
        )
    builder = recipe.get("builder")
    if type(builder) is not str:
        raise ValueError(
            "Malformed embedding recipe: expected a dict with a string 'builder' "
            f"key; got 'builder' = {reprlib.repr(builder)}."
        )
    kwargs = recipe.get("kwargs", {})
    if type(kwargs) is not dict:
        raise ValueError(
            f"Malformed embedding recipe: 'kwargs' must be a dict; got "
            f"{type(kwargs).__name__}."
        )
    if any(type(key) is not str for key in kwargs):
        raise ValueError(
            "Malformed embedding recipe: the keys of 'kwargs' must be strings."
        )
    name = builder
    cls = _REGISTRY.get(name)
    if cls is None:
        known = ", ".join(registered_architectures()) or "(none)"
        raise ValueError(
            f"This model file was saved with embedding architecture "
            f"{_ERROR_REPR.repr(name)}, which is not registered in this process. "
            f"Registered: {known}. Import the module that registers it "
            f"(@equine.embedding_architecture({_ERROR_REPR.repr(name)})), or pass the "
            f"architecture yourself with load(..., embedding_model=YourModule(...))."
        )
    try:
        # Checked before the copy, which recurses through every nested value.
        _check_plain(kwargs)
        # A copy, so a constructor that mutates its arguments cannot change the recipe.
        return cls(**copy.deepcopy(kwargs))
    except Exception as error:
        raise ValueError(
            f"Could not build embedding architecture {_ERROR_REPR.repr(name)} from its "
            f"recorded recipe: {_truncate(_message_or_type(error))}"
        ) from error
