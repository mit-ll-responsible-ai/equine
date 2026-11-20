# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""The support API of both model classes.

``Equine`` declares ``get_support`` and ``update_support`` as abstract methods.
``update_support`` takes its options by keyword (``calib_frac`` and ``label_names`` on
``EquineProtonet``, ``support_size`` on ``EquineGP``); passing one by position raises
a ``TypeError`` that names the keywords and leaves the model as it was.
``EquineProtonet.support``, ``support_embeddings`` and ``prototypes`` are read-only
views of the inner model's support.
"""

import math
import pickle
import re
import warnings
from collections import OrderedDict

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation
from builders import quick_gp, quick_protonet
from conftest import BasicEmbeddingModel, assert_model_unchanged, model_state
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq

NAMES = ["a", "b", "c"]


def _protonet():
    return eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)


def _gp():
    return eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )


def test_equine_declares_the_support_methods_abstract() -> None:
    assert {"get_support", "update_support"} <= eq.Equine.__abstractmethods__


@pytest.mark.parametrize(
    "build, keywords",
    [
        (quick_protonet, {"calib_frac": 0.5}),
        (quick_protonet, {"calib_frac": 0.5, "label_names": NAMES}),
        (quick_gp, {"support_size": 10}),
    ],
    ids=["protonet", "protonet-label-names", "gp"],
)
def test_keyword_options_do_not_warn(build, keywords) -> None:
    model, x, y = build()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.update_support(x, y, **keywords)


_PROTONET_BY_POSITION = (
    "EquineProtonet.update_support() takes calib_frac and label_names by keyword "
    "only, but {} option(s) were given by position: call update_support(support_x, "
    "support_y, calib_frac=...), adding label_names=... by keyword."
)
_GP_BY_POSITION = (
    "EquineGP.update_support() takes support_size by keyword only, but {} option(s) "
    "were given by position: call update_support(support_x, support_y, "
    "support_size=...)."
)


def _trained_gp_with_support():
    return quick_gp(vis_support=True)


@pytest.mark.parametrize(
    "build, call, repair, message",
    [
        (
            quick_protonet,
            lambda m, x, y: m.update_support(x, y, 0.2),
            lambda m, x, y: m.update_support(x, y, calib_frac=0.2),
            _PROTONET_BY_POSITION.format(1),
        ),
        (
            quick_protonet,
            lambda m, x, y: m.update_support(x, y, 0.2, NAMES),
            lambda m, x, y: m.update_support(x, y, calib_frac=0.2, label_names=NAMES),
            _PROTONET_BY_POSITION.format(2),
        ),
        (
            quick_protonet,
            lambda m, x, y: m.update_support(x, y, 0.2, label_names=NAMES),
            lambda m, x, y: m.update_support(x, y, calib_frac=0.2, label_names=NAMES),
            _PROTONET_BY_POSITION.format(1),
        ),
        (
            quick_protonet,
            lambda m, x, y: m.update_support(x, y, 0.2, calib_frac=0.2),
            lambda m, x, y: m.update_support(x, y, calib_frac=0.2),
            _PROTONET_BY_POSITION.format(1),
        ),
        (
            quick_protonet,
            lambda m, x, y: m.update_support(x, y, 0.2, calib_x=x[::4], calib_y=y[::4]),
            lambda m, x, y: m.update_support(
                x, y, calib_frac=0.2, calib_x=x[::4], calib_y=y[::4]
            ),
            _PROTONET_BY_POSITION.format(1),
        ),
        (
            quick_protonet,
            lambda m, x, y: m.update_support(x, y, 0.2, calib_x=x[::4]),
            lambda m, x, y: m.update_support(
                x, y, calib_frac=0.2, calib_x=x[::4], calib_y=y[::4]
            ),
            _PROTONET_BY_POSITION.format(1),
        ),
        (
            _trained_gp_with_support,
            lambda m, x, y: m.update_support(x, y, 10),
            lambda m, x, y: m.update_support(x, y, support_size=10),
            _GP_BY_POSITION.format(1),
        ),
        (
            _trained_gp_with_support,
            lambda m, x, y: m.update_support(x, y, 10, 10),
            lambda m, x, y: m.update_support(x, y, support_size=10),
            _GP_BY_POSITION.format(2),
        ),
        (
            _trained_gp_with_support,
            lambda m, x, y: m.update_support(x, y, 10, support_size=10),
            lambda m, x, y: m.update_support(x, y, support_size=10),
            _GP_BY_POSITION.format(1),
        ),
    ],
    ids=[
        "protonet (x, y, f)",
        "protonet (x, y, f, names)",
        "protonet (x, y, f, label_names=names)",
        "protonet (x, y, f, calib_frac=f)",
        "protonet (x, y, f, calib_x=, calib_y=)",
        "protonet (x, y, f, calib_x=)",
        "gp (x, y, n)",
        "gp (x, y, n, n)",
        "gp (x, y, n, support_size=n)",
    ],
)
def test_options_passed_by_position_are_refused_naming_the_keywords(
    build, call, repair, message
) -> None:
    """The refusal names the keywords and leaves the support, its KDEs (Protonet) or
    embeddings (GP) and the state dict as they were; the keyword call then updates
    the support."""
    model, x, y = build()
    kept = (
        "outlier_score_kde"
        if isinstance(model, eq.EquineProtonet)
        else "support_embeddings"
    )
    before = getattr(model, kept)
    state = {name: t.clone() for name, t in model.state_dict().items()}
    support = {label: rows.clone() for label, rows in model.get_support().items()}
    with pytest.raises(TypeError) as refused:
        call(model, x, y)
    assert str(refused.value) == message
    assert getattr(model, kept) is before
    assert list(model.state_dict()) == list(state)
    for name, t in state.items():
        assert torch.equal(model.state_dict()[name], t)
    assert list(model.get_support()) == list(support)
    for label, rows in support.items():
        assert torch.equal(model.get_support()[label], rows)
    assert model.label_names is None

    repair(model, x, y)
    assert getattr(model, kept) is not before
    assert list(model.get_support()) == list(range(CLASSES))


def _calibration_set(x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"calib_x": x[::4].clone(), "calib_y": y[::4].clone()}


def _rows_and_labels_message(rows: int, labels: int) -> str:
    return (
        f"support_x has {rows} row(s) and support_y {labels} label(s); give one "
        "label per row."
    )


def _calib_frac_message(value: object) -> str:
    return f"calib_frac must be a number strictly between 0 and 1; got {value!r}."


def _longer(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x[:10] + 100.0])


@pytest.mark.parametrize(
    ("changes", "message", "repair"),
    [
        *[
            pytest.param(
                lambda x, y, value=value: {"calib_frac": value},
                _calib_frac_message(value),
                lambda x, y: {"calib_frac": 0.5},
                id=f"calib_frac={value}",
            )
            for value in (1.5, 1.0, 0.0, -0.25, math.nan)
        ],
        pytest.param(
            lambda x, y: {"calib_frac": 1.5, **_calibration_set(x, y)},
            _calib_frac_message(1.5),
            lambda x, y: {"calib_frac": None},
            id="calib_frac=1.5-with-calib_x",
        ),
        pytest.param(
            lambda x, y: {"support_x": _longer(x)},
            _rows_and_labels_message(130, 120),
            lambda x, y: {"support_x": x},
            id="support_x-longer-calib_frac",
        ),
        pytest.param(
            lambda x, y: {"support_x": x[:110]},
            _rows_and_labels_message(110, 120),
            lambda x, y: {"support_x": x},
            id="support_x-shorter-calib_frac",
        ),
        pytest.param(
            lambda x, y: {
                "support_x": _longer(x),
                "calib_frac": None,
                **_calibration_set(x, y),
            },
            _rows_and_labels_message(130, 120),
            lambda x, y: {"support_x": x},
            id="support_x-longer-calib_x",
        ),
        pytest.param(
            lambda x, y: {
                "support_x": x[:110],
                "calib_frac": None,
                **_calibration_set(x, y),
            },
            _rows_and_labels_message(110, 120),
            lambda x, y: {"support_x": x},
            id="support_x-shorter-calib_x",
        ),
        pytest.param(
            lambda x, y: {"support_x": torch.tensor(0.0)},
            "support_x must have a row dimension; got shape (); pass a tensor with "
            "one entry per row along its first dimension.",
            lambda x, y: {"support_x": x},
            id="0-d-support_x",
        ),
    ],
)
def test_protonet_update_support_refuses_bad_input(changes, message, repair) -> None:
    """Each refusal names the argument, draws no random numbers and leaves the
    model as it was; the call its message asks for replaces the support."""
    model, x, y = quick_protonet()
    args = {"support_x": x, "support_y": y, "calib_frac": 0.5, **changes(x, y)}
    before = model_state(model)
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match=re.escape(message)):
        model.update_support(**args)
    assert_model_unchanged(model, before)
    assert torch.equal(torch.get_rng_state(), rng), "the refusal drew random numbers"

    args.update(repair(x, y))
    model.update_support(**args)
    assert list(model.get_support()) == list(range(CLASSES))
    assert model.outlier_score_kde[0] is not before["outlier_score_kde"][0][1]


@pytest.mark.parametrize(
    ("calib_frac", "error", "message"),
    [
        *[
            pytest.param(
                value,
                BeartypeCallHintParamViolation,
                None,
                marks=pytest.mark.contracts,
                id=f"int-{value}-contracts-on",
            )
            for value in (0, 1)
        ],
        *[
            pytest.param(
                value,
                ValueError,
                _calib_frac_message(value),
                marks=pytest.mark.skipif(
                    __debug__, reason="beartype refuses an int calib_frac first"
                ),
                id=f"int-{value}-python-O",
            )
            for value in (0, 1)
        ],
    ],
)
def test_protonet_update_support_refuses_an_int_calib_frac(
    calib_frac, error, message
) -> None:
    """An int ``calib_frac`` is a type error while contracts are on and a
    ``ValueError`` under python -O; the model is left as it was, and the float the
    message asks for replaces the support."""
    model, x, y = quick_protonet()
    before = model_state(model)
    rng = torch.get_rng_state()
    with pytest.raises(error, match=None if message is None else re.escape(message)):
        model.update_support(x, y, calib_frac=calib_frac)
    assert_model_unchanged(model, before)
    assert torch.equal(torch.get_rng_state(), rng), "the refusal drew random numbers"

    model.update_support(x, y, calib_frac=0.5)
    assert list(model.get_support()) == list(range(CLASSES))


def _support_size_message(shown: str) -> str:
    return (
        "support_size must be a positive integer (an int >= 1, not a bool); got "
        f"{shown}."
    )


@pytest.mark.parametrize(
    ("changes", "message", "repair", "rows"),
    [
        pytest.param(
            lambda x, y: {"support_size": 0},
            _support_size_message("int 0"),
            lambda x, y: {"support_size": 5},
            5,
            id="support_size=0",
        ),
        pytest.param(
            lambda x, y: {"support_size": -1},
            _support_size_message("int -1"),
            lambda x, y: {"support_size": 5},
            5,
            id="support_size=-1",
        ),
        pytest.param(
            lambda x, y: {"support_size": True},
            _support_size_message("bool"),
            lambda x, y: {"support_size": 5},
            5,
            id="support_size=True",
        ),
        pytest.param(
            lambda x, y: {"support_x": x[:0], "support_y": y[:0]},
            "support_x and support_y hold no examples; give at least one example "
            "and its label.",
            lambda x, y: {"support_x": x, "support_y": y},
            10,
            id="no-examples",
        ),
    ],
)
def test_gp_update_support_refuses_bad_input(changes, message, repair, rows) -> None:
    """Each refusal names the argument, draws no random numbers and leaves the
    model as it was (its support, the support embeddings and the prototypes
    included); the call its message asks for replaces the support."""
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = _gp()
    model.update_support(x, y, support_size=10)
    args = {"support_x": x, "support_y": y, "support_size": 10, **changes(x, y)}
    before = model_state(model)
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match=re.escape(message)):
        model.update_support(**args)
    assert_model_unchanged(model, before)
    assert torch.equal(torch.get_rng_state(), rng), "the refusal drew random numbers"

    args.update(repair(x, y))
    model.update_support(**args)
    assert [len(r) for r in model.get_support().values()] == [rows] * CLASSES
    assert model.get_prototypes().shape[0] == CLASSES


@pytest.mark.parametrize("support_size", [40, 100])
def test_gp_update_support_keeps_every_example_of_a_single_class(support_size) -> None:
    """A support of one class keeps all of its examples when ``support_size`` is at
    least their number."""
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = _gp()
    model.update_support(x[:40], y[:40], support_size=support_size)
    support = model.get_support()
    assert list(support) == [0]
    assert sorted(support[0][:, 0].tolist()) == sorted(x[:40, 0].tolist())
    assert model.get_prototypes().shape[0] == 1


def _no_support_message(method: str) -> str:
    if method.startswith("EquineGP."):
        train_call = "train_model (with vis_support=True)"
    else:
        train_call = "train_model"
    return (
        f"{method} needs a support, and this model has none; call {train_call} or "
        "update_support first."
    )


@pytest.mark.parametrize(
    ("build", "method"),
    [
        pytest.param(_gp, "compute_prototypes", id="gp-compute_prototypes"),
        pytest.param(_gp, "get_support", id="gp-get_support"),
        pytest.param(_gp, "get_prototypes", id="gp-get_prototypes"),
        pytest.param(
            lambda: quick_gp()[0],
            "get_support",
            id="gp-trained-without-vis_support-get_support",
        ),
        pytest.param(_protonet, "get_support", id="protonet-get_support"),
        pytest.param(_protonet, "get_prototypes", id="protonet-get_prototypes"),
    ],
)
def test_a_model_without_a_support_refuses_to_read_it(build, method) -> None:
    """The refusal names the method and leaves the model as it was; after the
    update_support call the message asks for, the method returns one entry per
    class."""
    _, x, y = separable_dataset()
    model = build()
    name = f"{type(model).__name__}.{method}"
    before = model_state(model)
    with pytest.raises(ValueError, match=re.escape(_no_support_message(name))):
        getattr(model, method)()
    assert_model_unchanged(model, before)

    if isinstance(model, eq.EquineGP):
        model.update_support(x, y, support_size=10)
    else:
        model.update_support(x, y.float(), calib_frac=0.5)
    assert len(getattr(model, method)()) == CLASSES


@pytest.mark.parametrize(
    "build, option",
    [(_protonet, {"support_size": 10}), (_gp, {"calib_frac": 0.5})],
    ids=["protonet", "gp"],
)
def test_the_other_class_option_is_refused(build, option) -> None:
    _, x, y = separable_dataset()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build().update_support(x, y, **option)


@pytest.mark.parametrize(
    "build, call, message",
    [
        (
            _protonet,
            lambda m, x, y: m.update_support(x, y),
            "missing required keyword argument: 'calib_frac'",
        ),
        (
            _gp,
            lambda m, x, y: m.update_support(x, y),
            "missing required keyword argument: 'support_size'",
        ),
    ],
    ids=["protonet-missing", "gp-missing"],
)
def test_a_missing_option_is_refused(build, call, message) -> None:
    _, x, y = separable_dataset()
    with pytest.raises(TypeError, match=message):
        call(build(), x, y.float())


def test_untrained_protonet_support_attributes_are_empty() -> None:
    model = _protonet()
    assert len(model.support) == 0
    assert len(model.support_embeddings) == 0
    assert model.prototypes.numel() == 0


@pytest.mark.parametrize(
    "name, value",
    [
        ("support", OrderedDict()),
        ("support_embeddings", OrderedDict()),
        ("prototypes", torch.zeros(CLASSES, CLASSES)),
    ],
)
def test_protonet_support_attributes_are_read_only(name, value) -> None:
    model = _protonet()
    with pytest.raises(AttributeError, match="has no setter"):
        setattr(model, name, value)


def _assert_attributes_are_the_support(model) -> None:
    assert list(model.support) == list(range(CLASSES))
    assert model.support is model.get_support()
    assert model.support_embeddings is model.model.support_embeddings
    assert list(model.support_embeddings) == list(model.support)
    assert model.prototypes is model.get_prototypes()


def test_protonet_support_attributes_after_train_model_and_load(tmp_path) -> None:
    model, _, _ = quick_protonet()
    _assert_attributes_are_the_support(model)
    path = str(tmp_path / "protonet.eq")
    model.save(path)
    _assert_attributes_are_the_support(eq.EquineProtonet.load(path))


def test_protonet_given_only_its_support_saves_a_file_that_loads(tmp_path) -> None:
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = _protonet()
    assert model.train_summary["modelType"] == "EquineProtonet"
    model.update_support(x, y, calib_frac=0.2, label_names=NAMES)
    path = str(tmp_path / "support_only.eq")
    model.save(path)
    loaded = eq.load_equine_model(path)
    assert isinstance(loaded, eq.EquineProtonet)
    assert loaded.get_label_names() == NAMES


def test_protonet_support_attributes_follow_a_dtype_cast() -> None:
    model, _, _ = quick_protonet()
    model.to(torch.float64)
    _assert_attributes_are_the_support(model)
    for tensors in (model.support.values(), model.support_embeddings.values()):
        assert all(t.dtype == torch.float64 for t in tensors)
    assert model.prototypes.dtype == torch.float64


def test_gp_load_keeps_the_support_embeddings(tmp_path) -> None:
    model, _, _ = quick_gp(vis_support=True)
    path = str(tmp_path / "gp.eq")
    model.save(path)
    loaded = eq.EquineGP.load(path)
    assert list(model.support_embeddings) == list(range(CLASSES))
    assert list(loaded.support_embeddings) == list(model.support_embeddings)
    for label, embeddings in model.support_embeddings.items():
        assert torch.equal(loaded.support_embeddings[label], embeddings)
    assert torch.equal(loaded.prototypes, model.prototypes)


def test_unpickled_protonet_drops_instance_support_entries() -> None:
    """Unpickling an ``EquineProtonet`` whose instance dict holds ``support``,
    ``support_embeddings`` and ``prototypes`` entries drops them, and dtype and
    device moves work."""
    model = eq.EquineProtonet(BasicEmbeddingModel(6, 3), 3)
    model.__dict__.update(
        support=OrderedDict(),
        support_embeddings=OrderedDict(),
        prototypes=torch.Tensor(),
    )
    restored = pickle.loads(pickle.dumps(model))
    assert not {"support", "support_embeddings", "prototypes"} & set(vars(restored))
    restored.to(torch.float64)
    restored.cpu()
