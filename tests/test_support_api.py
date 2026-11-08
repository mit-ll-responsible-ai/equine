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

import pickle
import warnings
from collections import OrderedDict

import icontract
import pytest
import torch
from conftest import BasicEmbeddingModel
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq

NAMES = ["a", "b", "c"]


def _protonet():
    return eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)


def _gp():
    return eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )


def _trained_protonet():
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = _protonet()
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    return model, x, y.float()


def _trained_gp(vis_support: bool = False):
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = _gp()
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=1,
        batch_size=32,
        vis_support=vis_support,
    )
    return model, x, y.long()


def test_equine_declares_the_support_methods_abstract() -> None:
    assert {"get_support", "update_support"} <= eq.Equine.__abstractmethods__


@pytest.mark.parametrize(
    "build, keywords",
    [
        (_trained_protonet, {"calib_frac": 0.5}),
        (_trained_protonet, {"calib_frac": 0.5, "label_names": NAMES}),
        (_trained_gp, {"support_size": 10}),
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
    return _trained_gp(vis_support=True)


@pytest.mark.parametrize(
    "build, call, repair, message",
    [
        (
            _trained_protonet,
            lambda m, x, y: m.update_support(x, y, 0.2),
            lambda m, x, y: m.update_support(x, y, calib_frac=0.2),
            _PROTONET_BY_POSITION.format(1),
        ),
        (
            _trained_protonet,
            lambda m, x, y: m.update_support(x, y, 0.2, NAMES),
            lambda m, x, y: m.update_support(x, y, calib_frac=0.2, label_names=NAMES),
            _PROTONET_BY_POSITION.format(2),
        ),
        (
            _trained_protonet,
            lambda m, x, y: m.update_support(x, y, 0.2, label_names=NAMES),
            lambda m, x, y: m.update_support(x, y, calib_frac=0.2, label_names=NAMES),
            _PROTONET_BY_POSITION.format(1),
        ),
        (
            _trained_protonet,
            lambda m, x, y: m.update_support(x, y, 0.2, calib_frac=0.2),
            lambda m, x, y: m.update_support(x, y, calib_frac=0.2),
            _PROTONET_BY_POSITION.format(1),
        ),
        (
            _trained_protonet,
            lambda m, x, y: m.update_support(x, y, 0.2, calib_x=x[::4], calib_y=y[::4]),
            lambda m, x, y: m.update_support(
                x, y, calib_frac=0.2, calib_x=x[::4], calib_y=y[::4]
            ),
            _PROTONET_BY_POSITION.format(1),
        ),
        (
            _trained_protonet,
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


@pytest.mark.contracts
def test_out_of_range_calib_frac_violates_a_contract() -> None:
    model, x, y = _trained_protonet()
    support = {label: rows.clone() for label, rows in model.get_support().items()}
    with pytest.raises(icontract.ViolationError, match="calib_frac"):
        model.update_support(x, y, calib_frac=1.5)
    assert list(model.get_support()) == list(support)
    for label, rows in support.items():
        assert torch.equal(model.get_support()[label], rows)
    assert model.label_names is None


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
    model, _, _ = _trained_protonet()
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
    model, _, _ = _trained_protonet()
    model.to(torch.float64)
    _assert_attributes_are_the_support(model)
    for tensors in (model.support.values(), model.support_embeddings.values()):
        assert all(t.dtype == torch.float64 for t in tensors)
    assert model.prototypes.dtype == torch.float64


def test_gp_load_keeps_the_support_embeddings(tmp_path) -> None:
    model, _, _ = _trained_gp(vis_support=True)
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
