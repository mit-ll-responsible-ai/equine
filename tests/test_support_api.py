# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""The support API of both model classes.

``Equine`` declares ``get_support`` and ``update_support`` as abstract methods.
``update_support`` takes its options by keyword (``calib_frac`` and ``label_names`` on
``EquineProtonet``, ``support_size`` on ``EquineGP``). Passing them by position emits a
``DeprecationWarning`` at the caller's line, then updates the support as the keywords
do, with the same type and range checks; this is also checked under ``python -O``.
``EquineProtonet.support``, ``support_embeddings`` and ``prototypes`` are read-only
views of the inner model's support.
"""

import copy
import json
import os
import pickle
import subprocess
import sys
import warnings
from collections import OrderedDict

import icontract
import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation
from conftest import BasicEmbeddingModel
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq

NAMES = ["a", "b", "c"]


def _protonet(cls=eq.EquineProtonet):
    return cls(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)


def _gp(cls=eq.EquineGP):
    return cls(
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


# Each positional form, the model it is called on and the keywords it stands for.
LEGACY_FORMS = {
    "protonet (x, y, f)": (
        _trained_protonet,
        lambda m, x, y: m.update_support(x, y, 0.5),
        {"calib_frac": 0.5},
    ),
    "protonet (x, y, f, names)": (
        _trained_protonet,
        lambda m, x, y: m.update_support(x, y, 0.5, NAMES),
        {"calib_frac": 0.5, "label_names": NAMES},
    ),
    "protonet (x, y, f, label_names=names)": (
        _trained_protonet,
        lambda m, x, y: m.update_support(x, y, 0.5, label_names=NAMES),
        {"calib_frac": 0.5, "label_names": NAMES},
    ),
    "gp (x, y, n)": (
        _trained_gp,
        lambda m, x, y: m.update_support(x, y, 10),
        {"support_size": 10},
    ),
}


def legacy_forms_report() -> list[dict]:
    """
    Run each positional form and its keyword form on two copies of one trained model,
    both after ``torch.manual_seed(1)``, the keyword form with warnings raised as
    errors. Per form, return the ``update_support`` warnings of the positional call
    (category name, whether it points at this file, whether it points at the line of
    the call), whether the two copies hold the same support and prototypes, and their
    label names. The result is JSON-serializable.
    """
    report = []
    trained = {}
    for form, (build, legacy, keywords) in LEGACY_FORMS.items():
        if build not in trained:
            trained[build] = build()
        model, x, y = trained[build]
        by_position, by_keyword = copy.deepcopy(model), copy.deepcopy(model)
        torch.manual_seed(1)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            legacy(by_position, x, y)
        torch.manual_seed(1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            by_keyword.update_support(x, y, **keywords)
        support, expected = by_position.get_support(), by_keyword.get_support()
        report.append(
            {
                "form": form,
                "warnings": [
                    [
                        w.category.__name__,
                        w.filename == __file__,
                        w.lineno == legacy.__code__.co_firstlineno,
                    ]
                    for w in record
                    if "update_support" in str(w.message)
                ],
                "same_support": list(support) == list(expected)
                and all(torch.equal(support[k], expected[k]) for k in expected),
                "same_prototypes": torch.equal(
                    by_position.get_prototypes(), by_keyword.get_prototypes()
                ),
                "support_rows": sum(len(rows) for rows in support.values()),
                "label_names": [by_position.label_names, by_keyword.label_names],
            }
        )
    return report


def _assert_like_the_keywords(report: list[dict]) -> None:
    assert [row["form"] for row in report] == list(LEGACY_FORMS)
    for row in report:
        assert row["warnings"] == [["DeprecationWarning", True, True]], row
        assert row["same_support"] and row["same_prototypes"], row
        # f = 0.5 keeps half of the 120 rows; n = 10 keeps 10 rows of each class.
        assert row["support_rows"] == (30 if row["form"].startswith("gp") else 60), row
        names = NAMES if "names" in row["form"] else None
        assert row["label_names"] == [names, names], row


def test_positional_options_warn_and_update_like_the_keywords() -> None:
    _assert_like_the_keywords(legacy_forms_report())


def test_positional_options_warn_and_update_like_the_keywords_under_python_O() -> None:
    tests = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(eq.__file__)))
    path = [package_root, tests, os.environ.get("PYTHONPATH", "")]
    code = (
        "import json, sys\n"
        "import test_support_api as t\n"
        "print(json.dumps({'optimize': sys.flags.optimize, "
        "'report': t.legacy_forms_report()}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(path)),
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.splitlines()[-1])
    assert result["optimize"] == 1
    _assert_like_the_keywords(result["report"])


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda m, x, y: m.update_support(x, y, calib_frac=1.5), "calib_frac"),
        (lambda m, x, y: m.update_support(x, y, 1.5), None),
        (lambda m, x, y: m.update_support(x, y, 1.5, NAMES), None),
        (lambda m, x, y: m.update_support(x, y, 0.0, label_names=NAMES), None),
    ],
    ids=[
        "calib_frac=f",
        "(x, y, f)",
        "(x, y, f, names)",
        "(x, y, f, label_names=names)",
    ],
)
def test_out_of_range_calib_frac_violates_a_contract(call, match) -> None:
    model, x, y = _trained_protonet()
    support = {label: rows.clone() for label, rows in model.get_support().items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(icontract.ViolationError, match=match):
            call(model, x, y)
    assert list(model.get_support()) == list(support)
    for label, rows in support.items():
        assert torch.equal(model.get_support()[label], rows)
    assert model.label_names is None


@pytest.mark.parametrize(
    "build, call",
    [
        (_trained_protonet, lambda m, x, y: m.update_support(x, y, 10)),
        (_trained_protonet, lambda m, x, y: m.update_support(x, y, 0.5, "a")),
        (_trained_gp, lambda m, x, y: m.update_support(x, y, 0.5)),
    ],
    ids=["protonet-int-calib-frac", "protonet-str-label-names", "gp-float-size"],
)
def test_positional_option_of_the_wrong_type_is_refused_by_beartype(
    build, call
) -> None:
    model, x, y = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(
            BeartypeCallHintParamViolation, match=r"update_support\(\) parameter"
        ):
            call(model, x, y)


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
            _protonet,
            lambda m, x, y: m.update_support(x, y, 0.5, calib_frac=0.5),
            "multiple values for argument 'calib_frac'",
        ),
        (
            _protonet,
            lambda m, x, y: m.update_support(x, y, 0.5, NAMES, label_names=NAMES),
            "multiple values for argument 'label_names'",
        ),
        (
            _protonet,
            lambda m, x, y: m.update_support(x, y, 0.5, NAMES, NAMES),
            "at most 2 options by position",
        ),
        (
            _gp,
            lambda m, x, y: m.update_support(x, y),
            "missing required keyword argument: 'support_size'",
        ),
        (
            _gp,
            lambda m, x, y: m.update_support(x, y, 10, support_size=10),
            "multiple values for argument 'support_size'",
        ),
        (
            _gp,
            lambda m, x, y: m.update_support(x, y, 10, 10),
            "at most 1 option by position",
        ),
    ],
    ids=[
        "protonet-missing",
        "protonet-calib-frac-twice",
        "protonet-label-names-twice",
        "protonet-too-many",
        "gp-missing",
        "gp-support-size-twice",
        "gp-too-many",
    ],
)
def test_a_missing_or_repeated_option_is_refused_without_a_warning(
    build, call, message
) -> None:
    _, x, y = separable_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=message):
            call(build(), x, y.float())


class _ProtonetPassingOptionsByPosition(eq.EquineProtonet):
    """Counts the calls of its ``update_support`` and passes the options on by
    position."""

    def update_support(self, support_x, support_y, calib_frac, label_names=None):
        self.calls = getattr(self, "calls", 0) + 1
        super().update_support(support_x, support_y, calib_frac, label_names)


class _GPPassingSizeByPosition(eq.EquineGP):
    """Counts the calls of its ``update_support`` and passes ``support_size`` on by
    position."""

    def update_support(self, support_x, support_y, support_size):
        self.calls = getattr(self, "calls", 0) + 1
        super().update_support(support_x, support_y, support_size)


@pytest.mark.parametrize(
    "build, option, y_dtype",
    [
        (lambda: _protonet(_ProtonetPassingOptionsByPosition), 0.5, torch.float),
        (lambda: _gp(_GPPassingSizeByPosition), 10, torch.long),
    ],
    ids=["protonet", "gp"],
)
def test_an_override_passing_options_by_position_is_called_once(
    build, option, y_dtype
) -> None:
    model = build()
    _, x, y = separable_dataset()
    with pytest.warns(DeprecationWarning, match="update_support") as record:
        model.update_support(x, y.to(y_dtype), option)
    found = [w for w in record if "update_support" in str(w.message)]
    assert len(found) == 1
    assert found[0].filename == __file__
    assert model.calls == 1
    assert list(model.get_support()) == list(range(CLASSES))


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


def test_protonet_support_attributes_follow_a_dtype_cast() -> None:
    model, _, _ = _trained_protonet()
    model.to(torch.float64)
    _assert_attributes_are_the_support(model)
    for tensors in (model.support.values(), model.support_embeddings.values()):
        assert all(t.dtype == torch.float64 for t in tensors)
    assert model.prototypes.dtype == torch.float64


def test_gp_load_keeps_the_support_embeddings(tmp_path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
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
