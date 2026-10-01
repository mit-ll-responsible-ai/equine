# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""
The model layer: EquineProtonet and EquineGP save and load (``save``, ``load``,
``load_equine_model``). Embedding models persist as a recipe + state_dict;
TorchScript only behind allow_executable / trust_executable for one transition
release. The file-reading layer underneath (``load_checkpoint``) is
tested in test_safe_loading.py.
"""

import io
import os
import subprocess
import sys
import warnings
from collections import OrderedDict
from typing import Any

import numpy as np
import pytest
import torch
from conftest import BasicEmbeddingModel, _rewrite_zip

import equine as eq
import equine.equine_gp
import equine.equine_protonet
import equine.utils


class UnregisteredNet(torch.nn.Module):
    """An embedding with no recipe (also what torch.jit.script yields)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(6, 3)

    def forward(self, x):
        return self.lin(x)


def _dataset(seed: int = 0):
    torch.manual_seed(seed)
    X = torch.rand(120, 6)
    Y = torch.tensor([0] * 40 + [1] * 40 + [2] * 40)
    return torch.utils.data.TensorDataset(X, Y), X


def train_protonet(embedding, **kwargs):
    dataset, X = _dataset()
    model = eq.EquineProtonet(embedding, 3, **kwargs)
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    return model, X


def train_gp(embedding, **kwargs):
    dataset, X = _dataset()
    model = eq.EquineGP(embedding, 3, 3, num_random_features=16, **kwargs)
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.001),
        num_epochs=1,
        batch_size=32,
        vis_support=True,
    )
    return model, X


TRAINERS = [
    pytest.param(train_protonet, eq.EquineProtonet, id="protonet"),
    pytest.param(train_gp, eq.EquineGP, id="gp"),
]


@pytest.fixture
def recipe_builds(monkeypatch) -> list[set[str]]:
    """Record the device types of every module the loader builds from a recipe."""
    # Not a torch.empty spy: torch caches tensor-creation functions on first device-context entry.
    built_on: list[set[str]] = []
    real_build = equine.utils.build_from_recipe

    def spy(recipe):
        module = real_build(recipe)
        built_on.append({p.device.type for p in module.parameters()})
        return module

    monkeypatch.setattr(equine.utils, "build_from_recipe", spy)
    return built_on


def _rewrite(path: str, **changes) -> dict:
    """Load a saved file, apply ``changes`` (a value of ``...`` deletes the key), save it back."""
    ckpt = torch.load(path, weights_only=True)
    for key, value in changes.items():
        if value is ...:
            del ckpt[key]
        else:
            ckpt[key] = value
    torch.save(ckpt, path)
    return ckpt


def _scripted(module: torch.nn.Module) -> torch.jit.ScriptModule:
    """``torch.jit.script`` that also works on Python 3.14 (see prepare_jit_module)."""
    return torch.jit.script(equine.utils.prepare_jit_module(module))


def _archive_of(module: torch.nn.Module) -> torch.Tensor:
    buffer = io.BytesIO()
    torch.jit.save(_scripted(module), buffer)
    return equine.utils._jit_archive_to_tensor(buffer)


def assert_same(a, b, X):
    oa, ob = a.predict(X[:8]), b.predict(X[:8])
    assert torch.allclose(oa.classes, ob.classes, atol=1e-6)
    assert torch.allclose(oa.ood_scores, ob.ood_scores, atol=1e-6)


# --- data-only files -----------------------------------------------------------


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_registered_embedding_saves_data_only_and_round_trips(
    tmp_path, train, cls
) -> None:
    model, X = train(BasicEmbeddingModel(6, 3))
    model.label_names = ["a", "b", "c"]
    path = str(tmp_path / "m.eq")
    model.save(path)

    ckpt = torch.load(path, weights_only=True)  # must not raise
    assert ckpt["equine_format_version"] == 2
    assert ckpt["contains_executable"] is False
    assert "embed_jit_save" not in ckpt
    assert ckpt["embedding_recipe"] == {
        "builder": "equine.tests.basic",
        "kwargs": {"tensor_dim": 6, "num_classes": 3},
    }
    assert set(ckpt["embedding_state_dict"]) == set(model.embedding_model.state_dict())

    for reloaded in (cls.load(path), eq.load_equine_model(path)):
        assert isinstance(reloaded, cls)
        assert isinstance(reloaded.embedding_model, BasicEmbeddingModel)
        assert reloaded.label_names == ["a", "b", "c"]
        support = reloaded.get_support()  # type: ignore[attr-defined]
        assert all(isinstance(k, int) for k in support.keys())
        assert_same(model, reloaded, X)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_numpy_string_names_are_saved_as_plain_strings(tmp_path, train, cls) -> None:
    """LabelEncoder.inverse_transform yields numpy strings, which the
    weights-only unpickler refuses; save() must write plain str."""
    model, X = train(BasicEmbeddingModel(6, 3))
    model.label_names = [np.str_(name) for name in ("a", "b", "c")]
    model.feature_names = [np.str_(f"f{i}") for i in range(6)]
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)  # must not raise
    assert ckpt["label_names"] == ["a", "b", "c"]
    for loader in (cls.load, eq.load_equine_model):
        reloaded = loader(path)
        assert reloaded.label_names == ["a", "b", "c"]
        assert reloaded.feature_names == [f"f{i}" for i in range(6)]
        names = reloaded.label_names + reloaded.feature_names
        assert all(type(name) is str for name in names)
        assert_same(model, reloaded, X)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_caller_supplied_architecture_overrides_recipe(tmp_path, train, cls) -> None:
    model, X = train(eq.MLP(6, [16], 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    for loader in (cls.load, eq.load_equine_model):
        fresh = eq.MLP(6, [16], 3)
        reloaded = loader(path, embedding_model=fresh)
        assert reloaded.embedding_model is fresh
        assert_same(model, reloaded, X)

        # An override whose architecture differs from the recipe is used as given,
        # so the stored weights must fit it; here they cannot.
        different = eq.MLP(6, [8, 8], 3)
        with pytest.raises(ValueError, match="does not match the weights") as err:
            loader(path, embedding_model=different)
        assert "MLP" in str(err.value)
        assert len(str(err.value)) < 1000


def test_protonet_load_onto_an_explicit_device_round_trips(tmp_path) -> None:
    model, X = train_protonet(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    reloaded = eq.EquineProtonet.load(path, device="cpu")
    assert reloaded.device == "cpu"
    assert {p.device.type for p in reloaded.embedding_model.parameters()} == {"cpu"}
    assert_same(model, reloaded, X)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_save_and_load_flags_are_keyword_only(tmp_path, train, cls) -> None:
    """A positional True must not silently mean allow_executable,
    allow_unsafe_legacy_format or trust_executable."""
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    with pytest.raises(TypeError):
        model.save(path, True)  # type: ignore[misc]
    model.save(path)
    positional = (path, None, True) if cls is eq.EquineProtonet else (path, True)
    with pytest.raises(TypeError):
        cls.load(*positional)  # type: ignore[misc]
    with pytest.raises(TypeError):
        eq.load_equine_model(path, True)  # type: ignore[misc]


# --- transition bridge -------------------------------------------------------------


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_unregistered_embedding_is_refused_without_opt_in(tmp_path, train, cls) -> None:
    model, _ = train(UnregisteredNet())
    with pytest.raises(ValueError) as err:
        model.save(str(tmp_path / "m.eq"))
    message = str(err.value)
    assert "UnregisteredNet" in message
    assert "embedding_architecture" in message
    assert "allow_executable=True" in message
    assert "trust_executable=True" in message


@pytest.mark.parametrize("train, cls", TRAINERS)
# The transition's FutureWarning is asserted in its own test.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_executable_opt_in_is_flagged_and_needs_trust(tmp_path, train, cls) -> None:
    model, X = train(_scripted(UnregisteredNet()))
    path = str(tmp_path / "m.eq")
    model.save(path, allow_executable=True)

    ckpt = torch.load(path, weights_only=True)  # still safe to *read*
    assert ckpt["contains_executable"] is True
    assert ckpt["embed_jit_save"].dtype == torch.uint8

    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="trust_executable=True"):
            loader(path)
        assert_same(model, loader(path, trust_executable=True), X)


def _future_warnings(record) -> list:
    return [w for w in record if issubclass(w.category, FutureWarning)]


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_executable_branch_warns_that_it_goes_away(tmp_path, train, cls) -> None:
    """Saving TorchScript and loading it under trust each emit a FutureWarning
    that points at the caller and names the migration route."""
    model, _ = train(_scripted(UnregisteredNet()))
    path = str(tmp_path / "m.eq")
    with pytest.warns(FutureWarning, match="embedding_model=") as record:
        model.save(path, allow_executable=True)
    assert [w.filename for w in _future_warnings(record)] == [__file__]

    for loader in (cls.load, eq.load_equine_model):
        with pytest.warns(FutureWarning, match="embedding_model=") as record:
            loader(path, trust_executable=True)
        assert [w.filename for w in _future_warnings(record)] == [__file__]


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_data_only_paths_do_not_warn_about_the_transition(tmp_path, train, cls) -> None:
    model, _ = train(_scripted(BasicEmbeddingModel(6, 3)))
    flagged = str(tmp_path / "flagged.eq")
    with pytest.warns(FutureWarning):
        model.save(flagged, allow_executable=True)
    data_only = str(tmp_path / "data_only.eq")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        # migrating: the archive's weights go into a registered module
        migrated = cls.load(
            flagged, trust_executable=True, embedding_model=BasicEmbeddingModel(6, 3)
        )
        migrated.save(data_only)
        for loader in (cls.load, eq.load_equine_model):
            loader(data_only)
    assert _future_warnings(record) == []


@pytest.mark.parametrize("train, cls", TRAINERS)
# The transition's FutureWarning is asserted in its own test.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_archive_without_flag_still_needs_trust(tmp_path, train, cls) -> None:
    """A hand-edited file cannot bypass the check by dropping the flag."""
    model, _ = train(_scripted(UnregisteredNet()))
    path = str(tmp_path / "m.eq")
    model.save(path, allow_executable=True)
    ckpt = torch.load(path, weights_only=True)
    del ckpt["contains_executable"]
    torch.save(ckpt, path)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="trust_executable=True"):
            loader(path)


@pytest.mark.parametrize("train, cls", TRAINERS)
@pytest.mark.parametrize("with_archive", [True, False], ids=["archive", "flag-only"])
def test_executable_flag_needs_trust_even_beside_a_recipe(
    tmp_path, train, cls, with_archive
) -> None:
    """The flag and the archive are each enough to require trust (spec section 5),
    even when the file also carries a usable recipe."""
    model, X = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    extra = (
        {"embed_jit_save": _archive_of(BasicEmbeddingModel(6, 3))}
        if with_archive
        else {}
    )
    _rewrite(path, contains_executable=True, **extra)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="trust_executable=True"):
            loader(path)
        assert_same(model, loader(path, trust_executable=True), X)


@pytest.mark.parametrize("train, cls", TRAINERS)
# The transition's FutureWarning is asserted in its own test.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_migration_from_executable_file_to_recipe(tmp_path, train, cls) -> None:
    model, X = train(_scripted(BasicEmbeddingModel(6, 3)))
    flagged = str(tmp_path / "flagged.eq")
    model.save(flagged, allow_executable=True)

    migrated = cls.load(
        flagged, trust_executable=True, embedding_model=BasicEmbeddingModel(6, 3)
    )
    data_only = str(tmp_path / "migrated.eq")
    migrated.save(data_only)  # no flag needed any more

    ckpt = torch.load(data_only, weights_only=True)
    assert ckpt["contains_executable"] is False
    assert ckpt["embedding_recipe"]["builder"] == "equine.tests.basic"
    assert_same(model, eq.load_equine_model(data_only), X)


@pytest.mark.parametrize("train, cls", TRAINERS)
# The transition's FutureWarning is asserted in its own test.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_migration_without_trust_is_refused(tmp_path, train, cls) -> None:
    model, _ = train(_scripted(BasicEmbeddingModel(6, 3)))
    flagged = str(tmp_path / "flagged.eq")
    model.save(flagged, allow_executable=True)
    with pytest.raises(ValueError, match="trust_executable=True"):
        cls.load(flagged, embedding_model=BasicEmbeddingModel(6, 3))


# --- recipe / weights consistency ----------------------------------------------------


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_recipe_that_disagrees_with_weights_is_refused_before_allocation(
    tmp_path, train, cls, recipe_builds
) -> None:
    """A tampered recipe describing a huge network must fail on the meta device,
    before any real tensor is allocated."""
    model, _ = train(eq.MLP(6, [16], 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    ckpt["embedding_recipe"]["kwargs"]["hidden_sizes"] = [10**6]  # ~10M params if built
    torch.save(ckpt, path)

    recipe_builds.clear()  # save() checks its recipe on the meta device too
    with pytest.raises(ValueError, match="does not match the weights"):
        cls.load(path)
    assert recipe_builds == [{"meta"}], (
        f"recipe built off the meta device: {recipe_builds}"
    )


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_recipe_without_weights_is_refused(tmp_path, train, cls) -> None:
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    _rewrite(path, embedding_state_dict=...)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="does not match the weights"):
            loader(path)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_malformed_state_dict_is_refused(tmp_path, train, cls) -> None:
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    key = next(iter(model.embedding_model.state_dict()))
    for bad in ([1, 2], torch.zeros(3), {key: 1}, {0: torch.zeros(1)}):
        _rewrite(path, embedding_state_dict=bad)
        for load in (
            cls.load,
            eq.load_equine_model,
            lambda p: cls.load(p, embedding_model=BasicEmbeddingModel(6, 3)),
        ):
            with pytest.raises(ValueError, match="embedding_state_dict is malformed"):
                load(path)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_override_that_disagrees_with_weights_is_refused(tmp_path, train, cls) -> None:
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="does not match the weights"):
            loader(path, embedding_model=eq.MLP(6, [4], 3))


@pytest.mark.parametrize("train, cls", TRAINERS)
@pytest.mark.parametrize("override", [False, True], ids=["recipe", "override"])
def test_mismatch_errors_stay_bounded(tmp_path, train, cls, override) -> None:
    """A file-supplied parameter name of any length yields a short error."""
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    state_dict = dict(model.embedding_model.state_dict())
    state_dict["A" * 1_000_000] = torch.zeros(1)
    _rewrite(path, embedding_state_dict=state_dict)
    kwargs: dict[str, Any] = (
        {"embedding_model": BasicEmbeddingModel(6, 3)} if override else {}
    )
    with pytest.raises(ValueError, match="does not match the weights") as err:
        cls.load(path, **kwargs)
    assert "1 parameter(s) differ" in str(err.value)
    assert len(str(err.value)) < 1000


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_override_with_no_weights_in_file_is_refused(tmp_path, train, cls) -> None:
    """An override must not silently hand back its own untrained weights."""
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    _rewrite(path, embedding_recipe=..., embedding_state_dict=...)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="contains no embedding weights"):
            loader(path, embedding_model=BasicEmbeddingModel(6, 3))


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_file_with_neither_recipe_nor_archive_is_refused(tmp_path, train, cls) -> None:
    """Without an override there is nothing to rebuild the embedding from."""
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    _rewrite(path, embedding_recipe=..., embedding_state_dict=...)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="no recipe and no archive"):
            loader(path)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_weights_that_fail_to_load_raise_value_error(tmp_path, train, cls) -> None:
    """Failures inside load_state_dict (here: bad extra state) surface as ValueError."""

    @eq.embedding_architecture("equine.tests.extra_state")
    class WithExtraState(torch.nn.Module):
        def __init__(self, d: int = 6) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(d, 3)
            self.version = 1

        def get_extra_state(self):
            return {"version": self.version}

        def set_extra_state(self, state) -> None:
            self.version = int(state["version"])

        def forward(self, x):
            return self.lin(x)

    model, X = train(WithExtraState())
    path = str(tmp_path / "m.eq")
    model.save(path)
    assert_same(model, cls.load(path), X)  # extra state round-trips

    state_dict = dict(model.embedding_model.state_dict())
    state_dict["_extra_state"] = "not a dict"
    _rewrite(path, embedding_state_dict=state_dict)
    with pytest.raises(ValueError, match="Could not load the embedding weights"):
        cls.load(path)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_tied_weights_round_trip(tmp_path, train, cls) -> None:
    """Shared parameters are one tensor in the file; that is not a storage trick."""

    @eq.embedding_architecture("equine.tests.tied")
    class Tied(torch.nn.Module):
        def __init__(self, d: int = 6) -> None:
            super().__init__()
            self.a = torch.nn.Linear(d, d)
            self.b = torch.nn.Linear(d, d)
            self.b.weight = self.a.weight

        def forward(self, x):
            return self.b(torch.relu(self.a(x)))[:, :3]

    model, X = train(Tied())
    path = str(tmp_path / "m.eq")
    model.save(path)
    reloaded = cls.load(path)
    assert reloaded.embedding_model.a.weight is reloaded.embedding_model.b.weight
    assert_same(model, reloaded, X)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_constructor_that_mutates_its_arguments_round_trips(
    tmp_path, train, cls
) -> None:
    """save() rebuilds the recipe to check it; a constructor that edits a list
    argument in place must not corrupt the recipe that is then written."""

    @eq.embedding_architecture("equine.tests.mutates_arguments")
    class MutatesArguments(torch.nn.Module):
        def __init__(self, in_features: int, hidden: list) -> None:
            super().__init__()
            hidden.insert(0, in_features)  # in place: the caller's list changes
            self.layers = torch.nn.Sequential(
                *(torch.nn.Linear(a, b) for a, b in zip(hidden, hidden[1:]))
            )

        def forward(self, x):
            return self.layers(x)

    model, X = train(MutatesArguments(6, [8, 3]))
    path = str(tmp_path / "m.eq")
    model.save(path)
    stored = torch.load(path, weights_only=True)["embedding_recipe"]
    assert stored == {
        "builder": "equine.tests.mutates_arguments",
        "kwargs": {"in_features": 6, "hidden": [8, 3]},
    }
    for loader in (cls.load, eq.load_equine_model):
        assert_same(model, loader(path), X)


# --- storage tricks ------------------------------------------------------------------


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_expanded_weights_are_refused_before_any_build(
    tmp_path, train, cls, recipe_builds
) -> None:
    """Weights that are expanded views of one element match a huge recipe's shapes
    while the file stays tiny; the real build would then allocate gigabytes."""
    model, _ = train(eq.MLP(6, [16], 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    with torch.device("meta"):
        huge = eq.MLP(6, [50_000_000], 3)  # ~500M parameters, ~2 GB if built
    _rewrite(
        path,
        embedding_recipe=eq.embedding_recipe(huge),
        embedding_state_dict={
            k: torch.zeros(1).expand(v.shape) for k, v in huge.state_dict().items()
        },
    )
    recipe_builds.clear()  # save() checks its recipe on the meta device too
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="storage is smaller than its shape"):
            loader(path)
    assert recipe_builds == [], f"recipe was built before the refusal: {recipe_builds}"


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_expanded_support_tensor_is_refused(tmp_path, train, cls) -> None:
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    label = next(iter(ckpt["support"]))
    ckpt["support"][label] = torch.zeros(1).expand(1_000_000, 6)
    torch.save(ckpt, path)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="storage is smaller than its shape"):
            loader(path)


def _views_of_one_storage(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The module's state_dict shapes, each a distinct view of one shared storage
    that is only as large as the biggest tensor."""
    shapes = {k: v.shape for k, v in module.state_dict().items()}
    base = torch.zeros(max(shape.numel() for shape in shapes.values()))
    return {k: base[: shape.numel()].view(shape) for k, shape in shapes.items()}


@pytest.mark.parametrize("train, cls", TRAINERS)
@pytest.mark.parametrize("override", [False, True], ids=["recipe", "override"])
def test_weights_sharing_one_storage_are_refused(
    tmp_path, train, cls, override, recipe_builds
) -> None:
    """Each tensor is honestly stored, but together they view one small storage,
    so a small file would claim a much larger network."""
    model, _ = train(eq.MLP(6, [16], 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    with torch.device("meta"):
        deep = eq.MLP(6, [64] * 8, 3)  # ~30K parameters over one 16 KB storage
    _rewrite(
        path,
        embedding_recipe=eq.embedding_recipe(deep),
        embedding_state_dict=_views_of_one_storage(deep),
    )
    kwargs: dict[str, Any] = (
        {"embedding_model": eq.MLP(6, [64] * 8, 3)} if override else {}
    )
    recipe_builds.clear()  # save() checks its recipe on the meta device too
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="too small for the architecture"):
            loader(path, **kwargs)
    assert all(devices == {"meta"} for devices in recipe_builds), recipe_builds


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_non_zip_file_is_refused_before_any_build(
    tmp_path, train, cls, recipe_builds
) -> None:
    """torch's pre-1.6 format sizes storages from the pickle alone, so a tiny file
    could claim any amount of weights."""
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    torch.save(
        torch.load(path, weights_only=True), path, _use_new_zipfile_serialization=False
    )
    recipe_builds.clear()  # save() checks its recipe on the meta device too
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="not a PyTorch zip archive"):
            loader(path)
    assert recipe_builds == []


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_truncated_weight_records_are_refused_before_any_build(
    tmp_path, train, cls, recipe_builds
) -> None:
    """Records shorter than the storages the pickle declares: a small file that
    claims the weights of a ~10M-parameter network."""
    model, _ = train(eq.MLP(6, [16], 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    with torch.device("meta"):
        huge = eq.MLP(6, [1_000_000], 3)
    ckpt = torch.load(path, weights_only=True)
    ckpt["embedding_recipe"] = eq.embedding_recipe(huge)
    ckpt["embedding_state_dict"] = {
        k: torch.zeros(v.shape, dtype=torch.uint8) for k, v in huge.state_dict().items()
    }
    full = str(tmp_path / "full.eq")
    torch.save(ckpt, full)
    _rewrite_zip(
        full,
        path,
        lambda name, data: (
            data[:4] if "/data/" in name and len(data) > 100_000 else data
        ),
    )
    assert os.path.getsize(path) < 100_000
    recipe_builds.clear()
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(
            ValueError, match="storage larger than the data it contains"
        ):
            loader(path)
    assert recipe_builds == []


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_loaded_model_does_not_depend_on_its_file(tmp_path, train, cls) -> None:
    model, X = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    reloaded = cls.load(path)
    os.truncate(path, 0)  # would fault tensors still backed by the file
    assert_same(model, reloaded, X)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_support_shared_across_labels_is_refused(tmp_path, train, cls) -> None:
    """One stored tensor referenced by every label: a small file, per-label work."""
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    shared = next(iter(ckpt["support"].values()))
    _rewrite(path, support={label: shared for label in ckpt["support"]})
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="alias one another"):
            loader(path)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_recipe_that_allocates_off_meta_is_not_built_for_real(
    tmp_path, train, cls, recipe_builds
) -> None:
    """save() refuses such a module; a crafted file naming it is refused at load
    after the meta build, before the real one."""
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    defective = _embedding_with_defect("explicit_device")
    _rewrite(
        path,
        embedding_recipe=eq.embedding_recipe(defective),
        embedding_state_dict=defective.state_dict(),
    )
    recipe_builds.clear()
    with pytest.raises(ValueError, match="off the meta device"):
        cls.load(path)
    assert recipe_builds == [{"meta"}], recipe_builds


# --- support storage ------------------------------------------------------------------


def test_protonet_saves_only_the_support_rows(tmp_path) -> None:
    """Each label's support is a view of all that class's training rows; saving
    the view would write (and leak) every one of them."""
    torch.manual_seed(0)
    X = torch.rand(3000, 6)
    Y = torch.tensor([0] * 1000 + [1] * 1000 + [2] * 1000)
    model = eq.EquineProtonet(BasicEmbeddingModel(6, 3), 3)
    model.train_model(
        torch.utils.data.TensorDataset(X, Y),
        num_episodes=2,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    path = str(tmp_path / "m.eq")
    model.save(path)

    ckpt = torch.load(path, weights_only=True)
    stored = list(ckpt["support"].values())
    stored += [kde["dataset"] for kde in ckpt["outlier_kde"].values()]
    for tensor in stored:
        assert (
            tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        )
    assert all(tuple(t.shape) == (10, 6) for t in ckpt["support"].values())
    assert os.path.getsize(path) < X.numel() * X.element_size() / 2


def test_gp_saves_only_the_support_rows(tmp_path) -> None:
    """EquineGP's support comes from generate_support too, so it is a view of all
    that class's training rows."""
    torch.manual_seed(0)
    X = torch.rand(3000, 6)
    Y = torch.tensor([0] * 1000 + [1] * 1000 + [2] * 1000)
    model = eq.EquineGP(BasicEmbeddingModel(6, 3), 3, 3, num_random_features=16)
    model.train_model(
        torch.utils.data.TensorDataset(X, Y),
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.001),
        num_epochs=1,
        batch_size=500,
        vis_support=True,
        support_size=10,
    )
    in_memory = list(model.support.values())
    assert any(
        t.untyped_storage().nbytes() > t.numel() * t.element_size() for t in in_memory
    ), "support is no longer a view; this test would pass vacuously"
    path = str(tmp_path / "m.eq")
    model.save(path)

    ckpt = torch.load(path, weights_only=True)
    for tensor in ckpt["support"].values():
        assert (
            tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        )
    assert all(tuple(t.shape) == (10, 6) for t in ckpt["support"].values())
    assert os.path.getsize(path) < X.numel() * X.element_size() / 2


def test_support_slices_of_one_tensor_round_trip(tmp_path) -> None:
    """Support set by hand from slices of one tensor is saved label by label."""
    model, X = train_protonet(BasicEmbeddingModel(6, 3))
    base = X[:30].clone()
    model.model.update_support(  # type: ignore[operator]
        OrderedDict((label, base[10 * label : 10 * (label + 1)]) for label in range(3))
    )
    path = str(tmp_path / "m.eq")
    model.save(path)
    assert_same(model, eq.EquineProtonet.load(path), X)


# --- save-time self-check ------------------------------------------------------------


def _embedding_with_defect(defect: str) -> torch.nn.Module:
    """A registered embedding that the loader could not rebuild from its recipe."""

    @eq.embedding_architecture(f"equine.tests.defect_{defect}")
    class Defective(torch.nn.Module):
        def __init__(self, d: int = 6) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(d, 3)
            if defect == "reads_values":
                self.scale = float(self.lin.weight.abs().max().item())  # fails on meta
            if defect == "explicit_device":
                self.register_buffer("table", torch.zeros(4, device="cpu"))

        def forward(self, x):
            return self.lin(x)

    module = Defective()
    if defect == "modified_after_construction":
        module.adapter = torch.nn.Linear(2, 2)
    return module


@pytest.mark.parametrize("train, cls", TRAINERS)
@pytest.mark.parametrize(
    "defect", ["modified_after_construction", "reads_values", "explicit_device"]
)
def test_save_refuses_an_embedding_its_recipe_cannot_rebuild(
    tmp_path, train, cls, defect
) -> None:
    model, _ = train(_embedding_with_defect(defect))
    with pytest.raises(ValueError, match="cannot be rebuilt from its recipe") as err:
        model.save(str(tmp_path / "m.eq"))
    assert "embedding_architecture" in str(err.value)


def _mlp_with(change) -> torch.nn.Module:
    mlp = eq.MLP(6, [8], 3)
    change(mlp)
    return mlp


@pytest.mark.parametrize("train, cls", TRAINERS)
@pytest.mark.parametrize(
    "change",
    [
        lambda mlp: mlp.net.__setitem__(1, torch.nn.Tanh()),
        lambda mlp: mlp.net.__setitem__(1, torch.nn.LeakyReLU(0.3)),
        lambda mlp: setattr(mlp.net[1], "inplace", True),
    ],
    ids=["swapped-activation", "swapped-with-hyperparameter", "reconfigured"],
)
def test_save_refuses_submodules_changed_after_construction(
    tmp_path, train, cls, change
) -> None:
    """Same parameter names and shapes, but not what the recipe rebuilds: the
    file would load a different network."""
    model, _ = train(_mlp_with(change))
    path = tmp_path / "m.eq"
    with pytest.raises(ValueError, match="constructor argument") as err:
        model.save(str(path))
    message = str(err.value)
    assert "cannot be rebuilt from its recipe" in message
    assert "'net.1'" in message
    assert not path.exists()


def test_submodule_check_is_independent_of_the_device() -> None:
    """The check compares a meta-device rebuild with the live module, so layer
    descriptions must not mention the device."""
    layers = [
        torch.nn.Linear(4, 3),
        torch.nn.Conv2d(2, 3, 3, stride=2, padding=1),
        torch.nn.BatchNorm1d(3),
        torch.nn.LayerNorm(3),
        torch.nn.Embedding(10, 3, padding_idx=0),
        torch.nn.LSTM(4, 3, num_layers=2),
        torch.nn.MultiheadAttention(4, 2),
        torch.nn.Dropout(0.2),
        torch.nn.PReLU(3),
    ]
    with torch.device("meta"):
        on_meta = [
            torch.nn.Linear(4, 3),
            torch.nn.Conv2d(2, 3, 3, stride=2, padding=1),
            torch.nn.BatchNorm1d(3),
            torch.nn.LayerNorm(3),
            torch.nn.Embedding(10, 3, padding_idx=0),
            torch.nn.LSTM(4, 3, num_layers=2),
            torch.nn.MultiheadAttention(4, 2),
            torch.nn.Dropout(0.2),
            torch.nn.PReLU(3),
        ]
    for live, skeleton in zip(layers, on_meta):
        assert equine.utils._submodule_difference(skeleton, live) is None, live


def _with_extra_state(extra: Any) -> torch.nn.Module:
    """A registered embedding whose get_extra_state returns ``extra``."""

    @eq.embedding_architecture("equine.tests.extra_state_value")
    class ExtraStateValue(torch.nn.Module):
        def __init__(self, d: int = 6) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(d, 3)
            self.restored: Any = None

        def get_extra_state(self):
            return extra

        def set_extra_state(self, state) -> None:
            self.restored = state

        def forward(self, x):
            return self.lin(x)

    return ExtraStateValue()


@pytest.mark.parametrize("train, cls", TRAINERS)
@pytest.mark.parametrize(
    "state, got",
    [({"tags": {"a"}}, "got set"), ({"mean": np.float64(0.5)}, "got float64")],
    ids=["set", "numpy-scalar"],
)
def test_save_refuses_extra_state_the_loader_would_refuse(
    tmp_path, train, cls, state, got
) -> None:
    model, _ = train(_with_extra_state(state))
    path = tmp_path / "m.eq"
    with pytest.raises(ValueError, match="Extra state") as err:
        model.save(str(path))
    message = str(err.value)
    assert "ExtraStateValue" in message
    assert f"_extra_state[{next(iter(state))!r}]" in message  # names the field
    assert got in message
    assert not path.exists()


def _extra_state_reading_values() -> torch.nn.Module:
    """A registered embedding whose get_extra_state reads a tensor's value, which
    a meta tensor does not have."""

    @eq.embedding_architecture("equine.tests.extra_state_reads_values")
    class ExtraStateReadsValues(torch.nn.Module):
        def __init__(self, d: int = 6) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(d, 3)

        def get_extra_state(self):
            return {"scale": float(self.lin.weight.abs().max().item())}

        def set_extra_state(self, state) -> None:
            pass

        def forward(self, x):
            return self.lin(x)

    return ExtraStateReadsValues()


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_extra_state_that_cannot_run_on_meta_is_refused(tmp_path, train, cls) -> None:
    module = _extra_state_reading_values()
    model, _ = train(module)
    path = tmp_path / "m.eq"
    with pytest.raises(ValueError, match="could not run on the meta device") as err:
        model.save(str(path))
    assert "cannot be rebuilt from its recipe" in str(err.value)
    assert not path.exists()

    # A crafted file naming such a recipe is refused with a ValueError as well.
    genuine, _ = train(BasicEmbeddingModel(6, 3))
    genuine.save(str(path))
    _rewrite(
        str(path),
        embedding_recipe=eq.embedding_recipe(module),
        embedding_state_dict=module.state_dict(),
    )
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="could not run on the meta device") as err:
            loader(str(path))
        assert len(str(err.value)) < 1000


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_extra_state_with_tensors_round_trips(tmp_path, train, cls) -> None:
    state = {
        "stats": torch.arange(3.0),
        "order": OrderedDict(a=[1, "x"]),
        "shape": torch.Size([2, 3]),
    }
    model, X = train(_with_extra_state(state))
    path = str(tmp_path / "m.eq")
    model.save(path)
    reloaded = cls.load(path)
    restored = reloaded.embedding_model.restored
    assert torch.equal(restored["stats"], state["stats"])
    assert restored["order"] == state["order"]
    assert type(restored["shape"]) is torch.Size
    assert restored["shape"] == state["shape"]
    assert_same(model, reloaded, X)


# --- head weights against settings -----------------------------------------------------


def _refuse_construction(self, *args, **kwargs) -> None:
    raise AssertionError("the model was constructed before its settings were checked")


@pytest.mark.parametrize("e, n, c", [(3, 16, 3), (8, 4, 2), (5, 5, 1), (1, 3, 2)])
def test_expected_laplace_state_matches_the_model(e, n, c) -> None:
    """The settings-derived expectation must match what EquineGP builds exactly,
    or genuine files would be refused."""
    model = eq.EquineGP(torch.nn.Identity(), e, c, num_random_features=n)
    built = {
        k: tuple(v.shape)
        for k, v in model.model.state_dict().items()
        if "feature_extractor" not in k
    }
    expected = equine.equine_gp._expected_laplace_state(
        {"emb_out_dim": e, "num_classes": c, "num_random_features": n}
    )
    assert {k: tuple(v.shape) for k, v in expected.items()} == built
    assert all(v.is_meta for v in expected.values())


def test_expected_laplace_state_defaults_like_the_constructor() -> None:
    """Files written before num_random_features was saved were built with 1024."""
    expected = equine.equine_gp._expected_laplace_state(
        {"emb_out_dim": 3, "num_classes": 2}
    )
    assert tuple(expected["precision"].shape) == (1024, 1024)


@pytest.mark.parametrize(
    "change",
    [
        {"num_random_features": 20_000},  # precision + covariance ~3.2 GB if built
        {"emb_out_dim": 20_000},  # random_matrix ~1.6 GB
        {"num_classes": 10**6},
    ],
    ids=["num_random_features", "emb_out_dim", "num_classes"],
)
def test_gp_settings_larger_than_the_stored_head_are_refused_before_construction(
    tmp_path, monkeypatch, change
) -> None:
    model, _ = train_gp(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    ckpt["settings"].update(change)
    torch.save(ckpt, path)
    assert os.path.getsize(path) < 100_000
    monkeypatch.setattr(eq.EquineGP, "__init__", _refuse_construction)
    for loader in (eq.EquineGP.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="laplace_model_save does not match"):
            loader(path)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda sd: sd.pop("precision"), "1 parameter"),  # was silently eye(n)
        (lambda sd: sd.update(extra=torch.zeros(1)), "1 parameter"),
        (lambda sd: sd.update(precision=torch.zeros(4, 4)), "1 parameter"),
    ],
    ids=["missing", "unexpected", "wrong-shape"],
)
def test_gp_head_weights_must_match_the_settings_exactly(
    tmp_path, monkeypatch, edit, message
) -> None:
    model, _ = train_gp(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    edit(ckpt["laplace_model_save"])
    torch.save(ckpt, path)
    monkeypatch.setattr(eq.EquineGP, "__init__", _refuse_construction)
    with pytest.raises(ValueError, match="laplace_model_save does not match") as err:
        eq.EquineGP.load(path)
    assert message in str(err.value)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"settings": {"num_random_features": 0}}, "positive integer"),
        ({"settings": {"emb_out_dim": "3"}}, "positive integer"),
        ({"settings": {"num_classes": True}}, "positive integer"),
        ({"settings": {"num_random_features": 2**40}}, "too large to represent"),
        ({"settings": {"num_random_features": 2**70}}, "too large to represent"),
        ({"laplace_model_save": [1, 2]}, "laplace_model_save is malformed"),
    ],
    ids=["zero", "string", "bool", "overflow", "huge", "malformed-head"],
)
def test_gp_malformed_settings_are_refused_before_construction(
    tmp_path, monkeypatch, changes, message
) -> None:
    model, _ = train_gp(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    for key, value in changes.items():
        if isinstance(value, dict):
            ckpt[key].update(value)
        else:
            ckpt[key] = value
    torch.save(ckpt, path)
    monkeypatch.setattr(eq.EquineGP, "__init__", _refuse_construction)
    with pytest.raises(ValueError, match=message):
        eq.EquineGP.load(path)


def test_protonet_head_weights_must_match_its_identity_head(
    tmp_path, monkeypatch
) -> None:
    model, _ = train_protonet(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    _rewrite(path, model_head_save={"weight": torch.zeros(3, 3)})
    monkeypatch.setattr(eq.EquineProtonet, "__init__", _refuse_construction)
    for loader in (eq.EquineProtonet.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="model_head_save does not match"):
            loader(path)


@pytest.mark.parametrize("emb_out_dim", [10**9, 4, 2, 0])
def test_protonet_emb_out_dim_is_bounded_by_the_embedding_output(
    tmp_path, monkeypatch, emb_out_dim
) -> None:
    """update_support allocates torch.ones(emb_out_dim); the embedding here
    outputs 3 features."""
    model, X = train_protonet(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    ckpt["settings"]["emb_out_dim"] = emb_out_dim
    torch.save(ckpt, path)

    def refuse(self, support) -> None:
        raise AssertionError("update_support ran before emb_out_dim was checked")

    with monkeypatch.context() as patch:
        patch.setattr(equine.equine_protonet.Protonet, "update_support", refuse)
        for loader in (eq.EquineProtonet.load, eq.load_equine_model):
            with pytest.raises(ValueError, match="emb_out_dim"):
                loader(path)

    ckpt["settings"]["emb_out_dim"] = 3
    torch.save(ckpt, path)
    assert_same(model, eq.EquineProtonet.load(path), X)


@pytest.mark.parametrize("emb_out_dim", [True, "3", 3.0, None], ids=repr)
def test_protonet_emb_out_dim_must_be_an_integer(
    tmp_path, monkeypatch, emb_out_dim
) -> None:
    """bool passes the constructor's int check but then breaks update_support."""
    model, _ = train_protonet(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    ckpt["settings"]["emb_out_dim"] = emb_out_dim
    torch.save(ckpt, path)
    monkeypatch.setattr(eq.EquineProtonet, "__init__", _refuse_construction)
    for loader in (eq.EquineProtonet.load, eq.load_equine_model):
        with pytest.raises(ValueError, match=r"settings\['emb_out_dim'\] must be"):
            loader(path)


# --- settings keys ------------------------------------------------------------------


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_unknown_settings_keys_are_refused(tmp_path, train, cls) -> None:
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    ckpt["settings"].update({"k" * 10_000: 1, "bogus": 2, "other": 3, "more": 4})
    torch.save(ckpt, path)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="does not accept") as err:
            loader(path)
        message = str(err.value)
        assert "4 key(s)" in message
        assert cls.__name__ in message
        assert len(message) < 600  # the 10,000-character key is truncated


@pytest.mark.parametrize("train, cls", TRAINERS)
@pytest.mark.parametrize(
    "settings", [None, [1, 2], {1: 2}], ids=["none", "list", "int-key"]
)
def test_malformed_settings_are_refused(tmp_path, train, cls, settings) -> None:
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    _rewrite(path, settings=settings)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="settings are malformed"):
            loader(path)


@pytest.mark.parametrize("use_temperature", [True, False])
def test_gp_use_temperature_from_0_1_5_is_dropped_with_a_warning(
    tmp_path, use_temperature
) -> None:
    """EquineGP.__init__ lost use_temperature in 0.1.6; files saved by 0.1.5
    still carry it and must load (and migrate) with unchanged predictions."""
    model, X = train_gp(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    without_key = eq.EquineGP.load(path)
    ckpt = torch.load(path, weights_only=True)
    ckpt["settings"]["use_temperature"] = use_temperature
    old = str(tmp_path / "old.eq")
    torch.save(ckpt, old)

    for loader in (eq.EquineGP.load, eq.load_equine_model):
        with pytest.warns(UserWarning, match="use_temperature.*since 0.1.6") as record:
            reloaded = loader(old)
        assert_same(without_key, reloaded, X)
        dropped = [w for w in record if "use_temperature" in str(w.message)]
        assert len(dropped) == 1
        assert dropped[0].filename == __file__, "warning must point at the caller"

    resaved = str(tmp_path / "resaved.eq")
    reloaded.save(resaved)
    assert "use_temperature" not in torch.load(resaved, weights_only=True)["settings"]


_HUGE = "x" * 100_000


@pytest.mark.parametrize(
    "train, cls, settings, message",
    [
        pytest.param(
            train_protonet,
            eq.EquineProtonet,
            {"cov_type": _HUGE},
            "cov_type",
            id="protonet-cov_type",
        ),
        pytest.param(
            train_protonet,
            eq.EquineProtonet,
            {"cov_type": 3},
            "cov_type",
            id="protonet-cov_type-int",
        ),
        pytest.param(
            train_protonet,
            eq.EquineProtonet,
            {"device": _HUGE},
            "device",
            id="protonet-device",
        ),
        pytest.param(
            train_gp, eq.EquineGP, {"device": _HUGE}, "device", id="gp-device"
        ),
        pytest.param(
            train_gp, eq.EquineGP, {"device": 0}, "device", id="gp-device-int"
        ),
    ],
)
def test_file_supplied_settings_give_bounded_errors(
    tmp_path, monkeypatch, train, cls, settings, message
) -> None:
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    ckpt["settings"].update(settings)
    torch.save(ckpt, path)
    monkeypatch.setattr(cls, "__init__", _refuse_construction)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match=message) as err:
            loader(path)
        assert len(str(err.value)) < 1000


@pytest.mark.parametrize(
    "train_summary, message",
    [
        ({"modelType": _HUGE}, "Unknown model type"),
        ([1, 2], "malformed train_summary"),
        ({"modelType": 5}, "malformed train_summary"),
        ({}, "malformed train_summary"),
        (..., "malformed train_summary"),
    ],
    ids=["unknown-type", "list", "non-str-type", "no-type", "missing"],
)
def test_load_equine_model_bounds_train_summary_errors(
    tmp_path, train_summary, message
) -> None:
    model, _ = train_protonet(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    _rewrite(path, train_summary=train_summary)
    with pytest.raises(ValueError, match=message) as err:
        eq.load_equine_model(path)
    assert len(str(err.value)) < 1000


@pytest.mark.parametrize(
    "train, cls, missing",
    [
        pytest.param(
            train_protonet,
            eq.EquineProtonet,
            ["settings", "outlier_kde", "model_head_save"],
            id="protonet",
        ),
        pytest.param(
            train_gp,
            eq.EquineGP,
            ["support", "laplace_model_save", "num_data", "train_batch_size"],
            id="gp",
        ),
    ],
)
def test_missing_checkpoint_entries_are_listed(tmp_path, train, cls, missing) -> None:
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    _rewrite(path, **dict.fromkeys(missing, ...))
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="missing") as err:
            loader(path)
        assert all(repr(key) in str(err.value) for key in missing)


def test_protonet_support_the_embedding_cannot_read_is_a_value_error(
    tmp_path,
) -> None:
    """The width probe runs the embedding on a stored support row."""
    model, _ = train_protonet(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    ckpt["support"] = {label: torch.zeros(10, 7) for label in ckpt["support"]}
    torch.save(ckpt, path)
    for loader in (eq.EquineProtonet.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="support") as err:
            loader(path)
        assert len(str(err.value)) < 1000


@pytest.mark.parametrize("train, cls", TRAINERS)
@pytest.mark.parametrize(
    "support",
    [
        [torch.zeros(10, 6)],
        {"0": torch.zeros(10, 6)},
        {0: [1.0, 2.0]},
        {(0, 1): torch.zeros(10, 6)},
        {True: torch.zeros(10, 6)},
        {1.5: torch.zeros(10, 6)},
        {float("inf"): torch.zeros(10, 6)},
    ],
    ids=["list", "str-key", "not-a-tensor", "tuple-key", "bool-key", "fraction", "inf"],
)
def test_malformed_support_is_refused(tmp_path, train, cls, support) -> None:
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    _rewrite(path, support=support)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="support is malformed"):
            loader(path)


def test_support_labels_that_collide_are_refused() -> None:
    rows = torch.zeros(2, 6)
    with pytest.raises(ValueError, match="support is malformed"):
        equine.utils._support_from_file({0: rows, torch.tensor(0): rows.clone()})


def _kde_state(n: int = 20) -> dict:
    torch.manual_seed(0)
    return {"dataset": torch.rand(1, n, dtype=torch.float64), "bw_factor": 0.5}


@pytest.mark.parametrize(
    "outlier_kde",
    [
        [1, 2],
        {"0": _kde_state()},
        {True: _kde_state()},
        {0.5: _kde_state()},
        {0: torch.zeros(3)},
        {0: {"bw_factor": 0.5}},
        {0: {**_kde_state(), "extra": 1}},
        {0: {"dataset": [0.1, 0.2, 0.3], "bw_factor": 0.5}},
        {0: {"dataset": torch.rand(1, 2, 3), "bw_factor": 0.5}},
        {0: {"dataset": torch.rand(2, 10), "bw_factor": 0.5}},
        {0: {"dataset": torch.rand(1, 10), "bw_factor": True}},
        {0: {"dataset": torch.rand(1, 10), "bw_factor": "0.5"}},
        {0: {"dataset": torch.rand(1, 10), "bw_factor": -1.0}},
        {0: {"dataset": torch.rand(1, 1), "bw_factor": 0.5}},
        {0: {"dataset": torch.rand(1, 10), "bw_factor": _HUGE}},
    ],
    ids=[
        "list",
        "str-label",
        "bool-label",
        "fraction-label",
        "not-a-dict",
        "no-dataset",
        "extra-key",
        "dataset-not-a-tensor",
        "dataset-3d",
        "dataset-2-rows",
        "bw-bool",
        "bw-str",
        "bw-negative",
        "kde-fails",
        "bw-huge-str",
    ],
)
def test_malformed_outlier_kde_is_refused(tmp_path, outlier_kde) -> None:
    model, _ = train_protonet(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    _rewrite(path, outlier_kde=outlier_kde)
    for loader in (eq.EquineProtonet.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="outlier_kde") as err:
            loader(path)
        assert len(str(err.value)) < 1000


def test_outlier_kde_accepts_a_1d_dataset(tmp_path) -> None:
    model, _ = train_protonet(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    for state in ckpt["outlier_kde"].values():
        state["dataset"] = state["dataset"].reshape(-1).clone()
    torch.save(ckpt, path)
    eq.EquineProtonet.load(path)


def test_outlier_kde_objects_need_the_legacy_opt_in() -> None:
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(np.random.default_rng(0).random(20))
    with pytest.raises(ValueError, match="outlier_kde"):
        equine.utils._outlier_kde_from_file({0: kde}, legacy=False)
    assert equine.utils._outlier_kde_from_file({0: kde}, legacy=True)[0] is kde
    with pytest.raises(ValueError, match="outlier_kde"):
        equine.utils._outlier_kde_from_file(
            {0: _kde_state(), torch.tensor(0): _kde_state()}, legacy=False
        )


@pytest.mark.parametrize(
    "train, cls, settings, message",
    [
        pytest.param(
            train_protonet,
            eq.EquineProtonet,
            {"init_temperature": _HUGE},
            "init_temperature",
            id="protonet-temperature-str",
        ),
        pytest.param(
            train_gp,
            eq.EquineGP,
            {"init_temperature": True},
            "init_temperature",
            id="gp-temperature-bool",
        ),
        pytest.param(
            train_protonet,
            eq.EquineProtonet,
            {"relative_mahal": 1},
            "relative_mahal",
            id="relative_mahal-int",
        ),
        pytest.param(
            train_protonet,
            eq.EquineProtonet,
            {"use_temperature": "yes"},
            "use_temperature",
            id="use_temperature-str",
        ),
        pytest.param(
            train_gp,
            eq.EquineGP,
            {"feature_names": [1, 2]},
            "feature_names",
            id="feature_names-ints",
        ),
        pytest.param(
            train_protonet,
            eq.EquineProtonet,
            {"label_names": _HUGE},
            "label_names",
            id="label_names-str",
        ),
        pytest.param(
            train_protonet,
            eq.EquineProtonet,
            {"device": "meta"},
            "device",
            id="protonet-meta",
        ),
        pytest.param(train_gp, eq.EquineGP, {"device": "meta"}, "device", id="gp-meta"),
    ],
)
def test_file_supplied_setting_values_are_type_checked(
    tmp_path, monkeypatch, train, cls, settings, message
) -> None:
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    ckpt["settings"].update(settings)
    torch.save(ckpt, path)
    monkeypatch.setattr(cls, "__init__", _refuse_construction)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match=message) as err:
            loader(path)
        assert len(str(err.value)) < 1000


@pytest.mark.parametrize("train, cls", TRAINERS)
@pytest.mark.parametrize("key", ["feature_names", "label_names"])
def test_top_level_names_are_type_checked(tmp_path, train, cls, key) -> None:
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    _rewrite(path, **{key: {"a": 1}})
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match=key):
            loader(path)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
@pytest.mark.parametrize("train, cls", TRAINERS)
def test_a_device_this_machine_lacks_is_refused(tmp_path, train, cls) -> None:
    model, X = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    _rewrite(
        path,
        settings={**torch.load(path, weights_only=True)["settings"], "device": "cuda"},
    )
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="not available on this machine"):
            loader(path)
    if cls is eq.EquineProtonet:  # the caller's device= replaces the file's
        assert_same(model, eq.EquineProtonet.load(path, device="cpu"), X)


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_constructor_failures_are_bounded_value_errors(
    tmp_path, monkeypatch, train, cls
) -> None:
    model, _ = train(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)

    def explode(self, *args, **kwargs) -> None:
        raise RuntimeError("x" * 100_000)

    monkeypatch.setattr(cls, "__init__", explode)
    for loader in (cls.load, eq.load_equine_model):
        with pytest.raises(ValueError, match=cls.__name__) as err:
            loader(path)
        assert len(str(err.value)) < 1000


def test_gp_support_the_embedding_cannot_read_is_a_value_error(tmp_path) -> None:
    model, _ = train_gp(BasicEmbeddingModel(6, 3))
    path = str(tmp_path / "m.eq")
    model.save(path)
    ckpt = torch.load(path, weights_only=True)
    ckpt["support"] = {label: torch.zeros(10, 7) for label in ckpt["support"]}
    torch.save(ckpt, path)
    for loader in (eq.EquineGP.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="support") as err:
            loader(path)
        assert len(str(err.value)) < 1000


# --- unknown recipe -----------------------------------------------------------------


@pytest.mark.parametrize("train, cls", TRAINERS)
def test_unknown_recipe_is_actionable_in_a_fresh_interpreter(
    tmp_path, train, cls
) -> None:
    @eq.embedding_architecture("equine.tests.only_here")
    class OnlyHere(torch.nn.Module):
        def __init__(self, d: int = 6) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(d, 3)

        def forward(self, x):
            return self.lin(x)

    model, _ = train(OnlyHere())
    path = str(tmp_path / "m.eq")
    model.save(path)
    code = (
        "import sys, equine as eq\n"
        "try:\n"
        f"    eq.load_equine_model({path!r})\n"
        "except ValueError as e:\n"
        "    print(e); sys.exit(3)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 3, proc.stderr
    assert "'equine.tests.only_here'" in proc.stdout
    assert "not registered" in proc.stdout
    assert "embedding_model=" in proc.stdout
