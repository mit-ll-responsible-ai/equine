# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Device-parametrized behaviour tests for both model classes.

Tests training, prediction, ``update_support``, saving and loading, moving a
model with ``to()``, input dtypes and the metric helpers, and checks that every
tensor a model stores is on its device. Every test runs on CPU and on whichever
accelerator this machine has (see ``conftest.available_devices()``); on CI
(Ubuntu, no accelerator) only the CPU case is collected.
"""

import pytest
import torch
import torchmetrics
from conftest import (
    BasicEmbeddingModel,
    assert_on_device,
    assert_valid_prediction,
    available_devices,
    stored_tensors,
)
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq
import equine.utils
from equine.equine_gp import _inverse_via_cholesky

pytestmark = pytest.mark.device


def devices() -> list:
    """Parametrization over ``available_devices()``; every case but the CPU one carries the ``accelerator`` marker."""
    return [
        pytest.param(device, marks=[] if device == "cpu" else [pytest.mark.accelerator])
        for device in available_devices()
    ]


def _protonet(device: str, use_temperature: bool = False):
    """Short float32 training on ``device``."""
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        use_temperature=use_temperature,
        device=device,
    )
    model.train_model(
        dataset,
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    return model, x, y


def _gp(device: str, **train_kwargs):
    """Short float32 training on ``device``."""
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        CLASSES,
        num_random_features=16,
        device=device,
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=2,
        batch_size=32,
        **train_kwargs,
    )
    return model, x, y


_BUILDERS = pytest.mark.parametrize("build", [_protonet, _gp], ids=["protonet", "gp"])


def _assert_same_predictions(
    expected: eq.EquineOutput, actual: eq.EquineOutput
) -> None:
    torch.testing.assert_close(
        actual.classes.cpu(), expected.classes.cpu(), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        actual.ood_scores.cpu(), expected.ood_scores.cpu(), atol=1e-5, rtol=0
    )


# --- EquineProtonet -----------------------------------------------------------


@pytest.mark.parametrize("device", devices())
def test_protonet_trains_and_predicts(device):
    model, x, _ = _protonet(device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_protonet_with_temperature_predicts(device):
    model, x, _ = _protonet(device, use_temperature=True)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_protonet_update_support(device):
    model, x, y = _protonet(device)
    model.update_support(x, y.float(), 0.5)
    assert_on_device(model, device)  # the new support and its embeddings too
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_protonet_save_load_round_trip(device, tmp_path):
    model, x, _ = _protonet(device)
    before = model.predict(x[:5])
    path = str(tmp_path / "protonet.eq")
    model.save(path)

    generic = eq.load_equine_model(path)  # lands on the saved device
    assert_on_device(generic, device)
    _assert_same_predictions(before, generic.predict(x[:5]))

    explicit = eq.EquineProtonet.load(path, device)
    _assert_same_predictions(before, explicit.predict(x[:5]))


# --- both classes -------------------------------------------------------------


@_BUILDERS
@pytest.mark.parametrize("device", devices())
def test_stored_tensors_on_device(build, device):
    model, _, _ = build(device)
    assert_on_device(model, device)


# --- EquineGP -----------------------------------------------------------------


@pytest.mark.parametrize("device", devices())
def test_gp_trains(device):
    model, _, _ = _gp(device)
    assert model.model.precision.device.type == torch.device(device).type


@pytest.mark.parametrize("device", devices())
def test_gp_predicts(device):
    model, x, _ = _gp(device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_gp_update_support(device):
    model, x, y = _gp(device)
    model.update_support(x, y.long(), 10)
    assert set(model.support) == set(range(CLASSES))
    assert model.prototypes.shape[0] == CLASSES  # one prototype per class
    assert_on_device(model, device)  # the new support and its embeddings too


@pytest.mark.parametrize("device", devices())
def test_gp_seen_count_mirrors_the_seen_data_buffer(device, tmp_path):
    """The asserts in the Laplace forward read a Python int, not the device buffer.

    ``seen_data`` is a buffer (saved in the state_dict and the file) and ``_seen_count``
    follows it through training, a reset and a load.
    """
    model, x, _ = _gp(device)
    dataset, _, _ = separable_dataset()
    assert isinstance(model.model._seen_count, int)
    assert model.model._seen_count == int(model.model.seen_data) == len(dataset)

    path = str(tmp_path / "gp.eq")
    model.save(path)
    for loaded in (eq.EquineGP.load(path, device), eq.load_equine_model(path)):
        assert loaded.model._seen_count == int(loaded.model.seen_data) == len(dataset)
        assert_valid_prediction(loaded.predict(x[:5]), 5, CLASSES)

    model.model.reset_precision_matrix()
    assert model.model._seen_count == int(model.model.seen_data) == 0


@pytest.mark.parametrize("device", devices())
def test_gp_seen_count_follows_load_state_dict(device):
    """Weights copied through the public ``nn.Module.load_state_dict`` bring the counter along.

    ``seen_data`` arrives with the state_dict, and the Python mirror the forward asserts
    read follows it.
    """
    trained, x, _ = _gp(device)
    dataset, _, _ = separable_dataset()
    fresh = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        CLASSES,
        num_random_features=16,
        device=device,
    )
    fresh.model.set_training_params(len(dataset), 32)
    fresh.load_state_dict(trained.state_dict())
    assert int(fresh.model.seen_data) == len(dataset)
    assert fresh.model._seen_count == len(dataset)
    fresh.eval()
    _assert_same_predictions(trained.predict(x[:5]), fresh.predict(x[:5]))


@pytest.mark.parametrize("device", devices())
def test_gp_vis_support_training(device):
    model, _, _ = _gp(device, vis_support=True, support_size=10)
    assert set(model.support) == set(range(CLASSES))


@pytest.mark.parametrize("device", devices())
def test_gp_save_load_round_trip(device, tmp_path):
    model, x, _ = _gp(device)
    before = model.predict(x[:5])
    path = str(tmp_path / "gp.eq")
    model.save(path)

    generic = eq.load_equine_model(path)  # lands on the saved device
    # Every stored tensor is on the device, seen_data included.
    assert_on_device(generic, device)
    _assert_same_predictions(before, generic.predict(x[:5]))


@pytest.mark.parametrize("device", devices())
def test_gp_deprecated_device_type_setter_moves_the_module(device):
    """Assigning the deprecated ``device_type`` moves the model, support included, along with ``device``."""
    model, x, _ = _gp("cpu", vis_support=True, support_size=10)
    with pytest.warns(DeprecationWarning, match="device_type"):
        model.device_type = device
    assert model.device == device
    assert_on_device(model, device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_gp_load_onto_device(device, tmp_path):
    model, x, _ = _gp("cpu")
    before = model.predict(x[:5])
    path = str(tmp_path / "gp.eq")
    model.save(path)

    loaded = eq.EquineGP.load(path, device)
    assert loaded.device == device
    assert_on_device(loaded, device)
    _assert_same_predictions(before, loaded.predict(x[:5]))

    generic = eq.load_equine_model(path, device=device)
    assert generic.device == device
    assert_on_device(generic, device)
    _assert_same_predictions(before, generic.predict(x[:5]))


_WITH_SUPPORT = pytest.mark.parametrize(
    "build",
    [
        lambda device: _protonet(device, use_temperature=True),
        lambda device: _gp(device, vis_support=True, support_size=10),
    ],
    ids=["protonet", "gp"],
)


@_WITH_SUPPORT
@pytest.mark.parametrize("device", devices())
def test_to_moves_every_stored_tensor(build, device):
    """``model.to(target)`` alone moves every stored tensor and updates ``device``.

    From the CPU to ``device`` and back: the support, its embeddings, the
    prototypes and the covariances move with the parameters, and the
    predictions do not change.
    """
    for source, target in (("cpu", device), (device, "cpu")):
        model, x, _ = build(source)
        before = model.predict(x[:5])
        model.to(target)
        assert_on_device(model, target)
        named = [model.device]
        if isinstance(model, eq.EquineProtonet):
            named.append(model.model.device)
        assert {torch.device(d).type for d in named} == {torch.device(target).type}
        after = model.predict(x[:5])
        assert after.classes.device.type == torch.device(target).type
        torch.testing.assert_close(
            after.classes.cpu(), before.classes.cpu(), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            after.ood_scores.cpu(), before.ood_scores.cpu(), atol=1e-4, rtol=0
        )
        # cdist checks that the embeddings and the prototypes share a device.
        torch.cdist(after.embeddings, model.get_prototypes())


@_WITH_SUPPORT
def test_to_meta_moves_every_stored_tensor(build):
    """The same move onto ``meta`` (no data)."""
    model, _, _ = build("cpu")
    model.to("meta")
    assert_on_device(model, "meta")
    assert model.device == "meta"
    if isinstance(model, eq.EquineProtonet):
        assert model.model.device == "meta"


@_WITH_SUPPORT
def test_dtype_casts_reach_every_stored_tensor(build):
    """``double()`` casts every stored floating tensor and keeps ``device``.

    The support, its embeddings, the prototypes and the covariances become
    float64 with the parameters.
    """
    model, x, _ = build("cpu")
    model.double()
    floating = {
        name: t.dtype
        for name, t in stored_tensors(model).items()
        if t.is_floating_point()
    }
    assert set(floating.values()) == {torch.float64}, floating
    assert model.device == "cpu"
    out = model.predict(x[:5])
    assert out.classes.dtype == torch.float64
    assert_valid_prediction(out, 5, CLASSES)


@_BUILDERS
@pytest.mark.parametrize("device", devices())
def test_loaders_read_the_file_onto_the_cpu(build, device, tmp_path, monkeypatch):
    """With ``device=``, every loader reads the file onto the CPU and the model moves to ``device``."""
    model, x, _ = build("cpu")
    before = model.predict(x[:5])
    path = str(tmp_path / "m.eq")
    model.save(path)
    read_onto: list[str] = []
    copy_off_the_file = equine.utils._copy_off_the_file

    def spy(tensors, map_location):
        read_onto.append(torch.device(map_location or "cpu").type)
        copy_off_the_file(tensors, map_location)

    monkeypatch.setattr(equine.utils, "_copy_off_the_file", spy)
    for loader in (type(model).load, eq.load_equine_model):
        loaded = loader(path, device)
        assert_on_device(loaded, device)
        _assert_same_predictions(before, loaded.predict(x[:5]))
    assert read_onto == ["cpu", "cpu"]


@pytest.mark.parametrize("device", devices())
def test_metric_helpers_compute_where_the_predictions_are(device):
    """The public metric helpers take predictions on any device, with labels on the CPU or on that device."""
    model, x, y = _gp(device)
    out = model.predict(x)
    on_cpu = eq.EquineOutput(
        classes=out.classes.cpu(),
        ood_scores=out.ood_scores.cpu(),
        embeddings=out.embeddings.cpu(),
    )
    expected = eq.generate_model_metrics(on_cpu, y)
    for labels in (y, y.to(device)):
        metrics = eq.generate_model_metrics(out, labels)
        assert metrics["confusionMatrix"] == expected["confusionMatrix"]
        for key in ("accuracy", "microF1Score"):
            assert metrics[key].device.type == torch.device(device).type
            assert float(metrics[key]) == pytest.approx(float(expected[key]))
        for key in ("brierScore", "brierSkillScore", "expectedCalibrationError"):
            assert metrics[key] == pytest.approx(expected[key], abs=1e-6)
        for helper, key in (
            (eq.brier_score, "brierScore"),
            (eq.brier_skill_score, "brierSkillScore"),
            (eq.expected_calibration_error, "expectedCalibrationError"),
        ):
            assert helper(out.classes, labels) == pytest.approx(expected[key], abs=1e-6)


@pytest.mark.parametrize("device", devices())
def test_gp_inverse_helper_stays_on_the_device(device):
    """The GP's covariance inversion runs where its input is and returns there."""
    a = torch.tensor([[4.0, 1.0], [1.0, 3.0]], device=device)
    inverse = _inverse_via_cholesky(a)
    assert inverse.device == a.device
    torch.testing.assert_close((a @ inverse).cpu(), torch.eye(2))
    with pytest.raises(AssertionError, match="Precision matrix inversion failed"):
        _inverse_via_cholesky(-a)


@pytest.mark.parametrize("device", devices())
def test_gp_linear_algebra_runs_on_the_models_device(device, monkeypatch):
    """Training and predicting run the GP's inversion and entropy on the model's device, with no copy elsewhere."""
    seen: set[tuple[str, str]] = set()

    def recording(real, name):
        def op(x, *args, **kwargs):
            seen.add((name, x.device.type))
            return real(x, *args, **kwargs)

        return op

    for module, name in (
        (torch.linalg, "cholesky_ex"),
        (torch, "cholesky_inverse"),
        (torch.special, "entr"),
    ):
        monkeypatch.setattr(module, name, recording(getattr(module, name), name))
    model, x, _ = _gp(device)
    model.predict(x[:5])
    device_type = torch.device(device).type
    assert seen == {
        ("cholesky_ex", device_type),
        ("cholesky_inverse", device_type),
        ("entr", device_type),
    }


def test_gp_inverse_on_the_cpu_is_the_plain_cholesky_inverse():
    """On the CPU the helper is exactly ``cholesky_ex`` then ``cholesky_inverse``."""
    torch.manual_seed(0)
    m = torch.rand(5, 5, dtype=torch.float64)
    a = m @ m.T + torch.eye(5, dtype=torch.float64)
    expected = torch.cholesky_inverse(torch.linalg.cholesky_ex(a)[0])
    assert torch.equal(_inverse_via_cholesky(a), expected)


# --- input dtype --------------------------------------------------------------


@_BUILDERS
@pytest.mark.parametrize("device", devices())
def test_predict_accepts_float64_input(build, device):
    """Inputs are cast to the embedding's parameter dtype at the model boundary."""
    model, x, _ = build(device)
    assert_valid_prediction(model.predict(x[:5].double()), 5, CLASSES)


@_BUILDERS
@pytest.mark.parametrize("device", devices())
def test_update_support_accepts_float64_input(build, device):
    """Support goes through the model boundary: moved to the device and cast to the embedding dtype."""
    model, x, y = build(device)
    if isinstance(model, eq.EquineProtonet):
        model.update_support(x.double(), y.float(), 0.5)
        support = model.model.support
    else:
        model.update_support(x.double(), y.long(), 10)
        support = model.support
    assert {t.dtype for t in support.values()} == {torch.float32}
    assert_on_device(model, device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_gp_train_model_accepts_float64_dataset(device):
    """The GP training, validation and calibration loops cast batches like predict does."""
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    dataset = torch.utils.data.TensorDataset(x.double(), y)
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        CLASSES,
        num_random_features=16,
        device=device,
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=2,
        batch_size=32,
        validation_dataset=dataset,
        val_metrics=[
            torchmetrics.classification.MulticlassAccuracy(CLASSES).to(device)
        ],
    )
    model.calibrate_model(dataset)
    assert_on_device(model, device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_protonet_train_model_accepts_float64_dataset(device):
    """Episodes, support and the calibration split all cross the model boundary."""
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    dataset = torch.utils.data.TensorDataset(x.double(), y)
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, device=device
    )
    model.train_model(
        dataset,
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    assert_on_device(model, device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_protonet_train_model_returns_the_callers_calibration_split(device):
    """``calib_x``/``calib_y`` come back as the caller split them.

    The copies moved to the device and cast to the model dtype are internal to
    ``train_model``; the returned tensors stay on the CPU in the dataset's dtypes.
    """
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = eq.EquineProtonet(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, device=device
    )
    result = model.train_model(
        torch.utils.data.TensorDataset(x.double(), y),
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    assert set(result) == {"train_summary", "calib_x", "calib_y"}
    assert result["calib_x"].device.type == result["calib_y"].device.type == "cpu"
    assert result["calib_x"].dtype == torch.float64
    assert result["calib_y"].dtype == y.dtype
    assert result["calib_x"].shape == (len(result["calib_y"]), FEATURES)


@pytest.mark.parametrize("device", devices())
def test_protonet_identity_embedding(device):
    """A Protonet over raw features: emb_out_dim is the input width.

    Without a parameter anywhere (the head is an Identity too), the support is set
    directly; inputs are moved but have no dtype to be cast to.
    """
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = eq.EquineProtonet(torch.nn.Identity(), FEATURES, device=device)
    assert model.model.emb_out_dim == x.shape[1]
    model.eval()
    model.update_support(x, y.float(), 0.5)
    assert_on_device(model, device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_gp_identity_embedding_casts_to_the_head_dtype(device):
    """With a parameter-less embedding the Laplace head's parameters set the input
    dtype; float64 input works."""
    torch.manual_seed(0)
    dataset, x, _ = separable_dataset()
    model = eq.EquineGP(
        torch.nn.Identity(), FEATURES, CLASSES, num_random_features=16, device=device
    )
    assert model.num_deep_features == x.shape[1]
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=2,
        batch_size=32,
    )
    assert_on_device(model, device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)
    assert_valid_prediction(model.predict(x[:5].double()), 5, CLASSES)


class _TokenEmbedding(torch.nn.Module):
    """Embedding model whose input is integer token ids, mean-pooled then projected.

    ``inputs`` records the dtype and device type of every input it receives.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(num_embeddings=50, embedding_dim=8)
        self.linear = torch.nn.Linear(8, CLASSES)
        self.inputs: set[tuple[torch.dtype, str]] = set()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.add((x.dtype, x.device.type))
        return self.linear(self.embed(x).mean(dim=1))


@pytest.mark.parametrize("device", devices())
def test_integer_inputs_are_moved_but_not_cast(device):
    """The dtype cast applies to floating inputs only: nn.Embedding indices stay integer."""
    torch.manual_seed(0)
    y = torch.arange(90) % CLASSES
    # class-specific ids, all below the 50 rows of the embedding table
    x = torch.randint(0, 47, (90, 4)) // CLASSES * CLASSES + y[:, None]
    embedding = _TokenEmbedding()
    model = eq.EquineProtonet(embedding, CLASSES, device=device)
    model.train_model(
        torch.utils.data.TensorDataset(x, y),
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)
    assert embedding.inputs == {(torch.int64, torch.device(device).type)}


# --- device attribute ---------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(
            lambda: eq.EquineProtonet(
                BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, device="cpu"
            ),
            id="protonet",
        ),
        pytest.param(
            lambda: eq.EquineGP(
                BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, device="cpu"
            ),
            id="gp",
        ),
    ],
)
def test_device_attribute_is_a_string(make):
    assert isinstance(make().device, str)


@_BUILDERS
@pytest.mark.parametrize(
    "move",
    [
        pytest.param(lambda model: model.cpu(), id="cpu"),
        pytest.param(lambda model: model.float(), id="float"),
        pytest.param(lambda model: model.to("cpu"), id="to"),
    ],
)
def test_moving_a_cpu_model_that_records_cuda(build, move):
    """``cpu()``, ``float()`` and ``to("cpu")`` work on a CPU model whose ``device`` is ``"cuda"``, and set it to ``"cpu"``.

    A model pickled on a CUDA host and unpickled onto the CPU is in that state; the test
    sets ``device`` directly.
    """
    model, x, _ = build("cpu")
    before = model.predict(x[:5])
    holders = [model, model.model] if isinstance(model, eq.EquineProtonet) else [model]
    for holder in holders:
        holder.device = "cuda"
    move(model)
    assert [holder.device for holder in holders] == ["cpu"] * len(holders)
    assert_on_device(model, "cpu")
    _assert_same_predictions(before, model.predict(x[:5]))
