# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Device-parametrized behaviour tests for both model classes (#188).

Every test runs on CPU and on whichever accelerator this machine has (see
``conftest.available_devices()``); on CI (Ubuntu, no accelerator) only the
CPU case is collected. A case that fails on an accelerator is marked with a
*strict* xfail through ``devices(xfail=...)``, and the mark is removed when
the path is fixed. A strict xfail that starts passing fails the run, so a
mark cannot go stale.
"""

import pytest
import torch
import torchmetrics
from conftest import (
    BasicEmbeddingModel,
    assert_on_device,
    assert_valid_prediction,
    available_devices,
)
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq
from equine.equine_gp import _entr, _inverse_via_cholesky

pytestmark = pytest.mark.device


def devices(
    *,
    xfail: str | None = None,
    raises: type[BaseException] | tuple[type[BaseException], ...] = RuntimeError,
    every_device: bool = False,
) -> list:
    """Parametrization over ``available_devices()``.

    When ``xfail`` is given, the accelerator case (every case, with
    ``every_device=True``) gets a strict xfail with that reason and the
    exception type(s) observed today in ``raises``, so every test states next
    to its signature whether (and why) it fails today. Otherwise the CPU case
    is plain; it is never marked for an accelerator-only failure.
    """
    params = []
    for device in available_devices():
        marks = []
        if xfail is not None and (every_device or device != "cpu"):
            marks.append(pytest.mark.xfail(strict=True, raises=raises, reason=xfail))
        params.append(pytest.param(device, marks=marks))
    return params


def _protonet(device: str, use_temperature: bool = False):
    """Short float32 training on ``device``; the float64 ``golden_data.trained_protonet`` must stay byte-stable and takes no device."""
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
    """Short float32 training on ``device``; the float64 ``golden_data.trained_gp`` must stay byte-stable and takes no device."""
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

    ``seen_data`` stays a buffer (state_dict and file format unchanged) and
    ``_seen_count`` follows it through training, a reset and a load, so no
    forward has to synchronize with the accelerator to read the counter.
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

    ``seen_data`` arrives with the state_dict; the Python mirror the forward
    asserts read must follow it, or a model rebuilt this way refuses to
    predict ("Not seen sufficient data for precision matrix").
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
    # seen_data too: load_checkpoint returns CPU tensors without device=, and
    # predictions do not show where that buffer landed.
    assert_on_device(generic, device)
    _assert_same_predictions(before, generic.predict(x[:5]))


@pytest.mark.parametrize("device", devices())
def test_gp_deprecated_device_type_setter_moves_the_module(device):
    """Assigning the deprecated ``device_type`` moves the module along with ``device``."""
    model, x, _ = _gp("cpu")
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


@pytest.mark.parametrize("device", devices())
def test_gp_inverse_and_entropy_helpers_return_to_the_device(device):
    """The GP's covariance inversion and entropy work on every device and stay there.

    On MPS both run on a CPU copy (torch 2.6 has no MPS kernel for
    ``linalg.cholesky_ex`` or ``special.entr``, and 2.6 to 2.9 none for
    ``cholesky_inverse``); the CPU and CUDA run the plain ops, which the
    golden tests pin.
    """
    a = torch.tensor([[4.0, 1.0], [1.0, 3.0]], device=device)
    inverse = _inverse_via_cholesky(a)
    assert inverse.device == a.device
    torch.testing.assert_close((a @ inverse).cpu(), torch.eye(2))
    with pytest.raises(AssertionError, match="Precision matrix inversion failed"):
        _inverse_via_cholesky(-a)

    p = torch.tensor([0.0, 0.25, 0.75], device=device)
    entropy = _entr(p)
    assert entropy.device == p.device
    torch.testing.assert_close(entropy.cpu(), torch.special.entr(p.cpu()))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_gp_on_mps_survives_the_kernels_torch_2_6_lacks(monkeypatch):
    """Simulates torch 2.6 on current torch: no MPS ``linalg.cholesky_ex`` or ``special.entr``.

    Each op raises NotImplementedError for an MPS tensor, as torch 2.6 does,
    and runs normally otherwise; a GP trained and queried on MPS must still
    work, which it does only because ``_inverse_via_cholesky`` and ``_entr``
    hand those ops CPU copies.
    """
    called_on: list[str] = []

    def without_mps_kernel(real, name):
        def op(x, *args, **kwargs):
            if x.device.type == "mps":
                raise NotImplementedError(f"{name} has no MPS kernel (torch 2.6)")
            called_on.append(f"{name}:{x.device.type}")
            return real(x, *args, **kwargs)

        return op

    monkeypatch.setattr(
        torch.linalg,
        "cholesky_ex",
        without_mps_kernel(torch.linalg.cholesky_ex, "cholesky_ex"),
    )
    monkeypatch.setattr(
        torch.special, "entr", without_mps_kernel(torch.special.entr, "entr")
    )
    model, x, _ = _gp("mps")
    assert_on_device(model, "mps")
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)
    assert set(called_on) == {"cholesky_ex:cpu", "entr:cpu"}


def test_gp_inverse_on_the_cpu_is_the_plain_cholesky_inverse():
    """On the CPU the helper is exactly ``cholesky_ex`` then ``cholesky_inverse`` (goldens)."""
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
        # torchmetrics keeps its state where it is told to; the model moves
        # the batches, so the metric must be on the same device.
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
    """``calib_x``/``calib_y`` come back as the caller split them, as in 0.1.8.

    The copies moved to the device and cast to the model dtype are internal
    to ``train_model``; the returned tensors stay on the CPU in the dataset's
    dtypes, so ``result["calib_y"].numpy()`` keeps working.
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

    Without a parameter anywhere (the head is an Identity too) there is
    nothing to train, so the support is set directly; inputs are moved but
    have no dtype to be cast to.
    """
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = eq.EquineProtonet(torch.nn.Identity(), FEATURES, device=device)
    assert model.model.emb_out_dim == x.shape[1]
    model.eval()  # the support covariance is an inference-time quantity
    model.update_support(x, y.float(), 0.5)
    assert_on_device(model, device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_gp_identity_embedding_casts_to_the_head_dtype(device):
    """With a parameter-less embedding the Laplace head's parameters set the input dtype, so float64 input still works."""
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
    """Embedding model whose input is integer token ids, mean-pooled then projected."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(num_embeddings=50, embedding_dim=8)
        self.linear = torch.nn.Linear(8, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.embed(x).mean(dim=1))


@pytest.mark.parametrize("device", devices())
def test_integer_inputs_are_moved_but_not_cast(device):
    """The dtype cast applies to floating inputs only: nn.Embedding indices stay integer."""
    torch.manual_seed(0)
    y = torch.arange(90) % CLASSES
    # class-specific ids, all below the 50 rows of the embedding table
    x = torch.randint(0, 47, (90, 4)) // CLASSES * CLASSES + y[:, None]
    model = eq.EquineProtonet(_TokenEmbedding(), CLASSES, device=device)
    model.train_model(
        torch.utils.data.TensorDataset(x, y),
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    assert x.dtype == torch.int64
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


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
