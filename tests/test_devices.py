# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Device-parametrized behaviour tests for both model classes (#188).

Every test runs on CPU and on whichever accelerator this machine has (see
``conftest.available_devices()``); on CI (Ubuntu, no accelerator) only the
CPU case is collected. The accelerator case carries a *strict* xfail only
where the path fails before the device-handling fixes (#170, #173, #177,
#188, #216). The CPU case is never marked for an accelerator-only failure;
the float64 input test and the CPU-only #216 case are marked on CPU
deliberately because they fail there too. The fixes flip the marked cases to
passing by removing the marks, and a strict xfail that starts passing fails
the run so a mark cannot go stale.
"""

import pytest
import torch
from conftest import (
    BasicEmbeddingModel,
    assert_on_device,
    assert_valid_prediction,
    available_devices,
)
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq

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


@pytest.mark.parametrize(
    "device",
    devices(
        xfail="#170: temperature buffer stays on CPU",
        raises=RuntimeError,
    ),
)
def test_protonet_with_temperature_predicts(device):
    model, x, _ = _protonet(device, use_temperature=True)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_protonet_update_support(device):
    model, x, y = _protonet(device)
    model.update_support(x, y.float(), 0.5)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize("device", devices())
def test_protonet_save_load_round_trip(device, tmp_path):
    model, x, _ = _protonet(device)
    before = model.predict(x[:5])
    path = str(tmp_path / "protonet.eq")
    model.save(path)

    generic = eq.load_equine_model(path)  # lands on the saved device
    _assert_same_predictions(before, generic.predict(x[:5]))

    explicit = eq.EquineProtonet.load(path, device)
    _assert_same_predictions(before, explicit.predict(x[:5]))


# --- both classes -------------------------------------------------------------


@_BUILDERS
@pytest.mark.parametrize(
    "device",
    devices(
        xfail=(
            "#170/#177: temperature stays on CPU; "
            "Protonet raw support[label] and GP _Laplace.seen_data too"
        ),
        raises=AssertionError,
    ),
)
def test_stored_tensors_on_device(build, device):
    model, _, _ = build(device)
    assert_on_device(model, device)


# --- EquineGP -----------------------------------------------------------------


@pytest.mark.parametrize("device", devices())
def test_gp_trains(device):
    model, _, _ = _gp(device)
    assert model.model.precision.device.type == torch.device(device).type


# EquineGP.forward already moves X to the device, so the predict paths get
# past compute_embeddings (#177) and fail on the missing MPS kernel (#173).
_GP_NO_CHOLESKY_KERNEL = (
    "#173: aten::cholesky_inverse has no MPS kernel (predict moves its input)"
)


@pytest.mark.parametrize(
    "device", devices(xfail=_GP_NO_CHOLESKY_KERNEL, raises=NotImplementedError)
)
def test_gp_predicts(device):
    model, x, _ = _gp(device)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.parametrize(
    "device",
    devices(xfail="#177: compute_embeddings does not move its input"),
)
def test_gp_update_support(device):
    model, x, y = _gp(device)
    model.update_support(x, y.long(), 10)
    assert set(model.support) == set(range(CLASSES))
    assert model.prototypes.shape[0] == CLASSES  # one prototype per class


@pytest.mark.parametrize(
    "device",
    devices(xfail="#177: compute_embeddings does not move its input"),
)
def test_gp_vis_support_training(device):
    model, _, _ = _gp(device, vis_support=True, support_size=10)
    assert set(model.support) == set(range(CLASSES))


@pytest.mark.parametrize(
    "device", devices(xfail=_GP_NO_CHOLESKY_KERNEL, raises=NotImplementedError)
)
def test_gp_save_load_round_trip(device, tmp_path):
    model, x, _ = _gp(device)
    before = model.predict(x[:5])
    path = str(tmp_path / "gp.eq")
    model.save(path)

    generic = eq.load_equine_model(path)  # lands on the saved device
    _assert_same_predictions(before, generic.predict(x[:5]))


@pytest.mark.skip(reason="#188: EquineGP.load does not take device= yet")
@pytest.mark.parametrize("device", devices())
def test_gp_load_onto_device(device, tmp_path):
    model, x, _ = _gp("cpu")
    before = model.predict(x[:5])
    path = str(tmp_path / "gp.eq")
    model.save(path)

    loaded = eq.EquineGP.load(path, device)
    assert_on_device(loaded, device)
    _assert_same_predictions(before, loaded.predict(x[:5]))


# --- input dtype --------------------------------------------------------------


@_BUILDERS
@pytest.mark.parametrize(
    "device",
    devices(
        xfail=(
            "#188: inputs are not cast to the embedding dtype yet "
            "(CPU raises mat1/mat2 dtype mismatch; MPS has no float64)"
        ),
        raises=(RuntimeError, TypeError),
        every_device=True,
    ),
)
def test_predict_accepts_float64_input(build, device):
    model, x, _ = build(device)
    assert_valid_prediction(model.predict(x[:5].double()), 5, CLASSES)


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
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="#216: EquineGP.device is a torch.device",
            ),
        ),
    ],
)
def test_device_attribute_is_a_string(make):
    assert isinstance(make().device, str)
