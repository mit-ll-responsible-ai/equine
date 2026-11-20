import copy
import math
import re
import warnings

import numpy as np
import pytest
import torch
import torchmetrics
from beartype.roar import BeartypeCallHintParamViolation
from builders import quick_gp
from conftest import (
    LABEL_KINDS,
    BasicEmbeddingModel,
    assert_model_unchanged,
    assert_on_device,
    assert_valid_prediction,
    generate_random_string_list,
    labels_of_kind,
    model_state,
    random_dataset,
    stored_tensors,
    use_basic_embedding_model,
    use_save_load_model_tests,
)
from golden_data import CLASSES, FEATURES, ROWS, separable_dataset
from hypothesis import given, settings

import equine as eq


def _sgd(model):
    """Cross-entropy and SGD with momentum and weight decay for ``model``."""
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001
    )
    return torch.nn.CrossEntropyLoss(), optimizer


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=10)
def test_equine_gp_train_from_scratch(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, _ = use_basic_embedding_model(
        random_dataset
    )

    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn, optimizer = _sgd(model)
    _ = model.train_model(dataset, loss_fn, optimizer, num_epochs=2)

    batch = X[1:10]
    assert_valid_prediction(model.predict(batch), len(batch), num_classes)


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=10)
def test_equine_gp_train_from_scratch_with_temperature(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, _ = use_basic_embedding_model(
        random_dataset
    )

    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn, optimizer = _sgd(model)
    before = model.temperature.item()
    assert before == 1.0, "init_temperature defaults to 1.0"
    train_dict = model.train_model(dataset, loss_fn, optimizer, num_epochs=2)
    assert "train_summary" in train_dict

    model.calibrate_model(dataset, 1, 0.01)
    after = model.temperature.item()
    assert after != before, "calibrate_model must move the temperature"
    assert after > 0

    batch = X[1:10]
    assert_valid_prediction(model.predict(batch), len(batch), num_classes)


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=10)
def test_equine_gp_train_from_scratch_with_scheduler(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, _ = use_basic_embedding_model(
        random_dataset
    )

    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn, optimizer = _sgd(model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
    train_dict = model.train_model(
        dataset, loss_fn, optimizer, scheduler=scheduler, num_epochs=5
    )
    assert "train_summary" in train_dict
    assert np.isclose(scheduler.get_last_lr()[0], 0.00001)

    batch = X[1:10]
    assert_valid_prediction(model.predict(batch), len(batch), num_classes)


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=10)
def test_equine_gp_train_from_scratch_with_validation(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, _ = use_basic_embedding_model(
        random_dataset
    )

    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn, optimizer = _sgd(model)
    _ = model.train_model(
        dataset,
        loss_fn,
        optimizer,
        validation_dataset=dataset,
        val_metrics=[
            torchmetrics.classification.MulticlassAccuracy(num_classes),
            torchmetrics.classification.MulticlassCalibrationError(num_classes),
        ],
        num_epochs=2,
    )
    batch = X[1:10]
    assert_valid_prediction(model.predict(batch), len(batch), num_classes)


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=2)
def test_equine_gp_save_load(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, _ = use_basic_embedding_model(
        random_dataset
    )

    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn, optimizer = _sgd(model)
    model.train_model(dataset, loss_fn, optimizer, num_epochs=2)

    use_save_load_model_tests(model, X)


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_equine_gp_save_load_with_temperature(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, _ = use_basic_embedding_model(
        random_dataset
    )

    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn, optimizer = _sgd(model)
    before = model.temperature.item()
    train_dict = model.train_model(dataset, loss_fn, optimizer, num_epochs=2)
    assert "train_summary" in train_dict

    model.calibrate_model(dataset, 1, 0.01)
    calibrated = model.temperature.item()
    assert calibrated != before, "calibrate_model must move the temperature"
    assert calibrated > 0

    new_model = use_save_load_model_tests(model, X)
    assert new_model.temperature.item() == pytest.approx(calibrated), (
        "temperature changed on reload"
    )


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_equine_gp_save_load_with_vis(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, train_kwargs = use_basic_embedding_model(
        random_dataset
    )

    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn, optimizer = _sgd(model)
    model.train_model(
        dataset,
        loss_fn,
        optimizer,
        num_epochs=2,
        vis_support=True,
        support_size=train_kwargs["support_size"],
    )

    new_model = use_save_load_model_tests(model, X)

    assert new_model.support is not None, "support was not saved"
    assert new_model.prototypes is not None, "prototypes were not saved"
    assert model.support.keys() == new_model.get_support().keys(), (
        "Support keys changed on reload"
    )
    assert (
        torch.nn.functional.mse_loss(model.prototypes, new_model.get_prototypes())
        <= 1e-7
    ), "Prototypes changed on reload"


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_equine_gp_save_load_with_feature_and_label_names(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, _ = use_basic_embedding_model(
        random_dataset
    )

    # without feature and labels names
    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn, optimizer = _sgd(model)
    model.train_model(dataset, loss_fn, optimizer, num_epochs=2)

    new_model = use_save_load_model_tests(model, X)

    assert new_model.get_feature_names() is None, "feature_names changed on reload"
    assert new_model.get_label_names() is None, "label_names changed on reload"

    feature_names = generate_random_string_list(X.shape[1])
    label_names = generate_random_string_list(num_classes)

    model = eq.EquineGP(
        embedding_model,
        num_classes,
        num_classes,
        feature_names=feature_names,
        label_names=label_names,
    )
    loss_fn, optimizer = _sgd(model)
    model.train_model(dataset, loss_fn, optimizer, num_epochs=2)

    new_model = use_save_load_model_tests(model, X)

    assert new_model.get_feature_names() == feature_names, (
        "feature_names changed on reload"
    )
    assert new_model.get_label_names() == label_names, "label_names changed on reload"


def test_validation_metrics_are_reset_between_epochs() -> None:
    dataset, x, y = separable_dataset()
    # float labels, like the dataset test_equine_gp_train_from_scratch_with_validation passes
    val = torch.utils.data.TensorDataset(x[:64], y[:64].float())
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=CLASSES)
    # Primes the metric with an update from before training.
    metric.update(torch.zeros(4, CLASSES), torch.zeros(4, dtype=torch.long))
    # Record how many updates each epoch's compute() sees. train_model calls
    # compute() once per epoch, after iterating the validation set.
    seen: list[int] = []
    orig_compute = metric.compute

    def recording_compute():
        seen.append(metric.update_count)
        return orig_compute()

    metric.compute = recording_compute
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.01),
        num_epochs=3,
        batch_size=32,
        validation_dataset=val,
        val_metrics=[metric],
    )
    # 64 validation rows in batches of 32: 2 updates per epoch. With a reset
    # every epoch, each compute() sees 2 updates; without one the count grows.
    assert seen == [2, 2, 2]


def test_assigning_device_type_is_refused_naming_device() -> None:
    """Assigning ``device_type`` raises and leaves the model on its device; reading it
    raises; ``model.to`` moves the model and ``device`` names where it is."""
    model, _, _ = quick_gp()
    with pytest.raises(AttributeError) as refused:
        model.device_type = "meta"
    assert str(refused.value) == (
        "EquineGP.device_type was removed: read EquineGP.device (a str) instead; "
        "to move the model, call model.to(value)."
    )
    assert model.device == "cpu"
    assert_on_device(model, "cpu")
    with pytest.raises(AttributeError, match="device_type"):
        model.device_type
    assert not hasattr(model, "device_type")

    model.to("meta")
    assert model.device == "meta"
    assert_on_device(model, "meta")


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_a_failed_move_leaves_the_model_usable() -> None:
    """A move that raises leaves ``device`` as it was and the model usable."""
    model, x, _ = quick_gp()
    with pytest.raises((AssertionError, RuntimeError)):
        model.to("cuda")
    assert model.device == "cpu"
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


def test_validation_metrics_may_be_a_generator() -> None:
    dataset, x, y = separable_dataset()
    val = torch.utils.data.TensorDataset(x[:64], y[:64])
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    metrics = [
        torchmetrics.classification.MulticlassAccuracy(num_classes=CLASSES)
        for _ in range(2)
    ]
    out = model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.01),
        num_epochs=2,
        batch_size=32,
        validation_dataset=val,
        val_metrics=(metric for metric in metrics),
    )
    assert [len(values) for values in out["val_metrics"]] == [2, 2]
    # 64 rows in batches of 32, reset each epoch.
    assert [metric.update_count for metric in metrics] == [2, 2]


def test_save_refuses_an_untrained_model(tmp_path) -> None:
    model = eq.EquineGP(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES)
    with pytest.raises(RuntimeError, match="has not been trained"):
        model.save(str(tmp_path / "untrained.eq"))
    assert not (tmp_path / "untrained.eq").exists()


def test_update_support_keeps_a_class_with_one_example() -> None:
    model, x, _ = quick_gp()
    y = torch.cat([torch.zeros(6), torch.ones(1)]).long()
    model.update_support(x[:7], y, support_size=5)
    assert [len(rows) for rows in model.get_support().values()] == [5, 1]


def test_laplace_refuses_more_rows_than_num_data_in_one_epoch() -> None:
    model, x, _ = quick_gp()
    model.model.train()
    model.model.reset_precision_matrix()
    model.model(x[:120])
    with pytest.raises(RuntimeError, match="Did not reset precision matrix"):
        model.model(x[:1])


@pytest.mark.parametrize(
    "dtype, default_dtype, precision_dtype",
    [
        (torch.float16, torch.float32, torch.float32),
        (torch.bfloat16, torch.float32, torch.float32),
        (torch.float32, torch.float32, torch.float32),
        (torch.float64, torch.float32, torch.float64),
        (torch.float32, torch.float64, torch.float64),
        (torch.float32, torch.float16, torch.float32),
        (torch.bfloat16, torch.float16, torch.float32),
        (torch.float16, torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
    ],
)
def test_train_model_accumulates_the_precision_in_the_promoted_dtype(
    dtype, default_dtype, precision_dtype
) -> None:
    """After train_model under ``default_dtype`` the precision matrix has
    ``precision_dtype`` and equals the identity plus the sum of ``k.t() @ k`` over
    the training set, with ``k`` the eval-mode features of the final weights, to
    within the rounding of ``precision_dtype``."""
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    ).to(dtype)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(default_dtype)
    try:
        model.train_model(
            torch.utils.data.TensorDataset(x.to(dtype), y.to(dtype)),
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.05),
            num_epochs=1,
            batch_size=32,
        )
    finally:
        torch.set_default_dtype(previous)
    with torch.no_grad():
        features = [model.compute_embeddings(c).double() for c in x.to(dtype).split(32)]
    expected = torch.eye(16, dtype=torch.float64) + sum(k.t() @ k for k in features)
    precision = model.model.precision
    assert precision.dtype == precision_dtype
    error = (precision.double() - expected).abs().max() / expected.abs().max()
    assert error <= 16 * torch.finfo(precision_dtype).eps


def test_a_float64_gp_with_a_float32_precision_accumulates_in_float64() -> None:
    """A float32 precision buffer in a float64 model becomes float64 at the next
    train_model."""
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    data = torch.utils.data.TensorDataset(x.double(), y.float())
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    ).double()
    model.model.precision = model.model.precision.float()
    model.train_model(
        data,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=1,
        batch_size=32,
    )
    assert model.model.precision.dtype == torch.float64


def _batchnorm_dropout_embedding() -> torch.nn.Sequential:
    """An embedding from ``FEATURES`` to ``CLASSES`` with BatchNorm and dropout."""
    return torch.nn.Sequential(
        torch.nn.Linear(FEATURES, 16),
        torch.nn.BatchNorm1d(16),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(16, CLASSES),
    )


def _independent_precision(model: eq.EquineGP, x: torch.Tensor) -> torch.Tensor:
    """The ridge identity plus the sum of ``k.t() @ k`` in float64 on the CPU, with
    ``k`` the eval-mode ``compute_embeddings`` of ``x`` in batches of 32."""
    model.eval()
    with torch.no_grad():
        k = torch.cat([model.compute_embeddings(c).cpu().double() for c in x.split(32)])
    eye = torch.eye(k.shape[1], dtype=torch.float64)
    return model.model.ridge_penalty * eye + k.t() @ k


@pytest.mark.parametrize(
    "make_embedding, rows, batch_size",
    [
        (lambda: BasicEmbeddingModel(FEATURES, CLASSES), ROWS, 32),
        (_batchnorm_dropout_embedding, ROWS, 32),
        (lambda: BasicEmbeddingModel(FEATURES, CLASSES), 1, 1),
    ],
    ids=["basic", "batchnorm_dropout", "one_row"],
)
def test_train_model_builds_the_precision_from_the_final_weights(
    make_embedding, rows, batch_size
) -> None:
    """After train_model the precision matrix of a float64 GP is the ridge identity
    plus the sum of ``k.t() @ k`` over every training example, with ``k`` the
    eval-mode features of the final weights. 120 rows in batches of 32 end with a
    batch of 24."""
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    x, y = x[:rows].double(), y[:rows]
    model = eq.EquineGP(
        make_embedding(), CLASSES, CLASSES, num_random_features=16
    ).double()
    model.train_model(
        torch.utils.data.TensorDataset(x, y),
        torch.nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=1e-2),
        num_epochs=3,
        batch_size=batch_size,
    )
    assert model.model._seen_count == int(model.model.seen_data) == rows
    torch.testing.assert_close(
        model.model.precision, _independent_precision(model, x), rtol=1e-12, atol=1e-12
    )


def test_precision_rebuild_runs_the_embedding_in_eval_mode() -> None:
    """After 3 epochs of 4 training batches, the embedding runs 4 more batches in
    eval mode with gradients off; its BatchNorm buffers are those left by the last
    training batch, and every module is in eval mode afterwards."""
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    embedding = _batchnorm_dropout_embedding()
    batchnorm = embedding[1]
    modes: list[bool] = []
    grads: list[bool] = []
    buffers: list[dict[str, torch.Tensor]] = []

    def record_buffers(module, args, output):
        if module.training:
            buffers.append({k: v.clone() for k, v in module.named_buffers()})

    embedding.register_forward_pre_hook(
        lambda module, args: modes.append(module.training)
    )
    embedding.register_forward_pre_hook(
        lambda module, args: grads.append(torch.is_grad_enabled())
    )
    batchnorm.register_forward_hook(record_buffers)
    model = eq.EquineGP(embedding, CLASSES, CLASSES, num_random_features=16)
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=1e-2),
        num_epochs=3,
        batch_size=32,
    )
    assert modes == [True] * 12 + [False] * 4
    assert len(grads) == 16 and all(grads[:12])
    assert grads[12:] == [False] * 4
    assert int(batchnorm.num_batches_tracked) == 12
    for name, value in batchnorm.named_buffers():
        assert torch.equal(value, buffers[-1][name]), name
    assert not any(module.training for module in model.modules())


@pytest.mark.parametrize("subset", [False, True], ids=["tensor_dataset", "subset"])
def test_precision_rebuild_draws_no_global_random_numbers(subset) -> None:
    """Rebuilding the precision of a trained GP in training mode whose embedding has
    BatchNorm and dropout leaves the global random number generator's state as it
    was, through either kind of batch loader, gives the precision train_model left,
    and leaves every module in eval mode."""
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    if subset:
        dataset = torch.utils.data.Subset(dataset, list(range(ROWS)))
    model = eq.EquineGP(
        _batchnorm_dropout_embedding(), CLASSES, CLASSES, num_random_features=16
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=2,
        batch_size=32,
    )
    trained = model.model.precision.clone()
    model.train()
    state = torch.get_rng_state()
    model._rebuild_precision(dataset, 32)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(model.model.precision, trained)
    assert not any(module.training for module in model.modules())


class _Failing(torch.nn.Module):
    """``BasicEmbeddingModel`` whose ``fail_at``-th forward in the mode ``fail_in``
    (``"train"`` or ``"eval"``, counted in ``calls``) raises ``error``."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = BasicEmbeddingModel(FEATURES, CLASSES)
        self.fail_in = "train"
        self.fail_at: int | None = None
        self.error: BaseException = RuntimeError("embedding failed")
        self.calls = {"train": 0, "eval": 0}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = "train" if self.training else "eval"
        self.calls[mode] += 1
        if mode == self.fail_in and self.calls[mode] == self.fail_at:
            raise self.error
        return self.inner(x)


class _Boom(Exception):
    """The error raised at a failure point."""


def _fail_on_call(function, n: int, error: BaseException):
    """``function``, except that its ``n``-th call raises ``error``."""
    calls = [0]

    def wrapped(*args, **kwargs):
        calls[0] += 1
        if calls[0] == n:
            raise error
        return function(*args, **kwargs)

    return wrapped


def _assert_untrained(model: eq.EquineGP, dataset, x, tmp_path) -> None:
    """The training parameters are cleared, the precision matrix is the ridge
    identity with no rows seen, no module is in training mode, and forward, predict,
    calibrate_model and save refuse the model."""
    laplace = model.model
    assert laplace.training_parameters_set is False
    assert (laplace.num_data, laplace.train_batch_size) == (0, 0)
    assert laplace._seen_count == 0
    assert int(laplace.seen_data) == 0
    identity = torch.eye(laplace.precision.shape[0], dtype=laplace.precision.dtype)
    assert torch.equal(laplace.precision, identity * laplace.ridge_penalty)
    assert not any(module.training for module in model.modules())
    for method, call in (
        ("forward", lambda: model(x)),
        ("predict", lambda: model.predict(x)),
        ("calibrate_model", lambda: model.calibrate_model(dataset)),
    ):
        with pytest.raises(
            ValueError,
            match=rf"^EquineGP\.{method} needs a trained model, and this model has "
            r"not been trained; call train_model first\.$",
        ):
            call()
    with pytest.raises(
        RuntimeError,
        match=r"^The model has not been trained; call train_model before save\.$",
    ):
        model.save(str(tmp_path / "model.eq"))
    assert not (tmp_path / "model.eq").exists()


_FAILURE_POINTS = [
    "embedding_second_batch",
    "embedding_last_batch",
    "keyboard_interrupt",
    "loss",
    "optimizer_step",
    "scheduler_step",
    "validation_metric",
    "precision_rebuild",
    "update_support",
    "train_summary",
]


@pytest.mark.parametrize("trained_before", [False, True], ids=["fresh", "trained"])
@pytest.mark.parametrize("point", _FAILURE_POINTS)
def test_an_error_in_train_model_leaves_the_model_untrained(
    point, trained_before, tmp_path, monkeypatch
) -> None:
    """2 epochs of 4 batches (32, 32, 32 and 24 rows). train_model raises the error
    of each failure point: the embedding on epoch 2's second or last training batch,
    a KeyboardInterrupt on epoch 1's second batch, the loss or the optimizer step on
    the sixth batch, the scheduler after epoch 1, a validation metric in epoch 2,
    the embedding on the precision rebuild's last batch, update_support
    (vis_support) and the train summary. The model is then untrained, and
    train_model with no epochs leaves no rows in its precision matrix. train_model
    then trains it again."""
    torch.manual_seed(0)
    dataset, x, _ = separable_dataset()
    embedding = _Failing()
    model = eq.EquineGP(embedding, CLASSES, CLASSES, num_random_features=16)

    def train(**options) -> dict:
        options = {"num_epochs": 2, "batch_size": 32, **options}
        loss_fn = options.pop("loss_fn", torch.nn.CrossEntropyLoss())
        opt = options.pop("opt", None) or torch.optim.SGD(model.parameters(), lr=0.05)
        return model.train_model(dataset, loss_fn, opt, **options)

    if trained_before:
        train()
    error: BaseException = (
        KeyboardInterrupt() if point == "keyboard_interrupt" else _Boom(point)
    )
    options: dict = {}
    if point in (
        "embedding_second_batch",
        "embedding_last_batch",
        "keyboard_interrupt",
        "precision_rebuild",
    ):
        embedding.calls = {"train": 0, "eval": 0}
        embedding.fail_in = "eval" if point == "precision_rebuild" else "train"
        embedding.fail_at = {
            "embedding_second_batch": 6,
            "embedding_last_batch": 8,
            "keyboard_interrupt": 2,
            "precision_rebuild": 4,
        }[point]
        embedding.error = error
    elif point == "loss":
        options["loss_fn"] = _fail_on_call(torch.nn.CrossEntropyLoss(), 6, error)
    elif point == "optimizer_step":
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        opt.step = _fail_on_call(opt.step, 6, error)
        options["opt"] = opt
    elif point == "scheduler_step":
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1)
        scheduler.step = _fail_on_call(scheduler.step, 1, error)
        options.update(opt=opt, scheduler=scheduler)
    elif point == "validation_metric":
        metric = _SumOfTopProbabilities()
        metric.update = _fail_on_call(metric.update, 5, error)
        options.update(validation_dataset=dataset, val_metrics=[metric])
    elif point == "update_support":
        monkeypatch.setattr(
            eq.EquineGP,
            "update_support",
            _fail_on_call(eq.EquineGP.update_support, 1, error),
        )
        options.update(vis_support=True, support_size=10)
    else:
        monkeypatch.setattr(
            eq.equine_gp,
            "generate_train_summary",
            _fail_on_call(eq.equine_gp.generate_train_summary, 1, error),
        )

    with pytest.raises(type(error)) as raised:
        train(**options)
    assert raised.value is error
    embedding.fail_at = None
    monkeypatch.undo()
    _assert_untrained(model, dataset, x, tmp_path)

    train(num_epochs=0)
    assert model.model._seen_count == 0
    with pytest.raises(
        RuntimeError, match=r"^Not seen sufficient data for precision matrix$"
    ):
        model.predict(x)

    train()
    torch.testing.assert_close(
        model.model.precision.double(),
        _independent_precision(model, x),
        rtol=1e-5,
        atol=1e-4,
    )
    assert_valid_prediction(model.predict(x), ROWS, CLASSES)


@pytest.mark.parametrize("trained_before", [False, True], ids=["fresh", "trained"])
def test_an_error_before_the_first_epoch_leaves_the_model_unchanged(
    trained_before,
) -> None:
    """A validation metric whose ``to`` raises: train_model raises the error before
    it sets the training parameters, and the model is unchanged, including a model
    trained before on 96 of the 120 rows."""
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    if trained_before:
        model.train_model(
            torch.utils.data.TensorDataset(x[:96], y[:96].float()),
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.05),
            num_epochs=2,
            batch_size=32,
        )
    before = model_state(model)
    metric = _SumOfTopProbabilities()
    error = _Boom("to")
    metric.to = _fail_on_call(metric.to, 1, error)
    with pytest.raises(_Boom) as raised:
        model.train_model(
            dataset,
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.05),
            num_epochs=2,
            batch_size=32,
            validation_dataset=dataset,
            val_metrics=[metric],
        )
    assert raised.value is error
    assert_model_unchanged(model, before)


@pytest.mark.parametrize("trained_before", [False, True], ids=["fresh", "trained"])
def test_train_model_refuses_vis_support_without_tensors(trained_before) -> None:
    """vis_support=True with a Subset is refused before training, and the model is
    unchanged. Both repairs the message names work: a TensorDataset with
    vis_support=True, and vis_support=False followed by update_support."""
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    subset = torch.utils.data.Subset(dataset, list(range(ROWS)))
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )

    def train(data, **options) -> dict:
        return model.train_model(
            data,
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.05),
            num_epochs=2,
            batch_size=32,
            **options,
        )

    if trained_before:
        train(dataset)
    before = model_state(model)
    with pytest.raises(
        ValueError,
        match=r"^vis_support=True takes the support from dataset\.tensors, and a "
        r"Subset has no tensors; pass a TensorDataset, or pass vis_support=False "
        r"and call update_support\(support_x, support_y, support_size=\.\.\.\) "
        r"after train_model\.$",
    ):
        train(subset, vis_support=True, support_size=10)
    assert_model_unchanged(model, before)

    train(dataset, vis_support=True, support_size=10)
    assert sorted(model.support) == list(range(CLASSES))
    train(subset)
    model.update_support(x, y, support_size=10)
    assert sorted(model.support) == list(range(CLASSES))


class _SumOfTopProbabilities(torchmetrics.Metric):
    """The sum over the examples of the largest softmax probability of ``preds``."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "total", default=torch.zeros((), dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.total += torch.softmax(preds.double(), dim=1).max(dim=1).values.sum()

    def compute(self) -> torch.Tensor:
        return self.total


def test_the_last_validation_pass_uses_the_rebuilt_precision() -> None:
    """The last epoch's validation metric equals the same metric computed afterwards
    from the returned model's logits, in the same batches of 32."""
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    validation, x_val, _ = separable_dataset(seed=5)
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    result = model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=3,
        batch_size=32,
        validation_dataset=validation,
        val_metrics=[_SumOfTopProbabilities()],
    )
    metric = _SumOfTopProbabilities()
    with torch.no_grad():
        for batch in x_val.split(32):
            metric.update(model(batch), torch.empty(0))
    torch.testing.assert_close(
        result["val_metrics"][0][-1], metric.compute(), rtol=0, atol=1e-9
    )


@pytest.mark.parametrize("num_epochs", [0, -1])
def test_train_model_without_epochs_leaves_the_precision_and_mode(num_epochs) -> None:
    """With no epoch to run, train_model passes nothing through the embedding and
    leaves the precision matrix and the training mode as they were."""
    model, x, y = quick_gp()
    model.train()
    calls: list[int] = []
    model.embedding_model.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].shape[0])
    )
    precision = model.model.precision.clone()
    result = model.train_model(
        torch.utils.data.TensorDataset(x, y),
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=num_epochs,
        batch_size=32,
    )
    assert calls == []
    assert torch.equal(model.model.precision, precision)
    assert result["train_loss"] == []
    assert model.training


def _bfloat16_gp(classes: int):
    """An untrained bfloat16 GP over a small MLP, float32 rows of ``classes``
    shifted normal clusters of 200, their labels, and the generator that drew them.
    Seeds torch with 0 first."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    x = torch.cat(
        [torch.randn(200, 4, generator=gen) + 3.0 * i for i in range(classes)]
    )
    y = torch.arange(classes).repeat_interleave(200).float()
    embedding = torch.nn.Sequential(
        torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8)
    )
    return eq.EquineGP(embedding, 8, classes).to(torch.bfloat16), x, y, gen


def test_a_bfloat16_gp_predicts_after_training() -> None:
    """The precision matrix of a trained bfloat16 GP is positive definite and
    predict inverts it."""
    model, x, y, _ = _bfloat16_gp(3)
    model.train_model(
        torch.utils.data.TensorDataset(x.bfloat16(), y),
        torch.nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=1e-2),
        num_epochs=3,
        batch_size=32,
    )
    assert torch.linalg.eigvalsh(model.model.precision.double()).min() >= 0.99
    output = model.predict(x[:20].bfloat16())
    assert bool(torch.isfinite(output.ood_scores).all())


@pytest.mark.parametrize(
    "default_dtype", [torch.float16, torch.bfloat16, torch.float64]
)
def test_predict_does_not_depend_on_the_default_dtype(default_dtype) -> None:
    """A float32 GP's first predict under ``default_dtype`` gives the probabilities,
    OOD scores and covariance it gives under float32."""
    model, x, _ = quick_gp()
    reference = copy.deepcopy(model)
    expected = reference.predict(x)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(default_dtype)
    try:
        output = model.predict(x)
    finally:
        torch.set_default_dtype(previous)
    assert torch.equal(output.classes, expected.classes)
    assert torch.equal(output.ood_scores, expected.ood_scores)
    assert torch.equal(model.model.covariance, reference.model.covariance)


def test_a_float64_gp_computes_predict_in_float64() -> None:
    """A float64 GP's covariance is the inverse of its precision plus a float64
    jitter, and its OOD scores are normalized by the float64 maximum entropy."""
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    ).double()
    model.train_model(
        torch.utils.data.TensorDataset(x.double(), y.float()),
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=1,
        batch_size=32,
    )
    output = model.predict(x.double())
    precision = model.model.precision
    jitter = 1e-7 * torch.eye(16, dtype=torch.float64)
    assert torch.equal(
        model.model.covariance, eq.equine_gp._inverse_via_cholesky(precision + jitter)
    )
    uniform = torch.full((CLASSES,), 1 / CLASSES, dtype=torch.float64)
    expected = (
        torch.special.entr(output.classes).sum(1) / torch.special.entr(uniform).sum()
    )
    assert output.ood_scores.dtype == torch.float64
    assert torch.equal(output.ood_scores, expected)


def test_a_float32_gp_computes_predict_in_float32() -> None:
    """A float32 GP's covariance is the inverse of its precision plus a float32
    jitter, and its OOD scores are the float32 entropy ratio of its probabilities,
    clamped to [0, 1]."""
    model, x, _ = quick_gp()
    output = model.predict(x)
    precision = model.model.precision
    jitter = 1e-7 * torch.eye(16, dtype=torch.float32)
    assert torch.equal(
        model.model.covariance, eq.equine_gp._inverse_via_cholesky(precision + jitter)
    )
    uniform = torch.ones(CLASSES, dtype=torch.float32) / CLASSES
    entropy = torch.special.entr(output.classes).sum(1)
    expected = (entropy / torch.special.entr(uniform).sum()).clamp(0, 1)
    assert output.ood_scores.dtype == torch.float32
    assert torch.equal(output.ood_scores, expected)


def test_a_bfloat16_gp_computes_ood_scores_in_float32() -> None:
    """A bfloat16 GP's OOD scores are the float32 entropy ratio of its logits, clamped
    to [0, 1] and cast to bfloat16."""
    classes = 10
    model, x, y, gen = _bfloat16_gp(classes)
    model.train_model(
        torch.utils.data.TensorDataset(x.bfloat16(), y),
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=1e-2),
        num_epochs=3,
        batch_size=32,
    )
    queries = torch.cat(
        [x, torch.randn(2000, 4, generator=gen) * 3.0 * classes]
    ).bfloat16()
    scores = model.predict(queries).ood_scores
    with torch.no_grad():
        logits, _ = model._forward_with_embeddings(queries)
    entropy = torch.special.entr(torch.softmax(logits.float(), dim=1)).sum(1)
    uniform = torch.ones(classes, dtype=torch.float32) / classes
    expected = (entropy / torch.special.entr(uniform).sum()).clamp(0, 1).bfloat16()
    assert scores.dtype == torch.bfloat16
    assert torch.equal(scores, expected)
    assert bool(((scores >= 0) & (scores <= 1)).all())


def test_a_gp_cast_to_bfloat16_after_training_predicts() -> None:
    """A trained GP cast to bfloat16, whose precision matrix is then bfloat16,
    predicts with OOD scores in [0, 1]."""
    model, x, _ = quick_gp()
    model.to(torch.bfloat16)
    assert model.model.precision.dtype == torch.bfloat16
    scores = model.predict(x.bfloat16()).ood_scores
    assert bool(((scores >= 0) & (scores <= 1)).all())


def test_nearly_equal_logits_keep_ood_scores_at_most_one() -> None:
    """A float64 GP whose logits are nearly equal (a temperature of 1e9) gives OOD
    scores in [0, 1]."""
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    ).double()
    model.train_model(
        torch.utils.data.TensorDataset(x.double(), y.float()),
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=1,
        batch_size=32,
    )
    model.temperature.fill_(1e9)
    scores = model.predict(x.double()).ood_scores
    assert bool(((scores >= 0) & (scores <= 1)).all())


def _recording_loss(losses: list[torch.Tensor]):
    """Cross-entropy that appends each batch's loss tensor to ``losses``."""

    def loss_fn(yhats, labels):
        loss = torch.nn.functional.cross_entropy(yhats, labels)
        losses.append(loss)
        return loss

    return loss_fn


def test_train_model_returns_the_mean_batch_loss_of_each_epoch() -> None:
    """``train_loss`` holds, per epoch, the mean of that epoch's batch losses."""
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    losses: list[torch.Tensor] = []
    result = model.train_model(
        dataset,
        _recording_loss(losses),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=3,
        batch_size=32,
    )
    batches = -(-ROWS // 32)
    assert len(losses) == 3 * batches
    values = [float(loss.detach().double()) for loss in losses]
    expected = [
        sum(values[e * batches : (e + 1) * batches]) / batches for e in range(3)
    ]
    assert all(type(value) is float for value in result["train_loss"])
    assert result["train_loss"] == pytest.approx(expected, rel=1e-6)


def test_train_model_reads_no_batch_loss_on_the_host(monkeypatch) -> None:
    """No batch loss, or a view of one, is read with ``item()``, ``float()`` or
    ``tolist()`` during training, and nothing read that way requires grad."""
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    read: list[torch.Tensor] = []
    for name in ("item", "__float__", "tolist"):
        original = getattr(torch.Tensor, name)

        def recording(self, *args, _original=original, **kwargs):
            read.append(self)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, recording)
    losses: list[torch.Tensor] = []
    model.train_model(
        dataset,
        _recording_loss(losses),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=2,
        batch_size=32,
    )
    monkeypatch.undo()
    assert losses
    storages = {loss.untyped_storage().data_ptr() for loss in losses}
    assert read
    assert storages.isdisjoint(t.untyped_storage().data_ptr() for t in read)
    assert not any(t.requires_grad for t in read)


def test_a_float8_loss_is_summed_in_float32() -> None:
    """A loss_fn returning a float8 loss trains, and train_loss is the mean of its
    batch losses."""
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    losses: list[torch.Tensor] = []
    record = _recording_loss(losses)
    result = model.train_model(
        dataset,
        lambda yhats, labels: record(yhats, labels).to(torch.float8_e4m3fn),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=1,
        batch_size=32,
    )
    values = [float(loss.detach().to(torch.float8_e4m3fn).double()) for loss in losses]
    assert result["train_loss"] == pytest.approx([sum(values) / len(values)], rel=1e-6)


def _host_reads(monkeypatch, batch_size: int) -> int:
    """The number of ``item()``, ``float()`` and ``tolist()`` calls during a
    two-epoch ``train_model`` with ``batch_size``."""
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    calls = [0]
    for name in ("item", "__float__", "tolist"):
        original = getattr(torch.Tensor, name)

        def counting(self, *args, _original=original, **kwargs):
            calls[0] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, counting)
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=2,
        batch_size=batch_size,
    )
    monkeypatch.undo()
    return calls[0]


def test_train_model_reads_from_the_host_as_often_for_any_number_of_batches(
    monkeypatch,
) -> None:
    """Training with one batch per epoch and with 15 batches per epoch reads tensors
    on the host the same number of times."""
    assert _host_reads(monkeypatch, ROWS) == _host_reads(monkeypatch, 8)


@pytest.mark.parametrize("shape", [(), (1,)])
def test_a_bfloat16_gp_sums_its_epoch_loss_in_float32(shape) -> None:
    """The epoch loss of a bfloat16 GP is summed in float32, for a 0-d loss and a
    one-element loss."""
    model, x, y, _ = _bfloat16_gp(3)
    losses: list[torch.Tensor] = []
    record = _recording_loss(losses)
    result = model.train_model(
        torch.utils.data.TensorDataset(x.bfloat16(), y),
        lambda yhats, labels: record(yhats, labels).reshape(shape),
        torch.optim.SGD(model.parameters(), lr=1e-2),
        num_epochs=1,
        batch_size=8,
    )
    assert len(losses) == 75
    assert all(loss.dtype == torch.bfloat16 for loss in losses)
    expected = sum(float(loss.detach().double()) for loss in losses) / 75
    assert result["train_loss"] == pytest.approx([expected], rel=1e-6)


def test_predict_refuses_a_model_that_has_seen_too_little_data() -> None:
    model = eq.EquineGP(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES)
    model.model.set_training_params(120, 32)
    with pytest.raises(RuntimeError, match="Not seen sufficient data"):
        model.predict(torch.rand(2, FEATURES))


def _assert_no_autograd_history(model) -> None:
    """Every tensor ``stored_tensors`` finds on ``model`` has no grad_fn."""
    for name, t in stored_tensors(model).items():
        assert t.grad_fn is None, f"{name} still has a grad_fn"


def test_equine_gp_deepcopy_after_training_with_vis_support() -> None:
    """A GP trained with ``vis_support=True`` keeps no autograd history."""
    model, _, _ = quick_gp(vis_support=True, support_size=10)
    _assert_no_autograd_history(model)
    copy.deepcopy(model)


def test_laplace_jl_projects_through_its_own_random_matrix_after_deepcopy() -> None:
    """A deep copy's projection reads its own ``random_matrix``."""
    model, _, _ = quick_gp()
    model_copy = copy.deepcopy(model)
    model_copy.model.random_matrix.zero_()
    f = torch.randn(5, model.num_deep_features)
    assert model.model.jl(f).abs().sum().item() > 0
    assert model_copy.model.jl(f).abs().sum().item() == 0.0


def test_laplace_jl_is_identity_without_a_projection() -> None:
    """Without a GP projection, ``jl`` returns its input unchanged."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        laplace = eq.equine_gp._Laplace(
            feature_extractor=torch.nn.Identity(),
            num_deep_features=4,
            num_gp_features=0,
            normalize_gp_features=False,
            num_random_features=0,
            num_outputs=3,
            feature_scale=None,
            mean_field_factor=25,
        )
    x = torch.randn(5, 4)
    assert laplace.jl(x) is x


def test_equine_gp_predict_and_forward_accept_a_single_1d_example() -> None:
    """A single unbatched example gets a batch dimension, as ``EquineProtonet`` does."""
    model, x, _ = quick_gp()

    batched_out = model.predict(x[:1])
    single_out = model.predict(x[0])
    assert single_out.classes.shape == (1, CLASSES)
    assert torch.allclose(single_out.classes, batched_out.classes)
    assert torch.allclose(single_out.ood_scores, batched_out.ood_scores)

    batched_logits = model(x[:1])
    single_logits = model(x[0])
    assert single_logits.shape == (1, CLASSES)
    assert torch.allclose(single_logits, batched_logits)


def test_laplace_counts_an_unbatched_training_example_as_one_row() -> None:
    model, x, _ = quick_gp()
    model.model.train()
    model.model.reset_precision_matrix()
    model.model(x[0])
    assert model.model.seen_data == 1
    assert model.model._seen_count == 1
    # The remaining rows complete the epoch without the reset check firing.
    model.model(x[1:120])


@pytest.fixture
def bounded_random_ortho(monkeypatch):
    calls = []
    real = eq.equine_gp._random_ortho

    def bounded(n, m):
        calls.append(1)
        if len(calls) > 100:
            raise AssertionError("_RandomFourierFeatures loops without end")
        return real(n, m)

    monkeypatch.setattr(eq.equine_gp, "_random_ortho", bounded)


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"emb_out_dim": 0, "num_classes": 2, "num_random_features": 8}, "emb_out_dim"),
        (
            {"emb_out_dim": -1, "num_classes": 2, "num_random_features": 8},
            "emb_out_dim",
        ),
        ({"emb_out_dim": 4, "num_classes": 0, "num_random_features": 8}, "num_classes"),
        (
            {"emb_out_dim": 4, "num_classes": 2, "num_random_features": 0},
            "num_random_features",
        ),
    ],
)
def test_equine_gp_refuses_non_positive_sizes(
    bounded_random_ortho, kwargs, name
) -> None:
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        eq.EquineGP(torch.nn.Identity(), **kwargs)


def test_equine_gp_refuses_one_class(bounded_random_ortho) -> None:
    with pytest.raises(ValueError, match="num_classes must be at least 2, got 1"):
        eq.EquineGP(
            torch.nn.Identity(), emb_out_dim=4, num_classes=1, num_random_features=8
        )
    model = eq.EquineGP(
        torch.nn.Identity(), emb_out_dim=4, num_classes=2, num_random_features=8
    )
    assert model.num_outputs == 2


@pytest.mark.parametrize("label_dtype", [torch.float32, torch.long])
@pytest.mark.parametrize("shift", [0, 7])
def test_update_support_draws_one_permutation_per_class(label_dtype, shift) -> None:
    """The support and the RNG state after the call equal a replay of one
    ``randperm`` over each class's rows in ascending label order, each class keeping
    its first ``min(count, support_size)`` rows. The class of label ``1 + shift``
    has one example."""
    support_y = torch.tensor([0] * 6 + [1] + [2] * 9) + shift
    order = torch.randperm(len(support_y), generator=torch.Generator().manual_seed(1))
    support_y = support_y[order].to(label_dtype)
    support_x = torch.rand(
        len(support_y), FEATURES, generator=torch.Generator().manual_seed(2)
    )
    support_x[:, 0] = torch.arange(len(support_y))
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        CLASSES + shift,
        num_random_features=16,
    )
    for seed in range(5):
        torch.manual_seed(seed)
        model.update_support(support_x, support_y, support_size=5)
        state = torch.get_rng_state()

        torch.manual_seed(seed)
        expected = []
        for label in torch.unique(support_y).tolist():
            rows = torch.argwhere(support_y == label).reshape(-1)
            expected.append((label, support_x[rows[torch.randperm(len(rows))][:5]]))
        support = model.get_support()
        assert list(support) == [label for label, _ in expected]
        assert [len(rows) for rows in support.values()] == [5, 1, 5]
        for label, rows in expected:
            assert torch.equal(support[label], rows)
        assert torch.equal(torch.get_rng_state(), state)


def test_update_support_refuses_support_x_and_support_y_of_different_lengths() -> None:
    """The refusal comes before any random draw, for every seed."""
    model = eq.EquineGP(torch.nn.Identity(), 4, 3)
    x = torch.rand(20, 4)
    y = torch.arange(20) % 3
    for seed in range(20):
        torch.manual_seed(seed)
        state = torch.get_rng_state()
        with pytest.raises(ValueError, match="support_x has 19 examples"):
            model.update_support(x[1:], y, support_size=3)
        assert torch.equal(torch.get_rng_state(), state)


GP_LABEL_KINDS = LABEL_KINDS + [
    f"{kind} {dtype}"
    for kind in ("ignored", "negative", "too large")
    for dtype in ("float", "long")
]


def _gp_labels_of_kind(y: torch.Tensor, kind: str) -> tuple[torch.Tensor, str]:
    """``labels_of_kind``, plus labels 0, 1 and 2 with label 2 replaced by a value
    outside the classes (-100, -1 or ``CLASSES``), as float or long."""
    if kind in LABEL_KINDS:
        return labels_of_kind(y, kind)
    name, dtype = kind.rsplit(" ", 1)
    value = {"ignored": -100, "negative": -1, "too large": CLASSES}[name]
    labels = torch.where(y == 2, value, y)
    labels = labels.float() if dtype == "float" else labels.long()
    return labels, f"must be class indices from 0 to {CLASSES - 1}, got {value}\\."


@pytest.mark.parametrize("labels", GP_LABEL_KINDS)
def test_train_model_refuses_labels_that_are_not_1d_whole_numbers(labels) -> None:
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    y, message = _gp_labels_of_kind(y, labels)
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    model.eval()
    state = copy.deepcopy(model.state_dict())
    summary = copy.deepcopy(model.train_summary)
    with pytest.raises(ValueError, match=f"dataset labels {message}"):
        model.train_model(
            torch.utils.data.TensorDataset(x, y),
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.05),
            num_epochs=1,
            batch_size=32,
        )
    after = model.state_dict()
    assert all(torch.equal(state[key], after[key]) for key in state)
    assert not model.training
    assert not model.model.training_parameters_set
    assert model.train_summary == summary


def _rows_of_each_class(rows_per_class: int) -> torch.utils.data.TensorDataset:
    """The first ``rows_per_class`` rows of each class of ``separable_dataset``."""
    dataset, _, y = separable_dataset()
    rows = torch.cat(
        [torch.argwhere(y == label).reshape(-1)[:rows_per_class] for label in range(3)]
    )
    return torch.utils.data.TensorDataset(*dataset[rows])


def _support_size_message(shown: str) -> str:
    return (
        "support_size must be a positive integer (an int >= 1, not a bool); got "
        f"{shown}."
    )


@pytest.mark.parametrize(
    ("rows_per_class", "options", "message", "repair"),
    [
        *[
            pytest.param(
                40,
                {"batch_size": 32, "vis_support": True, "support_size": value},
                _support_size_message(shown),
                {"support_size": 5},
                id=f"vis_support-support_size={value}",
            )
            for value, shown in ((0, "int 0"), (-1, "int -1"), (True, "bool"))
        ],
        pytest.param(
            40,
            {"batch_size": 200},
            "batch_size (200) is larger than the dataset (120 rows); pass a "
            "batch_size of at most 120.",
            {"batch_size": 120},
            id="batch_size-200-on-120-rows",
        ),
        pytest.param(
            16,
            {},
            "batch_size (64) is larger than the dataset (48 rows); pass a "
            "batch_size of at most 48.",
            {"batch_size": 48},
            id="default-batch_size-on-48-rows",
        ),
    ],
)
def test_train_model_refuses_bad_input(
    rows_per_class, options, message, repair
) -> None:
    """Each refusal names the argument, draws no random numbers and leaves the
    model untrained; the call its message asks for trains it."""
    torch.manual_seed(0)
    dataset = _rows_of_each_class(rows_per_class)
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = model_state(model)
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match=re.escape(message)):
        model.train_model(dataset, torch.nn.CrossEntropyLoss(), optimizer, 1, **options)
    assert_model_unchanged(model, before)
    assert torch.equal(torch.get_rng_state(), rng), "the refusal drew random numbers"

    options.update(repair)
    model.train_model(dataset, torch.nn.CrossEntropyLoss(), optimizer, 1, **options)
    assert model.model.training_parameters_set
    assert model.predict(dataset.tensors[0][:5]).classes.shape == (5, CLASSES)
    if options.get("vis_support"):
        assert [len(r) for r in model.get_support().values()] == [5] * CLASSES


@pytest.mark.parametrize("support_size", [0, -1])
def test_train_model_without_vis_support_does_not_read_support_size(
    support_size,
) -> None:
    """Without ``vis_support``, ``support_size`` is not read: any value trains the
    model and leaves the support empty."""
    torch.manual_seed(0)
    dataset = _rows_of_each_class(40)
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        1,
        batch_size=32,
        support_size=support_size,
    )
    assert model.model.training_parameters_set
    assert len(model.support) == 0
    assert model.predict(dataset.tensors[0][:5]).classes.shape == (5, CLASSES)


def _epochs_message(shown: str) -> str:
    return (
        "num_calibration_epochs must be a non-negative integer (an int >= 0, not a "
        f"bool; 0 leaves the temperature unchanged); got {shown}."
    )


def _lr_message(value: object) -> str:
    return (
        "calibration_lr must be a finite float above 0 (not an int or a bool), such as "
        f"0.01 or 1.0; got {value!r}."
    )


@pytest.mark.parametrize(
    ("trained", "changes", "message", "repair"),
    [
        *[
            pytest.param(
                True,
                {"num_calibration_epochs": value},
                _epochs_message(shown),
                {"num_calibration_epochs": 1},
                id=f"num_calibration_epochs={value}",
            )
            for value, shown in ((-1, "int -1"), (True, "bool"))
        ],
        *[
            pytest.param(
                True,
                {"calibration_lr": value},
                _lr_message(value),
                {"calibration_lr": 0.01},
                id=f"calibration_lr={value}",
            )
            for value in (0.0, -0.01, math.nan, math.inf)
        ],
        pytest.param(
            False,
            {},
            "EquineGP.calibrate_model needs a trained model, and this model has not "
            "been trained; call train_model first.",
            {},
            id="untrained",
        ),
    ],
)
def test_calibrate_model_refuses_bad_input(trained, changes, message, repair) -> None:
    """Each refusal names the argument (or asks for training), draws no random
    numbers and leaves the model as it was, the temperature and the Laplace buffers
    included; the call its message asks for calibrates the temperature."""
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    if trained:
        model.train_model(
            dataset,
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.05),
            num_epochs=1,
            batch_size=32,
        )
    with torch.no_grad():
        model.temperature.fill_(0.5)
    args = {"num_calibration_epochs": 1, "calibration_lr": 0.01, **changes}
    before = model_state(model)
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match=re.escape(message)):
        model.calibrate_model(dataset, **args)
    assert_model_unchanged(model, before)
    assert torch.equal(torch.get_rng_state(), rng), "the refusal drew random numbers"

    if not trained:
        model.train_model(
            dataset,
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.05),
            num_epochs=1,
            batch_size=32,
        )
    args.update(repair)
    model.calibrate_model(dataset, **args)
    assert model.temperature.item() != 0.5
    assert not model.temperature.requires_grad


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        *[
            pytest.param(
                changes,
                BeartypeCallHintParamViolation,
                None,
                marks=pytest.mark.contracts,
                id=f"{name}-contracts-on",
            )
            for changes, name in (
                ({"calibration_lr": 0}, "int-calibration_lr-0"),
                ({"calibration_lr": 1}, "int-calibration_lr-1"),
                ({"calibration_lr": True}, "bool-calibration_lr"),
                ({"num_calibration_epochs": 1.0}, "float-num_calibration_epochs"),
            )
        ],
        *[
            pytest.param(
                changes,
                ValueError,
                message,
                marks=pytest.mark.skipif(
                    __debug__, reason="beartype refuses the wrong type first"
                ),
                id=f"{name}-python-O",
            )
            for changes, name, message in (
                ({"calibration_lr": 0}, "int-calibration_lr-0", _lr_message(0)),
                ({"calibration_lr": 1}, "int-calibration_lr-1", _lr_message(1)),
                (
                    {"calibration_lr": True},
                    "bool-calibration_lr",
                    _lr_message(True),
                ),
                (
                    {"num_calibration_epochs": 1.0},
                    "float-num_calibration_epochs",
                    _epochs_message("float"),
                ),
            )
        ],
    ],
)
def test_calibrate_model_refuses_a_wrong_number_type(changes, error, message) -> None:
    """An int or bool ``calibration_lr`` or a float ``num_calibration_epochs`` is a
    type error while contracts are on and a ``ValueError`` under python -O; the model
    is left as it was, and the values the message asks for calibrate the
    temperature."""
    model, _, _ = quick_gp()
    dataset = separable_dataset()[0]
    with torch.no_grad():
        model.temperature.fill_(0.5)
    before = model_state(model)
    rng = torch.get_rng_state()
    with pytest.raises(error, match=None if message is None else re.escape(message)):
        model.calibrate_model(dataset, **changes)
    assert_model_unchanged(model, before)
    assert torch.equal(torch.get_rng_state(), rng), "the refusal drew random numbers"

    model.calibrate_model(dataset, num_calibration_epochs=1, calibration_lr=0.01)
    assert model.temperature.item() != 0.5


def test_calibrate_model_with_0_epochs_leaves_the_temperature_unchanged() -> None:
    model, _, _ = quick_gp()
    dataset = separable_dataset()[0]
    with torch.no_grad():
        model.temperature.fill_(0.5)
    model.calibrate_model(dataset, num_calibration_epochs=0)
    assert model.temperature.item() == 0.5
    assert not model.temperature.requires_grad


@pytest.mark.parametrize("labels", GP_LABEL_KINDS)
def test_calibrate_model_refuses_labels_that_are_not_1d_whole_numbers(labels) -> None:
    model, x, y = quick_gp()
    y, message = _gp_labels_of_kind(y, labels)
    with torch.no_grad():
        model.temperature.fill_(0.5)
    with pytest.raises(ValueError, match=f"dataset labels {message}"):
        model.calibrate_model(
            torch.utils.data.TensorDataset(x, y), calibration_batch_size=16
        )
    assert model.temperature.item() == 0.5
    assert not model.temperature.requires_grad


@pytest.mark.parametrize("error", ["label", "interrupt"])
def test_calibrate_model_restores_the_temperature_when_a_later_batch_raises(
    error,
) -> None:
    model, x, y = quick_gp()
    y = y.float()
    if error == "label":
        y[-1] = 0.5
    forward_calls = []

    def count_and_interrupt(*_):
        forward_calls.append(None)
        if error == "interrupt" and len(forward_calls) == 3:
            raise KeyboardInterrupt

    model.model.register_forward_hook(count_and_interrupt)
    with torch.no_grad():
        model.temperature.fill_(0.5)
    expected = ValueError if error == "label" else KeyboardInterrupt
    with pytest.raises(expected):
        model.calibrate_model(
            torch.utils.data.TensorDataset(x, y),
            calibration_lr=0.1,
            calibration_batch_size=4,
        )
    assert len(forward_calls) > 0
    assert model.temperature.item() == 0.5
    assert not model.temperature.requires_grad


def test_calibrate_model_leaves_the_temperature_as_it_was_when_the_optimizer_raises(
    monkeypatch,
) -> None:
    model, x, y = quick_gp()

    def refuse(*args, **kwargs):
        raise ValueError("refused")

    monkeypatch.setattr(torch.optim, "Adam", refuse)
    with torch.no_grad():
        model.temperature.fill_(0.5)
    with pytest.raises(ValueError, match="refused"):
        model.calibrate_model(torch.utils.data.TensorDataset(x, y.float()))
    assert model.temperature.item() == 0.5
    assert not model.temperature.requires_grad


@pytest.mark.parametrize("labels", GP_LABEL_KINDS)
def test_update_support_refuses_labels_that_are_not_1d_whole_numbers(labels) -> None:
    model, x, y = quick_gp()
    model.update_support(x, y.float(), support_size=5)
    support = {label: rows.clone() for label, rows in model.support.items()}
    prototypes = model.prototypes.clone()
    y, message = _gp_labels_of_kind(y, labels)
    state = torch.get_rng_state()
    with pytest.raises(ValueError, match=f"support_y {message}"):
        model.update_support(x, y, support_size=5)
    assert torch.equal(torch.get_rng_state(), state)
    assert list(model.support) == list(support)
    assert all(
        torch.equal(model.support[label], rows) for label, rows in support.items()
    )
    assert torch.equal(model.prototypes, prototypes)


def _float64_laplace(mean_field_factor):
    """A float64 ``_Laplace`` over identity features, seeded with 0, trained on one
    epoch of 40 random rows and set to eval mode."""
    torch.manual_seed(0)
    laplace = eq.equine_gp._Laplace(
        feature_extractor=torch.nn.Identity(),
        num_deep_features=FEATURES,
        num_gp_features=FEATURES,
        normalize_gp_features=True,
        num_random_features=32,
        num_outputs=CLASSES,
        feature_scale=2.0,
        mean_field_factor=mean_field_factor,
        ridge_penalty=0.5,
    ).to(torch.float64)
    laplace.set_training_params(num_data=40, batch_size=40)
    laplace.train()
    laplace(torch.randn(40, FEATURES, dtype=torch.float64))
    return laplace.eval()


@pytest.mark.parametrize("batch_size", [1, 7])
def test_laplace_eval_logits_equal_the_full_covariance_formula(batch_size) -> None:
    """In float64, eval-mode logits are ``beta(k)`` divided by
    ``sqrt(1 + diag(k @ covariance @ k.T * ridge_penalty) * mean_field_factor)``.
    """
    laplace = _float64_laplace(mean_field_factor=25)
    x = 2 * torch.randn(batch_size, FEATURES, dtype=torch.float64)

    with torch.no_grad():
        logits, k = laplace._forward_with_features(x)
        mean = laplace.beta(k)
        pred_cov = k @ ((laplace.covariance @ k.t()) * 0.5)
        expected = mean / torch.sqrt(1.0 + torch.diag(pred_cov) * 25).unsqueeze(-1)

    assert logits.dtype == torch.float64
    assert not torch.allclose(logits, mean), "the variance must change the logits"
    torch.testing.assert_close(logits, expected, rtol=0, atol=1e-12)


def test_mean_field_logits_receives_one_variance_per_example(monkeypatch) -> None:
    """``mean_field_logits`` receives a ``(batch_size,)`` variance."""
    model, x, _ = quick_gp()
    original = eq.equine_gp._Laplace.mean_field_logits
    shapes = []

    def recording(self, logits, pred_var, mean_field_factor):
        shapes.append(tuple(pred_var.shape))
        return original(self, logits, pred_var, mean_field_factor)

    monkeypatch.setattr(eq.equine_gp._Laplace, "mean_field_logits", recording)
    model.predict(x[:5])
    model(x[:5])
    assert shapes == [(5,), (5,)]


def test_laplace_without_mean_field_returns_the_full_covariance() -> None:
    """With ``mean_field_factor=None``, eval mode returns ``beta(k)`` and the full
    ``k @ covariance @ k.T * ridge_penalty`` matrix."""
    laplace = _float64_laplace(mean_field_factor=None)
    x = 2 * torch.randn(7, FEATURES, dtype=torch.float64)

    with torch.no_grad():
        (pred, pred_cov), k = laplace._forward_with_features(x)
        identity = torch.eye(32, dtype=torch.float64)
        covariance = torch.linalg.inv(laplace.precision + 1e-7 * identity)
        expected = k @ ((covariance @ k.t()) * 0.5)

    assert pred_cov.shape == (7, 7)
    torch.testing.assert_close(pred, laplace.beta(k), atol=1e-12, rtol=0)
    torch.testing.assert_close(pred_cov, expected, atol=1e-12, rtol=0)


class _ReversedFeatures(torch.utils.data.TensorDataset):
    """A TensorDataset whose examples have their features in reverse order."""

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x.flip(0), y


@pytest.mark.parametrize(
    "kind",
    [
        "tensors",
        "transposed",
        "permuted",
        "channels_last",
        "grayscale",
        "channels_last_rows",
        "subclass",
    ],
)
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("batch_size", [1, 7, 30, 64])
def test_batch_loader_matches_dataloader(kind, shuffle, batch_size) -> None:
    """``_batch_loader`` yields the batches of ``DataLoader(..., shuffle=shuffle)``,
    with the same strides, and leaves the same RNG state."""
    from equine.equine_gp import _batch_loader, _row_major_batch

    gen = torch.Generator().manual_seed(3)
    x = torch.rand(30, 4, generator=gen)
    if kind == "transposed":
        x = x.t().contiguous().t()
    elif kind == "permuted":
        x = torch.rand(30, 3, 4, generator=gen).permute(0, 2, 1)
    elif kind == "channels_last":
        x = torch.rand(30, 2, 3, 3, generator=gen)
        x = x.to(memory_format=torch.channels_last)
    elif kind == "grayscale":
        x = torch.rand(30, 3, 3, 1, generator=gen).permute(0, 3, 1, 2)
    elif kind == "channels_last_rows":
        x = torch.rand(60, 3, 2, 2, generator=gen)
        x = x.to(memory_format=torch.channels_last).view(30, 2, 3, 2, 2)
    dataset_class = (
        _ReversedFeatures if kind == "subclass" else torch.utils.data.TensorDataset
    )
    dataset = dataset_class(x, torch.arange(30).float())
    torch.manual_seed(0)
    expected = list(
        torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    )
    state = torch.get_rng_state()
    loader = _batch_loader(dataset, batch_size, shuffle=shuffle)
    batched = kind not in ("channels_last_rows", "subclass")
    assert (loader.collate_fn is _row_major_batch) == batched
    torch.manual_seed(0)
    got = list(loader)
    assert torch.equal(torch.get_rng_state(), state)
    assert len(got) == len(expected)
    assert {type(batch) for batch in got} == {type(batch) for batch in expected}
    for (x_e, y_e), (x_g, y_g) in zip(expected, got):
        assert torch.equal(x_g, x_e)
        assert torch.equal(y_g, y_e)
        assert y_g.dtype == y_e.dtype
        assert (x_g.stride(), y_g.stride()) == (x_e.stride(), y_e.stride())


BATCH_DTYPE_NAMES = [
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]


@pytest.mark.parametrize("dtype", BATCH_DTYPE_NAMES)
def test_batch_loader_reads_each_listed_dtype_a_batch_at_a_time(dtype) -> None:
    """For features of each dtype in ``_BATCH_DTYPES``, ``_batch_loader`` indexes the
    dataset once per batch and yields the batches of ``DataLoader``."""
    from equine.equine_gp import _batch_loader, _row_major_batch

    gen = torch.Generator().manual_seed(3)
    x = torch.randn(12, 2, 4, generator=gen).permute(0, 2, 1)
    if dtype.startswith("complex"):
        x = torch.complex(x, -x)
    x = x.to(getattr(torch, dtype))
    dataset = torch.utils.data.TensorDataset(x, torch.arange(12))
    loader = _batch_loader(dataset, 5, shuffle=True)
    assert loader.collate_fn is _row_major_batch
    torch.manual_seed(0)
    expected = list(torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True))
    torch.manual_seed(0)
    got = list(loader)
    assert len(got) == len(expected)
    for (x_e, y_e), (x_g, y_g) in zip(expected, got):
        assert x_g.dtype == x_e.dtype
        assert torch.equal(x_g, x_e)
        assert x_g.stride() == x_e.stride()
        assert torch.equal(y_g, y_e)


def test_batch_loader_reads_every_other_dtype_through_dataloader() -> None:
    """Features of a dtype outside ``_BATCH_DTYPES`` keep the ``DataLoader``."""
    from equine.equine_gp import _BATCH_DTYPES, _batch_loader, _row_major_batch

    listed = {getattr(torch, name) for name in BATCH_DTYPE_NAMES}
    assert _BATCH_DTYPES == listed
    checked = set()
    dtypes = {
        value
        for value in (getattr(torch, name) for name in dir(torch))
        if isinstance(value, torch.dtype)
    }
    for dtype in dtypes - listed:
        try:
            x = torch.zeros(12, 8, dtype=torch.uint8).view(dtype)
        except (RuntimeError, TypeError):
            try:
                x = torch.zeros(12, 4, dtype=dtype)
            except (RuntimeError, TypeError):
                continue
        dataset = torch.utils.data.TensorDataset(x, torch.arange(12))
        assert (
            _batch_loader(dataset, 5, shuffle=True).collate_fn is not _row_major_batch
        )
        checked.add(dtype)
    assert {
        torch.bool,
        torch.uint4,
        torch.bits8,
        torch.int4,
        torch.float8_e4m3fn,
    } <= checked


def _features_of_kind(kind):
    """Twelve rows of four features as a nested, jagged, quantized, sparse, uint4 or
    bits8 tensor."""
    if kind in ("uint4", "bits8"):
        return torch.randint(0, 16, (12, 4), dtype=torch.uint8).view(
            getattr(torch, kind)
        )
    rows = [torch.rand(4) for _ in range(12)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "nested":
            return torch.nested.nested_tensor(rows)
        if kind == "jagged":
            return torch.nested.nested_tensor(rows, layout=torch.jagged)
        if kind == "quantized":
            scales = torch.full((12,), 0.1)
            zero_points = torch.zeros(12, dtype=torch.long)
            return torch.quantize_per_channel(
                torch.stack(rows), scales, zero_points, 0, torch.quint8
            )
    return torch.stack(rows).to_sparse()


def _comparable(x):
    """``x`` as a tensor that ``torch.equal`` compares."""
    if x.is_quantized:
        return x.dequantize()
    if x.dtype in (torch.uint4, torch.bits8):
        return x.view(torch.uint8)
    return x


@pytest.mark.parametrize(
    "kind", ["nested", "jagged", "quantized", "sparse", "uint4", "bits8"]
)
def test_batch_loader_reads_other_tensor_kinds_as_dataloader_does(kind) -> None:
    """For nested, quantized, sparse, uint4 and bits8 features, ``_batch_loader`` is
    a ``DataLoader`` and yields its batches, or raises the same exception type."""
    from equine.equine_gp import _batch_loader, _row_major_batch

    torch.manual_seed(0)
    dataset = torch.utils.data.TensorDataset(_features_of_kind(kind), torch.arange(12))
    assert _batch_loader(dataset, 4, shuffle=True).collate_fn is not _row_major_batch

    def batches(loader):
        try:
            return [(x.is_quantized, _comparable(x), y) for x, y in loader]
        except Exception as error:
            return type(error)

    torch.manual_seed(0)
    expected = batches(torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True))
    torch.manual_seed(0)
    got = batches(_batch_loader(dataset, 4, shuffle=True))
    if isinstance(expected, type):
        assert got is expected
        return
    assert isinstance(got, list)
    assert len(got) == len(expected)
    for (q_e, x_e, y_e), (q_g, x_g, y_g) in zip(expected, got):
        assert q_g == q_e
        assert torch.equal(x_g, x_e)
        assert x_g.stride() == x_e.stride()
        assert torch.equal(y_g, y_e)


def test_train_model_on_a_tensor_dataset_matches_a_subset_of_it() -> None:
    """A TensorDataset (read a batch at a time) and a Subset holding all of its
    rows (read an example at a time) train the same model from the same seed."""
    dataset, _, _ = separable_dataset()
    subset = torch.utils.data.Subset(dataset, list(range(len(dataset))))
    states = []
    for data in (dataset, subset):
        torch.manual_seed(0)
        model = eq.EquineGP(
            BasicEmbeddingModel(FEATURES, CLASSES),
            CLASSES,
            CLASSES,
            num_random_features=16,
        )
        model.train_model(
            data,
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.05),
            num_epochs=3,
            batch_size=32,
            validation_dataset=data,
            val_metrics=[torchmetrics.Accuracy(task="multiclass", num_classes=CLASSES)],
        )
        model.calibrate_model(data, num_calibration_epochs=2, calibration_batch_size=16)
        states.append(model.state_dict())
    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), key


def test_training_and_calibration_batches_are_shuffled_and_validation_and_rebuild_batches_are_not(
    monkeypatch,
) -> None:
    """The loaders in order: training (shuffled), validation, the precision rebuild
    over the training set (its own generator), calibration (shuffled)."""
    import equine.equine_gp as equine_gp

    calls = []
    batch_loader = equine_gp._batch_loader

    def recording_batch_loader(dataset, batch_size, shuffle, generator=None):
        calls.append((dataset, shuffle, generator is None))
        return batch_loader(dataset, batch_size, shuffle, generator)

    monkeypatch.setattr(equine_gp, "_batch_loader", recording_batch_loader)
    train, _, _ = separable_dataset()
    validation = torch.utils.data.TensorDataset(*train.tensors)
    calibration = torch.utils.data.TensorDataset(*train.tensors)
    torch.manual_seed(0)
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        CLASSES,
        num_random_features=16,
    )
    model.train_model(
        train,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=1,
        batch_size=32,
        validation_dataset=validation,
        val_metrics=[torchmetrics.Accuracy(task="multiclass", num_classes=CLASSES)],
    )
    model.calibrate_model(calibration, calibration_batch_size=16)
    expected = [
        (train, True, True),
        (validation, False, True),
        (train, False, False),
        (calibration, True, True),
    ]
    assert len(calls) == len(expected)
    for call, expected_call in zip(calls, expected):
        assert call[0] is expected_call[0]
        assert call[1:] == expected_call[1:]
