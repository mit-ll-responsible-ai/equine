import copy
import warnings
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torchmetrics
from conftest import (
    BasicEmbeddingModel,
    assert_valid_prediction,
    generate_random_string_list,
    random_dataset,
    stored_tensors,
    use_basic_embedding_model,
    use_save_load_model_tests,
)
from golden_data import CLASSES, FEATURES, separable_dataset
from hypothesis import given, settings

import equine as eq


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=10)
def test_equine_gp_train_from_scratch(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, _ = use_basic_embedding_model(
        random_dataset
    )

    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
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
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
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
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
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
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
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
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
    model.train_model(dataset, loss_fn, optimizer, num_epochs=2)

    use_save_load_model_tests(model, X, tmp_filename="gp_save_load.eq")


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=1)
def test_equine_gp_save_load_with_temperature(random_dataset) -> None:
    dataset, num_classes, X, embedding_model, _ = use_basic_embedding_model(
        random_dataset
    )

    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
    before = model.temperature.item()
    train_dict = model.train_model(dataset, loss_fn, optimizer, num_epochs=2)
    assert "train_summary" in train_dict

    model.calibrate_model(dataset, 1, 0.01)
    calibrated = model.temperature.item()
    assert calibrated != before, "calibrate_model must move the temperature"
    assert calibrated > 0

    new_model = use_save_load_model_tests(
        model, X, tmp_filename="gp_save_load_with_temperature.eq"
    )
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
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
    model.train_model(
        dataset,
        loss_fn,
        optimizer,
        num_epochs=2,
        vis_support=True,
        support_size=train_kwargs["support_size"],
    )

    new_model = use_save_load_model_tests(
        model, X, tmp_filename="gp_save_load_with_vis.eq"
    )

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
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
    model.train_model(dataset, loss_fn, optimizer, num_epochs=2)

    new_model = use_save_load_model_tests(
        model, X, tmp_filename="gp_save_load_no_feature_and_label_names.eq"
    )

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
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
    model.train_model(dataset, loss_fn, optimizer, num_epochs=2)

    new_model = use_save_load_model_tests(
        model, X, tmp_filename="gp_save_load_with_feature_and_label_names.eq"
    )

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


def _gp_trained_on_cpu():
    torch.manual_seed(0)
    dataset, x, _ = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=1,
        batch_size=32,
    )
    return model, x


def _device_type_warning(record) -> warnings.WarningMessage:
    """The one device_type DeprecationWarning in ``record``, checked to point at the
    caller.
    """
    found = [w for w in record if "device_type" in str(w.message)]
    assert len(found) == 1
    assert found[0].category is DeprecationWarning
    assert found[0].filename == __file__, "the warning must point at the caller"
    return found[0]


def test_device_type_is_deprecated_alias(monkeypatch) -> None:
    model, x = _gp_trained_on_cpu()
    with pytest.warns(DeprecationWarning, match="device_type") as record:
        assert model.device_type == "cpu"
    message = str(_device_type_warning(record).message)
    # The replacement it names moves the model and updates device.
    assert "call model.to(value)" in message
    assert model.device == "cpu"

    # Assigning a new value moves the module and changes ``device``; predict still
    # works.
    monkeypatch.setattr(model, "to", Mock(wraps=model.to))
    with pytest.warns(DeprecationWarning, match="device_type") as record:
        model.device_type = "cpu:0"
    _device_type_warning(record)
    model.to.assert_called_once_with("cpu:0")
    assert model.device == "cpu:0"
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_device_type_failed_move_leaves_the_model_usable() -> None:
    """A move that raises leaves ``device`` as it was and the model usable; the module
    moves before ``device`` changes.
    """
    model, x = _gp_trained_on_cpu()
    with pytest.warns(DeprecationWarning, match="device_type"):
        with pytest.raises((AssertionError, RuntimeError)):
            model.device_type = "cuda"
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


def test_predict_refuses_an_empty_batch() -> None:
    model, _ = _gp_trained_on_cpu()
    with pytest.raises(ValueError, match="X is empty"):
        model.predict(torch.empty(0, FEATURES))


@pytest.mark.parametrize("value", [float("nan"), float("-inf")], ids=["nan", "inf"])
def test_predict_refuses_non_finite_embeddings(value) -> None:
    model, _ = _gp_trained_on_cpu()
    with pytest.raises(ValueError, match="embeddings of X hold NaN or infinite"):
        model.predict(torch.full((3, FEATURES), value))


class _ImputingEmbedding(torch.nn.Module):
    """A linear embedding that reads a NaN feature as 0."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(FEATURES, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.nan_to_num(x))


def test_predict_accepts_nan_input_that_the_embedding_handles() -> None:
    torch.manual_seed(0)
    dataset, x, _ = separable_dataset()
    model = eq.EquineGP(_ImputingEmbedding(), CLASSES, CLASSES, num_random_features=16)
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=1,
        batch_size=32,
    )
    rows = x[:5].clone()
    rows[:, 0] = float("nan")
    expected = model.predict(torch.nan_to_num(rows))
    out = model.predict(rows)
    assert torch.equal(out.classes, expected.classes)
    assert torch.equal(out.ood_scores, expected.ood_scores)


def test_update_support_keeps_a_class_with_one_example() -> None:
    model, x = _gp_trained_on_cpu()
    y = torch.cat([torch.zeros(6), torch.ones(1)]).long()
    model.update_support(x[:7], y, support_size=5)
    assert [len(rows) for rows in model.get_support().values()] == [5, 1]


def test_laplace_refuses_more_rows_than_num_data_in_one_epoch() -> None:
    model, x = _gp_trained_on_cpu()
    model.model.train()
    model.model.reset_precision_matrix()
    model.model(x[:120])
    with pytest.raises(RuntimeError, match="Did not reset precision matrix"):
        model.model(x[:1])


def test_predict_refuses_a_model_that_has_seen_too_little_data() -> None:
    model = eq.EquineGP(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES)
    model.model.set_training_params(120, 32)
    with pytest.raises(RuntimeError, match="Not seen sufficient data"):
        model.predict(torch.rand(2, FEATURES))


def _assert_no_autograd_history(model) -> None:
    """Every tensor ``stored_tensors`` finds on ``model`` has no grad_fn."""
    for name, t in stored_tensors(model).items():
        assert t.grad_fn is None, f"{name} still has a grad_fn"


def test_equine_gp_deepcopy_after_training() -> None:
    """A trained GP keeps no autograd history; ``copy.deepcopy`` works."""
    model, _ = _gp_trained_on_cpu()
    _assert_no_autograd_history(model)
    assert model.model.precision.grad_fn is None
    copy.deepcopy(model)


def test_equine_gp_deepcopy_after_training_with_vis_support() -> None:
    """A GP trained with ``vis_support=True`` keeps no autograd history."""
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES),
        CLASSES,
        CLASSES,
        num_random_features=16,
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=1,
        batch_size=32,
        vis_support=True,
        support_size=10,
    )
    _assert_no_autograd_history(model)
    copy.deepcopy(model)


def test_equine_gp_loaded_from_a_file_keeps_no_autograd_history(tmp_path) -> None:
    torch.manual_seed(0)
    dataset, _, _ = separable_dataset()
    model = eq.EquineGP(
        BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=1,
        batch_size=32,
        vis_support=True,
        support_size=10,
    )
    path = str(tmp_path / "gp.eq")
    model.save(path)
    loaded = eq.load_equine_model(path)
    _assert_no_autograd_history(loaded)
    copy.deepcopy(loaded)


def test_laplace_jl_projects_through_its_own_random_matrix_after_deepcopy() -> None:
    """A deep copy's projection reads its own ``random_matrix``."""
    model, _ = _gp_trained_on_cpu()
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
    model, x = _gp_trained_on_cpu()

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
    model, x = _gp_trained_on_cpu()
    model.model.train()
    model.model.reset_precision_matrix()
    model.model(x[0])
    assert model.model.seen_data == 1
    assert model.model._seen_count == 1
    # The remaining rows complete the epoch without the reset check firing.
    model.model(x[1:120])
