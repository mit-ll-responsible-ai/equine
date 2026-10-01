import numpy as np
import pytest
import torch
import torchmetrics
from conftest import (
    BasicEmbeddingModel,
    assert_valid_prediction,
    generate_random_string_list,
    random_dataset,
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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#171: EquineGP.train_model never resets val_metrics between epochs",
)
def test_validation_metrics_are_reset_between_epochs() -> None:
    dataset, x, y = separable_dataset()
    # float labels, like the dataset test_equine_gp_train_from_scratch_with_validation passes
    val = torch.utils.data.TensorDataset(x[:64], y[:64].float())
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=CLASSES)
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
    # train_model iterates the validation set with a DataLoader of the training
    # batch_size and calls metric.update once per batch: 64 rows / 32 = 2
    # updates per epoch. reset() zeroes update_count, so an implementation that
    # resets the metric once per epoch (before its updates or right after
    # compute()) shows every compute() exactly 2 updates; today the count
    # accumulates and each epoch's compute() covers all prior epochs.
    assert seen == [2, 2, 2]
