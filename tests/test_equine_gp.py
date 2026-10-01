import numpy as np
import pytest
import torch
import torchmetrics
from conftest import (
    assert_valid_prediction,
    generate_random_string_list,
    random_dataset,
    use_basic_embedding_model,
    use_save_load_model_tests,
)
from hypothesis import given, settings

import equine as eq


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=10)
def test_equine_gp_train_from_scratch(random_dataset) -> None:
    dataset, num_classes, X, embedding_model = use_basic_embedding_model(random_dataset)

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
    dataset, num_classes, X, embedding_model = use_basic_embedding_model(random_dataset)

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
    dataset, num_classes, X, embedding_model = use_basic_embedding_model(random_dataset)

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
    dataset, num_classes, X, embedding_model = use_basic_embedding_model(random_dataset)

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
    dataset, num_classes, X, embedding_model = use_basic_embedding_model(random_dataset)

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
    dataset, num_classes, X, embedding_model = use_basic_embedding_model(random_dataset)

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
    dataset, num_classes, X, embedding_model = use_basic_embedding_model(random_dataset)

    model = eq.EquineGP(embedding_model, num_classes, num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )
    model.train_model(dataset, loss_fn, optimizer, num_epochs=2, vis_support=True)

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
    dataset, num_classes, X, embedding_model = use_basic_embedding_model(random_dataset)

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
