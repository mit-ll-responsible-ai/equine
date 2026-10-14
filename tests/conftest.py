# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import os
import zipfile
from random import choice
from string import ascii_lowercase, digits

import pytest
import torch
from hypothesis import strategies as st

import equine as eq


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "accelerator: needs a CUDA or MPS device (skipped without one)"
    )


def _rewrite_zip(src: str, dst: str, edit) -> None:
    """Copy a torch archive entry by entry (uncompressed) through ``edit(name, data)``."""
    with (
        zipfile.ZipFile(src) as zin,
        zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zout,
    ):
        for info in zin.infolist():
            zout.writestr(info, edit(info.filename, zin.read(info.filename)))


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Snapshot and restore the registry's name -> class mapping so tests can
    reuse names freely.

    This isolates registered NAMES only: registering a class wraps that
    class's ``__init__`` in place, and that wrapping is never undone when the
    snapshot is restored. A class defined at module scope and registered by
    one test would still carry that wrapped ``__init__`` in a later test, so
    tests that register a name must define the class inside the test body (a
    fresh class object each time), not reuse a module-level one.

    This fixture runs once per test *function*, not once per Hypothesis
    example inside a ``@given``-decorated test; registrations made across
    examples within the same test accumulate in the same snapshot.
    """
    from equine import registry

    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))


@eq.embedding_architecture("equine.tests.basic")
class BasicEmbeddingModel(torch.nn.Module):
    def __init__(self, tensor_dim: int, num_classes: int) -> None:
        super(BasicEmbeddingModel, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(tensor_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear_relu_stack(x)
        return logits


@st.composite
def random_dataset(draw):
    dataset_row_count = draw(st.integers(min_value=100, max_value=100))
    dataset_col_count = draw(st.integers(min_value=1, max_value=1000))
    shape = (dataset_row_count, dataset_col_count)
    dataset_x = torch.rand(shape)

    num_classes = draw(
        st.integers(min_value=3, max_value=int(dataset_row_count / 30))
    )  # requires at least 30 examples per class
    way = draw(st.integers(min_value=2, max_value=num_classes))
    dataset_y = []
    for i in range(num_classes):
        dataset_y += [i] * int(dataset_row_count / num_classes)

    dataset_y += [0] * (dataset_row_count - len(dataset_y))
    dataset_y = torch.Tensor(dataset_y)

    dataset = torch.utils.data.TensorDataset(dataset_x, dataset_y)  # type: ignore

    return dataset, num_classes, way


def use_basic_embedding_model(random_dataset):
    dataset, num_classes, _ = random_dataset
    X, _ = dataset.tensors
    embedding_model = BasicEmbeddingModel(X.shape[1], num_classes)
    return dataset, num_classes, X, embedding_model


def use_save_load_model_tests(model, X, tmp_filename: str = "tmp.eq"):
    old_output = model.predict(X[1:10])
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)
    model.save(tmp_filename)
    new_model = eq.load_equine_model(tmp_filename)
    new_output = new_model.predict(X[1:10])
    assert (
        torch.nn.functional.mse_loss(old_output.classes, new_output.classes) <= 1e-7
    ), "Class predictions changed on reload"
    assert (
        torch.nn.functional.mse_loss(old_output.ood_scores, new_output.ood_scores)
        <= 1e-7
    ), "OOD predictions changed on reload"

    return new_model, tmp_filename


# return a list of random strings
# based off https://stackoverflow.com/a/34485032
def generate_random_string_list(list_length: int, str_length: int = 3):
    chars = ascii_lowercase + digits
    return [
        "".join(choice(chars) for _ in range(str_length)) for _ in range(list_length)
    ]
