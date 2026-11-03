# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import glob
import math
import os
import tempfile
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
    Snapshot and restore the registry's name -> class mapping. Tests can reuse names
    freely.

    This isolates registered NAMES only: registering a class wraps that
    class's ``__init__`` in place, and that wrapping is never undone when the
    snapshot is restored. Tests that register a name must define the class inside the
    test body (a fresh class object each time), not reuse a module-level one.

    This fixture runs once per test *function*, not once per Hypothesis
    example inside a ``@given``-decorated test; registrations made across
    examples within the same test accumulate in the same snapshot.
    """
    from equine import registry

    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))


@pytest.fixture(autouse=True)
def _restore_torch_rng_state():
    """Restore the torch RNG state after each test, undoing any reseeding
    done by ``random_dataset``.

    Runs once per test *function*, not once per Hypothesis example inside a
    ``@given``-decorated test.
    """
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


@pytest.fixture(scope="session", autouse=True)
def _no_stray_model_files_in_cwd():
    """The test session must not leave ``*.eq`` files in cwd.

    Snapshots the ``*.eq`` files in the working directory when the session
    starts and, at session teardown, fails if any new ones appeared. Under
    xdist this runs on every worker at that worker's end and covers every
    writer, wherever it was collected.
    """
    before = sorted(glob.glob("*.eq"))
    yield
    new_files = sorted(set(glob.glob("*.eq")) - set(before))
    assert new_files == [], (
        f"tests wrote model files into the working directory: {new_files}"
    )


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
    """A labelled dataset plus the training arguments that fit its shape.

    Returns ``(dataset, num_classes, train_kwargs)``. ``train_kwargs`` is passed to
    ``EquineProtonet.train_model`` (GP tests take what they need from it) and is sized
    to the drawn dataset. Torch is seeded from a drawn integer. This reseeds the
    process-global torch RNG; tests that run later in the same worker inherit that state
    unless restored (see ``_restore_torch_rng_state``).

    Shape: 2..5 balanced classes, every class has >= 30 rows, 120 <= rows <= 200.
    """
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    torch.manual_seed(seed)
    num_classes = draw(st.sampled_from([3, 2, 4, 5]))
    rows_per_class = draw(
        st.integers(
            min_value=max(30, math.ceil(120 / num_classes)),
            max_value=200 // num_classes,
        )
    )
    rows = rows_per_class * num_classes
    cols = draw(st.integers(min_value=1, max_value=64))
    dataset_x = torch.rand(rows, cols)
    dataset_y = torch.arange(num_classes).repeat_interleave(rows_per_class).float()
    dataset = torch.utils.data.TensorDataset(dataset_x, dataset_y)  # type: ignore

    # Each class keeps at least per_class_train rows after train_model holds out
    # round(calib_frac * r) of them; the sizes below fit within that.
    calib_frac = 0.2
    per_class_train = int(rows_per_class * (1 - calib_frac))
    way = min(3, num_classes)
    support_size = min(10, per_class_train - 5)
    episode_size = min(50, way * (per_class_train - support_size))
    train_kwargs = {
        "calib_frac": calib_frac,
        "way": way,
        "support_size": support_size,
        "episode_size": episode_size,
    }
    return dataset, num_classes, train_kwargs


def use_basic_embedding_model(random_dataset):
    dataset, num_classes, train_kwargs = random_dataset
    X, _ = dataset.tensors
    embedding_model = BasicEmbeddingModel(X.shape[1], num_classes)
    return dataset, num_classes, X, embedding_model, train_kwargs


def assert_valid_prediction(
    out: eq.EquineOutput, num_rows: int, num_classes: int
) -> None:
    """Shape and value checks on the EquineOutput that predict() returns."""
    assert out.classes.shape == (num_rows, num_classes)
    assert out.ood_scores.shape == (num_rows,)
    assert torch.isfinite(out.classes).all()
    assert torch.isfinite(out.ood_scores).all()
    assert torch.all(out.classes >= 0) and torch.all(out.classes <= 1)
    row_sums = out.classes.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert torch.all(out.ood_scores >= 0) and torch.all(out.ood_scores <= 1)


def use_save_load_model_tests(model, X, tmp_filename: str = "tmp.eq"):
    """Save, reload through load_equine_model, and assert predictions are unchanged.

    Writes into a temporary directory that is removed on return.
    """
    old_output = model.predict(X[1:10])
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, tmp_filename)
        model.save(path)
        new_model = eq.load_equine_model(path)
    new_output = new_model.predict(X[1:10])
    assert (
        torch.nn.functional.mse_loss(old_output.classes, new_output.classes) <= 1e-7
    ), "Class predictions changed on reload"
    assert (
        torch.nn.functional.mse_loss(old_output.ood_scores, new_output.ood_scores)
        <= 1e-7
    ), "OOD predictions changed on reload"
    return new_model


# return a list of random strings
# based off https://stackoverflow.com/a/34485032
def generate_random_string_list(list_length: int, str_length: int = 3):
    chars = ascii_lowercase + digits
    return [
        "".join(choice(chars) for _ in range(str_length)) for _ in range(list_length)
    ]
