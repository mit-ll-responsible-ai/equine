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
    config.addinivalue_line(
        "markers",
        "device: parametrized over available_devices() (informational)",
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


@pytest.fixture(autouse=True)
def _restore_torch_rng_state():
    """Undo the reseeding done by ``random_dataset`` so later tests don't inherit RNG state.

    Runs once per test *function*, not once per Hypothesis example inside a
    ``@given``-decorated test.
    """
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


@pytest.fixture(scope="session", autouse=True)
def _no_stray_model_files_in_cwd():
    """Regression guard for #224: the test session must not leave ``*.eq`` files in cwd.

    Snapshots the ``*.eq`` files in the working directory when the session
    starts and, at session teardown, fails if any new ones appeared. Under
    xdist this runs on every worker at that worker's end, so every writer is
    covered no matter where it was collected.
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


class CountingEmbedding(torch.nn.Module):
    """``BasicEmbeddingModel`` that counts its ``forward`` calls (#173, #182, #212).

    Not registered in the architecture registry: tests build it directly and
    read ``calls`` to assert how many embedding passes an operation made.
    """

    def __init__(self, tensor_dim: int, num_classes: int) -> None:
        super().__init__()
        self.inner = BasicEmbeddingModel(tensor_dim, num_classes)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.inner(x)


class RecordingEmbedding(torch.nn.Module):
    """``BasicEmbeddingModel`` behind a ``Dropout(0.5)`` that records the mode of every ``forward`` (#209, #179).

    Mode-sensitive on purpose: in training mode the dropout makes two forwards
    of the same input differ, so a test can tell whether an entry point
    computed in eval mode. ``modes`` holds ``self.training`` as seen by each
    forward; ``fail_next`` makes the next forward raise ``RuntimeError`` once,
    to check what an entry point leaves behind when its body raises. Not
    registered in the architecture registry.
    """

    def __init__(self, tensor_dim: int, num_classes: int) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
        self.inner = BasicEmbeddingModel(tensor_dim, num_classes)
        self.modes: list[bool] = []
        self.fail_next = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("embedding failed")
        self.modes.append(self.training)
        return self.inner(self.dropout(x))


@st.composite
def random_dataset(draw):
    """A labelled dataset plus the training arguments that fit its shape.

    Returns ``(dataset, num_classes, train_kwargs)``. ``train_kwargs`` is passed
    to ``EquineProtonet.train_model`` (GP tests take what they need from it):
    the defaults (way=3, support_size=25, episode_size=100) do not fit every
    shape this strategy draws (2 classes < way=3; 30-row classes keep only 24
    training rows < support_size=25), so generate_episode would raise. Torch is
    seeded from a drawn integer so hypothesis can replay and shrink failing
    examples. This reseeds the process-global torch RNG; tests that run later
    in the same worker inherit that state unless restored (see
    ``_restore_torch_rng_state``).

    Shape: 2..5 balanced classes, every class has >= 30 rows, 120 <= rows <= 200.
    """
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    torch.manual_seed(seed)
    num_classes = draw(st.integers(min_value=2, max_value=5))
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

    # train_model splits with stratified_train_test_split, which holds out
    # round(count * calib_frac) rows of every class, so each class keeps
    # r - round(0.2 r) >= floor(0.8 r) training rows; per_class_train is that
    # lower bound. At the current shape bounds it is >= 24, so support_size is
    # always 10 and leaves >= 14 query rows per class; the min() on support_size
    # only engages if the shape bounds are lowered. The min() on episode_size
    # does engage for 4-5 classes with 30-33 rows per class.
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
    """Shape and value checks on an EquineOutput from predict().

    predict() output only; do not use on forward(), which will return logits.
    """
    assert out.classes.shape == (num_rows, num_classes)
    assert out.ood_scores.shape == (num_rows,)
    assert torch.isfinite(out.classes).all()
    assert torch.isfinite(out.ood_scores).all()
    assert torch.all(out.classes >= 0) and torch.all(out.classes <= 1)
    row_sums = out.classes.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert torch.all(out.ood_scores >= 0) and torch.all(out.ood_scores <= 1)


def available_devices() -> list[str]:
    """CPU plus whichever accelerator this machine has. CI (Ubuntu) has none."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def stored_tensors(model: eq.Equine) -> dict[str, torch.Tensor]:
    """Every tensor a trained equine model keeps: parameters, buffers, support,
    prototypes, covariances. Used to assert device placement after train/load.

    The derived tensors live on the inner module for the Protonet
    (``model.model.prototypes`` ...) and on the wrapper for the GP
    (``model.prototypes``, ``model.support``); both are looked up and a
    missing attribute is simply absent from the result.
    """
    found: dict[str, torch.Tensor] = {}
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        found[name] = t
    holders = [("", model)]
    inner = getattr(model, "model", None)
    if inner is not None:
        holders.append(("model.", inner))
    for attr in ("prototypes", "covariance", "global_mean", "global_covariance"):
        for prefix, holder in holders:
            t = getattr(holder, attr, None)
            if torch.is_tensor(t) and t.numel() > 0:
                found[f"{prefix}{attr}"] = t
    for attr in ("support", "support_embeddings"):
        for prefix, holder in holders:
            d = getattr(holder, attr, None) or {}
            for label, t in d.items():
                if torch.is_tensor(t) and t.numel() > 0:
                    found[f"{prefix}{attr}[{label}]"] = t
    return found


def assert_on_device(model: eq.Equine, device: str) -> None:
    """Fail unless every tensor in ``stored_tensors(model)`` is on ``device``."""
    wrong = {
        name: str(t.device)
        for name, t in stored_tensors(model).items()
        if t.device.type != torch.device(device).type
    }
    assert wrong == {}, f"tensors not on {device}: {wrong}"


def use_save_load_model_tests(model, X, tmp_filename: str = "tmp.eq"):
    """Save, reload through load_equine_model, and assert predictions are unchanged.

    Writes into a temporary directory that is removed on return. Not a pytest
    fixture on purpose: hypothesis' function_scoped_fixture health check rejects
    ``tmp_path`` inside ``@given`` tests.
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
