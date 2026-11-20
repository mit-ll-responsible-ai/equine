# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""EquineProtonet's model head: a ``Linear(emb_out_dim, emb_out_dim)`` built as the
identity without drawing random numbers, computing in the dtype of its input, trained
by ``train_model``, and saved and loaded with its weights."""

import copy

import pytest
import torch
from conftest import BasicEmbeddingModel, available_devices
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq
from equine import _model_file


class _FirstFeatures(torch.nn.Module):
    """An embedding model without parameters: the first ``width`` features."""

    def __init__(self, width: int = CLASSES) -> None:
        super().__init__()
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : self.width] * 1


def _basic() -> torch.nn.Module:
    return BasicEmbeddingModel(FEATURES, CLASSES)


# id -> (default dtype at construction, embedding, dtype the model is cast to, data dtype)
_DTYPE_CASES = {
    "float32": (torch.float32, _basic, None, torch.float32),
    "float64-model": (torch.float32, _basic, torch.float64, torch.float64),
    "float64-default": (torch.float64, _basic, None, torch.float64),
    "float64-embedding-only": (
        torch.float32,
        lambda: _basic().double(),
        None,
        torch.float64,
    ),
    "bfloat16-model": (torch.float32, _basic, torch.bfloat16, torch.bfloat16),
    "bfloat16-embedding-only": (
        torch.float32,
        lambda: _basic().to(torch.bfloat16),
        None,
        torch.bfloat16,
    ),
    "float16-model": (torch.float32, _basic, torch.float16, torch.float16),
    "float16-embedding-only": (
        torch.float32,
        lambda: _basic().half(),
        None,
        torch.float16,
    ),
    "no-parameters-float32": (torch.float32, _FirstFeatures, None, torch.float32),
    "no-parameters-float64": (torch.float32, _FirstFeatures, None, torch.float64),
    "no-parameters-bfloat16": (torch.float32, _FirstFeatures, None, torch.bfloat16),
}


def _devices() -> list:
    return [
        pytest.param(device, marks=[] if device == "cpu" else [pytest.mark.accelerator])
        for device in available_devices()
    ]


def _build(
    case: str, device: str
) -> tuple[eq.EquineProtonet, torch.Tensor, torch.Tensor]:
    """An untrained model for ``case`` on ``device``, and the data in its dtype."""
    default, embedding, cast, dtype = _DTYPE_CASES[case]
    if device != "cpu" and dtype == torch.float64:
        pytest.skip(f"{device} has no float64")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(default)
    try:
        torch.manual_seed(0)
        model = eq.EquineProtonet(embedding(), CLASSES, device=device)
        if cast is not None:
            model.to(cast)
    finally:
        torch.set_default_dtype(previous)
    _, x, y = separable_dataset()
    return model, x.to(device=device, dtype=dtype), y.to(device).float()


@pytest.mark.parametrize("default", [torch.float32, torch.float64], ids=str)
def test_head_is_an_identity_linear_built_without_random_numbers(default) -> None:
    """The constructor leaves the global RNG state as it was. Built with
    uninitialised memory filled with NaN, the head is a ``Linear(CLASSES,
    CLASSES)`` with weight I and bias 0 in the default dtype."""
    previous = torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.set_default_dtype(default)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        embedding = _basic()
        state = torch.get_rng_state()
        model = eq.EquineProtonet(embedding, CLASSES)
        assert torch.equal(torch.get_rng_state(), state)
    finally:
        torch.set_default_dtype(previous)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill
    head = model.model.model_head
    assert isinstance(head, torch.nn.Linear)
    assert (head.in_features, head.out_features) == (CLASSES, CLASSES)
    assert torch.equal(head.weight, torch.eye(CLASSES, dtype=default))
    assert torch.equal(head.bias, torch.zeros(CLASSES, dtype=default))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("case", list(_DTYPE_CASES))
def test_untrained_head_predicts_as_an_identity_head(case, device) -> None:
    """Before training, ``update_support`` and ``predict`` give the same bits as the
    same model with a ``torch.nn.Identity`` head, and the embeddings keep the
    embedding model's dtype."""
    model, x, y = _build(case, device)
    reference = copy.deepcopy(model)
    reference.model.model_head = torch.nn.Identity()
    support = torch.cat([torch.arange(k * 40, k * 40 + 20) for k in range(CLASSES)])
    calib = torch.cat([torch.arange(k * 40 + 20, k * 40 + 35) for k in range(CLASSES)])
    queries = x[
        torch.cat([torch.arange(k * 40 + 35, k * 40 + 40) for k in range(CLASSES)])
    ]
    for m in (model, reference):
        torch.manual_seed(1)
        m.update_support(x[support], y[support], calib_x=x[calib], calib_y=y[calib])
    with torch.no_grad():
        embeddings = model.model.compute_embeddings(queries)
        expected = reference.model.compute_embeddings(queries)
    assert embeddings.dtype == expected.dtype
    assert torch.equal(embeddings, expected)
    out, ref = model.predict(queries), reference.predict(queries)
    assert out.classes.dtype == ref.classes.dtype
    assert torch.equal(out.classes, ref.classes)
    assert torch.equal(out.ood_scores, ref.ood_scores)


@pytest.mark.parametrize("embedding", ["frozen", "no-parameters"])
def test_train_model_trains_the_head_of_a_frozen_embedding(embedding) -> None:
    """``train_model`` on a frozen embedding model, or one without parameters, trains
    the head and leaves the embedding model's weights as they were."""
    torch.manual_seed(0)
    X, Y = torch.randn(400, 16), torch.arange(400) % 4
    if embedding == "frozen":
        backbone = torch.nn.Linear(16, 8).requires_grad_(False)
    else:
        backbone = _FirstFeatures(8)
    before = [p.detach().clone() for p in backbone.parameters()]
    model = eq.EquineProtonet(backbone, emb_out_dim=8)
    model.train_model(
        torch.utils.data.TensorDataset(X, Y),
        num_episodes=10,
        way=4,
        support_size=10,
        episode_size=40,
    )
    assert not torch.equal(model.model.model_head.weight, torch.eye(8))
    assert all(torch.equal(a, b) for a, b in zip(before, backbone.parameters()))


@pytest.mark.parametrize("emb_out_dim", [1, 2, 4])
def test_emb_out_dim_other_than_the_embedding_width_is_refused(
    tmp_path, emb_out_dim
) -> None:
    """The embedding model outputs 3 features. With another ``emb_out_dim``,
    ``update_support`` raises ``ValueError`` naming both; built with
    ``emb_out_dim=3``, the model takes the same support, saves, and loads."""
    _, x, y = separable_dataset()
    model = eq.EquineProtonet(_basic(), emb_out_dim)
    with pytest.raises(
        ValueError,
        match=rf"has shape \(\d+, 3\), but the model head takes "
        rf"emb_out_dim={emb_out_dim} features in its last dimension; build the "
        "model with emb_out_dim set to",
    ):
        model.update_support(x, y.float(), calib_frac=0.5)
    repaired = eq.EquineProtonet(_basic(), 3)
    repaired.update_support(x, y.float(), calib_frac=0.5)
    path = str(tmp_path / "m.eq")
    repaired.save(path)
    loaded = eq.EquineProtonet.load(path)
    assert torch.equal(loaded.predict(x[:9]).classes, repaired.predict(x[:9]).classes)


# id -> (default dtype at construction, embedding, dtype the model is cast to,
# data dtype, trained with train_model (else update_support only), device)
_ROUND_TRIPS = {
    "float32": (torch.float32, "basic", None, torch.float32, True, "cpu"),
    "float64-model": (
        torch.float32,
        "basic",
        torch.float64,
        torch.float64,
        True,
        "cpu",
    ),
    "float64-default": (torch.float64, "basic", None, torch.float64, True, "cpu"),
    "float64-embedding-only": (
        torch.float32,
        "basic-float64",
        None,
        torch.float64,
        True,
        "cpu",
    ),
    "bfloat16-model": (
        torch.float32,
        "basic",
        torch.bfloat16,
        torch.bfloat16,
        True,
        "cpu",
    ),
    "float16-model-untrained": (
        torch.float32,
        "basic",
        torch.float16,
        torch.float16,
        False,
        "cpu",
    ),
    "no-parameters": (torch.float32, "no-parameters", None, torch.float32, True, "cpu"),
    "float32-accelerator": (
        torch.float32,
        "basic",
        None,
        torch.float32,
        True,
        "accelerator",
    ),
}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            marks=[pytest.mark.accelerator] if case.endswith("accelerator") else [],
        )
        for case in _ROUND_TRIPS
    ],
)
def test_head_weights_round_trip_exactly(tmp_path, case) -> None:
    """For each dtype, embedding and device the constructor takes, ``save`` stores
    the head weights in their dtype; ``load`` restores them, and the loaded model
    predicts the same bits."""
    default, embedding, cast, dtype, train, device = _ROUND_TRIPS[case]
    if device == "accelerator":
        device = next((d for d in available_devices() if d != "cpu"), "")
        if not device:
            pytest.skip("no accelerator")

    @eq.embedding_architecture("equine.tests.first_features")
    class FirstFeatures(_FirstFeatures):
        def __init__(self, width: int) -> None:
            super().__init__(width)

    previous = torch.get_default_dtype()
    torch.set_default_dtype(default)
    try:
        torch.manual_seed(0)
        backbone = {
            "basic": _basic,
            "basic-float64": lambda: _basic().double(),
            "no-parameters": lambda: FirstFeatures(CLASSES),
        }[embedding]()
        model = eq.EquineProtonet(backbone, CLASSES, device=device)
    finally:
        torch.set_default_dtype(previous)
    if cast is not None:
        model.to(cast)
    head = model.model.model_head
    _, x, y = separable_dataset()
    x, y = x.to(device=device, dtype=dtype), y.to(device).float()
    if train:
        model.train_model(
            torch.utils.data.TensorDataset(x, y),
            num_episodes=5,
            calib_frac=0.2,
            support_size=10,
            way=3,
            episode_size=30,
        )
        assert not torch.equal(
            head.weight.cpu(), torch.eye(CLASSES, dtype=head.weight.dtype)
        )
    else:
        model.update_support(x, y, calib_frac=0.5)
    path = str(tmp_path / "m.eq")
    model.save(path)
    stored = _model_file.read(path)["model_head_save"]
    assert set(stored) == {"weight", "bias"}
    assert {t.dtype for t in stored.values()} == {head.weight.dtype}
    loaded = eq.EquineProtonet.load(path)
    for name, tensor in head.state_dict().items():
        back = loaded.model.model_head.state_dict()[name]
        assert back.dtype == tensor.dtype
        assert torch.equal(back.cpu(), tensor.cpu())
    out, back = model.predict(x[:9]), loaded.predict(x[:9])
    assert torch.equal(out.classes, back.classes)
    assert torch.equal(out.ood_scores, back.ood_scores)


@pytest.mark.parametrize("default", [torch.float32, torch.float64], ids=str)
def test_file_without_head_weights_loads_the_identity_head(tmp_path, default) -> None:
    """A model whose head is a ``torch.nn.Identity`` saves no head weights. Loaded
    under the default dtype it was trained in, the file has the identity head and
    predicts the same bits; loaded under float32, the head returns the embedding
    model's output in its dtype."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(default)
    try:
        torch.manual_seed(0)
        _, x, y = separable_dataset()
        model = eq.EquineProtonet(_basic(), CLASSES)
        model.model.model_head = torch.nn.Identity()
        model.train_model(
            torch.utils.data.TensorDataset(x, y.float()),
            num_episodes=5,
            calib_frac=0.2,
            support_size=10,
            way=3,
            episode_size=30,
        )
        path = str(tmp_path / "m.eq")
        model.save(path)
        assert _model_file.read(path)["model_head_save"] == {}
        loaded = eq.EquineProtonet.load(path)
    finally:
        torch.set_default_dtype(previous)
    head = loaded.model.model_head
    assert isinstance(head, torch.nn.Linear)
    assert torch.equal(head.weight, torch.eye(CLASSES, dtype=default))
    assert torch.equal(head.bias, torch.zeros(CLASSES, dtype=default))
    out, back = model.predict(x[:9]), loaded.predict(x[:9])
    assert torch.equal(out.classes, back.classes)
    assert torch.equal(out.ood_scores, back.ood_scores)
    under_float32 = eq.EquineProtonet.load(path)
    with torch.no_grad():
        embeddings = under_float32.model.compute_embeddings(x[:9])
        expected = under_float32.embedding_model(x[:9])
    assert embeddings.dtype == expected.dtype == default
    assert torch.equal(embeddings, expected)


def test_head_weights_that_are_not_a_dict_are_refused(tmp_path) -> None:
    """A ``model_head_save`` that is not a dict of tensors is refused as malformed."""
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = eq.EquineProtonet(_basic(), CLASSES)
    model.update_support(x, y.float(), calib_frac=0.5)
    path = str(tmp_path / "m.eq")
    model.save(path)
    checkpoint = _model_file.read(path)
    checkpoint["model_head_save"] = [torch.eye(CLASSES)]
    _model_file.write(path, checkpoint)
    for loader in (eq.EquineProtonet.load, eq.load_equine_model):
        with pytest.raises(ValueError, match="model_head_save is malformed"):
            loader(path)


def test_load_measures_the_embedding_width_in_eval_mode(tmp_path) -> None:
    """A model whose embedding model holds a BatchNorm layer loads with the running
    statistics it was saved with and predicts the same bits."""

    @eq.embedding_architecture("equine.tests.batch_norm")
    class Normed(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(FEATURES, width)
            self.norm = torch.nn.BatchNorm1d(width)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(self.linear(x))

    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = eq.EquineProtonet(Normed(CLASSES), CLASSES)
    model.train_model(
        torch.utils.data.TensorDataset(x, y.float()),
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    path = str(tmp_path / "m.eq")
    model.save(path)
    loaded = eq.EquineProtonet.load(path)
    saved = model.embedding_model.state_dict()
    for name, tensor in loaded.embedding_model.state_dict().items():
        assert torch.equal(tensor, saved[name]), name
    out, back = model.predict(x[:9]), loaded.predict(x[:9])
    assert torch.equal(out.classes, back.classes)
    assert torch.equal(out.ood_scores, back.ood_scores)


@pytest.mark.accelerator
def test_load_under_an_accelerator_default_device(tmp_path) -> None:
    """A file saved on the CPU loads on the CPU while an accelerator is torch's
    default device, and predicts the same bits."""
    device = next((d for d in available_devices() if d != "cpu"), "")
    if not device:
        pytest.skip("no accelerator")
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = eq.EquineProtonet(_basic(), CLASSES)
    model.update_support(x, y.float(), calib_frac=0.5)
    path = str(tmp_path / "m.eq")
    model.save(path)
    with torch.device(device):
        loaded = eq.EquineProtonet.load(path)
    assert loaded.device == "cpu"
    out, back = model.predict(x[:9]), loaded.predict(x[:9])
    assert torch.equal(out.classes, back.classes)
    assert torch.equal(out.ood_scores, back.ood_scores)


def test_load_moves_the_embedding_to_the_cpu_before_measuring_its_width(
    tmp_path,
) -> None:
    """``load`` moves the embedding model to the CPU (``.to("cpu")``) before it runs the
    support row that measures the embedding's width."""
    events: list[str] = []

    @eq.embedding_architecture("equine.tests.recording_moves")
    class Recording(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(FEATURES, width)

        def to(self, *args, **kwargs):
            events.append(f"to {args[0] if args else kwargs}")
            return super().to(*args, **kwargs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            events.append("forward")
            return self.linear(x)

    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = eq.EquineProtonet(Recording(CLASSES), CLASSES)
    model.update_support(x, y.float(), calib_frac=0.5)
    path = str(tmp_path / "m.eq")
    model.save(path)
    events.clear()
    eq.EquineProtonet.load(path)
    assert events[:2] == ["to cpu", "forward"]
