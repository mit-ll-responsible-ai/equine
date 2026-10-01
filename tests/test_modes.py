# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Train/eval mode handling for both model classes (#209, #179).

``predict`` and ``update_support`` are inference entry points: they compute
in eval mode whatever mode the caller left the model in, and leave the model
in that mode afterwards; a GP ``predict`` in training mode used to accumulate
into the Laplace precision matrix (#209). ``Protonet.update_support`` must compute the global moments in either
mode so that a freshly constructed (training-mode) model can be given a
support set (#179). ``train_model`` must leave the wrapper and the inner
module in the same (eval) mode. CPU only, seeded, short training like
``tests/test_inference.py``.

The models are built on ``conftest.RecordingEmbedding`` (a dropout plus a
record of the mode each forward ran in), so "computes in eval mode" is
asserted directly: every forward an entry point makes reports eval mode, and
its result equals the one computed after ``model.eval()``. With a
mode-insensitive embedding these tests would pass with ``_eval_mode`` removed
from the entry points.
"""

import pytest
import torch
from conftest import BasicEmbeddingModel, RecordingEmbedding, assert_valid_prediction
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq


def _protonet():
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineProtonet(
        RecordingEmbedding(FEATURES, CLASSES), CLASSES, use_temperature=True
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


def _gp():
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineGP(
        RecordingEmbedding(FEATURES, CLASSES), CLASSES, CLASSES, num_random_features=16
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=2,
        batch_size=32,
    )
    return model, x, y


def _assert_same_output(actual: eq.EquineOutput, expected: eq.EquineOutput) -> None:
    torch.testing.assert_close(actual.classes, expected.classes)
    torch.testing.assert_close(actual.ood_scores, expected.ood_scores)
    torch.testing.assert_close(actual.embeddings, expected.embeddings)


def _ran_in_eval_mode(embedding: RecordingEmbedding) -> None:
    """Every embedding forward since ``modes`` was cleared saw eval mode."""
    assert embedding.modes, "the entry point did not embed anything"
    assert not any(embedding.modes), embedding.modes


_BUILDERS = pytest.mark.parametrize("build", [_protonet, _gp], ids=["protonet", "gp"])


def test_protonet_update_support_on_untrained_model():
    """A freshly constructed EquineProtonet accepts a support set (#179).

    Pins #179 end to end. The wrapper's switch to eval mode is enough for
    it, so this passes with the inner ``Protonet.update_support`` change
    reverted; that change (the global moments computed in training mode too)
    is guarded by
    ``test_protonet_update_support_in_train_mode_computes_global_moments``.
    The eval-mode computation and the mode restore are pinned by the
    ``*_restores_callers_mode`` tests below.
    """
    torch.manual_seed(0)
    _, x, y = separable_dataset()
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES)
    assert model.training  # nn.Module default: the configuration that used to fail
    model.update_support(x, y.float(), 0.5)
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


def test_protonet_update_support_in_train_mode_computes_global_moments():
    """The inner Protonet recomputes the global moments in training mode too (#179).

    A direct call to the inner ``update_support`` must leave the moments
    tracking the support in either mode, not only in eval mode. (Only the
    episode loop of ``train_model`` opts out, see the next test.)
    """
    model, x, y = _protonet()
    stale_mean = model.model.global_mean.clone()
    stale_cov = model.model.global_covariance.clone()
    model.train()
    support = eq.utils.generate_support(
        x, y, support_size=5, selected_labels=list(range(CLASSES))
    )
    model.model.update_support(support)
    assert model.model.global_mean.shape == stale_mean.shape
    assert not torch.equal(model.model.global_mean, stale_mean)
    assert model.model.global_covariance.shape == stale_cov.shape
    assert not torch.equal(model.model.global_covariance, stale_cov)
    assert torch.isfinite(model.model.global_covariance).all()
    assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)


def test_protonet_train_model_computes_global_moments_once(monkeypatch):
    """The episode loop skips the global moments; nothing reads them there.

    Episodes need only the prototypes and the covariance. The moments feed
    the OOD calibration, so the final full-support update of ``train_model``
    (in eval mode) is the one call that computes them.
    """
    real = eq.equine_protonet.Protonet.compute_global_moments
    modes: list[bool] = []

    def counting(self):
        modes.append(self.training)
        return real(self)

    monkeypatch.setattr(eq.equine_protonet.Protonet, "compute_global_moments", counting)
    model, _, _ = _protonet()  # 5 episodes
    assert modes == [False]
    assert model.model.global_mean.shape == (CLASSES,)


def test_gp_predict_after_train_mode_does_not_touch_precision():
    """predict() after model.train() leaves the Laplace buffers alone (#209)."""
    model, x, _ = _gp()
    model.train()
    precision = model.model.precision.clone()
    seen_data = model.model.seen_data.clone()
    for _ in range(3):
        assert_valid_prediction(model.predict(x[:5]), 5, CLASSES)
    torch.testing.assert_close(model.model.precision, precision, atol=0, rtol=0)
    torch.testing.assert_close(model.model.seen_data, seen_data, atol=0, rtol=0)


def test_gp_train_model_leaves_wrapper_and_inner_in_eval():
    model, _, _ = _gp()
    assert model.training is False
    assert model.model.training is False


def test_protonet_train_model_leaves_wrapper_and_inner_in_eval():
    model, _, _ = _protonet()
    assert model.training is False
    assert model.model.training is False


_MODES = pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])


def _set_mode(model, training):
    model.train(training)
    assert model.training is training and model.model.training is training


def _assert_mode(model, training):
    assert model.training is training
    assert model.model.training is training


@_MODES
@_BUILDERS
def test_predict_restores_callers_mode(build, training):
    """predict() computes in eval mode and leaves wrapper and inner module in
    the mode the caller had them in (#209)."""
    model, x, _ = build()
    _set_mode(model, training)
    model.embedding_model.modes.clear()
    out = model.predict(x[:5])
    _ran_in_eval_mode(model.embedding_model)
    _assert_mode(model, training)
    assert_valid_prediction(out, 5, CLASSES)
    model.eval()
    _assert_same_output(out, model.predict(x[:5]))


@_MODES
def test_protonet_update_support_restores_callers_mode(training):
    """update_support() computes the support covariance, the moments and the
    OOD calibration in eval mode (PRED_COV_TYPE, not the training cov_type),
    whatever mode the caller is in, and restores that mode (#179)."""
    model, x, y = _protonet()
    _set_mode(model, training)
    model.embedding_model.modes.clear()
    torch.manual_seed(1)  # the calibration split and support draw
    model.update_support(x, y.float(), 0.5)
    _ran_in_eval_mode(model.embedding_model)
    _assert_mode(model, training)
    covariance = model.model.covariance.clone()
    out = model.predict(x[:5])
    assert_valid_prediction(out, 5, CLASSES)

    model.eval()
    torch.manual_seed(1)
    model.update_support(x, y.float(), 0.5)
    assert torch.equal(model.model.covariance, covariance)
    _assert_same_output(out, model.predict(x[:5]))


@_MODES
def test_gp_update_support_restores_callers_mode(training):
    model, x, y = _gp()
    _set_mode(model, training)
    precision = model.model.precision.clone()
    seen_data = model.model.seen_data.clone()
    model.embedding_model.modes.clear()
    torch.manual_seed(1)  # the support draw
    model.update_support(x, y.long(), 10)
    _ran_in_eval_mode(model.embedding_model)
    _assert_mode(model, training)
    torch.testing.assert_close(model.model.precision, precision, atol=0, rtol=0)
    torch.testing.assert_close(model.model.seen_data, seen_data, atol=0, rtol=0)
    prototypes = model.prototypes.clone()
    out = model.predict(x[:5])
    assert_valid_prediction(out, 5, CLASSES)

    model.eval()
    torch.manual_seed(1)
    model.update_support(x, y.long(), 10)
    torch.testing.assert_close(model.prototypes, prototypes)
    _assert_same_output(out, model.predict(x[:5]))


def _predict(model, x, y):
    model.predict(x[:5])


def _update_support(model, x, y):
    if isinstance(model, eq.EquineProtonet):
        model.update_support(x, y.float(), 0.5)
    else:
        model.update_support(x, y.long(), 10)


@pytest.mark.parametrize(
    "entry_point", [_predict, _update_support], ids=["predict", "update_support"]
)
@_MODES
@_BUILDERS
def test_mode_restored_when_the_entry_point_raises(build, training, entry_point):
    """An exception inside predict/update_support still leaves the caller's mode in place."""
    model, x, y = build()
    _set_mode(model, training)
    model.embedding_model.fail_next = True
    with pytest.raises(RuntimeError, match="embedding failed"):
        entry_point(model, x, y)
    _assert_mode(model, training)
    assert model.embedding_model.training is training


def _batchnorm_embedding() -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(FEATURES, 16),
        torch.nn.BatchNorm1d(16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, CLASSES),
    )


def _protonet_with_batchnorm():
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineProtonet(_batchnorm_embedding(), CLASSES)
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    return model, x, y


def _gp_with_batchnorm():
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineGP(
        _batchnorm_embedding(), CLASSES, CLASSES, num_random_features=16
    )
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=2,
        batch_size=32,
    )
    return model, x, y


@pytest.mark.parametrize(
    "entry_point", [_predict, _update_support], ids=["predict", "update_support"]
)
@pytest.mark.parametrize(
    "build", [_protonet_with_batchnorm, _gp_with_batchnorm], ids=["protonet", "gp"]
)
def test_frozen_embedding_stays_frozen(build, entry_point):
    """The frozen-backbone fine-tune idiom survives an inference call in between.

    ``model.train(); model.embedding_model.eval()`` keeps BatchNorm statistics
    (and dropout) fixed while the head trains. predict/update_support must
    restore every submodule's own mode, not re-propagate the wrapper's: the
    embedding stays in eval mode and its running statistics are untouched.
    """
    model, x, y = build()
    bn = model.embedding_model[1]
    model.train()
    model.embedding_model.eval()
    running_mean = bn.running_mean.clone()
    entry_point(model, x, y)
    assert model.training is True
    assert model.embedding_model.training is False
    assert bn.training is False
    torch.testing.assert_close(bn.running_mean, running_mean, atol=0, rtol=0)
    # and the reverse mix is kept too
    model.eval()
    model.embedding_model.train()
    entry_point(model, x, y)
    assert model.training is False
    assert model.embedding_model.training is True


def test_gp_manual_fine_tune_loop_with_predict_per_epoch():
    """A hand-written fine-tune loop (model.train(), reset_precision_matrix(),
    training batches through model(xs), predict on validation data at the end
    of each epoch) runs for two epochs: predict does not consume the training
    budget of the precision matrix and does not knock the model out of
    training mode (#209)."""
    model, x, y = _gp()
    val = x[:5]
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    out = None
    for _ in range(2):
        model.model.reset_precision_matrix()
        for start in range(0, x.shape[0], 32):
            opt.zero_grad()
            xs = x[start : start + 32]
            ys = y[start : start + 32].long()
            loss = loss_fn(model(xs), ys)
            loss.backward()
            opt.step()
        assert model.training and model.model.training
        out = model.predict(val)
        assert model.training and model.model.training
    assert out is not None
    assert_valid_prediction(out, 5, CLASSES)


def test_gp_train_mode_forward_invalidates_the_cached_covariance():
    """A predict issued mid-epoch must not freeze the Laplace covariance for the rest of the epoch (#209).

    predict computes in eval mode, which caches the covariance from the
    precision matrix accumulated so far and clears ``recompute_covariance``.
    Training forwards after it keep accumulating, so they must mark the cache
    stale; otherwise every later eval-mode predict reuses a covariance that
    misses the epoch's last batches.
    """
    model, x, _ = _gp()
    laplace = model.model
    num_data = laplace.num_data
    assert num_data == x.shape[0] and laplace.train_batch_size == 32
    model.train()
    laplace.reset_precision_matrix()
    with torch.no_grad():
        for start in range(0, 96, 32):  # 96 > num_data - train_batch_size
            model(x[start : start + 32])
    model.predict(x[:5])  # allowed here; caches a covariance from 96 rows
    assert laplace.recompute_covariance is False
    with torch.no_grad():
        model(x[96:num_data])  # the epoch's last batch
    assert laplace.recompute_covariance is True
    model.eval()
    out = model.predict(x[:5])
    covariance = laplace.covariance.clone()
    laplace.recompute_covariance = True  # force a recompute from the full precision
    reference = model.predict(x[:5])
    assert torch.equal(laplace.covariance, covariance)
    _assert_same_output(out, reference)
