# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Inference-path tests for both model classes (#173, #182, #212).

``predict`` must run the embedding model once per call and return tensors
that carry no autograd graph (it runs under ``torch.no_grad()``, not
``torch.inference_mode()``, so a caller can still feed the outputs into
autograd). ``update_support`` must embed each support class once and, for
the Protonet, the calibration set once. CPU only, seeded, short training
like ``tests/test_devices.py``.
"""

from collections import OrderedDict

import icontract
import pytest
import torch
from conftest import CountingEmbedding
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq


def _protonet():
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    emb = CountingEmbedding(FEATURES, CLASSES)
    model = eq.EquineProtonet(emb, CLASSES, use_temperature=True)
    model.train_model(
        dataset,
        num_episodes=5,
        calib_frac=0.2,
        support_size=10,
        way=3,
        episode_size=30,
    )
    return model, emb, x, y


def _gp():
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    emb = CountingEmbedding(FEATURES, CLASSES)
    model = eq.EquineGP(emb, CLASSES, CLASSES, num_random_features=16)
    model.train_model(
        dataset,
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.05),
        num_epochs=2,
        batch_size=32,
    )
    return model, emb, x, y


_BUILDERS = pytest.mark.parametrize("build", [_protonet, _gp], ids=["protonet", "gp"])


@_BUILDERS
def test_predict_embeds_once(build):
    model, emb, x, _ = build()
    emb.calls = 0
    model.predict(x[:5])
    assert emb.calls == 1


@_BUILDERS
def test_predict_outputs_carry_no_autograd(build):
    model, _, x, _ = build()
    out = model.predict(x[:5])
    assert out.classes.grad_fn is None
    assert out.embeddings.grad_fn is None
    assert not out.ood_scores.requires_grad


@_BUILDERS
def test_predict_outputs_are_plain_tensors_usable_in_autograd(build):
    """Guards against ``torch.inference_mode()``: its tensors cannot enter autograd.

    Every output is used as ``predict`` returns it; a clone would turn an
    inference tensor back into an ordinary one and hide the regression.
    """
    model, _, x, _ = build()
    out = model.predict(x[:5])
    for t in (out.classes, out.ood_scores, out.embeddings):
        assert not t.is_inference()
        w = torch.ones((), requires_grad=True)
        (t * w).sum().backward()  # inference tensors cannot be saved for backward


def test_gp_update_support_embeds_once_per_class():
    model, emb, x, y = _gp()
    emb.calls = 0
    model.update_support(x, y.long(), 10)
    assert emb.calls == CLASSES


def test_protonet_update_support_embeds_calibration_set_once():
    """One pass over the calibration set plus one per support class."""
    model, emb, x, y = _protonet()
    emb.calls = 0
    model.update_support(x, y.float(), 0.5)
    assert emb.calls == 1 + CLASSES


def test_gp_untrained_forward_and_predict_violate_precondition():
    """The training-parameters precondition guards the single-pass helper, so
    neither entry point computes anything or touches the Laplace buffers."""
    torch.manual_seed(0)
    _, x, _ = separable_dataset()
    model = eq.EquineGP(CountingEmbedding(FEATURES, CLASSES), CLASSES, CLASSES)
    precision = model.model.precision.clone()
    seen_data = model.model.seen_data.clone()
    with pytest.raises(icontract.ViolationError):
        model.forward(x[:5])
    assert model.training and model.model.training  # nn.Module's default, kept
    with pytest.raises(icontract.ViolationError):
        model.predict(x[:5])
    assert model.training and model.model.training  # restored on the way out
    torch.testing.assert_close(model.model.precision, precision, atol=0, rtol=0)
    torch.testing.assert_close(model.model.seen_data, seen_data, atol=0, rtol=0)


def test_gp_compute_prototypes_reembeds_and_reflects_changed_support():
    """Public compute_prototypes keeps its base semantics: it embeds the
    support again (one pass per class) and reflects whatever the support is."""
    model, emb, x, y = _gp()
    model.update_support(x, y.long(), 10)
    emb.calls = 0
    first = model.compute_prototypes()
    assert emb.calls == CLASSES
    torch.testing.assert_close(first, model.prototypes)

    shifted = OrderedDict(
        (label, support + 1.0) for label, support in model.support.items()
    )
    model.support = shifted
    second = model.compute_prototypes()
    assert emb.calls == 2 * CLASSES
    assert not torch.allclose(first, second)
    expected = torch.stack(
        [model.compute_embeddings(s).mean(dim=0) for s in shifted.values()]
    )
    torch.testing.assert_close(second, expected)
    assert list(model.support_embeddings) == list(shifted)
