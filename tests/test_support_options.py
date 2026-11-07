# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""EquineProtonet's pred_cov_type and ood_cov_type: prediction, OOD scores and
save/load."""

import pickle

import pytest
import torch
from conftest import BasicEmbeddingModel
from golden_data import CLASSES, FEATURES, separable_dataset

import equine as eq
from equine import CovType, _model_file


def _trained_protonet(**kwargs):
    torch.manual_seed(0)
    dataset, x, y = separable_dataset()
    model = eq.EquineProtonet(BasicEmbeddingModel(FEATURES, CLASSES), CLASSES, **kwargs)
    model.train_model(
        dataset, num_episodes=5, calib_frac=0.2, support_size=10, way=3, episode_size=30
    )
    return model, x, y.float()


def _heads(trained, x, y, option):
    """Heads on ``trained``'s embedding, one per covariance type for ``option``,
    given the same support."""
    heads = {}
    for cov in (CovType.DIAGONAL, CovType.UNIT):
        torch.manual_seed(0)
        heads[cov] = eq.EquineProtonet(
            trained.embedding_model, CLASSES, **{option: cov}
        )
        heads[cov].update_support(x, y, calib_frac=0.2)
        heads[cov].eval()
    return heads


@pytest.mark.parametrize(
    "pred,ood", [(CovType.UNIT, CovType.UNIT), (CovType.DIAGONAL, CovType.UNIT)]
)
def test_covariance_types_run_and_round_trip(tmp_path, pred, ood) -> None:
    model, x, _ = _trained_protonet(
        pred_cov_type=pred, ood_cov_type=ood, use_temperature=True
    )
    assert model.pred_cov_type == pred and model.ood_cov_type == ood
    out = model.predict(x[:10])
    assert out.classes.shape == (10, CLASSES)
    assert torch.all((out.ood_scores >= 0) & (out.ood_scores <= 1))
    path = str(tmp_path / "cov.eq")
    model.save(path)
    loaded = eq.load_equine_model(path)
    assert loaded.pred_cov_type == pred and loaded.ood_cov_type == ood
    assert torch.allclose(loaded.predict(x[:10]).classes, out.classes, atol=1e-6)
    assert torch.allclose(loaded.predict(x[:10]).ood_scores, out.ood_scores, atol=1e-6)


def test_a_default_model_saves_neither_covariance_type(tmp_path) -> None:
    model, _, _ = _trained_protonet()
    path = str(tmp_path / "m.eq")
    model.save(path)
    settings = _model_file.read(path)["settings"]
    assert "pred_cov_type" not in settings
    assert "ood_cov_type" not in settings


def test_unit_covariance_changes_predictions() -> None:
    trained, x, y = _trained_protonet()
    heads = _heads(trained, x, y, "pred_cov_type")
    diag, unit = (
        heads[CovType.DIAGONAL].model.covariance,
        heads[CovType.UNIT].model.covariance,
    )
    assert unit.shape == diag.shape
    assert torch.all(unit == unit.flatten()[0])
    assert not torch.all(diag == diag.flatten()[0])
    probs = {cov: head.predict(x).classes for cov, head in heads.items()}
    assert not torch.allclose(probs[CovType.DIAGONAL], probs[CovType.UNIT], atol=1e-4)


def test_ood_covariance_type_sets_the_global_moments() -> None:
    trained, x, y = _trained_protonet()
    heads = _heads(trained, x, y, "ood_cov_type")
    diag, unit = (
        heads[CovType.DIAGONAL].model.global_covariance,
        heads[CovType.UNIT].model.global_covariance,
    )
    assert torch.all(unit == unit.flatten()[0])
    assert not torch.all(diag == diag.flatten()[0])
    scores = {cov: head.predict(x).ood_scores for cov, head in heads.items()}
    assert not torch.allclose(scores[CovType.DIAGONAL], scores[CovType.UNIT])


def test_a_pickle_without_covariance_types_restores_the_defaults() -> None:
    model, x, y = _trained_protonet()
    del model.model.__dict__["pred_cov_type"], model.model.__dict__["ood_cov_type"]
    restored = pickle.loads(pickle.dumps(model))
    assert restored.pred_cov_type == CovType.DIAGONAL
    assert restored.ood_cov_type == CovType.DIAGONAL
    restored.update_support(x, y, calib_frac=0.2)


@pytest.mark.parametrize("key", ["cov_type", "pred_cov_type", "ood_cov_type"])
def test_load_refuses_a_covariance_type_and_says_how_to_repair(tmp_path, key) -> None:
    model, _, _ = _trained_protonet()
    path = str(tmp_path / "m.eq")
    model.save(path)
    saved = _model_file.read(path)
    saved["settings"][key] = "spherical"
    _model_file.write(path, saved)
    with pytest.raises(
        ValueError,
        match=rf"settings\['{key}'\] .* is not one of .* Save the model again",
    ):
        eq.load_equine_model(path)
    model.save(path)
    assert isinstance(eq.load_equine_model(path), eq.EquineProtonet)
