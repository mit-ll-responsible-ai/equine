# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Golden-value tests: seeded models must keep producing these exact numbers.

The golden models are built, trained and queried in float64 (see
golden_data.golden_dtype).
"""

import pytest
import torch
from golden_data import (
    far_ood_batch,
    golden_dtype,
    query_batch,
    separable_dataset,
    trained_gp,
    trained_protonet,
)

# Generated inside golden_dtype() by
# golden_data.trained_*().predict(golden_data.query_batch()) on CPU.
PROTONET_CLASSES = [
    [1.0, 1.7414158e-16, 1.3058541e-11],
    [4.3842212e-07, 0.9999992, 3.6218552e-07],
    [3.4795155e-06, 2.5310339e-07, 0.99999627],
]
PROTONET_OOD = [0.93407185, 0.18135909, 0.74240538]
GP_CLASSES = [
    [0.76350259, 0.13617651, 0.1003209],
    [0.13126601, 0.77616825, 0.092565735],
    [0.1201439, 0.11438467, 0.76547142],
]
GP_OOD = [0.6446385, 0.62214935, 0.64370546]

# Absolute tolerance of every golden comparison (they pass rtol=0).
GOLDEN_ATOL = 1e-6


@pytest.mark.parametrize(
    "make, expected_classes, expected_ood",
    [
        pytest.param(trained_protonet, PROTONET_CLASSES, PROTONET_OOD, id="protonet"),
        pytest.param(trained_gp, GP_CLASSES, GP_OOD, id="gp"),
    ],
)
def test_predictions_match_golden_values(make, expected_classes, expected_ood) -> None:
    with golden_dtype():
        out = make().predict(query_batch())
    # class identity is platform-stable: one query per class, in class order
    assert out.classes.argmax(dim=1).tolist() == [0, 1, 2]
    assert torch.allclose(
        out.classes,
        torch.tensor(expected_classes, dtype=out.classes.dtype),
        atol=GOLDEN_ATOL,
        rtol=0,
    )
    assert torch.allclose(
        out.ood_scores,
        torch.tensor(expected_ood, dtype=out.ood_scores.dtype),
        atol=GOLDEN_ATOL,
        rtol=0,
    )


@pytest.mark.parametrize("make", [trained_protonet, trained_gp], ids=["protonet", "gp"])
def test_far_out_of_distribution_queries_score_higher(make) -> None:
    """OOD scores must separate far-OOD queries from in-distribution ones."""
    with golden_dtype():
        model = make()
        in_dist = model.predict(query_batch()).ood_scores
        far = model.predict(far_ood_batch()).ood_scores
    assert torch.all(in_dist >= 0) and torch.all(in_dist <= 1)
    assert torch.all(far >= 0) and torch.all(far <= 1)
    assert far.mean() > in_dist.mean()


def test_protonet_ood_scores_lie_in_unit_interval_on_training_data() -> None:
    with golden_dtype():
        _, x, _ = separable_dataset()
        ood = trained_protonet().predict(x).ood_scores
    assert torch.all(ood >= 0) and torch.all(ood <= 1)
