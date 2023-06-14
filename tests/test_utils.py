# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import equine as eq
import torch
from hypothesis import given
from hypothesis import strategies as st
import hypothesis.extra.numpy as hnp


@st.composite
def support_dataset(draw):
    support_sz = draw(st.integers(min_value=2, max_value=25))
    task_count = draw(st.integers(min_value=2, max_value=10))
    tasks = [x for x in range(task_count)]
    dataset_row_count = draw(
        st.integers(min_value=(support_sz + 1) * task_count, max_value=1000)
    )
    dataset_col_count = draw(st.integers(min_value=1, max_value=1000))
    shape = (dataset_row_count, dataset_col_count)
    dataset_x = torch.rand(shape)
    dataset_y = []
    for task in tasks:
        dataset_y += [task] * (support_sz + 1)

    if len(dataset_x) > len(dataset_y):
        pad = (
            torch.randint(0, task_count, (len(dataset_x) - len(dataset_y), 1))
            .flatten()
            .tolist()
        )
        dataset_y += pad

    dataset_y = torch.Tensor(dataset_y)
    way = draw(st.integers(min_value=2, max_value=task_count))

    return dataset_x, dataset_y, support_sz, tasks, way


@given(dataset=support_dataset())
def test_generate_support(dataset) -> None:
    train_x, train_y, support_sz, tasks, way = dataset
    eq.utils.generate_support(train_x, train_y, support_sz, tasks)


@st.composite
def draw_two_tensors(draw):
    num_classes = draw(st.integers(min_value=2, max_value=128))
    num_examples = draw(st.integers(min_value=2, max_value=100))
    yh = draw(
        hnp.arrays(
            shape=st.just((num_examples, num_classes)),
            dtype="float32",
            elements=st.floats(
                allow_nan=False,
                allow_infinity=False,
                width=32,
                min_value=9.9999998245167e-15,
                max_value=1.0,
            ),
        ).map(torch.tensor)
    )
    yh = yh / yh.sum(dim=1).unsqueeze(-1)
    true_class = draw(
        st.lists(
            st.integers(min_value=0, max_value=num_classes - 1),
            min_size=num_examples,
            max_size=num_examples,
        ).map(torch.tensor)
    )
    return yh, true_class


@given(draw_two_tensors())
def test_brier_score(two_tensors) -> None:
    yh, yt = two_tensors
    assert eq.utils.brier_score(yh, yt) >= 0.0


@given(draw_two_tensors())
def test_brier_skill_score(two_tensors) -> None:
    yh, yt = two_tensors
    assert eq.utils.brier_skill_score(yh, yt) <= 1.0


@given(draw_two_tensors())
def test_ece(two_tensors) -> None:
    yh, yt = two_tensors
    assert eq.utils.expected_calibration_error(yh, yt) <= 1.0


@given(draw_two_tensors())
def test_metric_summary(two_tensors) -> None:
    yh, yt = two_tensors
    eq_out = eq.EquineOutput(yh, torch.ones(yt.size()), torch.ones(yt.size()))
    metrics = eq.utils.generate_model_metrics(eq_out, yt)
    assert "accuracy" in metrics
    assert "microF1Score" in metrics
    assert "brierScore" in metrics
    assert "brierSkillScore" in metrics
    assert "expectedCalibrationError" in metrics
