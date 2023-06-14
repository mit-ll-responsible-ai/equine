# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import torch
from hypothesis import strategies as st


class BasicEmbeddingModel(torch.nn.Module):
    def __init__(self, tensor_dim, num_classes):
        super(BasicEmbeddingModel, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(tensor_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x):
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
