# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
import torch
from hypothesis import given, settings, strategies as st
import pytest
import os

import equine as eq
from conftest import BasicEmbeddingModel, random_dataset


@given(
    data_shape=st.tuples(
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
    ),
    num_classes=st.integers(min_value=2, max_value=256),
)
def test_compute_embeddings(data_shape, num_classes):
    queries = torch.rand(data_shape)
    embed_model = BasicEmbeddingModel(data_shape[1], num_classes)
    model = eq.EquineProtonet(embed_model, num_classes)
    model.model.compute_embeddings(queries)


@st.composite
def embeddings(draw):
    dataset_row_count = draw(st.integers(min_value=100, max_value=1000))
    dataset_col_count = draw(st.integers(min_value=3, max_value=10))
    shape = (dataset_row_count, dataset_col_count)
    support_embeddings = torch.rand(shape)
    query_embeddings = torch.rand(shape)
    ptrs = [0]
    curr_idx = 0

    for i in range(dataset_col_count - 1):
        curr_idx += int(dataset_row_count / dataset_col_count)
        ptrs.append(curr_idx)

    ptrs.append(dataset_row_count - 1)
    ptrs = torch.Tensor(ptrs).to(torch.long)

    return support_embeddings, ptrs, query_embeddings


# @given(
#     data_shape=st.tuples(
#         st.integers(min_value=1, max_value=1000),
#         st.integers(min_value=1, max_value=1000),
#     ),
#     num_classes=st.integers(min_value=2, max_value=256),
# )
# def test_compute_covariance(data_shape,num_classes):
#     queries = torch.rand(data_shape)
#     embed_model = BasicEmbeddingModel(data_shape[1], num_classes)
#     ep = eq.EquineProtonet(embed_model, num_classes)
#     embeddings = ep.model.compute_embeddings(queries)
#     ep.model.compute_covariance(embeddings, eq.CovType.UNIT)
#
#
# @given(gen_embeddings=embeddings())
# def test_compute_covariance_by_type(gen_embeddings):
#     embeddings, _, _ = gen_embeddings
#     protonet = eq.torch.Protonet()
#     protonet._compute_covariance_by_type(embeddings, protonet.train_cov)
#
#
# @given(gen_embeddings=embeddings())
# def test_compute_shared_covariance(gen_embeddings):
#     embeddings, ptrs, _ = gen_embeddings
#     protonet = eq.torch.Protonet()
#     cov = protonet._compute_covariance(embeddings, ptrs, protonet.train_cov)
#     protonet._compute_shared_covariance(embeddings, ptrs, cov)
#
#
# @given(gen_embeddings=embeddings())
# def test_compute_prototypes(gen_embeddings):
#     embeddings, ptrs, _ = gen_embeddings
#     protonet = eq.torch.Protonet()
#     protonet._compute_prototypes(embeddings, ptrs)
#
#
# @given(gen_embeddings=embeddings())
# def test_compute_distance(gen_embeddings):
#     embeddings, ptrs, _ = gen_embeddings
#     protonet = eq.torch.Protonet()
#     cov = protonet._compute_covariance(embeddings, ptrs, protonet.train_cov)
#     prototypes = protonet._compute_prototypes(embeddings, ptrs)
#     protonet._compute_distance(embeddings, prototypes, cov)
#
#
# @given(gen_embeddings=embeddings())
# def test_compute_classes(gen_embeddings):
#     embeddings, ptrs, _ = gen_embeddings
#     protonet = eq.torch.Protonet()
#     cov = protonet._compute_covariance(embeddings, ptrs, protonet.train_cov)
#     prototypes = protonet._compute_prototypes(embeddings, ptrs)
#     dist = protonet._compute_distance(embeddings, prototypes, cov)
#     protonet._compute_classes(dist)
#
#
# @given(random_dataset=random_dataset())
# def test_forward(random_dataset):
#     dataset, num_classes = random_dataset
#     X, _ = dataset.tensors
#     embed_model = BasicEmbeddingModel(X.shape[1], num_classes)
#     protonet = eq.EquineProtonet(embed_model, num_classes).model
#     results = protonet(query_embed, support_embed, support_ptrs)
#     assert len(results.classes) == len(query_embed), "Same number of outputs as inputs"


@given(random_dataset=random_dataset())
@settings(deadline=None)
def test_train_episodes(random_dataset):
    dataset, num_classes, way = random_dataset
    num_shot = 3
    num_episodes = 10
    episode_size = 512

    X, Y = dataset.tensors
    num_deep_features = 32
    embed_model = BasicEmbeddingModel(X.shape[1], num_deep_features)
    model = eq.EquineProtonet(embed_model, num_deep_features)
    model.train_model(
        torch.utils.data.TensorDataset(X, Y),
        way=way,
        support_size=num_shot,
        num_episodes=num_episodes,
        episode_size=episode_size,
    )

    assert model.model.training is False, "Model leaves training mode"
    assert len(model.model.support) == num_classes
    # Test on multiple predictions
    eq_out = model.predict(X)
    assert len(eq_out.classes) == len(X)
    assert len(eq_out.ood_scores) == len(X)
    # Test on single prediction
    pred_out = model(X[0])
    assert len(pred_out) == 1, "Single prediction works"
    eq_out = model.predict(X[0])
    assert len(eq_out.classes) == 1, "Single prediction works"


@given(random_dataset=random_dataset())
@settings(deadline=None)
def test_train_episodes_with_temperature(random_dataset):
    dataset, num_classes = random_dataset
    way = 3
    num_shot = 3
    num_episodes = 10
    episode_size = 512

    X, Y = dataset.tensors
    num_deep_features = 32
    embed_model = BasicEmbeddingModel(X.shape[1], num_deep_features)
    model = eq.EquineProtonet(embed_model, num_deep_features, use_temperature=True)
    _, cal_x, cal_y = model.train_model(
        torch.utils.data.TensorDataset(X, Y),
        way=way,
        support_size=num_shot,
        num_episodes=num_episodes,
        episode_size=episode_size,
    )

    assert model.model.training is False, "Model leaves training mode"
    # Test on multiple predictions
    eq_out = model.predict(X)
    assert len(eq_out.classes) == len(X)
    assert len(eq_out.ood_scores) == len(X)
    # Test on single prediction
    pred_out = model(X[0])
    assert len(pred_out) == 1, "Single prediction works"
    eq_out = model.predict(X[0])
    assert len(eq_out.classes) == 1, "Single prediction works"

    model.calibrate_temperature(cal_x, cal_y, 1, 0.01)


@given(random_dataset=random_dataset())
def test_predict_fail_before_training(random_dataset):
    dataset, num_classes, _ = random_dataset
    X, _ = dataset.tensors
    embed_model = BasicEmbeddingModel(X.shape[1], num_classes)
    model = eq.EquineProtonet(embed_model, num_classes)
    with pytest.raises(ValueError):
        model(X)
    with pytest.raises(ValueError):
        model.predict(X)


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=2)
def test_equine_protonet_save_load(random_dataset) -> None:
    dataset, num_classes, _ = random_dataset
    X, Y = dataset.tensors
    embedding_model = BasicEmbeddingModel(X.shape[1], num_classes)

    model = eq.EquineProtonet(embedding_model, num_classes)
    model.train_model(torch.utils.data.TensorDataset(X, Y), num_episodes=10)

    old_output = model.predict(X[1:10])
    tmp_filename = "tmp_eq_proto.pt"
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)
    model.save(tmp_filename)
    new_model = eq.EquineProtonet.load(tmp_filename)
    new_output = new_model.predict(X[1:10])
    assert (
        torch.nn.functional.mse_loss(old_output.classes, new_output.classes) <= 1e-7
    ), "Predictions changed on reload"
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)  # Cleanup


@given(random_dataset=random_dataset())
@settings(deadline=None, max_examples=2)
def test_equine_protonet_save_load_with_temperature(random_dataset) -> None:
    dataset, num_classes = random_dataset
    X, Y = dataset.tensors
    embedding_model = BasicEmbeddingModel(X.shape[1], num_classes)

    model = eq.EquineProtonet(embedding_model, num_classes, use_temperature=True)
    model.train_model(torch.utils.data.TensorDataset(X, Y), num_episodes=10)

    old_output = model.predict(X[1:10])
    tmp_filename = "tmp_eq_proto.pt"
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)
    model.save(tmp_filename)
    new_model = eq.EquineProtonet.load(tmp_filename)
    new_output = new_model.predict(X[1:10])
    assert (
        torch.nn.functional.mse_loss(old_output.classes, new_output.classes) <= 1e-7
    ), "Predictions changed on reload"
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)  # Cleanup
