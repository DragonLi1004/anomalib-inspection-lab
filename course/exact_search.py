# Copyright (C) 2026 Course contributors
# SPDX-License-Identifier: Apache-2.0
"""Memory-efficient execution of upstream PatchCore, without feature compression.

Forward follows anomalib lib/v2.5.1 PatchcoreModel.forward. Intermediate feature
references are released as soon as their last consumer finishes. Optional query
chunking retains the upstream distance arithmetic and all memory-bank entries.
"""
from types import MethodType
import torch
from anomalib.data import InferenceBatch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def memory_forward(self, input_tensor):
    if self.training:
        return PatchcoreModel.forward(self, input_tensor)
    input_tensor = input_tensor.type(self.memory_bank.dtype)
    output_size = input_tensor.shape[-2:]
    if self.tiler:
        input_tensor = self.tiler.tile(input_tensor)
    with torch.no_grad():
        features = self.feature_extractor(input_tensor)
    features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
    embedding = self.generate_embedding(features)
    del features
    if self.tiler:
        embedding = self.tiler.untile(embedding)
    batch_size, _, width, height = embedding.shape
    embedding = self.reshape_embedding(embedding)
    if self.memory_bank.size(0) == 0:
        raise ValueError('Memory bank is empty. Cannot provide anomaly scores')
    patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
    patch_scores = patch_scores.reshape((batch_size, -1))
    locations = locations.reshape((batch_size, -1))
    pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
    del embedding, locations
    patch_scores = patch_scores.reshape((batch_size, 1, width, height))
    anomaly_map = self.anomaly_map_generator(patch_scores, output_size)
    return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)


def chunked_neighbors(self, embedding, n_neighbors):
    scores, locations = [], []
    # Reuse bank norms within this call only: no persistent cache or extra weights.
    y_norm = self.memory_bank.pow(2).sum(dim=-1, keepdim=True).T
    for x in embedding.split(self.course_query_chunk):
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        distances = torch.matmul(x, self.memory_bank.T)
        distances.mul_(-2).add_(x_norm).add_(y_norm).clamp_min_(0).sqrt_()
        values, indices = distances.min(1) if n_neighbors == 1 else distances.topk(n_neighbors, largest=False, dim=1)
        scores.append(values)
        locations.append(indices)
        del distances
    return torch.cat(scores), torch.cat(locations)


def configure_exact(model, query_chunk=None, enabled=True):
    """Configure an upstream torch model after loading; enabled=False restores it.

    No checkpoint tensors or persistent buffers are added. Chunked GEMM can have
    device-dependent floating-point differences; validate on the target device.
    The reference-release-only variant (query_chunk=None) preserves operations.
    """
    if type(model) is not PatchcoreModel:
        raise TypeError('Exact mode requires an original PatchcoreModel, not PCA or another subclass')
    if query_chunk is not None and (type(query_chunk) is not int or query_chunk <= 0):
        raise ValueError('query_chunk must be a positive integer or None')
    model.forward = MethodType(memory_forward if enabled else PatchcoreModel.forward, model)
    model.course_query_chunk = query_chunk
    neighbor_fn = chunked_neighbors if enabled and query_chunk is not None else PatchcoreModel.nearest_neighbors
    model.nearest_neighbors = MethodType(neighbor_fn, model)
    return model
