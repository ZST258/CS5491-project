from __future__ import annotations

import numpy as np


def build_knn_edge_index(positions: np.ndarray, k: int) -> np.ndarray:
    num_nodes = positions.shape[0]
    if num_nodes <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    k_actual = min(k, num_nodes - 1)
    if k_actual <= 0:
        return np.zeros((2, 0), dtype=np.int64)

    # L1 distance matrix, fully vectorized.
    distances = np.abs(positions[:, None, :] - positions[None, :, :]).sum(axis=-1)
    np.fill_diagonal(distances, np.inf)

    # Select the k nearest neighbors per node without sorting the full row.
    knn_idx = np.argpartition(distances, kth=k_actual - 1, axis=1)[:, :k_actual]

    src = np.repeat(np.arange(num_nodes, dtype=np.int64), k_actual)
    dst = knn_idx.reshape(-1).astype(np.int64, copy=False)
    return np.stack([src, dst], axis=0)
