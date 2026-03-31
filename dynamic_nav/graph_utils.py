from __future__ import annotations

import numpy as np


def build_knn_edge_index(positions: np.ndarray, k: int) -> np.ndarray:
    """Return directed kNN edges for a small graph."""
    num_nodes = positions.shape[0]
    if num_nodes <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    edges: list[tuple[int, int]] = []
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    for src in range(num_nodes):
        order = np.argsort(distances[src])
        neighbors = [dst for dst in order if dst != src][: min(k, num_nodes - 1)]
        edges.extend((src, dst) for dst in neighbors)
    return np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
