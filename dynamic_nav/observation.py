from __future__ import annotations

from typing import Any

import numpy as np


def pad_node_features(node_features: np.ndarray, max_nodes: int) -> np.ndarray:
    padded = np.zeros((max_nodes, node_features.shape[1]), dtype=np.float32)
    length = min(max_nodes, node_features.shape[0])
    padded[:length] = node_features[:length]
    return padded


def flatten_observation(observation: dict[str, Any], max_nodes: int) -> np.ndarray:
    original_nodes = np.asarray(observation["node_features"], dtype=np.float32)
    node_count = int(original_nodes.shape[0])
    node_features = pad_node_features(original_nodes, max_nodes)
    # Mark padded rows with a sentinel type value (-1.0) so MLP can distinguish
    # real nodes from padding. The last column is the TYPE field per config.
    if node_count < max_nodes:
        node_features[node_count:, -1] = -1.0

    # Use the global_features as provided by the environment. The environment
    # is responsible for including any normalized node_count if desired.
    global_features = np.asarray(observation["global_features"], dtype=np.float32)
    action_mask = np.asarray(observation["action_mask"], dtype=np.float32)
    return np.concatenate([node_features.reshape(-1), global_features, action_mask], axis=0).astype(np.float32)
