from __future__ import annotations

from typing import Any

import numpy as np


def pad_node_features(node_features: np.ndarray, max_nodes: int) -> np.ndarray:
    padded = np.zeros((max_nodes, node_features.shape[1]), dtype=np.float32)
    length = min(max_nodes, node_features.shape[0])
    padded[:length] = node_features[:length]
    return padded


def flatten_observation(observation: dict[str, Any], max_nodes: int) -> np.ndarray:
    node_features = pad_node_features(np.asarray(observation["node_features"], dtype=np.float32), max_nodes)
    global_features = np.asarray(observation["global_features"], dtype=np.float32)
    action_mask = np.asarray(observation["action_mask"], dtype=np.float32)
    return np.concatenate([node_features.reshape(-1), global_features, action_mask], axis=0).astype(np.float32)
