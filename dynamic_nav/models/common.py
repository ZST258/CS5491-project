from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from dynamic_nav.config import GLOBAL_FEATURE_DIM, NUM_ACTIONS
from dynamic_nav.observation import flatten_observation


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor


def observation_to_tensors(observation: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "node_features": torch.as_tensor(observation["node_features"], dtype=torch.float32, device=device),
        "edge_index": torch.as_tensor(observation["edge_index"], dtype=torch.long, device=device),
        "global_features": torch.as_tensor(observation["global_features"], dtype=torch.float32, device=device),
        "action_mask": torch.as_tensor(observation["action_mask"], dtype=torch.float32, device=device),
        "node_count": torch.as_tensor(
            observation.get("node_count", [observation["node_features"].shape[0]]),
            dtype=torch.long,
            device=device,
        ),
    }


def batch_to_tensors(observations: list[dict[str, Any]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [observation_to_tensors(observation, device=device) for observation in observations]


def masked_categorical(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.distributions.Categorical:
    # AMP(fp16) 安全：避免 -1e9 在 half 下溢出
    if logits.dtype in (torch.float16, torch.bfloat16):
        mask_value = torch.finfo(logits.dtype).min  # fp16约 -65504
    else:
        mask_value = -1e9
    masked_logits = logits.masked_fill(action_mask <= 0, mask_value)
    return torch.distributions.Categorical(logits=masked_logits)


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class BasePolicy(nn.Module):
    model_name = "base"

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.to(self.device)

    def forward_batch(self, observations: list[dict[str, Any]]) -> PolicyOutput:
        raise NotImplementedError

    # 新增：子类可重写（PredictivePolicy 会重写）
    def forward_from_latent(self, latents: torch.Tensor) -> PolicyOutput:
        raise NotImplementedError("forward_from_latent is not implemented for this policy.")

    def auxiliary_loss(self, observations: list[dict[str, Any]], actions: np.ndarray, dones: np.ndarray) -> torch.Tensor:
        return torch.zeros((), device=self.device)

    # 新增：默认无 latent 版，子类可重写
    def auxiliary_loss_from_latent(self, latents: torch.Tensor, actions: np.ndarray, dones: np.ndarray) -> torch.Tensor:
        # 回退到 observation 版本（需要子类自己支持时再覆盖）
        return torch.zeros((), device=self.device)

    def act(self, observation: dict[str, Any]) -> tuple[int, float, float]:
        output = self.forward_batch([observation])
        action_mask = torch.as_tensor(observation["action_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = masked_categorical(output.logits, action_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(output.value.squeeze(0).item())

    def evaluate_actions(
        self,
        observations: list[dict[str, Any]],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward_batch(observations)
        action_masks = torch.stack(
            [torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=self.device) for obs in observations], dim=0
        )
        dist = masked_categorical(output.logits, action_masks)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, output.value.squeeze(-1)

    # 新增：用 latent 做 actor/critic 前向，避免重复 encoder
    def evaluate_actions_from_latent(
        self,
        latents: torch.Tensor,
        action_masks: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward_from_latent(latents)
        dist = masked_categorical(output.logits, action_masks)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, output.value.squeeze(-1)


def flatten_batch(observations: list[dict[str, Any]], max_nodes: int, device: torch.device) -> torch.Tensor:
    flat = [flatten_observation(observation, max_nodes=max_nodes) for observation in observations]
    return torch.as_tensor(np.asarray(flat), dtype=torch.float32, device=device)


def graph_inputs(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_count = int(np.asarray(observation.get("node_count", [observation["node_features"].shape[0]])).item())
    node_features = np.asarray(observation["node_features"], dtype=np.float32)[:node_count]
    edge_index = np.asarray(observation["edge_index"], dtype=np.int64)
    valid_edges = edge_index[:, np.any(edge_index != 0, axis=0)]
    if valid_edges.size == 0:
        valid_edges = np.zeros((2, 0), dtype=np.int64)
    global_features = np.asarray(observation["global_features"], dtype=np.float32)
    return node_features, valid_edges, global_features


class GraphAttentionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn_proj = nn.Linear(hidden_dim * 2, 1)
        self.output_proj = nn.Linear(hidden_dim + GLOBAL_FEATURE_DIM, output_dim)
        self.activation = nn.ELU()

    def forward_single(
        self, node_features: torch.Tensor, edge_index: torch.Tensor, global_features: torch.Tensor
    ) -> torch.Tensor:
        # hidden: [N, H]
        hidden = self.activation(self.input_proj(node_features))
        num_nodes = hidden.shape[0]
        device = hidden.device

        # 构建带自环边
        if edge_index.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]
            self_loop = torch.arange(num_nodes, device=device)
            src = torch.cat([src, self_loop], dim=0)
            dst = torch.cat([dst, self_loop], dim=0)
        else:
            src = torch.arange(num_nodes, device=device)
            dst = torch.arange(num_nodes, device=device)

        # 边级 attention（避免 N*N pairwise）
        edge_feat = torch.cat([hidden[src], hidden[dst]], dim=-1)  # [E, 2H]
        edge_score = self.attn_proj(edge_feat).squeeze(-1)  # [E]

        # 按 dst 做 softmax（简化实现：循环节点，通常 N 不大）
        attn = torch.zeros_like(edge_score)
        for node in range(num_nodes):
            mask = (dst == node)
            if mask.any():
                probs = torch.softmax(edge_score[mask].float(), dim=0).to(attn.dtype)
                attn[mask] = probs

        # 聚合
        messages = hidden[src] * attn.unsqueeze(-1)  # [E, H]
        aggregated = torch.zeros_like(hidden)  # [N, H]
        aggregated.index_add_(0, dst, messages)

        pooled = aggregated.mean(dim=0)  # [H]
        return self.output_proj(torch.cat([pooled, global_features], dim=0))

    def forward_batch(self, observations: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
        latents = []
        for observation in observations:
            node_features, edge_index, global_features = graph_inputs(observation)
            node_tensor = torch.as_tensor(node_features, dtype=torch.float32, device=device)
            edge_tensor = torch.as_tensor(edge_index, dtype=torch.long, device=device)
            global_tensor = torch.as_tensor(global_features, dtype=torch.float32, device=device)
            latents.append(self.forward_single(node_tensor, edge_tensor, global_tensor))
        return torch.stack(latents, dim=0)