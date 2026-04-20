from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dynamic_nav.config import GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM, NUM_ACTIONS
from .common import BasePolicy, PolicyOutput

# Keep these exports for predictive.py compatibility.
EDGE_FEATURE_DIM = 5
EDGE_AGENT_GOAL = 0
EDGE_AGENT_OBS = 1
EDGE_OBS_OBS = 2
EDGE_SELF = 3
NUM_EDGE_TYPES = 4


def build_graph_from_obs(
    node_tensors: torch.Tensor,
    node_counts: torch.Tensor,
    edge_index_raw: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build edge-aware graph with self-loops and edge features.

    Returns:
        ei_full    : [B, 2, E_total]          knn edges + self-loops
        edge_feat  : [B, E_total, EDGE_FEATURE_DIM]
        node_mask  : [B, N]                   True = valid node
        edge_valid : [B, E_total]             True = valid edge
        src_c      : [B, E_total]             clamped source indices
        dst_c      : [B, E_total]             clamped destination indices
    """
    B, N, _ = node_tensors.shape

    node_mask = torch.arange(N, device=device).unsqueeze(0) < node_counts.unsqueeze(1)

    src_raw = edge_index_raw[:, 0, :]
    dst_raw = edge_index_raw[:, 1, :]
    nc_exp = node_counts.unsqueeze(1)
    valid_knn = (src_raw >= 0) & (dst_raw >= 0) & (src_raw < nc_exp) & (dst_raw < nc_exp)

    # append self-loops
    loop = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    loop_ei = torch.stack([loop, loop], dim=1)                # [B, 2, N]
    ei_full = torch.cat([edge_index_raw, loop_ei], dim=2)  # [B, 2, E_total]
    valid_full = torch.cat([valid_knn, node_mask],    dim=1)  # [B, E_total]

    E_total = ei_full.shape[2]
    src = ei_full[:, 0, :]   # [B, E_total]
    dst = ei_full[:, 1, :]

    # edge type classification
    is_self = src == dst
    is_ag = ((src == 0) & (dst == 1)) | ((src == 1) & (dst == 0))
    is_ao = ((src == 0) | (dst == 0) | (src == 1) | (dst == 1)) & ~is_ag & ~is_self
    etypes = torch.where(
        is_self, torch.tensor(EDGE_SELF,       device=device),
        torch.where(
            is_ag,  torch.tensor(EDGE_AGENT_GOAL, device=device),
            torch.where(
                is_ao, torch.tensor(EDGE_AGENT_OBS,  device=device),
                torch.tensor(EDGE_OBS_OBS,    device=device),
            ),
        ),
    )

    type_onehot = torch.zeros(B, E_total, NUM_EDGE_TYPES, dtype=torch.float32, device=device)
    type_onehot.scatter_(2, etypes.unsqueeze(2), 1.0)

    # spatial distance feature — clamped indices only used for gathering,
    # invalid edges are zeroed out via edge_valid below.
    src_c = src.clamp(0, N - 1)
    dst_c = dst.clamp(0, N - 1)
    positions = node_tensors[:, :, :2]
    src_pos = torch.gather(positions, 1, src_c.unsqueeze(2).expand(-1, -1, 2))
    dst_pos = torch.gather(positions, 1, dst_c.unsqueeze(2).expand(-1, -1, 2))
    space_dist = (src_pos - dst_pos).abs().sum(-1, keepdim=True) / 2.0  # [B, E, 1]

    edge_feat = torch.cat([type_onehot, space_dist], dim=2)  # [B, E, 5]

    src_node_valid = node_mask.gather(1, src_c)
    dst_node_valid = node_mask.gather(1, dst_c)
    edge_valid = valid_full & src_node_valid & dst_node_valid  # [B, E]

    edge_feat = edge_feat * edge_valid.unsqueeze(2).float()

    return ei_full, edge_feat, node_mask, edge_valid, src_c, dst_c


class EdgeAwareGNN(nn.Module):
    """
    Edge-aware Graph Transformer block.

    Edge bias is written into the dense [B, nh, N, N] attention matrix via
    vectorised index assignment (no Python head loop, no scatter_add).
    Padding nodes are cleanly handled by masking both key and query dimensions
    before softmax, then replacing any resulting NaNs with 0.
    """

    def __init__(self, dim: int, edge_dim: int = EDGE_FEATURE_DIM, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        self.edge_bias = nn.Linear(edge_dim, num_heads, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # [B, N, dim]
        edge_feat: torch.Tensor,   # [B, E, edge_dim]
        node_mask: torch.Tensor,   # [B, N]
        edge_valid: torch.Tensor,  # [B, E]
        src_c: torch.Tensor,       # [B, E]  clamped source indices
        dst_c: torch.Tensor,       # [B, E]  clamped destination indices
    ) -> torch.Tensor:
        B, N, _ = x.shape
        nh, hd = self.num_heads, self.head_dim

        q = self.q(x).view(B, N, nh, hd).transpose(1, 2)  # [B, nh, N, hd]
        k = self.k(x).view(B, N, nh, hd).transpose(1, 2)
        v = self.v(x).view(B, N, nh, hd).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, nh, N, N]

        edge_bias_raw = self.edge_bias(edge_feat).permute(0, 2, 1)  # [B, nh, E]
        edge_bias_raw = edge_bias_raw * edge_valid.unsqueeze(1).float()

        flat_idx = (dst_c * N + src_c).unsqueeze(1).expand(-1, nh, -1)  # [B, nh, E]
        bias_flat = torch.zeros(B, nh, N * N, dtype=x.dtype, device=x.device)
        bias_flat.scatter_add_(2, flat_idx, edge_bias_raw)
        attn = attn + bias_flat.reshape(B, nh, N, N)

        fill_value = torch.finfo(attn.dtype).min
        attn = attn.masked_fill(~node_mask.unsqueeze(1).unsqueeze(2), fill_value)
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, self.dim)
        x = self.norm1(x + self.out(out))
        x = self.norm2(x + self.ffn(x))
        # Padding nodes are cleared here; intermediate computations above do
        # not affect correctness.
        return x * node_mask.unsqueeze(-1).float()


class NodeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.agent_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.goal_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.obs_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, node_tensors: torch.Tensor, node_counts: torch.Tensor) -> torch.Tensor:
        agent_h = self.agent_proj(node_tensors[:, 0, :]).unsqueeze(1)
        goal_h = self.goal_proj(node_tensors[:, 1, :]).unsqueeze(1)
        obs_h = self.obs_proj(node_tensors[:, 2:, :])

        max_obs = obs_h.shape[1]
        device = node_tensors.device
        obs_counts = (node_counts - 2).clamp(min=0)
        valid_mask = torch.arange(max_obs, device=device).unsqueeze(0) < obs_counts.unsqueeze(1)
        obs_h = obs_h * valid_mask.unsqueeze(-1).to(obs_h.dtype)

        return torch.cat([agent_h, goal_h, obs_h], dim=1)


class GNNPolicy(BasePolicy):
    model_name = "gnn"

    def __init__(
        self,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout:    float = 0.0,
        device:     str = "cpu",
    ):
        super().__init__(device=device)
        self.latent_dim = latent_dim

        self.node_encoder = NodeEncoder(NODE_FEATURE_DIM, latent_dim)
        self.gnn_layers = nn.ModuleList([
            EdgeAwareGNN(latent_dim, edge_dim=EDGE_FEATURE_DIM, num_heads=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.global_proj = nn.Linear(GLOBAL_FEATURE_DIM, latent_dim)
        # agent token + goal token + projected global features
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(latent_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(latent_dim, 1)
        self.to(self.device)

    def encode(self, observations: list[dict]) -> torch.Tensor:
        device = self.device

        node_tensors = torch.stack([
            torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
            for obs in observations
        ], dim=0)

        node_counts = torch.as_tensor([
            int(np.asarray(obs.get("node_count", [obs["node_features"].shape[0]])).item())
            for obs in observations
        ], dtype=torch.long, device=device)

        edge_index_raw = torch.stack([
            torch.as_tensor(obs["edge_index"], dtype=torch.long, device=device)
            for obs in observations
        ], dim=0)

        _, edge_feat, node_mask, edge_valid, src_c, dst_c = build_graph_from_obs(
            node_tensors, node_counts, edge_index_raw, device
        )

        node_h = self.node_encoder(node_tensors, node_counts)
        for gnn_layer in self.gnn_layers:
            node_h = gnn_layer(node_h, edge_feat, node_mask, edge_valid, src_c, dst_c)

        agent_latent = node_h[:, 0, :]   # [B, H]
        goal_latent = node_h[:, 1, :]   # [B, H]

        global_h = torch.stack([
            torch.as_tensor(obs["global_features"], dtype=torch.float32, device=device)
            for obs in observations
        ], dim=0)

        return self.output_proj(
            torch.cat([agent_latent, goal_latent, self.global_proj(global_h)], dim=-1)
        )

    def forward_batch(self, observations: list[dict]) -> PolicyOutput:
        latent = self.encode(observations)
        return PolicyOutput(logits=self.policy_head(latent), value=self.value_head(latent))

    def forward_from_latent(self, latents: torch.Tensor) -> PolicyOutput:
        return PolicyOutput(logits=self.policy_head(latents), value=self.value_head(latents))

    def get_value_last_layer(self) -> nn.Linear | None:
        return self.value_head
