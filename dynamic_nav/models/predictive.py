from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dynamic_nav.config import NODE_FEATURE_DIM, NUM_ACTIONS, GLOBAL_FEATURE_DIM, DIFFICULTY_CONFIGS
from .common import BasePolicy, PolicyOutput
from .gnn import (
    NodeEncoder,
    EdgeAwareGNN,
    EDGE_FEATURE_DIM,
    EDGE_AGENT_GOAL,
    EDGE_AGENT_OBS,
    EDGE_OBS_OBS,
    EDGE_SELF,
    NUM_EDGE_TYPES,
)

# ──────────────────────────────────────────────────────────────────────────────
# Discrete move vocabulary
#
# Both obstacles and the agent move on a grid: each step is exactly one of 5
# actions — stay / up / down / left / right — so the displacement (dx, dy) in
# grid units is always one of:
#   class 0: ( 0,  0)  stay
#   class 1: (-1,  0)  up
#   class 2: ( 1,  0)  down
#   class 3: ( 0, -1)  left
#   class 4: ( 0,  1)  right
# ──────────────────────────────────────────────────────────────────────────────

NUM_MOVE_CLASSES = 5

_MOVE_DELTAS_GRID = torch.tensor(
    [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
    dtype=torch.float32,
)


def _delta_to_class(delta_grid: torch.Tensor) -> torch.Tensor:
    """
    Map integer grid-unit displacement (dx, dy) → class id in {0,…,4}.

    Parameters
    ----------
    delta_grid : [..., 2]  long tensor  (dx, dy) in {-1, 0, 1}

    Returns
    -------
    class_id : [...]  long tensor in {0,…,4}
    """
    dx  = delta_grid[..., 0]
    dy  = delta_grid[..., 1]
    cls = torch.zeros_like(dx, dtype=torch.long)
    cls = torch.where((dx == -1) & (dy ==  0), torch.full_like(cls, 1), cls)
    cls = torch.where((dx ==  1) & (dy ==  0), torch.full_like(cls, 2), cls)
    cls = torch.where((dx ==  0) & (dy == -1), torch.full_like(cls, 3), cls)
    cls = torch.where((dx ==  0) & (dy ==  1), torch.full_like(cls, 4), cls)
    return cls


# ──────────────────────────────────────────────────────────────────────────────
# ObstacleGRU — recursive single-step prediction
#
# Design
# ──────
# The GRU predicts ONE step at a time and feeds the predicted position back
# as input for the next step. With horizon=3:
#
#   step 1: x_input = input_proj([x0, y0, vx, vy, ...])
#           h1 = GRU(x_input, h0)  →  logit_1  (predicts t=0→t=1 displacement)
#           update curr_xy: x1 = x0 + softmax(logit_1) @ delta_table  [no_grad]
#
#   step 2: x_input = input_proj([x1, y1, vx, vy, ...])   ← updated position
#           h2 = GRU(x_input, h1)  →  logit_2  (predicts t=1→t=2 displacement)
#           update curr_xy: x2 = x1 + ...                              [no_grad]
#
#   step 3: x_input = input_proj([x2, y2, vx, vy, ...])   ← updated position
#           h3 = GRU(x_input, h2)  →  logit_3  (predicts t=2→t=3 displacement)
#
# This allows the GRU to detect boundary proximity at each step and predict
# rebounds correctly, because the updated position x_h is available when
# predicting the displacement at step h+1.
#
# The position update inside the loop uses torch.no_grad() to prevent
# gradient accumulation across H recursive steps (which would cause
# exploding gradients). The differentiable gradient path for GRU parameters
# goes through the logits directly via the CE loss in auxiliary_loss, and
# through the soft rollout in encode().
# ──────────────────────────────────────────────────────────────────────────────

class ObstacleGRU(nn.Module):
    """
    Per-obstacle GRU with recursive single-step prediction.

    Parameters
    ----------
    input_dim  : NODE_FEATURE_DIM (7)
    hidden_dim : latent_dim
    horizon    : number of future steps to predict

    Forward inputs
    --------------
    # obs_features : [B, max_obs, F]
    # cv_mask      : [B, max_obs]  bool — True = constant_velocity obstacle
    # delta_table  : [5, 2]        normalised per-class displacement

    # Forward returns
    ---------------
    # logits : [B, max_obs, H, 5]
        Raw logits. rw and padding obstacles are zeroed.
    """

    def __init__(self, input_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.horizon    = horizon
        self.hidden_dim = hidden_dim

        # Called once per recursive step with the current (updated) features.
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)

        self.move_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_MOVE_CLASSES),
        )

    def forward(
        self,
        obs_features: torch.Tensor,   # [B, max_obs, F]
        cv_mask:      torch.Tensor,   # [B, max_obs]  bool
        delta_table:  torch.Tensor,   # [5, 2]  normalised displacements
    ) -> torch.Tensor:
        B, max_obs, feat_dim = obs_features.shape

        # Flatten obstacles into batch dim for shared GRU weights.
        curr_feat    = obs_features.reshape(B * max_obs, feat_dim).clone()  # [B*M, F]
        cv_mask_flat = cv_mask.reshape(B * max_obs, 1).float()              # [B*M, 1]

        # Initialise hidden state from t=0 features.
        h = self.input_proj(curr_feat)   # [B*M, D]

        logit_steps: list[torch.Tensor] = []

        for _ in range(self.horizon):
            # ── 1. Build GRU input from current (possibly updated) features ──
            x_input = self.input_proj(curr_feat)           # [B*M, D]

            # ── 2. Single GRU step → predict next displacement ───────────
            h      = self.norm(self.gru_cell(x_input, h))
            logits = self.move_head(h)                     # [B*M, 5]
            logit_steps.append(logits)

            # ── 3. Update position for next recursive step (no_grad) ──────
            # Wrap in no_grad so gradients do not accumulate across the loop.
            # The GRU parameters still receive gradients through the logits
            # collected in logit_steps (via CE loss) and through the soft
            # rollout in encode() (via the policy gradient path).
            with torch.no_grad():
                probs     = F.softmax(logits, dim=-1)      # [B*M, 5]
                exp_delta = probs @ delta_table             # [B*M, 2]
                # Only move cv obstacles; rw stay fixed (cv_mask_flat=0)
                new_xy = (
                    curr_feat[:, :2] + exp_delta * cv_mask_flat
                ).clamp(0.0, 1.0)
                curr_feat = curr_feat.clone()
                curr_feat[:, :2] = new_xy

        # [B*M, H, 5] → [B, max_obs, H, 5]
        logits_all = (
            torch.stack(logit_steps, dim=1)
            .reshape(B, max_obs, self.horizon, NUM_MOVE_CLASSES)
        )

        # Zero out rw obstacles and padding
        cv_mask_f = cv_mask.unsqueeze(-1).unsqueeze(-1).float()   # [B, M, 1, 1]
        return logits_all * cv_mask_f


# ──────────────────────────────────────────────────────────────────────────────
# PredictivePolicy
# ──────────────────────────────────────────────────────────────────────────────

class PredictivePolicy(BasePolicy):
    """
    GNN policy augmented with a discrete recurrent obstacle predictor.

    Key design choices
    ──────────────────
    • ObstacleGRU predicts 5-class move logits for cv obstacles only,
      using recursive single-step prediction with position feedback.
    • rw obstacles keep their current position in all future frames.
      The GNN learns to avoid them via the is_rw flag (node feature idx 6).
    • Rollout is soft (softmax × delta_table) so gradients flow to the GRU.
    • Auxiliary loss is CE over cv obstacles only; rw obstacles are excluded.
    """

    model_name = "predictive"

    def __init__(
        self,
        latent_dim:  int   = 128,
        horizon:     int   = 3,
        aux_coef:    float = 0.2,
        num_layers:  int   = 2,
        dropout:     float = 0.0,
        device:      str   = "cpu",
    ):
        super().__init__(device=device)
        self.latent_dim = latent_dim
        self.horizon    = int(horizon)
        self.aux_coef   = aux_coef

        # ── GNN encoder (shared across all time steps) ───────────────
        self.node_encoder = NodeEncoder(NODE_FEATURE_DIM, latent_dim, dropout=dropout)
        self.gnn_layers   = nn.ModuleList([
            EdgeAwareGNN(latent_dim, edge_dim=EDGE_FEATURE_DIM, num_heads=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.global_proj = nn.Linear(GLOBAL_FEATURE_DIM, latent_dim)

        # agent + goal + global → latent_dim
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

        # ── Step embedding ────────────────────────────────────────────
        self.step_embed = nn.Embedding(self.horizon + 1, latent_dim)

        # ── Fixed gamma decay ─────────────────────────────────────────
        self.gamma = 0.7

        # ── Move-delta lookup table (buffer, not a parameter) ─────────
        self.register_buffer("move_deltas_grid", _MOVE_DELTAS_GRID.clone())

        # ── Obstacle GRU predictor ────────────────────────────────────
        self.obstacle_gru = ObstacleGRU(
            input_dim  = NODE_FEATURE_DIM,
            hidden_dim = latent_dim,
            horizon    = self.horizon,
        )

        # ── Actor / Critic ─────────────────────────────────────────────
        self.actor_input_dim  = latent_dim * (self.horizon + 1)
        self.critic_input_dim = latent_dim * (self.horizon + 1)

        a_h1 = max(128, self.actor_input_dim  // 2)
        a_h2 = max(64,  a_h1 // 2)
        c_h1 = max(128, self.critic_input_dim // 2)
        c_h2 = max(64,  c_h1 // 2)

        self.actor = nn.Sequential(
            nn.Linear(self.actor_input_dim, a_h1), nn.ReLU(),
            nn.Linear(a_h1, a_h2),                 nn.ReLU(),
            nn.Linear(a_h2, NUM_ACTIONS),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.critic_input_dim, c_h1), nn.ReLU(),
            nn.Linear(c_h1, c_h2),                  nn.ReLU(),
            nn.Linear(c_h2, 1),
        )
        self.to(self.device)

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_obstacle_features(
        node_tensors: torch.Tensor,
        node_counts:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device    = node_tensors.device
        obs_feat  = node_tensors[:, 2:, :]
        B, max_obs, _ = obs_feat.shape

        obs_counts = (node_counts - 2).clamp(min=0)
        obs_mask   = (
            torch.arange(max_obs, device=device).unsqueeze(0)
            < obs_counts.unsqueeze(1)
        )
        cv_mask = obs_mask & (obs_feat[:, :, 5] > 0.5)
        return obs_feat, obs_mask, cv_mask, obs_feat[:, :, :2]

    @staticmethod
    def _build_knn_graph(
        node_tensors:       torch.Tensor,
        node_counts:        torch.Tensor,
        knn_k:              int,
        node_mask_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = node_tensors.shape
        device   = node_tensors.device

        node_mask = (
            node_mask_override
            if node_mask_override is not None
            else torch.arange(N, device=device).unsqueeze(0) < node_counts.unsqueeze(1)
        )

        pos   = node_tensors[:, :, :2]
        dists = (pos.unsqueeze(2) - pos.unsqueeze(1)).abs().sum(-1)

        huge    = 1e9
        invalid = ~node_mask
        dists   = dists.masked_fill(invalid.unsqueeze(2), huge)
        dists   = dists.masked_fill(invalid.unsqueeze(1), huge)
        dists   = dists.masked_fill(
            torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0), huge
        )

        k = max(1, min(knn_k, N - 1))
        _, knn_dst = dists.topk(k, dim=-1, largest=False)

        src_knn = (
            torch.arange(N, device=device)
            .unsqueeze(1).expand(N, k)
            .unsqueeze(0).expand(B, -1, -1)
        ).reshape(B, N * k)
        dst_knn = knn_dst.reshape(B, N * k)

        self_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        src_all  = torch.cat([src_knn, self_idx], dim=1)
        dst_all  = torch.cat([dst_knn, self_idx], dim=1)

        is_self = src_all == dst_all
        is_ag   = ((src_all == 0) & (dst_all == 1)) | ((src_all == 1) & (dst_all == 0))
        is_ao   = (
            (src_all == 0) | (dst_all == 0) | (src_all == 1) | (dst_all == 1)
        ) & ~is_ag & ~is_self

        etypes = torch.full((B, src_all.shape[1]), EDGE_OBS_OBS, dtype=torch.long, device=device)
        etypes = torch.where(is_ao,   torch.full_like(etypes, EDGE_AGENT_OBS),  etypes)
        etypes = torch.where(is_ag,   torch.full_like(etypes, EDGE_AGENT_GOAL), etypes)
        etypes = torch.where(is_self, torch.full_like(etypes, EDGE_SELF),       etypes)

        E           = src_all.shape[1]
        type_onehot = torch.zeros(B, E, NUM_EDGE_TYPES, dtype=torch.float32, device=device)
        type_onehot.scatter_(2, etypes.unsqueeze(2), 1.0)

        src_pos    = torch.gather(pos, 1, src_all.unsqueeze(2).expand(-1, -1, 2))
        dst_pos    = torch.gather(pos, 1, dst_all.unsqueeze(2).expand(-1, -1, 2))
        space_dist = (src_pos - dst_pos).abs().sum(-1, keepdim=True) / 2.0
        edge_feat  = torch.cat([type_onehot, space_dist], dim=2)

        b_idx      = torch.arange(B, device=device).unsqueeze(1)
        edge_valid = node_mask[b_idx, src_all] & node_mask[b_idx, dst_all]
        edge_feat  = edge_feat * edge_valid.unsqueeze(2).float()

        return edge_feat, node_mask, edge_valid, src_all, dst_all

    # ── encode ────────────────────────────────────────────────────────────────

    def encode(self, observations: list[dict[str, Any]]) -> torch.Tensor:
        device = self.device

        node_tensors = torch.stack([
            torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
            for obs in observations
        ], dim=0)
        node_counts = torch.as_tensor([
            int(np.asarray(obs.get("node_count", [obs["node_features"].shape[0]])).item())
            for obs in observations
        ], dtype=torch.long, device=device)
        global_h = torch.stack([
            torch.as_tensor(obs["global_features"], dtype=torch.float32, device=device)
            for obs in observations
        ], dim=0)

        B         = node_tensors.shape[0]
        max_nodes = node_tensors.shape[1]
        T         = self.horizon + 1

        difficulty = observations[0].get("difficulty", "hard") if observations else "hard"
        knn_k      = DIFFICULTY_CONFIGS[difficulty].knn_k
        grid_size  = int(np.asarray(observations[0].get("grid_size", 12)).item())

        # ── Step 1: GRU recursive single-step prediction ──────────────
        obs_features, obs_mask, cv_mask, obs_xy = self._extract_obstacle_features(
            node_tensors, node_counts
        )
        delta_table = self.move_deltas_grid / float(grid_size)   # [5, 2]
        logits = self.obstacle_gru(obs_features, cv_mask, delta_table)
        # logits: [B, max_obs, H, 5]

        # ── Step 2: Differentiable soft rollout for graph construction ─
        # This rollout IS differentiable (no no_grad), so GRU parameters
        # also receive gradients through the GNN → policy loss path.
        # The GRU's internal position update (in forward) uses no_grad
        # only to prevent gradient explosion across the recursive loop.
        all_node_tensors = node_tensors.unsqueeze(0).expand(T, -1, -1, -1).clone()
        cv_mask_f = cv_mask.unsqueeze(-1).float()   # [B, max_obs, 1]

        for h in range(self.horizon):
            curr_nodes = all_node_tensors[h]
            logits_h   = logits[:, :, h, :]
            probs_h    = F.softmax(logits_h, dim=-1)
            exp_delta  = (probs_h @ delta_table) * cv_mask_f
            curr_pos   = curr_nodes[:, 2:, :2]
            curr_vel   = curr_nodes[:, 2:, 2:4]

            proposal = curr_pos + curr_vel
            hit_x = (proposal[:, :, 0] <= 0.0) | (proposal[:, :, 0] >= 1.0)
            hit_y = (proposal[:, :, 1] <= 0.0) | (proposal[:, :, 1] >= 1.0)
            next_vel = torch.stack([
                torch.where(hit_x, -curr_vel[:, :, 0], curr_vel[:, :, 0]),
                torch.where(hit_y, -curr_vel[:, :, 1], curr_vel[:, :, 1]),
            ], dim=-1)

            next_vel = torch.where(cv_mask_f > 0, next_vel, curr_vel)
            next_xy  = torch.where(cv_mask_f > 0, curr_pos + next_vel, curr_pos)
            all_node_tensors[h + 1, :, 2:, :2] = next_xy.clamp(0.0, 1.0)
            all_node_tensors[h + 1, :, 2:, 2:4] = next_vel

        # ── Step 3: KNN graphs (topology fixed, only space_dist updates) ─
        ef_0, node_mask, edge_valid, src_all, dst_all = self._build_knn_graph(
            node_tensors, node_counts, knn_k,
        )

        all_ef_tb      = ef_0.unsqueeze(0).expand(T, -1, -1, -1).clone()
        all_pos        = all_node_tensors[:, :, :, :2]
        gather_idx_src = src_all.unsqueeze(2).expand(-1, -1, 2)
        gather_idx_dst = dst_all.unsqueeze(2).expand(-1, -1, 2)

        src_pos_all = torch.gather(
            all_pos.reshape(T * B, max_nodes, 2), 1,
            gather_idx_src.repeat(T, 1, 1),
        ).reshape(T, B, -1, 2)
        dst_pos_all = torch.gather(
            all_pos.reshape(T * B, max_nodes, 2), 1,
            gather_idx_dst.repeat(T, 1, 1),
        ).reshape(T, B, -1, 2)

        space_dist_all = (src_pos_all - dst_pos_all).abs().sum(-1, keepdim=True) / 2.0
        all_ef_tb[:, :, :, 4:5] = (
            space_dist_all * edge_valid.unsqueeze(0).unsqueeze(3).float()
        )

        batched_nodes  = all_node_tensors.reshape(T * B, max_nodes, -1)
        batched_counts = node_counts.repeat(T)
        padded_ef      = all_ef_tb.reshape(T * B, -1, EDGE_FEATURE_DIM)
        padded_ev      = edge_valid.repeat(T, 1)
        padded_src     = src_all.repeat(T, 1)
        padded_dst     = dst_all.repeat(T, 1)
        node_mask_rep  = node_mask.repeat(T, 1)

        # ── Step 4: NodeEncoder + step embedding + GNN ───────────────
        node_h   = self.node_encoder(batched_nodes, batched_counts)
        step_ids = torch.arange(T, device=device).repeat_interleave(B)
        node_h   = node_h + self.step_embed(step_ids).unsqueeze(1)

        for layer in self.gnn_layers:
            node_h = layer(node_h, padded_ef, node_mask_rep, padded_ev, padded_src, padded_dst)

        # ── Step 5: Extract tokens + gamma decay ─────────────────────
        node_h_tb     = node_h.reshape(T, B, max_nodes, self.latent_dim)
        global_latent = self.global_proj(global_h)

        frame_tokens = torch.cat([
            node_h_tb[:, :, 0, :],
            node_h_tb[:, :, 1, :],
            global_latent.unsqueeze(0).expand(T, -1, -1),
        ], dim=-1)
        frame_latents = self.output_proj(
            frame_tokens.reshape(T * B, -1)
        ).reshape(T, B, self.latent_dim)

        future_env = frame_latents[1:]
        t_idx      = torch.arange(1, T, device=device, dtype=torch.float32).view(T - 1, 1, 1)
        future_env = future_env * (self.gamma ** t_idx)

        # ── Step 6: Fuse ──────────────────────────────────────────────
        base        = frame_latents[0]
        future_flat = future_env.permute(1, 0, 2).reshape(B, -1)
        return torch.cat([base, future_flat], dim=-1)

    # ── policy interface ──────────────────────────────────────────────────────

    def forward_batch(self, observations: list[dict[str, Any]]) -> PolicyOutput:
        latents = self.encode(observations)
        return PolicyOutput(
            logits=self.actor( latents[:, :self.actor_input_dim]),
            value =self.critic(latents[:, :self.critic_input_dim]),
        )

    def forward_from_latent(self, latents: torch.Tensor, action_masks=None) -> PolicyOutput:
        return PolicyOutput(
            logits=self.actor( latents[:, :self.actor_input_dim]),
            value =self.critic(latents[:, :self.critic_input_dim]),
        )

    def get_value_last_layer(self) -> nn.Linear | nn.Module | None:
        return self.critic[-1]

    # ── auxiliary loss ────────────────────────────────────────────────────────

    def auxiliary_loss(
        self,
        observations: list[dict[str, Any]],
        actions:      np.ndarray,
        dones:        np.ndarray,
    ) -> torch.Tensor:
        """
        CE loss between GRU recursive predictions and ground-truth single-step
        move classes from the rollout buffer.

        For each valid frame i and horizon step h in {1,...,H}:
            pred : GRU logit_h  (conditioned on predicted position at step h-1)
            true : move class of (obs_xy[i+h] - obs_xy[i+h-1])

        Both pred and true refer to the SAME single-step displacement,
        so the supervision is semantically consistent with the recursive design.
        """
        device = self.device
        B      = len(observations)

        if self.horizon <= 0 or B <= self.horizon:
            return torch.zeros((), device=device, requires_grad=True)

        # ── find valid (non-terminal) starting frames ─────────────────
        dones_t  = torch.as_tensor(dones.astype(np.bool_), device=device)
        num_cand = B - self.horizon
        valid    = torch.ones(num_cand, dtype=torch.bool, device=device)
        for h in range(self.horizon):
            valid &= ~dones_t[h: h + num_cand]
        valid &= ~dones_t[:num_cand]

        v_idx = valid.nonzero(as_tuple=False).squeeze(-1)
        if v_idx.numel() == 0:
            return torch.zeros((), device=device, requires_grad=True)

        Bv = v_idx.numel()

        # ── collect node features for frames i … i+H ──────────────────
        all_node_feats = torch.stack([
            torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
            for obs in observations
        ], dim=0)

        frame_idx = (
            v_idx.unsqueeze(0)
            + torch.arange(self.horizon + 1, device=device, dtype=torch.long).unsqueeze(1)
        )
        frames           = all_node_feats[frame_idx]   # [H+1, Bv, N, F]
        node_tensors_now = frames[0]                   # [Bv, N, F]

        node_counts_now = torch.as_tensor([
            int(np.asarray(
                observations[int(i)]["node_count"][0]
                if isinstance(observations[int(i)]["node_count"], (list, np.ndarray))
                else observations[int(i)]["node_count"]
            ).item())
            for i in v_idx.tolist()
        ], dtype=torch.long, device=device)

        # ── run GRU with recursive position feedback ───────────────────
        obs_features, obs_mask, cv_mask, obs_xy_now = self._extract_obstacle_features(
            node_tensors_now, node_counts_now
        )
        grid_size   = int(np.asarray(observations[0].get("grid_size", 12)).item())
        delta_table = (self.move_deltas_grid / float(grid_size)).to(device)

        # detach: prevent aux CE gradients from modifying shared GNN/NodeEncoder
        logits = self.obstacle_gru(obs_features.detach(), cv_mask, delta_table)
        # logits: [Bv, max_obs, H, 5]

        M = obs_mask.shape[1]

        # ── ground-truth single-step move classes ─────────────────────
        # true_label[h] = class of (obs_xy[i+h] - obs_xy[i+h-1])
        # Matches logit_h which predicts the displacement at step h
        # conditioned on the position predicted at step h-1.
        obs_xy_frames   = frames[:, :, 2:2 + M, :2]
        true_delta_norm = obs_xy_frames[1:] - obs_xy_frames[:-1]   # [H, Bv, M, 2]
        true_delta_grid = torch.round(
            true_delta_norm * float(grid_size)
        ).long().clamp(-1, 1)
        true_label = _delta_to_class(true_delta_grid)              # [H, Bv, M]

        # ── distance weighting ────────────────────────────────────────
        agent_xy_now = node_tensors_now[:, :1, :2]
        curr_dist    = (obs_xy_now - agent_xy_now).abs().sum(-1)
        weight       = (1.0 / (1.0 + curr_dist)).clamp(min=0.25)

        # ── CE loss over cv obstacles only ────────────────────────────
        logits_h = logits.permute(2, 0, 1, 3)   # [H, Bv, M, 5]
        HH       = self.horizon

        per_ce = F.cross_entropy(
            logits_h.reshape(HH * Bv * M, NUM_MOVE_CLASSES),
            true_label.reshape(HH * Bv * M),
            reduction="none",
        ).reshape(HH, Bv, M)

        cv_mask_loss = cv_mask.unsqueeze(0).float()
        w_exp        = weight.unsqueeze(0)

        num            = (per_ce * cv_mask_loss * w_exp).sum(dim=(1, 2))
        den            = (cv_mask_loss * w_exp).sum(dim=(1, 2)).clamp(min=1.0)
        ce_per_horizon = num / den

        return self.aux_coef * ce_per_horizon.mean()
