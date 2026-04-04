from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class TierConfig:
    tier1_enabled: bool = False
    tier2_enabled: bool = False
    tier1_strength: float = 0.15
    tier2_cost_penalty: float = 0.01


class TieredQuantumAdapter(nn.Module):
    """
    Tier 1 + Tier 2 MVP adapter.

    Tier 1: lightweight post-quantum denoiser/calibrator (no extra quantum calls)
    Tier 2: small policy over gate-scale actions with policy-gradient style update
    """

    def __init__(self, n_qubits: int, cfg: TierConfig):
        super().__init__()
        self.cfg = cfg

        self.tier1_calibrator = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Tanh(),
            nn.Linear(n_qubits, n_qubits),
        )

        # Actions: 0=identity, 1=attenuate, 2=amplify
        self.tier2_policy = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

        self.register_buffer("reward_baseline", torch.tensor(0.0))
        self._last_log_probs: torch.Tensor | None = None
        self._last_action_cost: torch.Tensor | None = None
        self._last_policy_active = False

    def adapt_input(self, q_input: torch.Tensor, training: bool) -> torch.Tensor:
        if not (self.cfg.tier2_enabled and training):
            self._last_policy_active = False
            self._last_log_probs = None
            self._last_action_cost = None
            return q_input

        logits = self.tier2_policy(q_input.detach())
        dist = Categorical(logits=logits)
        actions = dist.sample()  # (B,)
        log_probs = dist.log_prob(actions)

        # Per-sample scaling factors to emulate policy-selected gate intensity.
        scales = torch.tensor([1.0, 0.9, 1.1], device=q_input.device, dtype=q_input.dtype)
        factors = scales[actions].unsqueeze(1)

        costs = torch.tensor([0.0, 0.005, 0.015], device=q_input.device, dtype=q_input.dtype)
        action_cost = costs[actions]

        self._last_policy_active = True
        self._last_log_probs = log_probs
        self._last_action_cost = action_cost
        return q_input * factors

    def adapt_output(self, q_out: torch.Tensor) -> torch.Tensor:
        if not self.cfg.tier1_enabled:
            return q_out

        correction = self.tier1_calibrator(q_out)
        return q_out + self.cfg.tier1_strength * correction

    def policy_loss(self, batch_reward: torch.Tensor) -> torch.Tensor:
        if not (self.cfg.tier2_enabled and self._last_policy_active):
            return torch.tensor(0.0, device=batch_reward.device)

        assert self._last_log_probs is not None
        assert self._last_action_cost is not None

        reward = batch_reward.detach().float().mean()
        self.reward_baseline.mul_(0.95).add_(0.05 * reward)
        advantage = (reward - self.reward_baseline).detach()

        pg = -(self._last_log_probs * advantage).mean()
        cost_penalty = self.cfg.tier2_cost_penalty * self._last_action_cost.mean()
        return pg + cost_penalty
