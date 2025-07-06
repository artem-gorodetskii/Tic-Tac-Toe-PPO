from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.config import Config
from tictactoe.model import ActorCriticModel


def _normalize(tensor: Tensor, eps: float = 1e-8) -> Tensor:
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)


def _centralize(tensor: Tensor) -> Tensor:
    return tensor - tensor.mean()


class PPOAgent(BaseAgent):
    r"""Proximal Policy Optimization (PPO) agent for Tic-Tac-Toe."""

    def __init__(
        self,
        params: Config,
        device: torch.device,
    ) -> None:
        super().__init__(params)
        self.device = device

        self.model = ActorCriticModel(self.params).to(self.device)
        actor_params, critic_params = self.model.get_weight_decay_groups()
        self.actor_optimizer = torch.optim.AdamW(
            actor_params,
            lr=self.params.actor_lr,
            eps=self.params.actor_opt_eps,
            betas=self.params.actor_opt_betas,
        )
        self.critic_optimizer = torch.optim.AdamW(
            critic_params,
            lr=self.params.critic_lr,
            eps=self.params.critic_opt_eps,
            betas=self.params.critic_opt_betas,
        )
        self.reset_memory()
        self.old_model = deepcopy(self.model)

    def reset_memory(self) -> None:
        r"""Clear the agent's stored trajectories and rewards."""
        self.log_probs: List[Tensor] = []
        self.states: List[Tensor] = []
        self.actions: List[Tensor] = []
        self.values: List[Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def set_eval_mode(self) -> None:
        r"""Set the agent's model to evaluation mode."""
        self.model.eval()

    @torch.no_grad()
    def act(self, board: List[int]) -> int:
        r"""Select an action given the current board state.

        Args:
            board: current game board.

        Returns:
            The action index chosen by the agent.
        """
        state = torch.LongTensor(board).to(self.device)
        logits, value = self.model(state.unsqueeze(0))

        mask = torch.tensor(
            [0 if c == self.params.free_cell_val else 1 for c in board],
            dtype=torch.bool,
            device=self.device,
        )
        logits = logits.masked_fill(mask, -float("inf"))

        dist = Categorical(logits=logits)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value.squeeze(0))

        return action.item()

    def give_reward(
        self, reward: float, done: bool, opponent_move: bool = False
    ) -> None:
        r"""Store the reward for the last action taken by the agent.

        Args:
            reward: reward received after taking an action.
            done: whether the episode has ended.
            opponent_move: whether the reward was received after the opponent's move.
        """
        if opponent_move:
            # If an agent fails after the opponent's turn,
            # add the reward to his last turn.
            self.rewards[-1] += reward
        else:
            self.rewards.append(reward)

        self.dones.append(done)

    def update_policy(self) -> None:
        r"""Update the policy and value networks using stored trajectories."""
        if not self.rewards or not self.states:
            return

        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        values = torch.stack(self.values)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)

        if self.params.normalize_rewards:
            rewards = _normalize(rewards)

        if self.params.use_gae:
            returns = self._compute_gae(rewards, values, dones).detach()
        else:
            returns = self._compute_returns(rewards, dones).detach()

        advantages = (returns - values).detach()

        if self.params.normalize_advantages:
            advantages = _normalize(advantages)

        if self.params.centralize_advantages:
            advantages = _centralize(advantages)

        logits, value_preds = self.model(states)
        dists = Categorical(logits=logits)
        log_probs = dists.log_prob(actions)
        entropy = dists.entropy().mean()

        with torch.no_grad():
            old_logits = self.old_model.actor_forward(states)
            old_dists = Categorical(logits=old_logits)
            old_log_probs = old_dists.log_prob(actions)

        # Finding the ratio (pi_theta / pi_theta__old).
        ratios = torch.exp(log_probs - old_log_probs)

        # Finding Surrogate Loss.
        surr_loss_1 = ratios * advantages
        surr_loss_2 = (
            torch.clamp(ratios, 1 - self.params.clip_eps, 1 + self.params.clip_eps)
            * advantages
        )
        entropy_bonus = -self.params.entropy_coef * entropy
        policy_loss = -torch.mean(torch.min(surr_loss_1, surr_loss_2)) + entropy_bonus

        value_loss = self.params.value_loss_coef * F.mse_loss(value_preds, returns)

        step_now = self.model.get_step() + 1
        accum = self.params.grad_accum_steps

        if step_now % accum == 0:
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        if step_now % accum == 0:
            if self.params.actor_clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.actor_parameters(),
                    self.params.actor_clip_grad_norm,
                    norm_type=2.0,
                )
            if self.params.critic_clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.critic_parameters(),
                    self.params.critic_clip_grad_norm,
                    norm_type=2.0,
                )
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.model.increment_step()
        self._update_old_policy()
        self.reset_memory()

    def _compute_returns(
        self,
        rewards: Tensor,
        dones: Tensor,
    ) -> Tensor:
        # Compute n-step returns.
        T = len(rewards)
        returns = torch.zeros(T, dtype=torch.float32, device=self.device)
        # Discounted reward.
        disc_reward = 0.0

        for t in reversed(range(T)):
            not_done = 1.0 - float(dones[t])
            disc_reward = rewards[t] + self.params.gamma * disc_reward * not_done
            returns[t] = disc_reward

        return returns.unsqueeze(-1)

    def _compute_gae(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
    ) -> Tensor:
        # Compute Generalized Advantage Estimation.
        if not len(rewards) == len(values):
            raise RuntimeError("Mismatch between rewards and values.")

        T = len(rewards)

        # Add a fake value.
        fake_value = torch.tensor([0.0], device=self.device).unsqueeze(-1)
        values = torch.cat([values, fake_value])
        gaes = torch.zeros(T, dtype=torch.float32, device=self.device)
        future_gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            not_done = 1.0 - float(dones[t])
            delta = (
                rewards[t] + self.params.gamma * values[t + 1] * not_done - values[t]
            )
            future_gae = (
                delta + self.params.gamma * self.params.lam * not_done * future_gae
            )
            gaes[t] = future_gae + values[t]

        return gaes.unsqueeze(-1)

    def _update_old_policy(self) -> None:
        if (
            self.model.get_step() - self.old_model.get_step()
            >= self.params.old_policy_update_interval
        ):
            self.old_model.load_state_dict(self.model.state_dict())
            self.old_model.set_step(self.model.get_step())
