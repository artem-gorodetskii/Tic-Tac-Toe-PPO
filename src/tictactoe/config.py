from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Config:
    r"""Configuration parameters for Tic-Tac-Toe training and inference."""

    # Common.
    seed: int = 42  # Random seed.

    # Environment.
    board_size: int = 3  # Board side size.
    board_center: int = 4  # Index of center cell.
    free_cell_val: int = 0  # Value representing an empty cell.
    opponent_player: int = 1  # Player ID for the opponent.
    agent_player: int = 2  # Player ID for the agent.
    history_size: int = 5  # Size of the move history.

    # Model.
    num_states: int = 3  # Number of possible cell states.
    input_dim: int = 9  # Total number of cells in the board.
    actor_hid_dim: int = 64  # Hidden dimension for actor network.
    actor_fwd_dim: int = 128  # Feedforward dimension in actor transformer.
    actor_nheads: int = 1  # Number of attention heads in actor.
    actor_nlayers: int = 2  # Number of transformer layers in actor.
    critic_hid_dim: int = 64  # Hidden dimension for critic network.
    critic_fwd_dim: int = 128  # Feedforward dimension in critic transformer.
    critic_nheads: int = 1  # Number of attention heads in critic.
    critic_nlayers: int = 2  # Number of transformer layers in critic.

    # PPO Agent.
    gamma: float = 0.98  # Discount factor.
    use_gae: bool = True  # Whether to use Generalized Advantage Estimation.
    normalize_rewards: bool = False  # Whether to normalize rewards.
    normalize_advantages: bool = False  # Whether to normalize advantages.
    centralize_advantages: bool = False  # Whether to subtract mean from advantages.
    lam: float = 0.95  # GAE lambda parameter.
    clip_eps: float = 0.2  # PPO clip epsilon.
    value_loss_coef: float = 1.0  # Coefficient for value loss term.
    old_policy_update_interval: int = 10  # Steps before syncing old and new policies.
    entropy_coef: float = 0.002  # Coefficient for entropy bonus.

    # Optimization.
    actor_lr: float = 1e-4  # Learning rate for actor.
    actor_weight_decay: float = 0.0  # Weight decay for actor.
    actor_opt_eps: float = 1e-8  # Optimizer epsilon for actor.
    actor_opt_betas: Tuple[float, float] = (0.9, 0.99)  # Betas for actor optimizer.
    actor_clip_grad_norm: Optional[float] = None  # Max norm for actor gradients.
    critic_lr: float = 1e-4  # Learning rate for critic.
    critic_weight_decay: float = 0.0  # Weight decay for critic.
    critic_opt_eps: float = 1e-8  # Optimizer epsilon for critic.
    critic_opt_betas: Tuple[float, float] = (0.9, 0.99)  # Betas for critic optimizer.
    critic_clip_grad_norm: Optional[float] = None  # Max norm for critic gradients.
    grad_accum_steps: int = 1  # Number of gradient accumulation steps.
    # Layer name markers to exclude from weight decay.
    weight_decay_blacklist: List[str] = field(
        default_factory=lambda: [
            "embedding",
            "attn",
            "bias",
            "norm",
            "actor_head",
            "critic_head",
        ]
    )

    # Reward.
    win: float = 3.0  # Reward for winning.
    lose: float = -1.0  # Penalty for losing.
    draw: float = 0.5  # Reward for draw.
    move_bonus: float = 0.02  # Bonus for any move.
    fast_win_bonus: float = 2.0  # Bonus for fast victory.
    fast_win_moves: int = 5  # Threshold for "fast" win in moves.
    center_bonus: float = 0.3  # Bonus for choosing the center cell.
    two_in_row_bonus: float = 0.8  # Bonus for forming a doublet.
    block_opponent_bonus: float = 0.8  # Bonus for blocking opponent doublet.
    block_miss_penalty: float = -1.0  # Penalty for not blocking a threat.
    miss_win_opportunity_penalty: float = -0.8  # Penalty for missing own win.
    miss_win_opportunity_decay: float = 0.8  # Decay factor for repeated misses.
    thoughtful_victory_bonus: float = 1.5  # Extra reward for strategic win.

    # Rule-based agent behavior.
    rb_agent_win_prob: float = 0.33  # Probability to select winning move if available.
    rb_agent_block_prob: float = 0.33  # Probability to block if threat exists.
    rb_agent_center_prob: float = 0.5  # Probability to pick center if free.
    rb_agent_coners_prob: float = 0.5  # Probability to prefer corners.
    rb_agent_sides_prob: float = 0.4  # Probability to prefer sides.

    # Trainer.
    num_episodes: int = 1000000  # Number of episodes to train.

    # API.
    # Path to save/load model checkpoints.
    checkpoint_path: Path = Path("checkpoints/best_agent.pt")
    # Path to the HTML page of the frontend application.
    frontend_index_path: Path = Path("frontend/index.html")
    # Name of the main frontend directory.
    frontend_dir_name: str = "frontend"
    # Name of the static frontend directory.
    static_dir_name: str = "static"

    def __post_init__(self) -> None:
        r"""Validates configuration parameters after dataclass initialization.

        This method ensures that key constraints on the configuration fields
        are satisfied to prevent misconfiguration and runtime errors.
        """
        if self.board_size >= self.board_center:
            raise ValueError("board_size must be less than board_center.")

        if (
            self.free_cell_val in [self.agent_player, self.opponent_player]
            or self.agent_player == self.opponent_player
        ):
            raise ValueError(
                "free_cell_val, agent_player and opponent_player must be distinct."
            )

        if not (0 < self.history_size < 6):
            raise ValueError("history_size must be between 1 and 5.")

        if self.num_states != 3:
            raise ValueError("num_states must be 3.")

        if self.input_dim != self.board_size**2:
            raise ValueError("input_dim must be board_size squared.")

        if self.actor_hid_dim % self.actor_nheads != 0:
            raise ValueError("actor_hid_dim must be divisible by actor_nheads.")

        if self.critic_hid_dim % self.critic_nheads != 0:
            raise ValueError("critic_hid_dim must be divisible by critic_nheads.")

        if self.old_policy_update_interval < self.grad_accum_steps:
            raise ValueError("old_policy_update_interval must be >= grad_accum_steps.")

        if self.centralize_advantages and self.normalize_advantages:
            raise ValueError("Only one of these parameters can be True.")


CFG = Config()
