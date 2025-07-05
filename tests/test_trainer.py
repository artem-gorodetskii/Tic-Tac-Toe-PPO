from typing import List
from unittest import TestCase

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents.ppo_agent import PPOAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.agents.rule_based_agent import RuleBasedAgent
from tictactoe.config import CFG
from tictactoe.environment import TicTacToeEnv
from tictactoe.trainer import Trainer
from tictactoe.utils import get_available_device, set_seed


class _MockAgent(BaseAgent):
    def act(self, board: List[int]) -> int:
        return next(i for i, x in enumerate(board) if x == CFG.free_cell_val)


class TestTrainer(TestCase):
    r"""Unit tests for the Trainer class."""

    @classmethod
    def setUp(cls) -> None:
        r"""Set up test environment for all test cases."""
        set_seed(CFG.seed)
        cls.device = get_available_device()
        cls.env = TicTacToeEnv(params=CFG)
        cls.agent = PPOAgent(params=CFG, device=cls.device)
        cls.num_episodes = 2

        cls.trainer = Trainer(
            params=CFG,
            env=cls.env,
            opponent=_MockAgent(params=CFG),
            agent=cls.agent,
        )

    def test_get_first_player(self) -> int:
        r"""Test get_first_player() method."""
        for _ in range(10):
            player = self.trainer.get_first_player()
            self.assertIn(
                player,
                [CFG.agent_player, CFG.opponent_player],
                "Should return a valid player.",
            )

    def test_get_rolling_win_rate_initial(self) -> None:
        r"""Test that the initial rolling win rate is zero."""
        self.assertEqual(
            self.trainer.get_rolling_win_rate(), 0, "Initial win rate should be 0."
        )

    def _test_training(self, opponent: BaseAgent) -> None:
        expected_winners = [CFG.agent_player, CFG.opponent_player, 0]
        try:
            trainer = Trainer(
                params=CFG,
                env=self.env,
                opponent=opponent,
                agent=self.agent,
            )
            for winner, reward, win_rate in trainer.train(self.num_episodes):
                self.assertIn(winner, expected_winners, "Invalid winner.")
                self.assertIsInstance(reward, float, "Reward should be a float.")
                self.assertIsInstance(win_rate, int, "Win rate should be int.")

        except Exception as e:
            self.fail(f"Training raised an exception: {e}")

    def test_training_with_random_agent(self) -> None:
        r"""Test training using a RandomAgent as the opponent."""
        opponent = RandomAgent(params=CFG)
        self._test_training(opponent)

    def test_training_with_rule_based_agent(self) -> None:
        r"""Test training using a RuleBasedAgent as the opponent."""
        opponent = RuleBasedAgent(params=CFG)
        self._test_training(opponent)

    def test_training_self_play(self) -> None:
        """Test self-play training using PPOAgent as both agent and opponent."""
        opponent = PPOAgent(params=CFG, device=self.device)
        self._test_training(opponent)
