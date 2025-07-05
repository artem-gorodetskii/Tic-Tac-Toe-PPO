from copy import deepcopy
from unittest import TestCase

import torch

from tictactoe.agents.ppo_agent import PPOAgent
from tictactoe.agents.rule_based_agent import RuleBasedAgent
from tictactoe.config import CFG
from tictactoe.environment import TicTacToeEnv


class TestPPOAgent(TestCase):
    r"""Unit tests for the PPOAgent class."""

    @classmethod
    def setUp(cls) -> None:
        r"""Set up test environment for all test cases."""
        cls.device = torch.device("cpu")
        cls.agent = PPOAgent(CFG, cls.device)

    def test_act_and_memory(self) -> None:
        r"""Test that act() produces a valid action and updates agent memory."""
        board = [CFG.free_cell_val] * CFG.input_dim
        move = self.agent.act(board)
        self.assertIsInstance(move, int, "Action should be an int.")
        self.assertEqual(len(self.agent.states), 1, "State should be stored after act.")
        self.assertEqual(
            len(self.agent.actions), 1, "Action should be stored after act."
        )
        self.assertEqual(
            len(self.agent.log_probs), 1, "Log prob should be stored after act."
        )

    def test_give_reward(self) -> None:
        r"""Test that give_reward() correctly stores reward and done flag."""
        board = [CFG.free_cell_val] * CFG.input_dim
        self.agent.act(board)
        self.agent.give_reward(1.0, done=False)
        self.assertEqual(self.agent.rewards, [1.0], "Reward should be recorded.")
        self.assertEqual(self.agent.dones, [False], "Done flag should be recorded.")

    def test_update_policy(self) -> None:
        r"""Test that update_policy() updates model parameters correctly."""
        env = TicTacToeEnv(params=CFG)
        opponent = RuleBasedAgent(params=CFG)
        board = env.create_board()

        for _ in range(10):
            reward = 0.0
            move = self.agent.act(board)
            board_after = env.record_move(board, move, is_agent=True)
            reward += env.compute_intermediate_reward(board, board_after, move)

            winner, line = env.check_winner(board_after)
            if winner is not None:
                reward += env.compute_reward_episode_end(
                    winner=winner,
                    line=line,
                    total_moves=env.count_moves(board_after),
                )
                self.agent.give_reward(reward, done=True)
                break
            else:
                self.agent.give_reward(reward, done=True)

            board = board_after
            move = opponent.act(board)
            board = env.record_move(board, move, is_agent=False)

            winner, line = env.check_winner(board)
            if winner is not None:
                reward = env.compute_reward_episode_end(
                    winner=winner,
                    line=line,
                    total_moves=env.count_moves(board),
                )
                self.agent.give_reward(reward, done=True, opponent_move=True)
                break

        state_before = deepcopy(list(self.agent.model.parameters()))

        try:
            self.agent.update_policy()
        except Exception as e:
            self.fail(f"update_policy raised unexpected exception: {e}.")

        state_after = list(self.agent.model.parameters())
        any_changed = any(
            not torch.equal(w1, w2)
            for w1, w2 in zip(state_before, state_after, strict=False)
        )
        self.assertTrue(
            any_changed, "Model parameters did not change after update_policy."
        )
