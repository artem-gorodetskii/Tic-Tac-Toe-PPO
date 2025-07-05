from unittest import TestCase

from tictactoe.config import CFG
from tictactoe.environment import TicTacToeEnv


class TestTicTacToeEnv(TestCase):
    r"""Unit tests for the TicTacToeEnv class."""

    @classmethod
    def setUp(cls) -> None:
        r"""Set up test environment for all test cases."""
        cls.env = TicTacToeEnv(params=CFG)

    def test_create_board(self) -> None:
        r"""Test that the created board has the correct size and initial values."""
        board = self.env.create_board()
        self.assertEqual(
            len(board), CFG.board_size**2, "Board should have correct number of cells."
        )
        self.assertTrue(
            all(cell == CFG.free_cell_val for cell in board),
            "All cells should be initialized with the free cell value.",
        )

    def test_check_winner_horizontal(self) -> None:
        r"""Test horizontal win detection logic."""
        board = self.env.create_board()
        board[0:3] = [CFG.agent_player] * 3
        winner, line = self.env.check_winner(board)
        expected = CFG.agent_player
        self.assertEqual(winner, expected, "Agent should be detected as the winner.")
        expected = [0, 1, 2]
        self.assertEqual(line, expected, "Winning line should be the top row.")

    def test_check_draw(self) -> None:
        r"""Test that a full board with no winner results in a draw."""
        board = [
            CFG.agent_player,
            CFG.opponent_player,
            CFG.agent_player,
            CFG.opponent_player,
            CFG.agent_player,
            CFG.opponent_player,
            CFG.opponent_player,
            CFG.agent_player,
            CFG.opponent_player,
        ]
        winner, line = self.env.check_winner(board)
        self.assertEqual(
            winner, 0, "Game should be a draw when all cells are filled with no winner."
        )
        self.assertEqual(line, [], "No winning line expected in draw.")

    def test_record_move(self) -> None:
        r"""Test that a valid move updates the board correctly."""
        board = self.env.create_board()
        move = 4
        board_ = self.env.record_move(board, move, is_agent=True)
        self.assertEqual(
            board_[move], CFG.agent_player, "Agent move should update the board."
        )
        self.assertNotEqual(board, board_, "Original board should not be modified.")

    def test_invalid_move_raises(self) -> None:
        r"""Test that making a move to a non-free cell raises an error."""
        board = self.env.create_board()
        board[0] = CFG.agent_player

        with self.assertRaises(
            RuntimeError,
            msg="Should raise if trying to move into a non-free cell.",
        ):
            self.env.record_move(board, 0, is_agent=False)

    def test_count_moves(self) -> None:
        r"""Test that move count returns the correct number of occupied cells."""
        board = self.env.create_board()
        board[0] = CFG.agent_player
        board[1] = CFG.opponent_player
        moves = self.env.count_moves(board)
        self.assertEqual(moves, 2, "Should count only 2 moves.")

    def test_intermediate_reward_validity(self) -> None:
        r"""Test that the intermediate reward is computed and returned as a float."""
        board = self.env.create_board()
        board[0] = CFG.agent_player
        board[1] = CFG.agent_player
        board_before = board.copy()
        board_after = self.env.record_move(board, 2, is_agent=True)
        reward = self.env.compute_intermediate_reward(board_before, board_after, 2)
        self.assertIsInstance(reward, float, "Invalid type for intermediate reward.")

    def test_compute_reward_episode_end_win(self) -> None:
        r"""Test reward computation when the agent wins the game."""
        board = [
            CFG.agent_player,
            CFG.agent_player,
            CFG.agent_player,
            CFG.opponent_player,
            CFG.agent_player,
            CFG.free_cell_val,
            CFG.opponent_player,
            CFG.free_cell_val,
            CFG.opponent_player,
        ]
        _, line = self.env.check_winner(board)
        reward = self.env.compute_reward_episode_end(
            winner=CFG.agent_player,
            line=line,
            total_moves=7,
        )
        self.assertGreaterEqual(reward, CFG.win, "Incorect reward.")

    def test_compute_reward_episode_end_draw(self) -> None:
        r"""Test reward computation when the game ends in a draw."""
        board = [
            CFG.agent_player,
            CFG.opponent_player,
            CFG.agent_player,
            CFG.opponent_player,
            CFG.agent_player,
            CFG.opponent_player,
            CFG.opponent_player,
            CFG.agent_player,
            CFG.opponent_player,
        ]
        _, line = self.env.check_winner(board)
        reward = self.env.compute_reward_episode_end(
            winner=0,
            line=line,
            total_moves=9,
        )
        self.assertEqual(reward, CFG.draw, "Incorect reward.")

    def test_update_history_adds_pattern(self) -> None:
        r"""Test that the agent's doublet pattern is added to history."""
        board = self.env.create_board()
        board[0] = CFG.agent_player
        board[1] = CFG.agent_player
        board[2] = CFG.free_cell_val
        self.env._update_history(board, CFG.agent_player)

        self.assertEqual(
            len(self.env.agent_doublet_history), 1, "Should record a doublet pattern."
        )

    def test_blocked_opponent_opportunity_true(self) -> None:
        r"""Test that blocking an opponent’s potential win is detected."""
        before = self.env.create_board()
        before[0] = CFG.opponent_player
        before[1] = CFG.opponent_player

        after = before.copy()
        after[2] = CFG.agent_player

        result = self.env._blocked_opponent_opportunity(before, after)
        self.assertTrue(result, "Agent should be detected as having blocked opponent.")

    def test_reward_for_win(self) -> None:
        r"""Test that a win yields positive reward."""
        reward = self.env.compute_reward_episode_end(
            winner=CFG.agent_player, line=[0, 1, 2]
        )
        self.assertGreater(reward, 0.0, "Winning should yield positive reward.")

    def test_reward_for_draw(self) -> None:
        r"""Test that a draw yields the draw reward value."""
        reward = self.env.compute_reward_episode_end(winner=0, line=[])
        self.assertEqual(reward, CFG.draw, "Draw should yield draw reward.")

    def test_reward_for_loss(self) -> None:
        r"""Test that a loss yields the correct loss penalty."""
        reward = self.env.compute_reward_episode_end(
            winner=CFG.opponent_player, line=[3, 4, 5]
        )
        self.assertEqual(reward, CFG.lose, "Loss should yield loss penalty.")

    def test_miss_win_opportunity_penalty(self) -> None:
        r"""Test that missing a win opportunity leads to a penalty."""
        board = self.env.create_board()

        board[0] = CFG.agent_player
        board[1] = CFG.agent_player
        board[2] = CFG.free_cell_val

        # The agent made a move and created a potential win.
        self.env._update_history(board, CFG.agent_player)
        # On the next move, the agent misses the opportunity to win.
        self.env._update_history(board, CFG.agent_player)

        reward = self.env._miss_win_opportunity()
        self.assertLess(
            reward, 0.0, "Should receive penalty for missing win opportunity."
        )

    def test_thoughtful_victory(self) -> None:
        r"""Test that a thoughtful (non-random) victory is recognized correctly."""
        board = self.env.create_board()

        board[0] = CFG.agent_player
        self.env._update_history(board, CFG.agent_player)
        board[1] = CFG.opponent_player

        board[4] = CFG.agent_player
        self.env._update_history(board, CFG.agent_player)
        board[3] = CFG.opponent_player

        board[8] = CFG.agent_player
        self.env._update_history(board, CFG.agent_player)

        winner, line = self.env.check_winner(board)
        self.assertEqual(winner, CFG.agent_player, "The agent had to win.")

        status = self.env._thoughtful_victory(line)
        self.assertTrue(status, "This victory had to be thought out.")
