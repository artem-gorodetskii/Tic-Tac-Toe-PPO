import random
from typing import List

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.config import Config
from tictactoe.environment import WIN_PATTERNS


class RuleBasedAgent(BaseAgent):
    r"""Agent that chooses moves based on predefined rules and probabilities."""

    def __init__(self, params: Config) -> None:
        super().__init__(params)

        self.corners = [
            0,
            self.params.board_size - 1,
            self.params.board_size**2 - self.params.board_size,
            self.params.board_size**2 - 1,
        ]

        self.sides = []
        for i in range(self.params.board_size**2):
            if i not in self.corners and i != self.params.board_center:
                self.sides.append(i)

    def act(self, board: List[int]) -> int:
        r"""Decide the next move based on the current board state.

        Args:
            board: current board.

        Returns:
            The position of the chosen move.
        """
        # Win or block.
        for a, b, c in WIN_PATTERNS:
            line = [board[a], board[b], board[c]]
            pos = [a, b, c]

            # Try to win.
            if (
                random.random() <= self.params.rb_agent_win_prob
                and line.count(self.params.agent_player) == 2
                and line.count(self.params.free_cell_val) == 1
            ):
                return pos[line.index(self.params.free_cell_val)]

            # Try to block.
            if (
                random.random() <= self.params.rb_agent_block_prob
                and line.count(self.params.opponent_player) == 2
                and line.count(self.params.free_cell_val) == 1
            ):
                return pos[line.index(self.params.free_cell_val)]

        # Center.
        if (
            random.random() <= self.params.rb_agent_center_prob
            and board[self.params.board_center] == self.params.free_cell_val
        ):
            return self.params.board_center

        # Coners.
        available_corners = [
            i for i in self.corners if board[i] == self.params.free_cell_val
        ]
        if random.random() <= self.params.rb_agent_coners_prob and available_corners:
            return random.choice(available_corners)

        # Sides.
        available_sides = [
            i for i in self.sides if board[i] == self.params.free_cell_val
        ]
        if random.random() <= self.params.rb_agent_sides_prob and available_sides:
            return random.choice(available_sides)

        # Random move.
        empty_cells = [
            i for i, cell in enumerate(board) if cell == self.params.free_cell_val
        ]
        if empty_cells:
            return random.choice(empty_cells)

        return -1
