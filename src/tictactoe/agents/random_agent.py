import random
from typing import List

from tictactoe.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    r"""Agent that selects moves randomly from available empty cells."""

    def act(self, board: List[int]) -> int:
        r"""Decide the next move based on the current board state.

        Args:
            board: current board.

        Returns:
            The position of the chosen move.
        """
        empty_cells = [i for i, c in enumerate(board) if c == self.params.free_cell_val]
        if not empty_cells:
            return -1
        return random.choice(empty_cells)
