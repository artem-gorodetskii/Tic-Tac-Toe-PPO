from abc import ABC, abstractmethod
from typing import List

from tictactoe.config import Config


class BaseAgent(ABC):
    r"""Abstract base class for Tic-Tac-Toe agents."""

    def __init__(self, params: Config) -> None:
        self.params = params

    @abstractmethod
    def act(self, board: List[int]) -> int:
        r"""Decide the next move based on the current board state.

        Args:
            board: current board.

        Returns:
            The position of the chosen move.
        """
        pass
