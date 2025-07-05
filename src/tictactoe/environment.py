from collections import deque
from typing import Deque, List, Optional, Tuple

from tictactoe.config import Config

WIN_PATTERNS: List[List[int]] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
]


class _History:
    def __init__(
        self,
        history_size: int,
    ) -> None:
        assert history_size > 0
        self.history_size = history_size
        self.deque: Deque[List[int, List[Tuple[int, int, int]]]] = deque()

    def add(
        self,
        pattern: List[int],
        step: int,
    ) -> None:
        pattern_ = tuple(pattern)

        if not self.deque or self.deque[-1][0] != step:
            self.deque.append([step, [pattern_]])
        else:
            self.deque[-1][1].append(pattern_)

        if len(self.deque) > self.history_size:
            self.deque.popleft()

    def get_last_update_step(self) -> Optional[int]:
        if self.deque:
            return self.deque[-1][0]
        return None

    def __len__(self) -> int:
        return len(self.deque)


class TicTacToeEnv:
    r"""Class for the Tic-Tac-Toe environment."""

    def __init__(
        self,
        params: Config,
    ) -> None:
        self.params = params
        self._init_history()

    def create_board(self) -> List[int]:
        r"""Create an empty game board.

        Returns:
            A list representing the game board.
        """
        board = [self.params.free_cell_val] * (self.params.board_size**2)
        return board

    def check_winner(self, board: List[int]) -> Tuple[Optional[int], List[int]]:
        r"""Check for a winner on the board.

        Args:
            board: current state of the game board.

        Returns:
            The winning player (or None) and the winning line (or empty list).
        """
        for a, b, c in WIN_PATTERNS:
            if (
                board[a] != self.params.free_cell_val
                and board[a] == board[b]
                and board[a] == board[c]
            ):
                return board[a], [a, b, c]

        if all(c != self.params.free_cell_val for c in board):
            return 0, []

        return None, []

    def count_moves(self, board: List[int]) -> int:
        r"""Count the number of moves.

        Args:
            board: current state of the game board.

        Returns:
            The number of non-empty cells.
        """
        return sum(1 for x in board if x != self.params.free_cell_val)

    def record_move(self, board: List[int], move: int, is_agent: bool) -> List[int]:
        r"""Record a move on the board and return the updated board.

        Args:
            board: current state of the game board.
            move: index of the move to apply.
            is_agent: whether the move is made by the agent or oppponent.

        Returns:
            The updated game board.
        """
        if not board[move] == self.params.free_cell_val:
            raise RuntimeError("Invalid move.")

        board_ = board.copy()
        player = self.params.agent_player if is_agent else self.params.opponent_player
        board_[move] = player
        return board_

    def compute_reward_episode_end(
        self,
        winner: int,
        line: List[int],
        total_moves: int = 0,
    ) -> float:
        r"""Compute the final reward at the end of an episode.

        Args:
            winner: winning player.
            line: winning cell indices.
            total_moves: total number of moves made in the game.

        Returns:
            The episode-end reward.
        """
        if winner == self.params.agent_player:
            reward = self.params.win

            if total_moves <= self.params.fast_win_moves:
                reward += self.params.fast_win_bonus

            if self._thoughtful_victory(line):
                reward += self.params.thoughtful_victory_bonus

        elif winner == 0:
            reward = self.params.draw

        else:
            reward = self.params.lose

        self._init_history()

        return reward

    def compute_intermediate_reward(
        self,
        board_before: List[int],
        board_after: List[int],
        move: int,
    ) -> float:
        r"""Compute the intermediate reward after a move.

        Args:
            board_before: board state before the move.
            board_after: board state after the move.
            move: move made by the agent.

        Returns:
            The reward for this intermediate step.
        """
        self._update_history(board_after, self.params.agent_player)

        reward = self.params.move_bonus

        if move == self.params.board_center:
            reward += self.params.center_bonus

        reward += self.params.two_in_row_bonus * self._count_two_in_row(
            self.agent_doublet_history
        )
        block_status = self._blocked_opponent_opportunity(board_before, board_after)

        if block_status is not None:
            if block_status:
                reward += self.params.block_opponent_bonus
            else:
                reward += self.params.block_miss_penalty

        reward += self._miss_win_opportunity()

        return reward

    def _init_history(self) -> None:
        self.agent_doublet_history = _History(self.params.history_size)
        self.opponent_doublet_history = _History(self.params.history_size)
        self.history_step = 0

    def _update_history(self, board: List[int], player: int) -> None:
        self.history_step += 1

        for a, b, c in WIN_PATTERNS:
            cells = [board[a], board[b], board[c]]

            if cells.count(player) == 2 and cells.count(self.params.free_cell_val) == 1:
                if player == self.params.agent_player:
                    self.agent_doublet_history.add(
                        pattern=[a, b, c], step=self.history_step
                    )
                elif player == self.params.opponent_player:
                    self.opponent_doublet_history.add(
                        pattern=[a, b, c], step=self.history_step
                    )

    def _count_two_in_row(self, history: _History) -> int:
        last_update_step = history.get_last_update_step()

        if last_update_step is None or last_update_step != self.history_step:
            return 0

        return len(history.deque[-1][1])

    def _blocked_opponent_opportunity(
        self, board_before: List[int], board_after: List[int]
    ) -> Optional[bool]:
        for a, b, c in WIN_PATTERNS:
            line_before = [board_before[a], board_before[b], board_before[c]]
            line_after = [board_after[a], board_after[b], board_after[c]]
            if (
                line_before.count(self.params.opponent_player) == 2
                and line_before.count(self.params.free_cell_val) == 1
            ):
                # If after the agent's move there is no longer such a threat.
                if not (
                    line_after.count(self.params.opponent_player) == 2
                    and line_after.count(self.params.free_cell_val) == 1
                ):
                    return True
                else:
                    return False
        return None

    def _miss_win_opportunity(self) -> float:
        if not self.agent_doublet_history:
            return 0.0

        reward_ = 0.0
        current_step = self.history_step
        last_step = self.agent_doublet_history.get_last_update_step()

        if last_step != current_step:
            return reward_

        curr_patterns = set(self.agent_doublet_history.deque[-1][1])

        for i in range(len(self.agent_doublet_history) - 2, -1, -1):
            prev_step, prev_patterns = self.agent_doublet_history.deque[i]
            if curr_patterns & set(prev_patterns):
                time_gap = current_step - prev_step
                factor = self.params.miss_win_opportunity_decay ** (time_gap - 1)
                reward_ += self.params.miss_win_opportunity_penalty * factor

        return reward_

    def _thoughtful_victory(self, pattern: List[int]) -> bool:
        if not len(self.agent_doublet_history):
            return False

        prev_step = self.agent_doublet_history.deque[-1][0]

        if self.history_step - prev_step != 1:
            return False

        prev_prev_step = None
        prev_prev_pattern = None

        if len(self.agent_doublet_history) > 1:
            prev_prev_step, prev_prev_pattern = self.agent_doublet_history.deque[-2]

        if prev_prev_step is None:
            return True

        return tuple(pattern) not in set(prev_prev_pattern)
