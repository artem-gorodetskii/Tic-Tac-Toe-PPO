import random
from collections import deque
from typing import Generator, Tuple

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents.ppo_agent import PPOAgent
from tictactoe.config import CFG, Config
from tictactoe.environment import TicTacToeEnv


class Trainer:
    r"""Agent training class for the Tic-Tac-Toe environment."""

    def __init__(
        self,
        params: Config,
        env: TicTacToeEnv,
        opponent: BaseAgent,
        agent: PPOAgent,
    ) -> None:
        self.params = params
        self.env = env
        self.opponent = opponent
        self.agent = agent

        self.last_100_wins = deque(maxlen=100)

    def get_first_player(self) -> int:
        r"""Randomly select the first player for the next episode.

        Returns:
            Player identifier (agent or opponent) who will move first.
        """
        return random.choice([self.params.opponent_player, self.params.agent_player])

    def get_rolling_win_rate(self) -> int:
        r"""Calculate the rolling win rate over the last 100 episodes.

        Returns:
            Number of agent wins in the last 100 episodes.
        """
        return sum(self.last_100_wins)

    def play_episode(self, first_player: int) -> Tuple[int, float]:
        r"""Play a full game episode between the agent and the opponent.

        Args:
            first_player: player who will move first in the episode.

        Returns:
            A tuple containing the winning player and the total episode reward.
        """
        board = self.env.create_board()
        total_moves = 0
        episode_reward = 0.0
        current_player = first_player

        while True:
            board_after = None

            if current_player == self.params.opponent_player:
                move = self.opponent.act(board)
                board = self.env.record_move(board, move, is_agent=False)
                reward = 0.0
            else:
                move = self.agent.act(board)
                board_after = self.env.record_move(board, move, is_agent=True)
                reward = self.env.compute_intermediate_reward(
                    board_before=board, board_after=board_after, move=move
                )
                episode_reward += reward

            board_eval = board_after if board_after is not None else board
            total_moves += 1
            winner, line = self.env.check_winner(board_eval)

            if winner is not None:
                final_reward = self.env.compute_reward_episode_end(
                    winner, line, total_moves
                )
                reward += final_reward
                episode_reward += final_reward

                opponent_move = current_player != self.params.agent_player
                self.agent.give_reward(reward, done=True, opponent_move=opponent_move)
                self.agent.update_policy()

                self.last_100_wins.append(int(winner == self.params.agent_player))

                return winner, episode_reward

            if current_player == self.params.agent_player:
                self.agent.give_reward(reward, done=False)

            current_player = (
                self.params.opponent_player
                if current_player == self.params.agent_player
                else self.params.agent_player
            )
            if board_after is not None:
                board = board_after

    def train(
        self,
        num_episodes: int = CFG.num_episodes,
    ) -> Generator[Tuple[int, float, int], None, None]:
        r"""Train the agent over multiple episodes.

        Args:
            num_episodes: number of training episodes.

        Yields:
            A generator yielding tuples of (winner, episode_reward, rolling_win_rate).
        """
        for _ in range(num_episodes):
            first_player = self.get_first_player()
            winner, episode_reward = self.play_episode(first_player)
            win_rate = self.get_rolling_win_rate()

            yield winner, episode_reward, win_rate
