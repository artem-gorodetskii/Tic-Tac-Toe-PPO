from collections import deque
from contextlib import asynccontextmanager
from copy import deepcopy
from threading import Lock
from typing import AsyncGenerator

from fastapi import FastAPI

from tictactoe.agents.ppo_agent import PPOAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.config import CFG
from tictactoe.environment import TicTacToeEnv
from tictactoe.utils import get_available_device


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    r"""Application lifespan context manager for FastAPI.

    Args:
        app: FastAPI application instance.

    Yields:
        AsyncGenerator that manages the application state.
    """
    env = TicTacToeEnv(params=CFG)
    agent = PPOAgent(params=CFG, device=get_available_device())
    best_agent = deepcopy(agent)

    app.state.env = env
    app.state.agent = agent
    app.state.opponent_type = "random"
    app.state.opponent_agent = RandomAgent(params=CFG)
    app.state.best_agent = best_agent
    app.state.best_win_rate = 0
    app.state.is_training = False
    app.state.stop_training = False
    app.state.train_logs = deque(maxlen=1000)
    app.state.train_logs_lock = Lock()

    yield
