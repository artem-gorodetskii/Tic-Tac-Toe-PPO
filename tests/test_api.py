import time
from collections import deque
from copy import deepcopy
from threading import Lock
from unittest import TestCase

from fastapi.testclient import TestClient

from tictactoe.agents.ppo_agent import PPOAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.api.main import app
from tictactoe.config import CFG
from tictactoe.environment import TicTacToeEnv
from tictactoe.utils import get_available_device


class TestAPI(TestCase):
    r"""Tests for the FastAPI application endpoints."""

    @classmethod
    def setUp(cls) -> None:
        r"""Set up test environment for all test cases."""
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

        cls.app = app
        cls.client = TestClient(app=cls.app)

        cls.wait_time = 0.25

    def test_env_config(self) -> None:
        r"""Test that the /env_config endpoint returns correct configuration."""
        response = self.client.get("/env_config")
        self.assertEqual(response.status_code, 200, "The status code must be 200.")
        self.assertEqual(response.status_code, 200, "The status code must be 200.")
        data = response.json()
        self.assertEqual(
            data["board_size"], CFG.board_size, "Invalid board_size value."
        )
        self.assertEqual(
            data["free_cell_val"], CFG.free_cell_val, "Invalid free_cell_val value."
        )
        self.assertEqual(
            data["agent_player"], CFG.agent_player, "Invalid agent_player value."
        )
        self.assertEqual(
            data["opponent_player"],
            CFG.opponent_player,
            "Invalid opponent_player value.",
        )

    def test_agent_play_empty_board(self) -> None:
        r"""Test /agent-play endpoint on an empty board."""
        board = [0] * 9
        response = self.client.post("/agent-play", json={"board": board})
        self.assertEqual(response.status_code, 200, "The status code must be 200.")
        data = response.json()
        self.assertIn("move", data, "The data must contain a filed 'move'.")
        self.assertIn("gameOver", data, "The data must contain a filed 'gameOver'.")
        self.assertIn("winner", data, "The data must contain a filed 'winner'.")
        self.assertIn("line", data, "The data must contain a filed 'line'.")
        expected = data["move"] == -1 or (0 <= data["move"] < 9)
        self.assertTrue(expected, "Invalid move value.")

    def test_agent_play_win(self) -> None:
        r"""Test /agent-play endpoint when agent has a winning move."""
        board = [2, 2, 0, 1, 2, 1, 1, 1, 0]
        response = self.client.post("/agent-play", json={"board": board})
        self.assertEqual(response.status_code, 200, "The status code must be 200.")
        data = response.json()
        self.assertTrue(data["winner"] == 2, "The agent had to win.")
        expected = data["line"] in [[0, 1, 2], [0, 4, 8]]
        self.assertTrue(expected, "Invalid win pattern.")

    def test_train_start_and_stop(self) -> None:
        r"""Test the /train and /stop-training endpoints."""
        response = self.client.get("/train")
        self.assertEqual(response.status_code, 200, "The status code must be 200.")
        self.assertTrue(
            response.json()["status"] in ["training started", "already training"],
            "The training status must be active.",
        )
        time.sleep(self.wait_time)

        self.assertTrue(
            len(self.app.state.train_logs) > 0, "Train log must be not empty."
        )

        response = self.client.get("/stop-training")
        self.assertEqual(response.status_code, 200, "The status code must be 200.")
        self.assertEqual(
            response.json()["status"], "stopping", "The train log must not be empty."
        )
        time.sleep(self.wait_time)

    def test_set_opponent(self) -> None:
        r"""Test the /set_opponent endpoint."""
        response = self.client.post("/set_opponent", json={"opponent": "rule_based"})
        self.assertEqual(response.status_code, 200, "The status code must be 200.")
        self.assertTrue(app.state.opponent_type == "rule_based", "Wrong opponent type.")

        response = self.client.post("/set_opponent", json={"opponent": "random"})
        self.assertEqual(response.status_code, 200, "The status code must be 200.")
        self.assertTrue(app.state.opponent_type == "random", "Wrong opponent type.")

        response = self.client.post("/set_opponent", json={"opponent": "self"})
        self.assertEqual(response.status_code, 200, "The status code must be 200.")
        self.assertTrue(app.state.opponent_type == "self", "Wrong opponent type.")

        response = self.client.get("/train")
        self.assertEqual(response.status_code, 200, "The status code must be 200.")
        response = self.client.post("/set_opponent", json={"opponent": "self"})
        time.sleep(self.wait_time)

        self.assertEqual(response.status_code, 400, "The status code must be 400.")
        response = self.client.get("/stop-training")
        self.assertEqual(response.status_code, 200, "The status code must be 200.")
