import asyncio
import json
import time
from copy import deepcopy
from threading import Thread
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tictactoe.agents.random_agent import RandomAgent  # noqa
from tictactoe.agents.rule_based_agent import RuleBasedAgent  # noqa
from tictactoe.config import CFG
from tictactoe.environment import TicTacToeEnv
from tictactoe.trainer import Trainer

router = APIRouter()


@router.get("/train")
async def start_training(request: Request) -> JSONResponse:
    r"""Start training the PPO agent.

    Args:
        request: FastAPI request object.

    Returns:
       Status message indicating training start
       or that training is already running.
    """
    state = request.app.state

    if state.is_training:
        return JSONResponse({"status": "already training"})

    state.stop_training = False

    with state.train_logs_lock:
        state.train_logs.clear()

    def _run_training() -> None:
        try:
            if state.is_training:
                return

            state.is_training = True

            # Start training from the best model and init best win rate.
            state.agent = deepcopy(state.best_agent)
            state.agent.reset_memory()
            state.best_win_rate = 0

            trainer = Trainer(
                params=CFG,
                env=TicTacToeEnv(params=CFG),
                opponent=state.opponent_agent,
                agent=state.agent,
            )

            for step, (_, _, rolling_win_rate) in enumerate(trainer.train()):
                if state.stop_training:
                    break

                log = {"step": step, "rolling_win_rate": rolling_win_rate}

                with state.train_logs_lock:
                    state.train_logs.append(log)

                if rolling_win_rate > state.best_win_rate:
                    state.best_win_rate = rolling_win_rate
                    best_agent = deepcopy(trainer.agent)
                    best_agent.reset_memory()
                    state.best_agent = best_agent

                time.sleep(0.0001)

        except Exception as e:
            raise RuntimeError(f"Training thread failed: {e}.") from e

        finally:
            state.is_training = False

        state.is_training = False

    Thread(target=_run_training, daemon=True).start()
    return JSONResponse({"status": "training started"})


@router.get("/train/stream")
async def train_stream(request: Request) -> StreamingResponse:
    r"""Stream training progress logs to the client.

    Args:
        request: FastAPI request object.

    Returns:
        A stream of training logs.
    """
    state = request.app.state

    async def _event_stream() -> AsyncGenerator[str, None]:
        last_step = -1

        while True:
            if not state.is_training or state.stop_training:
                break

            with state.train_logs_lock:
                logs_snapshot = list(state.train_logs)

            for log in logs_snapshot:
                if log["step"] <= last_step:
                    continue
                yield f"data: {json.dumps(log)}\n\n"
                last_step = log["step"]

            await asyncio.sleep(0.0005)

        yield "event: stopped\ndata: {}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@router.get("/stop-training")
async def stop_training_route(request: Request) -> JSONResponse:
    r"""Stop the ongoing training process.

    Args:
        request: FastAPI request object.

    Returns:
        Status message indicating training is stopping.
    """
    request.app.state.stop_training = True
    return JSONResponse({"status": "stopping"})


@router.post("/set_opponent")
async def set_opponent(request: Request) -> JSONResponse:
    r"""Set the opponent agent type for training.

    Args:
        request: FastAPI request object.

    Returns:
        Status message indicating success or error.
    """
    if request.app.state.is_training:
        return JSONResponse(
            {"error": "Cannot change opponent while training"}, status_code=400
        )

    data = await request.json()
    opponent_type = data.get("opponent")

    if opponent_type == "random":
        opponent = RandomAgent(params=CFG)
    elif opponent_type == "rule_based":
        opponent = RuleBasedAgent(params=CFG)
    elif opponent_type == "self":
        opponent = deepcopy(request.app.state.best_agent)
        opponent.reset_memory()
        opponent.set_eval_mode()
    else:
        return JSONResponse({"error": "Invalid opponent type"}, status_code=400)

    request.app.state.opponent_type = opponent_type
    request.app.state.opponent_agent = opponent

    return JSONResponse({"status": "ok"})
