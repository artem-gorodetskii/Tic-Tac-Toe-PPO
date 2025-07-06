from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/agent-play")
async def agent_play(request: Request) -> JSONResponse:
    r"""Handles agent's move in the Tic-Tac-Toe game.

    Args:
        request: incoming request.

    Returns:
        JSONResponse containing the agent's move, game over status,
        winner (if any), and the winning line positions.
    """
    data = await request.json()
    board = data.get("board", [])

    env = request.app.state.env
    agent = request.app.state.best_agent

    total_moves = env.count_moves(board)
    winner, line = env.check_winner(board)

    if winner is not None:
        reward = env.compute_reward_episode_end(winner, line, total_moves)
        agent.give_reward(reward, done=True, opponent_move=True)
        agent.update_policy()
        return JSONResponse(
            {"move": -1, "gameOver": True, "winner": winner, "line": line}
        )

    move = agent.act(board)
    board_after = env.record_move(board, move, is_agent=True)
    winner, line = env.check_winner(board_after)

    reward = env.compute_intermediate_reward(
        board_before=board,
        board_after=board_after,
        move=move,
    )
    if winner is not None:
        reward += env.compute_reward_episode_end(winner, line, total_moves)
        agent.give_reward(reward, done=True)
        agent.update_policy()
        return JSONResponse(
            {"move": move, "gameOver": True, "winner": winner, "line": line}
        )
    else:
        agent.give_reward(reward, done=False)
        return JSONResponse(
            {"move": move, "gameOver": False, "winner": None, "line": []}
        )
