from fastapi import APIRouter
from fastapi.responses import JSONResponse

from tictactoe.config import CFG

router = APIRouter()


@router.get("/env_config")
async def get_env_config() -> JSONResponse:
    """Returns the environment configuration parameters.

    Returns:
        A response containing key environment settings.
    """
    return JSONResponse(
        {
            "board_size": CFG.board_size,
            "free_cell_val": CFG.free_cell_val,
            "agent_player": CFG.agent_player,
            "opponent_player": CFG.opponent_player,
        }
    )
