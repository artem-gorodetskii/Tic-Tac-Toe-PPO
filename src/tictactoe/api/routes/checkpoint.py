import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from tictactoe.config import CFG
from tictactoe.utils import get_app_root

CHECKPOINT_PATH = get_app_root() / CFG.checkpoint_path

router = APIRouter()


@router.post("/save_checkpoint")
def save_checkpoint(request: Request) -> JSONResponse:
    r"""Saves the model state of the best agent to a checkpoint file.

    Args:
        request: incoming FastAPI request.

    Returns:
        A response indicating the checkpoint has been saved.
    """
    torch.save(request.app.state.best_agent.model.state_dict(), CHECKPOINT_PATH)
    return JSONResponse({"status": "Checkpoint saved"})


@router.post("/load_checkpoint")
def load_checkpoint(request: Request) -> JSONResponse:
    r"""Loads the model state from a checkpoint file, if it exists.

    Args:
        request: incoming FastAPI request.

    Returns:
        A response indicating whether the checkpoint was
        successfully loaded or not found.
    """
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH)
        request.app.state.best_agent.model.load_state_dict(checkpoint)
        return JSONResponse({"status": "Checkpoint loaded"})
    return JSONResponse({"status": "No checkpoint found", "error": True})
