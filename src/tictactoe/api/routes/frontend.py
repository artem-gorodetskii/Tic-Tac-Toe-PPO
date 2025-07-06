from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from tictactoe.config import CFG

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def root() -> str:
    r"""Serves the main HTML page of the frontend application.

    Returns:
        The contents of the index.html file as a string.
    """
    return CFG.frontend_index_path.read_text(encoding="utf-8")
