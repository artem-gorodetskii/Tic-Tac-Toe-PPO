from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def root() -> str:
    r"""Serves the main HTML page of the frontend application.

    Returns:
        The contents of the `index.html` file as a string.
    """
    index_path = Path("frontend/index.html")
    return index_path.read_text(encoding="utf-8")
