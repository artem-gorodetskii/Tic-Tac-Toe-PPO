from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
def health_check() -> JSONResponse:
    r"""Health check endpoint to verify the API service is running.

    Returns:
        JSONResponse object indicating the service status.
    """
    return JSONResponse({"status": "ok"})
