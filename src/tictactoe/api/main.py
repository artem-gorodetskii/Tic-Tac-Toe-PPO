from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tictactoe.api.lifespan import lifespan
from tictactoe.api.routes import (
    checkpoint,
    frontend,
    game,
    health,
    training,
)
from tictactoe.api.routes import (
    config as config_route,
)
from tictactoe.config import CFG
from tictactoe.utils import get_app_root, set_seed

set_seed(CFG.seed)

app = FastAPI(lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(directory=get_app_root() / CFG.frontend_dir_name, html=True),
    name=CFG.static_dir_name,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes.
app.include_router(training.router)
app.include_router(game.router)
app.include_router(config_route.router)
app.include_router(frontend.router)
app.include_router(health.router)
app.include_router(checkpoint.router)
