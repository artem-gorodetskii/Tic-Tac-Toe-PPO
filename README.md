# Tic-Tac-Toe PPO Agent

An AI-powered Tic-Tac-Toe agent trained with Proximal Policy Optimization (PPO), built using PyTorch and FastAPI. This project includes training utilities, REST API endpoints, and a simple web interface for playing against the agent.


### 🚀 Features
- PPO-based agent with Generalized Advantage Estimation (GAE)
- Multiple opponent types: random, rule-based, and self-play
- REST API powered by FastAPI for training and gameplay
- Save and load the best-performing agent checkpoint
- Web interface for human-agent interaction
- Live training metric logging
- Full unit test suite


### 🧩 Project Structure
```bash
.
├── src/tictactoe/           # Core logic (agents, model, environment, API)
│   ├── agents/              # Agent implementations
│   ├── api/                 # FastAPI app and API routes
│   ├── model.py             # Actor-Critic model
│   ├── environment.py       # Game environment and rewards
│   ├── trainer.py           # Training loop logic
│   └── config.py            # Configuration parameters
├── frontend/                # Web interface (HTML + JS)
├── tests/                   # Unit tests for key components
├── checkpoints/             # Saved agent models
├── .devcontainer/           # VS Code Dev Container setup
├── Dockerfile               # Docker container
├── docker-compose.yml       # Full-stack orchestration
├── pyproject.toml           # Poetry configuration
```

### ▶️ Quick Start

Ensure you have Docker and Docker Compose installed. Then launch the app:
```bash
docker-compose up --build
```

The game UI will be available at: `http://localhost:8000`.


### 🎮 UI Overview
The web UI allows you to interact with the agent and control training directly
from your browser (see screenshot below).

<p align="center">
    <br>
        <img alt="img-name" src="assets/web_interface.png" width="500">
    <br>
        <em>Tic-Tac-Toe user interface.</em>
    <br>
</p>

UI Components:

1. Game Board
    - 3×3 interactive grid for playing against the agent.
    - You always play first.

2. `↻ Next round` button
    - Resets the board and starts a new round after a game ends.
    - After reset, the opponent plays first.

3. Training Controls
    - `▶ Run Training`: Starts PPO agent training in the background.
    - `↓ Save Checkpoint`: Saves the current agent to disk.
    - `↑ Load Checkpoint`: Loads the best saved agent from disk.
    - `⚁ Random Agent`: Opponent plays randomly.
    - `⚙ Rule-Based Agent`: Uses simple heuristics.
    - `↺ Self-Play`: Agent plays against a copy of itself.

4. Training Progress Graph
    - Displays the win rate of the PPO agent across training steps (max 100 games per point).
    - Red line shows the highest recorded win rate.
    - The graph updates in a sliding-window manner, displaying only the latest 1000 points.


### 🧪 Running Tests
```bash
poetry run pytest
```

### 📄 License
[MIT License](https://github.com/artem-gorodetskii/Tic-Tac-Toe-PPO/blob/main/LICENSE)
