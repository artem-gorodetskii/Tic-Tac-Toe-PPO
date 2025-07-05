FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="/app/src"

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-dev

COPY src ./src
COPY frontend ./frontend

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "tictactoe.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
