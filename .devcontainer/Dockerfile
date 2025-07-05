FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl build-essential git tree && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash vscode

USER vscode
ENV POETRY_HOME="/home/vscode/.poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /workspace
