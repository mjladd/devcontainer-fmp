FROM python:3.11-slim-bookworm

RUN apt update && \
    apt upgrade -y

RUN apt install g++ gcc curl libsndfile1 libsndfile1-dev zsh git make -y

## install oh-my-zsh
# with robbyrussell" and git/aws plugins
RUN sh -c "$(curl -L  https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p git  \
    -a 'alias sl=ls' \
    -a "alias ls='ls -p'" \
    -a 'unsetopt AUTO_REMOVE_SLASH' \
    -a 'export PYTHONDONTWRITEBYTECODE=1' \
    -a 'export PATH="/root/.local/bin:$PATH"'

# Change default SHELL
RUN chsh -s /usr/bin/zsh

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY ./pyproject.toml /tmp/pyproject.toml

# Create virtual environment and install dependencies using uv
WORKDIR /tmp
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r pyproject.toml

# Configure nbdime for git
RUN /opt/venv/bin/nbdime config-git --enable --global

# Set environment variables
ENV PYTHONBUFFERED=TRUE
ENV DEVCONTAINER=true
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
