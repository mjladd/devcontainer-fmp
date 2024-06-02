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
    -a 'export PYTHONDONTWRITEBYTECODE=1'


# Change default SHELL
RUN chsh -s /usr/bin/zsh

# install uv
#RUN curl -LsSf https://astral.sh/uv/install.sh | sh

#RUN cd /workspaces/python-devcontainer &&  /root/.cargo/bin/uv venv
#RUN  /workspaces/python-devcontainer/.venv/bin/activate

COPY ./requirements.txt requirements.txt
#RUN /root/.cargo/bin/uv pip install --upgrade pip
#RUN /root/.cargo/bin/uv pip  install -r requirements.txt
# RUN source /root/.cargo/env
ENV PYTHONBUFFERED=TRUE
ENV DEVCONTAINER=true
