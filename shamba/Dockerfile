FROM python:3.10.16-slim

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl -fsSL https://pyenv.run | bash

# Set environment variables to make pyenv available in future RUN commands
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

# # Initialize pyenv in any future shell
RUN echo 'eval "$(pyenv init --path)"' >> /etc/profile.d/pyenv.sh
RUN echo 'eval "$(pyenv init -)"' >> /root/.bashrc

# Install Python 3.10 using pyenv
RUN pyenv install 3.10.16
RUN pyenv global 3.10.16

# Install poetry
ENV POETRY_HOME="/root/.local"
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

# Set the working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY pyproject.toml poetry.lock __init__.py ./
COPY model/ ./model/

RUN poetry env use python3.10  # Explicitly specify the Python version for poetry

# Install dependencies using pipenv
# RUN poetry lock
RUN poetry config virtualenvs.create false
RUN poetry install --no-root --no-interaction
RUN poetry env use 3.10
