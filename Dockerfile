FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (used to honor uv.lock at install time so production builds
# match the lockfile exactly — `pip install .` re-resolves and ignores
# the lockfile).
RUN pip install --no-cache-dir uv

# Install Python dependencies from the lockfile (production deps only).
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
# Batteries are version-controlled content — bake them in so the eval CLI
# (and the future battery-trigger API) can run them wherever the agent runs.
COPY eval/ ./eval/
RUN uv sync --locked --no-dev

# Make the venv's executables findable so CMD can call uvicorn directly.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV ENVIRONMENT=docker
# Build-time code-provenance stamps. CI passes these (deploy-production.yml);
# local compose forwards GIT_COMMIT/GIT_BRANCH from the shell when set. The
# image has no .git, so the eval CLI's get_git_info() reads these instead.
ARG AGENT_VERSION=""
ENV AGENT_VERSION=$AGENT_VERSION
ARG GIT_COMMIT=""
ENV GIT_COMMIT=$GIT_COMMIT
ARG GIT_BRANCH=""
ENV GIT_BRANCH=$GIT_BRANCH

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
