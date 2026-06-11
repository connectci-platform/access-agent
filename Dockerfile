FROM python:3.12-slim

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
RUN uv sync --locked --no-dev

# Make the venv's executables findable so CMD can call uvicorn directly.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV ENVIRONMENT=docker
# Build-time version stamp (CI passes --build-arg AGENT_VERSION="$(git describe --tags --always --dirty)").
ARG AGENT_VERSION=""
ENV AGENT_VERSION=$AGENT_VERSION

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
