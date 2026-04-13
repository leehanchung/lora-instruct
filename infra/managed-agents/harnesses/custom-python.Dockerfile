# Harness: Custom Python agent loop
#
# Runs the built-in agent loop (src.runtime.agent_loop) with tool
# implementations for bash, file ops, web fetch, etc.
#
# Build:
#   docker build -f harnesses/custom-python.Dockerfile -t managed-agents-runtime:latest .

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git ca-certificates \
    build-essential \
    jq vim less \
    && rm -rf /var/lib/apt/lists/*

# Node.js (for tool execution)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml poetry.lock* ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main

# Copy application code
COPY src/ ./src/

# Non-root user
RUN groupadd -g 1000 agent && useradd -u 1000 -g agent -m -d /home/agent agent
RUN mkdir -p /workspace && chown -R agent:agent /workspace /app

ENV HOME=/home/agent
WORKDIR /workspace
USER agent

ENTRYPOINT ["python3", "-m", "src.runtime.agent_loop"]
