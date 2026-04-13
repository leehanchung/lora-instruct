# Harness: SWE-agent (Princeton)
#
# Autonomous software engineering agent that resolves GitHub issues.
# Runs with gVisor (runsc) by default — stronger isolation because
# it executes arbitrary code from unknown repositories.
#
# Build:
#   docker build -f harnesses/swe-agent.Dockerfile -t managed-agents-swe-agent:latest .

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git ca-certificates \
    build-essential \
    jq vim less \
    && rm -rf /var/lib/apt/lists/*

# Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install SWE-agent
RUN pip install --no-cache-dir swe-agent

# Non-root user
RUN groupadd -g 1000 agent && useradd -u 1000 -g agent -m -d /home/agent agent
RUN mkdir -p /workspace && chown -R agent:agent /workspace

ENV HOME=/home/agent
WORKDIR /workspace
USER agent

ENTRYPOINT ["python", "-m", "swe_agent.run"]
