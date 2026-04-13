# Harness: Claude Code + MCP plugins (Discord, Slack, etc.)
#
# This is the primary harness for running Claude Code as a long-lived bot.
# Includes: claude CLI, Bun (MCP server runtime), Node.js, Python, Go.
#
# Build:
#   docker build -f harnesses/claude-code.Dockerfile -t managed-agents-claude-code:latest .

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git ca-certificates gnupg \
    build-essential pkg-config \
    python3.11 python3-pip python3-venv \
    jq vim less zip unzip \
    bash zsh netcat dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Bun (Discord MCP server runtime)
ENV BUN_INSTALL="/usr/local/bun"
RUN curl -fsSL https://bun.sh/install | BUN_INSTALL=/usr/local/bun bash
ENV PATH="/usr/local/bun/bin:$PATH"

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Pre-install Discord plugin
RUN mkdir -p /opt/claude-plugins && \
    cd /opt/claude-plugins && \
    git clone --depth 1 https://github.com/anthropics/claude-plugins-official.git && \
    cd claude-plugins-official/external_plugins/discord && \
    bun install --frozen-lockfile

# Common Python packages
RUN python3 -m pip install --no-cache-dir \
    httpx requests beautifulsoup4 pyyaml numpy pandas

# Non-root user
RUN groupadd -g 1000 agent && useradd -u 1000 -g agent -m -d /home/agent agent

# Directories
RUN mkdir -p /workspace /home/agent/.claude/channels/discord && \
    chown -R agent:agent /workspace /home/agent

ENV CLAUDE_PLUGINS_DIR="/opt/claude-plugins/claude-plugins-official/external_plugins" \
    TERM=xterm-256color \
    NODE_ENV=production \
    HOME=/home/agent

WORKDIR /workspace
USER agent

ENTRYPOINT ["claude"]
