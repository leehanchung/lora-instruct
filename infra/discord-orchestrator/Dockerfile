# Dockerfile for the always-on Discord bot process.
# This does NOT run Claude Code — it only hosts the bot that dispatches to Modal.

FROM python:3.14-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv

# Copy project files
COPY pyproject.toml .python-version ./
RUN uv pip install --system .

COPY src/ src/

# The bot is lightweight — ~50MB RAM
CMD ["python", "-m", "src.bot.main"]
