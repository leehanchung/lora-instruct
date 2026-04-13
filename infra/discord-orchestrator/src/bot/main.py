"""Bot entrypoint — sets up the Discord client and registers event handlers."""

from __future__ import annotations

import asyncio
import sys

import discord
import structlog

from src.bot.handlers import MessageHandler
from src.bot.session_manager import SessionManager
from src.config import Settings
from src.modal_dispatch.sandbox import SandboxDispatcher

logger = structlog.get_logger()


def create_bot(settings: Settings) -> discord.Client:
    """Create and configure the Discord client."""
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True

    client = discord.Client(intents=intents)

    # ── Wire up components ───────────────────────────────────
    session_manager = SessionManager(ttl_seconds=settings.session_ttl_seconds)
    dispatcher = SandboxDispatcher(settings=settings)
    handler = MessageHandler(
        settings=settings,
        session_manager=session_manager,
        dispatcher=dispatcher,
    )

    # ── Event handlers ───────────────────────────────────────
    @client.event
    async def on_ready():
        logger.info("bot.ready", user=str(client.user), guilds=len(client.guilds))

    @client.event
    async def on_message(message: discord.Message):
        # Ignore own messages
        if message.author == client.user:
            return

        # Ignore bots
        if message.author.bot:
            return

        channel = message.channel

        # Reply inside an existing thread → handle as continuation
        if isinstance(channel, discord.Thread):
            await handler.handle_thread_reply(message)
            return

        # New message in a monitored channel → create thread + dispatch
        if isinstance(channel, discord.TextChannel):
            if channel.name.startswith(settings.discord_channel_prefix):
                await handler.handle_channel_message(message)

    return client


def main() -> None:
    """Entrypoint for `bot` console script."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
    )

    try:
        settings = Settings()  # type: ignore[call-arg]
    except Exception as e:
        logger.error("config.invalid", error=str(e))
        sys.exit(1)

    client = create_bot(settings)
    asyncio.run(client.start(settings.discord_bot_token))


if __name__ == "__main__":
    main()
