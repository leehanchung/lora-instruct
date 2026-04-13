"""Bot entrypoint — sets up the Discord client and registers event handlers."""

from __future__ import annotations

import asyncio
import re
import sys

import discord
import structlog

from delulu_discord.dispatcher import SandboxDispatcher
from delulu_discord.handlers import MessageHandler
from delulu_discord.session_manager import SessionManager
from delulu_discord.settings import Settings

logger = structlog.get_logger()


def _strip_mention(content: str, bot_id: int) -> str:
    """Remove the bot's own @-mention tokens from a message, leaving the prompt."""
    # Discord user mentions: <@ID> or <@!ID>
    return re.sub(rf"<@!?{bot_id}>", "", content).strip()


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
        # Ignore own messages and other bots
        if message.author == client.user or message.author.bot:
            return

        bot_user = client.user
        if bot_user is None:
            return

        bot_mentioned = bot_user in message.mentions
        channel = message.channel

        # Thread reply: auto-continue if this thread is already ours, otherwise
        # require an explicit @-mention to pull us into the conversation.
        if isinstance(channel, discord.Thread):
            owns_thread = (
                channel.owner_id == bot_user.id
                or session_manager.get_session(channel.id) is not None
            )
            if not (owns_thread or bot_mentioned):
                return
            prompt = _strip_mention(message.content, bot_user.id)
            if not prompt:
                return
            await handler.handle_thread_reply(message, prompt)
            return

        # Top-level channel message: only respond when explicitly mentioned.
        if isinstance(channel, discord.TextChannel):
            if not bot_mentioned:
                return
            prompt = _strip_mention(message.content, bot_user.id)
            if not prompt:
                return
            await handler.handle_channel_message(message, prompt)

    return client


def main() -> None:
    """Entrypoint for `delulu-discord` console script."""
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
