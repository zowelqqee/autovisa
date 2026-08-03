"""Точка входа: конфигурация, регистрация хендлеров, запуск polling.

Запуск:
    python -m bot.main

Режим — polling (getUpdates). Webhook сознательно не используется: для одного
VPS без домена и SSL это лишняя инфраструктура.
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from telegram import BotCommand, BotCommandScopeChat
from telegram.ext import (
    AIORateLimiter,
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from . import db
from .followup import shutdown_scheduler, start_scheduler
from .google_sheets import GoogleSheetsCRM
from .handlers import (
    error_handler,
    help_command,
    human_command,
    message_handler,
    non_text_handler,
    reset_command,
    resume_command,
    start_command,
    stats_command,
    voice_handler,
)
from .openai_client import OpenAIChatClient
from .voice import VoiceClient

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Стандартный logging: формат, уровень, приглушение болтливых библиотек."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        level=getattr(logging, level, logging.INFO),
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext.Application").setLevel(logging.INFO)
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)


def require_env(name: str) -> str:
    """Читает обязательную переменную окружения или завершает процесс."""
    value = os.getenv(name)
    if not value:
        logger.critical("Не задана обязательная переменная окружения %s (см. .env.example)", name)
        raise SystemExit(1)
    return value


async def on_startup(application: Application) -> None:
    """Инициализация БД, клиентов OpenAI и планировщика follow-up."""
    await db.init_db()

    # Интеграция необязательна: при отсутствии service-account настроек бот
    # продолжает работать как раньше, а подробная инструкция есть в README.
    application.bot_data["crm"] = GoogleSheetsCRM.from_env()

    chat = OpenAIChatClient()
    application.bot_data["chat"] = chat
    stats = await chat.usage.usage()
    logger.info(
        "OpenAI готов: model=%s effort=%s. За сутки уже израсходовано: "
        "%s запросов, %s токенов",
        chat.model,
        chat.effort,
        stats.day_calls,
        stats.day_tokens,
    )

    # Голосовой режим переиспользует HTTP-клиент текстового: те же ключ,
    # прокси и пул соединений.
    if os.getenv("VOICE_ENABLED", "1") == "1":
        voice = VoiceClient(chat.client)
        application.bot_data["voice"] = voice
        logger.info(
            "Голосовой режим готов: %s -> %s (голос %s)",
            voice.stt_model, voice.tts_model, voice.voice,
        )
    else:
        logger.info("Голосовой режим выключен через VOICE_ENABLED=0")

    start_scheduler(application)

    client_commands = [
        BotCommand("start", "начать заново"),
        BotCommand("help", "справка и список услуг"),
        BotCommand("reset", "очистить историю диалога"),
        BotCommand("human", "позвать оператора-человека"),
    ]
    await application.bot.set_my_commands(client_commands)

    manager_chat_id = application.bot_data.get("manager_chat_id")
    if manager_chat_id:
        await application.bot.set_my_commands(
            client_commands + [
                BotCommand("resume", "снять паузу: /resume <user_id>"),
                BotCommand("stats", "расход запросов и токенов"),
            ],
            scope=BotCommandScopeChat(chat_id=manager_chat_id),
        )

    me = await application.bot.get_me()
    logger.info("Бот запущен: @%s (id=%s)", me.username, me.id)


async def on_shutdown(application: Application) -> None:
    """Graceful shutdown: планировщик, HTTP-клиент, соединение с БД."""
    shutdown_scheduler()
    chat: OpenAIChatClient | None = application.bot_data.get("chat")
    if chat is not None:
        # Голосовой клиент использует этот же HTTP-клиент — закрывать отдельно
        # не нужно.
        await chat.aclose()
    await db.close_db()
    logger.info("Бот остановлен")


def build_application() -> Application:
    """Собирает `Application` с хендлерами и (опционально) прокси."""
    token = require_env("TELEGRAM_BOT_TOKEN")
    require_env("OPENAI_API_KEY")
    manager_chat_id = int(require_env("MANAGER_CHAT_ID"))

    builder = ApplicationBuilder().token(token)
    builder.rate_limiter(AIORateLimiter())
    builder.post_init(on_startup)
    builder.post_shutdown(on_shutdown)

    # Прокси для Telegram API — только если явно задан TELEGRAM_PROXY_URL.
    proxy_url = os.getenv("TELEGRAM_PROXY_URL") or None
    if proxy_url:
        logger.info("Telegram API через прокси")
        builder.request(HTTPXRequest(proxy=proxy_url, connection_pool_size=8))
        builder.get_updates_request(HTTPXRequest(proxy=proxy_url, connection_pool_size=1))

    application = builder.build()
    application.bot_data["manager_chat_id"] = manager_chat_id

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("human", human_command))
    application.add_handler(CommandHandler("resume", resume_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE, message_handler)
    )
    application.add_handler(
        MessageHandler(filters.VOICE & filters.ChatType.PRIVATE, voice_handler)
    )
    application.add_handler(
        MessageHandler(
            (filters.PHOTO | filters.Document.ALL | filters.VIDEO | filters.AUDIO)
            & filters.ChatType.PRIVATE,
            non_text_handler,
        )
    )
    application.add_error_handler(error_handler)

    return application


def main() -> None:
    load_dotenv()
    configure_logging()

    application = build_application()
    # drop_pending_updates=True — после рестарта не отвечаем на «протухшие»
    # сообщения, накопившиеся, пока бот лежал.
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
