"""Follow-up напоминания зависшим лидам (APScheduler).

Правило: если по пользователю нет записи в `leads` и от его последнего
сообщения прошло больше `FOLLOWUP_HOURS` часов — отправляем одно напоминание
и ставим флаг `followup_sent`.

Максимум одно напоминание на «зависание»: флаг сбрасывается в
`db.upsert_user()`, то есть при следующем сообщении пользователя цикл может
повториться, но спама не будет.

Текст напоминания статический — обращения к модели здесь нет, поэтому
follow-up ничего не стоит.

Вторая задача планировщика — уборка журнала `api_calls`, по которому
считается расход запросов и токенов.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.error import Forbidden, TelegramError
from telegram.ext import Application

from . import db

logger = logging.getLogger(__name__)

FOLLOWUP_HOURS = int(os.getenv("FOLLOWUP_HOURS", "6"))
FOLLOWUP_INTERVAL_MINUTES = int(os.getenv("FOLLOWUP_INTERVAL_MINUTES", "15"))

FOLLOWUP_TEXT = "Остались вопросы? Я на связи 🙂"

_scheduler: Optional[AsyncIOScheduler] = None


async def followup_job(application: Application) -> None:
    """Один проход планировщика: находит зависших и шлёт напоминание."""
    try:
        stale = await db.get_stale_users(FOLLOWUP_HOURS)
    except Exception:  # noqa: BLE001 — job не должен падать и убивать планировщик
        logger.exception("Не удалось получить список зависших пользователей")
        return

    if not stale:
        logger.debug("Follow-up: некому напоминать")
        return

    logger.info("Follow-up: кандидатов на напоминание — %s", len(stale))

    for user in stale:
        try:
            await application.bot.send_message(chat_id=user.chat_id, text=FOLLOWUP_TEXT)
        except Forbidden:
            # Бот заблокирован — помечаем флаг, чтобы не пытаться снова.
            logger.info("Follow-up пропущен, бот заблокирован: user_id=%s", user.user_id)
            await db.mark_followup_sent(user.user_id)
            continue
        except TelegramError as exc:
            logger.error("Ошибка follow-up для user_id=%s: %s", user.user_id, exc)
            continue
        except Exception:  # noqa: BLE001
            logger.exception("Неожиданная ошибка follow-up для user_id=%s", user.user_id)
            continue

        await db.mark_followup_sent(user.user_id)
        logger.info(
            "Follow-up отправлен: user_id=%s (молчал с %s)",
            user.user_id,
            user.last_message_at,
        )


async def cleanup_job(application: Application) -> None:
    """Раз в сутки чистит журнал обращений к API от старых записей."""
    try:
        chat = application.bot_data.get("chat")
        if chat is not None:
            await chat.usage.cleanup()
    except Exception:  # noqa: BLE001
        logger.exception("Не удалось почистить журнал api_calls")


def start_scheduler(application: Application) -> AsyncIOScheduler:
    """Создаёт и запускает планировщик. Вызывается из `on_startup`."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        cleanup_job,
        trigger="interval",
        hours=24,
        args=[application],
        id="api_calls_cleanup",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        followup_job,
        trigger="interval",
        minutes=FOLLOWUP_INTERVAL_MINUTES,
        args=[application],
        id="followup",
        # Если предыдущий запуск ещё идёт или процесс «проспал» — не копим
        # очередь одинаковых задач.
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )
    scheduler.start()
    _scheduler = scheduler

    logger.info(
        "Планировщик follow-up запущен: проверка каждые %s мин, порог %s ч",
        FOLLOWUP_INTERVAL_MINUTES,
        FOLLOWUP_HOURS,
    )
    return scheduler


def shutdown_scheduler() -> None:
    """Останавливает планировщик. Вызывается из `on_shutdown`."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Планировщик follow-up остановлен")
