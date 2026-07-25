"""Учёт обращений к OpenAI API: запросы и токены.

Задача модуля — видимость расхода, а не ограничение: на текущем тарифе
жёстких квот нет. Счётчики нужны, чтобы команда `/stats` показывала
менеджерам реальное потребление, а в логах было видно стоимость диалогов.

Данные лежат в таблице `api_calls` в SQLite, поэтому перезапуск сервиса
не обнуляет статистику.

Опциональные лимиты (`OPENAI_RPM`, `OPENAI_RPD`, `OPENAI_TPD`) **по умолчанию
выключены** — значение 0 означает «без ограничения». Они пригодятся, если
понадобится подстраховаться от неожиданного счёта или перейти на тариф
с квотами; тогда запрос отсекается до отправки, а не ловится по 429.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import db

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover — Python < 3.9
    ZoneInfo = None  # type: ignore[assignment]

DEFAULT_QUOTA_TZ = "UTC"


class QuotaExceeded(RuntimeError):
    """Сработал опциональный лимит — запрос к API не отправлялся.

    Attributes:
        scope: "minute" | "day" | "tokens".
        retry_after: через сколько секунд можно повторить (None — неизвестно).
        used / limit: текущее потребление, для сообщений и логов.
    """

    def __init__(
        self,
        scope: str,
        *,
        retry_after: Optional[float] = None,
        used: int = 0,
        limit: int = 0,
    ) -> None:
        self.scope = scope
        self.retry_after = retry_after
        self.used = used
        self.limit = limit
        super().__init__(f"Лимит {scope} исчерпан: {used}/{limit}")


@dataclass(slots=True)
class Usage:
    """Снимок потребления за минуту и за сутки."""

    minute_calls: int
    day_calls: int
    day_tokens: int
    minute_limit: int  # 0 — без ограничения
    day_limit: int
    token_limit: int
    day_resets_at: datetime

    @property
    def unlimited(self) -> bool:
        return not (self.minute_limit or self.day_limit or self.token_limit)


class UsageTracker:
    """Журнал обращений к API + опциональные лимиты."""

    def __init__(
        self,
        *,
        rpm: Optional[int] = None,
        rpd: Optional[int] = None,
        tpd: Optional[int] = None,
        quota_timezone: Optional[str] = None,
        max_wait_seconds: float = 65.0,
    ) -> None:
        # 0 = без ограничения (значение по умолчанию).
        self.rpm = int(rpm if rpm is not None else os.getenv("OPENAI_RPM", "0"))
        self.rpd = int(rpd if rpd is not None else os.getenv("OPENAI_RPD", "0"))
        self.tpd = int(tpd if tpd is not None else os.getenv("OPENAI_TPD", "0"))
        self.max_wait_seconds = max_wait_seconds
        self._tz = _load_timezone(quota_timezone or os.getenv("OPENAI_QUOTA_TZ", DEFAULT_QUOTA_TZ))
        self._lock = asyncio.Lock()

        if self.rpm or self.rpd or self.tpd:
            logger.info(
                "Лимиты OpenAI: %s RPM, %s RPD, %s токенов/сутки (0 — без ограничения)",
                self.rpm, self.rpd, self.tpd,
            )

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #

    async def begin_call(self) -> int:
        """Регистрирует начало запроса, возвращает id записи журнала.

        Если заданы лимиты — проверяет их и может подождать освобождения
        минутного слота.

        Raises:
            QuotaExceeded: лимит исчерпан и ждать бесполезно.
        """
        async with self._lock:
            if self.rpd or self.tpd:
                await self._check_daily()
            if self.rpm:
                await self._wait_for_minute_slot()
            return await db.record_api_call()

    async def finish_call(self, call_id: int, tokens: int) -> None:
        """Дописывает фактический расход токенов после ответа модели."""
        if tokens > 0:
            await db.update_api_call_tokens(call_id, tokens)

    async def usage(self) -> Usage:
        """Текущее потребление — для `/stats`, логов и диагностики."""
        minute_calls = await db.count_api_calls_since(_iso(_utcnow() - timedelta(seconds=60)))
        day_start = _iso(self._day_start())
        return Usage(
            minute_calls=minute_calls,
            day_calls=await db.count_api_calls_since(day_start),
            day_tokens=await db.sum_api_tokens_since(day_start),
            minute_limit=self.rpm,
            day_limit=self.rpd,
            token_limit=self.tpd,
            day_resets_at=self._next_day_start(),
        )

    async def cleanup(self, keep_days: int = 30) -> None:
        """Подчищает старые записи журнала (вызывается по расписанию)."""
        removed = await db.purge_old_api_calls(keep_days)
        if removed:
            logger.debug("Журнал api_calls: удалено %s старых записей", removed)

    # ------------------------------------------------------------------ #
    # Внутреннее (работает, только если лимиты заданы)
    # ------------------------------------------------------------------ #

    async def _check_daily(self) -> None:
        day_start = _iso(self._day_start())
        reset_at = self._next_day_start()
        retry_after = max(0.0, (reset_at - _utcnow()).total_seconds())

        if self.rpd:
            used = await db.count_api_calls_since(day_start)
            if used >= self.rpd:
                logger.warning("Суточный лимит запросов исчерпан: %s/%s", used, self.rpd)
                raise QuotaExceeded("day", retry_after=retry_after, used=used, limit=self.rpd)

        if self.tpd:
            spent = await db.sum_api_tokens_since(day_start)
            # Оценка стоимости следующего запроса: до отправки точный расход
            # неизвестен, поэтому берём средний по уже сделанным запросам.
            estimate = int(await db.avg_api_tokens_since(day_start)) or 2000
            if spent + estimate > self.tpd:
                logger.warning(
                    "Суточный лимит токенов исчерпан: %s + ~%s > %s", spent, estimate, self.tpd
                )
                raise QuotaExceeded(
                    "tokens", retry_after=retry_after, used=spent, limit=self.tpd
                )

    async def _wait_for_minute_slot(self) -> None:
        """Ждёт освобождения слота в минутном окне (если задан RPM)."""
        window_start = _iso(_utcnow() - timedelta(seconds=60))
        used = await db.count_api_calls_since(window_start)
        if used < self.rpm:
            return

        oldest = await db.oldest_api_call_since(window_start)
        if oldest is None:  # гонка: окно опустело между запросами
            return

        free_at = _parse(oldest) + timedelta(seconds=60)
        delay = max(0.0, (free_at - _utcnow()).total_seconds()) + 0.5

        if delay > self.max_wait_seconds:
            raise QuotaExceeded("minute", retry_after=delay, used=used, limit=self.rpm)

        logger.info("Минутный лимит (%s/%s), жду %.1fs", used, self.rpm, delay)
        await asyncio.sleep(delay)

    # -- границы суток -------------------------------------------------- #

    def _day_start(self) -> datetime:
        local_now = _utcnow().astimezone(self._tz)
        midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.astimezone(timezone.utc)

    def _next_day_start(self) -> datetime:
        local_now = _utcnow().astimezone(self._tz)
        midnight = (local_now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return midnight.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #


def _load_timezone(name: str):
    """Загружает часовой пояс; при отсутствии tzdata откатывается на UTC."""
    if ZoneInfo is None or name.upper() == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 — нет базы tzdata в системе
        logger.warning("Часовой пояс %r недоступен, считаю сутки по UTC", name)
        return timezone.utc


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
