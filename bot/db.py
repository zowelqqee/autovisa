"""SQLite-слой: пользователи, история диалогов, лиды.

Используется стандартный `sqlite3` (без внешних зависимостей). Все блокирующие
вызовы обёрнуты в `asyncio.to_thread`, поэтому из хендлеров можно вызывать
любую функцию модуля через `await` — event loop не блокируется.

Соединение одно на процесс (`check_same_thread=False`) и защищено
`threading.Lock`, т.к. `asyncio.to_thread` выполняет работу в пуле потоков.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "autovisa.db")

_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Модели
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Message:
    """Одно сообщение диалога (таблица conversations)."""

    role: str  # "user" | "assistant"
    content: str
    created_at: str


@dataclass(slots=True)
class StaleUser:
    """Пользователь, которому пора отправить follow-up."""

    user_id: int
    chat_id: int
    first_name: str
    last_message_at: str


# --------------------------------------------------------------------------- #
# Схема
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id         INTEGER PRIMARY KEY,
    chat_id         INTEGER NOT NULL,
    username        TEXT,
    first_name      TEXT,
    created_at      TEXT NOT NULL,
    last_message_at TEXT NOT NULL,
    followup_sent   INTEGER NOT NULL DEFAULT 0,
    paused          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_conversations_user
    ON conversations (user_id, id);

-- Карточка клиента. Четыре основных поля вынесены в колонки (по ним удобно
-- искать и выгружать), остальные поля карточки лежат в `card` одним JSON —
-- их состав меняется вместе с промптом, и заводить под каждое колонку смысла нет.
CREATE TABLE IF NOT EXISTS leads (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    name       TEXT,
    contact    TEXT,
    service    TEXT,
    context    TEXT,
    card       TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_leads_user ON leads (user_id);

-- Журнал обращений к OpenAI API: сколько запросов и токенов израсходовано.
-- Лежит в БД, чтобы статистика переживала перезапуск процесса.
-- `tokens` заполняется уже после ответа модели (до запроса расход неизвестен).
CREATE TABLE IF NOT EXISTS api_calls (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    tokens     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_api_calls_created ON api_calls (created_at);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Низкоуровневые синхронные операции (выполняются в thread pool)
# --------------------------------------------------------------------------- #


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        directory = os.path.dirname(os.path.abspath(DB_PATH))
        if directory:
            os.makedirs(directory, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
    return _conn


def _execute(sql: str, params: Iterable[Any] = ()) -> None:
    with _lock:
        conn = _connect()
        conn.execute(sql, tuple(params))
        conn.commit()


def _query(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    with _lock:
        conn = _connect()
        return conn.execute(sql, tuple(params)).fetchall()


def _init_db_sync() -> None:
    with _lock:
        conn = _connect()
        conn.executescript(_SCHEMA)
        _migrate(conn)
        conn.commit()
    logger.info("База данных инициализирована: %s", os.path.abspath(DB_PATH))


def _migrate(conn: sqlite3.Connection) -> None:
    """Досоздаёт колонки, появившиеся после первого релиза.

    `CREATE TABLE IF NOT EXISTS` не меняет уже существующую таблицу, поэтому
    новые колонки добавляются здесь — иначе обновлённый бот упадёт на БД,
    созданной предыдущей версией.
    """
    api_columns = {row["name"] for row in conn.execute("PRAGMA table_info(api_calls)")}
    if api_columns and "tokens" not in api_columns:
        conn.execute("ALTER TABLE api_calls ADD COLUMN tokens INTEGER NOT NULL DEFAULT 0")
        logger.info("Миграция: в api_calls добавлена колонка tokens")

    lead_columns = {row["name"] for row in conn.execute("PRAGMA table_info(leads)")}
    if lead_columns and "card" not in lead_columns:
        conn.execute("ALTER TABLE leads ADD COLUMN card TEXT")
        logger.info("Миграция: в leads добавлена колонка card")

    user_columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
    if user_columns and "paused" not in user_columns:
        conn.execute("ALTER TABLE users ADD COLUMN paused INTEGER NOT NULL DEFAULT 0")
        logger.info("Миграция: в users добавлена колонка paused")


def _close_sync() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


# --------------------------------------------------------------------------- #
# Публичный async-API
# --------------------------------------------------------------------------- #


async def init_db() -> None:
    """Создаёт таблицы, если их ещё нет. Вызывать один раз при старте."""
    await asyncio.to_thread(_init_db_sync)


async def close_db() -> None:
    """Закрывает соединение (вызывать при graceful shutdown)."""
    await asyncio.to_thread(_close_sync)


async def upsert_user(
    user_id: int,
    chat_id: int,
    username: Optional[str],
    first_name: Optional[str],
) -> None:
    """Создаёт пользователя или обновляет метаданные и `last_message_at`.

    Любая новая активность пользователя сбрасывает флаг `followup_sent`,
    чтобы напоминание можно было отправить снова в следующем «зависании».
    """
    now = _utcnow()
    await asyncio.to_thread(
        _execute,
        """
        INSERT INTO users (user_id, chat_id, username, first_name,
                           created_at, last_message_at, followup_sent)
        VALUES (?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(user_id) DO UPDATE SET
            chat_id         = excluded.chat_id,
            username        = excluded.username,
            first_name      = excluded.first_name,
            last_message_at = excluded.last_message_at,
            followup_sent   = 0
        """,
        (user_id, chat_id, username, first_name or "", now, now),
    )


async def add_message(user_id: int, role: str, content: str) -> None:
    """Добавляет сообщение в историю диалога."""
    if role not in ("user", "assistant"):
        raise ValueError(f"Недопустимая роль: {role!r}")
    await asyncio.to_thread(
        _execute,
        "INSERT INTO conversations (user_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?)",
        (user_id, role, content, _utcnow()),
    )


async def get_history(user_id: int, limit: int = 20) -> list[Message]:
    """Возвращает последние `limit` сообщений в хронологическом порядке."""
    rows = await asyncio.to_thread(
        _query,
        "SELECT role, content, created_at FROM conversations "
        "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    )
    return [
        Message(role=r["role"], content=r["content"], created_at=r["created_at"])
        for r in reversed(rows)
    ]


async def create_lead(
    user_id: int,
    name: Optional[str],
    contact: Optional[str],
    service: Optional[str],
    context: Optional[str],
    card: Optional[dict[str, str]] = None,
) -> int:
    """Записывает карточку клиента и возвращает id записи.

    `card` — остальные поля карточки (гражданство, адрес, проблемные места
    и т.д.); складываются в одну колонку как JSON.
    """
    now = _utcnow()
    card_json = json.dumps(card, ensure_ascii=False) if card else None

    def _insert() -> int:
        with _lock:
            conn = _connect()
            cur = conn.execute(
                "INSERT INTO leads (user_id, name, contact, service, context, card, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, name, contact, service, context, card_json, now),
            )
            conn.commit()
            return int(cur.lastrowid)

    lead_id = await asyncio.to_thread(_insert)
    logger.info("Создан лид #%s для user_id=%s", lead_id, user_id)
    return lead_id


async def has_lead(user_id: int) -> bool:
    """True, если по пользователю уже есть запись в `leads`."""
    rows = await asyncio.to_thread(
        _query, "SELECT 1 FROM leads WHERE user_id = ? LIMIT 1", (user_id,)
    )
    return bool(rows)


async def get_stale_users(hours: int) -> list[StaleUser]:
    """Пользователи без лида, молчащие дольше `hours`, и без отправленного follow-up."""
    threshold = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat(timespec="seconds")
    rows = await asyncio.to_thread(
        _query,
        """
        SELECT u.user_id, u.chat_id, u.first_name, u.last_message_at
        FROM users u
        WHERE u.followup_sent = 0
          AND u.last_message_at < ?
          AND NOT EXISTS (SELECT 1 FROM leads l WHERE l.user_id = u.user_id)
        """,
        (threshold,),
    )
    return [
        StaleUser(
            user_id=r["user_id"],
            chat_id=r["chat_id"],
            first_name=r["first_name"] or "",
            last_message_at=r["last_message_at"],
        )
        for r in rows
    ]


async def mark_followup_sent(user_id: int) -> None:
    """Ставит флаг `followup_sent`, чтобы не слать напоминание повторно."""
    await asyncio.to_thread(
        _execute, "UPDATE users SET followup_sent = 1 WHERE user_id = ?", (user_id,)
    )


async def set_paused(user_id: int, value: bool) -> None:
    """Ставит/снимает паузу — пока она активна, бот не отвечает автоматически."""
    await asyncio.to_thread(
        _execute, "UPDATE users SET paused = ? WHERE user_id = ?", (int(value), user_id)
    )


async def is_paused(user_id: int) -> bool:
    """True, если для пользователя включена ручная пауза (ждём оператора)."""
    rows = await asyncio.to_thread(
        _query, "SELECT paused FROM users WHERE user_id = ?", (user_id,)
    )
    return bool(rows and rows[0]["paused"])


async def reset_conversation(user_id: int) -> None:
    """Удаляет историю диалога пользователя (для команды /reset)."""
    await asyncio.to_thread(
        _execute, "DELETE FROM conversations WHERE user_id = ?", (user_id,)
    )


# --------------------------------------------------------------------------- #
# Учёт обращений к OpenAI API (запросы и токены)
# --------------------------------------------------------------------------- #


async def record_api_call(timestamp: Optional[str] = None) -> int:
    """Фиксирует факт обращения к API и возвращает id записи.

    Записывается ДО запроса: провалившийся запрос у Google тоже, как правило,
    списывается с квоты, поэтому считать нужно попытки, а не успехи.
    Расход токенов на этот момент неизвестен — он дописывается через
    `update_api_call_tokens()` уже после ответа модели.
    """
    ts = timestamp or _utcnow()

    def _insert() -> int:
        with _lock:
            conn = _connect()
            cur = conn.execute(
                "INSERT INTO api_calls (created_at, tokens) VALUES (?, 0)", (ts,)
            )
            conn.commit()
            return int(cur.lastrowid)

    return await asyncio.to_thread(_insert)


async def update_api_call_tokens(call_id: int, tokens: int) -> None:
    """Дописывает фактический расход токенов к записи журнала."""
    await asyncio.to_thread(
        _execute,
        "UPDATE api_calls SET tokens = ? WHERE id = ?",
        (int(tokens), call_id),
    )


async def sum_api_tokens_since(since_iso: str) -> int:
    """Сумма израсходованных токенов начиная с `since_iso`."""
    rows = await asyncio.to_thread(
        _query,
        "SELECT COALESCE(SUM(tokens), 0) AS n FROM api_calls WHERE created_at >= ?",
        (since_iso,),
    )
    return int(rows[0]["n"]) if rows else 0


async def avg_api_tokens_since(since_iso: str) -> float:
    """Средний расход токенов на запрос (только по записям с ненулевым расходом).

    Используется как оценка стоимости следующего запроса: до отправки точный
    расход неизвестен, а превышать суточный лимит по токенам нельзя.
    """
    rows = await asyncio.to_thread(
        _query,
        "SELECT AVG(tokens) AS a FROM api_calls WHERE created_at >= ? AND tokens > 0",
        (since_iso,),
    )
    value = rows[0]["a"] if rows else None
    return float(value) if value else 0.0


async def count_api_calls_since(since_iso: str) -> int:
    """Количество обращений к API начиная с момента `since_iso` (UTC ISO-8601)."""
    rows = await asyncio.to_thread(
        _query,
        "SELECT COUNT(*) AS n FROM api_calls WHERE created_at >= ?",
        (since_iso,),
    )
    return int(rows[0]["n"]) if rows else 0


async def oldest_api_call_since(since_iso: str) -> Optional[str]:
    """Время самого раннего обращения начиная с `since_iso` (или None).

    Нужно, чтобы вычислить, когда освободится слот скользящего окна RPM.
    """
    rows = await asyncio.to_thread(
        _query,
        "SELECT MIN(created_at) AS ts FROM api_calls WHERE created_at >= ?",
        (since_iso,),
    )
    return rows[0]["ts"] if rows and rows[0]["ts"] else None


async def purge_old_api_calls(keep_days: int = 3) -> int:
    """Удаляет старые записи журнала. Возвращает число удалённых строк."""
    threshold = (
        datetime.now(timezone.utc) - timedelta(days=keep_days)
    ).isoformat(timespec="seconds")

    def _delete() -> int:
        with _lock:
            conn = _connect()
            cur = conn.execute("DELETE FROM api_calls WHERE created_at < ?", (threshold,))
            conn.commit()
            return cur.rowcount

    return await asyncio.to_thread(_delete)
