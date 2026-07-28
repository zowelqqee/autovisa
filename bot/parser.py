"""Парсинг служебных тегов в ответах модели + защита от prompt injection.

Два направления:

1. `sanitize_user_input()` — вход. Обезвреживает в сообщении пользователя всё,
   что похоже на системные теги (`[LEAD_CAPTURED]`, `[/LEAD_CAPTURED]`,
   `[ESCALATE: ...]`), ДО записи в историю. Иначе пользователь мог бы
   «подделать» лид или эскалацию.

2. `parse_model_response()` — выход. Извлекает из ответа модели блок
   `[LEAD_CAPTURED]` и тег `[ESCALATE: ...]`, вырезает их из текста и
   возвращает чистое сообщение для отправки в Telegram.

Ожидаемый формат карточки клиента (закрывающий тег необязателен, поля —
любое подмножество `CARD_FIELDS`, модель заполняет только известное):

    [LEAD_CAPTURED]
    name: Иван
    contact: @ivan / +374 xx xxx xxx
    citizenship: РФ
    city: Ереван
    services: ИП, счёт
    problems: адрес не подтверждён, собственник за границей
    next_step: проверить договор аренды и доверенность
    [/LEAD_CAPTURED]

Эскалация:

    [ESCALATE: клиент просит юриста по нестандартному основанию]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 1. Санитайзер пользовательского ввода
# --------------------------------------------------------------------------- #

# Всё, что похоже на системный тег: [LEAD_CAPTURED], [/LEAD_CAPTURED],
# [ESCALATE], [ESCALATE: ...], а также варианты с пробелами и другим регистром.
_INJECTION_RE = re.compile(
    r"\[\s*/?\s*(?:LEAD_CAPTURED|ESCALATE)\b[^\]]*\]",
    re.IGNORECASE,
)

# Одиночные упоминания без скобок — на случай, если модель «додумает» скобки.
_BARE_TAG_RE = re.compile(r"\b(LEAD_CAPTURED|ESCALATE)\b", re.IGNORECASE)

# Невидимые символы, которыми можно замаскировать тег (zero-width, BOM).
_INVISIBLE_RE = re.compile(r"[​-‏  ﻿]")

MAX_USER_INPUT_CHARS = 4000


def sanitize_user_input(text: str, max_chars: int = MAX_USER_INPUT_CHARS) -> str:
    """Обезвреживает системные теги в сообщении пользователя.

    Теги не удаляются, а «ломаются» заменой квадратных скобок на круглые —
    так менеджер при разборе диалога видит, что клиент писал, а модель уже не
    может воспринять это как служебную разметку.
    """
    if not text:
        return ""

    cleaned = _INVISIBLE_RE.sub("", text)

    def _defuse_bracketed(match: re.Match[str]) -> str:
        inner = match.group(0)[1:-1]
        logger.warning("Обезврежен системный тег во вводе пользователя: %r", match.group(0))
        return f"({inner})"

    cleaned = _INJECTION_RE.sub(_defuse_bracketed, cleaned)
    cleaned = _BARE_TAG_RE.sub(lambda m: m.group(0).replace("_", "‑"), cleaned)

    cleaned = cleaned.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + " […сообщение обрезано]"
    return cleaned


# --------------------------------------------------------------------------- #
# 2. Парсер ответа модели
# --------------------------------------------------------------------------- #

_LEAD_BLOCK_RE = re.compile(
    r"\[LEAD_CAPTURED\](?P<body>.*?)(?:\[/LEAD_CAPTURED\]|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_ESCALATE_RE = re.compile(
    r"\[ESCALATE\s*:?\s*(?P<reason>[^\]]*)\]",
    re.IGNORECASE,
)

# Поля карточки клиента: внутреннее имя → подпись для менеджера.
# Порядок определяет вид уведомления в чате менеджеров.
CARD_FIELDS: tuple[tuple[str, str], ...] = (
    ("name", "Имя"),
    ("contact", "Контакт"),
    ("citizenship", "Гражданство"),
    ("city", "Город нахождения"),
    ("entry_date", "Дата въезда"),
    ("entry_doc", "Документ въезда"),
    ("status", "Статус пребывания"),
    ("social_number", "Социальный номер"),
    ("address", "Адрес / жильё"),
    ("rental", "Договор аренды"),
    ("owner_available", "Собственник доступен"),
    ("goal", "Цель"),
    ("service", "Нужные услуги"),
    ("urgency", "Срочность"),
    ("problems", "Проблемные места"),
    ("documents", "Документы на руках"),
    ("next_step", "Следующее действие"),
    ("context", "Контекст диалога"),
)

_CARD_KEYS = tuple(key for key, _ in CARD_FIELDS)

# Синонимы ключей внутри блока карточки: модель может назвать поле
# по-русски или чуть иначе.
_FIELD_ALIASES: dict[str, str] = {
    "name": "name", "имя": "name", "клиент": "name",
    "contact": "contact", "контакт": "contact", "контакты": "contact",
    "телефон": "contact", "phone": "contact", "telegram": "contact",
    "citizenship": "citizenship", "гражданство": "citizenship",
    "city": "city", "город": "city", "город нахождения": "city",
    "location": "city", "местонахождение": "city",
    "entry_date": "entry_date", "дата въезда": "entry_date", "въезд": "entry_date",
    "entry_doc": "entry_doc", "документ въезда": "entry_doc", "паспорт": "entry_doc",
    "status": "status", "статус": "status", "статус пребывания": "status",
    "внж": "status",
    "social_number": "social_number", "социальный номер": "social_number",
    "соцномер": "social_number", "соцкарта": "social_number",
    "address": "address", "адрес": "address", "жильё": "address", "жилье": "address",
    "rental": "rental", "договор": "rental", "договор аренды": "rental",
    "аренда": "rental",
    "owner_available": "owner_available", "собственник": "owner_available",
    "собственник доступен": "owner_available",
    "goal": "goal", "цель": "goal", "задача": "goal",
    "service": "service", "services": "service", "услуга": "service",
    "услуги": "service", "нужные услуги": "service", "запрос": "service",
    "urgency": "urgency", "срочность": "urgency",
    "problems": "problems", "проблемные места": "problems", "проблемы": "problems",
    "риски": "problems",
    "documents": "documents", "документы": "documents",
    "документы на руках": "documents",
    "next_step": "next_step", "следующее действие": "next_step",
    "следующий шаг": "next_step",
    "context": "context", "контекст": "context", "детали": "context",
    "комментарий": "context", "summary": "context",
}

_FIELD_LINE_RE = re.compile(r"^\s*[-*]?\s*(?P<key>[\wЀ-ӿ ]+?)\s*[:=]\s*(?P<value>.*)$")

# Заглушки, которыми модель иногда заполняет неизвестные поля вместо того,
# чтобы их пропустить. Такое значение равносильно пустому: иначе в карточке
# менеджера появится «Имя: не указано», а в БД — лид без реальных данных.
_PLACEHOLDER_VALUES = frozenset(
    {
        "", "-", "—", "–", "n/a", "na", "null", "none", "нет", "нет данных",
        "не указано", "не указан", "не указана", "неизвестно", "не сообщил",
        "не сообщила", "не задан", "уточняется", "?", "не известно",
    }
)


def _clean_value(value: str) -> Optional[str]:
    """Нормализует значение поля карточки: заглушка → None."""
    value = value.strip().strip(".").strip()
    return None if value.casefold() in _PLACEHOLDER_VALUES else (value or None)


@dataclass(slots=True)
class Lead:
    """Карточка клиента из блока [LEAD_CAPTURED].

    Все поля опциональны: модель заполняет только то, что реально узнала из
    диалога. Пустые поля в уведомление менеджеру не попадают.
    """

    name: Optional[str] = None
    contact: Optional[str] = None
    citizenship: Optional[str] = None
    city: Optional[str] = None
    entry_date: Optional[str] = None
    entry_doc: Optional[str] = None
    status: Optional[str] = None
    social_number: Optional[str] = None
    address: Optional[str] = None
    rental: Optional[str] = None
    owner_available: Optional[str] = None
    goal: Optional[str] = None
    service: Optional[str] = None
    urgency: Optional[str] = None
    problems: Optional[str] = None
    documents: Optional[str] = None
    next_step: Optional[str] = None
    context: Optional[str] = None

    def is_empty(self) -> bool:
        return not any(getattr(self, key) for key in _CARD_KEYS)

    def filled(self) -> list[tuple[str, str]]:
        """Заполненные поля в порядке карточки: [(подпись, значение)]."""
        result = []
        for key, label in CARD_FIELDS:
            value = getattr(self, key)
            if value:
                result.append((label, value))
        return result

    def extras(self) -> dict[str, str]:
        """Поля сверх четырёх основных — в БД идут одним JSON-полем."""
        primary = ("name", "contact", "service", "context")
        return {
            key: getattr(self, key)
            for key in _CARD_KEYS
            if key not in primary and getattr(self, key)
        }


@dataclass(slots=True)
class ParsedResponse:
    """Результат разбора ответа модели."""

    text: str  # Очищенный текст для пользователя
    lead: Optional[Lead] = None  # Заполнен, если был [LEAD_CAPTURED]
    escalation: Optional[str] = None  # Причина, если был [ESCALATE: ...]

    @property
    def has_lead(self) -> bool:
        return self.lead is not None

    @property
    def has_escalation(self) -> bool:
        return self.escalation is not None


def _parse_lead_body(body: str) -> Lead:
    """Разбирает тело блока лида построчно в формате `ключ: значение`."""
    lead = Lead()
    current_key: Optional[str] = None

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _FIELD_LINE_RE.match(line)
        if match:
            key = _FIELD_ALIASES.get(match.group("key").strip().casefold())
            value = match.group("value").strip()
            if key:
                current_key = key
                setattr(lead, key, _clean_value(value))
                continue

        # Продолжение многострочного значения (обычно context).
        if current_key:
            previous = getattr(lead, current_key) or ""
            setattr(lead, current_key, f"{previous} {line}".strip())

    return lead


def parse_model_response(raw: str) -> ParsedResponse:
    """Извлекает служебные теги и возвращает очищенный текст.

    Теги всегда вырезаются из текста, даже если разобрать содержимое
    не удалось — пользователь не должен видеть служебную разметку.
    """
    if not raw:
        return ParsedResponse(text="")

    text = raw
    lead: Optional[Lead] = None
    escalation: Optional[str] = None

    lead_match = _LEAD_BLOCK_RE.search(text)
    if lead_match:
        parsed = _parse_lead_body(lead_match.group("body"))
        if parsed.is_empty():
            logger.warning("Блок [LEAD_CAPTURED] найден, но полей в нём нет")
        lead = parsed
        text = text[: lead_match.start()] + text[lead_match.end() :]

    escalate_match = _ESCALATE_RE.search(text)
    if escalate_match:
        reason = escalate_match.group("reason").strip()
        escalation = reason or "причина не указана"
        text = text[: escalate_match.start()] + text[escalate_match.end() :]

    # Подчищаем возможные висячие закрывающие теги и лишние пустые строки.
    text = re.sub(r"\[/?\s*(?:LEAD_CAPTURED|ESCALATE)[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = markdown_to_telegram_html(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return ParsedResponse(text=text, lead=lead, escalation=escalation)


# Сообщения клиенту уходят с parse_mode=HTML (см. bot/handlers.py), поэтому
# markdown модели конвертируем в теги Telegram, а не вырезаем. Одиночное
# подчёркивание не трогаем — оно встречается в телеграм-никах вида
# @ivan_petrov.
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
_MD_ITALIC_RE = re.compile(r"\*(?!\s)([^*\n]+?)(?<!\s)\*")
_MD_CODE_RE = re.compile(r"`{1,3}([^`]+)`{1,3}", re.DOTALL)
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^\s{0,3}\*\s+", re.MULTILINE)


def _escape_html(text: str) -> str:
    """Экранирование для parse_mode=HTML — в отличие от `_esc()`, не подменяет
    пустую строку на «—»: здесь это тело сообщения, а не поле карточки."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def markdown_to_telegram_html(text: str) -> str:
    """Конвертирует markdown модели в HTML-теги, которые понимает Telegram.

    Сначала экранируются HTML-спецсимволы (иначе `<`/`>`/`&` в тексте клиента
    сломали бы parse_mode=HTML), и только потом markdown-маркеры превращаются
    в теги — экранирование их не затрагивает, это разные символы.
    """
    text = _escape_html(text)
    text = _MD_BOLD_RE.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", text)
    text = _MD_BULLET_RE.sub("• ", text)
    text = _MD_ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _MD_CODE_RE.sub(r"\1", text)
    text = _MD_HEADING_RE.sub("", text)
    return text


# --------------------------------------------------------------------------- #
# Форматирование уведомлений менеджеру
# --------------------------------------------------------------------------- #


def format_lead_notification(
    lead: Lead,
    *,
    user_id: int,
    username: Optional[str],
    first_name: Optional[str],
) -> str:
    """Карточка клиента для группового чата менеджеров.

    Печатаются только заполненные поля — пустые строки в карточке лишь
    мешают читать.
    """
    handle = f"@{username}" if username else "—"
    lines = ["🟢 <b>Карточка клиента</b>", ""]
    lines.extend(f"<b>{label}:</b> {_esc(value)}" for label, value in lead.filled())
    lines.append("")
    lines.append(f"<b>Telegram:</b> {_esc(first_name)} ({_esc(handle)})")
    lines.append(f"<b>user_id:</b> <code>{user_id}</code>")
    return "\n".join(lines)


def format_escalation_notification(
    reason: str,
    history: list[tuple[str, str]],
    *,
    user_id: int,
    username: Optional[str],
    first_name: Optional[str],
) -> str:
    """Текст уведомления об эскалации + последние сообщения диалога.

    `history` — список пар (role, content) в хронологическом порядке.
    """
    handle = f"@{username}" if username else "—"
    lines = [
        "🔴 <b>Эскалация — нужен менеджер</b>\n",
        f"<b>Причина:</b> {_esc(reason)}\n",
        f"<b>Telegram:</b> {_esc(first_name)} ({_esc(handle)})",
        f"<b>user_id:</b> <code>{user_id}</code>\n",
        "<b>Последние сообщения:</b>",
    ]
    if not history:
        lines.append("<i>история пуста</i>")
    for role, content in history:
        who = "👤 Клиент" if role == "user" else "🤖 Бот"
        snippet = content if len(content) <= 400 else content[:400] + "…"
        lines.append(f"{who}: {_esc(snippet)}")
    return "\n".join(lines)


def _esc(value: Optional[str]) -> str:
    """Экранирование для Telegram parse_mode=HTML."""
    if not value:
        return "—"
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
