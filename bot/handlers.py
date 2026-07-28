"""Обработчики Telegram: команды, текстовые и голосовые сообщения, ошибки.

Два режима работают с общей историей в `conversations`:

- **текст** → `OpenAIChatClient` (Responses API);
- **голос** → распознавание (STT) → та же чат-модель → синтез (TTS).

Голос проходит через ту же текстовую модель, поэтому теги [LEAD_CAPTURED]
вырезаются ДО озвучки: клиент их не слышит, а лид фиксируется. В историю
пишутся расшифровки речи, поэтому текст и голос делят один контекст.

Поток обработки текстового сообщения:

    сообщение
      → sanitize_user_input()        (защита от prompt injection)
      → db.add_message(role="user")
      → db.get_history(limit)
      → OpenAIChatClient.complete_or_fallback()
      → parse_model_response()      (вырезаем [LEAD_CAPTURED] / [ESCALATE])
      → db.create_lead() + уведомление менеджеру  (если был лид)
      → уведомление менеджеру с последними 5 сообщениями (если была эскалация)
      → db.add_message(role="assistant") + отправка чистого текста пользователю

Весь ввод-вывод асинхронный: обращения к SQLite обёрнуты в `asyncio.to_thread`
внутри `bot/db.py`, HTTP-запросы — через AsyncOpenAI и PTB.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import Forbidden, TelegramError
from telegram.ext import ContextTypes

from . import db
from .knowledge_base import COMPANY, services_summary
from .openai_client import OpenAIChatClient
from .parser import (
    ParsedResponse,
    format_escalation_notification,
    format_lead_notification,
    parse_model_response,
    sanitize_user_input,
)
from .usage import QuotaExceeded
from .voice import VoiceClient, VoiceError

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE = 4096
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "20"))
ESCALATION_HISTORY_SIZE = 5

# Сообщения при срабатывании опциональных лимитов (по умолчанию выключены).
# Формулировки не упоминают лимиты и модель: для клиента это просто
# занятость, а не техническая деталь.
QUOTA_MINUTE_REPLY = (
    "Сейчас отвечаю сразу нескольким людям — напишите, пожалуйста, то же "
    "самое через минуту, и я вернусь к вам."
)
QUOTA_DAY_REPLY = (
    "На сегодня я уже не успею ответить сама. Оставьте, пожалуйста, имя и "
    "контакт (телефон или @username) — менеджер свяжется с вами в ближайший "
    "рабочий день."
)

# Голосовой режим выключен через VOICE_ENABLED=0.
VOICE_UNAVAILABLE_REPLY = (
    "Голосовые пока не слушаю — напишите текстом, отвечу сразу же."
)
VOICE_FAILED_REPLY = (
    "Не расслышала — попробуйте, пожалуйста, ещё раз голосом или просто "
    "напишите текстом."
)

# Блокировка на пользователя: не даём обрабатывать два его сообщения
# одновременно, иначе история диалога уйдёт в API в перепутанном порядке.
_user_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

WELCOME = (
    "Здравствуйте! Я — {name}, помогаю с переездом и документами в Армении: "
    "ВНЖ, прописка, соцкарта, счета, гражданство, визы.\n\n"
    "Расскажите вашу ситуацию — вместе разберёмся, что делать и в какие сроки."
)

HELP = (
    "<b>Что я могу</b>\n\n"
    "Отвечаю на вопросы по релокации в Армению и помогаю подобрать услугу.\n\n"
    "<b>Услуги и сроки:</b>\n{services}\n\n"
    "<b>Команды:</b>\n"
    "/start — начать заново\n"
    "/help — эта справка\n"
    "/reset — очистить историю диалога\n"
    "/human — позвать оператора-человека\n\n"
    "Просто напишите вопрос своими словами — этого достаточно."
)


# --------------------------------------------------------------------------- #
# Команды
# --------------------------------------------------------------------------- #


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start — приветствие и регистрация пользователя."""
    user, chat = update.effective_user, update.effective_chat
    if not user or not chat:
        return

    await db.upsert_user(user.id, chat.id, user.username, user.first_name)
    logger.info("Старт диалога: user_id=%s username=%s", user.id, user.username)
    await update.message.reply_text(WELCOME.format(name=COMPANY["name"]))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help — краткая справка со списком услуг."""
    await update.message.reply_text(
        HELP.format(services=services_summary()), parse_mode=ParseMode.HTML
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/reset — очистка истории диалога (лиды сохраняются)."""
    user = update.effective_user
    if not user:
        return
    await db.reset_conversation(user.id)
    logger.info("История очищена: user_id=%s", user.id)
    await update.message.reply_text("Хорошо, начинаем с чистого листа. Расскажите, что у вас за ситуация?")


async def human_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/human — клиент сам зовёт оператора. Ставит бота на паузу для этого чата."""
    message = update.message
    user, chat = update.effective_user, update.effective_chat
    if not message or not user or not chat:
        return

    await db.upsert_user(user.id, chat.id, user.username, user.first_name)
    await db.set_paused(user.id, True)
    logger.info("Клиент вызвал оператора (/human): user_id=%s", user.id)

    recent = await db.get_history(user.id, limit=ESCALATION_HISTORY_SIZE)
    await _notify_manager(
        context,
        format_escalation_notification(
            "Клиент нажал /human — просит оператора",
            [(m.role, m.content) for m in recent],
            user_id=user.id,
            username=user.username,
            first_name=user.first_name,
        ),
    )

    await message.reply_text(
        "Передала ваш запрос менеджеру, он подключится в ближайшее время. "
        "Можете написать что-то ещё — я всё передам."
    )


async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/resume <user_id> — снимает паузу после /human. Только в чате менеджеров."""
    chat = update.effective_chat
    message = update.message
    if not chat or not message or chat.id != context.application.bot_data.get("manager_chat_id"):
        return

    if not context.args:
        await message.reply_text("Укажите user_id: /resume 123456789 (он есть в карточке выше).")
        return

    try:
        target_id = int(context.args[0])
    except ValueError:
        await message.reply_text("user_id должен быть числом — возьмите его из карточки клиента.")
        return

    await db.set_paused(target_id, False)
    logger.info("Пауза снята менеджером: user_id=%s", target_id)
    await message.reply_text(f"Готово, бот снова отвечает пользователю {target_id}.")


# --------------------------------------------------------------------------- #
# Основной обработчик сообщений
# --------------------------------------------------------------------------- #


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/stats — расход запросов и токенов. Работает только в чате менеджеров."""
    chat = update.effective_chat
    if not chat or chat.id != context.application.bot_data.get("manager_chat_id"):
        return

    client: OpenAIChatClient = context.application.bot_data["chat"]
    u = await client.usage.usage()
    voice_on = "включён" if context.application.bot_data.get("voice") else "выключен"
    limits = "без ограничений" if u.unlimited else (
        f"лимиты: {u.minute_limit or '∞'} RPM / {u.day_limit or '∞'} RPD / {u.token_limit or '∞'} токенов"
    )
    await update.message.reply_text(
        f"Модель: {client.model} (effort={client.effort})\n"
        f"Голосовой режим: {voice_on}\n\n"
        f"За минуту: {u.minute_calls} запросов\n"
        f"За сутки: {u.day_calls} запросов, {u.day_tokens:,} токенов\n"
        f"{limits}".replace(",", " ")
    )


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает любое текстовое сообщение от пользователя."""
    message = update.message
    user, chat = update.effective_user, update.effective_chat
    if not message or not message.text or not user or not chat:
        return

    text = sanitize_user_input(message.text)
    if not text:
        await message.reply_text("Не удалось разобрать сообщение. Напишите, пожалуйста, текстом.")
        return

    async with _user_locks[user.id]:
        await db.upsert_user(user.id, chat.id, user.username, user.first_name)
        await db.add_message(user.id, "user", text)

        if await db.is_paused(user.id):
            # Клиент уже вызвал оператора (/human) — сообщение сохранено в
            # историю для контекста, но модель не дёргаем: ждём /resume.
            return

        history = await db.get_history(user.id, limit=HISTORY_LIMIT)

        chat_client: OpenAIChatClient = context.application.bot_data["chat"]
        typing = asyncio.create_task(_keep_typing(context, chat.id))
        try:
            raw, ok = await chat_client.complete_or_fallback(history)
        except QuotaExceeded as exc:
            await _handle_quota_exceeded(context, chat.id, user, exc)
            return
        finally:
            typing.cancel()

        if not ok:
            # Fallback не пишем в историю: это не реплика модели, и она
            # исказила бы контекст следующего запроса.
            await _send_long(context, chat.id, raw)
            return

        parsed = parse_model_response(raw)
        await _handle_tags(context, user, parsed)

        reply = parsed.text.strip()
        if not reply:
            # Модель вернула только служебные теги — подстраховываемся,
            # чтобы пользователь не остался без ответа.
            reply = "Записал(а). Чем ещё помочь по вашей ситуации?"
            logger.warning("Ответ модели состоял только из тегов, user_id=%s", user.id)

        await db.add_message(user.id, "assistant", reply)
        await _send_long(context, chat.id, reply)


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Голосовое сообщение: распознаём → отвечает чат-модель → озвучиваем.

    Голос идёт через ту же модель и тот же промпт, что и текст, поэтому
    служебные теги вырезаются ДО синтеза речи: клиент их не слышит, а лид
    фиксируется точно так же, как в переписке.
    """
    message = update.message
    user, chat = update.effective_user, update.effective_chat
    if not message or not message.voice or not user or not chat:
        return

    voice: VoiceClient | None = context.application.bot_data.get("voice")
    if voice is None:
        await message.reply_text(VOICE_UNAVAILABLE_REPLY)
        return

    async with _user_locks[user.id]:
        await db.upsert_user(user.id, chat.id, user.username, user.first_name)

        indicator = asyncio.create_task(_keep_typing(context, chat.id, ChatAction.TYPING))
        try:
            voice_file = await message.voice.get_file()
            ogg = bytes(await voice_file.download_as_bytearray())
            user_text = sanitize_user_input(await voice.transcribe(ogg))
        except VoiceError as exc:
            indicator.cancel()
            logger.error("Не удалось распознать голосовое user_id=%s: %s", user.id, exc)
            await message.reply_text(VOICE_FAILED_REPLY)
            return
        except TelegramError as exc:
            indicator.cancel()
            logger.error("Не удалось скачать голосовое user_id=%s: %s", user.id, exc)
            await message.reply_text(VOICE_FAILED_REPLY)
            return

        if not user_text:
            indicator.cancel()
            await message.reply_text(VOICE_FAILED_REPLY)
            return

        await db.add_message(user.id, "user", user_text)

        if await db.is_paused(user.id):
            # См. message_handler — клиент уже вызвал оператора через /human.
            indicator.cancel()
            return

        history = await db.get_history(user.id, limit=HISTORY_LIMIT)

        chat_client: OpenAIChatClient = context.application.bot_data["chat"]
        try:
            raw, ok = await chat_client.complete_or_fallback(history)
        except QuotaExceeded as exc:
            indicator.cancel()
            await _handle_quota_exceeded(context, chat.id, user, exc)
            return
        finally:
            indicator.cancel()

        if not ok:
            await _send_long(context, chat.id, raw)
            return

        parsed = parse_model_response(raw)
        await _handle_tags(context, user, parsed)

        reply = parsed.text.strip() or "Записал(а). Чем ещё помочь по вашей ситуации?"
        await db.add_message(user.id, "assistant", reply)

        # Озвучиваем уже очищенный от тегов текст.
        recording = asyncio.create_task(
            _keep_typing(context, chat.id, ChatAction.RECORD_VOICE)
        )
        try:
            reply_voice = await voice.synthesize(reply)
        except VoiceError as exc:
            logger.error("Синтез речи не удался: %s — отвечаю текстом", exc)
            await _send_long(context, chat.id, reply)
            return
        finally:
            recording.cancel()

        try:
            await context.bot.send_voice(
                chat_id=chat.id,
                voice=reply_voice,
                # Расшифровка в подписи: удобно читать в шумном месте и видно,
                # что именно бот ответил.
                caption=reply[:1000],
            )
        except Forbidden:
            logger.info("Пользователь заблокировал бота: chat_id=%s", chat.id)
        except TelegramError as exc:
            logger.error("Не удалось отправить голосовой ответ: %s", exc)
            await _send_long(context, chat.id, reply)


async def non_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Фото, документы, видео — просим написать текстом."""
    if update.message:
        await update.message.reply_text(
            "Файлы пока не открываю, только текст и голос — опишите в двух "
            "словах, что у вас за вопрос, а документы передадим менеджеру, "
            "когда понадобится."
        )


# --------------------------------------------------------------------------- #
# Обработчик ошибок
# --------------------------------------------------------------------------- #


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Логирует необработанные исключения и извиняется перед пользователем."""
    logger.exception("Необработанная ошибка при обработке апдейта", exc_info=context.error)

    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Что-то не задалось с моей стороны — повторите, пожалуйста, ещё раз, сейчас отвечу.",
            )
        except TelegramError:
            logger.warning("Не удалось отправить сообщение об ошибке пользователю")


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #


async def _notify_manager(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    """Отправляет уведомление в групповой чат менеджеров.

    Ошибка отправки не должна ронять диалог с клиентом — только лог.
    """
    manager_chat_id = context.application.bot_data.get("manager_chat_id")
    if not manager_chat_id:
        logger.error("MANAGER_CHAT_ID не задан — уведомление не отправлено")
        return
    try:
        await context.bot.send_message(
            chat_id=manager_chat_id, text=text, parse_mode=ParseMode.HTML
        )
    except TelegramError as exc:
        logger.error("Не удалось отправить уведомление менеджерам: %s", exc)


async def _handle_tags(
    context: ContextTypes.DEFAULT_TYPE,
    user: Any,
    parsed: ParsedResponse,
) -> None:
    """Обрабатывает [LEAD_CAPTURED] и [ESCALATE] из ответа модели.

    Общий код для текстового и голосового режимов: теги в обоих случаях
    приходят из одной и той же модели с одним промптом.
    """
    if parsed.has_lead and parsed.lead is not None:
        # Карточка без контакта не пригодна для работы и при этом отключила бы
        # follow-up (он смотрит, есть ли лид). Ждём, пока контакт появится.
        if not parsed.lead.contact:
            logger.info(
                "Карточка без контакта для user_id=%s — лид не создаём", user.id
            )
            return

        await db.create_lead(
            user_id=user.id,
            name=parsed.lead.name,
            contact=parsed.lead.contact,
            service=parsed.lead.service,
            context=parsed.lead.context,
            card=parsed.lead.extras(),
        )
        await _notify_manager(
            context,
            format_lead_notification(
                parsed.lead,
                user_id=user.id,
                username=user.username,
                first_name=user.first_name,
            ),
        )

    if parsed.has_escalation and parsed.escalation:
        recent = await db.get_history(user.id, limit=ESCALATION_HISTORY_SIZE)
        await _notify_manager(
            context,
            format_escalation_notification(
                parsed.escalation,
                [(m.role, m.content) for m in recent],
                user_id=user.id,
                username=user.username,
                first_name=user.first_name,
            ),
        )


async def _handle_quota_exceeded(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user: object,
    exc: QuotaExceeded,
) -> None:
    """Реакция на срабатывание опционального лимита запросов.

    Сообщение пользователя уже сохранено в истории, поэтому после снятия
    лимита диалог продолжится с полным контекстом. Ответ-заглушку в историю
    не пишем — это не реплика модели.
    """
    user_id = getattr(user, "id", 0)

    if exc.scope == "minute":
        logger.warning("Минутный лимит для user_id=%s (%s/%s)", user_id, exc.used, exc.limit)
        await _send_long(context, chat_id, QUOTA_MINUTE_REPLY)
        return

    logger.error(
        "Суточная квота исчерпана (%s/%s), user_id=%s остался без ответа",
        exc.used,
        exc.limit,
        user_id,
    )
    await _send_long(context, chat_id, QUOTA_DAY_REPLY)

    # Менеджеру сообщаем один раз в сутки, чтобы не превращать чат в спам.
    today = datetime.now(timezone.utc).date().isoformat()
    if context.application.bot_data.get("quota_alert_date") != today:
        context.application.bot_data["quota_alert_date"] = today
        username = getattr(user, "username", None)
        handle = f"@{username}" if username else "—"
        await _notify_manager(
            context,
            "⚠️ <b>Дневной лимит запросов к модели исчерпан</b>\n\n"
            f"Клиенты получают ответ-заглушку с просьбой оставить контакт.\n"
            f"Первый затронутый: {handle} (<code>{user_id}</code>).\n\n"
            "Лимит сбросится по расписанию квоты. "
            "Если это повторяется — нужен платный тариф или другая модель.",
        )


async def _send_long(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str) -> None:
    """Отправляет текст, разбивая его на части по лимиту Telegram."""
    for chunk in _split_text(text):
        try:
            await context.bot.send_message(chat_id=chat_id, text=chunk)
        except Forbidden:
            logger.info("Пользователь заблокировал бота: chat_id=%s", chat_id)
            return
        except TelegramError as exc:
            logger.error("Ошибка отправки сообщения в чат %s: %s", chat_id, exc)
            return


def _split_text(text: str, limit: int = TELEGRAM_MAX_MESSAGE) -> list[str]:
    """Режет длинный текст по абзацам/строкам, не разрывая слова без нужды."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remainder = text
    while len(remainder) > limit:
        window = remainder[:limit]
        split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
        if split_at <= 0:
            split_at = limit
        chunks.append(remainder[:split_at].strip())
        remainder = remainder[split_at:].strip()
    if remainder:
        chunks.append(remainder)
    return chunks


async def _keep_typing(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    action: str = ChatAction.TYPING,
) -> None:
    """Показывает «печатает…» / «записывает голосовое», пока идёт запрос."""
    try:
        while True:
            await context.bot.send_chat_action(chat_id=chat_id, action=action)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass
    except TelegramError:
        pass
