"""Предполётная проверка окружения: `python -m bot.preflight`.

Проверяет всё, что нужно боту для старта, и печатает понятный отчёт:

- заданы ли обязательные переменные окружения;
- жив ли ключ OpenAI (запрос списка моделей — токены не тратятся);
- доступна ли выбранная чат-модель;
- валиден ли токен Telegram;
- виден ли чат менеджеров и может ли бот туда писать.

Ничего никуда не отправляет: все вызовы читающие, сообщение в группу
менеджеров не уходит.
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

OK = "  ✓"
FAIL = "  ✗"
WARN = "  !"


def _mask(value: str) -> str:
    """Показывает только хвост секрета — чтобы не светить его в терминале."""
    return f"…{value[-4:]}" if len(value) > 8 else "(слишком короткий)"


async def check_env() -> tuple[bool, dict[str, str]]:
    print("Переменные окружения")
    values: dict[str, str] = {}
    ok = True
    for name in ("TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "MANAGER_CHAT_ID"):
        value = os.getenv(name, "")
        if value:
            shown = value if name == "MANAGER_CHAT_ID" else _mask(value)
            print(f"{OK} {name} = {shown}")
            values[name] = value
        else:
            print(f"{FAIL} {name} не задан")
            ok = False
    return ok, values


async def check_openai(api_key: str) -> bool:
    print("\nOpenAI")
    from openai import AsyncOpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    client = AsyncOpenAI(api_key=api_key)
    try:
        available = {m.id async for m in client.models.list()}
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} ключ не работает: {type(exc).__name__}: {str(exc)[:120]}")
        return False
    finally:
        await client.close()

    print(f"{OK} ключ рабочий, доступно моделей: {len(available)}")
    if model in available:
        print(f"{OK} чат-модель {model} доступна")
    else:
        print(f"{FAIL} чат-модель {model} недоступна на этом ключе")
        return False

    if os.getenv("VOICE_ENABLED", "1") == "1":
        for env_name, default in (
            ("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
            ("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        ):
            name = os.getenv(env_name, default)
            mark = OK if name in available else WARN
            note = "" if name in available else " — голосовой режим не заработает"
            print(f"{mark} голосовая модель {name} доступна{note}")
    else:
        print(f"{WARN} голосовой режим выключен (VOICE_ENABLED=0)")
    return True


async def check_telegram(token: str, manager_chat_id: str) -> bool:
    print("\nTelegram")
    from telegram import Bot
    from telegram.error import InvalidToken, TelegramError

    bot = Bot(token)
    try:
        async with bot:
            me = await bot.get_me()
            print(f"{OK} токен рабочий — бот @{me.username} (id={me.id})")

            try:
                chat_id = int(manager_chat_id)
            except ValueError:
                print(f"{FAIL} MANAGER_CHAT_ID={manager_chat_id!r} — не число")
                return False

            if chat_id > 0:
                print(
                    f"{WARN} MANAGER_CHAT_ID={chat_id} положительный — похоже на личку.\n"
                    "      Нужен ID ГРУППЫ (отрицательный, у супергрупп с -100)."
                )

            try:
                chat = await bot.get_chat(chat_id)
                print(f"{OK} чат менеджеров виден: {chat.title!r} ({chat.type})")
                member = await bot.get_chat_member(chat_id, me.id)
                if member.status in ("administrator", "member", "creator"):
                    print(f"{OK} бот состоит в чате (статус: {member.status})")
                else:
                    print(f"{FAIL} бот не участник чата (статус: {member.status})")
                    return False
            except TelegramError as exc:
                print(f"{FAIL} чат {chat_id} недоступен: {exc}")
                print("      Добавьте бота в группу и отправьте туда любое сообщение.")
                return False
    except InvalidToken:
        print(f"{FAIL} токен отклонён Telegram (Unauthorized)")
        print("      Возьмите актуальный в @BotFather: /mybots → бот → API Token")
        return False
    except TelegramError as exc:
        print(f"{FAIL} Telegram недоступен: {exc}")
        return False
    return True


async def main() -> int:
    load_dotenv()
    print("Предполётная проверка Non-Stop Visa бота\n")

    env_ok, values = await check_env()
    if not env_ok:
        print("\nИтог: заполните .env (см. .env.example) и повторите.")
        return 1

    openai_ok = await check_openai(values["OPENAI_API_KEY"])
    telegram_ok = await check_telegram(
        values["TELEGRAM_BOT_TOKEN"], values["MANAGER_CHAT_ID"]
    )

    print()
    if openai_ok and telegram_ok:
        print("Итог: всё готово. Запускайте: python -m bot.main")
        return 0
    print("Итог: есть проблемы — см. отметки ✗ выше.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
