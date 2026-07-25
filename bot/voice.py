"""Голосовой режим: распознавание речи и синтез через OpenAI Audio API.

Голосовой ход устроен в три шага, и все три работают в форматах, которые
Telegram отдаёт и принимает без перекодирования:

    голосовое OGG/Opus из Telegram
      → gpt-4o-mini-transcribe        (STT принимает OGG напрямую)
      → та же чат-модель, что и в тексте  (см. bot/openai_client.py)
      → gpt-4o-mini-tts, response_format="opus"  → готовый Ogg/Opus в Telegram

Отсюда два следствия:

1. **Конвертация не нужна** — ни ffmpeg, ни какой-либо иной системной
   зависимости: форматы совпадают на обоих концах.
2. **Теги работают и в голосе.** Чат-модель отвечает текстом, поэтому блок
   [LEAD_CAPTURED] вырезается ДО озвучки: клиент его не слышит, а лид
   фиксируется так же, как в текстовом режиме.

Расшифровки обеих реплик пишутся в общую историю диалога, поэтому клиент
может начать голосом и продолжить текстом без потери контекста.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

DEFAULT_STT_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "alloy"

# Инструкция голосу: та же роль, что и в тексте, но озвученная.
TTS_INSTRUCTIONS = (
    "Speak in a calm, friendly and professional tone, like a competent "
    "consultant on the phone. Natural pace, no exaggerated emotion."
)

# Ограничение на длину входящего голосового: защита от случайной
# многоминутной записи (у OpenAI лимит 25 МБ на файл).
MAX_VOICE_BYTES = 20 * 1024 * 1024


class VoiceError(RuntimeError):
    """Не удалось обработать голосовое сообщение."""


class VoiceClient:
    """STT + TTS поверх OpenAI Audio API."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        stt_model: Optional[str] = None,
        tts_model: Optional[str] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        # HTTP-клиент переиспользуется общий с текстовым режимом —
        # отдельное соединение и второй пул ни к чему.
        self._client = client
        self.stt_model = stt_model or os.getenv("OPENAI_STT_MODEL", DEFAULT_STT_MODEL)
        self.tts_model = tts_model or os.getenv("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)
        self.voice = voice or os.getenv("OPENAI_TTS_VOICE", DEFAULT_VOICE)
        # Подсказка языка заметно повышает точность на коротких записях.
        self.language = language or os.getenv("OPENAI_STT_LANGUAGE", "ru")

    async def transcribe(self, ogg: bytes, filename: str = "voice.ogg") -> str:
        """Голосовое из Telegram (OGG/Opus) → текст.

        Raises:
            VoiceError: файл пуст, слишком велик или API вернул ошибку.
        """
        if not ogg:
            raise VoiceError("Пустое голосовое сообщение")
        if len(ogg) > MAX_VOICE_BYTES:
            raise VoiceError(f"Голосовое слишком большое: {len(ogg) // 1024} КБ")

        buffer = io.BytesIO(ogg)
        buffer.name = filename  # SDK определяет формат по имени файла

        try:
            result = await self._client.audio.transcriptions.create(
                model=self.stt_model,
                file=buffer,
                language=self.language,
            )
        except openai.APIError as exc:
            raise VoiceError(f"Распознавание не удалось: {exc}") from exc

        text = (getattr(result, "text", "") or "").strip()
        if not text:
            raise VoiceError("Речь не распознана")
        logger.info("STT: распознано %s символов", len(text))
        return text

    async def synthesize(self, text: str) -> bytes:
        """Текст → голосовое в формате Telegram (Ogg/Opus).

        Raises:
            VoiceError: пустой текст или ошибка API.
        """
        text = (text or "").strip()
        if not text:
            raise VoiceError("Нечего озвучивать")

        try:
            response = await self._client.audio.speech.create(
                model=self.tts_model,
                voice=self.voice,
                input=text,
                # opus — нативный формат голосовых Telegram, перекодировать
                # ничего не нужно.
                response_format="opus",
                instructions=TTS_INSTRUCTIONS,
            )
            audio = await response.aread()
        except openai.APIError as exc:
            raise VoiceError(f"Синтез речи не удался: {exc}") from exc

        if not audio:
            raise VoiceError("Синтез вернул пустой файл")
        logger.info("TTS: сгенерировано %s КБ", len(audio) // 1024)
        return audio
