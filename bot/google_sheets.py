"""Неблокирующая синхронизация лидов с CRM в Google Sheets.

В самой CRM нет технической колонки с Telegram user_id: её не добавляем,
чтобы не менять согласованную структуру листа. Связка ``user_id → номер строки``
хранится локально в SQLite (см. ``crm_rows`` в :mod:`bot.db`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build

from . import db

if TYPE_CHECKING:
    from .parser import Lead

logger = logging.getLogger(__name__)

SCOPES = ("https://www.googleapis.com/auth/spreadsheets",)
CRM_HEADERS = (
    "Имя клиента",
    "Контакт",
    "Дата обращения",
    "Источник",
    "Тип услуги",
    "Статус",
    "Ответственный",
    "Оплата",
    "Комментарий",
    "Следующий шаг",
)
DEFAULT_WORKSHEET = "CRM"
MOSCOW_TZ = ZoneInfo("Europe/Moscow")


@dataclass(slots=True)
class GoogleSheetsCRM:
    """Клиент CRM. Все синхронные вызовы Google API выполняются в thread pool."""

    spreadsheet_id: str
    worksheet: str
    service: Any

    @classmethod
    def from_env(cls) -> "GoogleSheetsCRM | None":
        """Собирает клиент из service-account JSON или возвращает None, если CRM не настроена.

        Поддерживаются два безопасных способа передачи ключа:

        * ``GOOGLE_SERVICE_ACCOUNT_FILE`` — путь к JSON-файлу;
        * ``GOOGLE_SERVICE_ACCOUNT_JSON`` — полное содержимое JSON (удобно для Render).
        """
        spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "").strip()
        credentials_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
        credentials_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

        if not spreadsheet_id or not (credentials_file or credentials_json):
            logger.warning(
                "Google Sheets CRM выключена: задайте GOOGLE_SHEETS_SPREADSHEET_ID и "
                "GOOGLE_SERVICE_ACCOUNT_FILE либо GOOGLE_SERVICE_ACCOUNT_JSON (см. README)."
            )
            return None

        try:
            if credentials_json:
                credentials = service_account.Credentials.from_service_account_info(
                    json.loads(credentials_json), scopes=SCOPES
                )
            else:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_file, scopes=SCOPES
                )
            service = build("sheets", "v4", credentials=credentials, cache_discovery=False)
        except Exception:
            # Не даём необязательной интеграции остановить запуск бота.
            logger.exception("Не удалось инициализировать Google Sheets CRM")
            return None

        return cls(
            spreadsheet_id=spreadsheet_id,
            worksheet=os.getenv("GOOGLE_SHEETS_WORKSHEET", DEFAULT_WORKSHEET).strip()
            or DEFAULT_WORKSHEET,
            service=service,
        )

    async def upsert_lead(self, user_id: int, lead: "Lead", fallback_name: str | None) -> None:
        """Создаёт или обновляет CRM-заявку пользователя.

        Исключения намеренно подавляются здесь: лид и сообщение уже сохранены
        в SQLite, а клиентский ответ Telegram не должен зависеть от Google API.
        При ошибке остаётся подробная запись в отдельном логгере этого модуля.
        """
        try:
            row = await db.get_crm_row(user_id, self.spreadsheet_id, self.worksheet)
            if row:
                await asyncio.to_thread(self._update_row, row, lead, fallback_name)
                logger.info("CRM обновлена: user_id=%s, row=%s", user_id, row)
                return

            row = await asyncio.to_thread(self._append_row, lead, fallback_name)
            await db.save_crm_row(user_id, self.spreadsheet_id, self.worksheet, row)
            logger.info("CRM создана: user_id=%s, row=%s", user_id, row)
        except Exception:
            logger.exception(
                "Ошибка записи CRM в Google Sheets (user_id=%s, spreadsheet=%s, worksheet=%s)",
                user_id,
                self.spreadsheet_id,
                self.worksheet,
            )

    @property
    def _sheet_range(self) -> str:
        # В имени листа допустим апостроф, поэтому экранируем его по A1 notation.
        return "'" + self.worksheet.replace("'", "''") + "'"

    def _append_row(self, lead: "Lead", fallback_name: str | None) -> int:
        values = [[
            lead.name or fallback_name or "",
            lead.contact or "",
            datetime.now(MOSCOW_TZ).strftime("%d.%m.%Y %H:%M"),
            "бот",
            lead.service or "",
            "новый",
            "",
            "",
            "",
            "",
        ]]
        response = (
            self.service.spreadsheets()
            .values()
            .append(
                spreadsheetId=self.spreadsheet_id,
                range=f"{self._sheet_range}!A:J",
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body={"values": values},
            )
            .execute()
        )
        updated_range = response.get("updates", {}).get("updatedRange", "")
        match = re.search(r"![A-Z]+(\d+):", updated_range)
        if not match:
            raise RuntimeError(f"Google Sheets не вернул номер созданной строки: {updated_range!r}")
        return int(match.group(1))

    def _update_row(self, row: int, lead: "Lead", fallback_name: str | None) -> None:
        """Обновляет только данные, полученные от клиента.

        Дату/источник/статус и ручные поля менеджеров не перезаписываем при
        повторном обращении. Пустые значения от модели тоже не стирают CRM.
        """
        updates: list[dict[str, object]] = []
        name = lead.name or fallback_name
        if name:
            updates.append({"range": f"{self._sheet_range}!A{row}", "values": [[name]]})
        if lead.contact:
            updates.append({"range": f"{self._sheet_range}!B{row}", "values": [[lead.contact]]})
        if lead.service:
            updates.append({"range": f"{self._sheet_range}!E{row}", "values": [[lead.service]]})
        if not updates:
            return
        (
            self.service.spreadsheets()
            .values()
            .batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body={"valueInputOption": "USER_ENTERED", "data": updates},
            )
            .execute()
        )
