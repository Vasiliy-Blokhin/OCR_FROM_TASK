"""Логирование сервиса: консоль + файл logs/ocr.log с ротацией.

Уровень — OCR_LOG_LEVEL (INFO по умолчанию; DEBUG добавляет сырые ответы модели и детали обработки).
К каждой записи добавляется идентификатор запроса, чтобы в логе отличать параллельные документы.
"""
import contextvars
import logging
from logging.handlers import RotatingFileHandler

from app import config

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_configured = False


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.rid = request_id.get()
        return True


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(rid)s] %(name)s: %(message)s")
    root = logging.getLogger("ocr")
    root.setLevel(config.LOG_LEVEL)
    root.propagate = False

    console = logging.StreamHandler()
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    file = RotatingFileHandler(
        config.LOG_DIR / "ocr.log", maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    for handler in (console, file):
        handler.setFormatter(fmt)
        handler.addFilter(_RequestIdFilter())
        root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ocr.{name}")
