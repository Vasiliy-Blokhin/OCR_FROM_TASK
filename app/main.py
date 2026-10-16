import asyncio
import base64
import sys
import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app import config
from app.docx_export import build_docx
from app.log import get_logger, request_id, setup_logging
from app.ocr import OCREngine
from app.pdf import load_document
from app.scheduler import Scheduler, collect

setup_logging()
log = get_logger("api")
scheduler: Scheduler | None = None

# Как часто проверять, не отключился ли клиент, пока его страницы в работе
DISCONNECT_CHECK_SECONDS = 2


def _devices() -> list[str]:
    """Устройства для реплик модели: GPU по кругу, без GPU — CPU."""
    gpus = torch.cuda.device_count()
    if not gpus:
        return ["cpu"] * config.REPLICAS
    return [f"cuda:{i % gpus}" for i in range(config.REPLICAS)]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global scheduler
    log.info("Старт сервиса: Python %s (%s)", sys.version.split()[0], sys.executable)
    log.info("Пути: проект %s, модель %s, логи %s", config.ROOT, config.MODEL_PATH, config.LOG_DIR / "ocr.log")
    devices = _devices()
    log.info("Реплик модели: %d на устройствах %s; лимит одновременных запросов %d",
             len(devices), ", ".join(devices), config.MAX_ACTIVE_REQUESTS)
    engines = [OCREngine(device) for device in devices]
    scheduler = Scheduler(engines)
    log.info("Сервис готов принимать запросы: POST /ocr")
    yield
    log.info("Остановка сервиса")


# Только один эндпоинт: служебные /docs и /openapi.json отключены
app = FastAPI(title="OCR", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class OCRResult(BaseModel):
    pages: dict[str, str]  # номер страницы -> распознанный текст (Markdown)
    docx: str  # Word-документ со структурой страниц, base64


def _filename(name: str | None) -> str:
    """Имя файла для лога. curl из Windows-консоли шлёт его в cp1251, а сервер читает как latin-1."""
    if not name:
        return "без имени"
    try:
        return name.encode("latin-1").decode("cp1251")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


@app.post("/ocr")
async def ocr(request: Request, file: UploadFile = File(...)) -> OCRResult:
    rid = uuid.uuid4().hex[:8]
    request_id.set(rid)
    started = time.time()
    data = await file.read()
    log.info("Запрос от %s: файл «%s», %s, %.1f КБ", request.client.host if request.client else "?",
             _filename(file.filename), file.content_type, len(data) / 1024)

    if scheduler.active_requests >= config.MAX_ACTIVE_REQUESTS:
        log.warning("Отказ 503: уже %d активных запросов (лимит OCR_MAX_ACTIVE_REQUESTS=%d)",
                    scheduler.active_requests, config.MAX_ACTIVE_REQUESTS)
        raise HTTPException(status_code=503, detail="Сервис перегружен, повторите запрос позже",
                            headers={"Retry-After": "60"})
    try:
        # Рендер страниц — нагрузка на CPU: выполняем вне цикла событий, чтобы не тормозить другие запросы
        images = await run_in_threadpool(load_document, data)
    except Exception:
        log.exception("Не удалось открыть файл «%s»", _filename(file.filename))
        raise HTTPException(
            status_code=400, detail="Ожидается PDF или изображение (JPG, PNG, TIFF, BMP, WebP)"
        )
    log.info("Страниц к распознаванию: %d (подготовка %.1f с)", len(images), time.time() - started)

    job = scheduler.submit(rid, images)
    pending = {asyncio.wrap_future(f) for f in job.futures}
    try:
        while pending:
            done, pending = await asyncio.wait(pending, timeout=DISCONNECT_CHECK_SECONDS,
                                               return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                task.result()  # ошибка страницы прерывает весь запрос
            if pending and await request.is_disconnected():
                log.warning("Клиент отключился, не дождавшись ответа")
                scheduler.cancel(job)
                raise HTTPException(status_code=499, detail="Клиент отключился")
            if done:
                elapsed = time.time() - started
                log.info("Прогресс: %d/%d стр., прошло %.0f с, осталось ≈%.0f с", job.done, job.total,
                         elapsed, elapsed / max(job.done, 1) * (job.total - job.done))
    except HTTPException:
        raise
    except Exception:
        scheduler.cancel(job)
        log.exception("Запрос прерван из-за ошибки распознавания")
        raise HTTPException(status_code=500, detail="Ошибка распознавания, подробности в логе сервиса")

    pages = collect(job)
    docx_started = time.time()
    docx = await run_in_threadpool(build_docx, pages)
    log.info("Word собран: %.1f КБ за %.2f с", len(docx) / 1024, time.time() - docx_started)
    log.info("Готово: %d стр. за %.1f с (%.1f с/стр.)", len(pages), time.time() - started,
             (time.time() - started) / max(len(pages), 1))
    return OCRResult(
        pages={str(i): page.text for i, page in enumerate(pages, start=1)},
        docx=base64.b64encode(docx).decode("ascii"),
    )
