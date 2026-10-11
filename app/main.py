from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.ocr import OCREngine
from app.pdf import render_pages

engine: OCREngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = OCREngine()
    yield


app = FastAPI(title="OCR", lifespan=lifespan)


@app.post("/ocr")
def ocr(file: UploadFile = File(...)) -> dict[str, str]:
    data = file.file.read()
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Ожидается PDF-файл")
    try:
        pages = render_pages(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось открыть PDF")
    return {str(i): engine.recognize_page(page) for i, page in enumerate(pages, start=1)}
