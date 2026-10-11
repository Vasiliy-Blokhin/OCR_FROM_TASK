import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.docx_export import build_docx
from app.ocr import OCREngine
from app.pdf import render_pages

engine: OCREngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = OCREngine()
    yield


app = FastAPI(title="OCR", lifespan=lifespan)


class OCRResult(BaseModel):
    pages: dict[str, str]  # номер страницы -> распознанный текст (Markdown)
    docx: str  # Word-документ со структурой страниц, base64


@app.post("/ocr")
def ocr(file: UploadFile = File(...)) -> OCRResult:
    data = file.file.read()
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Ожидается PDF-файл")
    try:
        images = render_pages(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось открыть PDF")
    pages = [engine.recognize_page(img) for img in images]
    return OCRResult(
        pages={str(i): page.text for i, page in enumerate(pages, start=1)},
        docx=base64.b64encode(build_docx(pages)).decode("ascii"),
    )
