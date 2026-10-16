"""Прогон OCR по PDF без HTTP-сервера: python scripts/run_local.py file1.pdf [file2.jpg ...]

Для каждого файла сохраняются results/<имя>.json (ключ — номер страницы) и results/<имя>.docx.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT  # noqa: E402
from app.log import setup_logging  # noqa: E402
from app.docx_export import build_docx  # noqa: E402
from app.ocr import OCREngine  # noqa: E402
from app.pdf import load_document  # noqa: E402

out_dir = ROOT / "results"
out_dir.mkdir(exist_ok=True)

setup_logging()
engine = OCREngine()
for path in map(Path, sys.argv[1:]):
    pages = []
    for i, img in enumerate(load_document(path.read_bytes()), start=1):
        t = time.time()
        pages.append(engine.recognize_page(img, i))
        print(f"{path.name} стр. {i}: {time.time() - t:.1f} с", flush=True)
    result = {str(i): page.text for i, page in enumerate(pages, start=1)}
    out = out_dir / f"{path.stem}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{path.stem}.docx").write_bytes(build_docx(pages))
    print(f"-> {out} (+ .docx)")
