"""Прогон OCR по PDF без HTTP-сервера: python scripts/run_local.py file1.pdf [file2.pdf ...]

Результат для каждого файла сохраняется в results/<имя>.json (ключ — номер страницы).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT  # noqa: E402
from app.ocr import OCREngine  # noqa: E402
from app.pdf import render_pages  # noqa: E402

out_dir = ROOT / "results"
out_dir.mkdir(exist_ok=True)

engine = OCREngine()
for path in map(Path, sys.argv[1:]):
    result = {}
    for i, page in enumerate(render_pages(path.read_bytes()), start=1):
        t = time.time()
        result[str(i)] = engine.recognize_page(page)
        print(f"{path.name} стр. {i}: {time.time() - t:.1f} с", flush=True)
    out = out_dir / f"{path.stem}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
