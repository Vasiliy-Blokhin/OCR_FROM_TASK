"""Точка входа OCR-сервиса: python main.py

При первом запуске ставит зависимости из vendor/wheels (без интернета) в vendor/site-packages,
затем поднимает HTTP-сервер с эндпоинтом POST /ocr на 0.0.0.0:8000.
Порт и адрес задаются переменными OCR_HOST / OCR_PORT.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WHEELS = ROOT / "vendor" / "wheels"
SITE = ROOT / "vendor" / "site-packages"
MARKER = SITE / ".installed"


def ensure_dependencies():
    if sys.version_info[:2] != (3, 12):
        sys.exit(
            f"Нужен Python 3.12 (сейчас {sys.version.split()[0]}): пакеты в vendor/wheels собраны под 3.12. "
            "Запустите ./start.sh — он использует встроенный Python из vendor/python."
        )
    if MARKER.exists():
        return
    if not WHEELS.is_dir():
        sys.exit("Нет vendor/wheels: подготовьте проект командой python scripts/prepare_offline.py")
    print("Первый запуск: установка зависимостей из vendor/wheels (без интернета)...", flush=True)
    # pip запускается прямо из своего колеса: в контейнере pip может быть не установлен
    pip_wheel = next(WHEELS.glob("pip-*.whl"))
    subprocess.run(
        [
            sys.executable, str(pip_wheel / "pip"), "install",
            "--no-index", "--find-links", str(WHEELS),
            "--target", str(SITE),
            "--no-warn-script-location", "--disable-pip-version-check",
            "-r", str(ROOT / "requirements.lock"),
        ],
        check=True,
    )
    MARKER.touch()


def main():
    ensure_dependencies()
    sys.path.insert(0, str(SITE))
    sys.path.insert(0, str(ROOT))
    # Никаких обращений к Hugging Face Hub: модель читается только из локальной папки
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("OCR_HOST", "0.0.0.0"),
        port=int(os.getenv("OCR_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
