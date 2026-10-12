"""Подготовка проекта к работе без интернета (запускается один раз на машине с интернетом).

Скачивает в папку проекта всё, что нужно в изолированном Linux-контейнере (x86_64):
  vendor/wheels/  — все Python-пакеты из requirements.lock (включая транзитивные и CUDA-библиотеки)
  vendor/python/  — автономный CPython 3.12 для Linux (для контейнеров без Python 3.12)
  models/<модель> — веса модели

Работает на Windows и Linux: колёса качаются под платформу Linux независимо от текущей ОС.
    python scripts/prepare_offline.py
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"
WHEELS = VENDOR / "wheels"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
PBS_RELEASE = "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"


def download_wheels():
    WHEELS.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pip", "download",
        "-r", str(ROOT / "requirements.lock"),
        "-d", str(WHEELS),
        "--extra-index-url", TORCH_INDEX,
        "--only-binary=:all:", "--no-deps",
        "--python-version", "3.12", "--implementation", "cp", "--abi", "cp312", "--abi", "abi3", "--abi", "none",
    ]
    # Колёса для glibc 2.17–2.28: подходят любому дистрибутиву с glibc >= 2.28 (максимум у torch)
    platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 16, -1)]
    platforms += ["manylinux2014_x86_64", "linux_x86_64"]
    for plat in platforms:
        cmd += ["--platform", plat]
    subprocess.run(cmd, check=True)
    # Колесо pip: main.py запускает установку из него, даже если в контейнере нет pip
    subprocess.run([sys.executable, "-m", "pip", "download", "pip==25.2", "--no-deps", "-d", str(WHEELS)], check=True)


def download_python():
    # Храним архивом: start.sh распакует его уже в Linux, с правильными правами и ссылками
    archive = VENDOR / "python.tar.gz"
    if archive.exists():
        print("Python уже скачан:", archive)
        return
    with urllib.request.urlopen(PBS_RELEASE) as resp:
        assets = json.load(resp)["assets"]
    asset = next(
        a for a in assets
        if a["name"].startswith("cpython-3.12.")
        and a["name"].endswith("-x86_64-unknown-linux-gnu-install_only.tar.gz")
    )
    print("Скачиваю", asset["name"])
    urllib.request.urlretrieve(asset["browser_download_url"], archive)


def download_model(repo: str):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub==2.2.0"], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "download_model.py"), "--repo", repo], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    download_wheels()
    download_python()
    if not args.skip_model:
        download_model(args.repo)
    print("Готово: проект можно копировать в контейнер без интернета.")
