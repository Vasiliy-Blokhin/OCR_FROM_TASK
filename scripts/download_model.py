"""Скачивает веса модели в локальную папку models/ (нужен интернет, запускается один раз)."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="Qwen/Qwen3-VL-8B-Instruct")
args = parser.parse_args()

target = ROOT / "models" / args.repo.split("/")[-1]
snapshot_download(repo_id=args.repo, local_dir=target)
print(f"Модель сохранена в {target}")
