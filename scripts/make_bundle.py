"""Сборка архива для переноса в контейнер: python scripts/make_bundle.py [--dry-run]

В архив dist/ocr-offline.tar (папка ocr/ внутри) попадает только то, что нужно для работы:
код, vendor/wheels, vendor/python.tar.gz, веса модели и документация. Локальные артефакты
этого компьютера — logs/, results/, __pycache__, .venv, vendor/site-packages — не попадают,
поэтому в контейнере не появятся пути и логи машины, где готовился проект.

Перед сборкой проверяется, что скачаны все пакеты из requirements.lock и веса модели.
"""
import argparse
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["main.py", "start.sh", "requirements.lock", "requirements.txt", "README.md", "DEPLOY.md",
         "Dockerfile", "docker-compose.yml", ".dockerignore", "vendor/python.tar.gz"]
DIRS = ["app", "vendor/wheels"]
SKIP = {"__pycache__", ".cache"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check(model_dir: Path | None) -> None:
    wheels = {_norm(w.name.split("-")[0]) for w in (ROOT / "vendor" / "wheels").glob("*.whl")}
    lock = [line.split("==")[0] for line in (ROOT / "requirements.lock").read_text().splitlines()
            if "==" in line]
    missing = [name for name in lock if _norm(name) not in wheels]
    if missing:
        sys.exit(f"В vendor/wheels нет пакетов: {', '.join(missing)} — запустите scripts/prepare_offline.py")
    if not any(wheels & {"pip"}):
        sys.exit("В vendor/wheels нет колеса pip — запустите scripts/prepare_offline.py")
    if model_dir and not (model_dir / "model.safetensors.index.json").exists():
        sys.exit(f"Нет весов модели в {model_dir}")
    if not (ROOT / "vendor" / "python.tar.gz").exists():
        sys.exit("Нет vendor/python.tar.gz — запустите scripts/prepare_offline.py")
    print(f"Проверка пройдена: {len(lock)} пакетов из requirements.lock + pip, "
          f"модель {model_dir.name if model_dir else 'не включается'}")


def collect(model_dir: Path | None) -> list[Path]:
    paths = [ROOT / f for f in FILES]
    for d in DIRS + ([str(model_dir.relative_to(ROOT))] if model_dir else []):
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not SKIP & set(p.relative_to(ROOT).parts):
                paths.append(p)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--dry-run", action="store_true", help="только проверить и показать состав")
    parser.add_argument("--without-model", action="store_true",
                        help="без весов модели (если модель передаётся отдельно в models/)")
    args = parser.parse_args()

    model_dir = None if args.without_model else ROOT / "models" / args.model
    check(model_dir)
    paths = collect(model_dir)
    size = sum(p.stat().st_size for p in paths)
    print(f"Файлов: {len(paths)}, объём {size / 1024**3:.1f} ГБ")
    if args.dry_run:
        for p in paths:
            if p.parent.name not in {"wheels"} and "models" not in p.parts:
                print("  ", p.relative_to(ROOT).as_posix())
        print("   + vendor/wheels/*.whl", "и " + model_dir.relative_to(ROOT).as_posix() if model_dir else "")
        sys.exit(0)

    out = ROOT / "dist" / "ocr-offline.tar"
    out.parent.mkdir(exist_ok=True)
    with tarfile.open(out, "w") as tar:
        for p in paths:
            info = tar.gettarinfo(p, arcname="ocr/" + p.relative_to(ROOT).as_posix())
            # Права и владелец — как для Linux, независимо от ОС, где собирается архив
            info.mode = 0o755 if p.suffix == ".sh" else 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            with open(p, "rb") as fh:
                tar.addfile(info, fh)
    print(f"Готово: {out} ({out.stat().st_size / 1024**3:.1f} ГБ)")
    print("В контейнере: tar -xf ocr-offline.tar && sh ocr/start.sh")
