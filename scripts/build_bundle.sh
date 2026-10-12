#!/usr/bin/env bash
# Сборка офлайн-архива на машине с интернетом и Docker.
# Результат: dist/ocr-offline.tar.gz (образ с моделью) + docker-compose.yml + DEPLOY.md —
# эту папку переносят в изолированный контур (см. DEPLOY.md).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen3-VL-8B-Instruct}"
IMAGE="ocr-offline:latest"

if [ ! -f "models/${MODEL}/model.safetensors.index.json" ]; then
    echo "Скачиваю модель ${MODEL}..."
    python3 -m pip install --quiet "huggingface_hub==2.2.0"
    python3 scripts/download_model.py --repo "Qwen/${MODEL}"
fi

docker build --build-arg MODEL="${MODEL}" -t "${IMAGE}" .

mkdir -p dist
echo "Сохраняю образ в dist/ocr-offline.tar.gz (несколько минут)..."
docker save "${IMAGE}" | gzip -1 > dist/ocr-offline.tar.gz
cp docker-compose.yml DEPLOY.md dist/
ls -lh dist
