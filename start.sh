#!/bin/sh
# Запуск OCR-сервиса: sh start.sh
# Если в системе есть Python 3.12 — используется он, иначе встроенный из vendor/python.tar.gz.
set -e
cd "$(dirname "$0")"
if command -v python3.12 >/dev/null 2>&1; then
    exec python3.12 main.py "$@"
fi
if [ ! -x vendor/python/bin/python3 ]; then
    echo "Распаковка встроенного Python 3.12..."
    tar -xzf vendor/python.tar.gz -C vendor
fi
exec vendor/python/bin/python3 main.py "$@"
