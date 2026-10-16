#!/bin/sh
# Запуск OCR-сервиса в Linux-контейнере: sh start.sh
# Если в системе есть Python 3.12 — используется он, иначе встроенный из vendor/python.tar.gz.
set -e
cd "$(dirname "$0")"

# Пакеты в vendor/wheels и встроенный Python собраны под Linux x86_64
if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "start.sh рассчитан на Linux x86_64 (сейчас: $(uname -s) $(uname -m))." >&2
    echo "Запускайте его внутри Linux-контейнера; на Windows — см. «Установка для разработки» в README.md." >&2
    exit 1
fi

if command -v python3.12 >/dev/null 2>&1; then
    exec python3.12 main.py "$@"
fi

if [ ! -x vendor/python/bin/python3 ]; then
    echo "Распаковка встроенного Python 3.12..."
    # Распаковка во временную папку: при сбое не остаётся недоделанного vendor/python
    rm -rf vendor/python.tmp
    mkdir vendor/python.tmp
    tar -xzf vendor/python.tar.gz -C vendor/python.tmp
    rm -rf vendor/python
    mv vendor/python.tmp/python vendor/python
    rmdir vendor/python.tmp
fi
exec vendor/python/bin/python3 main.py "$@"
