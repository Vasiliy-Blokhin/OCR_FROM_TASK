# Образ собирается без интернета: зависимости берутся из vendor/wheels, модель — из models/.
# Перед сборкой проект должен быть подготовлен: python scripts/prepare_offline.py
# CUDA-библиотеки поставляются в колёсах; на хосте нужен драйвер NVIDIA и NVIDIA Container Toolkit.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
# Установка зависимостей на этапе сборки (тот же код, что при первом запуске main.py)
RUN python -c "import main; main.ensure_dependencies()"

EXPOSE 8000
CMD ["python", "main.py"]
