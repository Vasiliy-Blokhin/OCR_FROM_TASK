# Офлайн-образ OCR: зависимости и веса модели внутри, интернет при запуске не нужен.
# CUDA-библиотеки поставляются в колёсах torch; на хосте нужен только драйвер NVIDIA
# и NVIDIA Container Toolkit.
FROM python:3.12-slim

ARG MODEL=Qwen3-VL-8B-Instruct

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OCR_MODEL_PATH=/app/models/${MODEL}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Веса копируются до кода: при правке кода слой модели (~17 ГБ) не пересобирается
COPY models/${MODEL} /app/models/${MODEL}
COPY app /app/app

RUN useradd --create-home ocr && chown -R ocr /app
USER ocr

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
