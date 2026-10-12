# Развёртывание OCR в изолированном контуре

Образ содержит все зависимости и веса модели (Qwen3-VL-8B), интернет не нужен.

## Требования к серверу

| Ресурс | Минимум | Рекомендуется |
|---|---|---|
| GPU | NVIDIA 8 ГБ VRAM (`OCR_QUANT=4bit`) | NVIDIA от 24 ГБ VRAM (L4, A10, A100, H100) |
| CPU | 4–6 ядер | 8 ядер |
| RAM | 16 ГБ | 32 ГБ |
| Диск | 60 ГБ (архив ~22 ГБ + загруженный образ ~25 ГБ; архив после `docker load` можно удалить) | 80 ГБ |
| ПО | Linux, драйвер NVIDIA с поддержкой CUDA 12.8 (≥ 570), Docker, NVIDIA Container Toolkit | — |

Проверка, что Docker видит GPU: `docker run --rm --gpus all ubuntu nvidia-smi`
(образ `ubuntu` должен быть в контуре; можно пропустить и смотреть на логи сервиса).

## Запуск

```bash
docker load -i ocr-offline.tar.gz
docker compose up -d
docker compose logs -f     # ждём строку "Application startup complete" (~1–2 мин, загрузка модели)
```

Без compose:

```bash
docker run -d --gpus all -p 8000:8000 --name ocr ocr-offline:latest
```

## Использование

```bash
curl -F "file=@document.pdf" http://localhost:8000/ocr > result.json
```

Ответ: `{"pages": {"1": "...", ...}, "docx": "<base64>"}`. Обработка идёт 25–60 с на страницу
на GPU 8 ГБ, поэтому у клиента должен быть большой таймаут (для документа на 40 страниц — до 40 минут).
Запросы обрабатываются по очереди.

## Настройки

Задаются в `docker-compose.yml` (секция `environment`):

| Переменная | По умолчанию | Описание |
|---|---|---|
| `OCR_QUANT` | `4bit` | `none` — без сжатия модели, для GPU от 24 ГБ: быстрее и точнее |
| `OCR_MAX_PIXELS` | `1200000` | Размер полосы страницы; на GPU от 24 ГБ — `4000000` |
| `OCR_DPI` | `300` | DPI рендера страниц |
| `OCR_TILE_WIDTH` | `1700` | Ширина страницы перед нарезкой на полосы |
