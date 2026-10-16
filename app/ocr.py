import math
import os
import re
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from app import config
from app.docx_export import Page
from app.log import get_logger
from app.pdf import horizontal_text_ratio, split_strips
from app.tables import (
    drop_duplicated_lines,
    expand_br_rows,
    is_table_row,
    strip_repeated_header,
    trailing_table_header,
)

log = get_logger("engine")


def _vram(device: str) -> str:
    if not device.startswith("cuda"):
        return "CPU"
    gb = 1024**3
    free, total = torch.cuda.mem_get_info(device)
    return (
        f"{device}: VRAM процесса {torch.cuda.memory_allocated(device) / gb:.2f} ГБ, "
        f"пик {torch.cuda.max_memory_allocated(device) / gb:.2f} ГБ; "
        f"свободно на GPU {free / gb:.2f} из {total / gb:.2f} ГБ"
    )


class OCREngine:
    """Одна копия модели на одном устройстве. Не потокобезопасна: страницы ей подаёт планировщик по одной."""

    def __init__(self, device: str | None = None):
        started = time.time()
        # Визуальный энкодер оставляем в bf16; lm_head, связанный с эмбеддингами, квантовать нельзя
        skip = ["visual"]
        if AutoConfig.from_pretrained(config.MODEL_PATH).tie_word_embeddings:
            skip.append("lm_head")
        quant = None
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        use_cuda = self.device.startswith("cuda")
        if not use_cuda:
            # bitsandbytes 4/8-bit работает только на GPU; без GPU модель грузится в bf16 (медленно)
            log.warning("GPU не найден: работа на CPU, без квантизации (медленно)")
        elif config.QUANT == "4bit":
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=skip,
            )
        elif config.QUANT == "8bit":
            quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)

        name = torch.cuda.get_device_name(self.device) if use_cuda else "CPU"
        log.info(
            "Загрузка модели %s: квантизация=%s, устройство=%s (%s), torch=%s",
            config.MODEL_PATH, config.QUANT if use_cuda else "none", self.device, name, torch.__version__,
        )
        log.info(
            "Параметры: DPI=%s, TILE_WIDTH=%s, MAX_PIXELS=%s, MAX_NEW_TOKENS=%s, "
            "repetition_penalty=%s (повтор %s)",
            config.DPI, config.TILE_WIDTH, config.MAX_PIXELS, config.MAX_NEW_TOKENS,
            config.REPETITION_PENALTY, config.RETRY_REPETITION_PENALTY,
        )
        self.processor = AutoProcessor.from_pretrained(config.MODEL_PATH)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.MODEL_PATH,
            dtype=torch.bfloat16,
            device_map={"": self.device},
            quantization_config=quant,
        ).eval()
        log.info("Модель загружена за %.1f с; %s", time.time() - started, _vram(self.device))

    @torch.inference_mode()
    def recognize_page(self, page: Image.Image, number: int = 1) -> Page:
        started = time.time()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

        landscape = self._is_sideways(page, number)
        if landscape:
            # Альбомные страницы в образцах подшиты так, что текст идёт снизу вверх:
            # поворачиваем на 90° по часовой стрелке
            page = page.rotate(270, expand=True)

        strips = split_strips(page)
        log.info(
            "Стр. %d: %dx%d px, полос %d (%s)", number, page.width, page.height, len(strips),
            ", ".join(f"{s.width}x{s.height}" for s in strips),
        )

        parts: list[str] = []
        header: list[str] | None = None  # шапка таблицы, если предыдущая полоса ею закончилась
        for i, strip in enumerate(strips, start=1):
            label = f"Стр. {number}, полоса {i}/{len(strips)}"
            prompt = config.PROMPT
            if header:
                prompt += config.CONTINUATION_PROMPT.format(header=" | ".join(header), n=len(header))
                log.info("%s: продолжение таблицы из %d столбцов: %s", label, len(header), " | ".join(header))
            text = self._recognize(strip, prompt, label)
            if not _has_content(text):
                log.info("%s: отброшена — нет текста, только «[неразборчиво]»/разметка", label)
                continue
            if header:
                text = strip_repeated_header(text, header)
            text = expand_br_rows(text)
            deduped = drop_duplicated_lines(text)
            if deduped != text:
                log.info("%s: удалены строки, дублирующие строки таблицы (%d → %d строк)", label,
                         len(text.splitlines()), len(deduped.splitlines()))
                text = deduped
            parts.append(text)
            header = trailing_table_header(text, header)

        # Освобождаем кеш аллокатора: иначе фрагментация на 8 ГБ GPU уводит память в shared RAM
        if self.device.startswith("cuda"):
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        log.info(
            "Стр. %d готова за %.1f с: %d символов, альбомная=%s; %s",
            number, time.time() - started, sum(map(len, parts)), landscape, _vram(self.device),
        )
        return Page(text="\n".join(parts), landscape=landscape)

    def _inputs(self, image: Image.Image, prompt: str = config.PROMPT):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        return self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

    def _is_sideways(self, page: Image.Image, number: int) -> bool:
        """Страница повёрнута на 90°? Геометрия строк, а в неясных случаях — уверенность модели."""
        ratio = horizontal_text_ratio(page)
        if ratio >= 1.5:
            log.info("Стр. %d: ориентация книжная (соотношение строк %.2f ≥ 1.5)", number, ratio)
            return False
        if ratio <= 0.5:
            log.info("Стр. %d: страница повёрнута (соотношение строк %.2f ≤ 0.5)", number, ratio)
            return True
        scale = min(1.0, math.sqrt(400_000 / (page.width * page.height)))
        small = page.resize((int(page.width * scale), int(page.height * scale)), Image.LANCZOS)
        upright = self._read_confidence(small)
        rotated = self._read_confidence(small.rotate(270, expand=True))
        log.info(
            "Стр. %d: ориентация неясна (соотношение %.2f), уверенность модели: как есть %.3f, "
            "с поворотом %.3f → %s",
            number, ratio, upright, rotated, "поворот" if rotated > upright else "как есть",
        )
        return rotated > upright

    def _read_confidence(self, image: Image.Image) -> float:
        """Средняя лог-вероятность первых токенов распознавания."""
        out = self.model.generate(
            **self._inputs(image),
            max_new_tokens=48,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        return sum(s.log_softmax(-1).max().item() for s in out.scores) / len(out.scores)

    def _recognize(self, image: Image.Image, prompt: str, label: str) -> str:
        inputs = self._inputs(image, prompt)
        log.debug("%s: изображение %dx%d, токенов на входе %d", label, image.width, image.height,
                  inputs["input_ids"].shape[1])
        text, looped = self._generate(inputs, config.REPETITION_PENALTY, label)
        if looped:
            # Модель зациклилась и не дописала текст — повторяем с более сильным штрафом за повторы
            log.warning("%s: зацикливание, повтор с repetition_penalty=%s", label, config.RETRY_REPETITION_PENALTY)
            retry, _ = self._generate(inputs, config.RETRY_REPETITION_PENALTY, label)
            log.info("%s: длина ответа без повтора %d, с повтором %d — берём %s", label, len(text), len(retry),
                     "повтор" if len(retry) > len(text) else "первый")
            text = max(text, retry, key=len)
        return text

    def _generate(self, inputs, repetition_penalty: float, label: str) -> tuple[str, bool]:
        prompt_len = inputs["input_ids"].shape[1]
        stopper = LoopStopper(self.processor.tokenizer, prompt_len)
        started = time.time()
        out = self.model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )
        elapsed = time.time() - started
        new_tokens = out.shape[1] - prompt_len
        raw = self.processor.decode(out[0, prompt_len:], skip_special_tokens=True)
        log.info(
            "%s: %d токенов на входе, %d сгенерировано за %.1f с (%.1f ток/с)%s%s",
            label, prompt_len, new_tokens, elapsed, new_tokens / max(elapsed, 1e-6),
            ", ЗАЦИКЛИВАНИЕ" if stopper.fired else "",
            ", достигнут лимит MAX_NEW_TOKENS" if new_tokens >= config.MAX_NEW_TOKENS else "",
        )
        log.debug("%s: ответ модели:\n%s", label, raw)
        return clean_text(raw), stopper.fired


# Фрагмент длиной 2–40 символов, повторённый подряд 6+ раз, или символ 20+ раз (зацикливание модели)
_LOOP_RE = re.compile(r"(.{2,40}?)\1{5,}|(.)\2{19,}", re.DOTALL)
# Линии для заполнения: ____, _ _ _, ....., -----
_FILL_RE = re.compile(r"(?:[_.\-…]\s?){4,}")
# Несколько «[неразборчиво]» подряд (в т.ч. недописанные) схлопываем в одно
_UNREADABLE_RE = re.compile(r"(?:\[неразборчиво\]?\s*){2,}")


class LoopStopper(StoppingCriteria):
    """Останавливает генерацию, если конец текста зациклился."""

    def __init__(self, tokenizer, prompt_len: int, check_every: int = 16):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.check_every = check_every
        self.fired = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        n = input_ids.shape[1] - self.prompt_len
        if n == 0 or n % self.check_every:
            return False
        # Пробелы и переводы строк нормализуем: модель часто повторяет фрагмент с разными разделителями
        tail = " ".join(self.tokenizer.decode(input_ids[0, -120:], skip_special_tokens=True).split())
        words = tail.split()
        # Последние 30 слов из 3 и менее разных — зацикливание вида «[неразборчиво] [неразборчиво] ...»
        if len(words) >= 30 and len(set(words[-30:])) <= 3:
            self.fired = True
            return True
        # Повтор должен доходить до конца текста (допускаем недописанный последний фрагмент)
        for m in _LOOP_RE.finditer(tail):
            if len(tail) - m.end() <= len(m.group(1) or m.group(2)):
                self.fired = True
                return True
        return False


def _has_content(text: str) -> bool:
    """Есть ли в ответе текст, а не только «[неразборчиво]» и разметка пустой таблицы."""
    return bool(_NOISE_RE.sub("", text))


_NOISE_RE = re.compile(r"\[неразборчиво\]?|<br>|[|#\-_:\s]")


def clean_text(text: str) -> str:
    # Линии для заполнения схлопываем только вне таблиц: в таблице «----» — это разделитель шапки
    text = "\n".join(line if is_table_row(line) else _FILL_RE.sub("___ ", line) for line in text.splitlines())
    text = _UNREADABLE_RE.sub("[неразборчиво]\n", text)
    text = _LOOP_RE.sub(lambda m: m.group(1) or m.group(2), text)
    return "\n".join(line.rstrip() for line in text.strip().splitlines())
