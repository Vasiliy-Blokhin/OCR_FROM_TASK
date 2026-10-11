import math
import os
import re
import threading

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
from app.pdf import horizontal_text_ratio, split_strips


class OCREngine:
    def __init__(self):
        # Визуальный энкодер оставляем в bf16; lm_head, связанный с эмбеддингами, квантовать нельзя
        skip = ["visual"]
        if AutoConfig.from_pretrained(config.MODEL_PATH).tie_word_embeddings:
            skip.append("lm_head")
        quant = None
        if config.QUANT == "4bit":
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=skip,
            )
        elif config.QUANT == "8bit":
            quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)

        self.processor = AutoProcessor.from_pretrained(config.MODEL_PATH)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            quantization_config=quant,
        ).eval()
        # Модель одна на процесс — запросы к GPU выполняем по очереди
        self._lock = threading.Lock()

    @torch.inference_mode()
    def recognize_page(self, page: Image.Image) -> Page:
        with self._lock:
            landscape = self._is_sideways(page)
            if landscape:
                # Альбомные страницы в образцах подшиты так, что текст идёт снизу вверх:
                # поворачиваем на 90° по часовой стрелке
                page = page.rotate(270, expand=True)
            texts = [self._recognize(strip) for strip in split_strips(page)]
            # Освобождаем кеш аллокатора: иначе фрагментация на 8 ГБ GPU уводит память в shared RAM
            torch.cuda.empty_cache()
        text = "\n".join(t for t in texts if t.replace("[неразборчиво]", "").strip())
        return Page(text=text, landscape=landscape)

    def _inputs(self, image: Image.Image):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": config.PROMPT},
            ],
        }]
        return self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

    def _is_sideways(self, page: Image.Image) -> bool:
        """Страница повёрнута на 90°? Геометрия строк, а в неясных случаях — уверенность модели."""
        ratio = horizontal_text_ratio(page)
        if ratio >= 1.5:
            return False
        if ratio <= 0.5:
            return True
        scale = min(1.0, math.sqrt(400_000 / (page.width * page.height)))
        small = page.resize((int(page.width * scale), int(page.height * scale)), Image.LANCZOS)
        return self._read_confidence(small.rotate(270, expand=True)) > self._read_confidence(small)

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

    def _recognize(self, image: Image.Image) -> str:
        inputs = self._inputs(image)
        text, looped = self._generate(inputs, repetition_penalty=1.05)
        if looped:
            # Модель зациклилась и не дописала текст — повторяем с более сильным штрафом за повторы
            retry, _ = self._generate(inputs, repetition_penalty=1.15)
            text = max(text, retry, key=len)
        return text

    def _generate(self, inputs, repetition_penalty: float) -> tuple[str, bool]:
        prompt_len = inputs["input_ids"].shape[1]
        stopper = LoopStopper(self.processor.tokenizer, prompt_len)
        out = self.model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )
        text = self.processor.decode(out[0, prompt_len:], skip_special_tokens=True)
        return clean_text(text), stopper.fired


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


def clean_text(text: str) -> str:
    text = _FILL_RE.sub("___ ", text)
    text = _UNREADABLE_RE.sub("[неразборчиво]\n", text)
    text = _LOOP_RE.sub(lambda m: m.group(1) or m.group(2), text)
    return "\n".join(line.rstrip() for line in text.strip().splitlines())
