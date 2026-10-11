import math

import cv2
import numpy as np
import pymupdf
from PIL import Image

from app import config


def _dark_mask(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L")) < 128


def crop_to_content(img: Image.Image, pad: float = 0.02) -> Image.Image:
    """Обрезает пустые поля скана, чтобы модель получила текст в большем разрешении."""
    dark = _dark_mask(img)
    h, w = dark.shape
    # Строки/столбцы с заметным количеством тёмных пикселей (игнорируем одиночные точки шума)
    rows = np.flatnonzero(dark.sum(axis=1) > max(3, w * 0.003))
    cols = np.flatnonzero(dark.sum(axis=0) > max(3, h * 0.003))
    if rows.size == 0 or cols.size == 0:
        return img
    py, px = int(h * pad), int(w * pad)
    box = (
        max(cols[0] - px, 0),
        max(rows[0] - py, 0),
        min(cols[-1] + px, w),
        min(rows[-1] + py, h),
    )
    return img.crop(box)


def horizontal_text_ratio(img: Image.Image) -> float:
    """Отношение «горизонтальных» строк текста к «вертикальным».

    Линии таблиц удаляются, символы размазываются вдоль одной оси и считается суммарная длина
    вытянутых пятен. >1 — текст идёт горизонтально, <1 — страница повёрнута на 90°.
    """
    gray = np.asarray(img.convert("L"))
    scale = 1000 / max(gray.shape)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ink = (gray < 160).astype(np.uint8) * 255
    h, w = ink.shape
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1)))
    rules |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 15)))
    text = cv2.subtract(ink, cv2.dilate(rules, np.ones((3, 3), np.uint8)))

    def elongated(kernel: np.ndarray, horizontal: bool) -> int:
        _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(text, kernel))
        ww, hh = stats[1:, 2], stats[1:, 3]
        if horizontal:
            return int(ww[(ww > 4 * hh) & (ww > 20)].sum())
        return int(hh[(hh > 4 * ww) & (hh > 20)].sum())

    return (elongated(np.ones((1, 9), np.uint8), True) + 1) / (elongated(np.ones((9, 1), np.uint8), False) + 1)


def split_strips(img: Image.Image) -> list[Image.Image]:
    """Масштабирует страницу до TILE_WIDTH по ширине и режет на горизонтальные полосы
    не больше MAX_PIXELS. Разрезы ставятся в самые «пустые» строки пикселей, чтобы не резать текст."""
    scale = min(1.0, config.TILE_WIDTH / img.width)
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    if img.width * img.height <= config.MAX_PIXELS:
        return [img]

    strip_h = config.MAX_PIXELS // img.width
    # Запас 15%: разрез может сместиться от целевой позиции к пустой строке
    n = math.ceil(img.height / (strip_h * 0.85))
    ink = _dark_mask(img).sum(axis=1)
    cuts = [0]
    for i in range(1, n):
        target = img.height * i // n
        lo = max(target - strip_h // 6, cuts[-1] + strip_h // 4)
        hi = min(target + strip_h // 6, cuts[-1] + strip_h, img.height - 1)
        cuts.append(lo + int(np.argmin(ink[lo:hi + 1])) if hi > lo else target)
    cuts.append(img.height)
    # Полосы без текста (поля, одиночные точки шума) не отправляем в модель —
    # на них она выдумывает «[неразборчиво]»
    text_rows = ink > max(3, img.width * 0.003)
    return [
        img.crop((0, top, img.width, bottom))
        for top, bottom in zip(cuts, cuts[1:])
        if text_rows[top:bottom].sum() >= 10
    ]


def render_pages(pdf_bytes: bytes) -> list[Image.Image]:
    """Рендерит страницы PDF и обрезает пустые поля."""
    pages = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=config.DPI, colorspace=pymupdf.csRGB)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(crop_to_content(img))
    return pages
