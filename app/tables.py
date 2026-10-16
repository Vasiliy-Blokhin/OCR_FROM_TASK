"""Работа с Markdown-таблицами в ответах модели."""
import re

# Разделитель шапки Markdown-таблицы: |---|:---:|
_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
_BR = re.compile(r"<br\s*/?>")


def is_table_row(line: str) -> bool:
    line = line.strip()
    return line.startswith("|") or line.count("|") >= 2


def split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def is_separator(cells: list[str]) -> bool:
    return any(cells) and all(_SEPARATOR_CELL.match(c) for c in cells if c)


def _join_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def expand_br_rows(text: str) -> str:
    """Делит строку таблицы, в которой модель склеила несколько записей через <br>.

    Если во всех непустых ячейках одинаковое число частей k > 1, это k отдельных записей:
    | 20.12<br>27.12 | ТО-2<br>ТО-2 | Елин<br>Елин |  ->  две строки таблицы.
    """
    out = []
    for line in text.splitlines():
        if not is_table_row(line):
            out.append(line)
            continue
        cells = split_row(line)
        parts = [_BR.split(c) if c else [] for c in cells]
        counts = {len(p) for p in parts if p}
        if len(counts) == 1 and (k := counts.pop()) > 1:
            for i in range(k):
                out.append(_join_row([p[i].strip() if p else "" for p in parts]))
        else:
            out.append(line)
    return "\n".join(out)


def trailing_table_header(text: str, previous: list[str] | None) -> list[str] | None:
    """Шапка таблицы, которой заканчивается текст полосы (None — текст закончился не таблицей).

    Если в хвостовой таблице нет своей шапки (это продолжение), возвращается шапка предыдущей полосы.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not is_table_row(lines[-1]):
        return None
    start = len(lines) - 1
    while start > 0 and is_table_row(lines[start - 1]):
        start -= 1
    block = lines[start:]
    if len(block) >= 2 and is_separator(split_row(block[1])):
        return split_row(block[0])
    return previous or split_row(block[0])


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def drop_duplicated_lines(text: str) -> str:
    """Убирает обычные строки, которые повторяют строку таблицы.

    В продолжении таблицы модель иногда выводит записи дважды: сначала текстом, потом таблицей.
    """
    lines = text.splitlines()
    rows = [_norm("".join(split_row(line))) for line in lines if is_table_row(line)]
    if not rows:
        return text
    out = []
    for line in lines:
        key = _norm(line)
        if not is_table_row(line) and not line.lstrip().startswith("#") and len(key) >= 6 \
                and any(key in row for row in rows):
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def strip_repeated_header(text: str, header: list[str]) -> str:
    """Убирает из продолжения таблицы повтор шапки и строки-разделители |---|."""
    norm = [c.lower() for c in header]
    out = []
    for line in text.splitlines():
        if is_table_row(line):
            cells = split_row(line)
            if is_separator(cells) or [c.lower() for c in cells] == norm:
                continue
        out.append(line)
    return "\n".join(out).strip()
