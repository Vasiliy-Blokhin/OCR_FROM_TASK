#!/bin/sh
# Перезапуск OCR-сервиса этой папки (например, после git pull): sh restart.sh
#
# Останавливает только сервис, запущенный из этой папки: процесс Python, выполняющий main.py,
# у которого рабочая папка — папка проекта. Другие процессы Python и копии сервиса в других
# папках не затрагиваются. Затем запускает сервис заново в фоне и ждёт готовности.
#
# Вывод сервиса: logs/service.out. Таймауты: OCR_STOP_TIMEOUT (60 с), OCR_START_TIMEOUT (900 с).
set -e
cd "$(dirname "$0")"
ROOT=$(pwd -P)
LOG_DIR=${OCR_LOG_DIR:-$ROOT/logs}
OUT="$LOG_DIR/service.out"
STOP_TIMEOUT=${OCR_STOP_TIMEOUT:-60}
START_TIMEOUT=${OCR_START_TIMEOUT:-900}

if [ "$(uname -s)" != "Linux" ]; then
    echo "restart.sh рассчитан на Linux (сейчас: $(uname -s))." >&2
    exit 1
fi

# PID процессов этой папки (рабочая папка = папка проекта), чья командная строка подходит под шаблон
pids_matching() {
    for dir in /proc/[0-9]*; do
        pid=${dir#/proc/}
        [ "$(readlink "$dir/cwd" 2>/dev/null)" = "$ROOT" ] || continue
        cmd=$(tr '\0' ' ' < "$dir/cmdline" 2>/dev/null) || continue
        case "$cmd" in
            $1) echo "$pid" ;;
        esac
    done
}

# Сервис этой папки: Python, выполняющий main.py
service_pids() { pids_matching '*python*[ ]main.py[ ]*'; }
# Сервис или start.sh, который ещё готовит запуск (распаковка встроенного Python)
alive_pids() { service_pids; pids_matching '*sh[ ]start.sh[ ]*'; }

# --- остановка ---
pids=$(service_pids)
if [ -z "$pids" ]; then
    echo "Запущенный сервис в $ROOT не найден — только запуск."
else
    echo "Остановка сервиса (PID $(echo $pids))..."
    kill -TERM $pids 2>/dev/null || true
    waited=0
    while [ -n "$(service_pids)" ] && [ "$waited" -lt "$STOP_TIMEOUT" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    pids=$(service_pids)
    if [ -n "$pids" ]; then
        # Сервис ждёт завершения текущих запросов; если они не успели — останавливаем принудительно
        echo "Сервис не остановился за $STOP_TIMEOUT с — принудительная остановка (PID $(echo $pids))."
        kill -KILL $pids 2>/dev/null || true
        sleep 2
    fi
    echo "Сервис остановлен."
fi

# --- запуск ---
mkdir -p "$LOG_DIR"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') запуск через restart.sh =====" >> "$OUT"
start_line=$(wc -l < "$OUT")
# setsid — чтобы сервис не завершился вместе с терминалом, из которого запущен restart.sh
if command -v setsid >/dev/null 2>&1; then
    setsid nohup sh start.sh >> "$OUT" 2>&1 < /dev/null &
else
    nohup sh start.sh >> "$OUT" 2>&1 < /dev/null &
fi
echo "Запуск... вывод сервиса: $OUT"

waited=0
while [ "$waited" -lt "$START_TIMEOUT" ]; do
    if tail -n +"$((start_line + 1))" "$OUT" | grep -q "Сервис готов принимать запросы"; then
        echo "Сервис запущен (PID $(echo $(service_pids))), порт ${OCR_PORT:-8000}. Готов за $waited с."
        exit 0
    fi
    if [ "$waited" -gt 5 ] && [ -z "$(alive_pids)" ]; then
        echo "Сервис завершился при запуске. Последние строки вывода:" >&2
        tail -n 30 "$OUT" >&2
        exit 1
    fi
    sleep 2
    waited=$((waited + 2))
done
echo "Сервис не сообщил о готовности за $START_TIMEOUT с (идёт установка пакетов или загрузка модели?)." >&2
echo "Следите за выводом: tail -f $OUT" >&2
exit 1
