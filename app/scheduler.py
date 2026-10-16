"""Планировщик страниц для одновременных запросов.

Каждый запрос — очередь своих страниц. Рабочие потоки (по одному на реплику модели) берут страницы
из запросов по кругу: документ на 40 страниц не задерживает запрос на 1 страницу дольше, чем на
обработку одной страницы. При отмене запроса (клиент отключился) его оставшиеся страницы снимаются.
"""
import itertools
import threading
import time
from collections import deque
from concurrent.futures import Future

from PIL import Image

from app.docx_export import Page
from app.log import get_logger, request_id
from app.ocr import OCREngine

log = get_logger("queue")


class RequestJob:
    def __init__(self, rid: str, images: list[Image.Image]):
        self.rid = rid
        self.total = len(images)
        self.pending: deque[tuple[int, Image.Image, Future]] = deque()
        self.futures: list[Future] = []
        for number, image in enumerate(images, start=1):
            future: Future = Future()
            self.pending.append((number, image, future))
            self.futures.append(future)
        self.created = time.time()
        self.last_served = 0  # «время» последней выданной страницы (счётчик планировщика); 0 — ещё не обслуживался
        self.done = 0
        self.cancelled = False


class Scheduler:
    def __init__(self, engines: list[OCREngine]):
        self._cond = threading.Condition()
        self._jobs: list[RequestJob] = []  # активные запросы в порядке поступления
        self._tick = 0  # счётчик выданных страниц
        self._busy = 0
        self.workers = len(engines)
        for n, engine in enumerate(engines):
            threading.Thread(target=self._worker, args=(n, engine), name=f"ocr-worker-{n}", daemon=True).start()

    @property
    def active_requests(self) -> int:
        with self._cond:
            return len(self._jobs)

    def stats(self) -> str:
        pending = sum(len(j.pending) for j in self._jobs)
        return (f"активных запросов {len(self._jobs)}, страниц в очереди {pending}, "
                f"занято реплик {self._busy}/{self.workers}")

    def submit(self, rid: str, images: list[Image.Image]) -> RequestJob:
        job = RequestJob(rid, images)
        with self._cond:
            ahead = sum(len(j.pending) for j in self._jobs)
            self._jobs.append(job)
            log.info("Запрос поставлен в очередь: %d стр.; перед ним страниц других запросов %d; %s",
                     job.total, ahead, self.stats())
            self._cond.notify_all()
        return job

    def cancel(self, job: RequestJob) -> None:
        with self._cond:
            if job.cancelled:
                return
            job.cancelled = True
            dropped = len(job.pending)
            for _, _, future in job.pending:
                future.cancel()
            job.pending.clear()
            self._remove(job)
            log.warning("Запрос %s отменён: снято страниц из очереди %d (уже готово %d/%d)",
                        job.rid, dropped, job.done, job.total)

    def _remove(self, job: RequestJob) -> None:
        if job in self._jobs:
            self._jobs.remove(job)

    def _next_page(self):
        """Страница запроса, который дольше всех не обслуживался (вызывается под блокировкой).

        Новые запросы идут раньше уже обслуженных, при равенстве — в порядке поступления:
        страницы разных запросов чередуются по кругу.
        """
        waiting = [job for job in self._jobs if job.pending]
        if not waiting:
            return None
        job = min(waiting, key=lambda j: j.last_served)  # min стабилен: при равенстве — более ранний
        self._tick += 1
        job.last_served = self._tick
        return job, job.pending.popleft()

    def _worker(self, n: int, engine: OCREngine) -> None:
        for _ in itertools.count():
            with self._cond:
                item = self._next_page()
                while item is None:
                    self._cond.wait()
                    item = self._next_page()
                job, (number, image, future) = item
                self._busy += 1
            request_id.set(job.rid)
            if not future.set_running_or_notify_cancel():
                with self._cond:
                    self._busy -= 1
                continue
            log.info("Реплика %d взяла стр. %d/%d (в очереди ждала %.1f с)", n, number, job.total,
                     time.time() - job.created)
            try:
                future.set_result(engine.recognize_page(image, number))
            except Exception as exc:  # ошибка страницы отдаётся запросу, рабочий поток живёт дальше
                log.exception("Ошибка распознавания стр. %d", number)
                future.set_exception(exc)
            finally:
                with self._cond:
                    self._busy -= 1
                    job.done += 1
                    if job.done == job.total or (job.cancelled and not job.pending):
                        self._remove(job)
                    log.info("Готово стр. %d/%d запроса (выполнено %d/%d); %s", number, job.total,
                             job.done, job.total, self.stats())


def collect(job: RequestJob) -> list[Page]:
    return [future.result() for future in job.futures]
