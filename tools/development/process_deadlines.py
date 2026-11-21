# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/tensor_report_card.py.

"""Run developer report tasks with per-process deadlines and owned cleanup."""

import math
import multiprocessing
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from types import TracebackType
from typing import Literal


@dataclass
class _Pending:
    process: BaseProcess
    recv: Connection
    start_time: float
    kwargs: dict[str, object]
    received: bool = False
    result: object = None


def _stop(process: BaseProcess) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)
    if process.is_alive():
        raise RuntimeError("report task could not be stopped")


class ProcessDeadlineExecutor:
    """Queue one-result pipe tasks and terminate overdue child processes.

    Tasks receive the original keyword arguments plus a sending Connection named
    cxn, and return their result through cxn.send. Each next_result call fills
    available worker slots and returns one completed task's kwargs and payload.
    Nonzero exits, missing payloads and timeouts return None. Use the context
    manager to clean up queued or running tasks when the consumer exits early.

    Like the original, submission retains and later mutates the kwargs mapping.
    Unlike the original, parent sending endpoints are closed after startup,
    received payloads are drained while workers run, and killed workers are
    joined. Returned kwargs retain the now-closed parent cxn endpoint.
    """

    def __init__(
        self,
        timeout: float,
        max_workers: int,
        *,
        start_method: Literal["spawn", "fork", "forkserver"] | None = None,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("task timeout must be finite and positive")
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer")
        self.timeout = timeout
        self.max_workers = max_workers
        self._context = multiprocessing.get_context(start_method)
        self.queued: list[tuple[Callable[..., None], dict[str, object]]] = []
        self.pending: list[_Pending] = []

    @property
    def work_to_be_done(self) -> int:
        """Return the number of running and queued tasks."""
        return len(self.queued) + len(self.pending)

    def submit(self, func: Callable[..., None], kwargs: dict[str, object]) -> None:
        """Queue a task accepting cxn and exactly one pipe result."""
        self.queued.append((func, kwargs))

    def _fill_workers(self) -> None:
        while self.queued and len(self.pending) < self.max_workers:
            func, kwargs = self.queued.pop(0)
            recv, send = self._context.Pipe(duplex=False)
            kwargs["cxn"] = send
            process = self._context.Process(target=func, kwargs=kwargs)
            start = time.monotonic()
            try:
                process.start()
            except BaseException:
                recv.close()
                self.queued.insert(0, (func, kwargs))
                raise
            finally:
                send.close()
            self.pending.append(_Pending(process, recv, start, kwargs))

    def _scan_pending(self) -> _Pending | None:
        for index, task in enumerate(self.pending):
            # Reading before exit avoids the original pipe-buffer deadlock for
            # large payloads. The task protocol permits one complete send.
            if not task.received and task.recv.poll():
                try:
                    task.result = task.recv.recv()
                except (EOFError, OSError):
                    task.result = None
                task.received = True
            if not task.process.is_alive():
                task.process.join(timeout=1)
                return self.pending.pop(index)
            if time.monotonic() - task.start_time > self.timeout:
                _stop(task.process)
                return self.pending.pop(index)
        return None

    def next_result(self) -> tuple[dict[str, object], object]:
        """Return a task outcome, or raise StopIteration when no work remains."""
        self._fill_workers()
        if not self.pending:
            raise StopIteration("no report tasks remain")
        while (finished := self._scan_pending()) is None:
            time.sleep(0.005)
        try:
            # A child can exit between the initial poll and is_alive. Once it
            # is joined, an available complete payload can be read safely.
            if not finished.received and finished.recv.poll():
                try:
                    finished.result = finished.recv.recv()
                except (EOFError, OSError):
                    finished.result = None
            result = finished.result if finished.process.exitcode == 0 else None
        finally:
            finished.recv.close()
        self._fill_workers()
        return finished.kwargs, result

    def close(self) -> None:
        """Discard queued work and terminate and join all owned children."""
        self.queued.clear()
        while self.pending:
            task = self.pending[0]
            _stop(task.process)
            task.recv.close()
            self.pending.pop(0)

    def __enter__(self) -> "ProcessDeadlineExecutor":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
