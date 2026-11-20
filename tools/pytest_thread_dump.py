"""Report long tests without traversing live interpreter frames from native code.

CPython's asynchronous faulthandler watchdog can race with frame destruction.
Taking Python frame references under the GIL keeps them alive while formatting.
Fatal-signal faulthandler support and the CI job's hard timeout remain separate.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import traceback
from collections.abc import Generator

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.getgroup("flagquantum-diagnostics").addoption(
        "--fq-thread-dump-timeout",
        type=float,
        default=0.0,
        help="Report Python thread stacks after this many seconds in one test",
    )


def pytest_configure(config: pytest.Config) -> None:
    timeout = float(config.getoption("--fq-thread-dump-timeout"))
    if not math.isfinite(timeout) or timeout < 0:
        raise pytest.UsageError("thread dump timeout must be finite and non-negative")
    if timeout and float(config.getini("faulthandler_timeout") or 0):
        raise pytest.UsageError(
            "Use only the Python thread watchdog for timed stack dumps"
        )


def _dump_thread_stacks(nodeid: str, timeout: float, descriptor: int) -> None:
    # Own a separate stream: pytest capture and the main thread cannot close it
    # or hold its Python IO lock while this diagnostic is being written.
    with os.fdopen(os.dup(descriptor), "w", encoding="utf-8") as stream:
        print(f"Test exceeded {timeout:g}s: {nodeid}", file=stream)
        for identifier, frame in sys._current_frames().items():
            print(f"Thread {identifier} (most recent call last):", file=stream)
            traceback.print_stack(frame, limit=100, file=stream)


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, object, object]:
    timeout = float(item.config.getoption("--fq-thread-dump-timeout"))
    if not timeout:
        return (yield)
    assert sys.__stderr__ is not None
    descriptor = os.dup(sys.__stderr__.fileno())
    errors: list[Exception] = []

    def report() -> None:
        try:
            _dump_thread_stacks(item.nodeid, timeout, descriptor)
        except Exception as error:
            # Transfer diagnostic-thread failures to pytest's owning thread.
            errors.append(error)

    timer = threading.Timer(timeout, report)
    timer.name = f"thread-dump:{item.nodeid}"
    timer.daemon = True
    timer.start()
    try:
        return (yield)
    finally:
        timer.cancel()
        timer.join(timeout=5)
        os.close(descriptor)
        if timer.is_alive():
            raise RuntimeError("The test thread diagnostic did not finish within 5s")
        if errors:
            raise RuntimeError("The test thread diagnostic failed") from errors[0]
