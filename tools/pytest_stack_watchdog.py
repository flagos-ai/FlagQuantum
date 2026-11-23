"""Print delayed CI stack snapshots without native lock-free frame traversal."""

from __future__ import annotations

import math
import os
import sys
import threading
import traceback
from collections.abc import Generator
from typing import TextIO

import pytest

OUTPUT: pytest.StashKey[TextIO] = pytest.StashKey()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--fq-stack-dump-after",
        type=float,
        default=0.0,
        help="Print Python thread stacks after this many seconds per test (0 disables).",
    )


def pytest_configure(config: pytest.Config) -> None:
    seconds = config.getoption("--fq-stack-dump-after")
    if not math.isfinite(seconds) or seconds < 0:
        raise pytest.UsageError("--fq-stack-dump-after must be finite and nonnegative")
    if not seconds:
        return
    if (
        config.pluginmanager.hasplugin("faulthandler")
        and float(config.getini("faulthandler_timeout") or 0.0) > 0
    ):
        raise pytest.UsageError(
            "--fq-stack-dump-after requires faulthandler_timeout=0; "
            "do not run both delayed stack dump mechanisms"
        )
    # xdist's stderr proxy has no file descriptor. Preserve the real destination
    # before per-test capture or user fixtures replace/close the stream.
    try:
        descriptor = sys.stderr.fileno()
    except (AttributeError, ValueError):
        descriptor = -1
    if descriptor < 0:
        if sys.__stderr__ is None:
            raise pytest.UsageError("The stack watchdog requires a stderr descriptor")
        descriptor = sys.__stderr__.fileno()
    config.stash[OUTPUT] = os.fdopen(os.dup(descriptor), "w", encoding="utf-8")


def pytest_unconfigure(config: pytest.Config) -> None:
    if OUTPUT in config.stash:
        config.stash[OUTPUT].close()
        del config.stash[OUTPUT]


def _dump_stacks(nodeid: str, seconds: float, output: TextIO) -> None:
    # The returned dictionary owns frame references. Python formats them under
    # normal interpreter synchronization, unlike faulthandler's native watchdog.
    frames = sys._current_frames()
    try:
        print(f"Python stack snapshot after {seconds:g}s: {nodeid}", file=output)
        for identifier, frame in sorted(frames.items()):
            print(f"Thread {identifier} (most recent call last):", file=output)
            traceback.print_stack(frame, limit=100, file=output)
        output.flush()
    finally:
        frames.clear()


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, None, None]:
    seconds = item.config.getoption("--fq-stack-dump-after")
    if not seconds:
        yield
        return
    watchdog = threading.Timer(
        seconds, _dump_stacks, args=(item.nodeid, seconds, item.config.stash[OUTPUT])
    )
    watchdog.name = "fq-stack-watchdog"
    watchdog.daemon = True
    watchdog.start()
    try:
        yield
    finally:
        watchdog.cancel()
        watchdog.join(timeout=5)
        if watchdog.is_alive():
            pytest.exit(
                "The CI stack watchdog did not stop within 5 seconds", returncode=3
            )
