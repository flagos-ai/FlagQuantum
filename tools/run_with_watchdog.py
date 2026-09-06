#!/usr/bin/env python3
"""Run a hardware command with phase-aware no-output and hard timeouts."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _diagnostics(directory: Path, payload: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "watchdog.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    commands = {
        "processes.log": ["ps", "-eo", "pid,ppid,stat,etime,rss,cmd"],
        "gpu.log": [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader",
        ],
    }
    for name, command in commands.items():
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=10)
            text = result.stdout + result.stderr
        except (OSError, subprocess.SubprocessError) as error:
            text = f"diagnostic unavailable: {error}\n"
        (directory / name).write_text(text, encoding="utf-8")
    stacks = []
    proc = Path("/proc")
    entries = proc.iterdir() if proc.is_dir() else ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            stack = (entry / "stack").read_text()
        except (OSError, UnicodeError):
            continue
        if "torchrun" in command or "python" in command:
            stacks.append(f"pid={entry.name} command={command}\n{stack}\n")
    (directory / "rank-stacks.log").write_text(
        "".join(stacks) or "kernel stacks unavailable\n", encoding="utf-8"
    )


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--stall-seconds", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.stall_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("positive timeouts and a command are required")
    started = last_progress = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    os.set_blocking(process.stdout.fileno(), False)
    reason = ""
    while process.poll() is None:
        try:
            chunk = os.read(process.stdout.fileno(), 65536)
        except BlockingIOError:
            chunk = b""
        if chunk:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            last_progress = time.monotonic()
        now = time.monotonic()
        if now - started > args.timeout_seconds:
            reason = "phase_timeout"
            break
        if now - last_progress > args.stall_seconds:
            reason = "no_progress_timeout"
            break
        time.sleep(0.1)
    tail = process.stdout.read() or b""
    if tail:
        sys.stdout.buffer.write(tail)
    if not reason:
        return int(process.returncode or 0)
    _diagnostics(
        args.diagnostics,
        {
            "schema": "flagquantum_watchdog_v1",
            "phase": args.phase,
            "reason": reason,
            "last_operation": " ".join(command),
            "elapsed_seconds": time.monotonic() - started,
            "cleanup_required": True,
        },
    )
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    return 124


if __name__ == "__main__":
    raise SystemExit(run())
