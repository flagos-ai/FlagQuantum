#!/usr/bin/env python3
"""Run a hardware command with phase-aware no-output and hard timeouts."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import FrameType


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
            process_command = (
                (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            )
            stack = (entry / "stack").read_text()
        except (OSError, UnicodeError):
            continue
        if "torchrun" in process_command or "python" in process_command:
            stacks.append(f"pid={entry.name} command={process_command}\n{stack}\n")
    (directory / "rank-stacks.log").write_text(
        "".join(stacks) or "kernel stacks unavailable\n", encoding="utf-8"
    )


def _process_group_rss(group: int | None) -> int:
    """Sum RSS (KiB from POSIX ps) for the owned group; never infer idle on errors."""
    result = subprocess.run(
        ["ps", "-axo", "pgid=,rss="],
        text=True,
        capture_output=True,
        check=True,
        timeout=0.05,
    )
    rows = result.stdout.splitlines()
    if not rows:
        raise ValueError("ps returned no process accounting records")
    total = 0
    for row in rows:
        fields = row.split()
        if len(fields) != 2:
            raise ValueError("ps returned malformed process accounting")
        owner, rss = (int(value) for value in fields)
        if owner < 0 or rss < 0:
            raise ValueError("ps returned invalid process accounting")
        if owner == group:
            total += rss * 1024
    return total


def _cancel(signum: int, frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def _signal_group(group: int, signum: int) -> None:
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        # macOS can report EPERM after TERM removes the last group member.
        # Accept that race only when independent accounting confirms absence.
        try:
            result = subprocess.run(
                ["ps", "-axo", "pgid="],
                text=True,
                capture_output=True,
                check=True,
                timeout=0.05,
            )
            rows = result.stdout.splitlines()
            if not rows or any(len(row.split()) != 1 for row in rows):
                raise ValueError("ps returned missing or malformed process groups")
            groups = [int(row.strip()) for row in rows]
            if any(owner < 0 for owner in groups):
                raise ValueError("ps returned invalid process groups")
        except (OSError, ValueError, subprocess.SubprocessError) as accounting_error:
            raise error from accounting_error
        if group in groups:
            raise


def _stop_group(process: subprocess.Popen[bytes]) -> None:
    # The launcher may exit while same-group children still own its output pipe.
    _signal_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    _signal_group(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--stall-seconds", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--rss-limit-bytes", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or any(
        not math.isfinite(value) or value <= 0
        for value in (args.stall_seconds, args.timeout_seconds)
    ):
        parser.error("positive timeouts and a command are required")
    if args.rss_limit_bytes is not None and args.rss_limit_bytes <= 0:
        parser.error("RSS limit must be positive")
    rss: dict[str, object] = {
        "rss_limit_bytes": args.rss_limit_bytes,
        "sampled_group_rss_max_bytes": 0,
        "rss_sample_count": 0,
        "rss_sample_gap_max_seconds": 0.0,
        "rss_max_allowed_gap_seconds": 0.1,
        "hard_allocation_limit": False,
        "rss_scope": "sum of owned process-group RSS; shared pages may be counted twice",
    }
    if args.rss_limit_bytes is not None:
        try:
            _process_group_rss(None)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            _diagnostics(
                args.diagnostics,
                {
                    **rss,
                    "reason": "rss_monitor_unavailable",
                    "detail": str(error),
                    "command_started": False,
                },
            )
            return 125
    started = last_progress = previous_sample = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    os.set_blocking(process.stdout.fileno(), False)
    previous_handler = None
    if (
        args.rss_limit_bytes is not None
        and threading.current_thread() is threading.main_thread()
    ):
        previous_handler = signal.signal(signal.SIGTERM, _cancel)
    reason = ""
    peak = samples = 0
    recent_samples: deque[tuple[float, int]] = deque(maxlen=32)
    maximum_gap = 0.0
    try:
        while process.poll() is None:
            if args.rss_limit_bytes is not None:
                try:
                    observed = _process_group_rss(process.pid)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    rss["detail"] = str(error)
                    reason = "rss_monitor_unavailable"
                    break
                sampled = time.monotonic()
                recent_samples.append((sampled - started, observed))
                maximum_gap = max(maximum_gap, sampled - previous_sample)
                previous_sample = sampled
                peak = max(peak, observed)
                samples += int(observed > 0)
                if maximum_gap > 0.1:
                    reason = "rss_sampling_gap_exceeded"
                    break
                if observed > args.rss_limit_bytes:
                    reason = "rss_limit_exceeded"
                    break
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
            time.sleep(0.01 if args.rss_limit_bytes is not None else 0.1)
        if not reason and args.rss_limit_bytes is not None:
            # Observe the final interval and refuse children that outlive the launcher.
            try:
                remaining = _process_group_rss(process.pid)
                maximum_gap = max(maximum_gap, time.monotonic() - previous_sample)
                peak = max(peak, remaining)
                if remaining:
                    reason = "rss_group_survived_launcher"
                elif maximum_gap > 0.1:
                    reason = "rss_sampling_gap_exceeded"
                elif not samples:
                    reason = "rss_no_live_sample"
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                rss["detail"] = str(error)
                reason = "rss_monitor_unavailable"
        cleanup_started = time.monotonic()
        if reason:
            _stop_group(process)
        tail = process.stdout.read() or b""
        if tail:
            sys.stdout.buffer.write(tail)
        rss.update(
            sampled_group_rss_max_bytes=peak,
            rss_sample_count=samples,
            rss_sample_gap_max_seconds=maximum_gap,
            rss_recent_samples_seconds_bytes=list(recent_samples),
        )
        payload = {
            "schema": "flagquantum_watchdog_v1",
            "phase": args.phase,
            "reason": reason or "completed",
            "last_operation": " ".join(command),
            "elapsed_seconds": time.monotonic() - started,
            "cleanup_required": bool(reason),
            "command_returncode": process.returncode,
            "termination_elapsed_seconds": (
                time.monotonic() - cleanup_started if reason else 0.0
            ),
            **(rss if args.rss_limit_bytes is not None else {}),
        }
        if reason:
            _diagnostics(args.diagnostics, payload)
            return 124
        if args.rss_limit_bytes is not None:
            args.diagnostics.mkdir(parents=True, exist_ok=True)
            (args.diagnostics / "watchdog.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )
        return int(process.returncode or 0)
    except BaseException as error:
        # Owned process boundary: preserve cancellation evidence, then propagate it.
        cleanup_started = time.monotonic()
        _stop_group(process)
        rss.update(
            sampled_group_rss_max_bytes=peak,
            rss_sample_count=samples,
            rss_sample_gap_max_seconds=maximum_gap,
            rss_recent_samples_seconds_bytes=list(recent_samples),
        )
        _diagnostics(
            args.diagnostics,
            {
                "schema": "flagquantum_watchdog_v1",
                "phase": args.phase,
                "reason": "supervisor_interrupted",
                "exception_type": type(error).__name__,
                "last_operation": " ".join(command),
                "cleanup_required": True,
                "command_returncode": process.returncode,
                "termination_elapsed_seconds": time.monotonic() - cleanup_started,
                **(rss if args.rss_limit_bytes is not None else {}),
            },
        )
        raise
    finally:
        if process.poll() is None or args.rss_limit_bytes is not None:
            _stop_group(process)
        process.stdout.close()
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(run())
