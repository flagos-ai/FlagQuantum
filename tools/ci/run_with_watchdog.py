#!/usr/bin/env python3
"""Run a hardware command with phase-aware no-output and hard timeouts."""

from __future__ import annotations

import sys as _entry_sys
from pathlib import Path as _EntryPath

if __package__ in (None, ""):
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


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


def _diagnostics(
    directory: Path, payload: dict[str, object], *, group: int | None = None
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "watchdog.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    observations = {"snapshot_error": "command_not_started"}
    if group is not None:
        if not __package__:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from tools.validation.hardware_telemetry import record_sample

        output = directory / "hardware-snapshot.jsonl"
        try:
            record_sample(output, group, time.monotonic())
            observations = json.loads(output.read_text().splitlines()[-1])[
                "observations"
            ]
        except (OSError, ValueError) as error:
            observations = {"snapshot_error": type(error).__name__}
    for name in ("processes", "gpu"):
        retained = {
            key: value
            for key, value in observations.items()
            if key == name or key.startswith((name + "_", "snapshot_"))
        }
        (directory / (name + ".log")).write_text(
            json.dumps(retained, sort_keys=True) + "\n", encoding="utf-8"
        )
    (directory / "rank-stacks.log").write_text(
        "Kernel stacks and process command lines are not collected automatically; "
        "owned numeric process accounting is retained in processes.log.\n",
        encoding="utf-8",
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
    parser.add_argument("--hardware-samples", type=Path)
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
    if args.hardware_samples is not None:
        if args.rss_limit_bytes is not None:
            parser.error(
                "hardware sampling cannot meet the strict RSS sampling cadence"
            )
        if not __package__:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        args.hardware_samples.parent.mkdir(parents=True, exist_ok=True)
        args.hardware_samples.touch(exist_ok=False)
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
                    "detail": type(error).__name__,
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
    next_hardware_sample = started
    try:
        while process.poll() is None:
            if (
                args.hardware_samples is not None
                and time.monotonic() >= next_hardware_sample
            ):
                from tools.validation.hardware_telemetry import record_sample

                record_sample(args.hardware_samples, process.pid, started)
                next_hardware_sample = time.monotonic() + 1.0
            if args.rss_limit_bytes is not None:
                try:
                    observed = _process_group_rss(process.pid)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    rss["detail"] = type(error).__name__
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
                rss["detail"] = type(error).__name__
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
            _diagnostics(args.diagnostics, payload, group=process.pid)
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
            group=process.pid,
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
