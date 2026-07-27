#!/usr/bin/env python3
"""Supervise multiple node launchers and fail the whole job as one unit."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def _terminate(process: subprocess.Popen[bytes], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=grace_seconds)


def run_jobs(
    jobs: list[dict[str, Any]],
    *,
    output_dir: Path,
    timeout_seconds: float,
    poll_seconds: float = 0.1,
    termination_grace_seconds: float = 5.0,
) -> dict[str, Any]:
    if len(jobs) < 2 or timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("at least two jobs and positive timeouts are required")
    names = [str(job.get("name", "")) for job in jobs]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("job names must be non-empty and unique")
    if any(not job.get("command") for job in jobs):
        raise ValueError("every job requires a command")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    processes: dict[str, subprocess.Popen[bytes]] = {}
    streams: dict[str, Any] = {}
    first_failure: str | None = None
    reason = "completed"
    cleanup_results: dict[str, int] = {}
    failure_detected_at: float | None = None
    try:
        for job in jobs:
            name = str(job["name"])
            stream = (output_dir / f"{name}.log").open("wb")
            streams[name] = stream
            processes[name] = subprocess.Popen(
                [str(item) for item in job["command"]],
                cwd=job.get("cwd"),
                env=job.get("env"),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        while True:
            failed = [
                name
                for name, process in processes.items()
                if process.poll() not in {None, 0}
            ]
            if failed:
                first_failure = failed[0]
                reason = "peer_launcher_failure"
                failure_detected_at = time.monotonic()
                break
            if all(process.poll() == 0 for process in processes.values()):
                break
            if time.monotonic() - started > timeout_seconds:
                reason = "job_timeout"
                failure_detected_at = time.monotonic()
                break
            time.sleep(poll_seconds)
        if reason != "completed":
            for process in processes.values():
                _terminate(process, termination_grace_seconds)
            for job in jobs:
                cleanup = job.get("cleanup_command")
                if cleanup:
                    completed = subprocess.run(
                        [str(item) for item in cleanup],
                        timeout=termination_grace_seconds,
                        check=False,
                    )
                    cleanup_results[str(job["name"])] = int(completed.returncode)
    finally:
        for process in processes.values():
            _terminate(process, termination_grace_seconds)
        for stream in streams.values():
            stream.close()
    elapsed = time.monotonic() - started
    failure_detected_seconds = (
        None if failure_detected_at is None else failure_detected_at - started
    )
    cleanup_elapsed_seconds = (
        0.0 if failure_detected_at is None else time.monotonic() - failure_detected_at
    )
    cleanup_failures = sorted(
        name for name, returncode in cleanup_results.items() if returncode != 0
    )
    cleanup_verified = reason == "completed" or (
        len(cleanup_results) == len(jobs) and not cleanup_failures
    )
    payload = {
        "schema": "flagquantum.multinode_watchdog.v1",
        "status": "passed" if reason == "completed" else "failed_closed",
        "reason": reason,
        "first_failure_job": first_failure,
        "elapsed_seconds": elapsed,
        "failure_detected_seconds": failure_detected_seconds,
        "cleanup_elapsed_seconds": cleanup_elapsed_seconds,
        "cleanup_required": reason != "completed",
        "cleanup_results": cleanup_results,
        "cleanup_verified": cleanup_verified,
        "cleanup_failures": cleanup_failures,
        "jobs": [
            {
                "name": name,
                "returncode": int(process.returncode or 0),
                "log": str(output_dir / f"{name}.log"),
            }
            for name, process in processes.items()
        ],
        "all_launchers_exited": all(
            process.poll() is not None for process in processes.values()
        ),
    }
    (output_dir / "watchdog.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--termination-grace-seconds", type=float, default=5.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    payload = run_jobs(
        config["jobs"],
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
        termination_grace_seconds=args.termination_grace_seconds,
    )
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(
        0 if payload["status"] == "passed" else 1 if payload["cleanup_verified"] else 2
    )


if __name__ == "__main__":
    main()
