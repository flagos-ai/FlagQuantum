"""One-shot, bounded launcher for the six FTS-02 CPU measurements."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from tools import fts02_native_boundary, run_with_watchdog
from tools.fts02_process_budget import (
    POINTS,
    CreationBudget,
    Reservation,
    is_git_provenance,
    isolation_identity,
    runtime_identity,
)

ROOT = Path(__file__).resolve().parents[1]
CAPACITY = 32 * 1024 * 1024
RESERVE = 1024 * 1024
RECEIPT_RESERVE = 64 * 1024
GIT_COMMANDS = (
    ("git", "rev-parse", "HEAD"),
    ("git", "status", "--porcelain", "--untracked-files=no"),
)


class Evidence:
    """Charge bytes before writes and reserve space for final records."""

    def __init__(self, reservation: Reservation | None = None) -> None:
        self.reservation = reservation
        self.written = 0
        self.diagnostics_truncated = False

    def write(self, path: Path, data: bytes, *, diagnostic: bool = False) -> None:
        limit = CAPACITY - (RECEIPT_RESERVE if diagnostic else RESERVE)
        if self.written + len(data) > limit:
            if not diagnostic:
                raise RuntimeError("FTS-02 evidence capacity exhausted")
            self.diagnostics_truncated = True
            data = data[: max(0, limit - self.written)]
        if self.reservation is not None:
            self.reservation.update(bytes_count=len(data), byte_limit=limit - 64 * 1024)
        self.written += len(data)
        with path.open("wb" if diagnostic else "ab") as stream:
            stream.write(data)


def command(point: str) -> tuple[str, list[str]]:
    if point in ("w2-x", "w2-z"):
        return "examples/qec/stim_pymatching_baseline.py", [
            "--code",
            "surface",
            "--distance",
            "9",
            "--basis",
            point[-1],
            "--shots",
            "4096",
            "--batch-size",
            "32",
        ]
    if point == "w3":
        return "benchmarks/qec_clifford_dem_construction.py", [
            "--distance",
            "9",
            "--rounds",
            "5",
        ]
    if point in POINTS:
        return "flagquantum.benchmarking", [
            "run",
            "qec_memory_sampling",
            "--shots",
            "16384",
            "--batch-size",
            point.split("-")[1],
            "--warmup",
            "3",
            "--iterations",
            "10",
        ]
    raise ValueError(f"unknown point: {point}")


def git_provenance(
    budget: CreationBudget, expected: str, reservation: Reservation | None = None
) -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_LAZY_FETCH": "1",
    }
    outputs = []
    for _, *args in GIT_COMMANDS:
        if reservation is not None:
            reservation.charge(budget.point, "provenance")
        budget.charge("provenance")
        outputs.append(
            subprocess.check_output(
                ["/usr/bin/git", "-c", "core.fsmonitor=false", *args],
                cwd=ROOT,
                env=env,
                timeout=10,
                text=True,
                preexec_fn=fts02_native_boundary.install,
            ).strip()
        )
    if outputs != [expected, ""]:
        raise RuntimeError("FTS-02 requires the pinned clean tracked source")


def target(point: str, revision: str) -> None:
    fts02_native_boundary.install()
    entry, args = command(point)
    expected = dict(zip(GIT_COMMANDS, (revision + "\n", ""), strict=True))

    def provenance(argv: object, *, cwd: Path, text: bool, timeout: int = 10) -> str:
        if cwd != ROOT or not text or not is_git_provenance(argv):
            raise RuntimeError("unexpected target provenance request")
        assert isinstance(argv, (tuple, list))
        return expected.pop(tuple(argv))

    context = (
        patch.object(subprocess, "check_output", provenance)
        if point not in ("w2-x", "w2-z")
        else nullcontext()
    )
    sys.argv = [entry, *args]
    with context:
        if entry.endswith(".py"):
            runpy.run_path(str(ROOT / entry), run_name="__main__")
        else:
            runpy.run_module(entry, run_name="__main__")


def run_campaign(output: Path, revision: str, reservation: Reservation) -> int:
    reservation.claim("campaign")
    if runtime_identity() != reservation.environment:
        reservation.update(flag="closed")
        raise RuntimeError("approved dependency lock mismatch")
    try:
        reservation.verify_isolation(isolation_identity(reservation.directory.parent))
    except (OSError, ValueError, RuntimeError):
        reservation.update(flag="closed")
        raise
    if output.parent != reservation.directory:
        raise ValueError("campaign output must share its reservation volume")
    output.mkdir()
    evidence = Evidence(reservation)
    budgets = {point: CreationBudget(point) for point in POINTS}
    current = None
    status = "failed"
    try:
        # Bind source before W2, which has no provenance subprocess of its own.
        def charge(point: str, role: str) -> None:
            reservation.charge(point, role)
            budgets[point].charge(role)

        def provenance(point: str) -> None:
            git_provenance(budgets[point], revision, reservation)

        provenance("w3")
        for point, budget in budgets.items():
            current = point
            directory = output / point
            directory.mkdir()
            if point.startswith("w6-"):
                provenance(point)
            seconds = "300" if point == "w3" else "120"

            def charge_target(role: str, point: str = point) -> None:
                charge(point, role)

            def emit_target(data: bytes, directory: Path = directory) -> None:
                evidence.write(directory / "target.log", data)

            result = run_with_watchdog.run(
                [
                    "--phase",
                    point,
                    "--stall-seconds",
                    seconds,
                    "--timeout-seconds",
                    seconds,
                    "--rss-limit-bytes",
                    "2147483648",
                    "--diagnostics",
                    str(directory),
                    "--",
                    sys.executable,
                    "-B",
                    "-m",
                    "tools.launch_fts02_cpu",
                    "--reservation",
                    str(reservation.directory),
                    "--target",
                    point,
                    "--revision",
                    revision,
                ],
                charge=charge_target,
                setup=fts02_native_boundary.install,
                emit=emit_target,
                write=lambda path, data: evidence.write(path, data, diagnostic=True),
            )
            if result or evidence.diagnostics_truncated:
                return result or 125
        status = "completed"
        return 0
    finally:
        # Diagnostics have their own ceiling below, leaving room for this record.
        record = {
            "status": status,
            "last_point": current,
            "revision": revision,
            "coordinator_attempts": 1,
            "reservation": reservation.summary(),
            "diagnostics_truncated": evidence.diagnostics_truncated,
            "evidence_bytes_charged_before_receipt": evidence.written,
            "point_attempts": {
                point: budget.snapshot() for point, budget in budgets.items()
            },
        }
        payload = json.dumps(record).encode()
        if len(payload) > RECEIPT_RESERVE:
            raise RuntimeError("FTS-02 receipt exceeds reserved capacity")
        reservation.update(bytes_count=len(payload), byte_limit=CAPACITY - 64 * 1024)
        (output / "receipt.json").write_bytes(payload)
        reservation.update(flag="closed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", choices=POINTS)
    group.add_argument("--output", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux" or Path.cwd() != ROOT:
        parser.error("requires Linux and the source checkout as working directory")
    if len(args.revision) != 40 or any(
        c not in "0123456789abcdef" for c in args.revision
    ):
        parser.error("requires a full source commit SHA")
    environment = runtime_identity()
    reservation = Reservation(args.reservation.resolve(), args.revision, environment)
    if args.target:
        # The parent charged this target before Popen; no target-side children are allowed.
        reservation._read()
        target(args.target, args.revision)
        return 0
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    return run_campaign(args.output.resolve(), args.revision, reservation)


if __name__ == "__main__":
    raise SystemExit(main())
