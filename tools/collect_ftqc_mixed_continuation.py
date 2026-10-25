"""Run only the separately approved seven-point continuation after controller review.

This private developer tool never edits the original campaign. The unchanged
watchdog owns workload cleanup and RSS sampling; this controller owns the whole
wall/storage budget and the new attempt ledger. No retry is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import signal
import subprocess
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SOURCE = "5cd642a69a0142b7c7e42013851c38dee701a3d5"
ORIGINAL_MANIFEST = "8e19e5ca2fc45467430bcf9eaf641d500d620b8163104a0deb1c697616fa2b3a"
CASE_SECONDS = 120.0
TOTAL_SECONDS = 900.0
RSS_BYTES = 2 * 1024**3
STORAGE_BYTES = 128 * 1024**2
FREE_BYTES = 512 * 1024**2
HEADROOM_BYTES = 2 * 1024**2


@dataclass
class Attempt:
    """One new attempt linked to an immutable original slot."""

    original_slot: int
    n: int
    t: int
    topology: str
    mode: str
    repetition: int
    status: str = "not_attempted"
    evidence: dict[str, Any] | None = None

    @property
    def label(self) -> str:
        return f"n{self.n}-t{self.t}-{self.topology}-{self.mode}-{self.repetition}"


def matrix() -> list[Attempt]:
    """Return exactly the failed warm-up and 27 previously unattempted slots."""
    points = [(8, 4, "ring", "tensor_network")]
    points.extend(
        (8, 8, topology, mode)
        for topology, mode in itertools.product(
            ("chain", "ring"), ("statevector", "mps", "tensor_network")
        )
    )
    return [
        Attempt(164 + index * 4 + repeat, *point, repeat)
        for index, point in enumerate(points)
        for repeat in range(4)
    ]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def retained_bytes(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def assess(directory: Path, returncode: int, attempt: Attempt | None) -> dict[str, Any]:
    """Retain supervision first; never parse a producer after supervision fails."""
    producer = directory / "producer.json"
    watchdog = directory / "supervisor/watchdog.json"
    result: dict[str, Any] = {
        "status": "failed",
        "failure_domain": "supervision",
        "returncode": returncode,
        "producer_state": "available" if producer.exists() else "absent",
    }
    if not watchdog.is_file():
        result["diagnostic"] = "Watchdog report absent"
        return result
    result["watchdog_sha256"] = digest(watchdog)
    try:
        report = read_json(watchdog)
    except (ValueError, OSError) as error:
        result["diagnostic"] = f"Invalid watchdog report: {error}"
        return result
    result["watchdog"] = report  # Preserve reason/detail, returns, RSS and cleanup.
    if (
        report.get("reason") != "completed"
        or report.get("command_returncode") != returncode
        or returncode not in (0, 1)
        or not positive(report.get("rss_sample_count"))
        or not bounded(report.get("sampled_group_rss_max_bytes"), RSS_BYTES)
        or not bounded(report.get("rss_sample_gap_max_seconds"), 0.1)
    ):
        result["diagnostic"] = "Supervision or command admission failed"
        return result
    if attempt is None:
        if returncode != 0:
            result["diagnostic"] = "Probe command failed"
            return result
        result.update(status="completed", failure_domain=None)
        return result
    result["failure_domain"] = "producer"
    try:
        result["producer_sha256"] = digest(producer)
        data = read_json(producer)
        if (
            data.get("n"),
            data.get("authored_t_count"),
            data.get("topology"),
            data.get("requested_mode"),
        ) != (attempt.n, attempt.t, attempt.topology, attempt.mode):
            raise ValueError("Producer changed workload or requested route")
        if data.get("status") == "refused" and returncode == 0:
            if not data.get("diagnostic"):
                raise ValueError("Capability refusal omitted diagnostic")
            result.update(status="refused", diagnostic=data["diagnostic"])
            return result
        errors = data.get("errors")
        if (
            returncode != 0
            or data.get("status") != "completed"
            or not isinstance(errors, dict)
            or len(errors) != 4
            or not all(bounded(value, 1e-12) for value in errors.values())
            or data.get("performance_targets_enabled") is not False
        ):
            raise ValueError("Independent numerical comparison or target policy failed")
    except (ValueError, OSError) as error:
        result["diagnostic"] = f"Producer report failed: {error}"
        return result
    result.update(status="completed", failure_domain=None)
    return result


def bounded(value: object, limit: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= limit
    )


def positive(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def stop(process: subprocess.Popen[bytes]) -> None:
    """Cancel the watchdog first so it can terminate its owned command group."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError(
                "Watchdog did not complete owned-group cleanup"
            ) from error


def check_budget(original: Path, output: Path, deadline: float, reserve: float) -> None:
    if time.monotonic() + reserve >= deadline:
        raise TimeoutError("Insufficient continuation wall budget including cleanup")
    if (
        retained_bytes(original) + retained_bytes(output)
        > STORAGE_BYTES - HEADROOM_BYTES
    ):
        raise OSError("Combined evidence storage headroom exhausted")


def execute(
    command: list[str], directory: Path, original: Path, output: Path, deadline: float
) -> tuple[int, str | None]:
    """Bound one watchdog invocation, preserving diagnostics after cancellation."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE")
    }
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.monotonic()
    error: str | None = None
    with (
        (directory / "stdout.log").open("wb") as stdout,
        (directory / "stderr.log").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            command, cwd="/private/tmp", env=env, stdout=stdout, stderr=stderr
        )
        try:
            while process.poll() is None:
                check_budget(
                    original, output, min(deadline, started + CASE_SECONDS), 20
                )
                time.sleep(0.02)
        except BaseException as cause:  # Process boundary retains cancellation cause.
            error = f"{type(cause).__name__}: {cause}"
        finally:
            stop(process)
    assert process.returncode is not None
    return process.returncode, error


def command(
    python: Path, watchdog: Path, directory: Path, attempt: Attempt | None
) -> list[str]:
    payload = (
        [str(python), "-c", "import time; time.sleep(0.25)"]
        if attempt is None
        else [
            str(python),
            "-m",
            "flagquantum.benchmarking",
            "run",
            "ftqc_mixed_reference",
            "--n",
            str(attempt.n),
            "--t",
            str(attempt.t),
            "--topology",
            attempt.topology,
            "--mode",
            attempt.mode,
            "--json-output",
            str(directory / "producer.json"),
        ]
    )
    return [
        str(python),
        str(watchdog),
        "--phase",
        "continuation-probe" if attempt is None else attempt.label,
        "--timeout-seconds",
        str(CASE_SECONDS),
        "--stall-seconds",
        str(CASE_SECONDS),
        "--rss-limit-bytes",
        str(RSS_BYTES),
        "--diagnostics",
        str(directory / "supervisor"),
        "--",
        *payload,
    ]


def admission(root: Path, watchdog: Path, deadline: float) -> dict[str, Any]:
    """Verify retained producer artifacts without running a study circuit."""
    original = root / "fts10-cpu-observations-20261025"
    manifest = original / "manifest.json"
    if digest(manifest) != ORIGINAL_MANIFEST:
        raise ValueError("Original manifest identity changed")
    old = read_json(manifest)
    if old["source_revision"] != SOURCE or digest(watchdog) != old["watchdog_sha256"]:
        raise ValueError("Original producer or unchanged watchdog identity changed")
    if digest(root / "collect_fts10_cpu.py") != old["controller_sha256"]:
        raise ValueError("Original controller identity changed")
    if len(old["cases"]) != 192:
        raise ValueError("Original matrix incomplete")
    for attempt in matrix():
        previous = old["cases"][attempt.original_slot]
        expected = "failed" if attempt.original_slot == 164 else "not_attempted"
        if previous["status"] != expected or (
            previous["n"],
            previous["t"],
            previous["topology"],
            previous["mode"],
            previous["repetition"],
        ) != (attempt.n, attempt.t, attempt.topology, attempt.mode, attempt.repetition):
            raise ValueError(
                "Continuation attempted to replace a completed observation"
            )
    install = read_json(root / "pr848-main-install-acceptance.json")
    python = root / "pr848-main-install-py312/bin/python"
    wheel = Path(install["wheel"]["path"])
    if (
        install["source_commit"] != SOURCE
        or install["wheel"] != old["installed_wheel"]
        or digest(wheel) != old["installed_wheel"]["sha256"]
    ):
        raise ValueError("Retained wheel admission changed")
    code = "import json,sys,importlib.metadata,flagquantum; print(json.dumps([sys.version_info[:3],importlib.metadata.version('torch'),flagquantum.__file__]))"
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("PYTHON")
    }
    runtime = json.loads(
        subprocess.check_output(
            [str(python), "-c", code],
            cwd="/private/tmp",
            env=env,
            text=True,
            timeout=min(20, max(0.01, deadline - time.monotonic())),
        )
    )
    if runtime[:2] != [[3, 12, 14], "2.14.1"]:
        raise ValueError("Retained runtime profile changed")
    package = Path(runtime[2]).parent
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("flagquantum/") and not name.endswith("/"):
                if (package.parent / name).read_bytes() != archive.read(name):
                    raise ValueError(
                        f"Installed producer differs from retained wheel: {name}"
                    )
    cpu = subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
    ).strip()
    if cpu != old["cpu_model"] or cpu != "Apple M5":
        raise ValueError("Original CPU profile changed")
    return {
        "source_revision": SOURCE,
        "original_manifest_sha256": ORIGINAL_MANIFEST,
        "original_controller_sha256": old["controller_sha256"],
        "watchdog_sha256": old["watchdog_sha256"],
        "installed_wheel": install["wheel"],
        "runtime": runtime[:2],
        "cpu_model": cpu,
    }


def campaign(root: Path, output: Path, watchdog: Path) -> dict[str, Any]:
    """Probe once and stop on the first failed/refused attempt; never overwrite."""
    started = time.monotonic()
    if shutil.disk_usage(output.parent).free < FREE_BYTES:
        raise OSError("Startup requires at least 512 MiB free")
    output.mkdir(exist_ok=False)
    deadline = started + TOTAL_SECONDS
    original = root / "fts10-cpu-observations-20261025"
    attempts = matrix()
    manifest: dict[str, Any] = {
        "status": "running",
        "original_manifest_sha256": ORIGINAL_MANIFEST,
        "controller_sha256": digest(Path(__file__)),
        "started_unix_seconds": time.time(),
        "budget": {
            "workload_processes": 28,
            "cumulative_attempts": 193,
            "total_seconds": TOTAL_SECONDS,
            "case_seconds": CASE_SECONDS,
            "rss_bytes": RSS_BYTES,
            "query_timeout_seconds": 0.05,
            "max_sampling_gap_seconds": 0.1,
            "combined_retained_bytes": STORAGE_BYTES,
        },
        "performance_targets_enabled": False,
    }

    def save() -> None:
        manifest["attempts"] = [
            dict(
                asdict(item),
                outer_role="host_warmup" if item.repetition == 0 else "measured",
            )
            for item in attempts
        ]
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["combined_retained_bytes_before_write"] = retained_bytes(
            original
        ) + retained_bytes(output)
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "manifest.json")

    save()
    try:
        check_budget(original, output, deadline, CASE_SECONDS + 20)
        manifest["provenance"] = admission(root, watchdog, deadline)
        for attempt in [None, *attempts]:
            check_budget(original, output, deadline, CASE_SECONDS + 20)
            directory = output / (
                "supervision-probe" if attempt is None else attempt.label
            )
            directory.mkdir()
            if attempt is not None:
                attempt.status = "running"
            save()
            returncode, outer_error = execute(
                command(
                    root / "pr848-main-install-py312/bin/python",
                    watchdog,
                    directory,
                    attempt,
                ),
                directory,
                original,
                output,
                deadline,
            )
            evidence = assess(directory, returncode, attempt)
            if outer_error is not None:
                evidence.update(status="failed", controller_error=outer_error)
            if attempt is None:
                manifest["probe"] = evidence
            else:
                attempt.evidence = evidence
                attempt.status = evidence["status"]
            save()
            if evidence["status"] != "completed":
                raise RuntimeError(
                    "First failure/refusal stops continuation; see retained evidence"
                )
        check_budget(original, output, deadline, 0)
        if digest(original / "manifest.json") != ORIGINAL_MANIFEST:
            raise ValueError("Original manifest changed during continuation")
        manifest["status"] = "completed"
    except BaseException as error:  # Campaign boundary retains all failures and gaps.
        manifest.update(
            status="failed", controller_error=f"{type(error).__name__}: {error}"
        )
        for attempt in attempts:
            if attempt.status == "running":
                attempt.status = "failed"
    finally:
        save()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watchdog", type=Path, required=True)
    parser.add_argument("--reviewed-controller-sha256", required=True)
    args = parser.parse_args()
    if digest(Path(__file__)) != args.reviewed_controller_sha256:
        parser.error(
            "Controller digest must match the separately reviewed implementation"
        )
    root, output, watchdog = (
        path.resolve() for path in (args.evidence_root, args.output, args.watchdog)
    )
    original = root / "fts10-cpu-observations-20261025"
    if output != root / "fts10-cpu-continuation-20261025":
        parser.error(
            "Only the single approved continuation output directory is admitted"
        )
    if output == original or original in output.parents or output in original.parents:
        parser.error("Continuation output must be separate from original evidence")

    def interrupted(signum: int, frame: object) -> None:
        raise InterruptedError(f"Controller received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    result = campaign(root, output, watchdog)
    print(result["status"])
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
