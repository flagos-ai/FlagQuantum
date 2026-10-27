"""Collect the separately approved FTS-10D matrix once, after main admission.

Invoke as python -m tools.collect_ftqc_phase_cost from the reviewed checkout.
CI tests inject synthetic payloads; no live numerical trial or probe is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from types import FrameType
from typing import Any

from tools.observe_ftqc_phase_cost import CASES, ROUTES, atomic, description, expansion
from tools.run_with_watchdog import _process_group_rss, _stop_group

SECONDS = 30.0
CLEANUP_RESERVE = 16.0
TOTAL_SECONDS = 1200.0
RSS_BYTES = 2 * 1024**3
STORAGE_BYTES = 128 * 1024**2
FILE_BYTES = 512 * 1024
FREE_BYTES = 512 * 1024**2


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def schedule() -> list[dict[str, Any]]:
    return [
        {
            "case": case,
            "route": ROUTES[(offset + index + repeat) % 4],
            "repetition": repeat,
            "role": "warmup" if repeat == 0 else "measured",
            "status": "not_attempted",
        }
        for repeat in range(4)
        for index, case in enumerate(CASES)
        for offset in range(4)
    ]


def retained_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def budget(output: Path, deadline: float, reserve: float = SECONDS) -> None:
    if time.monotonic() + reserve >= deadline:
        raise TimeoutError("campaign wall budget cannot admit another attempt")
    if retained_bytes(output) >= STORAGE_BYTES - 4 * 1024**2:
        raise OSError("evidence storage headroom exhausted")
    if shutil.disk_usage(output).free < FREE_BYTES:
        raise OSError("at least 512 MiB free space is required")


def admission(install_report: Path, source: str) -> dict[str, Any]:
    """Inspect retained installation bytes and metadata without a live worker."""
    report = json.loads(install_report.read_text())
    if (
        report["source_commit"] != source
        or report.get("fresh_core_install") is not True
    ):
        raise ValueError("require the accepted observer-main fresh installation")
    python = Path(report["python"]).absolute()
    if not python.is_file():
        raise ValueError("admitted interpreter is absent")
    sites = list(python.parent.parent.glob("lib/python*/site-packages"))
    if len(sites) != 1:
        raise ValueError("ambiguous installed package location")
    site = sites[0]
    wheel = Path(report["wheel"]["path"])
    if digest(wheel) != report["wheel"]["sha256"]:
        raise ValueError("retained wheel digest changed")
    files: dict[str, str] = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("flagquantum/") and not name.endswith("/"):
                data = archive.read(name)
                if (site / name).read_bytes() != data:
                    raise ValueError(f"installed wheel member changed: {name}")
                files[name] = hashlib.sha256(data).hexdigest()
    dependencies = {
        item.metadata["Name"]: {
            "version": item.version,
            "metadata_sha256": hashlib.sha256(
                (item.read_text("METADATA") or "").encode()
            ).hexdigest(),
        }
        for item in importlib.metadata.distributions(path=[str(site)])
    }
    torch_version = next(
        v["version"] for k, v in dependencies.items() if k.lower() == "torch"
    )
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=10
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True, timeout=10
    )
    if head != source or dirty:
        raise ValueError("controller checkout must be clean admitted observer main")
    cpu_model = platform.machine()
    if platform.system() == "Darwin":
        cpu_model = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
        ).strip()
    elif Path("/proc/cpuinfo").is_file():
        with Path("/proc/cpuinfo").open() as stream:
            cpu_model = next(
                (
                    line.partition(":")[2].strip()
                    for line in stream.read(1024**2).splitlines()
                    if line.startswith("model name")
                ),
                cpu_model,
            )
    return {
        "source_commit": source,
        "python": str(python),
        "python_executable_sha256": digest(python.resolve()),
        "runtime_versions": report["runtime_versions"],
        "site_packages": str(site),
        "wheel_sha256": digest(wheel),
        "package_members_sha256": files,
        "dependencies": dependencies,
        "torch_version": torch_version,
        "controller_sha256": digest(Path(__file__)),
        "worker_sha256": digest(root / "tools/observe_ftqc_phase_cost.py"),
        "watchdog_helpers_sha256": digest(root / "tools/run_with_watchdog.py"),
        "install_report_sha256": digest(install_report),
        "platform": platform.system(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "interpreter_config_sha256": digest(python.parent.parent / "pyvenv.cfg"),
        "max_jobs": "2",
    }


def limit_evidence_files() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))


def execute(
    command: list[str], directory: Path, output: Path, deadline: float
) -> dict[str, Any]:
    """Own one group from Popen through cleanup, including every failure's wall."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MAX_JOBS="2",
        PYTHONDONTWRITEBYTECODE="1",
    )
    started = previous_sample = time.monotonic()
    observations: list[tuple[float, int]] = []
    result: dict[str, Any] = {
        "status": "failed",
        "command": command,
        "rss_query_deadline_seconds": 0.05,
        "rss_max_gap_limit_seconds": 0.1,
        "rss_scope": "sum owned-group RSS; shared pages may count twice; sampled, not allocation cap",
    }
    process: subprocess.Popen[bytes] | None = None
    peak, maximum_gap = 0, 0.0
    with (
        (directory / "stdout.log").open("xb") as stdout,
        (directory / "stderr.log").open("xb") as stderr,
    ):
        try:
            process = subprocess.Popen(
                command,
                cwd=tempfile.gettempdir(),
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                preexec_fn=limit_evidence_files,
            )
            result["owned_pgid"] = process.pid
            while process.poll() is None:
                # Reserve the existing helper's TERM/KILL/wait allowance inside 30s.
                budget(output, min(deadline, started + SECONDS), CLEANUP_RESERVE)
                rss = _process_group_rss(process.pid)
                now = time.monotonic()
                maximum_gap = max(maximum_gap, now - previous_sample)
                previous_sample = now
                observations.append((now - started, rss))
                peak = max(peak, rss)
                if maximum_gap > 0.1 or rss > RSS_BYTES:
                    raise RuntimeError("RSS sample gap or group limit exceeded")
                time.sleep(0.01)
            final_rss = _process_group_rss(process.pid)
            maximum_gap = max(maximum_gap, time.monotonic() - previous_sample)
            if (
                final_rss
                or maximum_gap > 0.1
                or not any(rss > 0 for _, rss in observations)
            ):
                raise RuntimeError(
                    "group survived launcher or final monitoring interval invalid"
                )
            result.update(status="completed", returncode=process.returncode)
        except (
            BaseException
        ) as error:  # Process boundary preserves cancellation before cleanup.
            result["diagnostic"] = f"{type(error).__name__}: {error}"
        finally:
            cleanup = time.monotonic()
            if process is not None and result["status"] != "completed":
                try:
                    _stop_group(process)
                except (OSError, subprocess.SubprocessError) as error:
                    result["cleanup_error"] = f"{type(error).__name__}: {error}"
            result.update(
                outer_wall_seconds=time.monotonic() - started,
                cleanup_seconds=time.monotonic() - cleanup,
                sampled_group_rss_max_bytes=peak,
                rss_sample_gap_max_seconds=maximum_gap,
                rss_sample_count=sum(rss > 0 for _, rss in observations),
                samples_seconds_bytes=observations,
                returncode=None if process is None else process.returncode,
            )
    if result["outer_wall_seconds"] > SECONDS:
        result.update(
            status="failed",
            diagnostic="outer wall including cleanup exceeded 30 seconds",
        )
    atomic(directory / "supervision.json", result)
    return result


def compact_supervision(supervision: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in supervision.items()
        if key != "samples_seconds_bytes"
    }


def expected_work(item: dict[str, Any]) -> dict[str, int]:
    n, word = description(item["case"])
    branches = len(expansion(word))
    return {
        "public_executions": int(item["route"] != "candidate"),
        "state_constructions": branches + 2 ** n if item["route"] == "candidate" else 0,
        "overlap_queries": (
            branches**2 + branches * 2 ** n if item["route"] == "candidate" else 0
        ),
    }


def assess(
    directory: Path, supervision: dict[str, Any], item: dict[str, Any] | None
) -> dict[str, Any]:
    supervision = compact_supervision(supervision)
    if supervision["status"] != "completed" or supervision.get("returncode") not in (
        0,
        1,
    ):
        return {
            "status": "failed",
            "failure_domain": "supervision",
            "supervision": supervision,
        }
    if item is None:
        return {
            "status": "completed" if supervision.get("returncode") == 0 else "failed",
            "supervision": supervision,
        }
    try:
        report = json.loads((directory / "producer.json").read_text())
        if (report.get("case"), report.get("route")) != (item["case"], item["route"]):
            raise ValueError("worker changed admitted point")
        if supervision.get("returncode") != 0:
            raise ValueError(f"worker process failed: {report.get('diagnostic')}")
        if report.get("status") == "refused":
            if (
                item["route"] == "candidate"
                or report.get("refusal_stage") != "planning"
                or not report.get("diagnostic")
            ):
                raise ValueError("only explicit public planning refusal is nonfatal")
        elif report.get("status") == "completed":
            errors = report.get("errors")
            required = {"probability_error", "normalization_error"}
            if item["route"] == "candidate":
                required.update(("complex_amplitude_error", "complex_gram_error"))
            if any(report.get(k) != value for k, value in expected_work(item).items()):
                raise ValueError("worker exceeded or changed admitted work counters")
            if (
                not isinstance(errors, dict)
                or set(errors) != required
                or not all(
                    type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1e-12
                    for v in errors.values()
                )
                or report.get("performance_targets_enabled") is not False
            ):
                raise ValueError("independent comparison or target policy failed")
        else:
            raise ValueError("worker did not complete")
    except (ValueError, OSError) as error:
        return {
            "status": "failed",
            "failure_domain": "producer",
            "diagnostic": str(error),
            "supervision": supervision,
        }
    return {
        "status": report["status"],
        "producer_sha256": digest(directory / "producer.json"),
        "supervision": supervision,
    }


def summarize(output: Path, attempts: list[dict[str, Any]], deadline: float) -> None:
    """Derive medians/ranges from all three retained repeats, never a fastest subset."""
    points: list[dict[str, Any]] = []
    for case in CASES:
        for route in ROUTES:
            budget(output, deadline, 0)
            selected = [
                x for x in attempts if x["case"] == case and x["route"] == route
            ]
            fresh = [x for x in selected if x["repetition"] != 0]
            entry: dict[str, Any] = {
                "case": case,
                "route": route,
                "warmup_status": selected[0]["status"],
                "fresh_statuses": [x["status"] for x in fresh],
                "complete_three_repeats": len(fresh) == 3
                and all(x["status"] == "completed" for x in fresh),
            }
            if entry["complete_three_repeats"]:
                rows = []
                for item in fresh:
                    path = output / f"attempt-{item['ordinal']:03d}" / "producer.json"
                    data = json.loads(path.read_text())
                    rows.append(
                        {
                            "ordinal": item["ordinal"],
                            "producer_sha256": digest(path),
                            "numerical_seconds": data["numerical_seconds"],
                            "outer_wall_seconds": item["supervision"][
                                "outer_wall_seconds"
                            ],
                            "sampled_group_rss_max_bytes": item["supervision"][
                                "sampled_group_rss_max_bytes"
                            ],
                        }
                    )
                entry["fresh_observations"] = rows
                for metric in (
                    "numerical_seconds",
                    "outer_wall_seconds",
                    "sampled_group_rss_max_bytes",
                ):
                    values = [row[metric] for row in rows]
                    if not all(
                        type(v) in (int, float) and math.isfinite(v) and v >= 0
                        for v in values
                    ):
                        raise ValueError(
                            "summary requires finite nonnegative raw measurements"
                        )
                    entry[metric] = {
                        "median": statistics.median(values),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
            points.append(entry)
    atomic(
        output / "summary.json",
        {
            "points": points,
            "selection_made": False,
            "performance_targets_enabled": False,
            "interpretation": "Three observations only; no p95/p99, general winner, capacity or speedup certification",
        },
    )


def campaign(install_report: Path, output: Path, source: str) -> dict[str, Any]:
    started = time.monotonic()
    output = output.resolve()
    if any(
        part
        in (
            "fts10-cpu-observations-20261025",
            "fts10-cpu-continuation-20261025",
            "stage-a-observations-20261025",
        )
        for part in output.parts
    ):
        raise ValueError(
            "historical evidence trees cannot receive new collection files"
        )
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("collection evidence must be outside the clean checkout")
    if shutil.disk_usage(output.parent).free < FREE_BYTES:
        raise OSError("startup needs at least 512 MiB free")
    # Never mutate or reuse any existing campaign, including a failed one.
    output.mkdir(exist_ok=False)
    manifest: dict[str, Any] = {
        "status": "running",
        "attempts": schedule(),
        "started_unix_seconds": time.time(),
        "attempt_budget": 161,
        "performance_targets_enabled": False,
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "limits": {
            "worker_seconds": SECONDS,
            "cleanup_reserve_seconds": CLEANUP_RESERVE,
            "campaign_seconds": TOTAL_SECONDS,
            "rss_bytes": RSS_BYTES,
            "evidence_bytes": STORAGE_BYTES,
            "per_file_bytes": FILE_BYTES,
            "free_bytes": FREE_BYTES,
        },
    }
    deadline = started + TOTAL_SECONDS
    refused: set[tuple[str, str]] = set()

    def save() -> None:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["retained_bytes_before_write"] = retained_bytes(output)
        path = output / "manifest.json"
        # Only the evolving controller ledger is replaced; attempt files are immutable.
        temporary = output / "manifest.new"
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.replace(path)

    save()
    try:
        provenance = admission(install_report, source)
        atomic(output / "provenance.json", provenance)
        python = provenance["python"]
        worker = str(Path(__file__).with_name("observe_ftqc_phase_cost.py").resolve())
        for ordinal, item in enumerate([None, *manifest["attempts"]]):
            budget(output, deadline)
            if item is not None and (item["case"], item["route"]) in refused:
                item["reason"] = (
                    "earlier declared planning refusal; repetition unattempted"
                )
                save()
                continue
            directory = output / f"attempt-{ordinal:03d}"
            directory.mkdir()
            if item is not None:
                item.update(status="running", ordinal=ordinal)
            manifest["started_attempts"] = manifest.get("started_attempts", 0) + 1
            save()
            payload = (
                [python, "-c", "import time; time.sleep(1)"]
                if item is None
                else [
                    python,
                    worker,
                    "--case",
                    item["case"],
                    "--route",
                    item["route"],
                    "--provenance",
                    str(output / "provenance.json"),
                    "--output",
                    str(directory / "producer.json"),
                ]
            )
            supervision = execute(payload, directory, output, deadline)
            evidence = assess(directory, supervision, item)
            if item is None:
                manifest["probe"] = evidence
            else:
                item.update(evidence)
            save()
            if evidence["status"] == "failed":
                raise RuntimeError(
                    "first failure stops collection; remaining slots unattempted"
                )
            if evidence["status"] == "refused" and item is not None:
                refused.add((item["case"], item["route"]))
        summarize(output, manifest["attempts"], deadline)
        budget(output, deadline, 0)
        manifest["status"] = "completed_with_refusals" if refused else "completed"
    except BaseException as error:
        manifest.update(status="failed", diagnostic=f"{type(error).__name__}: {error}")
    finally:
        save()
    return manifest


def require_delivery(path: Path, source: str) -> None:
    data = json.loads(path.read_text())
    if (
        data.get("head") != source
        or data.get("all_main_admission_gates_passed") is not True
    ):
        raise ValueError(
            "collection requires the admitted observer-main delivery record"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--delivery-report", type=Path, required=True)
    args = parser.parse_args()

    def interrupted(signum: int, frame: FrameType | None) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if os.name != "posix":
        parser.error("the admitted RSS/process-group supervisor requires POSIX")
    require_delivery(args.delivery_report, args.source_commit)
    report = campaign(
        args.install_report.resolve(), args.output.resolve(), args.source_commit
    )
    print(
        json.dumps(
            {"status": report["status"], "attempts": report.get("started_attempts", 0)}
        )
    )
    return 0 if report["status"] in ("completed", "completed_with_refusals") else 1


if __name__ == "__main__":
    raise SystemExit(main())
