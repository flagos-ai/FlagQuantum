"""Measure Linux runner limits and budget pytest for two shared registrations."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

GIB = 1024**3


def controller_directories(proc: Path = Path("/proc")) -> dict[str, list[Path]]:
    """Resolve cgroup mounts and include every visible limiting ancestor."""
    result: dict[str, list[Path]] = {"cpu": [], "memory": []}
    try:
        memberships = [
            line.split(":", 2)
            for line in (proc / "self/cgroup").read_text().splitlines()
        ]
        mounts = (proc / "self/mountinfo").read_text().splitlines()
    except OSError:
        return result
    for line in mounts:
        before, after = line.split(" - ", 1)
        fields, filesystem = before.split(), after.split()
        kind = filesystem[0]
        if kind not in {"cgroup", "cgroup2"}:
            continue
        root, mount = Path(fields[3]), Path(fields[4])
        for _, controllers, relative in memberships:
            names = set(controllers.split(","))
            if kind == "cgroup2" and controllers:
                continue
            if kind == "cgroup" and not names.intersection(filesystem[2].split(",")):
                continue
            try:
                current = mount / Path(relative).relative_to(root)
            except ValueError:
                # A cgroup namespace can report / while mountinfo retains
                # the host root. Its visible mount root still limits the job.
                if relative != "/":
                    raise RuntimeError(
                        f"Cannot resolve cgroup membership {relative}"
                    ) from None
                current = mount
            directories = [current, *current.parents]
            directories = directories[: directories.index(mount) + 1]
            for controller in result:
                if kind == "cgroup2" or controller in names:
                    result[controller].extend(directories)
    return result


def read_limits(directories: dict[str, list[Path]]) -> tuple[list[float], list[int]]:
    """Read v1/v2 CPU bandwidth and memory limits; unlimited is not a quota."""
    cpu: list[float] = []
    memory: list[int] = []
    for directory in directories["cpu"]:
        try:
            if (directory / "cpu.max").exists():
                quota, period = (directory / "cpu.max").read_text().split()
                if quota != "max":
                    cpu.append(int(quota) / int(period))
            elif (directory / "cpu.cfs_quota_us").exists():
                quota_value = int((directory / "cpu.cfs_quota_us").read_text())
                period_value = int((directory / "cpu.cfs_period_us").read_text())
                if quota_value > 0:
                    cpu.append(quota_value / period_value)
        except (OSError, ValueError, ZeroDivisionError):
            raise RuntimeError(f"Cannot determine CPU quota at {directory}") from None
    for directory in directories["memory"]:
        for name in ["memory.max", "memory.limit_in_bytes"]:
            path = directory / name
            if path.exists():
                value = path.read_text().strip()
                if value != "max" and 0 < int(value) < 2**60:
                    memory.append(int(value))
    return cpu, memory


def select_workers(
    cpu: float, memory: int, shared: bool, dedicated: bool = False
) -> int:
    """Share a host between two registrations; bound memory and fan-out too."""
    if not shared:
        return max(1, min(2, math.floor(cpu)))
    if dedicated:
        return max(1, min(16, math.floor(cpu), memory // (2 * GIB)))
    return max(1, min(8, math.floor(cpu / 2), memory // (2 * 2 * GIB)))


def select_distributed_workers(cpu: float, memory: int, shared: bool) -> int:
    """Budget up to eight child ranks per launcher, with at most two launchers."""
    if not shared:
        return 1
    return max(
        1, min(2, math.floor(math.ceil(cpu / 2) / 8), memory // (2 * 8 * 2 * GIB))
    )


def main(*, dedicated: bool = False) -> None:
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    directories = controller_directories()
    if shared := os.environ.get("RUNNER_ENVIRONMENT") == "self-hosted":
        if not directories["cpu"] or not directories["memory"]:
            raise RuntimeError("Shared Linux runner cgroup limits are unavailable")
    quotas, memory_limits = read_limits(directories)
    physical_memory = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    cpu = min([float(len(affinity)), *quotas])
    memory = min([physical_memory, *memory_limits])
    workers = select_workers(cpu, memory, shared, dedicated)
    distributed_workers = select_distributed_workers(cpu, memory, shared)
    report = {
        "runner": os.environ.get("RUNNER_NAME"),
        "hostname": os.uname().nodename,
        "affinity": affinity,
        "cpu_quotas": quotas,
        "effective_cpus": cpu,
        "memory_limit_bytes": memory,
        "controller_directories": {
            key: [str(path) for path in paths] for key, paths in directories.items()
        },
        "load_average": os.getloadavg(),
        "workers": workers,
        "distributed_workers": distributed_workers,
        "threads_per_worker": 1 if shared else 2,
        "shared_registrations_budgeted": 2 if shared and not dedicated else 1,
        "dedicated_service": dedicated,
        "inherited_rank_environment": {
            key: os.environ[key]
            for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"]
            if key in os.environ
        },
    }
    report_directory = Path(
        tempfile.mkdtemp(prefix="fq-ci-report-", dir=os.environ.get("RUNNER_TEMP"))
    )
    payload = json.dumps(report, indent=2)
    (report_directory / "resources.json").write_text(payload + "\n")
    print(payload, flush=True)
    if "GITHUB_ENV" in os.environ:
        with Path(os.environ["GITHUB_ENV"]).open("a") as stream:
            stream.write(
                f"FQ_PYTEST_WORKERS={workers}\nFQ_DISTRIBUTED_WORKERS={distributed_workers}\n"
                f"FQ_CI_REPORT_DIR={report_directory}\n"
            )
            if shared:
                for name in [
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                ]:
                    stream.write(f"{name}=1\n")
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
            stream.write(f"### CPU budget\n\n```json\n{payload}\n```\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dedicated", action="store_true")
    main(dedicated=parser.parse_args().dedicated)
