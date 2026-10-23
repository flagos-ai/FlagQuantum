"""Verify the MC550 service boundary from inside an actual Actions job."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def verify() -> dict[str, object]:
    root = Path("/liuwei01/flagquantum-runners")
    slot = os.environ["FQ_MC550_SLOT"]
    peer = "cpu02" if slot != "cpu02" else "cpu01"
    if os.geteuid() == 0:
        raise RuntimeError("Runner must not execute as root")
    status = Path("/proc/self/status").read_text()
    if "CapEff:\t0000000000000000" not in status or "NoNewPrivs:\t1" not in status:
        raise RuntimeError("Capabilities or privilege escalation remain available")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 8 or affinity[4:] != [cpu + 80 for cpu in affinity[:4]]:
        raise RuntimeError(f"Unexpected physical-core allocation: {affinity}")
    membership = next(
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    cgroup = Path("/sys/fs/cgroup") / membership.lstrip("/")
    quota, period = map(int, (cgroup / "cpu.max").read_text().split())
    memory = int((cgroup / "memory.max").read_text())
    if quota / period != 4 or memory != 16 * 1024**3:
        raise RuntimeError("Per-runner CPU or memory cap is missing")
    parent = cgroup.parent
    pool_quota, pool_period = map(int, (parent / "cpu.max").read_text().split())
    if (
        pool_quota / pool_period != 64
        or int((parent / "memory.max").read_text()) != 256 * 1024**3
    ):
        raise RuntimeError("Aggregate pool resource cap is missing")
    if (
        int((cgroup / "memory.swap.max").read_text()) != 0
        or os.getpriority(os.PRIO_PROCESS, 0) < 10
    ):
        raise RuntimeError("Runner could consume shared swap or normal CPU priority")
    for path in [
        Path("/root"),
        Path("/data"),
        Path("/public-nfs"),
        root / "admin",
        root / "homes" / peer,
        root / "runners" / peer / ".credentials",
    ]:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except (PermissionError, FileNotFoundError):
            continue
        os.close(descriptor)
        raise RuntimeError(f"Unexpected access to protected host path: {path}")
    if Path("/run/docker.sock").exists():
        raise RuntimeError("Host Docker socket is exposed")
    for pattern in ["nvidia*", "dri/render*", "mtgpu*", "metax*", "musa*"]:
        if list(Path("/dev").glob(pattern)):
            raise RuntimeError(f"Accelerator device exposed: {pattern}")
    try:
        with tempfile.NamedTemporaryFile(dir=root / "runners" / slot):
            pass
    except PermissionError:
        pass
    else:
        raise RuntimeError("Runner can modify its own application directory")
    with tempfile.NamedTemporaryFile(dir=os.environ["RUNNER_TEMP"]):
        pass
    report = {
        "runner": os.environ["RUNNER_NAME"],
        "slot": slot,
        "affinity": affinity,
        "cpu_quota": quota / period,
        "memory_max_bytes": memory,
        "pool_cpu_quota": pool_quota / pool_period,
        "cgroup": membership,
        "uid": os.geteuid(),
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    verify()
