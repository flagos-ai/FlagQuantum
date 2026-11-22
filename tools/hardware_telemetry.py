"""Record read-only hardware observations during one watchdog-owned command."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import TypedDict


class TelemetrySample(TypedDict):
    observed_at_unix_seconds: float
    elapsed_seconds: float
    collection_seconds: float
    process_group: int
    observations: dict[str, str]


_COMMANDS = {
    "gpu": (
        "nvidia-smi",
        "--query-gpu=index,uuid,pstate,clocks.current.sm,clocks.current.memory,"
        "utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
        "--format=csv,nounits",
    ),
    # Include other processes to expose shared-host CPU load. Do not collect
    # arguments, environments or credentials from their command lines.
    "processes": ("ps", "-axo", "pid=,pgid=,psr=,pcpu=,rss=,comm="),
}


def record_sample(destination: Path, group: int, started: float) -> None:
    """Append one bounded diagnostic snapshot; unavailable data stays explicit."""
    before = time.monotonic()
    sample: TelemetrySample = {
        "observed_at_unix_seconds": time.time(),
        "elapsed_seconds": before - started,
        "collection_seconds": 0.0,
        "process_group": group,
        "observations": {},
    }
    observations = sample["observations"]
    observations["cuda_visible_devices"] = os.environ.get(
        "CUDA_VISIBLE_DEVICES", "unset"
    )
    for name, command in _COMMANDS.items():
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=0.5 if name == "gpu" else 0.25,
            )
        except (OSError, UnicodeError, subprocess.SubprocessError) as error:
            observations[f"{name}_error"] = f"{type(error).__name__}: {error}"
        else:
            observations[name] = result.stdout
            if result.returncode or result.stderr:
                observations[f"{name}_diagnostic"] = (
                    f"returncode={result.returncode}\n{result.stderr}"
                )
    try:
        observations["load_average_1_5_15_minutes"] = str(os.getloadavg())
    except OSError as error:
        observations["load_average_error"] = str(error)

    affinities = []
    for row in observations.get("processes", "").splitlines():
        fields = row.split(maxsplit=5)
        if len(fields) != 6 or not fields[0].isdigit() or not fields[1].isdigit():
            observations["affinity_error"] = "unrecognized process accounting row"
            continue
        if int(fields[1]) != group:
            continue
        pid = int(fields[0])
        try:
            allowed = sorted(os.sched_getaffinity(pid))
        except (AttributeError, OSError) as error:
            affinities.append(f"pid={pid} unavailable: {error}")
        else:
            affinities.append(f"pid={pid} allowed_cpus={allowed}")
    observations["owned_process_affinities"] = "\n".join(affinities)
    sample["collection_seconds"] = time.monotonic() - before
    # Stream snapshots rather than retaining a command-length in-memory history.
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sample, allow_nan=False) + "\n")
