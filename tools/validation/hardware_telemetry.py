"""Record read-only hardware observations during one watchdog-owned command."""

from __future__ import annotations

import sys as _entry_sys
from pathlib import Path as _EntryPath

if __package__ in (None, ""):
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import json
import os
import re
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
        "--query-gpu=index,pstate,clocks.current.sm,clocks.current.memory,"
        "utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ),
    # Filter transient accounting before persistence; never request command names.
    "processes": ("ps", "-axo", "pid=,pgid=,psr=,pcpu=,rss="),
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
    for name, command in _COMMANDS.items():
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=0.5 if name == "gpu" else 0.25,
            )
        except (OSError, UnicodeError, subprocess.SubprocessError) as error:
            observations[f"{name}_error"] = type(error).__name__
        else:
            if result.returncode:
                observations[f"{name}_error"] = f"exit_status={result.returncode}"
                continue
            if result.stderr:
                observations[f"{name}_diagnostic"] = "stderr_present"
            rows = []
            for row in result.stdout.splitlines():
                if name == "gpu":
                    fields = [field.strip() for field in row.split(",")]
                    valid = (
                        len(fields) == 9
                        and fields[0].isdigit()
                        and re.fullmatch(r"P[0-9]+|N/A|\[Not Supported\]", fields[1])
                        and all(
                            re.fullmatch(
                                r"[0-9]+(?:\.[0-9]+)?|N/A|\[Not Supported\]", value
                            )
                            for value in fields[2:]
                        )
                    )
                    if valid:
                        rows.append(", ".join(fields))
                    else:
                        observations["gpu_error"] = "unrecognized GPU accounting row"
                else:
                    fields = row.split()
                    if (
                        len(fields) != 5
                        or not all(fields[index].isdigit() for index in (0, 1, 2, 4))
                        or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", fields[3])
                    ):
                        observations["processes_error"] = (
                            "unrecognized process accounting row"
                        )
                    elif int(fields[1]) == group:
                        rows.append(" ".join(fields))
            observations[name] = "\n".join(rows)
            if name == "gpu" and not result.stdout.strip():
                observations["gpu_error"] = "empty GPU accounting"
    try:
        observations["load_average_1_5_15_minutes"] = str(os.getloadavg())
    except OSError as error:
        observations["load_average_error"] = type(error).__name__

    affinities = []
    for row in observations.get("processes", "").splitlines():
        fields = row.split()
        pid = int(fields[0])
        try:
            allowed = sorted(os.sched_getaffinity(pid))
        except (AttributeError, OSError) as error:
            affinities.append(f"pid={pid} unavailable: {type(error).__name__}")
        else:
            affinities.append(f"pid={pid} allowed_cpus={allowed}")
    observations["owned_process_affinities"] = "\n".join(affinities)
    sample["collection_seconds"] = time.monotonic() - before
    # Stream snapshots rather than retaining a command-length in-memory history.
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sample, allow_nan=False) + "\n")
