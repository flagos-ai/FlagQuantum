"""Profile only the twentieth backward of the batch-64 Adam case."""

from __future__ import annotations

import argparse
import runpy
import tempfile
from pathlib import Path

import torch
from run import WORKER, install_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", choices=("baseline", "candidate"))
    args = parser.parse_args()
    if args.route == "baseline":
        install_baseline()
    train = runpy.run_path(str(WORKER))["worker"]
    original = torch.autograd.backward
    count = 0

    def measured(*args, **kwargs):
        nonlocal count
        count += 1
        if count != 20:
            return original(*args, **kwargs)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
        ) as profile:
            result = original(*args, **kwargs)
        print(
            profile.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total", row_limit=22
            )
        )
        return result

    torch.autograd.backward = measured
    try:
        with tempfile.TemporaryDirectory() as directory:
            train(3, "Adam", "default_eager", Path(directory) / "worker.json", False)
    finally:
        torch.autograd.backward = original


if __name__ == "__main__":
    main()
