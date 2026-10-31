#!/usr/bin/env python3
"""Build the pinned FlagCX Ascend/FlagOS dependency in a dedicated environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REVISION = "6fe36310fb5c9bd2396f531d7bf78c08fc792340"
JSON_SHA256 = "44b4cba46af923a0f00bef52b6291d7d2ea37f32f5bafce4648a33e24c1ab874"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--json-include", required=True, type=Path)
    parser.add_argument("--wheel-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    json_include = args.json_include.resolve(strict=True)
    wheel_dir = args.wheel_dir.resolve()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True, timeout=15
    ).strip()
    if head != REVISION:
        raise RuntimeError(f"FlagCX must be checked out at {REVISION}; found {head}")
    header = json_include / "nlohmann" / "json.hpp"
    if hashlib.sha256(header.read_bytes()).hexdigest() != JSON_SHA256:
        raise RuntimeError("The nlohmann/json single header does not match the pin")
    spec = importlib.util.find_spec("torch_fl")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Install the validated Torch-FL wheel before building FlagCX"
        )
    patch = Path(__file__).parent / "patches" / "flagcx-v0.14.1-flagos.patch"
    subprocess.run(
        ["git", "apply", "--check", str(patch.resolve())],
        cwd=source,
        check=True,
        timeout=15,
    )
    subprocess.run(
        ["git", "apply", str(patch.resolve())], cwd=source, check=True, timeout=15
    )
    wheel_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        FLAGCX_ADAPTOR="ascend",
        FLAGCX_TORCH_BACKEND="flagos",
        FLAGOS_INSTALL_PATH=str(Path(spec.origin).parent),
        JSON_INCLUDE_DIR=str(json_include),
        MAX_JOBS="4",
    )
    # Upstream setup.py sizes its build from cpu_count rather than MAX_JOBS.
    launcher = (
        "import multiprocessing,runpy,sys;"
        "multiprocessing.cpu_count=lambda:4;"
        "sys.argv=['setup.py','bdist_wheel','--dist-dir',sys.argv[1]];"
        "runpy.run_path('setup.py',run_name='__main__')"
    )
    subprocess.run(
        [sys.executable, "-c", launcher, str(wheel_dir)],
        cwd=source,
        env=environment,
        check=True,
        timeout=3600,
    )


if __name__ == "__main__":
    main()
