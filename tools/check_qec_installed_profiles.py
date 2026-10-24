"""Check five real QEC installations on the Linux package runner."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import FrameType


def _cancel(signum: int, frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def _run(command: list[str], *, cwd: Path, log: Path, seconds: int = 240) -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE")
    }
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MAX_JOBS="2",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
    )
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=seconds)
        except BaseException:
            # Own process boundary: release installers/compiler children on cancellation.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def _profile(artifact: Path, profile: str, reports: Path, label: str) -> None:
    extras = {
        "core": "",
        "stim": "stim",
        "pymatching": "pymatching",
        "both": "stim,pymatching",
    }
    probe = Path(__file__).with_name("qec_installed_smoke.py").resolve()
    with tempfile.TemporaryDirectory(prefix="fq-qec-installed-") as directory:
        outside = Path(directory)
        python = outside / "venv/bin/python"
        _run(
            [sys.executable, "-m", "venv", str(outside / "venv")],
            cwd=outside,
            log=reports / f"{label}-venv.log",
            seconds=30,
        )
        # Use the declared build baseline and the existing package job's CPU source.
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "torch>=2.10,<2.11",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            cwd=outside,
            log=reports / f"{label}-torch.log",
        )
        requirement = str(artifact) + (
            f"[{extras[profile]}]" if extras[profile] else ""
        )
        build_flags = []
        if artifact.name.endswith(".tar.gz"):
            # Satisfy the declared requirements with CPU Torch already installed,
            # as the existing sdist CI does; isolated PyPI builds can pull CUDA.
            _run(
                [str(python), "-m", "pip", "install", "setuptools>=77.0.3", "wheel"],
                cwd=outside,
                log=reports / f"{label}-build-tools.log",
            )
            build_flags.append("--no-build-isolation")
        _run(
            [str(python), "-m", "pip", "install", *build_flags, requirement],
            cwd=outside,
            log=reports / f"{label}-install.log",
            seconds=360,
        )
        _run(
            [str(python), "-m", "pip", "check"],
            cwd=outside,
            log=reports / f"{label}-dependencies.log",
            seconds=30,
        )
        _run(
            [str(python), str(probe), profile],
            cwd=outside,
            log=reports / f"{label}-smoke.json",
            seconds=60,
        )


def main() -> int:
    signal.signal(signal.SIGTERM, _cancel)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    dist = args.dist_dir.resolve()
    wheels, sources = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        parser.error("expected one verified wheel and one sdist from the package job")
    manifest = json.loads((dist / "manifest.json").read_text(encoding="utf-8"))
    if (
        os.environ.get("GITHUB_SHA")
        and manifest["commit_sha"] != os.environ["GITHUB_SHA"]
    ):
        parser.error("artifact manifest does not describe this CI commit")
    reports = args.report_dir.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "artifact-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for profile in ("core", "stim", "pymatching", "both"):
        _profile(wheels[0], profile, reports, "wheel-" + profile)
    _profile(sources[0], "both", reports, "sdist-both")
    print("QEC installed acceptance passed: four real wheel profiles and sdist both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
