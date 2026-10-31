#!/usr/bin/env python3
"""Run the native 910C regression on an explicitly provisioned Torch-FL stack."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = (
    "tests/integration/test_ascend_circuit_workflows.py",
    "tests/integration/test_ascend_native_dynamics.py",
    "tests/test_ascend_910c_training.py",
)


def validate_report(report: Path, logs: Sequence[Path]) -> int:
    """Refuse an empty, skipped, failed, or CPU-fallback hardware run."""
    root = ET.parse(report).getroot()
    if root.tag not in ("testsuite", "testsuites"):
        raise RuntimeError("The Ascend regression did not produce a JUnit suite")
    cases = list(root.iter("testcase"))
    if not cases:
        raise RuntimeError("The Ascend regression collected no test cases")
    for case in cases:
        if any(case.find(tag) is not None for tag in ("skipped", "failure", "error")):
            raise RuntimeError(f"Incomplete Ascend regression: {case.get('name')}")
    for log in logs:
        if "[flagos cpu_fallback]" in log.read_text(encoding="utf-8"):
            raise RuntimeError(f"CPU fallback observed in {log.name}")
    return len(cases)


def _source_revision() -> str:
    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, timeout=30
        ).strip()

    revision = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise RuntimeError("The Ascend regression requires a clean Git checkout")
    if os.environ.get("GITHUB_SHA", revision) != revision:
        raise RuntimeError("The checkout does not match this workflow's GITHUB_SHA")
    return revision


def _probe(expected_devices: int) -> None:
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("This lane requires the provisioned Linux ARM64 910C host")
    if os.environ.get("FLAGQUANTUM_ACCELERATOR_MODEL") != "Ascend910C":
        raise RuntimeError("The provisioner must declare Ascend910C hardware identity")
    torch_fl = importlib.import_module("torch_fl")
    torch = importlib.import_module("torch")
    flagcx = importlib.import_module("flagcx")
    for module in (torch_fl, flagcx):
        if module.__file__ is None or not Path(module.__file__).is_relative_to(
            sys.prefix
        ):
            raise RuntimeError(
                "Install Torch-FL and FlagCX in the active Python environment"
            )
    if torch.version.cuda is not None:
        raise RuntimeError(
            "The qualified Ascend lane uses the CPU PyTorch distribution"
        )
    if torch.flagos.device_count() != expected_devices:
        raise RuntimeError(
            "The visible logical NPU count differs from the provisioned lane"
        )
    vendor = sys.modules.get("torch_npu")
    if vendor is not None and not getattr(vendor, "__torch_fl_shim__", False):
        raise RuntimeError(
            "A real torch_npu module was loaded in the pure Torch-FL lane"
        )
    if "libtorch_npu" in Path("/proc/self/maps").read_text(encoding="utf-8"):
        raise RuntimeError("A torch_npu extension was loaded in the pure Torch-FL lane")
    fq = importlib.import_module("flagquantum")
    if fq.__file__ is None or not Path(fq.__file__).is_relative_to(ROOT):
        raise RuntimeError("FlagQuantum must be imported from this exact checkout")
    torch.flagos.set_device(0)
    x = torch.tensor(
        [0.2 + 0.1j, -0.3 + 0.4j],
        dtype=torch.complex128,
        device="flagos:0",
        requires_grad=True,
    )
    value = (x.conj() * x).real.sum()
    value.backward()
    if (
        value.device.type != "flagos"
        or x.grad is None
        or x.grad.device.type != "flagos"
    ):
        raise RuntimeError(
            "The complex objective or its gradient left the native device"
        )
    torch.testing.assert_close(
        value.cpu(), torch.tensor(0.3, dtype=torch.float64), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(x.grad.cpu(), 2 * x.detach().cpu(), atol=1e-12, rtol=0)
    for package in ("torch", "torch_fl", "flagcx", "flag_gems", "flagtree"):
        print(f"{package}={importlib.metadata.version(package)}", flush=True)
    print(f"logical_npu_count={expected_devices}", flush=True)
    print(f"device_name={torch.flagos.get_device_name(0)}", flush=True)
    print("native_complex128_gradient_probe=passed", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-devices", type=int, default=2)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.expected_devices < 1:
        parser.error("--expected-devices must be positive")
    if args.probe:
        _probe(args.expected_devices)
        return 0
    revision = _source_revision()
    output = args.output_dir.resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        parser.error("Store raw CI artifacts outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.sha").write_text(revision + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        FLAGQUANTUM_TEST_ASCEND_910C="1",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        FLAGCX_TORCH_BACKEND="flagos",
        FLAGOS_DIST_STAGED_GLOO="0",
        FLAGOS_LOG="fallback",
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT), environment.get("PYTHONPATH", "")))
    )
    probe_log = output / "environment.log"
    with probe_log.open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--probe",
                "--expected-devices",
                str(args.expected_devices),
                "--output-dir",
                str(output),
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )
    report = output / "native.xml"
    command = [
        sys.executable,
        "tools/run_with_watchdog.py",
        "--phase",
        "ascend-native-regression",
        "--stall-seconds",
        "240",
        "--timeout-seconds",
        "1200",
        "--diagnostics",
        str(output / "watchdog"),
        "--",
        sys.executable,
        "-m",
        "pytest",
        *TESTS,
        "-v",
        "-s",
        f"--junitxml={report}",
    ]
    print(f"BEGIN Ascend native regression at {revision}", flush=True)
    native_log = output / "native.log"
    with native_log.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1300,
        )
    count = validate_report(report, (probe_log, native_log))
    print(f"PASS {count} native regression cases; no skips or CPU fallback", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
