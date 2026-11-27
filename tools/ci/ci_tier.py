#!/usr/bin/env python
"""Run FlagQuantum test tiers used by local and CI workflows.

This script intentionally orchestrates existing pytest markers and benchmark
audit commands. It does not run performance benchmarks or turn CPU distributed
semantics into scalability evidence.
"""

from __future__ import annotations

import sys as _entry_sys
from pathlib import Path as _EntryPath

if __package__ in (None, ""):
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

Command = tuple[str, ...]


@dataclass(frozen=True)
class CITier:
    name: str
    trigger: str
    proves: str
    does_not_prove: str
    commands: tuple[Command, ...]
    blocks_default_pr: bool = False

    def rendered_commands(
        self, python_executable: str = "python"
    ) -> tuple[Command, ...]:
        return tuple(
            tuple(
                python_executable if token == "{python}" else token for token in command
            )
            for command in self.commands
        )

    def command_lines(self, python_executable: str = "python") -> tuple[str, ...]:
        return tuple(
            " ".join(command) for command in self.rendered_commands(python_executable)
        )


CI_TIERS: dict[str, CITier] = {
    "pr-default": CITier(
        name="pr-default",
        trigger="Every pull request and ordinary local issue verification.",
        proves="Fast smoke and unit health for import, local Circuit behavior, IR, planner, and pure audit helpers.",
        does_not_prove="Distributed transport, benchmark performance, GPU execution, or scalability claims.",
        commands=(("{python}", "-m", "pytest", "-m", "smoke or unit", "-q"),),
        blocks_default_pr=True,
    ),
    "pr-runtime": CITier(
        name="pr-runtime",
        trigger="Pull requests that touch runtime, API, planner, compiler, or integration boundaries.",
        proves="Local cross-module runtime/API behavior selected by the integration marker.",
        does_not_prove="Multi-process transport, accelerator execution, or scalability claims.",
        commands=(
            ("{python}", "-m", "pytest", "-m", "integration and not slow", "-q"),
        ),
    ),
    "pr-distributed": CITier(
        name="pr-distributed",
        trigger="Pull requests that touch distributed planning/runtime metadata, audit, benchmark JSON, or release gates.",
        proves="CPU/local distributed semantics, fail-closed behavior, benchmark JSON contracts, and release-gate validation.",
        does_not_prove="Real multi-GPU or multi-node capacity expansion or scalability; CPU distributed tests are not release evidence.",
        commands=(
            ("{python}", "-m", "pytest", "-m", "distributed_cpu", "-q"),
            (
                "{python}",
                "-m",
                "pytest",
                "-m",
                "benchmark_contract or release_gate",
                "-q",
            ),
        ),
    ),
    "nightly": CITier(
        name="nightly",
        trigger="Scheduled nightly CPU-safe validation or explicit maintainer request.",
        proves="Broad non-benchmark, non-accelerator, non-multinode repository health on CPU.",
        does_not_prove="GPU kernels, multi-node transport, benchmark performance, or release-grade scalability.",
        commands=(
            (
                "{python}",
                "-m",
                "pytest",
                "-m",
                "not benchmark_contract and not scalability and not distributed_multinode and not distributed_accel and not gpu",
                "-q",
            ),
        ),
    ),
    "gpu-scheduled": CITier(
        name="gpu-scheduled",
        trigger="Scheduled or manual job on an explicit accelerator runner.",
        proves="Tests marked for accelerator-backed distributed behavior, and the device-bound Triton kernels, on the configured hardware.",
        does_not_prove="Multi-node transport or release scalability without benchmark audit evidence.",
        commands=(
            ("{python}", "-m", "pytest", "-m", "distributed_accel and gpu", "-q"),
            ("{python}", "-m", "pytest", "-m", "triton and gpu", "-q"),
        ),
    ),
    "multinode-scheduled": CITier(
        name="multinode-scheduled",
        trigger="Manual dispatch on the launch host, which is the node that can reach its peer over ssh; not scheduled, because the pair holds a device on each host.",
        proves="Three workloads, each one partitioned across two nodes on one revision -- forward, backward, optimizer step and checkpoint resume -- configured for the RoCE fabric, timed with warmup and repeats, with the NCCL route it actually took read back from the debug log and the host staging audited by a profiler.",
        does_not_prove="Release scalability: each probe reports release_gate_allowed and scalability_claim_allowed false, and the audit that could promote one runs separately.",
        commands=(
            (
                "{python}",
                "tools/validation/multinode_launch_plan.py",
                "--run",
                "--staging",
                "/nfs/fq-multinode-tier",
                "--checkpoint-directory",
                "/nfs/fq-multinode-tier-checkpoints",
                "--report-directory",
                "hardware-run",
                # The route is named rather than discovered: a lane that let
                # NCCL choose could record a socket run and leave the fabric
                # untested while looking like a pass. The debug log the launch
                # already keeps is what the probe reads back, and a log that
                # does not confirm the configured route fails the probe.
                "--interface",
                "ens22f0",
                "--transport",
                "rdma",
                # Warmup plus repeats between synchronizations, so the artifact
                # carries a measured duration rather than one unsynchronized
                # reading of the launch.
                "--measure",
            ),
            (
                "{python}",
                "tools/validation/multinode_launch_plan.py",
                "--run",
                "--probe",
                "mps",
                "--staging",
                "/nfs/fq-multinode-tier",
                "--checkpoint-directory",
                "/nfs/fq-multinode-tier-checkpoints",
                "--report-directory",
                "hardware-run-mps",
                # The route is named rather than discovered: a lane that let
                # NCCL choose could record a socket run and leave the fabric
                # untested while looking like a pass. The debug log the launch
                # already keeps is what the probe reads back, and a log that
                # does not confirm the configured route fails the probe.
                "--interface",
                "ens22f0",
                "--transport",
                "rdma",
                # Warmup plus repeats between synchronizations, so the artifact
                # carries a measured duration rather than one unsynchronized
                # reading of the launch.
                "--measure",
            ),
            (
                "{python}",
                "tools/validation/multinode_launch_plan.py",
                "--run",
                "--probe",
                "tn",
                "--staging",
                "/nfs/fq-multinode-tier",
                "--checkpoint-directory",
                "/nfs/fq-multinode-tier-checkpoints",
                "--report-directory",
                "hardware-run-tn",
                # The route is named rather than discovered: a lane that let
                # NCCL choose could record a socket run and leave the fabric
                # untested while looking like a pass. The debug log the launch
                # already keeps is what the probe reads back, and a log that
                # does not confirm the configured route fails the probe.
                "--interface",
                "ens22f0",
                "--transport",
                "rdma",
                # Warmup plus repeats between synchronizations, so the artifact
                # carries a measured duration rather than one unsynchronized
                # reading of the launch.
                "--measure",
            ),
        ),
    ),
    "release": CITier(
        name="release",
        trigger="Release candidate validation before promoting benchmark payloads or publishing release notes.",
        proves="Full non-performance-benchmark pytest coverage plus benchmark audit and release payload validation.",
        does_not_prove="Any claim outside the audited payload; release gate remains fail-closed.",
        commands=(
            ("{python}", "-m", "pytest", "-m", "not scalability", "-q"),
            (
                "{python}",
                "benchmarks/audit_results.py",
                "--input",
                "benchmarks/results",
            ),
            (
                "{python}",
                "benchmarks/audit_results.py",
                "--input",
                "benchmarks/results/scalability",
                "--require-scalability",
            ),
        ),
    ),
}


def run_commands(commands: Sequence[Command], *, cwd: Path, dry_run: bool) -> int:
    for command in commands:
        print("+ " + " ".join(command), flush=True)
        if dry_run:
            continue
        result = subprocess.run(command, cwd=cwd)
        if result.returncode != 0:
            return result.returncode
    return 0


@contextmanager
def _collection_directory() -> Iterator[Path]:
    if report := os.environ.get("FQ_CI_REPORT_DIR"):
        # CI retains collection evidence alongside the execution report.
        yield Path(tempfile.mkdtemp(prefix="collection-", dir=report))
    else:
        with tempfile.TemporaryDirectory(prefix="fq-pytest-selection-") as directory:
            yield Path(directory)


def _verified_selection(directory: Path) -> tuple[Path, bool]:
    payload = json.loads((directory / "selection.json").read_text())
    if not isinstance(payload, dict):
        raise ValueError("Selected pytest manifest must be an object")
    raw = (directory / "nodes.txt").read_bytes()
    nodes = raw.decode().splitlines()
    if (
        payload.get("schema") != "flagquantum_pytest_selection_v1"
        or payload.get("source_sha") != os.environ.get("GITHUB_SHA")
        or type(payload.get("tests")) is not int
        or payload["tests"] != len(nodes)
        or type(payload.get("collected_tests")) is not int
        or payload["collected_tests"] < len(nodes)
        or not isinstance(payload.get("collected_sha256"), str)
        or len(payload["collected_sha256"]) != 64
        or type(payload.get("shard_count")) is not int
        or payload["shard_count"] < 1
        or type(payload.get("shard_index")) is not int
        or not 0 <= payload["shard_index"] < payload["shard_count"]
        or not nodes
        or len(nodes) != len(set(nodes))
        or payload.get("sha256") != hashlib.sha256(raw).hexdigest()
        or type(payload.get("exclusive")) is not bool
        or (payload["exclusive"] and len(nodes) != 1)
    ):
        raise ValueError(
            "Selected pytest manifest does not match this source or content"
        )
    return directory / "nodes.txt", payload["exclusive"]


def _exclusive_environment(
    python: str, directory: Path, cwd: Path
) -> tuple[dict[str, str], int]:
    """Use the existing admission owner, retaining its original resource evidence."""
    environment = dict(os.environ)
    env_file = directory / "exclusive.env"
    result = subprocess.run(
        [python, "-I", str(cwd / "tools/ci/ci_cpu_budget.py"), "--exclusive-test"],
        cwd=cwd,
        env={**environment, "GITHUB_ENV": str(env_file)},
    )
    if result.returncode:
        return environment, result.returncode
    admitted = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    allowed = {
        "FQ_PYTEST_WORKERS",
        "FQ_DISTRIBUTED_WORKERS",
        "FQ_CI_REPORT_DIR",
        "FQ_CI_HOST_BUDGET",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    }
    if set(admitted) != allowed or admitted["FQ_PYTEST_WORKERS"] != "1":
        raise ValueError("Exclusive CPU admission returned an unexpected environment")
    environment.update(admitted)
    # Preserve serial collection, its initial admission and refusal evidence.
    destination_report = Path(admitted["FQ_CI_REPORT_DIR"])
    if destination_report.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Exclusive resource report must have independent storage")
    shutil.copytree(directory, destination_report / "collection")
    if destination := os.environ.get("GITHUB_ENV"):
        with Path(destination).open("a") as target:
            target.write(env_file.read_text())
    return environment, 0


def run_planned_partition(
    command: Command, *, cwd: Path, count: int, index: int, dry_run: bool
) -> int:
    """Collect a Runtime shard once, then execute exactly its verified node IDs."""
    if dry_run:
        print(
            "Collect this shard serially, then execute its verified node IDs:",
            flush=True,
        )
        return run_commands((command,), cwd=cwd, dry_run=True)
    with _collection_directory() as directory:
        environment = dict(os.environ)
        if report := environment.get("FQ_CI_REPORT_DIR"):
            resource = Path(report) / "resources.json"
            if not resource.is_file():
                raise ValueError("The admitted CPU resource report is missing")
            shutil.copyfile(resource, directory / "resources.json")
            environment["FQ_CI_REPORT_DIR"] = str(directory)
        selection = directory / "selection"
        collect = (
            *command,
            "--collect-only",
            "-n",
            "0",
            "-p",
            "tools.ci.pytest_shard",
            "--fq-shard-count",
            str(count),
            "--fq-shard-index",
            str(index),
            "--fq-write-selected-tests",
            str(selection),
            "--junitxml",
            str(directory / "collection.xml"),
        )
        print("+ " + " ".join(collect), flush=True)
        result = subprocess.run(collect, cwd=cwd, env=environment)
        if result.returncode:
            return result.returncode
        nodes, exclusive = _verified_selection(selection)
        environment = dict(os.environ)
        if exclusive and int(environment.get("FQ_PYTEST_WORKERS", "1")) > 1:
            environment, code = _exclusive_environment(command[0], directory, cwd)
            if code:
                return code
        execute = (
            *command,
            "@" + str(nodes),
            "-p",
            "tools.ci.pytest_shard",
            "--fq-shard-count",
            "1",
            "--fq-shard-index",
            "0",
        )
        if report := environment.get("FQ_CI_REPORT_DIR"):
            execute += ("--junitxml", str(Path(report) / "pytest.xml"))
        if exclusive:
            execute += ("-n", "1")
        print("+ " + " ".join(execute), flush=True)
        return subprocess.run(execute, cwd=cwd, env=environment).returncode


def print_tiers() -> None:
    for tier in CI_TIERS.values():
        blocking = "default-pr" if tier.blocks_default_pr else "on-demand"
        print(f"{tier.name} [{blocking}]")
        print(f"  trigger: {tier.trigger}")
        print(f"  proves: {tier.proves}")
        print(f"  does_not_prove: {tier.does_not_prove}")
        for command in tier.command_lines():
            print(f"  command: {command}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tier", nargs="?", choices=tuple(CI_TIERS), help="CI/test tier to run"
    )
    parser.add_argument("--list", action="store_true", help="List tiers and commands")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used when running tier commands; defaults to the current interpreter",
    )
    parser.add_argument("--command-index", type=int, help="Run one command of the tier")
    parser.add_argument(
        "--plan-shard",
        action="store_true",
        help="Collect a Runtime shard once before starting its test workers",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list:
        print_tiers()
        return 0
    if args.tier is None:
        print("error: tier is required unless --list is used", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[2]
    tier = CI_TIERS[args.tier]
    commands = tier.rendered_commands(args.python)
    if args.command_index is not None:
        if not 0 <= args.command_index < len(commands):
            print("error: command-index is outside this tier", file=sys.stderr)
            return 2
        commands = (commands[args.command_index],)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        print(
            "error: shard-index must be within a positive shard-count", file=sys.stderr
        )
        return 2
    if args.plan_shard:
        if args.tier != "pr-runtime" or len(commands) != 1:
            print(
                "error: planned collection is supported only for pr-runtime",
                file=sys.stderr,
            )
            return 2
        return run_planned_partition(
            commands[0],
            cwd=root,
            count=args.shard_count,
            index=args.shard_index,
            dry_run=args.dry_run,
        )
    if args.shard_count > 1:
        if any(command[1:3] != ("-m", "pytest") for command in commands):
            print("error: only pytest commands can be sharded", file=sys.stderr)
            return 2
        commands = tuple(
            (
                *command,
                "-p",
                "tools.ci.pytest_shard",
                "--fq-shard-count",
                str(args.shard_count),
                "--fq-shard-index",
                str(args.shard_index),
            )
            for command in commands
        )
    return run_commands(commands, cwd=root, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
