#!/usr/bin/env python
"""Run FlagQuantum test tiers used by local and CI workflows.

This script intentionally orchestrates existing pytest markers and benchmark
audit commands. It does not run performance benchmarks or turn CPU distributed
semantics into scalability evidence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
        commands=(("{python}", "-m", "pytest", "-m", "integration", "-q"),),
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
        proves="Tests marked for accelerator-backed distributed behavior on the configured hardware.",
        does_not_prove="Multi-node transport or release scalability without benchmark audit evidence.",
        commands=(
            ("{python}", "-m", "pytest", "-m", "distributed_accel and gpu", "-q"),
        ),
    ),
    "multinode-scheduled": CITier(
        name="multinode-scheduled",
        trigger="Scheduled or manual torchrun/cluster job with explicit rank placement.",
        proves="Selected multi-node transport candidates for the configured cluster.",
        does_not_prove="Release scalability unless the produced payload also passes the release gate.",
        commands=(("{python}", "-m", "pytest", "-m", "distributed_multinode", "-q"),),
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list:
        print_tiers()
        return 0
    if args.tier is None:
        print("error: tier is required unless --list is used", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    tier = CI_TIERS[args.tier]
    commands = tier.rendered_commands(args.python)
    return run_commands(commands, cwd=root, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
