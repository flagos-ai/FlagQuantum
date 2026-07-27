"""Discoverable command line entry point for benchmark runners."""

from __future__ import annotations

import argparse
import sys

from .registry import describe, names, resolve, specs


def _print_list() -> None:
    print("Available FlagQuantum benchmarks:\n")
    width = max(len(item.name) for item in specs())
    for item in specs():
        print(f"  {item.name:<{width}}  [{item.category}] {item.summary}")
    print("\nUse `flagquantum-benchmark info NAME` for requirements and an example.")


def _print_info(name: str) -> None:
    runner = describe(name)
    print(runner.name)
    print(f"  Category : {runner.category}")
    print(f"  Hardware : {runner.hardware}")
    print(f"  Summary  : {runner.summary}")
    if runner.example:
        print(f"  Example  : {runner.example}")


def main() -> int:
    raw_arguments = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="flagquantum-benchmark",
        description="Discover and run maintained FlagQuantum benchmarks.",
    )
    parser.add_argument("command", nargs="?", help="list, info, run, or a runner name")
    parser.add_argument("name", nargs="?", help="benchmark runner name")
    args, remainder = parser.parse_known_args()
    if args.command in (None, "list"):
        _print_list()
        return 0
    if args.command == "info":
        if args.name is None:
            parser.error("info requires a benchmark name")
        if args.name not in names():
            parser.error(
                f"unknown benchmark {args.name!r}; " "use `flagquantum-benchmark list`"
            )
        _print_info(args.name)
        return 0
    if args.command == "run":
        if args.name is None:
            parser.error("run requires a benchmark name")
        runner_name = args.name
    else:
        # Backward-compatible shorthand:
        # ``python -m flagquantum.benchmarking environment_probe ...``.
        runner_name = args.command
        remainder = raw_arguments[1:]
    if runner_name not in names():
        parser.error(
            f"unknown benchmark {runner_name!r}; use `flagquantum-benchmark list`"
        )
    sys.argv = [f"{parser.prog} run {runner_name}", *remainder]
    return resolve(runner_name)()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
