"""Discoverable command line entry point for benchmark runners."""

from __future__ import annotations

import argparse

from .registry import names, resolve


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.runners")
    parser.add_argument(
        "runner",
        nargs="?",
        choices=names(),
        help="runner to execute (omit to list available runners)",
    )
    args, remainder = parser.parse_known_args()
    if args.runner is None:
        parser.print_help()
        return 0
    # Delegate argument parsing to the selected runner to keep one contract.
    import sys

    sys.argv = [f"{parser.prog} {args.runner}", *remainder]
    return resolve(args.runner)()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
