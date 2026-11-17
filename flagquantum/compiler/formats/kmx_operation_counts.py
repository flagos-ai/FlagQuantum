# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix main_count_operations.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/cmd/main_count_operations.cc

"""Report exact physical operation counts from KMX or KMB."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from ._binary_streams import write_binary_result
from .kmx_circuit import KmxCircuit, read_circuit_file

_ROWS = (
    "NEG",
    "BIT_INVERT",
    "BIT_STORE0",
    "BIT_STORE1",
    "X",
    "Z",
    "R",
    "HMR",
    "CX",
    "CZ",
    "SWAP",
    "CSWAP",
    "CCX",
    "CCZ",
    "Z_POW",
    "INFO_TAG",
    "DEBUG_PRINT_EMPTY",
    "POP_CONDITION",
    "PUSH_CONDITION",
)
_DEBUG = ("DEBUG_PRINT_EMPTY", "DEBUG_PRINT_Q", "DEBUG_PRINT_C")
_MAGIC = ("CSWAP", "CCX", "CCZ", "Z_POW")
_RULES = (
    ("Metadata Operations", ("NEG", "INFO_TAG")),
    ("Condition Operations", ("PUSH_CONDITION", "POP_CONDITION")),
    ("Bit Operations", ("BIT_INVERT", "BIT_STORE0", "BIT_STORE1")),
    ("Pauli Operations", ("X", "Z")),
    ("Stabilizer Operations", ("CX", "CZ", "SWAP", "R", "HMR")),
    ("Z_POW+CCX+CCZ Operations", _MAGIC),
)
_DIVIDER = "-" * 36


def format_operation_counts(
    circuit: KmxCircuit, *, groups: bool = False, magic: bool = False
) -> str:
    """Render the fixed physical histogram and optional source group totals."""
    counts = Counter(record.opcode for record in circuit)

    def total(names: tuple[str, ...]) -> int:
        return sum(counts[name] for name in names)

    if magic:
        return f"{total(_MAGIC)}\n"

    lines = [
        f"{name:>25}: {total(_DEBUG) if name == _DEBUG[0] else counts[name]}"
        for name in _ROWS
    ]
    if groups:
        lines.extend(
            (
                _DIVIDER,
                f"{'Qubits':>25}: {circuit.qubits}",
                f"{'Bits':>25}: {circuit.bits}",
            )
        )
        lines.extend(f"{name:>25}: {total(ops)}" for name, ops in _RULES)
        lines.extend((_DIVIDER, f"{'Total Operations':>25}: {len(circuit.records)}"))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Count records from a path or standard input."""
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, exit_on_error=False
    )
    parser.add_argument("--in", dest="input", type=Path, help="default: stdin")
    parser.add_argument("--out", dest="output", type=Path, help="default: stdout")
    parser.add_argument(
        "--groups", nargs="?", const="", choices=("",), help="append category totals"
    )
    parser.add_argument(
        "--magic",
        nargs="?",
        const="",
        choices=("",),
        help="emit only the physical magic count",
    )
    try:
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            raise ValueError(f"Unrecognized command line argument {unknown[0]}")
        raw = sys.stdin.buffer.read() if args.input is None else args.input.read_bytes()
        result = format_operation_counts(
            read_circuit_file(raw),
            groups=args.groups is not None,
            magic=args.magic is not None,
        )
        write_binary_result(result.encode(), args.output, sys.stdout.buffer)
    except (argparse.ArgumentError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
