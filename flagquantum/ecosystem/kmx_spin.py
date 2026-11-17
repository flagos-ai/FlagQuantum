# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix main_spin.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/cmd/main_spin.cc

"""Benchmark repeated packed execution of a physical KMX or KMB circuit."""

import argparse
import sys
from pathlib import Path
from time import perf_counter_ns

from ..compiler.formats._binary_streams import write_binary_result
from ..compiler.formats.kmx_circuit import read_circuit_file
from ..compiler.formats.kmx_spin_report import format_spin_report
from ..simulation.physical_basis_spin import spin_basis

_MODES = tuple(
    f"{width}{suffix}" for width in (64, 128, 256, 512) for suffix in ("", "-polyfill")
)


def main(argv: list[str] | None = None) -> int:
    """Read and benchmark the entire input before opening the destination."""
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, exit_on_error=False
    )
    parser.add_argument("--in", dest="input", type=Path)
    parser.add_argument("--out", dest="output", type=Path)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--mode", choices=_MODES)
    try:
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            raise ValueError(f"Unrecognized command line argument {unknown[0]}")
        if args.mode is None or args.shots is None:
            raise ValueError("Spin requires both --mode and --shots")
        width = int(args.mode.partition("-")[0])
        raw = sys.stdin.buffer.read() if args.input is None else args.input.read_bytes()
        circuit = read_circuit_file(raw)
        start = perf_counter_ns()
        hashed = spin_basis(
            circuit.records,
            circuit.qubits,
            circuit.bits,
            shots=args.shots,
            width=width,
        )
        micros = (perf_counter_ns() - start) // 1000
        report = format_spin_report(
            hashed, width, len(circuit.records) * args.shots, micros
        )
        write_binary_result(report.encode(), args.output, sys.stdout.buffer)
    except (argparse.ArgumentError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
