# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix count_operations_sampled and SimInitInstruction.

"""Sample raw physical circuit operation counts from a KMX or KMB file."""

import argparse
import sys
from collections import Counter
from pathlib import Path
from random import Random

from ..compiler.formats._binary_streams import write_binary_result
from ..compiler.formats.kmx_circuit import KmxCircuit, read_circuit_file
from ..compiler.formats.kmx_sampled_report import format_sampled_counts
from ..simulation.physical_basis_shots import sample_basis_batch
from .kmx_sampling_init import (
    parse_register_initialization as parse_register_initialization,
)


def sampled_operation_counts(
    circuit: KmxCircuit, shots: int, *, init: str = "", rng: Random | None = None
) -> Counter[str]:
    """Accumulate packed input-basis trajectories in bounded batches."""
    if type(shots) is not int or not 0 <= shots <= (1 << 63) - 1:
        raise ValueError("Shots must be a nonnegative signed 64-bit integer")
    assignments = parse_register_initialization(circuit, init)
    rng = Random() if rng is None else rng
    counts: Counter[str] = Counter()
    for start in range(0, shots, 64):
        result = sample_basis_batch(
            circuit, shots=min(shots - start, 64), rng=rng, initialization=assignments
        )
        counts.update(result.counts)
    return counts


def main(argv: list[str] | None = None) -> int:
    """Sample a file or stdin stream and write the complete report."""
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, exit_on_error=False
    )
    parser.add_argument("--in", dest="input", type=Path)
    parser.add_argument("--out", dest="output", type=Path)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--init", default="")
    parser.add_argument("--groups", nargs="?", const="", choices=("",))
    try:
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            raise ValueError(f"Unrecognized command line argument {unknown[0]}")
        if args.shots is None:
            raise ValueError("Must specify a value for --shots")
        raw = sys.stdin.buffer.read() if args.input is None else args.input.read_bytes()
        circuit = read_circuit_file(raw)
        counts = sampled_operation_counts(circuit, args.shots, init=args.init)
        report = format_sampled_counts(
            circuit, counts, args.shots, groups=args.groups is not None
        )
        write_binary_result(report.encode(), args.output, sys.stdout.buffer)
    except (argparse.ArgumentError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
