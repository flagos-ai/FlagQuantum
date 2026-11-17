# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix count_operations_sampled and SimInitInstruction.

"""Sample raw physical circuit operation counts from a KMX or KMB file."""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from random import Random

from ..compiler.formats._binary_streams import write_binary_result
from ..compiler.formats.kmx_circuit import KmxCircuit, read_circuit_file
from ..compiler.formats.kmx_sampled_report import format_sampled_counts
from ..simulation.physical_basis_shots import RegisterInitialization, sample_basis_batch

_INIT = re.compile(r"r([0-9]*)([=<])(.*)")
_VALUE = re.compile(r"0b[01_]*|0x[0-9a-fA-F_]*|[0-9_]+")


def parse_register_initialization(
    circuit: KmxCircuit, text: str
) -> tuple[RegisterInitialization, ...]:
    """Resolve source integer initialization strings to physical members."""
    assignments = []
    for item in text.removesuffix(",").split(",") if text else ():
        match = _INIT.fullmatch(item)
        if match is None:
            raise ValueError(f"Invalid register initialization {item!r}")
        index, operator, payload = match.groups()
        register = int(index or "0")
        if register >= len(circuit.registers):
            raise ValueError(f"Initialization register r{register} is not declared")
        randomize = operator == "<" or payload == "*"
        if operator == "=" and payload == "*":
            value = 0
        else:
            payload = payload.strip(" ")
            if not _VALUE.fullmatch(payload):
                raise ValueError(f"Invalid register initialization value {payload!r}")
            base = {"0b": 2, "0x": 16}.get(payload[:2], 10)
            digits = (payload[2:] if base != 10 else payload).replace("_", "")
            value = int(digits or "0", base)
            if randomize and value == 0 and digits and base != 10:
                raise ValueError("Random initialization modulus is zero")
        assignments.append(
            RegisterInitialization(
                circuit.registers[register].members, value, randomize
            )
        )
    return tuple(assignments)


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
