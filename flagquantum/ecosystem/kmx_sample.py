# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix main_sample.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/cmd/main_sample.cc

"""Sample raw KMX or KMB registers and report the paired basis trajectories."""

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path
from random import Random
from typing import Literal, TextIO

from ..compiler.artifacts.phase_angle import PhaseAngle128
from ..compiler.formats._binary_streams import write_binary_result
from ..compiler.formats.kmx_circuit import KmxCircuit, read_circuit_file
from ..compiler.formats.kmx_sample_report import RegisterSample, format_sample_report
from ..simulation.physical_basis_shots import (
    PHYSICAL_SAMPLE_BATCH_SIZE,
    sample_basis_batch,
)
from .kmx_sampling_init import parse_register_initialization


def sample_registers(
    circuit: KmxCircuit,
    shots: int,
    *,
    init: str = "",
    rng: Random | None = None,
    debug: TextIO | None = None,
) -> Iterator[RegisterSample]:
    """Yield one packed-basis row per shot from a KMX or KMB circuit."""
    if type(shots) is not int or not 0 <= shots < 1 << 32:
        raise ValueError("Shots must be an unsigned 32-bit integer")
    assignments = parse_register_initialization(circuit, init)
    generator = Random() if rng is None else rng

    def observe(value: int, phase: bool) -> None:
        assert debug is not None
        if phase:
            angle = PhaseAngle128(value & ((1 << 64) - 1), value >> 64)
            debug.write(f" (phase={angle})\n")
        else:
            debug.write(str(value))

    for start in range(0, shots, PHYSICAL_SAMPLE_BATCH_SIZE):
        count = min(shots - start, PHYSICAL_SAMPLE_BATCH_SIZE)
        batch = sample_basis_batch(
            circuit,
            shots=count,
            rng=generator,
            initialization=assignments,
            capture_before=True,
            debug=observe if debug is not None else None,
        )
        assert batch.before is not None

        def columns(q: dict[int, int], b: dict[int, int], shot: int) -> tuple[int, ...]:
            return tuple(
                sum(
                    (((q if kind == "q" else b).get(wire, 0) >> shot) & 1) << i
                    for i, (kind, wire) in enumerate(register.members)
                )
                for register in circuit.registers
            )

        for shot in range(count):
            initial = columns(batch.before.qubits, batch.before.bits, shot)
            final = columns(batch.qubits, batch.bits, shot)
            yield RegisterSample(initial, final, bool((batch.sign_mask >> shot) & 1))


def main(argv: list[str] | None = None) -> int:
    """Read and sample the complete input before writing the report."""
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, exit_on_error=False
    )
    parser.add_argument("--in", dest="input", type=Path)
    parser.add_argument("--out", dest="output", type=Path)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--init", default="")
    parser.add_argument("--decimal", action="store_true")
    parser.add_argument("--hex", action="store_true")
    try:
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            raise ValueError(f"Unrecognized command line argument {unknown[0]}")
        raw = sys.stdin.buffer.read() if args.input is None else args.input.read_bytes()
        circuit = read_circuit_file(raw)
        radix: Literal[2, 10, 16] = 16 if args.hex else 10 if args.decimal else 2
        report = format_sample_report(
            circuit.registers,
            sample_registers(circuit, args.shots, init=args.init, debug=sys.stderr),
            radix=radix,
        )
        write_binary_result(report.encode(), args.output, sys.stdout.buffer)
    except (argparse.ArgumentError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
