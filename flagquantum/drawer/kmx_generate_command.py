# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix main_generate_circuit.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/cmd/main_generate_circuit.cc

"""Generate physical arithmetic and table circuits as KMX, KMB or text diagrams."""

import argparse
import sys
from math import inf
from pathlib import Path

from ..compiler.formats._binary_streams import write_binary_result
from ..compiler.formats.kmb import encode_kmb
from ..compiler.formats.unsigned_word import parse_unsigned_word
from ..compiler.registers.circuit_generation import (
    generate_addition,
    generate_classical_addition,
    generate_comparison,
    generate_modular_doubling,
    generate_table,
)
from .reversible_phase_analysis import render_reversible_phase_analysis_svg
from .reversible_phase_diagram import render_circuit_text


def _size(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 1_000_000:
        raise argparse.ArgumentTypeError("Expected a size between 0 and 1000000")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Build and validate the complete physical stream before writing output."""
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, exit_on_error=False
    )
    parser.add_argument(
        "--kind",
        choices=(
            "iadd",
            "iadd_classical",
            "iadd_classical_simple",
            "imul2_mod",
            "flip_if_lt",
            "lookup",
            "unlookup",
        ),
    )
    for flag in ("n", "w", "clean", "dirty"):
        parser.add_argument(f"--{flag}", type=_size, default=-1)
    for flag in (
        "kmb",
        "diagram",
        "controlled",
        "minimize_qubits",
        "validate",
        "skip_output",
    ):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--mod")
    parser.add_argument("--btol", type=float, default=inf)
    parser.add_argument("--flame_out", type=Path)
    parser.add_argument("--out", type=Path)
    try:
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            raise ValueError(f"Unrecognized command line argument {unknown[0]}")
        if not args.btol >= 0:
            raise ValueError("Bit tolerance must be nonnegative")
        n = args.n
        if (
            args.kind
            in ("iadd", "iadd_classical", "iadd_classical_simple", "flip_if_lt")
            and n == -1
        ):
            raise ValueError("Need '--n #'.")
        if args.kind == "iadd":
            circuit = generate_addition(
                n,
                clean_count=max(0, args.clean),
                dirty_count=max(0, args.dirty),
                controlled=args.controlled,
                minimize_qubits=args.minimize_qubits,
            )
        elif args.kind in ("iadd_classical", "iadd_classical_simple"):
            simple = args.kind == "iadd_classical_simple"
            circuit = generate_classical_addition(
                n,
                clean_count=n if args.clean == -1 else args.clean,
                dirty_count=(0 if simple else n) if args.dirty == -1 else args.dirty,
                controlled=args.controlled,
                minimize_qubits=args.minimize_qubits,
                btol=args.btol,
                simple=simple,
            )
        elif args.kind == "imul2_mod":
            modulus = None
            if args.mod is not None:
                if n != -1:
                    raise ValueError("Specify '--mod #' xor '--n #'.")
                modulus, n = parse_unsigned_word(args.mod)
            circuit = generate_modular_doubling(
                n,
                modulus=modulus,
                clean_count=n if args.clean == -1 else args.clean,
                dirty_count=max(0, args.dirty),
                controlled=args.controlled,
                minimize_qubits=args.minimize_qubits,
                btol=args.btol,
            )
        elif args.kind == "flip_if_lt":
            circuit = generate_comparison(
                n,
                clean_count=n if args.clean == -1 else args.clean,
                controlled=args.controlled,
                minimize_qubits=args.minimize_qubits,
                btol=args.btol,
            )
        elif args.kind in ("lookup", "unlookup"):
            n = 3 if n == -1 else n
            w = 2 if args.w == -1 else args.w
            erase = args.kind == "unlookup"
            clean = args.clean
            if clean == -1:
                clean = max(n, 1 << (n // 2)) if erase and w and n <= 30 else n
            circuit = generate_table(
                n, w, clean_count=clean, controlled=args.controlled, erase=erase
            )
        else:
            raise ValueError("Need '--kind'.")
        kmx = circuit.to_kmx()
        if args.flame_out is not None:
            args.flame_out.write_text(
                render_reversible_phase_analysis_svg(kmx, reference_qubit_count=n),
                encoding="utf-8",
            )
        if args.skip_output:
            return 0
        if args.diagram:
            data = (render_circuit_text(circuit, use_unicode=True) + "\n").encode()
        elif args.kmb:
            data = encode_kmb(kmx, file_text=True)
        else:
            data = (kmx + "\n").encode() if kmx else b""
        write_binary_result(data, args.out, sys.stdout.buffer)
    except (argparse.ArgumentError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
