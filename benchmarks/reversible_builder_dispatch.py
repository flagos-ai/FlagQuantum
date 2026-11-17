# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from locked Kickmix circuit_builder.perf.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/build/circuit_builder.perf.cc

"""Verify and time the locked circuit-builder workloads at native boundaries."""

import argparse
import json
from collections.abc import Callable
from hashlib import sha256
from statistics import median
from time import perf_counter_ns

from flagquantum.compiler.artifacts.reversible_pattern import bind_reversible_pattern
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseOperation
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.bits.bit_rotation import bit_rotation_ir
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.primitives._mixed_gates import _flip, _XBasisTarget
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction, IRValidationError

_WIDTH = 256
_TARGET = tuple(range(_WIDTH, 0, -1))
_PATTERN = (ReversiblePhaseOperation("ccx", (1, 0, 2)),)
_COLUMNS = (
    tuple(range(_WIDTH)),
    tuple(range(_WIDTH, 2 * _WIDTH)),
    tuple(range(1, _WIDTH)),
)


def _rotation() -> tuple[Instruction, ...]:
    return bit_rotation_ir(_TARGET, 1, control=0)


def _pattern() -> tuple[ReversiblePhaseOperation, ...]:
    return bind_reversible_pattern(_PATTERN, range(_WIDTH - 1), qubits=_COLUMNS)


def _mixed() -> list[tuple[Instruction, ...] | None]:
    results: list[tuple[Instruction, ...] | None] = []
    for lane in range(_WIDTH):
        bit = ClassicalFieldCoefficient(lane)
        literal = bool(lane % 2)
        controls = (lane, bit, literal)
        target = (lane, _XBasisTarget(bit), _XBasisTarget(literal))[lane // 27 % 3]
        try:
            results.append(_flip(target, controls[lane % 3], controls[lane // 3 % 3]))
        except IRValidationError as error:
            if "cannot repeat a qubit" not in str(error):
                raise
            results.append(None)
    return results


def _record(
    operation: ReversiblePhaseOperation,
) -> tuple[str, tuple[str, ...]]:
    entry = " ".join((operation.opcode.upper(), *(f"q{q}" for q in operation.qubits)))
    condition = operation.condition_bit
    return entry, () if condition is None else (f"b{condition}",)


def _mixed_records(
    results: list[tuple[Instruction, ...] | None],
) -> list[tuple[int, str | list[tuple[str, tuple[str, ...]]]]]:
    records: list[tuple[int, str | list[tuple[str, tuple[str, ...]]]]] = []
    for lane, gates in enumerate(results):
        if gates is None:
            records.append((lane, "collision"))
            continue
        # Provide the possible input bit for verification and discard its fixture definition.
        prefix = tuple(ClassicalXor((), slot).instruction() for slot in range(lane + 1))
        program = lower_reversible_phase(CircuitIR(_WIDTH, (*prefix, *gates)))
        records.append(
            (lane, [_record(op) for op in program.operations[len(prefix) :]])
        )
    return records


def _timing(function: Callable[[], object], iterations: int) -> int | float:
    times = []
    for _ in range(iterations):
        started = perf_counter_ns()
        result = function()
        times.append(perf_counter_ns() - started)
        del result
    return median(times)


def run(iterations: int) -> list[dict[str, int | float | str]]:
    """Verify emitted records and refusals, then time native construction."""
    rotation = _rotation()
    pattern = _pattern()
    mixed = _mixed()
    rotation_records = [
        _record(op)
        for op in lower_reversible_phase(CircuitIR(_WIDTH + 1, rotation)).operations
    ]
    for name, records, expected in (
        (
            "controlled_left_rotate",
            rotation_records,
            "ce10693dd89c04fb82979d341fdb294a5ee32491ec6809f3e1013d2bdc1ba279",
        ),
        (
            "pattern_ccx",
            [_record(op) for op in pattern],
            "81b597ce239c0062423b706f2828114305f2cce2be1ec47c2411a239dacdab8d",
        ),
        (
            "mixed_ccx",
            _mixed_records(mixed),
            "fe9e53521f09fb87943e9d07ee6d71acbdf2de628d24336e6f2f6b87fcf3b272",
        ),
    ):
        if sha256(repr(records).encode()).hexdigest() != expected:
            raise ValueError(f"Source admission, opcode or operand mismatch in {name}")

    return [
        {
            "workload": "controlled_left_rotate",
            "emitted": len(rotation),
            "median_ns": _timing(_rotation, iterations),
        },
        {
            "workload": "pattern_ccx",
            "emitted": len(pattern),
            "median_ns": _timing(_pattern, iterations),
        },
        {
            "workload": "mixed_ccx",
            "admitted_calls": sum(gates is not None for gates in mixed),
            "rejected_calls": mixed.count(None),
            "emitted": sum(len(gates) for gates in mixed if gates is not None),
            "median_ns": _timing(_mixed, iterations),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    print(json.dumps(run(args.iterations), indent=2))


if __name__ == "__main__":
    main()
