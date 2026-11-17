# SPDX-License-Identifier: Apache-2.0
# Workloads adapted from the locked Kickmix gen_iadd_classical.perf.cc.

"""Benchmark physical mixed addition and carry XOR generation."""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from statistics import median
from time import perf_counter_ns
from typing import Literal

from flagquantum.compiler.formats.kmb import KmxRegister, encode_kmb
from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
from flagquantum.compiler.formats.kmx_builder import KmxCircuitBuilder
from flagquantum.compiler.registers.arithmetic.physical_addition import (
    physical_carry_parity,
    physical_mixed_addition,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)

_CONSTANT = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
_Workload = tuple[str, int, int, Literal["text", "columns"] | None]
_WORKLOADS: tuple[_Workload, ...] = (
    ("carry_kmb_text", 0, 1762, "text"),
    ("carry_kmb_columns", 0, 1762, "columns"),
    ("carry_classical_bits", 1, 1792, None),
    ("carry_constant", 0, 1762, None),
    ("addition_two_clean", 2, 5792, None),
    ("addition_partial_clean", 3, 4747, None),
)


@dataclass(frozen=True)
class _WorkloadResult:
    text: str
    binary: bytes | None
    opcodes: Counter[str]


def run_workload(
    mode: int, *, kmb: Literal["text", "columns"] | None = None
) -> _WorkloadResult:
    """Build a paired source-layout workload with optional KMB serialization."""
    src, dst = tuple(range(255)), tuple(range(255, 510))
    ports = (src, dst, (510,), (511,))
    control, carry = 511, 510
    clean_stop = 612 if mode == 3 else 514
    clean = tuple(range(512, clean_stop))
    offset = tuple(
        ClassicalFieldCoefficient(bit) if mode == 1 else bool((_CONSTANT >> bit) & 1)
        for bit in reversed(range(256))
    )
    inputs = 256 if mode == 1 else 0
    if mode < 2:
        program, calls = physical_carry_parity(
            src[::-1],
            offset,
            dst[::-1],
            carry_in=carry,
            control=control,
            clean=clean,
            classical_width=inputs,
        )
    else:
        program, calls = physical_mixed_addition(
            src[::-1],
            offset,
            carry_in=carry,
            control=control,
            clean=clean,
            dirty=tuple(range(clean_stop, clean_stop + 256)),
            classical_width=inputs,
        )
    registers = tuple(
        KmxRegister("", tuple(("q", q) for q in wires)) for wires in ports
    )
    physical = encode_kmx_block_marks(program, calls).splitlines()[1:]
    declarations = [
        line for i, register in enumerate(registers) for line in register.to_kmx(i)
    ]
    text = "\n".join((*declarations, *physical))
    binary = None
    if kmb == "text":
        binary = encode_kmb(text)
    if kmb == "columns":
        builder = KmxCircuitBuilder()
        builder.replace_registers(registers)
        builder.append_kmx_text("\n".join(physical))
        binary = builder.to_kmb()
    return _WorkloadResult(
        text, binary, Counter(line.partition(" ")[0] for line in physical)
    )


def benchmark_classical_addition(
    *, warmup: int = 1, repeats: int = 5
) -> dict[str, object]:
    """Time four generators and two safe KMB serialization routes."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    outputs: dict[str, object] = {}
    for name, mode, records, writer in _WORKLOADS:
        times = []
        last = None
        for i in range(warmup + repeats):
            start = perf_counter_ns()
            last = run_workload(mode, kmb=writer)
            elapsed = perf_counter_ns() - start
            if sum(last.opcodes.values()) != records:
                raise RuntimeError(
                    f"Unexpected record count for {name}: {last.opcodes}"
                )
            if i >= warmup:
                times.append(elapsed)
        assert last is not None
        seconds = median(times) / 1e9
        outputs[name] = {
            "records_including_marks": records,
            "median_seconds": seconds,
            "records_per_second": records / seconds,
            "kmx_sha256": sha256(last.text.encode()).hexdigest(),
            "kmb_sha256": sha256(last.binary).hexdigest()
            if last.binary is not None
            else None,
        }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark_classical_addition(warmup=args.warmup, repeats=args.repeats),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
