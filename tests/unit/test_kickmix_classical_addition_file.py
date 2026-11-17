"""Compare complete mixed arithmetic streams and source benchmark workloads."""

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import TypedDict

import pytest
import torch
from typing_extensions import NotRequired

from benchmarks.benchmark_physical_classical_addition import (
    benchmark_classical_addition,
    run_workload,
)
from flagquantum.compiler.formats.kmx_block_marks import encode_kmx_block_marks
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.arithmetic.physical_addition import (
    physical_carry_parity,
    physical_mixed_addition,
    physical_vented_addition,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as C,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from tests.unit.measurement_checks import expand_reset_branches, forced_branch


class _Stream(TypedDict):
    case: list[int]
    sha256: str


class _Benchmark(TypedDict):
    sha256: str
    opcodes: dict[str, int]
    kmb_sha256: NotRequired[str]
    collision_free_sha256: NotRequired[str]


_SOURCE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "kickmix_classical_addition_file.json"
    ).read_text()
)
_STREAMS: list[_Stream] = _SOURCE["streams"]
_BENCHMARKS: dict[str, _Benchmark] = _SOURCE["benchmarks"]


def test_complete_physical_mixed_streams() -> None:
    for example in _STREAMS:
        mode, n, m, c, d, k, kind, carry_kind, enabled = example["case"]
        assert carry_kind == 1 and enabled
        target, dst = tuple(reversed(range(n))), tuple(reversed(range(n, n + k)))
        offset = (
            tuple(C(bit) for bit in reversed(range(m)))
            if kind == 1
            else tuple(bit % 3 == 0 for bit in reversed(range(m)))
        )
        carry = C(m if kind == 1 else 0)
        inputs = carry.slot + 1
        control = n + k
        clean = tuple(range(control + 1, control + 1 + c))
        if mode == 0:
            program, calls = physical_carry_parity(
                target,
                offset,
                dst,
                carry_in=carry,
                control=control,
                clean=clean,
                classical_width=inputs,
            )
        elif mode == 1:
            vents = tuple(range(inputs, inputs + max(n, k) + 1))
            program, calls = physical_vented_addition(
                target,
                offset,
                carry_copy=dst,
                vent_bits=vents,
                carry_in=carry,
                control=control,
                clean=clean,
                classical_width=inputs + len(vents),
            )
        else:
            program, calls = physical_mixed_addition(
                target,
                offset,
                clean=clean,
                dirty=tuple(range(control + 1 + c, control + 1 + c + d)),
                carry_in=carry,
                control=control,
                classical_width=inputs,
                simple=mode == 4,
                subtract=mode == 3,
            )
        body = "\n".join(encode_kmx_block_marks(program, calls).splitlines()[1:])
        assert hashlib.sha256(body.encode()).hexdigest() == example["sha256"], example


@pytest.mark.parametrize("mode", range(4))
def test_source_benchmark_workload_fingerprints(mode: int) -> None:
    expected = _BENCHMARKS[str(mode)]
    run = run_workload(mode, kmb="text" if mode == 0 else None)
    assert run.opcodes == expected["opcodes"]
    assert hashlib.sha256(run.text.encode()).hexdigest() == expected.get(
        "collision_free_sha256", expected["sha256"]
    )
    if mode == 1:
        aliased = run.text.replace(
            "HMR q512 b256\nCZ q510 q511 if b256",
            "HMR q512 b0\nCZ q510 q511 if b0",
        )
        assert hashlib.sha256(aliased.encode()).hexdigest() == expected["sha256"]
    if mode == 0:
        columns = run_workload(mode, kmb="columns")
        assert run.binary is not None and columns.binary == run.binary
        assert hashlib.sha256(run.binary).hexdigest() == expected["kmb_sha256"]


def test_runs_all_six_benchmark_modes() -> None:
    report = benchmark_classical_addition(warmup=0, repeats=1)
    assert len(report) == 6


@pytest.mark.parametrize("subtract", [False, True])
def test_physical_mixed_carry_and_work_on_every_complex_branch(subtract: bool) -> None:
    program, _ = physical_mixed_addition(
        (0, 1, 2),
        (C(2), C(1), C(0)),
        clean=(4, 5),
        dirty=(),
        control=3,
        carry_in=C(3),
        classical_width=4,
        subtract=subtract,
    )
    state = torch.zeros(64, dtype=torch.complex128)
    values = torch.arange(16)
    psi = torch.complex((values + 1).double(), (3 - values).double())
    psi /= psi.norm()
    state[values << 2] = psi
    input_bits = tuple(
        ClassicalXor((), bit, bit in (0, 2, 3)).instruction()
        for bit in program.input_bits
    )
    out = torch.zeros_like(state)
    target = ((values >> 1) + (-6 if subtract else 6) * (values & 1)) % 8
    out[((target << 1) | (values & 1)) << 2] = psi
    gates = (
        *input_bits,
        *expand_reset_branches(
            bind_reversible_phase_execution(program), first_slot=program.n_bits + 2
        ),
    )
    assert sum(g.name == "measure" for g in gates) == 4
    for outcomes in product((0, 1), repeat=4):
        torch.testing.assert_close(
            forced_branch(state, gates, outcomes), out / 4, atol=5e-14, rtol=5e-14
        )
