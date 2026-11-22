"""Compare full multi-control binding records and every small measured branch."""

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.control.physical_multi_control import (
    physical_multi_controlled_x,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.primitives._mixed_gates import _XBasisTarget
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import expand_reset_branches, forced_branch

pytestmark = pytest.mark.unit
_Pattern = Literal["q", "mixed", "duplicate", "false"]
_Port = int | bool | Bit
_Case = tuple[int, int, _Pattern, int, int, str]


def binding_ports(
    case: _Case,
) -> tuple[tuple[_Port, ...], tuple[int, ...], tuple[int, ...], int]:
    n, m, pattern, work, _, _ = case
    controls: tuple[_Port, ...] = tuple(range(n))
    if pattern == "mixed":
        controls = (Bit(2), *controls[: n // 2], True, Bit(0), *controls[n // 2 :])
    elif pattern == "duplicate":
        controls = (Bit(2), *controls, Bit(2))
    elif pattern == "false":
        controls = (*controls, Bit(0), False)
    quantum = n + m + 1
    return (
        controls,
        tuple(range(n, n + m)),
        tuple(reversed(range(quantum, quantum + work))),
        quantum,
    )


def test_complete_source_records_and_workspace_return_order() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("multi_control_binding.json")
        .read_text()
    )
    assert fixture["source_revision"] == "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453"
    cases: list[_Case] = fixture["cases"]
    assert cases
    for case in cases:
        controls, target, work, _ = binding_ports(case)
        used = max(0, case[0] - 2) if case[1] and case[2] != "false" else 0
        returned = (*reversed(work[:used]), *work[used:])
        streams = []
        for leases in (work, returned):
            program = physical_multi_controlled_x(controls, target, workspace=leases)
            text = encode_kmx(program)
            streams.append("\n".join(text.splitlines()[1:]))
            assert encode_kmx(decode_kmx(text)) == text
        records = "\n".join(stream for stream in streams if stream)
        assert len(records.splitlines()) == case[4], case
        assert hashlib.sha256(records.encode()).hexdigest() == case[5], case


@pytest.mark.parametrize(
    "n,targets,enabled",
    [(n, m, e) for n, m, e in product(range(5), (1, 2), (False, True))],
)
def test_all_small_branches_complex_amplitudes_and_work_zero(
    n: int, targets: int, enabled: bool
) -> None:
    logical = n + targets + 1
    count = max(0, n - 2)
    width = logical + count
    controls: tuple[_Port, ...] = (Bit(3), *range(n), Bit(3))
    program = physical_multi_controlled_x(
        controls, tuple(range(n, n + targets)), workspace=tuple(range(logical, width))
    )
    setup = (ClassicalXor((), 3, enabled).instruction(),)
    gates = expand_reset_branches(
        (*setup, *bind_reversible_phase_execution(program)), first_slot=100
    )
    inputs = torch.arange(1 << logical)
    amplitudes = torch.complex((inputs + 1).double(), (7 - 2 * inputs).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    initial[inputs << count] = amplitudes
    active = ((inputs >> (targets + 1)) == ((1 << n) - 1)) & enabled
    mask = sum(1 << (logical - 1 - wire) for wire in range(n, n + targets))
    expected = torch.zeros_like(initial)
    expected[(inputs ^ (active.to(torch.int64) * mask)) << count] = amplitudes
    events = sum(gate.name == "measure" for gate in gates)
    assert events == 2 * count
    total = torch.zeros_like(initial.real)
    for history in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, history)
        branch = (
            expected / 2 ** (events / 2)
            if enabled
            else (torch.zeros_like(expected) if any(history) else expected)
        )
        torch.testing.assert_close(actual, branch, atol=6e-14, rtol=6e-14)
        total += actual.abs().square()
    torch.testing.assert_close(total, expected.abs().square())
    if count:
        assert torch.count_nonzero(total.reshape(-1, 1 << count)[:, 1:]) == 0


def test_append_and_measured_primitive_counts() -> None:
    program = physical_multi_controlled_x(
        (0, Bit(2), 1, 2, 3), (4, 5), workspace=(7, 6)
    )
    counts = Counter(op.opcode for op in program.operations)
    assert counts == {
        "push_condition": 1,
        "pop_condition": 1,
        "cx": 2,
        "r": 2,
        "ccx": 3,
        "hmr": 2,
        "cz": 2,
    }
    circuit = DynamicCircuit(program.n_qubits, dtype=torch.complex128)
    for wire in range(4):
        circuit.x(wire)
    circuit.append(
        FieldOperation(
            program.n_qubits,
            (
                ClassicalXor((), 2, True).instruction(),
                *bind_reversible_phase_execution(program),
            ),
        )
    )
    result = run_dynamic(circuit, shots=8, seed=17, strategy="batched")
    expected = torch.zeros((8, 256), dtype=torch.complex128)
    expected[:, 252] = 1
    torch.testing.assert_close(result.final_states.reshape(8, -1), expected)


def test_word_conversion_precedes_skip_and_workspace_validation() -> None:
    for controls in (True, 0, Bit(0)):
        with pytest.raises(ValueError, match="sequence"):
            physical_multi_controlled_x(controls, (1,))
    with pytest.raises(ValueError, match="Z-basis"):
        physical_multi_controlled_x((False, _XBasisTarget(True)), (1,))
    with pytest.raises(ValueError, match="quantum-only"):
        physical_multi_controlled_x((False,), (Bit(0),))
    assert not physical_multi_controlled_x((0, 1, 2, False), (3,)).operations
    assert not physical_multi_controlled_x((0, 1, 2), ()).operations
    with pytest.raises(ValueError, match="n-2"):
        physical_multi_controlled_x((0, 1, 2), (3,))
    with pytest.raises(ValueError, match="distinct"):
        physical_multi_controlled_x((0, 1, 2), (3,), workspace=(0,))


def test_repeated_targets_retain_source_order_for_zero_and_one_controls() -> None:
    assert [op.opcode for op in physical_multi_controlled_x((), (0, 0)).operations] == [
        "x",
        "x",
    ]
    assert [
        op.qubits for op in physical_multi_controlled_x((1,), (0, 0)).operations
    ] == [(1, 0), (1, 0)]
