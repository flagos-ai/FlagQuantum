"""Check complete addition binding dispatch and coherent measurement branches."""

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Literal, cast

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.arithmetic.addition_binding import (
    physical_addition_binding,
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
_Kind = Literal["q", "b", "literal", "int"]
_Case = tuple[int, int, _Kind, str, int, str, bool, int, str]
_Port = int | bool | Bit
_Offset = int | tuple[_Port, ...]


def binding_ports(
    case: _Case,
) -> tuple[tuple[int, ...], _Offset, _Port, tuple[int, ...], int, int]:
    """Keep LSB source allocation, MSB native views and LIFO clean leases."""
    n, m, kind, control_kind, work, _, _, _, _ = case
    quantum = n + (m if kind == "q" else 0) + 1
    count = m if kind == "b" else 0
    offset: _Offset
    if kind == "int":
        offset = -(1 << (n - 1)) + 1 if n else 0
    elif kind == "q":
        offset = tuple(reversed(range(n, n + m)))
    elif kind == "b":
        offset = tuple(Bit(k) for k in reversed(range(m)))
    else:
        offset = tuple(k % 3 == 0 for k in reversed(range(m)))
    control: _Port = quantum - 1
    if control_kind == "b":
        control = Bit(count)
        count += 1
    elif control_kind != "q":
        control = control_kind == "true"
    return (
        tuple(reversed(range(n))),
        offset,
        control,
        tuple(reversed(range(quantum, quantum + work))),
        count,
        quantum,
    )


def test_complete_source_records_and_refusals() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("addition_binding.json")
        .read_text()
    )
    assert fixture["source_revision"] == "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453"
    cases: list[_Case] = fixture["cases"]
    refusals: list[_Case] = fixture["source_rejections"]
    assert cases and refusals
    for case in cases:
        target, offset, control, work, _, _ = binding_ports(case)
        program, calls = physical_addition_binding(
            offset,
            target=target,
            control=control,
            workspace=work,
            btol=float(case[5]),
            subtract=case[6],
        )
        encoded = encode_kmx_block_marks(program, calls)
        records = "\n".join(encoded.splitlines()[1:])
        assert len(records.splitlines()) == case[7], case
        assert hashlib.sha256(records.encode()).hexdigest() == case[8], case
        decoded, decoded_calls = decode_kmx_block_marks(encoded)
        assert encode_kmx_block_marks(decoded, decoded_calls) == encoded
    for case in refusals:
        target, offset, control, work, _, _ = binding_ports(case)
        with pytest.raises(ValueError):
            physical_addition_binding(
                offset,
                target=target,
                control=control,
                workspace=work,
                btol=float(case[5]),
                subtract=case[6],
            )


@pytest.mark.parametrize(
    "subtract,classical_offset,guard", list(product((False, True), repeat=3))
)
def test_complete_complex_branches_and_work_zero(
    subtract: bool, classical_offset: bool, guard: bool
) -> None:
    logical = 3 if classical_offset else 5
    clean = (logical + 1, logical)
    width = logical + 2
    target = (1, 0)
    offset: _Offset = (Bit(0), Bit(1)) if classical_offset else (3, 2)
    control = Bit(4)
    program, _ = physical_addition_binding(
        offset, target=target, control=control, workspace=clean, subtract=subtract
    )
    values = {0: True, 1: False, 2: False, 3: False, 4: guard}
    setup = tuple(
        ClassicalXor((), bit, values[bit]).instruction() for bit in program.input_bits
    )
    gates = expand_reset_branches(
        (*setup, *bind_reversible_phase_execution(program)), first_slot=100
    )
    indices = torch.arange(1 << logical)
    amplitude = torch.complex((indices + 1).double(), (7 - 2 * indices).double())
    amplitude /= amplitude.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    initial[indices << 2] = amplitude
    target_value = ((indices >> (logical - 1)) & 1) + 2 * (
        (indices >> (logical - 2)) & 1
    )
    rhs = (
        torch.full_like(indices, 2)
        if classical_offset
        else (((indices >> (logical - 3)) & 1) + 2 * ((indices >> (logical - 4)) & 1))
    )
    changed = (
        (target_value + (-rhs if subtract else rhs)) % 4 if guard else target_value
    )
    output = (
        indices
        ^ (((target_value ^ changed) & 1) << (logical - 1))
        ^ ((((target_value ^ changed) >> 1) & 1) << (logical - 2))
    )
    expected = torch.zeros_like(initial)
    expected[output << 2] = amplitude
    events = sum(gate.name == "measure" for gate in gates)
    total = torch.zeros_like(initial.real)
    for history in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, history)
        branch = (
            expected / 2 ** (events / 2)
            if guard
            else (torch.zeros_like(expected) if any(history) else expected)
        )
        torch.testing.assert_close(actual, branch, atol=5e-14, rtol=5e-14)
        total += actual.abs().square()
    torch.testing.assert_close(total, expected.abs().square())
    assert torch.count_nonzero(total.reshape(-1, 4)[:, 1:]) == 0


def test_append_classical_guard_and_resource_records() -> None:
    program, calls = physical_addition_binding(
        -1, target=(1, 0), control=Bit(3), workspace=(3, 2)
    )
    circuit = DynamicCircuit(program.n_qubits, dtype=torch.complex128)
    setup = (ClassicalXor((), 3, True).instruction(),)
    circuit.append(
        FieldOperation(
            program.n_qubits, (*setup, *bind_reversible_phase_execution(program))
        )
    )
    result = run_dynamic(circuit, shots=8, seed=19, strategy="batched")
    expected = torch.zeros((8, 16), dtype=torch.complex128)
    expected[:, 12] = 1
    torch.testing.assert_close(result.final_states.reshape(8, -1), expected)
    assert calls and all(call.stop <= len(program.operations) for call in calls)
    resources = Counter(op.opcode for op in program.operations)
    assert resources["push_condition"] == resources["pop_condition"]
    assert resources["push_condition"] >= 1


def test_conversion_skip_and_quantum_tolerance_order() -> None:
    program, calls = physical_addition_binding(
        object(), target=(), control=cast(int, _XBasisTarget(True))
    )
    assert not program.operations and not calls
    program, calls = physical_addition_binding(
        (0, Bit(0)), target=(1,), control=False, btol=2
    )
    assert not program.operations and not calls
    with pytest.raises(ValueError, match="range"):
        physical_addition_binding(4, target=(0, 1), control=False)
    with pytest.raises(ValueError, match="btol"):
        physical_addition_binding((2, 3), target=(0, 1), btol=1)
    with pytest.raises(ValueError, match="quantum-only"):
        physical_addition_binding((2, Bit(0)), target=(0, 1))


@pytest.mark.parametrize(
    "subtract,classical_offset", list(product((False, True), repeat=2))
)
def test_quantum_enable_all_measurement_branches(
    subtract: bool, classical_offset: bool
) -> None:
    logical = 4 if classical_offset else 6
    work = (logical + 1, logical)
    offset: _Offset = (Bit(0), Bit(1)) if classical_offset else (3, 2)
    control = logical - 2
    program, _ = physical_addition_binding(
        offset, target=(1, 0), control=control, workspace=work, subtract=subtract
    )
    setup = tuple(
        ClassicalXor((), bit, bit == 0).instruction() for bit in program.input_bits
    )
    gates = expand_reset_branches(
        (*setup, *bind_reversible_phase_execution(program)), first_slot=100
    )
    indices = torch.arange(1 << logical)
    amplitude = torch.complex((indices + 1).double(), (3 - indices).double())
    amplitude /= amplitude.norm()
    initial = torch.zeros(1 << (logical + 2), dtype=torch.complex128)
    initial[indices << 2] = amplitude
    value = ((indices >> (logical - 1)) & 1) + 2 * ((indices >> (logical - 2)) & 1)
    rhs = (
        torch.full_like(indices, 2)
        if classical_offset
        else (((indices >> (logical - 3)) & 1) + 2 * ((indices >> (logical - 4)) & 1))
    )
    changed = (value + (-rhs if subtract else rhs) * ((indices >> 1) & 1)) % 4
    output = (
        indices
        ^ (((value ^ changed) & 1) << (logical - 1))
        ^ ((((value ^ changed) >> 1) & 1) << (logical - 2))
    )
    expected = torch.zeros_like(initial)
    expected[output << 2] = amplitude
    events = sum(gate.name == "measure" for gate in gates)
    assert events >= 2
    for history in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            actual, expected / 2 ** (events / 2), atol=5e-14, rtol=5e-14
        )
        assert torch.count_nonzero(actual.reshape(-1, 4)[:, 1:]) == 0


@pytest.mark.parametrize("subtract,value", list(product((False, True), range(8))))
def test_three_bit_classical_word_all_branches(subtract: bool, value: int) -> None:
    program, _ = physical_addition_binding(
        (Bit(0), Bit(1), Bit(2)),
        target=(2, 1, 0),
        control=3,
        workspace=(7, 6, 5),
        subtract=subtract,
    )
    setup = tuple(
        ClassicalXor((), bit, bool(value & (1 << (2 - bit)))).instruction()
        for bit in program.input_bits
    )
    gates = expand_reset_branches(
        (*setup, *bind_reversible_phase_execution(program)), first_slot=100
    )
    indices = torch.arange(32)
    amplitudes = torch.complex((indices + 1).double(), (9 - indices).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[indices << 3] = amplitudes
    target_value = sum(((indices >> (4 - wire)) & 1) << wire for wire in range(3))
    updated = (
        target_value + (-value if subtract else value) * ((indices >> 1) & 1)
    ) % 8
    output = indices.clone()
    for wire in range(3):
        output ^= (((target_value ^ updated) >> wire) & 1) << (4 - wire)
    expected = torch.zeros_like(initial)
    expected[output << 3] = amplitudes
    events = sum(gate.name == "measure" for gate in gates)
    assert events == 4
    for history in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(actual, expected / 4, atol=6e-14, rtol=6e-14)
        assert torch.count_nonzero(actual.reshape(-1, 8)[:, 1:]) == 0


def test_measured_primitive_counts_are_separate_from_models() -> None:
    expected = (
        ((3, 2), {"ccx": 4, "r": 1, "hmr": 1, "cz": 1}),
        ((Bit(0), Bit(1)), {"x": 4, "cx": 4, "r": 1, "ccx": 1, "hmr": 1, "cz": 1}),
        (-1, {"x": 4, "cx": 2, "r": 1, "ccx": 1, "hmr": 1, "cz": 1}),
    )
    for offset, counts in expected:
        program, _ = physical_addition_binding(
            offset, target=(1, 0), control=4, workspace=(6, 5)
        )
        assert Counter(op.opcode for op in program.operations) == counts
