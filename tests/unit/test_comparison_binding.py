"""Compare the complete binding dispatch and its classical guard execution."""

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Literal

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
from flagquantum.compiler.registers.comparison.physical_comparison import (
    physical_comparison,
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

_Relation = Literal["lt", "gt", "eq"]
_Case = tuple[int, str, str, str, int, str, bool, _Relation, int, str]
_Port = int | bool | Bit


def binding_ports(
    width: int, kind: str, target_kind: str, control_kind: str, work: int
) -> tuple[
    tuple[int, ...],
    tuple[_Port, ...],
    int | _XBasisTarget | None,
    _Port,
    tuple[int, ...],
    int,
]:
    """Lay out the source's LSB allocations as native MSB views and LIFO leases."""
    quantum = width
    classical = 0
    right: list[_Port] = []
    for index in range(width):
        if kind == "q" or (kind == "m" and index % 3 == 0):
            right.append(quantum)
            quantum += 1
        elif kind == "b" or (kind == "m" and index % 3 == 1):
            right.append(Bit(classical))
            classical += 1
        else:
            right.append(index % 2 == 0)
    target: int | _XBasisTarget | None = quantum
    control: _Port = quantum + 1
    quantum += 2
    if target_kind == "c":
        target = _XBasisTarget(Bit(classical))
        classical += 1
    elif target_kind == "phase":
        target = None
    elif target_kind == "skip":
        target = _XBasisTarget(False)
    if control_kind in ("c", "same"):
        if control_kind == "same" and isinstance(target, _XBasisTarget):
            assert isinstance(target.outcome, Bit)
            control = target.outcome
        else:
            control = Bit(classical)
            classical += 1
    elif control_kind != "q":
        control = control_kind == "true"
    return (
        tuple(reversed(range(width))),
        tuple(reversed(right)),
        target,
        control,
        tuple(reversed(range(quantum, quantum + work))),
        classical,
    )


def test_complete_binding_records() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("comparison_binding.json")
        .read_text()
    )
    assert fixture["source_revision"] == "9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453"
    cases: list[_Case] = fixture["cases"]
    assert len(cases) > 2000
    for (
        width,
        kind,
        target,
        control,
        work,
        tolerance,
        equal,
        relation,
        size,
        digest,
    ) in cases:
        left, right, output, guard, clean, _ = binding_ports(
            width, kind, target, control, work
        )
        rhs: _Port | tuple[_Port, ...] = (
            sum(1 << index for index, bit in enumerate(right[::-1]) if bit is True)
            if kind == "int"
            else right
        )
        program, calls = physical_comparison(
            left,
            rhs,
            output,
            clean,
            relation=relation,
            control=guard,
            or_equal=equal,
            btol=float(tolerance),
        )
        text = encode_kmx_block_marks(program, calls)
        records = "\n".join(text.splitlines()[1:])
        assert len(records.splitlines()) == size
        assert hashlib.sha256(records.encode()).hexdigest() == digest, (
            width,
            kind,
            target,
            control,
            work,
            tolerance,
            equal,
            relation,
        )
        assert decode_kmx_block_marks(text) == (program, calls)


@pytest.mark.parametrize("relation", ["lt", "gt", "eq"])
@pytest.mark.parametrize("phase", [False, True])
@pytest.mark.parametrize(
    "target_guard,control_guard", list(product((False, True), repeat=2))
)
def test_all_guarded_measurement_branches(
    relation: _Relation, phase: bool, target_guard: bool, control_guard: bool
) -> None:
    program, _ = physical_comparison(
        (0, 1),
        (Bit(0), Bit(1)),
        _XBasisTarget(Bit(5)) if phase else 2,
        (4, 3),
        relation=relation,
        control=Bit(6),
    )
    assert program.input_bits == ((0, 1, 5, 6) if phase else (0, 1, 6))
    assert all(
        op.result_bit is None or op.result_bit >= 7
        for op in program.operations
        if op.opcode == "hmr"
    )
    values = {0: True, 1: False, 5: target_guard, 6: control_guard}
    setup = tuple(
        ClassicalXor((), bit, values[bit]).instruction() for bit in program.input_bits
    )
    gates = expand_reset_branches(
        (*setup, *bind_reversible_phase_execution(program)),
        first_slot=100,
    )
    indices = torch.arange(8)
    amplitudes = torch.complex((indices + 1).double(), (11 - indices).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[indices << 2] = amplitudes
    lhs = indices >> 1
    selected = {"lt": lhs < 2, "gt": lhs > 2, "eq": lhs == 2}[relation]
    selected &= control_guard and (target_guard if phase else True)
    expected = torch.zeros_like(initial)
    if phase:
        expected[indices << 2] = amplitudes * (1 - 2 * selected.int())
    else:
        expected[(indices ^ selected.int()) << 2] = amplitudes
    events = sum(gate.name == "measure" for gate in gates)
    total = torch.zeros_like(initial.real)
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, outcomes)
        # Inactive outer guards suppress measurements. Only the all-zero
        # postselection history has weight, with the original complex phase.
        if not control_guard or (phase and not target_guard):
            branch = expected if not any(outcomes) else torch.zeros_like(expected)
        else:
            branch = expected / 2 ** (events / 2)
        torch.testing.assert_close(actual, branch, atol=3e-14, rtol=3e-14)
        total += actual.abs().square()
    torch.testing.assert_close(total, expected.abs().square())
    assert torch.count_nonzero(total.reshape(8, 4)[:, 1:]) == 0


def test_normal_append_and_guarded_phase() -> None:
    program, _ = physical_comparison(
        (0,), (True,), _XBasisTarget(Bit(0)), (), control=Bit(1)
    )
    circuit = DynamicCircuit(1, dtype=torch.complex128)
    setup = tuple(
        ClassicalXor((), bit, True).instruction() for bit in program.input_bits
    )
    circuit.append(
        FieldOperation(1, (*setup, *bind_reversible_phase_execution(program)))
    )
    result = run_dynamic(circuit, shots=2, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1),
        torch.tensor([[-1, 0], [-1, 0]], dtype=torch.complex128),
    )


def test_skipped_binding_keeps_both_scopes_without_generator_marks() -> None:
    program, calls = physical_comparison(
        (0,), (True,), _XBasisTarget(Bit(2)), (), control=False
    )
    assert calls == ()
    assert [op.opcode for op in program.operations] == [
        "push_condition",
        "pop_condition",
    ]
    program, calls = physical_comparison(
        (0,), (True,), _XBasisTarget(False), (), control=Bit(3)
    )
    assert calls == ()
    assert program.input_bits == (3,)
    assert [op.opcode for op in program.operations] == [
        "push_condition",
        "pop_condition",
    ]


def test_shared_scope_preserves_primitives_and_nested_call_ranges() -> None:
    plain, plain_calls = physical_comparison(
        (0, 1, 2), (True, False, True), None, (3, 4, 5)
    )
    guarded, calls = physical_comparison(
        (0, 1, 2), (True, False, True), _XBasisTarget(Bit(7)), (3, 4, 5), control=Bit(7)
    )
    assert [op.condition_bit for op in guarded.operations[:2]] == [7, 7]
    assert [op.opcode for op in guarded.operations[-2:]] == ["pop_condition"] * 2
    assert Counter(op.opcode for op in plain.operations) == Counter(
        op.opcode for op in guarded.operations[2:-2]
    )
    assert calls[0].start == plain_calls[0].start + 2
    assert calls[0].stop == plain_calls[0].stop + 2
    text = encode_kmx_block_marks(guarded, calls)
    assert decode_kmx_block_marks(text) == (guarded, calls)


@pytest.mark.parametrize("relation", ["lt", "gt", "eq"])
def test_conversion_rejects_invalid_coefficients_before_skip(
    relation: _Relation,
) -> None:
    invalid: Any = None  # Invalid values are deliberate at this input boundary.
    with pytest.raises(ValueError, match="Expected a quantum wire"):
        physical_comparison(
            (0,), (invalid,), _XBasisTarget(False), (), relation=relation
        )


def test_known_original_empty_mixed_pool_failure_has_native_rejection() -> None:
    with pytest.raises(ValueError, match="at least width minus one"):
        physical_comparison((1, 0), (Bit(0), 2), 3, (), control=4)


def test_existing_scalar_word_and_basis_bindings_are_reused() -> None:
    direct = physical_comparison((0,), (True,), None, ())
    assert physical_comparison(0, 1, "|->", ()) == direct
    skipped, calls = physical_comparison(0, True, "|+>", ())
    assert skipped.operations == () and calls == ()
    with pytest.raises(ValueError, match="outside the operand word range"):
        physical_comparison((0,), 2, "|+>", ())
    with pytest.raises(ValueError, match="quantum-only"):
        physical_comparison((False,), True, "|+>", ())


def test_reference_workspace_refusals() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("comparison_binding.json")
        .read_text()
    )
    rejections: list[_Case] = fixture["source_rejections"]
    assert len(rejections) == 480
    assert fixture["unsafe_source_cases_excluded"] == 120
    for (
        width,
        kind,
        target,
        control,
        work,
        tolerance,
        equal,
        relation,
        _,
        _,
    ) in rejections:
        left, right, output, guard, clean, _ = binding_ports(
            width, kind, target, control, work
        )
        with pytest.raises(ValueError):
            physical_comparison(
                left,
                right,
                output,
                clean,
                relation=relation,
                control=guard,
                or_equal=equal,
                btol=float(tolerance),
            )
