"""Bind complete external Majorana objects to portable native SELECT consumers."""

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import (
    BlackBoxPrepare,
    _identity_prepare,
)
from flagquantum.algorithms.primitives.circuit_register_emitter import (
    CircuitRegisterEmitter,
)
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_majorana_file import source as source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _capture(source_object: Any) -> tuple[SelectBlockEncoding, CircuitRegisterEmitter]:
    emitter = import_register_emitter(source_object)
    # Only native layout tuples and restored IR cross into the consumer.
    emitter = replace(
        emitter,
        snapshots=(
            CircuitIR.from_json(emitter.snapshots[0].to_json()),
            CircuitIR.from_json(emitter.snapshots[1].to_json()),
        ),
    )
    selection = emitter.layout[:-1]
    target = emitter.layout[-1:]
    signal = BlackBoxPrepare(selection, (), _identity_prepare)
    return SelectBlockEncoding(selection, target, emitter, signal), emitter


def _expected_matrix(
    target_gate: Any, target_width: int, control_width: int
) -> torch.Tensor:
    import cirq

    target_matrix = torch.as_tensor(cirq.unitary(target_gate), dtype=torch.complex128)
    identity = torch.eye(2, dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    blocks = []
    for control, row in product(range(1 << control_width), range(target_width)):
        enabled = control == (1 << control_width) - 1
        factors = [
            (
                target_matrix
                if enabled and wire == row
                else z if enabled and wire < row else identity
            )
            for wire in range(target_width)
        ]
        matrix = factors[0]
        for factor in factors[1:]:
            matrix = torch.kron(matrix, factor)
        blocks.append(matrix)
    width = sum(block.shape[0] for block in blocks)
    result = torch.zeros((width, width), dtype=torch.complex128)
    start = 0
    for block in blocks:
        stop = start + block.shape[0]
        result[start:stop, start:stop] = block
        start = stop
    return result


@pytest.mark.parametrize("controls", (0, 1, 2))
@pytest.mark.parametrize("gate_name", ("H", "S", "Rz"))
def test_external_gate_and_multi_control_all_measurement_branches(
    source: Any, controls: int, gate_name: str
) -> None:
    import cirq
    from qualtran import BQUInt, QAny, Register
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    gate = {"H": cirq.H, "S": cirq.S, "Rz": cirq.rz(0.3)}[gate_name]
    original = source(
        Register("selection", BQUInt(1, 2)),
        control_regs=() if controls == 0 else (Register("enable", QAny(controls)),),
        target_gate=gate,
    )
    native, emitter = _capture(original)
    work, bits = emitter.requirements()
    selection_width = controls + 1
    logical_width = selection_width + 2
    logical_matrix = _expected_matrix(gate, 2, controls)
    generator = torch.Generator().manual_seed(97)
    coherent = torch.randn(
        1 << logical_width, generator=generator, dtype=torch.complex128
    )
    coherent /= coherent.norm()
    columns = [coherent]
    if gate_name == "S" and controls <= 1:
        columns.extend(torch.eye(1 << logical_width, dtype=torch.complex128).unbind(1))
    for adjoint in (False, True):
        ir = (
            Circuit(logical_width + work)
            .append(
                native.make_on(
                    selection=tuple(range(selection_width)),
                    target=tuple(range(selection_width, logical_width)),
                    workspace=tuple(range(logical_width, logical_width + work)),
                    classical_bits=tuple(range(bits)),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        assert ir.instructions == emitter.snapshots[int(adjoint)].instructions
        assert CircuitIR.from_json(ir.to_json()).instructions == ir.instructions
        events = Counter(g.name for g in ir.instructions)["measure"]
        model = get_cost_value(
            original.adjoint() if adjoint else original, QECGatesCost()
        )
        model_width = get_cost_value(
            original.adjoint() if adjoint else original, QubitCount()
        )
        ands = controls + 2 * int(gate_name == "S")
        assert model.measurement == model.and_bloq == ands
        assert model.toffoli == int(controls == 2)
        assert model.clifford == (15 if gate_name == "H" else 11)
        assert model.rotation == (0 if gate_name == "S" else 4)
        assert model_width == logical_width + 1 + controls + int(gate_name == "S")
        counts = Counter(g.name for g in ir.instructions)
        extra = 2 * int(adjoint and gate_name == "S")
        assert events == controls + extra
        assert counts["t"] + counts["tdg"] == 4 * (controls + extra)
        assert counts["ccx"] == int(controls == 2)
        assert counts["cx"] == (2, 10, 15)[controls] + 6 * extra + 4 * int(
            adjoint and gate_name == "Rz"
        )
        matrix = logical_matrix.mH if adjoint else logical_matrix
        for logical_state in columns:
            state = torch.zeros(1 << (logical_width + work), dtype=torch.complex128)
            state[:: 1 << work] = logical_state
            expected = torch.zeros_like(state)
            expected[:: 1 << work] = matrix @ logical_state
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(state, ir.instructions, history),
                    expected / 2 ** (events / 2),
                    atol=3e-12,
                    rtol=0,
                )


def test_named_nested_axes_use_row_major_targets_and_clean_accumulator(
    source: Any,
) -> None:
    import cirq
    from qualtran import BQUInt, QAny, Register

    original = source(
        (Register("row", BQUInt(1, 2)), Register("column", BQUInt(1, 2))),
        control_regs=(Register("enable", QAny(2)),),
        target_gate=cirq.H,
    )
    native, emitter = _capture(original)
    assert emitter.layout == (("enable", 2), ("row", 1), ("column", 1), ("target", 4))
    work, bits = emitter.requirements()
    logical = 8
    generator = torch.Generator().manual_seed(39)
    values = torch.randn(1 << logical, generator=generator, dtype=torch.complex128)
    values /= values.norm()
    matrix = _expected_matrix(cirq.H, 4, 2)
    for adjoint in (False, True):
        ir = (
            Circuit(logical + work)
            .append(
                native.make_on(
                    selection=(0, 1, 2, 3),
                    target=(4, 5, 6, 7),
                    workspace=tuple(range(logical, logical + work)),
                    classical_bits=tuple(range(bits)),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        assert ir.instructions == emitter.snapshots[int(adjoint)].instructions
        events = sum(g.name == "measure" for g in ir.instructions)
        state = torch.zeros(1 << (logical + work), dtype=torch.complex128)
        state[:: 1 << work] = values
        expected = torch.zeros_like(state)
        expected[:: 1 << work] = (matrix.mH if adjoint else matrix) @ values
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(state, ir.instructions, history),
                expected / 2 ** (events / 2),
                atol=3e-12,
                rtol=0,
            )


def test_symbolic_ports_require_binding_and_symbolic_extents_remain_source_errors(
    source: Any,
) -> None:
    import attrs
    import cirq
    from qualtran import BQUInt, Register
    from sympy import Symbol

    width = Symbol("n", positive=True, integer=True)
    extent = Symbol("L", positive=True, integer=True)
    declared = source(Register("selection", BQUInt(width, 2)), target_gate=cirq.H)
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(declared)
    resolved = attrs.evolve(
        declared, selection_regs=(Register("selection", BQUInt(1, 2)),)
    )
    native, emitter = _capture(resolved)
    assert native.selection_layout == (("control", 1), ("selection", 1))
    assert emitter.requirements() == (2, 1)
    unsupported = source(Register("selection", BQUInt(width, extent)))
    with pytest.raises(ValueError, match="Symbolic iteration size"):
        import_register_emitter(unsupported)


def test_uncontrolled_singleton_retains_locked_source_failure(source: Any) -> None:
    from qualtran import BQUInt, Register

    singleton = source(Register("selection", BQUInt(1, 1)), control_regs=())
    with pytest.raises(IndexError, match="list index out of range"):
        import_register_emitter(singleton)


def test_separate_source_control_registers_leave_dirty_accumulator(source: Any) -> None:
    import cirq
    from qualtran import BQUInt, QBit, Register

    original = source(
        Register("selection", BQUInt(1, 2)),
        control_regs=(Register("first", QBit()), Register("second", QBit())),
        target_gate=cirq.H,
    )
    _, emitter = _capture(original)
    forward = emitter.snapshots[0]
    logical_width = 5
    work = forward.n_wires - logical_width
    initial = torch.zeros(1 << forward.n_wires, dtype=torch.complex128)
    initial[(1 << 4) << work] = 1  # first=1, second=0, all other bits zero.
    events = sum(g.name == "measure" for g in forward.instructions)
    for history in product((0, 1), repeat=events):
        output = forced_branch(initial, forward.instructions, history)
        clean = output[:: 1 << work]
        assert clean.norm().item() < 1e-12
        assert output.norm().item() == pytest.approx(2 ** (-events / 2), abs=1e-12)
