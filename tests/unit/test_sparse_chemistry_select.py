"""Verify source-aligned sparse chemistry SELECT as an appendable operation."""

from itertools import product

import pytest
import torch

from flagquantum.algorithms.primitives.multiplexers.select_sparse import SelectSparse
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import walk_source_graph
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_sparse_chemistry_prepare import _assert_sparse_signature

pytestmark = pytest.mark.unit


def _layout(size, value):
    layout = SelectSparse.source_layout(size, control_value=value)
    width = sum(bits for _, bits in layout)
    work, slots = SelectSparse.source_requirements(size)
    ports = partition_register(tuple(range(width)), layout)
    scratch = tuple(range(width, width + work))
    bits = tuple(range(11, 11 + slots))
    return layout, ports, scratch, bits


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("size,value", tuple(product((4, 6), (None, 0, 1))))
def test_sparse_select_matches_pinned_composition(size, value):
    pytest.importorskip("qualtran")
    from qualtran._infra.quantum_graph import LeftDangle
    from qualtran.bloqs.basic_gates import SGate
    from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse as Source
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    layout, ports, scratch, bits = _layout(size, value)
    source = Source(size, value)
    _assert_sparse_signature(
        SelectSparse.source_signature(size, control_value=value), source.signature
    )
    assert layout == tuple((reg.name, reg.total_bits()) for reg in source.signature)
    if value is not None:
        assert source.decompose_bloq().final_soqs()["control"].binst == LeftDangle

    calls = []

    def leaf(bloq, inputs, desired):
        if isinstance(bloq, Source):
            return None
        assert all(inputs[name] == wires for name, wires in desired.items())
        calls.append(type(bloq))
        if isinstance(bloq, SGate):
            return (Instruction("s", inputs["q"]),), inputs
        assert isinstance(bloq, SelectedMajoranaFermion)
        active = len(scratch) if bloq.control_registers else len(scratch) - 1
        result = reference_arithmetic_gates(
            bloq,
            inputs,
            scratch[:active],
            dict(zip(scratch[1:active], bits[: active - 1], strict=True)),
            cirq_style=True,
            keep_pauli_leaves=True,
        )
        return result, inputs

    expected, returned = walk_source_graph(source, ports, ports, leaf)
    assert calls == [SGate, *([SelectedMajoranaFermion] * 4)]
    assert returned == ports
    op = SelectSparse.make_source_on(
        size, ports, workspace=scratch, classical_bits=bits, control_value=value
    )
    n_qubits = scratch[-1] + 1
    circuit = DynamicCircuit(n_qubits).append(op)
    assert_source(circuit.to_ir().instructions, expected)
    assert_source_model(op.resource_model(), source)
    report = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    assert report.logical.peak_logical_qubits == n_qubits
    assert report.emitted.t_count == report.logical.gates.total_t_count()
    assert report.emitted.unconditional["measure"] == report.logical.gates.measurement
    if value is not None:
        assert report.emitted.instruction_qubits == n_qubits - 1


def test_sparse_select_preserves_superposition_and_cleans_reused_workspace():
    _, ports, scratch, bits = _layout(4, 0)
    op = SelectSparse.make_source_on(
        4, ports, workspace=scratch, classical_bits=bits, control_value=0
    )
    n_qubits = scratch[-1] + 2  # Last wire is a spectator.
    gates = DynamicCircuit(n_qubits).append(op).to_ir().instructions
    events = sum(gate.name == "measure" for gate in gates)
    assert events == 10
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for word, system, spectator in product(range(256), range(16), (0, 1)):
        # Source order: external control, p, q, r, s, alpha, beta, flag.
        _, p, q, r, s, alpha, beta, flag = ((word >> i) & 1 for i in reversed(range(8)))
        index = (((word << 4) | system) << 4) | spectator
        coefficient = complex(1 + index % 17, index % 11 - 5)
        state[index] = coefficient
        phase = 1j if flag else 1 + 0j
        pairs = ((r, beta, True), (s, beta, False))
        if flag:
            pairs = ((p, alpha, True), (q, alpha, False), *pairs)
        for orbital, spin, is_y in pairs:
            address = 2 * orbital + spin
            bit = 1 << (3 - address)
            phase *= (-1) ** (system >> (4 - address)).bit_count()
            if is_y:
                phase *= -1j if system & bit else 1j
            system ^= bit
        destination = (((word << 4) | system) << 4) | spectator
        expected[destination] = phase * coefficient
    norm = state.norm()
    state /= norm
    expected /= norm * 2 ** (events / 2)
    inverse = op.emit(adjoint=True)
    assert inverse[-1] == Instruction("sdg", ports["flag_1b"])
    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        history = (pattern * events)[:events]
        torch.testing.assert_close(
            forced_branch(state, gates, history), expected, atol=2e-12, rtol=0
        )
        torch.testing.assert_close(
            forced_branch(expected, inverse, history),
            state * 2**-events,
            atol=2e-12,
            rtol=0,
        )


def test_sparse_select_validates_source_ports_and_resources():
    _, ports, scratch, bits = _layout(4, None)
    with pytest.raises(ValueError, match="two spatial orbitals"):
        SelectSparse.source_resource_model(2)
    with pytest.raises(ValueError, match="control value"):
        SelectSparse.source_layout(4, control_value=True)
    with pytest.raises(ValueError, match="port names"):
        SelectSparse.make_source_on(
            4, ports, workspace=scratch, classical_bits=bits, control_value=1
        )
    with pytest.raises(ValueError, match="matching work"):
        SelectSparse.make_source_on(
            4, ports, workspace=scratch, classical_bits=bits[:1]
        )
    layout = SelectSparse.source_layout(4, control_value=1)
    width = sum(n for _, n in layout)
    ports = partition_register(tuple(range(width)), layout)
    work, slots = SelectSparse.controlled_requirements(4)
    scratch = tuple(range(width, width + work))
    bits = tuple(range(slots))
    for value in (None, True, 2):
        with pytest.raises(ValueError):
            SelectSparse.make_controlled_on(
                4,
                ports,
                workspace=scratch,
                classical_bits=bits,
                control_value=value,
            )
    with pytest.raises(ValueError, match="distinct"):
        SelectSparse.make_controlled_on(
            4, ports, workspace=scratch, classical_bits=(0,) * slots
        )
    with pytest.raises(ValueError, match="matching work"):
        SelectSparse.make_controlled_on(
            4, ports, workspace=scratch[:-1], classical_bits=bits
        )


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("value", [0, 1])
def test_native_external_control_matches_full_complex_select(size, value):
    layout = SelectSparse.source_layout(size, control_value=value)
    logical = sum(width for _, width in layout)
    ports = partition_register(tuple(range(logical)), layout)
    work, slots = SelectSparse.controlled_requirements(size)
    workspace = tuple(range(logical, logical + work))
    bits = tuple(range(20, 20 + slots))
    width = logical + work + 1
    binding = dict(workspace=workspace, classical_bits=bits, control_value=value)
    op = SelectSparse.make_controlled_on(size, ports, **binding)
    ir = DynamicCircuit(width).append(op).to_ir()
    gates = ir.instructions
    inverse = SelectSparse.make_controlled_on(
        size, ports, **binding, adjoint=True
    ).emit()
    events = sum(gate.name == "measure" for gate in gates)
    assert sum(gate.name == "measure" for gate in inverse) == events
    report = estimate_ftqc_resources(op.resource_model(), emitted_ir=ir)
    assert report.emitted is not None
    assert report.logical.gates.and_bloq == report.logical.gates.measurement == events
    assert report.logical.peak_logical_qubits == logical + work
    assert (report.emitted.t_count, events, work, slots) == (
        (28, 7, 3, 2) if size == 2 else (60, 15, 4, 3)
    )
    state = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for word, system, spectator in product(
        range(1 << (logical - size)), range(1 << size), (0, 1)
    ):
        index = (((word << size) | system) << (work + 1)) | spectator
        coefficient = complex(1 + index % 17, index % 11 - 5)
        state[index] = coefficient
        control = word >> (logical - size - 1)
        phase = 1 + 0j
        if control == value:
            alpha, beta, flag = ((word >> i) & 1 for i in (2, 1, 0))
            if size == 2:
                p = q = r = s = 0
            else:
                p, q, r, s = ((word >> i) & 1 for i in (6, 5, 4, 3))
            phase = 1j if flag else 1 + 0j
            pairs = ((r, beta, True), (s, beta, False))
            if flag:
                pairs = ((p, alpha, True), (q, alpha, False), *pairs)
            for orbital, spin, is_y in pairs:
                address = 2 * orbital + spin
                mask = 1 << (size - 1 - address)
                phase *= (-1) ** (system >> (size - address)).bit_count()
                if is_y:
                    phase *= -1j if system & mask else 1j
                system ^= mask
        destination = (((word << size) | system) << (work + 1)) | spectator
        expected[destination] = phase * coefficient
    norm = state.norm()
    state /= norm
    expected /= norm * 2 ** (events / 2)
    histories = (
        product((0, 1), repeat=events)
        if size == 2
        else ((pattern * events)[:events] for pattern in ((0,), (1,), (0, 1), (1, 0)))
    )
    for history in histories:
        torch.testing.assert_close(
            forced_branch(state, gates, history), expected, atol=3e-12, rtol=0
        )
        torch.testing.assert_close(
            forced_branch(expected, inverse, history),
            state * 2**-events,
            atol=3e-12,
            rtol=0,
        )


@pytest.mark.ftqc_reference
def test_single_spatial_orbital_source_resource_limits() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse as Source
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = Source(2)
    _assert_sparse_signature(SelectSparse.source_signature(2), source.signature)
    assert len(source.decompose_bloq().bloq_instances) == 5
    with pytest.raises(AssertionError):
        get_cost_value(source, QECGatesCost())
    with pytest.raises(RuntimeError):
        get_cost_value(source, QubitCount())


@pytest.mark.parametrize("value", [None, 0, 1])
def test_single_spatial_orbital_exact_phase_and_control_passthrough(value):
    layout, ports, scratch, bits = _layout(2, value)
    total = sum(width for _, width in layout)
    assert scratch == bits == ()
    op = SelectSparse.make_source_on(
        2, ports, workspace=scratch, classical_bits=bits, control_value=value
    )
    gates = DynamicCircuit(total).append(op).to_ir().instructions
    assert not any(g.name in {"measure", "reset"} for g in gates)
    state = (
        torch.arange(1, 1 + (1 << total), dtype=torch.float64).to(torch.complex128) + 1j
    )
    state /= state.norm()
    expected = torch.empty_like(state)
    for index, amp in enumerate(state):
        alpha, beta, flag, sys_up, sys_down = (
            (index >> bit) & 1 for bit in reversed(range(5))
        )
        occupancies = sys_up, sys_down
        sign = (-1) ** (occupancies[beta] + flag * (1 + occupancies[alpha]))
        expected[index] = 1j * sign * amp
    torch.testing.assert_close(
        forced_branch(state, gates, ()), expected, atol=2e-12, rtol=0
    )
    torch.testing.assert_close(
        forced_branch(expected, op.emit(adjoint=True), ()), state, atol=2e-12, rtol=0
    )
