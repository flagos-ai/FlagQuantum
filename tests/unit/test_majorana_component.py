"""Stored Majorana graphs retain source ports, inlining and measured parity."""

from collections import Counter
from itertools import product
from typing import Literal

import pytest
import torch

from flagquantum.algorithms.primitives.hubbard_majorana import HubbardMajorana
from flagquantum.algorithms.primitives.hubbard_majorana_component import (
    hubbard_majorana_program,
)
from flagquantum.compiler.registers.lookup.majorana_component import majorana_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("pauli", ["X", "Y"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_complete_hubbard_and_shared_majorana_graphs(
    dimension: int, controlled: bool, pauli: Literal["X", "Y"], adjoint: bool
) -> None:
    import cirq
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        HubbardMajorannaOperator,
    )
    from qualtran.bloqs.multiplexers.selected_majorana_fermion import (
        SelectedMajoranaFermion,
    )

    source = HubbardMajorannaOperator(
        dimension, dimension, pauli, control_val=1 if controlled else None
    )
    native = HubbardMajorana(dimension, pauli, controlled)
    record = hubbard_majorana_program(native, adjoint=adjoint)
    assert verify_program_record(record) == ()
    compare_layers(
        record, source.adjoint() if adjoint else source, atomic_leaves(record)
    )
    width = (dimension - 1).bit_length()
    selection = tuple(
        Register(name, bounded_unsigned_register_type(bits, bound))
        for name, bits, bound in (
            ("x", width, dimension),
            ("y", width, dimension),
            ("spin", 1, 2),
        )
    )
    record = majorana_program(
        selection,
        pauli,
        control=Register("control", quantum_bit_type()) if controlled else None,
        adjoint=adjoint,
    )
    shared = SelectedMajoranaFermion(
        selection_regs=source.selection_registers,
        control_regs=source.control_registers,
        target_gate=cirq.X if pauli == "X" else cirq.Y,
    )
    compare_layers(
        record, shared.adjoint() if adjoint else shared, atomic_leaves(record)
    )


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_shared_small_complete_operator_every_measurement_history(
    controlled: bool, pauli: Literal["X", "Y", "Z"], adjoint: bool
) -> None:
    import cirq

    from tests.unit.test_majorana_source_binding import _expected_matrix

    selection = (Register("selection", bounded_unsigned_register_type(1, 2)),)
    record = majorana_program(
        selection,
        pauli,
        control=Register("control", quantum_bit_type()) if controlled else None,
        adjoint=adjoint,
    )
    logical = 3 + int(controlled)
    work = 1 + int(controlled)
    inputs = (
        *(((0,),) if controlled else ()),
        (int(controlled),),
        (1 + int(controlled), 2 + int(controlled)),
    )
    bound = bind_register_program(
        record, inputs, clean=tuple(range(logical, logical + work))
    )
    gates = bound.circuit.instructions
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    expected = _expected_matrix(
        {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}[pauli], 2, int(controlled)
    )
    if adjoint:
        expected = expected.mH
    events = Counter(g.name for g in gates)["measure"]
    for column in range(1 << logical):
        state = torch.zeros(1 << (logical + work), dtype=torch.complex128)
        state[column << work] = 1
        desired = torch.zeros_like(state)
        desired[:: 1 << work] = expected[:, column] / 2 ** (events / 2)
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(state, gates, history), desired, atol=2e-12, rtol=0
            )


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("pauli", ["X", "Y"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_hubbard_complex_amplitudes_private_work_and_literal_inventory(
    controlled: bool, pauli: Literal["X", "Y"], adjoint: bool
) -> None:
    operation = HubbardMajorana(2, pauli, controlled)
    ports: dict[str, tuple[int, ...]] = {}
    logical = 0
    for name, size in operation.layout:
        ports[name] = tuple(range(logical, logical + size))
        logical += size
    work, slots = operation.requirements()
    clean = tuple(range(logical, logical + work))
    bound = bind_register_program(
        hubbard_majorana_program(operation, adjoint=adjoint),
        tuple(ports.values()),
        clean=clean,
    )
    flat = operation._emit(ports, clean, tuple(range(slots)))
    assert Counter(g.name for g in bound.circuit.instructions) == Counter(
        g.name for g in flat
    )
    assert bound.outputs == tuple(ports.values())
    initial = torch.randn(
        1 << logical,
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(742),
    )
    initial /= initial.norm()
    state = torch.zeros(1 << (logical + work), dtype=torch.complex128)
    state[:: 1 << work] = initial
    from tests.unit.measurement_certificate import certify_erasure_histories

    events = Counter(g.name for g in flat)["measure"]
    assert (
        certify_erasure_histories(
            bound.circuit.instructions, n_wires=bound.circuit.n_wires, zero_wires=clean
        )
        == events
    )
    expected = forced_branch(state, flat, (0,) * events) * 2 ** (events / 2)
    # The Majorana is Hermitian. Its separately stored inverse must agree.
    histories = [(0,) * events, (1,) * events, tuple(i % 2 for i in range(events))]
    for history in histories:
        actual = forced_branch(state, bound.circuit.instructions, history) * 2 ** (
            events / 2
        )
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=0)
        torch.testing.assert_close(
            actual.reshape(-1, 1 << work)[:, 1:],
            torch.zeros_like(actual.reshape(-1, 1 << work)[:, 1:]),
            atol=2e-11,
            rtol=0,
        )


def test_original_hubbard_stored_refusals_stay_explicit() -> None:
    with pytest.raises(ValueError, match="singleton"):
        hubbard_majorana_program(HubbardMajorana(1))
    with pytest.raises(ValueError, match="zero control"):
        hubbard_majorana_program(HubbardMajorana(2, controlled=True, control_value=0))
