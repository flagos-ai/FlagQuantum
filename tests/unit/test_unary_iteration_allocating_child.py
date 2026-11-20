"""Unary subclass hooks preserve shaped ports and private clean child storage."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_ir,
)
from flagquantum.compiler.registers.control.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_unary_iteration_file import source as source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def allocating_source(source: Any) -> Any:
    cirq = pytest.importorskip("cirq")
    from attrs import frozen
    from qualtran import BQUInt, GateWithRegisters, QBit, Register, Signature
    from qualtran.bloqs.basic_gates import CNOT, CZ
    from qualtran.resource_counting import QubitCount

    @frozen
    class PrivateLeaf(GateWithRegisters):  # type: ignore[misc]
        @property
        def signature(self) -> Any:
            return Signature.build(control=1, target=1, phase=1)

        def decompose_from_registers(self, *, context: Any, **quregs: Any) -> Any:
            control, target, phase = (
                quregs[name][0] for name in ("control", "target", "phase")
            )
            work = context.qubit_manager.qalloc(1)
            yield cirq.CNOT(control, work[0])
            yield cirq.CZ(work[0], phase)
            yield cirq.CNOT(work[0], target)
            yield cirq.CNOT(control, work[0])
            context.qubit_manager.qfree(work)

        def build_call_graph(self, ssa: Any) -> dict[Any, int]:
            return {CNOT(): 3, CZ(): 1}

        def my_static_costs(self, cost_key: Any) -> Any:
            return 4 if isinstance(cost_key, QubitCount) else NotImplemented

    @frozen
    class AllocatingSelect(source.UnaryIterationGate):  # type: ignore[misc]
        controlled: bool

        @property
        def control_registers(self) -> tuple[Any, ...]:
            return (Register("enable", QBit()),) if self.controlled else ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (Register("selection", BQUInt(2, 3)),)

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return (Register("target", QBit()),)

        @property
        def extra_registers(self) -> tuple[Any, ...]:
            return (Register("phase", QBit(), shape=(2,)),)

        @property
        def signature(self) -> Any:
            # The base signature omits extra_registers. A subclass owns their
            # admission into its complete signature, as in the source protocol.
            return Signature(
                (
                    *self.control_registers,
                    *self.selection_registers,
                    *self.target_registers,
                    *self.extra_registers,
                )
            )

        def nth_operation(
            self, context: Any, control: Any, selection: int, target: Any, phase: Any
        ) -> Any:
            return PrivateLeaf().on_registers(
                control=control, target=target, phase=phase[selection % 2]
            )

        def nth_operation_callgraph(self, **values: int) -> set[tuple[Any, int]]:
            return {(PrivateLeaf(), 1)}

    return AllocatingSelect


@pytest.mark.parametrize("controlled,adjoint", tuple(product((False, True), repeat=2)))
def test_allocating_shaped_extra_ports_all_columns_and_branches(
    source: Any, allocating_source: Any, controlled: bool, adjoint: bool
) -> None:
    original = allocating_source(controlled)
    controls = int(controlled)
    logical = controls + 5
    parent_work = controls + 1
    work = parent_work + 1
    ports = {
        "selection": (controls, controls + 1),
        "target": (controls + 2,),
        "phase": (controls + 3, controls + 4),
    }
    if controlled:
        ports["enable"] = (0,)
    scratch = tuple(range(logical, logical + work))
    child_work = scratch[-1]

    def emit(index: int, predicate: int) -> tuple[Instruction, ...]:
        gates = (
            Instruction("cx", (predicate, child_work)),
            Instruction("cz", (child_work, ports["phase"][index % 2])),
            Instruction("cx", (child_work, ports["target"][0])),
            Instruction("cx", (predicate, child_work)),
        )
        return tuple(reversed(gates)) if adjoint else gates

    gates = indexed_component_ir(
        (ports["selection"],),
        (3,),
        emit,
        controls=ports.get("enable", ()),
        workspace=scratch[:-1],
        classical_bits=tuple(range(parent_work)),
        adjoint=adjoint,
    )
    direction = original.adjoint() if adjoint else original
    reference = reference_arithmetic_gates(
        direction,
        ports,
        scratch,
        dict(zip(scratch, range(work), strict=True)),
        cirq_style=not adjoint,
        keep_pauli_leaves=True,
    )
    if adjoint:
        # Generic source inversion recycles work across parent/child lifetimes.
        # The fixed-child native recipe has equal counts and channel, while its
        # raw physical allocation stream remains an explicit whole-file gap.
        assert Counter(g.name for g in gates) == Counter(g.name for g in reference)
        assert gates != reference
    else:
        assert_source_dependencies(gates, reference)
    model = indexed_component_cost(
        (2,),
        (3,),
        (ComponentCost("PrivateLeaf", 4, GateCost(clifford=4)),) * 3,
        target_width=3,
        child_target_width=2,
        control_count=controls,
        adjoint=adjoint,
    )
    assert_source_model(model, direction)
    native = (
        Circuit(logical + work).append(FieldOperation(logical + work, gates)).to_ir()
    )
    assert CircuitIR.from_json(native.to_json()) == native
    measured = Counter(g.name for g in gates)
    assert measured["measure"] == controls + 1
    assert measured["t"] + measured["tdg"] == 4 * (controls + 1)
    # Every logical column includes arbitrary values of the two phase ports.
    # The padded address follows the exact source traversal and is compared too.
    for basis in range(1 << logical):
        state = torch.zeros(1 << (logical + work), dtype=torch.complex128)
        state[basis << work] = torch.tensor(0.6 + 0.8j, dtype=state.dtype)
        for history in product((0, 1), repeat=measured["measure"]):
            actual = forced_branch(state, gates, history)
            expected = forced_branch(state, reference, history)
            torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
            assert torch.count_nonzero(actual.reshape(-1, 1 << work)[:, 1:]) == 0
    # The existing foreign boundary must also consume the full custom subclass,
    # preserve both snapshots, and stop retaining the foreign source instance.
    captured = import_register_emitter(original)
    assert tuple(name for name, _ in captured.layout) == tuple(
        register.name for register in original.signature
    )
    assert captured.requirements()[0] == work
    rebound = captured(
        ports, scratch, tuple(range(captured.requirements()[1])), adjoint
    )
    assert Counter(g.name for g in gates) == Counter(g.name for g in rebound)
    assert len(rebound) == len(gates)
    for basis in range(1 << logical):
        state = torch.zeros(1 << (logical + work), dtype=torch.complex128)
        state[basis << work] = torch.tensor(0.6 + 0.8j, dtype=state.dtype)
        for history in product((0, 1), repeat=measured["measure"]):
            actual = forced_branch(state, rebound, history)
            expected = forced_branch(state, reference, history)
            torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
            assert torch.count_nonzero(actual.reshape(-1, 1 << work)[:, 1:]) == 0
