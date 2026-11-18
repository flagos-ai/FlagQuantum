"""Verify prefix-aware folded source subclasses through native indexed traversal."""

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
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_unary_iteration_file import source as source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def folded_source(source: Any) -> Any:
    import cirq
    from attrs import frozen
    from qualtran import BQUInt, QAny, QBit, Register
    from qualtran.bloqs.basic_gates import CNOT, CZ

    @frozen
    class FoldedSelect(source.UnaryIterationGate):  # type: ignore[misc]
        controlled: bool

        @property
        def control_registers(self) -> tuple[Any, ...]:
            return (Register("control", QBit()),) if self.controlled else ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (Register("row", BQUInt(1, 2)), Register("column", BQUInt(2, 4)))

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return (Register("target", QAny(1)),)

        def _break_early(
            self, selection_index_prefix: tuple[int, ...], left: int, right: int
        ) -> bool:
            return selection_index_prefix == (0,) and right - left == 2

        def nth_operation(
            self, context: Any, control: Any, row: int, column: int, target: Any
        ) -> Any:
            gates = ((cirq.Z, cirq.Z, cirq.X, cirq.X), (cirq.X, cirq.Z, cirq.Z, cirq.X))
            return gates[row][column](target[0]).controlled_by(control)

        def nth_operation_callgraph(
            self, **selection_regs_name_to_val: int
        ) -> set[tuple[Any, int]]:
            row = selection_regs_name_to_val["row"]
            column = selection_regs_name_to_val["column"]
            gates = ((CZ(), CZ(), CNOT(), CNOT()), (CNOT(), CZ(), CZ(), CNOT()))
            return {(gates[row][column], 1)}

    return FoldedSelect


@pytest.mark.parametrize("controlled", (False, True))
def test_prefix_fold_native_source_stream_and_all_operator_columns(
    folded_source: Any, controlled: bool
) -> None:
    import cirq

    original = folded_source(controlled)
    controls = int(controlled)
    logical = controls + 4
    work = controls + 2
    ports = {
        "row": (controls,),
        "column": (controls + 1, controls + 2),
        "target": (controls + 3,),
    }
    if controlled:
        ports["control"] = (0,)
    scratch = tuple(range(logical, logical + work))
    slots = tuple(range(work))
    table = ("z", "z", "x", "x", "x", "z", "z", "x")
    visited: list[int] = []
    for adjoint in (False, True):
        visited.clear()

        def emit(index: int, predicate: int) -> tuple[Instruction, ...]:
            visited.append(index)
            return (Instruction("c" + table[index], (predicate, logical - 1)),)

        gates = indexed_component_ir(
            (ports["row"], ports["column"]),
            (2, 4),
            emit,
            controls=ports.get("control", ()),
            workspace=scratch,
            classical_bits=slots,
            adjoint=adjoint,
            break_early=lambda prefix, left, right: prefix == (0,)
            and right - left == 2,
        )
        assert visited == ([7, 6, 5, 4, 2, 0] if adjoint else [0, 2, 4, 5, 6, 7])
        model = indexed_component_cost(
            (1, 2),
            (2, 4),
            tuple(
                ComponentCost(str(index), 2, GateCost(clifford=1)) for index in range(8)
            ),
            target_width=1,
            control_count=controls,
            adjoint=adjoint,
            break_early=lambda prefix, left, right: prefix == (0,)
            and right - left == 2,
        )
        assert [int(call.component.name) for call in model.calls] == visited
        assert_source_model(model, original)
        reference = reference_arithmetic_gates(
            original.adjoint() if adjoint else original,
            ports,
            scratch,
            dict(zip(scratch, slots, strict=True)),
            cirq_style=not adjoint,
            keep_pauli_leaves=True,
        )
        assert_source_dependencies(gates, reference)
        ir = (
            Circuit(logical + work)
            .append(FieldOperation(logical + work, gates))
            .to_ir()
        )
        assert CircuitIR.from_json(ir.to_json()).instructions == gates
        counts = Counter(g.name for g in gates)
        events = counts["measure"]
        assert events == 4 + controls
        assert counts["t"] + counts["tdg"] == 4 * events
        for basis in range(1 << logical):
            state = torch.zeros(1 << (logical + work), dtype=torch.complex128)
            state[basis << work] = 1
            selected = (basis >> 1) & 7
            enabled = not controlled or bool(basis >> 4)
            leaf = cirq.unitary(
                {"x": cirq.X, "y": cirq.Y, "z": cirq.Z}[table[selected]]
            )
            values = torch.zeros(2, dtype=torch.complex128)
            values[basis & 1] = 1
            if enabled:
                values = torch.as_tensor(leaf, dtype=torch.complex128) @ values
            expected = torch.zeros_like(state)
            start = (basis >> 1) << (work + 1)
            expected[start : start + (2 << work) : 1 << work] = values
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(state, gates, history),
                    expected / 2 ** (events / 2),
                    atol=2e-12,
                    rtol=0,
                )


def test_root_fold_keeps_zero_selection_recipe_boundary(folded_source: Any) -> None:
    import cirq
    from attrs import frozen
    from qualtran._infra.gate_with_registers import get_named_qubits

    @frozen
    class RootFold(folded_source):  # type: ignore[misc]
        controlled: bool

        def _break_early(
            self, selection_index_prefix: tuple[int, ...], left: int, right: int
        ) -> bool:
            return not selection_index_prefix

    original = RootFold(False)
    with pytest.raises(
        NotImplementedError, match="Selection register must not be empty"
    ):
        list(
            cirq.flatten_op_tree(
                original.decompose_from_registers(
                    context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
                    **get_named_qubits(original.signature),
                )
            )
        )
    with pytest.raises(ValueError, match="zero-selection recipe"):
        indexed_component_ir(
            ((0,), (1, 2)),
            (2, 4),
            lambda *_: (),
            workspace=(4, 5),
            classical_bits=(0, 1),
            break_early=lambda *_: True,
        )

    with pytest.raises(ValueError, match="zero-selection recipe"):
        indexed_component_cost(
            (1, 2),
            (2, 4),
            (ComponentCost("CX", 2),) * 8,
            target_width=1,
            break_early=lambda *_: True,
        )


@pytest.mark.parametrize("controls", (0, 1, 2, 3))
@pytest.mark.parametrize("adjoint", (False, True))
def test_inactive_fold_model_reuses_the_existing_full_call_inventory(
    controls: int, adjoint: bool
) -> None:
    children = tuple(
        ComponentCost(str(index), 2, GateCost(clifford=index + 1))
        for index in range(15)
    )
    ordinary = indexed_component_cost(
        (3, 2),
        (5, 3),
        children,
        target_width=1,
        control_count=controls,
        adjoint=adjoint,
    )
    folded = indexed_component_cost(
        (3, 2),
        (5, 3),
        children,
        target_width=1,
        control_count=controls,
        adjoint=adjoint,
        break_early=lambda *_: False,
    )
    assert folded == ordinary
