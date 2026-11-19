"""Declared leaves compose and receive explicit actions, never implicit execution."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.component_graph import (
    copy_component_program,
    iter_component_calls,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_rebindings import NativeRebindings
from tests.unit.test_declared_component_contract import (
    test_declaration_call_retains_signature_and_linear_connections as declaration_contract,
)
from tests.unit.test_program_declarations import declared_program

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")


class NativeDeclarations(NativeRebindings):
    def call(
        self, name: str, signature: Signature, bindings: Mapping[str, object]
    ) -> object:
        self.builder.define(FunctionRecord(name, (), registers=tuple(signature)))
        return self.builder.call_dict(name, bindings)


@pytest.mark.parametrize("shape", [(), (2,), (0,)])
@pytest.mark.parametrize("directional", [False, True])
def test_native_replaces_source_declaration_contract(
    shape: tuple[int, ...], directional: bool
) -> None:
    declaration_contract(NativeDeclarations, shape, directional)


def test_declared_leaf_survives_traversal_copy_expansion_and_resource_queries() -> None:
    from flagquantum.compiler._component_trace.program_call_graph import (
        program_component_call_graph,
    )
    from flagquantum.compiler._component_trace.program_cost_query import (
        query_program_qec_gate_cost,
    )
    from flagquantum.compiler._component_trace.program_logical_width import (
        query_program_logical_width,
    )
    from flagquantum.compiler._component_trace.resource_generalization import (
        CliffordCall,
    )
    from flagquantum.compiler.registers.component_expansion import (
        NoComponentExpansionError,
        expand_component_calls_once,
    )
    from flagquantum.drawer.component_graph import export_component_graph_dot

    record = declared_program()
    assert len(list(iter_component_calls(record))) == 1
    copied = copy_component_program(record)
    assert verify_program_record(copied) == ()
    assert copied.functions[1] == record.functions[1]
    assert len(list(iter_component_calls(copied))) == 1
    with pytest.raises(NoComponentExpansionError):
        expand_component_calls_once(record)
    assert program_component_call_graph(record).leaf_counts == {"declared": 1}
    assert query_program_logical_width(record) == 3
    assert query_program_logical_width(record, "declared") == 3
    assert (
        query_program_qec_gate_cost(record, {"declared": CliffordCall("h", 1)}).clifford
        == 1
    )
    with pytest.raises(ValueError, match="Missing QEC terminal"):
        query_program_qec_gate_cost(record, {})
    dot = export_component_graph_dot(record)
    assert "declared" in dot and dot.count("->") == 2


@pytest.mark.parametrize("unreachable", [False, True])
def test_physical_entry_points_refuse_missing_bodies(unreachable: bool) -> None:
    record = declared_program()
    if unreachable:
        main = record.functions[0]
        block = main.blocks[0]
        record = replace(
            record,
            functions=(
                replace(
                    main,
                    blocks=(
                        replace(block, operations=(), return_values=block.arguments),
                    ),
                ),
                record.functions[1],
            ),
        )
    with pytest.raises(ValueError, match="requires component bodies"):
        lower_register_program(record)
    with pytest.raises(ValueError, match="requires component bodies"):
        bind_register_program(record, ((0, 1, 2),))


def test_explicit_classical_action_is_required_for_declared_leaf() -> None:
    from flagquantum.simulation.component_classical_state import ComponentClassicalState

    record = declared_program()
    seen: list[str] = []

    def action(function: FunctionRecord, values: dict[str, object]) -> object:
        assert function.blocks == ()
        seen.append(function.name)
        return values

    state = ComponentClassicalState(record, (0,), {"word": 5}, action)
    assert state.simulate() == {"word": 5}
    assert seen == ["declared"]


def test_explicit_tensor_supplier_keeps_full_complex_action() -> None:
    from flagquantum.simulation.tensor_network.component_dense import (
        contract_component_tensors,
    )
    from flagquantum.simulation.tensor_network.models import TensorNetworkNode
    from flagquantum.simulation.tensor_network.register_tensor_binding import (
        bind_register_tensor,
    )

    record = declared_program()
    diagonal = torch.arange(1, 9, dtype=torch.float64).to(torch.complex128) * (1 + 2j)
    matrix = torch.diag(diagonal)

    def supply(
        function: FunctionRecord, incoming: Any, outgoing: Any
    ) -> tuple[TensorNetworkNode, ...]:
        assert function.blocks == ()
        return (
            bind_register_tensor(
                matrix, Signature(function.registers or ()), incoming, outgoing
            ),
        )

    result = contract_component_tensors(
        record, (0,), supply, torch.empty((), dtype=torch.complex128)
    )
    torch.testing.assert_close(result, matrix, rtol=0, atol=0)


def test_explicit_adjoint_mapping_and_missing_hierarchy_refusal() -> None:
    from flagquantum.compiler.registers.component_adjoint import (
        adjoint_component_hierarchy,
        adjoint_component_program,
    )
    from flagquantum.compiler.registers.component_control import (
        control_component_hierarchy,
    )

    record = declared_program()
    inverse = adjoint_component_program(record, {"declared": "declared"})
    assert verify_program_record(inverse) == ()
    assert inverse.functions[1].blocks == ()
    with pytest.raises(ValueError, match="missing specialized adjoint"):
        adjoint_component_hierarchy(record, {})
    with pytest.raises(ValueError, match="missing specialized control"):
        control_component_hierarchy(record, {})
