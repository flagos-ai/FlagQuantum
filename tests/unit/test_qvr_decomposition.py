"""Complete variable rotation graphs preserve source tensors and private cleanup."""

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms import rotations_ui_export as native
from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.qvr_decomposition import qvr_child
from flagquantum.compiler.registers.arithmetic.integer_square_component import (
    integer_square_program,
)
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.phase.cost_function_component import (
    cost_function_phase_program,
)
from flagquantum.compiler.registers.phase.qvr_component import (
    qvr_phase_gradient_program,
    qvr_zpow_program,
)
from flagquantum.compiler.registers.phase.qvr_phase_gradient import qvr_phase_format
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_field_product_decomposition import source_walk
from tests.unit.test_rotations_ui_export import originals as originals

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
FACTORIES = (
    "qvr_zpow",
    "qvr_phase_gradient",
    "square_via_phase_gradient",
    "square_via_zpow_phasing",
)


@pytest.mark.parametrize(
    "identifier,graphs", tuple(zip(FACTORIES, (1, 2, 3, 2), strict=True))
)
def test_complete_file_recursive_graphs_and_fresh_source_drawings(
    identifier: str, graphs: int, originals: tuple[Any, dict[str, Any]]
) -> None:
    rows = source_walk(identifier, originals[1][identifier], catalog=native)
    assert sum(row["disposition"] == "exact_graph" for row in rows) == graphs
    assert all(row.get("drawing", "source_exact") == "source_exact" for row in rows)


@pytest.mark.parametrize("square,gradient", tuple(product((False, True), repeat=2)))
def test_full_source_tensor_all_histories_and_inverse_with_spectator(
    square: bool, gradient: bool
) -> None:
    from qualtran import QFxp
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.arithmetic import Square
    from qualtran.bloqs.rotations.phasing_via_cost_function import (
        PhasingViaCostFunction,
    )
    from qualtran.bloqs.rotations.quantum_variable_rotation import (
        QvrPhaseGradient,
        QvrZPow,
    )

    width = 2
    cost_width = 2 * width if square else width
    cost_name = "result" if square else "energy"
    cost = Register(cost_name, fixed_point_register_type(cost_width, cost_width))
    make = qvr_phase_gradient_program if gradient else qvr_zpow_program
    phase = make(cost, 0.5, 1.0)
    inverse_phase = make(cost, -0.5, 1.0)
    record = (
        cost_function_phase_program(
            integer_square_program(width),
            phase,
            integer_square_program(width, adjoint=True),
            cost_names=(cost_name,),
        )
        if square
        else phase
    )
    inverse_record = (
        cost_function_phase_program(
            integer_square_program(width),
            inverse_phase,
            integer_square_program(width, adjoint=True),
            cost_names=(cost_name,),
        )
        if square
        else inverse_phase
    )
    source_phase = (QvrPhaseGradient if gradient else QvrZPow)(
        SourceRegister(cost_name, QFxp(cost_width, cost_width)), 0.5, 1.0
    )
    source = (
        PhasingViaCostFunction(Square(width), source_phase) if square else source_phase
    )
    m = (
        qvr_phase_format(cost_width, cost_width, 0.5, 1.0).gradient_width
        if gradient
        else 0
    )
    logical = width + m
    work_size = (2 * width if square else 0) + max(
        width * (width - 1) if square else 0, m - 1
    )
    total = logical + work_size + 1
    wires = tuple(reversed(range(logical)))
    inputs = (wires[:width], wires[width:]) if gradient else (wires,)
    clean = tuple(range(logical + 1, total))
    bound = bind_register_program(record, inputs, clean=clean)
    assert bound.outputs == inputs and copy_component_program(record) == record
    gates = bound.circuit.instructions
    reverse_gates = bind_register_program(
        inverse_record, inputs, clean=clean
    ).circuit.instructions
    values = torch.arange(1, 2 ** (logical + 1) + 1, dtype=torch.float64)
    amplitudes = torch.complex(values, values.flip(0))
    amplitudes /= amplitudes.norm()
    order = (*wires, logical, *clean)
    restore = tuple(order.index(wire) for wire in range(total))
    initial_view = torch.zeros((2**logical, 2, 2**work_size), dtype=torch.complex128)
    initial_view[:, :, 0] = amplitudes.reshape(2**logical, 2)
    initial = initial_view.reshape((2,) * total).permute(restore).flatten()
    expected_view = torch.zeros_like(initial_view)
    expected_view[:, :, 0] = torch.as_tensor(
        source.tensor_contract()
    ) @ amplitudes.reshape(2**logical, 2)
    expected = expected_view.reshape((2,) * total).permute(restore).flatten()
    count = sum(g.name == "measure" for g in gates)
    reverse_count = sum(g.name == "measure" for g in reverse_gates)
    for history in product((0, 1), repeat=count):
        branch = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            branch, expected / 2 ** (count / 2), atol=8e-12, rtol=0
        )
        work = (
            branch.reshape((2,) * total).permute(order).reshape(-1, 2**work_size)[:, 1:]
        )
        torch.testing.assert_close(work, torch.zeros_like(work), atol=8e-12, rtol=0)
        for second in product((0, 1), repeat=reverse_count):
            recovered = forced_branch(branch, reverse_gates, second)
            torch.testing.assert_close(
                recovered,
                initial / 2 ** ((count + reverse_count) / 2),
                atol=8e-12,
                rtol=0,
            )
    circuit = DynamicCircuit(total, inputs=initial, dtype=initial.dtype)
    circuit.append(FieldOperation(total, gates))
    result = run_dynamic(circuit, shots=4, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), expected.expand(4, -1), atol=8e-12, rtol=0
    )
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit


@pytest.mark.parametrize("identifier", FACTORIES[1:])
def test_qualified_children_reject_changed_phase_or_feedback(identifier: str) -> None:
    record = native.decomposition(identifier)
    assert record is not None
    functions = []
    for function in record.functions:
        blocks = []
        for block in function.blocks:
            operations = []
            for operation in block.operations:
                if operation.opcode == "phase":
                    operation = replace(operation, opcode="rz")
                elif operation.opcode == "s":
                    operation = replace(operation, opcode="z")
                elif operation.opcode == "cx":
                    operation = replace(operation, operands=operation.operands[::-1])
                operations.append(operation)
            blocks.append(replace(block, operations=tuple(operations)))
        functions.append(replace(function, blocks=tuple(blocks)))
    changed = replace(record, functions=tuple(functions))
    checked = 0
    for function in record.functions:
        if qvr_child(record, function.name, native.attributes(identifier)) is not None:
            assert (
                qvr_child(changed, function.name, native.attributes(identifier)) is None
            )
            checked += 1
    assert checked > 0


def test_original_twelve_bit_program_primitive_counts() -> None:
    record = native.decomposition("qvr_phase_gradient")
    assert record is not None
    bound = bind_register_program(
        record, (tuple(range(12)), tuple(range(12, 31))), clean=tuple(range(31, 49))
    )
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert (counts["t"] + counts["tdg"], counts["cx"], counts["measure"]) == (
        484,
        1244,
        121,
    )
    assert bound.outputs == (tuple(range(12)), tuple(range(12, 31)))


def test_colliding_source_root_filename_keeps_both_dependency_bodies() -> None:
    from examples.algorithms.component_ui_dependencies import (
        ComponentFunction,
        dependency_graph,
    )
    from flagquantum.compiler._component_trace.program_call_graph import (
        program_component_call_graph,
    )

    roots = FACTORIES[2:]
    assert native.filename(roots[0]) == native.filename(roots[1])
    graph = dependency_graph(
        roots,
        owner=native._dependency_owner,
        program=native.decomposition,
        child_presentation=native._child_presentation,
    )
    phase_names: set[str] = set()
    phase_owners: set[str] = set()
    direct_calls = {
        native._dependency_owner(root): program_component_call_graph(
            record, record.functions[0].name
        ).direct_calls
        for root in roots
        if (record := native.decomposition(root)) is not None
    }
    for node, children in graph.items():
        if not isinstance(node, ComponentFunction):
            continue
        expected = {
            child for parent, child in direct_calls[node.owner] if parent == node.entry
        }
        for child in children:
            assert isinstance(child, ComponentFunction)
        assert {
            child.entry for child in children if isinstance(child, ComponentFunction)
        } == expected
        assert all(
            child.program == node.program
            for child in children
            if isinstance(child, ComponentFunction)
        )
        presentation = node.presentation
        if presentation is not None and presentation[0] in {
            "QvrZPow",
            "QvrPhaseGradient",
        }:
            phase_names.add(presentation[0])
            phase_owners.add(node.owner)
    assert phase_names == {
        "QvrZPow",
        "QvrPhaseGradient",
    }
    assert len(phase_owners) == 2


@pytest.mark.parametrize("width", (2, 3))
def test_qvr_root_retains_original_refusing_scaled_child(width: int) -> None:
    from flagquantum.core.ir.program import verify_program_record

    cost = Register("out", fixed_point_register_type(width, 0))
    record = qvr_phase_gradient_program(cost, 0.1875, 0.2)
    assert verify_program_record(record) == ()
    assert len(record.functions[0].blocks[0].operations) == 1
    assert len([function for function in record.functions if not function.blocks]) == 1
    gradient = qvr_phase_format(width, 0, 0.1875, 0.2).gradient_width
    with pytest.raises(ValueError):
        bind_register_program(
            record,
            (tuple(range(width)), tuple(range(width, width + gradient))),
            clean=tuple(range(width + gradient, width + gradient + 10)),
        )
