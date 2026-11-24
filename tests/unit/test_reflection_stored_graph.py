"""Frozen zero-reflection dependencies and complete complex branch operators."""

from itertools import product

import numpy as np
import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.algorithms.primitives.prepared_state_reflection import (
    PreparedStateReflection,
    zero_state_reflection_program,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.control.multi_control_component import (
    multi_control_z_program,
)
from flagquantum.compiler.registers.primitives.identity import register_identity_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("widths", [(1,), (2,), (1, 2), (1, 2, 3)])
@pytest.mark.parametrize("control", [None, 0, 1])
@pytest.mark.parametrize("coefficient", [1, -1, 1j, -1j, np.exp(0.31j)])
def test_zero_reflection_source_calls_and_complete_small_operators(
    widths: tuple[int, ...], control: int | None, coefficient: complex
) -> None:
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    reflection = PreparedStateReflection.around_zero(widths, control, coefficient)
    graph = zero_state_reflection_program(reflection)
    source = ReflectionUsingPrepare.reflection_around_zero(
        widths, control_val=control, global_phase=coefficient
    )
    # Identity is source-atomic; its native per-bit implementation is tested above.
    terminals = atomic_leaves(graph)
    terminals.update(f.name for f in graph.functions[1:] if f.registers == ())
    for function in graph.functions[1:]:
        if any(
            matches_component_definition(
                graph, function.name, register_identity_program(width)
            )
            for width in widths
        ):
            terminals.add(function.name)
    compare_layers(graph, source, terminals)
    assert verify_program_record(graph) == ()
    identity_cost = ComponentCost("IdentityPrepare", sum(widths))
    assert_source_model(reflection.resource_model(identity_cost, identity_cost), source)
    logical = sum(widths) + (control is not None)
    offset = int(control is not None)
    inputs: list[tuple[int, ...]] = [(0,)] if control is not None else []
    for width in widths:
        wires = tuple(range(offset, offset + width))
        inputs.append(wires)
        offset += width
    clean = tuple(range(logical, logical + sum(widths) - int(control is not None)))
    bound = bind_register_program(graph, tuple(inputs), clean=clean)
    copied = copy_component_program(graph)
    assert (
        bind_register_program(copied, tuple(inputs), clean=clean).circuit
        == bound.circuit
    )
    operation = _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
    appended = Circuit(bound.circuit.n_wires).append(operation).to_ir()
    assert appended.instructions == bound.circuit.instructions
    assert bound.circuit.from_json(bound.circuit.to_json()) == bound.circuit
    if sum(widths) <= 3:
        expected = torch.as_tensor(source.tensor_contract())
        torch.testing.assert_close(
            logical_operator(bound.circuit, logical), expected, atol=1e-11, rtol=0
        )
        count = sum(g.name == "measure" for g in bound.circuit.instructions)
        stride = 1 << (bound.circuit.n_wires - logical)
        initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
        initial[::stride] = torch.tensor(
            [complex(i + 1, 3 - i) for i in range(1 << logical)]
        )
        initial /= initial.norm()
        output = torch.zeros_like(initial)
        output[::stride] = expected @ initial[::stride]
        for history in product((0, 1), repeat=count):
            actual = forced_branch(initial, bound.circuit.instructions, history)
            torch.testing.assert_close(
                actual, output / 2 ** (count / 2), atol=1e-11, rtol=0
            )
        inverse = zero_state_reflection_program(
            PreparedStateReflection.around_zero(
                widths, control, coefficient.conjugate()
            )
        )
        backwards = bind_register_program(inverse, tuple(inputs), clean=clean).circuit
        torch.testing.assert_close(
            logical_operator(backwards, logical), expected.mH, atol=1e-11, rtol=0
        )
        for history in product((0, 1), repeat=2 * count):
            restored = forced_branch(
                initial, (*bound.circuit.instructions, *backwards.instructions), history
            )
            torch.testing.assert_close(restored, initial / 2**count, atol=1e-11, rtol=0)
        # The unchanged source generator and stored Bloq recipes use different
        # atomic phase forms. Full complex operators above include their phase.


@pytest.mark.parametrize(
    "values", [p for n in range(4) for p in product((0, 1), repeat=n)]
)
def test_multi_control_z_all_small_histories_and_complex_operators(
    values: tuple[int, ...],
) -> None:
    from qualtran.bloqs.mcmt import MultiControlZ

    graph = multi_control_z_program(values)
    width = len(values)
    bound = bind_register_program(
        graph,
        (*((i,) for i in range(width)), (width,)),
        clean=tuple(range(width + 1, 2 * width)),
    )
    expected = torch.as_tensor(MultiControlZ(values).tensor_contract())
    torch.testing.assert_close(
        logical_operator(bound.circuit, width + 1), expected, atol=1e-11, rtol=0
    )
    count = sum(g.name == "measure" for g in bound.circuit.instructions)
    stride = 1 << (bound.circuit.n_wires - width - 1)
    state = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    state[::stride] = torch.tensor(
        [complex(i + 1, i - 2) for i in range(1 << (width + 1))]
    )
    state /= state.norm()
    output = torch.zeros_like(state)
    output[::stride] = expected @ state[::stride]
    for history in product((0, 1), repeat=count):
        torch.testing.assert_close(
            forced_branch(state, bound.circuit.instructions, history),
            output / 2 ** (count / 2),
            atol=1e-11,
            rtol=0,
        )
