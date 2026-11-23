"""Repeated constructors retain the independently resolved child and helper closure."""

from collections import Counter
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_rotation_factories import rotation_factories
from flagquantum.compiler.formats.legacy_scalar_factories import scalar_phase_factories
from flagquantum.compiler.formats.legacy_su2_factories import su2_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.errors import SerializationError
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_legacy_word_factories import _factories, _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _catalogue() -> dict[str, ComponentFactory]:
    return {
        **_factories(),
        **rotation_factories(),
        **scalar_phase_factories(),
        **su2_factories(),
    }


def _resolve(source: Any, depth: int = 0) -> ProgramRecord:
    result = resolve_component_program(_library(source, depth), 0, _catalogue())
    assert verify_program_record(result) == ()
    assert len({function.name for function in result.functions}) == len(
        result.functions
    )
    return result


@pytest.mark.parametrize("width", (0, 1, 3))
@pytest.mark.parametrize("adjoint", (False, True))
def test_su2_child_closure_exact_calls_gates_complex_amplitudes_and_clean_carrier(
    width: int, adjoint: bool
) -> None:
    from qualtran.bloqs.basic_gates import OnEach, SU2RotationGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    child = SU2RotationGate(0.2, -0.3, 0.4, 0.7, eps=1e-9)
    if adjoint:
        child = child.adjoint()
    source = OnEach(width, child)
    program = _resolve(source)
    assert _native_trace(program) == _source_trace(
        _source(Signature(program.functions[0].registers or ())),
        source.decompose_bloq(),
    )
    wires = tuple(reversed(range(width)))
    carrier = width + 1
    bound = bind_register_program(program, (wires,), clean=(carrier,))
    assert bound.outputs == (wires,)
    assert (
        bound.circuit
        == bind_register_program(
            _resolve(source, 1), (wires,), clean=(carrier,)
        ).circuit
    )
    expected_gates = tuple(
        gate
        for wire in wires
        for gate in bind_register_program(
            _resolve(child), ((wire,),), clean=(carrier,)
        ).circuit.instructions
    )
    assert bound.circuit.instructions == expected_gates
    actual = Counter(gate.name for gate in bound.circuit.instructions)
    assert actual == Counter(
        {"rz": 2 * width, "rx": width, "x": 2 * width, "phase": 2 * width}
    )
    assert [event.kind for event in bound.events] == ["allocate", "free"] * width
    assert not any(
        gate.name in {"measure", "reset"} for gate in bound.circuit.instructions
    )
    assert Counter(get_cost_value(source, QECGatesCost()).asdict()) == Counter(
        {
            key: width * value
            for key, value in get_cost_value(child, QECGatesCost()).asdict().items()
        }
    )
    assert source.build_call_graph(None) == {child: width}
    assert get_cost_value(source, QubitCount()) == width
    for precision in (torch.complex64, torch.complex128):
        values = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
        amplitudes = torch.complex(values, 2 - values).to(precision)
        amplitudes /= amplitudes.norm()
        initial = torch.zeros(1 << (width + 2), dtype=precision)
        initial[::2] = amplitudes
        matrix = torch.eye(1, dtype=precision)
        for _ in range(width):
            matrix = torch.kron(
                matrix, torch.as_tensor(child.rotation_matrix, dtype=precision)
            )
        expected = torch.zeros_like(initial)
        expected[::2] = (matrix @ amplitudes.reshape(1 << width, 2)).flatten()
        circuit = Circuit(width + 2, inputs=initial, dtype=precision).append(
            FieldOperation(width + 2, bound.circuit.instructions)
        )
        torch.testing.assert_close(
            circuit.state().flatten(),
            expected,
            rtol=0,
            atol=3e-6 if precision == torch.complex64 else 4e-12,
        )
        assert not torch.count_nonzero(circuit.state().flatten()[1::2])


def test_symbolic_repetition_keeps_real_child_closure_and_refuses_finite_binding() -> (
    None
):
    from qualtran.bloqs.basic_gates import OnEach, SU2RotationGate

    width = sympy.Symbol("n", integer=True, positive=True)
    source = OnEach(width, SU2RotationGate(0.2, 0.3, 0.4, 0.5))
    program = _resolve(source)
    assert not program.functions[0].blocks
    assert any(function.blocks for function in program.functions[1:])
    with pytest.raises(ValueError):
        bind_register_program(program, ((0,),), clean=(1,))
    from qualtran import DecomposeTypeError

    with pytest.raises(DecomposeTypeError):
        source.decompose_bloq()


def test_nested_scalar_child_references_reuse_the_catalogue_and_missing_owner_fails() -> (
    None
):
    from qualtran.bloqs.basic_gates import OnEach, SU2RotationGate

    source = OnEach(3, SU2RotationGate(0.2, 0.3, -0.4, 0.1))
    catalogue = _catalogue()
    del catalogue["qualtran.bloqs.basic_gates.su2_rotation.SU2RotationGate"]
    with pytest.raises(SerializationError, match="No native factory"):
        resolve_component_program(_library(source), 0, catalogue)
