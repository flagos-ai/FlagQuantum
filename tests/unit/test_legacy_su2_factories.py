"""Complete SU(2) constructor graphs preserve source calls, phases and work ownership."""

from collections import Counter
from importlib import import_module
from math import pi
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_rotation_factories import rotation_factories
from flagquantum.compiler.formats.legacy_scalar_factories import scalar_phase_factories
from flagquantum.compiler.formats.legacy_su2_factories import su2_factories
from flagquantum.compiler.registers.phase.su2_rotation import su2_rotation_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.errors import SerializationError
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
KEY = "qualtran.bloqs.basic_gates.su2_rotation.SU2RotationGate"


def _resolve(source: Any, depth: int = 0) -> ProgramRecord:
    program = resolve_component_program(
        _library(source, depth),
        0,
        {**rotation_factories(), **scalar_phase_factories(), **su2_factories()},
    )
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize(
    "angles",
    (
        (0.2, 0.3, -0.4, 0.1),
        (0.2, 0.3, -0.4, 7.2),
        (0.0, 0.0, 0.0, 0.0),
        (pi / 2, 0.0, 0.0, 0.0),
    ),
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_named_constructor_preserves_source_calls_gates_complex_state_and_resources(
    angles: tuple[float, float, float, float], adjoint: bool
) -> None:
    from qualtran.bloqs.basic_gates import GlobalPhase, SU2RotationGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = SU2RotationGate(*angles, eps=1e-9)
    if adjoint:
        source = source.adjoint()
    named = _resolve(source)
    with pytest.raises(SerializationError, match="zero-port execution child"):
        _resolve(source.decompose_bloq(), 1)
    assert _native_trace(named) == _source_trace(
        _source(Signature.build(q=1)), source.decompose_bloq()
    )
    assert len(named.functions[0].blocks[0].operations) == 4
    before = ((0,),)
    bound = bind_register_program(named, before, clean=(2,))
    assert bound.outputs == before
    original_steps = tuple(
        instance.bloq for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
    )
    from flagquantum.compiler.registers.primitives.global_phase import scalar_phase_ir
    from flagquantum.compiler.registers.primitives.pauli_rotation import (
        pauli_rotation_ir,
    )

    expected_gates = tuple(
        gate
        for step in original_steps
        for gate in (
            scalar_phase_ir(2, float(step.exponent) * pi)
            if isinstance(step, GlobalPhase)
            else pauli_rotation_ir(
                "x" if type(step).__name__ == "Rx" else "z", 0, float(step.angle)
            )
        )
    )
    assert bound.circuit.instructions == expected_gates
    assert not any(g.name in {"measure", "reset"} for g in bound.circuit.instructions)
    calls = su2_rotation_calls(
        source.theta,
        source.phi,
        source.lambd,
        global_shift=source.global_shift,
        precision=source.eps,
    )
    model: Counter[str] = Counter()
    for call in calls:
        model.update(query_qec_gate_cost(call).asdict())
    assert dict(model) == get_cost_value(source, QECGatesCost()).asdict()
    assert get_cost_value(source, QubitCount()) == 1
    actual = Counter(g.name for g in bound.circuit.instructions)
    assert actual["rz"] == 2 and actual["rx"] == 1
    assert actual["phase"] == actual["x"] == 2
    assert [event.kind for event in bound.events] == ["allocate", "free"]
    for precision in (torch.complex64, torch.complex128):
        initial = torch.zeros(8, dtype=precision)
        amplitudes = torch.tensor([1 + 2j, -2j, 3 - 1j, -2 + 1j], dtype=precision)
        amplitudes /= amplitudes.norm()
        initial[::2] = amplitudes
        matrix = torch.as_tensor(source.rotation_matrix, dtype=precision)
        expected = torch.zeros_like(initial)
        expected[::2] = (matrix @ amplitudes.reshape(2, 2)).flatten()
        ir = CircuitIR.from_json(bound.circuit.to_json())
        circuit = Circuit(3, inputs=initial, dtype=precision).append(
            FieldOperation(3, ir.instructions)
        )
        torch.testing.assert_close(
            circuit.state().flatten(),
            expected,
            rtol=0,
            atol=2e-6 if precision == torch.complex64 else 3e-12,
        )
        assert not torch.count_nonzero(circuit.state().flatten()[1::2])


@pytest.mark.parametrize("example", ("_su2_rotation_gate", "_hadamard", "_t_gate"))
def test_complete_original_examples_and_factory_alternatives(example: str) -> None:
    from qualtran.bloqs.basic_gates import SU2RotationGate

    source = getattr(
        import_module("qualtran.bloqs.basic_gates.su2_rotation"), example
    )()
    sources = (
        source,
        SU2RotationGate.from_matrix(source.rotation_matrix),
        SU2RotationGate.from_euler_zxz_angles(
            0.3, 0.7, -0.5, global_shift=0.2, eps=None
        ),
    )
    for original in sources:
        bound = bind_register_program(_resolve(original), ((0,),), clean=(1,))
        initial = torch.tensor([1, 0, 2j, 0], dtype=torch.complex128) / 5**0.5
        expected = torch.zeros_like(initial)
        expected[::2] = torch.as_tensor(original.rotation_matrix) @ initial[::2]
        actual = (
            Circuit(2, inputs=initial, dtype=torch.complex128)
            .append(FieldOperation(2, bound.circuit.instructions))
            .state()
            .flatten()
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=3e-12)


def test_symbolic_source_decomposition_retains_call_angles_precision_and_binding() -> (
    None
):
    from qualtran.bloqs.basic_gates import GlobalPhase, SU2RotationGate

    theta, phi, lambd, alpha, eps = sympy.symbols(
        "theta phi lambd alpha eps", real=True
    )
    source = SU2RotationGate(sympy.sin(theta), phi, lambd, alpha, eps)
    program = _resolve(source)
    with pytest.raises(SerializationError, match="zero-port execution child"):
        _resolve(source.decompose_bloq(), 1)
    bound = bind_register_program(program, ((0,),), clean=(1,))
    assert _native_trace(program) == _source_trace(
        _source(Signature.build(q=1)), source.decompose_bloq()
    )
    children = source.decompose_bloq().bloq_instances
    assert sum(isinstance(instance.bloq, GlobalPhase) for instance in children) == 1
    for instance in children:
        assert instance.bloq.eps == (
            1e-11 if isinstance(instance.bloq, GlobalPhase) else eps / 3
        )
    values = {"theta": 0.37, "phi": -0.31, "lambd": 0.82, "alpha": -0.23}
    initial = torch.tensor([1, 0, 2j, 0], dtype=torch.complex128) / 5**0.5
    circuit = Circuit(2, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(2, bound.circuit.instructions)
    )
    actual = circuit.bind_parameters(values).state().flatten()
    numeric = SU2RotationGate(
        float(sympy.sin(0.37)), values["phi"], values["lambd"], values["alpha"]
    )
    expected = torch.zeros_like(initial)
    expected[::2] = torch.as_tensor(numeric.rotation_matrix) @ initial[::2]
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-12)
    with pytest.raises(ValueError, match="Symbolic"):
        source.rotation_matrix


def test_su2_constructor_rejects_invalid_parameters_and_signature() -> None:
    factory = su2_factories()[KEY]
    for args in (
        {},
        {"theta": 1, "phi": 2},
        {"theta": 1, "phi": 2, "lambd": "bad"},
        {"theta": 1, "phi": 2, "lambd": 3, "extra": 1},
    ):
        with pytest.raises(SerializationError):
            factory("bad", None, args)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"theta": 1, "phi": 2, "lambd": 3})
