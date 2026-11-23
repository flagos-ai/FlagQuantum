"""Scalar constructor closure preserves zero ports, phase and clean carrier ownership."""

from collections import Counter
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_rotation_factories import rotation_factories
from flagquantum.compiler.formats.legacy_scalar_factories import scalar_phase_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
KEY = "qualtran.bloqs.basic_gates.global_phase.GlobalPhase"


def _resolve(source: Any) -> ProgramRecord:
    program = resolve_component_program(
        _library(source, 0), 0, {**scalar_phase_factories(), **rotation_factories()}
    )
    assert verify_program_record(program) == ()
    return program


@pytest.mark.parametrize("exponent", (0.0, 0.5, -0.7, 1.7, -2.3))
@pytest.mark.parametrize("adjoint", (False, True))
def test_complete_scalar_constructor_source_phase_private_carrier_and_separate_model(
    exponent: float, adjoint: bool
) -> None:
    from qualtran.bloqs.basic_gates import GlobalPhase
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = GlobalPhase(exponent=exponent, eps=1e-9)
    if adjoint:
        source = source.adjoint()
    program = _resolve(source)
    assert program.functions[0].registers == ()
    bound = bind_register_program(program, (), clean=(2,))
    assert bound.outputs == ()
    actual = Counter(g.name for g in bound.circuit.instructions)
    assert actual == ({} if exponent == 0 else {"x": 2, "phase": 2})
    assert [event.kind for event in bound.events] == (
        [] if exponent == 0 else ["allocate", "free"]
    )
    assert [(gate.name, gate.wires) for gate in bound.circuit.instructions] == (
        []
        if exponent == 0
        else [("x", (2,)), ("phase", (2,)), ("x", (2,)), ("phase", (2,))]
    )
    assert get_cost_value(source, QECGatesCost()).asdict() == {}
    assert get_cost_value(source, QubitCount()) == 0
    for precision in (torch.complex64, torch.complex128):
        initial = torch.tensor(
            [1 + 2j, 0, -2j, 0, 3 - 1j, 0, -2 + 1j, 0], dtype=precision
        )
        initial /= initial.norm()
        circuit = Circuit(3, inputs=initial, dtype=precision)
        circuit.append(
            FieldOperation(bound.circuit.n_wires, bound.circuit.instructions)
        )
        torch.testing.assert_close(
            circuit.state().flatten(),
            initial * complex(source.tensor_contract()),
            rtol=0,
            atol=2e-6 if precision == torch.complex64 else 3e-12,
        )
        assert not torch.count_nonzero(circuit.state().flatten()[1::2])


@pytest.mark.parametrize("exponent", (0.37, -1.7))
def test_original_positive_control_resolves_existing_z_power_dependency(
    exponent: float,
) -> None:
    from qualtran.bloqs.basic_gates import GlobalPhase, ZPowGate

    source = GlobalPhase(exponent=exponent).controlled()
    assert isinstance(source, ZPowGate)
    bound = bind_register_program(_resolve(source), ((0,),))
    assert tuple(g.name for g in bound.circuit.instructions) == ("phase",)
    initial = torch.tensor([1 + 2j, -3 + 1j], dtype=torch.complex128)
    initial /= initial.norm()
    actual = (
        Circuit(1, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(1, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(
        actual,
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128) @ initial,
        rtol=0,
        atol=3e-12,
    )


def test_symbolic_global_phase_binds_through_existing_parameter_path() -> None:
    from qualtran.bloqs.basic_gates import GlobalPhase

    theta = sympy.Symbol("theta", real=True)
    bound = bind_register_program(
        _resolve(GlobalPhase(exponent=sympy.sin(theta))), (), clean=(0,)
    )
    initial = torch.tensor([1, 0, 1j, 0], dtype=torch.complex128) / 2**0.5
    circuit = Circuit(2, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(1, bound.circuit.instructions)
    )
    actual = circuit.bind_parameters({"theta": 0.37}).state().flatten()
    expected = initial * complex(
        GlobalPhase(exponent=sympy.sin(theta)).coefficient.subs(theta, 0.37)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-12)


def test_scalar_constructor_rejects_invalid_arguments_and_nonempty_ports() -> None:
    factory = scalar_phase_factories()[KEY]
    for args in (
        {},
        {"exponent": 1, "other": 2},
        {"exponent": "phase"},
        {"exponent": float("inf")},
    ):
        with pytest.raises((ValueError, SerializationError)):
            factory("bad", None, args)
    with pytest.raises(SerializationError, match="empty"):
        factory("bad", (Register("q", quantum_bit_type()),), {"exponent": 1})
