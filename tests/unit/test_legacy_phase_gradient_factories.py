"""Stored gradient constructors preserve full phases, operands and measured work."""

from itertools import product
from math import sqrt

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_phase_gradient_factories import (
    phase_gradient_factories,
)
from flagquantum.compiler.registers.phase.phase_addition import phase_add_workspace_size
from flagquantum.compiler.registers.phase.phase_gradient_component import (
    phase_gradient_state_program,
    phase_gradient_unitary_program,
)
from flagquantum.compiler.registers.phase.scaled_phase_component import (
    phase_add_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.core.value_math import SymbolicInt
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,controlled,exponent", tuple(product((1, 2, 3), (False, True), (-1.0, 0.37)))
)
def test_gradient_unitary_complete_stored_graph_and_joint_operator(
    width: int, controlled: bool, exponent: float
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary

    source = PhaseGradientUnitary(width, exponent, controlled)
    record = resolve_component_program(
        _library(source, 0), 0, phase_gradient_factories()
    )
    assert verify_program_record(record) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    for depth in (1, 2):
        assert (
            resolve_component_program(
                _library(source, depth), 0, phase_gradient_factories()
            )
            == record
        )
    inputs = (
        ((0,), tuple(range(1, width + 1))) if controlled else (tuple(range(width)),)
    )
    logical = width + controlled
    clean = (logical,) if controlled else ()
    bound = bind_register_program(record, inputs, clean=clean)
    assert bound == bind_register_program(
        phase_gradient_unitary_program(width, exponent, controlled=controlled),
        inputs,
        clean=clean,
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.tensor(cirq.unitary(source), dtype=torch.complex128),
        atol=3e-12,
        rtol=0,
    )


@pytest.mark.parametrize("width,exponent", tuple(product((1, 2, 4), (-1.0, 0.37))))
def test_gradient_state_output_ownership_and_full_complex_tensor(
    width: int, exponent: float
) -> None:
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState

    source = PhaseGradientState(width, exponent)
    record = resolve_component_program(
        _library(source, 0), 0, phase_gradient_factories()
    )
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    for depth in (1, 2):
        assert (
            resolve_component_program(
                _library(source, depth), 0, phase_gradient_factories()
            )
            == record
        )
    clean = tuple(range(width))
    bound = bind_register_program(record, (), clean=clean, allow_workspace_outputs=True)
    assert bound == bind_register_program(
        phase_gradient_state_program(width, exponent),
        (),
        clean=clean,
        allow_workspace_outputs=True,
    )
    assert bound.outputs == (clean,)
    torch.testing.assert_close(
        logical_operator(bound.circuit, width)[:, 0],
        torch.tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=3e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "control,sign,shift", tuple(product((None, 0, 1), (-1, 1), (0, 1)))
)
def test_atomic_phase_addition_joint_operator_and_shared_components(
    control: int | None, sign: int, shift: int
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

    source = AddIntoPhaseGrad(2, 3, right_shift=shift, sign=sign, controlled_by=control)
    record = resolve_component_program(
        _library(source, 0), 0, phase_gradient_factories()
    )
    inputs = ((0,), (1, 2), (3, 4, 5)) if control is not None else ((0, 1), (2, 3, 4))
    logical = 5 + (control is not None)
    work = phase_add_workspace_size(
        2, 3, right_shift=shift, controlled=control is not None
    )
    clean = tuple(range(logical, logical + work))
    bound = bind_register_program(record, inputs, clean=clean)
    assert bound == bind_register_program(
        phase_add_component_program(
            2,
            3,
            right_shift=shift,
            sign=sign,
            controlled=control is not None,
            control_value=1 if control is None else control,
        ),
        inputs,
        clean=clean,
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.tensor(cirq.unitary(source), dtype=torch.complex128),
        atol=8e-12,
        rtol=0,
    )


@pytest.mark.parametrize("control,sign", tuple(product((None, 0, 1), (-1, 1))))
def test_phase_addition_every_small_raw_history_with_coherent_spectator(
    control: int | None, sign: int
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

    source = AddIntoPhaseGrad(1, 2, sign=sign, controlled_by=control)
    record = resolve_component_program(
        _library(source, 0), 0, phase_gradient_factories()
    )
    inputs = ((0,), (1,), (2, 3)) if control is not None else ((0,), (1, 2))
    logical = 3 + (control is not None)
    work = phase_add_workspace_size(1, 2, controlled=control is not None)
    bound = bind_register_program(
        record, inputs, clean=tuple(range(logical + 1, logical + 1 + work))
    )
    raw = torch.arange(1, 2 ** (logical + 1) + 1, dtype=torch.float64)
    coherent = torch.complex(raw, raw.flip(0))
    coherent /= coherent.norm()
    expected = (
        torch.tensor(cirq.unitary(source), dtype=torch.complex128)
        @ coherent.reshape(2**logical, 2)
    ).flatten()
    initial = torch.zeros(2 ** (logical + 1 + work), dtype=torch.complex128)
    initial[:: 2**work] = coherent
    target = torch.zeros_like(initial)
    target[:: 2**work] = expected
    count = sum(g.name == "measure" for g in bound.circuit.instructions)
    for history in product((0, 1), repeat=count):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            target / sqrt(2**count),
            atol=8e-12,
            rtol=0,
        )


@pytest.mark.parametrize("gamma", (-0.5, 0.0, 1.5))
def test_scaled_phase_source_slices_and_full_complex_operator(gamma: float) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran import QFxp
    from qualtran.bloqs.rotations.phase_gradient import AddScaledValIntoPhaseReg

    source = AddScaledValIntoPhaseReg(QFxp(2, 2), 3, gamma, QFxp(3, 2))
    record = resolve_component_program(
        _library(source, 0), 0, phase_gradient_factories()
    )
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    bound = bind_register_program(record, ((0, 1), (2, 3, 4)), clean=(5, 6))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 5),
        torch.tensor(cirq.unitary(source), dtype=torch.complex128),
        atol=8e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "controlled,state", ((False, False), (True, False), (False, True))
)
def test_gradient_symbolic_phase_uses_existing_append_parameter_binding(
    controlled: bool, state: bool
) -> None:
    from qualtran.bloqs.rotations.phase_gradient import (
        PhaseGradientState,
        PhaseGradientUnitary,
    )

    theta = sympy.Symbol("theta")
    source = (
        PhaseGradientState(2, theta)
        if state
        else PhaseGradientUnitary(2, theta, controlled)
    )
    record = resolve_component_program(
        _library(source, 0), 0, phase_gradient_factories()
    )
    logical = 2 + controlled
    inputs = () if state else ((0,), (1, 2)) if controlled else ((0, 1),)
    clean = (0, 1) if state else (logical,) if controlled else ()
    bound = bind_register_program(
        record, inputs, clean=clean, allow_workspace_outputs=state
    )
    circuit = Circuit(bound.circuit.n_wires, dtype=torch.complex128)
    circuit.append(FieldOperation(bound.circuit.n_wires, bound.circuit.instructions))
    assert circuit.parameter_names == ("theta",)
    numeric = 0.37
    resolved = circuit.bind_parameters({"theta": numeric})
    if controlled:
        actual = logical_operator(resolved.to_ir(), logical)
        expected = PhaseGradientUnitary(2, numeric, True).tensor_contract()
        torch.testing.assert_close(
            actual, torch.tensor(expected, dtype=torch.complex128), atol=3e-12, rtol=0
        )
    else:
        expected = (
            PhaseGradientState(2, numeric).tensor_contract()
            if state
            else PhaseGradientUnitary(2, numeric).tensor_contract()
        )
        actual = resolved.state().flatten() if state else get_unitary(resolved)
        torch.testing.assert_close(
            actual, torch.tensor(expected, dtype=torch.complex128), atol=3e-12, rtol=0
        )


@pytest.mark.parametrize("width", (0, sympy.Symbol("n"), sympy.Integer(3)))
def test_gradient_original_width_boundaries_and_declarations(
    width: SymbolicInt,
) -> None:
    from qualtran.bloqs.rotations.phase_gradient import (
        PhaseGradientState,
        PhaseGradientUnitary,
    )

    for kind in (PhaseGradientUnitary, PhaseGradientState):
        with pytest.raises((KeyError, TypeError, ValueError)):
            kind(width).decompose_bloq()
        factory = phase_gradient_factories()[
            "qualtran.bloqs.rotations.phase_gradient." + kind.__name__
        ]
        result = factory("direct", None, {"bitsize": width})
        assert isinstance(result, FunctionRecord) and not result.blocks


@pytest.mark.parametrize("controlled", (False, True))
def test_symbolic_gradient_exact_adjoint_recovers_joint_operator(
    controlled: bool,
) -> None:
    from flagquantum.compiler.formats.symbolic_parameters import (
        symbolic_parameter_value,
    )

    parameter = symbolic_parameter_value(sympy.Symbol("theta"))
    logical = 2 + controlled
    inputs = ((0,), (1, 2)) if controlled else ((0, 1),)
    clean = (logical,) if controlled else ()
    circuit = Circuit(logical + controlled, dtype=torch.complex128)
    for inverse in (False, True):
        program = phase_gradient_unitary_program(
            2, parameter, controlled=controlled, adjoint=inverse
        )
        bound = bind_register_program(program, inputs, clean=clean)
        circuit.append(FieldOperation(circuit.n_qubits, bound.circuit.instructions))
    ir = circuit.bind_parameters({"theta": 0.37}).to_ir()
    torch.testing.assert_close(
        logical_operator(ir, logical),
        torch.eye(2**logical, dtype=torch.complex128),
        atol=8e-12,
        rtol=0,
    )


def test_original_examples_and_invalid_constructor_metadata() -> None:
    from qualtran.bloqs.rotations import phase_gradient as source

    from flagquantum.compiler.formats.legacy_component_resolution import (
        ResolvedArgument,
    )
    from flagquantum.errors import SerializationError

    examples = (
        source._phase_gradient_unitary,
        source._phase_gradient_unitary_symbolic,
        source._phase_gradient_state,
        source._add_into_phase_grad,
        source._add_scaled_val_into_phase_reg,
    )
    for example in examples:
        op = example.make()
        library = _library(op, 0)
        record = resolve_component_program(library, 0, phase_gradient_factories())
        assert record.functions[0].registers == library.entries[0].declaration.registers
        factory = phase_gradient_factories()[library.entries[0].declaration.name]
        arguments: dict[str, ResolvedArgument] = {
            a.name: a.value for a in library.entries[0].declaration.arguments
        }
        with pytest.raises(SerializationError, match="signature"):
            factory("bad", (), arguments)
        with pytest.raises(SerializationError):
            factory("bad", None, {**arguments, "extra": 1})
        with pytest.raises(SerializationError):
            factory("bad", None, {})


def test_scaled_signed_storage_and_symbolic_coefficients_remain_explicit_boundaries() -> (
    None
):
    from qualtran import QFxp
    from qualtran.bloqs.rotations.phase_gradient import AddScaledValIntoPhaseReg

    cases = (
        AddScaledValIntoPhaseReg(QFxp(3, 2, signed=True), 3, 0.5, QFxp(3, 2)),
        AddScaledValIntoPhaseReg(QFxp(2, 2), 3, sympy.Symbol("gamma"), QFxp(3, 2)),
    )
    for source in cases:
        library = _library(source, 0)
        record = resolve_component_program(library, 0, phase_gradient_factories())
        assert record.functions[0].registers == library.entries[0].declaration.registers
        assert not record.functions[0].blocks
        with pytest.raises(ValueError, match="bodies"):
            bind_register_program(record, ())
