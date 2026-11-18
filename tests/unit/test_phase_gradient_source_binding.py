"""Resolve the complete gradient file at its original directional boundaries."""

from collections import Counter
from dataclasses import replace
from math import pi, sqrt
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.circuit_register_emitter import (
    CircuitRegisterEmitter,
)
from flagquantum.compiler.registers.phase.phase_addition import phase_add_workspace_size
from flagquantum.compiler.registers.phase.phase_gradient import (
    phase_gradient_state_ir,
    phase_gradient_unitary_ir,
    source_phase_gradient_unitary_cost,
)
from flagquantum.compiler.registers.phase.scaled_phase_addition import (
    scaled_phase_add_ir,
    scaled_phase_add_terms,
    source_scaled_phase_add_cost,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_phase_gradient_file import source_module as source_module
from tests.unit.test_scaled_phase_addition import ScaledAddArguments

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def restored(
    source: Any, *, local_ports: tuple[str, ...] = ()
) -> CircuitRegisterEmitter:
    emitter = import_register_emitter(source, local_ports=local_ports)
    return replace(
        emitter,
        snapshots=(
            CircuitIR.from_json(emitter.snapshots[0].to_json()),
            CircuitIR.from_json(emitter.snapshots[1].to_json()),
        ),
    )


def through_operators(
    emitter: CircuitRegisterEmitter,
) -> tuple[torch.Tensor, torch.Tensor]:
    logical = sum(width for _, width in emitter.layout)
    work, slots = emitter.requirements()
    packed = BlackBoxPrepare(emitter.layout, (), emitter)
    matrices = []
    for adjoint in (False, True):
        ir = (
            Circuit(logical + work)
            .append(
                packed.make_on(
                    selection=tuple(range(logical)),
                    junk=(),
                    workspace=tuple(range(logical, logical + work)),
                    classical_bits=tuple(range(slots)),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        assert isinstance(ir, CircuitIR)
        matrices.append(logical_operator(ir, logical))
    torch.testing.assert_close(
        matrices[1] @ matrices[0],
        torch.eye(1 << logical, dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )
    return matrices[0], matrices[1]


@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("controlled", [False, True])
def test_original_symbolic_gradient_actual_execution(
    source_module: Any, width: int, controlled: bool
) -> None:
    from attrs import evolve

    declaration = source_module._phase_gradient_unitary_symbolic.make()
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(declaration)
    exponent = -0.375
    source = evolve(
        declaration, bitsize=width, exponent=exponent, is_controlled=controlled
    )
    assert_source_model(
        source_phase_gradient_unitary_cost(width, exponent, controlled=controlled),
        source,
    )
    emitter = restored(source)
    actual = through_operators(emitter)
    logical = width + controlled
    gradient = tuple(range(int(controlled), logical))
    for inverse in (False, True):
        gates = phase_gradient_unitary_ir(
            gradient,
            exponent=-exponent if inverse else exponent,
            control=0 if controlled else None,
            work=logical if controlled else None,
            classical_bit=0 if controlled else None,
        )
        native = CircuitIR(logical + controlled, gates)
        assert Counter(
            g.name for g in emitter.snapshots[int(inverse)].instructions
        ) == Counter(g.name for g in native.instructions)
        torch.testing.assert_close(
            actual[int(inverse)], logical_operator(native, logical), atol=1e-11, rtol=0
        )
    diagonal = torch.ones(1 << logical, dtype=torch.complex128)
    values = torch.arange(1 << width, dtype=torch.float64)
    phases = torch.exp(2j * pi * exponent * values / (1 << width))
    diagonal[-len(phases) :] = phases
    torch.testing.assert_close(actual[0], torch.diag(diagonal), atol=1e-11, rtol=0)


@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("exponent", [-1.0, 1.0, -0.375])
def test_right_gradient_output_has_local_ownership(
    source_module: Any, width: int, exponent: float
) -> None:
    source = source_module.PhaseGradientState(width, exponent)
    emitter = restored(source, local_ports=("phase_grad",))
    native = CircuitRegisterEmitter(
        (("phase_grad", width),),
        (
            CircuitIR(
                width, phase_gradient_state_ir(tuple(range(width)), exponent=exponent)
            ),
            CircuitIR(
                width,
                phase_gradient_state_ir(
                    tuple(range(width)), exponent=exponent, inverse=True
                ),
            ),
        ),
    )
    spectator = torch.tensor([1, 2j], dtype=torch.complex128) / sqrt(5)
    initial = torch.zeros(1 << (width + 1), dtype=torch.complex128)
    initial[:: 1 << width] = spectator
    gradient = torch.exp(
        2j
        * pi
        * exponent
        * torch.arange(1 << width, dtype=torch.float64)
        / (1 << width)
    ) / sqrt(1 << width)
    expected = torch.kron(spectator, gradient)
    for implementation in (emitter, native):
        oracle = BlackBoxPrepare(
            (), (), implementation, local_junk_layout=(("phase_grad", width),)
        )
        assert oracle.selection_width == oracle.junk_width == 0
        assert oracle.local_junk_width == width and implementation.requirements() == (
            0,
            0,
        )
        state = initial
        for adjoint in (False, True):
            ir = (
                Circuit(width + 1)
                .append(
                    oracle.make_on(
                        selection=(),
                        junk=(),
                        workspace=tuple(range(1, width + 1)),
                        classical_bits=(),
                        adjoint=adjoint,
                    )
                )
                .to_ir()
            )
            assert all(0 not in gate.wires for gate in ir.instructions)
            state = forced_branch(state, ir.instructions, ())
            torch.testing.assert_close(
                state, initial if adjoint else expected, atol=1e-11, rtol=0
            )


@pytest.mark.parametrize("coefficient", [3.25, -3.25])
def test_actual_admitted_mixed_fixed_point_scaled_add(
    source_module: Any, coefficient: float
) -> None:
    from qualtran import QFxp

    source = source_module.AddScaledValIntoPhaseReg(
        QFxp(5, 3), 4, coefficient, QFxp(4, 2)
    )
    args: ScaledAddArguments = dict(
        input_fraction_bits=3,
        coefficient=coefficient,
        coefficient_width=4,
        coefficient_fraction_bits=2,
    )
    assert_source_model(
        source_scaled_phase_add_cost(5, 3, 4, coefficient, 4, 2), source
    )
    emitter = restored(source)
    source_matrices = through_operators(emitter)
    terms = scaled_phase_add_terms(5, 3, 4, coefficient, 4, 2)
    work = max(
        phase_add_workspace_size(len(tuple(range(5))[part]), 4, right_shift=shift)
        for part, shift in terms
    )
    forward = None
    for inverse in (False, True):
        gates = scaled_phase_add_ir(
            tuple(range(5)),
            tuple(range(5, 9)),
            tuple(range(9, 9 + work)),
            tuple(range(work)),
            adjoint=inverse,
            **args,
        )
        native = CircuitIR(9 + work, gates)
        assert Counter(
            g.name for g in emitter.snapshots[int(inverse)].instructions
        ) == Counter(g.name for g in native.instructions)
        operator = logical_operator(native, 9)
        torch.testing.assert_close(
            source_matrices[int(inverse)], operator, atol=1e-11, rtol=0
        )
        if not inverse:
            forward = operator
    assert forward is not None
    expected = torch.zeros_like(forward)
    for x in range(32):
        for phase in range(16):
            result = source.on_classical_vals(x=x, phase_grad=phase)
            expected[x * 16 + int(result["phase_grad"]), x * 16 + phase] = 1
    torch.testing.assert_close(forward, expected, atol=1e-11, rtol=0)


def test_mixed_dtype_error_and_abstract_negative_model_remain(
    source_module: Any,
) -> None:
    from qualtran import BloqError, QFxp

    source = source_module.AddScaledValIntoPhaseReg(QFxp(3, 2), 4, 0.375, QFxp(3, 3))
    with pytest.raises(BloqError, match="register dtypes are not consistent"):
        import_register_emitter(source)
    addition = source_module.AddIntoPhaseGrad(1, 1)
    assert tuple(addition.build_call_graph(None).values()) == (-1,)
    from qualtran import DecomposeNotImplementedError

    with pytest.raises(DecomposeNotImplementedError):
        addition.decompose_bloq()


@pytest.mark.parametrize("controlled", [False, True])
def test_zero_width_unitary_keeps_original_port_failure(
    source_module: Any, controlled: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = source_module.PhaseGradientUnitary(0, is_controlled=controlled)
    assert not get_cost_value(source, QECGatesCost()).asdict()
    with pytest.raises(RuntimeError, match="phase_grad"):
        get_cost_value(source, QubitCount())
    with pytest.raises(KeyError, match="phase_grad"):
        import_register_emitter(source)


def test_zero_width_right_state_retains_its_declaration_and_failure(
    source_module: Any,
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = source_module.PhaseGradientState(0)
    (port,) = tuple(source.signature)
    assert port.name == "phase_grad" and port.bitsize == 0 and port.side.name == "RIGHT"
    with pytest.raises(KeyError):
        get_cost_value(source, QECGatesCost())
    with pytest.raises(RuntimeError, match="qubit count"):
        get_cost_value(source, QubitCount())
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(source, local_ports=("phase_grad",))
