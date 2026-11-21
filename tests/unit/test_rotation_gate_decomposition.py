"""Controlled rotation catalog graphs retain exact source calls and phases."""

from typing import Any

import pytest
import torch

from examples.algorithms import rotations_ui_export as native
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    controlled_rz_ir,
    controlled_z_power_ir,
)
from flagquantum.compiler.registers.primitives.controlled_rotation_component import (
    controlled_rz_program,
    controlled_z_power_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.parameters import Parameter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_field_product_decomposition import source_walk
from tests.unit.test_rotations_ui_export import originals as originals

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("identifier", ("cz_pow", "crz"))
def test_complete_controlled_source_calls_and_fresh_drawings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    rows = source_walk(identifier, originals[1][identifier], catalog=native)
    assert sum(row["disposition"] == "exact_graph" for row in rows) == 1
    assert all(row["disposition"] != "original_ignored_adjoint_defect" for row in rows)


@pytest.mark.parametrize("identifier", ("z_pow", "rz", "x_pow", "y_pow"))
def test_original_atomic_rotation_refusal(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    from qualtran import DecomposeTypeError

    with pytest.raises(DecomposeTypeError):
        originals[1][identifier].decompose_bloq()
    assert native.decomposition(identifier) is None


@pytest.mark.parametrize("controlled_power", (False, True))
@pytest.mark.parametrize("parameter", (0.123, -0.4, 0.0, 1.0))
def test_exact_component_stream_all_branches_work_zero_and_inverse(
    controlled_power: bool, parameter: float
) -> None:
    from qualtran.bloqs.basic_gates import CRz, CZPowGate

    record = (
        controlled_z_power_program(parameter)
        if controlled_power
        else controlled_rz_program(parameter)
    )
    assert copy_component_program(record) == record
    bound = bind_register_program(
        record, ((2,), (0,)), clean=(3,) if controlled_power else ()
    )
    gates = bound.circuit.instructions
    reference = (
        controlled_z_power_ir(2, 0, parameter, 3, 0)
        if controlled_power
        else controlled_rz_ir(2, 0, parameter)
    )
    assert gates == reference
    source = CZPowGate(parameter) if controlled_power else CRz(parameter)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    width = bound.circuit.n_wires
    values = torch.arange(1, 9, dtype=torch.float64)
    amplitudes = torch.complex(values, values.flip(0))
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    initial[:: (2 if controlled_power else 1)] = amplitudes
    order = (2, 0, 1, 3) if controlled_power else (2, 0, 1)
    reverse = tuple(order.index(wire) for wire in range(width))
    expected = (
        (matrix @ initial.reshape((2,) * width).permute(order).reshape(4, -1))
        .reshape((2,) * width)
        .permute(reverse)
        .flatten()
    )
    inverse = bind_register_program(
        (
            controlled_z_power_program(-parameter)
            if controlled_power
            else controlled_rz_program(-parameter)
        ),
        ((2,), (0,)),
        clean=(3,) if controlled_power else (),
    ).circuit.instructions
    for history in ((0,), (1,)) if controlled_power else ((),):
        branch = forced_branch(initial, gates, history)
        scale = 2**0.5 if controlled_power else 1
        torch.testing.assert_close(branch, expected / scale, atol=3e-12, rtol=0)
        if controlled_power:
            assert torch.count_nonzero(branch[1::2]) == 0
        for second in ((0,), (1,)) if controlled_power else ((),):
            recovered = forced_branch(branch, inverse, second)
            torch.testing.assert_close(
                recovered, initial / (scale**2), atol=3e-12, rtol=0
            )
    if controlled_power:
        from flagquantum.dynamic import DynamicCircuit
        from flagquantum.runtime.dynamic.execution import run_dynamic

        dynamic = DynamicCircuit(width, inputs=initial, dtype=initial.dtype)
        dynamic.append(FieldOperation(width, gates))
        trajectories = run_dynamic(dynamic, shots=16, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            trajectories.final_states.reshape(16, -1),
            expected.expand(16, -1),
            atol=3e-12,
            rtol=0,
        )
    else:
        actual = (
            Circuit(width, inputs=initial, dtype=initial.dtype)
            .append(FieldOperation(width, gates))
            .state()
            .flatten()
        )
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=0)
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit


def test_owned_symbolic_angle_through_catalog_append(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    import cirq
    import numpy as np

    record = native.decomposition("crz")
    assert record is not None
    bound = bind_register_program(record, ((1,), (0,)))
    initial = torch.tensor([1, 2j, -3, 4j], dtype=torch.complex128)
    initial /= initial.norm()
    circuit = Circuit(2, inputs=initial, dtype=initial.dtype).append(
        FieldOperation(2, bound.circuit.instructions)
    )
    assert circuit.parameter_names == (r"\theta",)
    for value in (0.0, 0.123, -0.7):
        source = originals[1]["crz"]
        matrix = cirq.unitary(
            cirq.resolve_parameters(
                source.decompose_bloq().to_cirq_circuit(), {r"\theta": value}
            )
        )
        expected = (
            torch.as_tensor(np.asarray(matrix), dtype=initial.dtype)
            @ initial.reshape(2, 2).T.flatten()
        )
        expected = expected.reshape(2, 2).T.flatten()
        torch.testing.assert_close(
            circuit.bind_parameters({r"\theta": value}).state().flatten(),
            expected,
            atol=3e-12,
            rtol=0,
        )
    differentiable = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    actual = circuit.bind_parameters({r"\theta": differentiable}).state().flatten()
    gradient = torch.autograd.grad(actual[1].real, differentiable)[0]
    expected_derivative = (initial[1] * torch.exp(-0.5j * differentiable) * -0.5j).real
    torch.testing.assert_close(gradient, expected_derivative, atol=3e-12, rtol=0)
    assert controlled_rz_program(Parameter(r"\theta")) == record


@pytest.mark.parametrize("parameter", (True, float("inf"), float("nan"), 10**400))
@pytest.mark.parametrize("factory", (controlled_rz_program, controlled_z_power_program))
def test_existing_concrete_angle_admission_is_reused(
    parameter: float, factory: Any
) -> None:
    with pytest.raises(ValueError, match="finite phase angle"):
        factory(parameter)
