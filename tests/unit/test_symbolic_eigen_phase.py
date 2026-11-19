"""Generic eigen-gate phases preserve both symbolic fields and full operators."""

from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.ecosystem.cirq import from_cirq, to_cirq
from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_cirq_feedback_import import source_branch

pytestmark = pytest.mark.unit
cirq = pytest.importorskip("cirq")
sympy = pytest.importorskip("sympy")
FAMILIES = (cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.ZZPowGate, cirq.CZPowGate)


def _gate(family: Any, *, resolved: bool = False) -> Any:
    theta, shift = sympy.symbols("theta shift")
    exponent = sympy.sin(theta)
    offset = sympy.cos(shift)
    if resolved:
        exponent = float(exponent.subs(theta, 0.37))
        offset = float(offset.subs(shift, 0.2))
    return family(exponent=exponent, global_shift=offset)


@pytest.mark.parametrize("family", FAMILIES)
def test_generic_eigen_fields_round_trip_and_match_original_matrix(family: Any) -> None:
    gate = _gate(family)
    qubits = tuple(cirq.LineQubit.range(cirq.num_qubits(gate)))
    original = cirq.Circuit(gate.on(*qubits))
    imported = from_cirq(original)
    native = Circuit.from_ir(imported, dtype=torch.complex128)
    assert native.parameter_names == ("shift", "theta")
    bound = native.bind_parameters({"theta": 0.37, "shift": 0.2})
    expected = cirq.unitary(_gate(family, resolved=True))
    torch.testing.assert_close(
        get_unitary(bound), torch.as_tensor(expected), atol=4e-15, rtol=0
    )
    restored = to_cirq(imported)
    np.testing.assert_allclose(
        cirq.unitary(cirq.resolve_parameters(restored, {"theta": 0.37, "shift": 0.2})),
        expected,
        atol=4e-15,
        rtol=0,
    )
    assert [g.name for g in imported.instructions][-4:] == ["x", "phase", "x", "phase"]
    assert len(imported.instructions) == 5
    # Original Cirq protocol omits the shift field from discovery and resolution.
    assert cirq.parameter_names(gate) == {"theta"}
    assert cirq.resolve_parameters(
        gate, {"theta": 0.37, "shift": 0.2}
    ).global_shift == sympy.cos(sympy.Symbol("shift"))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("polarity", (0, 1))
def test_generic_eigen_feedback_all_branches_with_clean_record(
    family: Any, polarity: int
) -> None:
    width = cirq.num_qubits(_gate(family))
    order = tuple(cirq.LineQubit.range(width + 1))
    record = order[-1]
    condition = cirq.SympyCondition(sympy.Eq(sympy.Symbol("m"), polarity))

    def circuit(gate: Any) -> Any:
        return cirq.Circuit(
            cirq.H(record),
            cirq.measure(record, key="m"),
            gate.on(*order[:-1]).with_classical_controls(condition),
            cirq.reset(record),
            strategy=cirq.InsertStrategy.NEW,
        )

    imported = import_cirq_feedback(circuit(_gate(family)), qubit_order=order).ir
    bound = (
        Circuit.from_ir(imported).bind_parameters({"theta": 0.37, "shift": 0.2}).to_ir()
    )
    initial = np.zeros(2 ** len(order), dtype=np.complex128)
    initial[::2] = np.arange(1, 2**width + 1) * (1 + 2j)
    initial /= np.linalg.norm(initial)
    for outcome in (0, 1):
        expected = source_branch(
            circuit(_gate(family, resolved=True)), order, initial, (outcome,)
        )
        actual = forced_branch(torch.as_tensor(initial), bound.instructions, (outcome,))
        torch.testing.assert_close(
            actual, torch.as_tensor(expected), atol=5e-15, rtol=0
        )
        assert torch.count_nonzero(actual[1::2]) == 0


@pytest.mark.parametrize("family", (cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate))
@pytest.mark.parametrize("values", ((0,), (1,), (0, 1), (1, 0)))
def test_generic_eigen_nested_controls_match_source_with_spectator(
    family: Any, values: tuple[int, ...]
) -> None:
    controls = len(values)
    qubits = tuple(cirq.LineQubit.range(controls + 1))
    operation = (
        _gate(family).on(qubits[-1]).controlled_by(*qubits[:-1], control_values=values)
    )
    program = import_cirq_register_optree(operation)
    bound = bind_register_program(
        program,
        tuple((i,) for i in range(controls + 1)),
        clean=tuple(range(controls + 1, controls + 4)),
    )
    # Keep the tensor spectator distinct from any clean decomposition workspace.
    width = bound.circuit.n_wires
    initial = torch.zeros(2 ** (width + 1), dtype=torch.complex128)
    logical = torch.arange(1, 2 ** (controls + 2) + 1, dtype=torch.float64).to(
        torch.complex128
    ) * (1 + 1j)
    logical /= logical.norm()
    for index in range(2 ** (controls + 1)):
        initial[
            index * 2 ** (width - controls) : index * 2 ** (width - controls) + 2
        ] = logical[2 * index : 2 * index + 2]
    circuit = Circuit(width + 1, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(width + 1, bound.circuit.instructions)
    )
    resolved = circuit.bind_parameters({"theta": 0.37, "shift": 0.2}).to_ir()
    expected_gate = (
        _gate(family, resolved=True)
        .on(qubits[-1])
        .controlled_by(*qubits[:-1], control_values=values)
    )
    expected = (
        torch.as_tensor(np.kron(cirq.unitary(expected_gate), np.eye(2))) @ logical
    )
    measurements = sum(g.name == "measure" for g in resolved.instructions)
    from itertools import product

    for outcomes in product((0, 1), repeat=measurements):
        actual = forced_branch(initial, resolved.instructions, outcomes)
        packed = torch.cat(
            [
                actual[
                    index * 2 ** (width - controls) : index * 2 ** (width - controls)
                    + 2
                ]
                for index in range(2 ** (controls + 1))
            ]
        )
        torch.testing.assert_close(
            packed, expected / 2 ** (measurements / 2), atol=8e-15, rtol=0
        )
        assert torch.count_nonzero(actual) == torch.count_nonzero(packed)
