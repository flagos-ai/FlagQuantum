"""Verify the remaining concrete foreign control protocols against frozen source."""

from typing import Any

import pytest

from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.ecosystem.cirq._control_values import (
    export_control_values,
    import_control_values,
)
from flagquantum.ecosystem.cirq._register_control import control_register_operation
from tests.unit.test_cirq_register_control import _assert_appended

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
cirq = pytest.importorskip("cirq")
q = pytest.importorskip("qualtran")


def _check_foreign_control(dtype: Any, source_dtype: Any, value: Any) -> None:
    """Exercise the SDK protocol independently from typed predicate admission."""
    from qualtran.bloqs.basic_gates.rotation import XPowGate

    register = Register("ctrl", dtype)
    width = int(register.bitsize)
    controls = np.asarray(cirq.LineQubit.range(width), dtype=object)
    target = np.asarray([cirq.LineQubit(width)], dtype=object)
    activation = np.empty((), dtype=object)
    activation.fill(value)
    values = (activation,)
    clause = export_control_values((register,), values)
    source_spec = q.CtrlSpec(source_dtype, values[0])
    assert clause == source_spec.to_cirq_cv()
    returned, decoded = import_control_values(clause, registers=(register,))
    assert returned == (register,)
    assert export_control_values(returned, decoded) == clause
    child = cirq.XPowGate(exponent=0.25).on(target[0])
    actual, outputs = control_register_operation(
        child,
        {"q": target},
        (register,),
        values,
        {"ctrl": controls},
    )
    original, original_outputs = (
        XPowGate(0.25)
        .controlled(source_spec)
        .as_cirq_op(cirq.SimpleQubitManager(), ctrl=controls, q=target)
    )
    assert tuple(outputs) == tuple(original_outputs)
    assert outputs["ctrl"] is controls and outputs["q"] is target
    assert (
        cirq.Circuit(actual).to_text_diagram()
        == cirq.Circuit(original).to_text_diagram()
    )
    np.testing.assert_allclose(
        cirq.unitary(actual), cirq.unitary(original), rtol=0, atol=2e-15
    )
    _assert_appended(actual, original)


@pytest.mark.parametrize("width,fraction", [(1, -2), (1, 0), (1, 1), (2, 1), (3, 3)])
def test_fixed_point_sdk_control_complete_stored_domain(
    width: int, fraction: int
) -> None:
    source = q.QFxp(width, fraction)
    for raw in range(1 << width):
        value = source.from_bits([(raw >> bit) & 1 for bit in reversed(range(width))])
        _check_foreign_control(
            fixed_point_register_type(width, fraction), source, value
        )


@pytest.mark.parametrize("prime,degree", [(2, 1), (2, 2), (3, 1), (5, 1), (3, 2)])
@pytest.mark.parametrize("polynomial", [False, True])
def test_field_sdk_control_complete_legal_domain(
    prime: int, degree: int, polynomial: bool
) -> None:
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
    from flagquantum.simulation.finite_field_polynomial_encoding import (
        FiniteFieldPolynomialEncoding,
    )

    field = FiniteFieldEncoding(prime, degree)
    encoding = FiniteFieldPolynomialEncoding(1, field) if polynomial else field
    source = q.QGFPoly(1, q.QGF(prime, degree)) if polynomial else q.QGF(prime, degree)
    for value in source.get_classical_domain():
        _check_foreign_control(encoding.register_type(), source, value)


def test_unresolved_symbolic_controls_retain_source_sdk_refusal() -> None:
    from sympy import Symbol

    from flagquantum.core.ir.program.register_types import unsigned_register_type

    width = Symbol("n", positive=True, integer=True)
    source = q.CtrlSpec(q.QUInt(width), 0)
    native = Register("ctrl", unsigned_register_type(width))
    with pytest.raises(ValueError, match="symbolic"):
        source.to_cirq_cv()
    with pytest.raises(ValueError, match="symbolic"):
        export_control_values((native,), (np.asarray(0),))


@pytest.mark.parametrize("family", [cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate])
@pytest.mark.parametrize("values", [(1, 1, 1), (0, 1, 0), (1, 0, 1, 0)])
@pytest.mark.parametrize("shift", [0, -0.5, 0.25])
def test_large_controlled_rotation_sdk_decomposition_keeps_complex_phase(
    family: Any, values: tuple[int, ...], shift: float
) -> None:
    qubits = cirq.LineQubit.range(len(values) + 1)
    operation = family(exponent=0.25, global_shift=shift).on(qubits[-1])
    controlled = operation.controlled_by(*qubits[:-1], control_values=values)
    _assert_appended(controlled, controlled)
