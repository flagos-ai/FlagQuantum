"""Complete polynomial addition graphs reuse typed fields and packed views."""

from dataclasses import replace

import pytest
import torch

from examples.algorithms.register_field_polynomial_addition import (
    FieldPolynomialAddition,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.finite_field.addition_component import (
    field_addition_program,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_addition import (
    gf2_polynomial_add_constant_ir,
    gf2_polynomial_add_ir,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)
from flagquantum.compiler.registers.finite_field.polynomial_addition_component import (
    field_polynomial_addition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.registers import RegisterType

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def field(width: int) -> RegisterType:
    return RegisterType(
        "finite_field",
        width,
        parameters=(("characteristic", 2), ("degree", width), ("order", 1 << width)),
    )


@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("degree", [0, 1, 4])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("constant", [False, True])
def test_complete_stream_and_stored_children(
    width: int, degree: int, adjoint: bool, constant: bool
) -> None:
    layout = FieldPolynomialLayout(degree, width)
    size = layout.logical_width
    source = tuple(reversed(range(size)))
    target = tuple(range(size, 2 * size))
    constants = ((1 << width) - 1,) if constant else None
    inputs = (target,) if constant else (source, target)
    program = field_polynomial_addition_program(
        layout, field(width), constants=constants, adjoint=adjoint
    )
    bound = bind_register_program(program, inputs)
    expected = (
        gf2_polynomial_add_constant_ir(target, constants, layout, adjoint=adjoint)
        if constants is not None
        else gf2_polynomial_add_ir(source, target, layout, adjoint=adjoint)
    )
    assert bound.circuit.instructions == expected
    assert bound.outputs == inputs
    assert bound.events == () and bound.classical_outputs == ()
    root = next(f for f in program.functions if f.name == "main")
    assert len(root.blocks[0].operations) == (2 if constant else 4) + degree + 1


@pytest.mark.parametrize("constant", [None, (0,), (3,), (1, 2)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_full_complex_append_spectator_clean_wire_and_restore(
    constant: tuple[int, ...] | None, adjoint: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import flagquantum.compiler.registers.finite_field.field_polynomial_addition as old

    def replaced(*args: object, **kwargs: object) -> None:
        raise AssertionError("Append must bind shared stored coefficient calls")

    monkeypatch.setattr(old, "gf2_polynomial_add_ir", replaced)
    monkeypatch.setattr(old, "gf2_polynomial_add_constant_ir", replaced)
    layout = FieldPolynomialLayout(1, 2)
    size = layout.logical_width
    data_width = size if constant is not None else 2 * size
    inputs = (
        (tuple(range(size)),)
        if constant is not None
        else (tuple(range(size)), tuple(range(size, 2 * size)))
    )
    operation = FieldPolynomialAddition(layout, field(2), inputs, constant, adjoint)
    state = torch.zeros(1 << (data_width + 2), dtype=torch.complex128)
    values = torch.arange(1, (1 << (data_width + 1)) + 1, dtype=torch.float64)
    state[::2] = torch.complex(values, torch.flip(values, (0,)))
    state /= torch.linalg.vector_norm(state)
    expected = torch.zeros_like(state)
    encoded_constant = 0
    for coefficient in constant or ():
        encoded_constant = (encoded_constant << 2) | coefficient
    for value in range(1 << data_width):
        output = (
            value ^ encoded_constant
            if constant is not None
            else value ^ (value >> size)
        )
        for spectator in (0, 1):
            expected[(output << 2) | (spectator << 1)] = state[
                (value << 2) | (spectator << 1)
            ]
    circuit = Circuit(data_width + 2, inputs=state, dtype=torch.complex128).append(
        operation
    )
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=1e-12, rtol=0)
    restored = circuit.append(replace(operation, adjoint=not adjoint)).state().flatten()
    torch.testing.assert_close(restored, state, atol=1e-12, rtol=0)
    assert torch.count_nonzero(restored[1::2]) == 0


def test_constant_child_exact_adjoint_order_and_contracts() -> None:
    inputs = ((0, 1, 2),)
    forward = bind_register_program(
        field_addition_program(field(3), constant=7), inputs
    )
    inverse = bind_register_program(
        field_addition_program(field(3), constant=7, adjoint=True), inputs
    )
    assert inverse.circuit.instructions == tuple(reversed(forward.circuit.instructions))
    for constants in [(), (4,), (1, 0, 0)]:
        with pytest.raises(ValueError):
            field_polynomial_addition_program(
                FieldPolynomialLayout(1, 2), field(2), constants=constants
            )
    with pytest.raises(ValueError):
        field_polynomial_addition_program(FieldPolynomialLayout(1, 2), field(3))
    with pytest.raises(ValueError):
        field_polynomial_addition_program(FieldPolynomialLayout(1, 2), field(2), adjoint=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        field_addition_program(field(2), adjoint=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("width,degree", [(1, 0), (2, 1), (3, 4), (5, 1), (8, 1)])
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
def test_unmodified_source_operands_feedback_and_resource_model(
    width: int, degree: int, constant: bool, adjoint: bool
) -> None:
    from galois import Poly
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic import GF2PolyAdd, GF2PolyAddK
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from tests.unit.test_field_multiplication import assert_gate_streams

    cirq = pytest.importorskip("cirq")
    source_type = QGFPoly(degree, QGF(2, width))
    layout = FieldPolynomialLayout(degree, width)
    size = layout.logical_width
    constants = ((1 << width) - 1,) if constant else None
    source = (
        GF2PolyAddK(
            source_type, Poly([(1 << width) - 1], field=source_type.qgf.gf_type)
        )
        if constant
        else GF2PolyAdd(source_type)
    )
    if adjoint:
        source = source.adjoint()
    inputs = (
        (tuple(range(size)),)
        if constant
        else (tuple(range(size)), tuple(range(size, 2 * size)))
    )
    bound = FieldPolynomialAddition(
        layout, field(width), inputs, constants, adjoint
    ).bind()
    np = pytest.importorskip("numpy")
    port_inputs = {
        name: np.array([cirq.LineQubit(w) for w in word])
        for name, word in zip(("f_x", "g_x")[: len(inputs)], inputs, strict=True)
    }
    circuit, outputs = (
        source.decompose_bloq().flatten().to_cirq_circuit_and_quregs(**port_inputs)
    )
    names = {cirq.X: "x", cirq.CNOT: "cx"}
    expected = [
        (names[op.gate], tuple(q.x for q in op.qubits))
        for op in circuit.all_operations()
    ]
    assert_gate_streams(
        [(g.name, g.wires) for g in bound.circuit.instructions], expected
    )
    assert (
        tuple(tuple(q.x for q in outputs[name].flat) for name in port_inputs)
        == bound.outputs
    )
    assert len(expected) == get_cost_value(source, QECGatesCost()).clifford
