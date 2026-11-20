"""Stored PLU and Frobenius calls preserve source gates and coherent inverses."""

from collections import Counter
from dataclasses import replace

import pytest
import torch

from examples.algorithms.register_field_frobenius import FieldFrobenius
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.finite_field.field_linear import gf2_frobenius
from flagquantum.compiler.registers.finite_field.frobenius_component import (
    field_frobenius_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.registers import RegisterType
from tests.unit.test_binary_linear_component import SourceCalls
from tests.unit.test_field_linear import _source_type
from tests.unit.test_field_multiplication import reference_gates

np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
FIELDS = ((1, 3), (2, 7), (3, 11), (4, 19), (5, 37), (8, 283))


def dtype(width: int, modulus: int) -> RegisterType:
    return RegisterType(
        "finite_field",
        width,
        parameters=(
            ("characteristic", 2),
            ("degree", width),
            ("order", 1 << width),
            ("polynomial", modulus),
            ("polynomial_field_characteristic", 2),
            ("polynomial_field_degree", 1),
            ("polynomial_field_primitive_element", 1),
            ("polynomial_field_modulus", 3),
        ),
    )


@pytest.mark.parametrize("width,modulus", FIELDS)
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("adjoint", [False, True])
def test_stored_graph_matches_existing_mathematical_owner(
    width: int, modulus: int, count: int, adjoint: bool
) -> None:
    wires = tuple(range(width))
    program = field_frobenius_program(
        dtype(width, modulus), count, modulus=modulus, adjoint=adjoint
    )
    bound = bind_register_program(program, (wires,))
    plan = gf2_frobenius(wires, count, modulus=modulus, adjoint=adjoint)
    assert bound.circuit.instructions == plan.instructions
    assert bound.outputs == plan.outputs
    assert bound.events == () and bound.classical_outputs == ()
    root = next(f for f in program.functions if f.name == "main")
    # split, one actual linear child, join retain separate stored definitions.
    assert len(root.blocks[0].operations) == 3
    assert set(Counter(g.name for g in bound.circuit.instructions)) <= {"cx"}


@pytest.mark.parametrize("width,modulus", FIELDS)
@pytest.mark.parametrize("count", [1, 2, 4])
def test_source_forward_stream_aliases_and_independent_model(
    width: int, modulus: int, count: int
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = GF2Square(_source_type(width, modulus), k=count)
    bound = FieldFrobenius(
        dtype(width, modulus), tuple(range(width)), modulus, count
    ).bind()
    gates, outputs = reference_gates(source, {"x": tuple(range(width))}, ())
    reader = SourceCalls()
    original_outputs = source.build_composite_bloq(reader, x=np.array(range(width)))
    assert [(g.name, g.wires) for g in bound.circuit.instructions] == reader.gates
    assert tuple(original_outputs["x"]) == outputs["x"]
    assert bound.outputs == (outputs["x"],)
    assert len(gates) == get_cost_value(source, QECGatesCost()).clifford


@pytest.mark.parametrize("width,modulus", FIELDS[:4])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_normal_append_full_complex_spectator_and_restoration(
    width: int, modulus: int, count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import flagquantum.compiler.registers.finite_field.field_linear as physical

    def replaced(*args: object, **kwargs: object) -> None:
        raise AssertionError("The append path must use the stored component")

    monkeypatch.setattr(physical, "gf2_frobenius", replaced)
    operation = FieldFrobenius(
        dtype(width, modulus), tuple(range(width)), modulus, count
    )
    bound = operation.bind()
    size = 1 << (width + 2)
    state = torch.zeros(size, dtype=torch.complex128)
    values = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    state[::2] = torch.complex(values, torch.flip(values, (0,)))
    state /= torch.linalg.vector_norm(state)
    expected = torch.zeros_like(state)
    field = _source_type(width, modulus).gf_type
    for value in range(1 << width):
        output = int(field(value) ** (2**count))
        physical_value = sum(
            ((output >> (width - 1 - i)) & 1) << (width + 1 - wire)
            for i, wire in enumerate(bound.outputs[0])
        )
        for spectator in (0, 1):
            expected[physical_value | (spectator << 1)] = state[
                (value << 2) | (spectator << 1)
            ]
    circuit = Circuit(width + 2, inputs=state, dtype=torch.complex128).append(operation)
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=1e-12, rtol=0)
    inverse = replace(operation, target=bound.outputs[0], adjoint=True)
    restored = circuit.append(inverse).state().flatten()
    torch.testing.assert_close(restored, state, atol=1e-12, rtol=0)
    assert torch.count_nonzero(restored[1::2]) == 0


def test_invalid_field_contracts() -> None:
    valid = dtype(3, 11)
    for wrong in [
        replace(valid, encoding="unsigned"),
        replace(valid, parameters=(("characteristic", 3),)),
        dtype(3, 13),
    ]:
        with pytest.raises(ValueError):
            field_frobenius_program(wrong, modulus=11)
    for count in [-1, True]:
        with pytest.raises(ValueError):
            field_frobenius_program(valid, count, modulus=11)
    with pytest.raises(ValueError):
        field_frobenius_program(valid, modulus=11, adjoint=1)  # type: ignore[arg-type]
