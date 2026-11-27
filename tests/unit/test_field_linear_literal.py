"""Source zero-pivot synthesis and signed iteration counts remain literal."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.finite_field.constant_multiplication_component import (
    constant_multiplication_decomposition_program,
    constant_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.field_multiplication import (
    FieldDecomposition,
    bind_field_outputs,
)
from flagquantum.compiler.registers.finite_field.linear_component import (
    binary_linear_decomposition_program,
    binary_linear_program,
)
from flagquantum.compiler.registers.finite_field.linear_resource_model import (
    source_linear_decomposition_resource_formula,
)
from flagquantum.compiler.registers.finite_field.monomial_component import (
    monomial_decomposition_program,
    monomial_multiplication_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core._binary_matrix import BinaryMatrix
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_binary_linear_component import SourceCalls
from tests.unit.test_field_frobenius_component import dtype
from tests.unit.test_field_linear import _source_type
from tests.unit.test_field_multiplication import assert_gate_streams, reference_gates

np = pytest.importorskip("numpy")
pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3))
def test_every_small_matrix_factors_source_gates_aliases_and_literal_model(
    width: int,
) -> None:
    from galois import GF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import SynthesizeLRCircuit

    for flat in product((0, 1), repeat=width * width):
        matrix = tuple(tuple(flat[r * width : (r + 1) * width]) for r in range(width))
        value = BinaryMatrix.from_dense([list(row) for row in matrix])
        factors = value.plu_with_zero_pivots()
        assert factors is not None
        expected = GF(2)(matrix).plu_decompose()
        for actual, original in zip(factors, expected, strict=True):
            np.testing.assert_array_equal(actual.dense(), original)
        assert factors[0].multiply(factors[1]).multiply(factors[2]) == value
        singular = any(
            not (row >> i & 1) for i, row in enumerate(factors[2].packed_rows)
        )
        assert (value.plu() is None) == singular
        for adjoint in (False, True):
            source: Any = SynthesizeLRCircuit(np.asarray(matrix), is_adjoint=adjoint)
            observer = SourceCalls()
            returned = source.build_composite_bloq(observer, q=np.arange(width))
            program = binary_linear_decomposition_program(matrix, adjoint=adjoint)
            bound = bind_register_program(
                program,
                tuple((i,) for i in range(width)),
                entry=program.functions[0].name,
            )
            assert [
                (g.name, g.wires) for g in bound.circuit.instructions
            ] == observer.gates
            assert bound.outputs == tuple((int(i),) for i in returned["q"])
            declared = source.build_call_graph(None)
            assert source_linear_decomposition_resource_formula(matrix).clifford == sum(
                declared.values()
            )
            assert bound.events == () and bound.classical_outputs == ()
        if singular:
            with pytest.raises(ValueError):
                binary_linear_program(matrix)


@pytest.mark.parametrize(
    "matrix", (((0, 0), (0, 0)), ((1, 1), (0, 0)), ((0, 0, 1), (1, 0, 1), (1, 0, 1)))
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_singular_source_full_complex_operator_and_correct_native_inverse(
    matrix: tuple[tuple[int, ...], ...], adjoint: bool
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import SynthesizeLRCircuit

    width = len(matrix)
    source = SynthesizeLRCircuit(np.asarray(matrix), is_adjoint=adjoint)
    program = binary_linear_decomposition_program(matrix, adjoint=adjoint)
    bound = bind_register_program(
        program, tuple((i,) for i in range(width)), entry=program.functions[0].name
    )
    fixed = bind_field_outputs(
        FieldDecomposition(bound.circuit.instructions, bound.outputs, ()),
        tuple((i,) for i in range(width)),
    )
    original = torch.as_tensor(
        source.decompose_bloq().flatten().tensor_contract(), dtype=torch.complex128
    )
    actual = get_unitary(
        Circuit(width, dtype=torch.complex128).append(FieldOperation(width, fixed))
    )
    torch.testing.assert_close(actual, original, rtol=0, atol=2e-12)
    initial = torch.zeros(1 << (width + 2), dtype=torch.complex128)
    initial[::2] = torch.randn(
        1 << (width + 1),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(5190),
    )
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    expected[::2] = (original @ initial[::2].reshape(1 << width, 2)).flatten()
    operation = FieldOperation(width + 2, fixed)
    circuit = Circuit(width + 2, inputs=initial, dtype=torch.complex128).append(
        operation
    )
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    inverse = Circuit(width + 2).append(operation).adjoint().to_ir()
    torch.testing.assert_close(
        circuit.compose(inverse).state().flatten(), initial, rtol=0, atol=2e-12
    )
    # Missing source diagonals can produce a negative declared count, while
    # its reversible physical circuit has nonnegative primitive inventory.
    model = source_linear_decomposition_resource_formula(matrix)
    if not any(any(row) for row in matrix):
        assert model.clifford == -width
        assert Counter(g.name for g in bound.circuit.instructions)["cx"] == 0


@pytest.mark.parametrize("width,modulus", ((1, 3), (2, 7), (3, 11)))
@pytest.mark.parametrize("adjoint", (False, True))
def test_all_source_constants_including_zero_and_declared_directions(
    width: int, modulus: int, adjoint: bool
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulK

    for constant in range(1 << width):
        source: Any = GF2MulK(_source_type(width, modulus), constant)
        if adjoint:
            source = source.adjoint()
        program = constant_multiplication_decomposition_program(
            dtype(width, modulus), constant, modulus=modulus, adjoint=adjoint
        )
        bound = bind_register_program(program, (tuple(range(width)),))
        gates, outputs = reference_gates(
            source, {"g": tuple(range(width))}, (), repair_adjoint=False
        )
        assert_gate_streams(
            [(g.name, g.wires) for g in bound.circuit.instructions], gates
        )
        assert bound.outputs == (outputs["g"],)
        fixed = bind_field_outputs(
            FieldDecomposition(bound.circuit.instructions, bound.outputs, ()),
            (tuple(range(width)),),
        )
        actual = get_unitary(
            Circuit(width, dtype=torch.complex128).append(FieldOperation(width, fixed))
        )
        original = torch.as_tensor(
            source.decompose_bloq().flatten().tensor_contract(), dtype=torch.complex128
        )
        torch.testing.assert_close(actual, original, atol=2e-12, rtol=0)
    with pytest.raises(ValueError):
        constant_multiplication_program(dtype(width, modulus), 0, modulus=modulus)


@pytest.mark.parametrize("count", (-7, -1, 0, 1, 3))
@pytest.mark.parametrize("left", (False, True))
def test_source_signed_iteration_count_is_not_signed_mathematical_power(
    count: int, left: bool
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2ShiftLeft,
        GF2ShiftRight,
    )

    width, modulus = 3, 11
    source = (GF2ShiftLeft if left else GF2ShiftRight)(
        _source_type(width, modulus), count
    )
    program = monomial_decomposition_program(
        dtype(width, modulus), count, modulus=modulus, adjoint=left
    )
    bound = bind_register_program(program, ((0, 1, 2),))
    gates, outputs = reference_gates(source, {"f": (0, 1, 2)}, (), repair_adjoint=False)
    assert_gate_streams([(g.name, g.wires) for g in bound.circuit.instructions], gates)
    assert bound.outputs == (outputs["f"],)
    fixed = bind_field_outputs(
        FieldDecomposition(bound.circuit.instructions, bound.outputs, ()), ((0, 1, 2),)
    )
    actual = get_unitary(
        Circuit(width, dtype=torch.complex128).append(FieldOperation(width, fixed))
    )
    original = torch.as_tensor(
        source.decompose_bloq().flatten().tensor_contract(), dtype=torch.complex128
    )
    torch.testing.assert_close(actual, original, atol=2e-12, rtol=0)
    if count < 0:
        assert bound.circuit.instructions == () and bound.outputs == ((0, 1, 2),)
        mathematical = bind_register_program(
            monomial_multiplication_program(
                dtype(width, modulus), count, modulus=modulus
            ),
            ((0, 1, 2),),
        )
        assert (
            mathematical.circuit.instructions or mathematical.outputs != bound.outputs
        )


@pytest.mark.parametrize(
    "operation,parameters",
    (
        ("GF2MulK", {"const": 0}),
        ("GF2ShiftRight", {"k": -2}),
        ("GF2ShiftLeft", {"k": -2}),
    ),
)
def test_catalog_constructor_selects_literal_zero_and_negative_source_paths(
    operation: str, parameters: dict[str, int]
) -> None:
    from examples.algorithms.field_product_definitions import (
        field_product_catalog_program,
    )
    from flagquantum.core.ir.program.signatures import Signature

    expected = (
        constant_multiplication_decomposition_program(dtype(3, 11), 0, modulus=11)
        if operation == "GF2MulK"
        else monomial_decomposition_program(
            dtype(3, 11), -2, modulus=11, adjoint=operation == "GF2ShiftLeft"
        )
    )
    ports = Signature(expected.functions[0].registers or ())
    assert field_product_catalog_program(operation, ports, parameters) == expected
