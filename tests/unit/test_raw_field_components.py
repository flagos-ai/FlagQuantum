"""Raw nonbinary source circuits retain word/degree differences and exact phases."""

from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.field_power_decomposition import field_power_program
from examples.algorithms.field_product_definitions import field_product_catalog_program
from flagquantum import Circuit
from flagquantum.compiler.registers.finite_field.constant_multiplication_component import (
    constant_matrix_decomposition_program,
    constant_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.inverse_decomposition import (
    field_inverse_degree_one_program,
)
from flagquantum.compiler.registers.finite_field.monomial_component import (
    monomial_decomposition_program,
    monomial_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.multiplication_component import (
    field_multiplication_decomposition_program,
    field_multiplication_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_multiplication import field_scale_matrix
from tests.unit.field_product_checks import bind_product, product_source_stream

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
FIELDS = ((3, 1), (3, 2), (5, 1))


def check_raw_execution(source: Any, program: ProgramRecord) -> None:
    bound, ports = bind_product(program)
    original_gates, returned = product_source_stream(source, bound, ports)
    assert bound.circuit.instructions == original_gates
    assert bound.outputs == tuple(returned.values())
    assert bound.classical_outputs == ()
    assert all(e.kind == "allocate" for e in bound.events)
    inputs = tuple(w for word in ports.values() for w in word)
    outputs = tuple(w for word in bound.outputs for w in word)
    count = max((*inputs, *outputs)) + 3
    values = torch.arange(1 << len(inputs), dtype=torch.int64)
    starts = torch.zeros_like(values)
    for i, wire in enumerate(inputs):
        starts |= ((values >> (len(inputs) - 1 - i)) & 1) << (count - 1 - wire)
    starts = starts.repeat_interleave(2)
    starts |= torch.arange(starts.numel(), dtype=torch.int64).remainder(2) << 1
    amplitude = torch.randn(
        starts.numel(),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(125),
    )
    amplitude /= amplitude.norm()
    destinations = starts.clone()
    for gate in original_gates:
        assert gate.name in {"cx", "ccx"}
        condition = torch.ones_like(destinations)
        for wire in gate.wires[:-1]:
            condition &= (destinations >> (count - 1 - wire)) & 1
        destinations ^= condition << (count - 1 - gate.wires[-1])
    state, expected = (
        torch.zeros(1 << count, dtype=torch.complex128) for _ in range(2)
    )
    state[starts], expected[destinations] = amplitude, amplitude
    operation = FieldOperation(count, bound.circuit.instructions)
    circuit = Circuit(count, inputs=state, dtype=torch.complex128).append(operation)
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=2e-12, rtol=0)
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    # Independently contract the complete original operator when small. Larger
    # raw accumulators use the exhaustive untouched-source permutation above.
    if len(outputs) <= 9:
        matrix = torch.as_tensor(
            source.decompose_bloq().flatten().tensor_contract(), dtype=torch.complex128
        )
        logical = matrix @ amplitude.reshape(1 << len(inputs), 2)
        values = torch.arange(1 << len(outputs), dtype=torch.int64)
        mapped = torch.zeros_like(values)
        for i, wire in enumerate(outputs):
            mapped |= ((values >> (len(outputs) - 1 - i)) & 1) << (count - 1 - wire)
        expected_tensor = torch.zeros_like(expected)
        expected_tensor[
            mapped.repeat_interleave(2)
            | torch.arange(logical.numel()).remainder(2) << 1
        ] = logical.flatten()
        torch.testing.assert_close(
            circuit.state().flatten(), expected_tensor, atol=2e-12, rtol=0
        )
    inverse = Circuit(count).append(operation).adjoint().to_ir()
    torch.testing.assert_close(
        circuit.compose(inverse).state().flatten(), state, atol=2e-12, rtol=0
    )


@pytest.mark.parametrize(
    "characteristic,degree,constant",
    tuple((p, d, k) for p, d in FIELDS for k in range(p**d)),
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_raw_constant_all_encodings_source_matrix_and_declared_direction(
    characteristic: int, degree: int, constant: int, adjoint: bool
) -> None:
    from qualtran._infra.data_types import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulK

    encoding = FiniteFieldEncoding(characteristic, degree)
    source: Any = GF2MulK(QGF(characteristic, degree), constant)
    matrix = tuple(
        tuple(int(bit) for bit in row) for row in field_scale_matrix(encoding, constant)
    )
    assert matrix == tuple(
        tuple(int(bit) for bit in row) for row in source.reduction_matrix_q
    )
    if adjoint:
        source = source.adjoint()
    program = constant_matrix_decomposition_program(
        encoding.register_type(), matrix, adjoint=adjoint
    )
    check_raw_execution(source, program)
    if not adjoint:
        assert (
            field_product_catalog_program(
                "GF2MulK",
                Signature(program.functions[0].registers or ()),
                {"const": constant},
            )
            == program
        )
    with pytest.raises(ValueError):
        constant_multiplication_program(
            encoding.register_type(), 1, modulus=int(encoding.gf_type.irreducible_poly)
        )


@pytest.mark.parametrize("characteristic,degree", FIELDS)
@pytest.mark.parametrize("count", (-1, 0, 1))
@pytest.mark.parametrize("left", (False, True))
def test_raw_shift_degree_not_word_width(
    characteristic: int, degree: int, count: int, left: bool
) -> None:
    from qualtran._infra.data_types import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2ShiftLeft,
        GF2ShiftRight,
    )

    encoding = FiniteFieldEncoding(characteristic, degree)
    modulus = int(encoding.gf_type.irreducible_poly)
    source = (GF2ShiftLeft if left else GF2ShiftRight)(
        QGF(characteristic, degree), count
    )
    program = monomial_decomposition_program(
        encoding.register_type(), count, modulus=modulus, adjoint=left
    )
    check_raw_execution(source, program)
    assert (
        field_product_catalog_program(
            "GF2ShiftLeft" if left else "GF2ShiftRight",
            Signature(program.functions[0].registers or ()),
            {"k": count},
        )
        == program
    )
    if degree == 1:
        bound, _ = bind_product(program)
        assert bound.circuit.instructions == ()
        assert bound.outputs == (tuple(reversed(range(int(encoding.bitsize)))),)
    with pytest.raises(ValueError):
        monomial_multiplication_program(encoding.register_type(), 1, modulus=modulus)


@pytest.mark.parametrize("characteristic,degree", FIELDS)
@pytest.mark.parametrize("accumulate", (False, True))
def test_raw_quadratic_all_inputs_and_arbitrary_targets(
    characteristic: int, degree: int, accumulate: bool
) -> None:
    from qualtran._infra.data_types import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2Multiplication
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    encoding = FiniteFieldEncoding(characteristic, degree)
    modulus = int(encoding.gf_type.irreducible_poly)
    source = GF2Multiplication(QGF(characteristic, degree), plus_equal_prod=accumulate)
    program = field_multiplication_decomposition_program(
        encoding.register_type(), modulus=modulus, plus_equal_prod=accumulate
    )
    check_raw_execution(source, program)
    bound, _ = bind_product(program)
    model = get_cost_value(source, QECGatesCost())
    assert (
        sum(g.name == "ccx" for g in bound.circuit.instructions)
        == model.toffoli
        == int(encoding.bitsize) ** 2
    )
    assert sum(g.name == "cx" for g in bound.circuit.instructions) == model.clifford
    assert (
        field_product_catalog_program(
            "GF2Multiplication",
            Signature(program.functions[0].registers or ()),
            {"plus_equal_prod": accumulate},
        )
        == program
    )
    with pytest.raises(ValueError):
        field_multiplication_program(encoding.register_type(), modulus=modulus)


@pytest.mark.parametrize("characteristic", (2, 3, 5))
def test_degree_one_inverse_original_raw_branch(characteristic: int) -> None:
    from qualtran._infra.data_types import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse

    encoding = FiniteFieldEncoding(characteristic, 1)
    program = field_inverse_degree_one_program(encoding.register_type())
    check_raw_execution(GF2Inverse(QGF(characteristic, 1)), program)
    assert (
        field_power_program(
            "GF2Inverse", Signature(program.functions[0].registers or ()), {}
        )
        == program
    )
    assert tuple(r.name for r in program.functions[0].registers or ()) == (
        "x",
        "result",
    )


@pytest.mark.parametrize("characteristic,degree", FIELDS)
def test_raw_accumulation_original_generic_adjoint(
    characteristic: int, degree: int
) -> None:
    from qualtran._infra.data_types import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2Multiplication

    encoding = FiniteFieldEncoding(characteristic, degree)
    source = GF2Multiplication(
        QGF(characteristic, degree), plus_equal_prod=True
    ).adjoint()
    program = field_multiplication_decomposition_program(
        encoding.register_type(),
        modulus=int(encoding.gf_type.irreducible_poly),
        plus_equal_prod=True,
        adjoint=True,
    )
    check_raw_execution(source, program)


def test_native_raw_compilation_and_append_do_not_load_source_sdks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    import flagquantum.compiler.registers.finite_field.field_multiplication as old

    encoding = FiniteFieldEncoding(3, 1)
    dtype = encoding.register_type()
    matrix = tuple(
        tuple(int(bit) for bit in row) for row in field_scale_matrix(encoding, 2)
    )
    modulus = int(encoding.gf_type.irreducible_poly)
    original = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "galois", "kickmix"}:
            raise AssertionError("Native raw execution must not import reference SDKs")
        return original(name, *args, **kwargs)

    def replaced(*args: object, **kwargs: object) -> None:
        raise AssertionError("The old public product emitter must not execute")

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.setattr(old, "gf2_multiplication", replaced)
    programs = (
        constant_matrix_decomposition_program(dtype, matrix),
        monomial_decomposition_program(dtype, 0, modulus=modulus),
        field_multiplication_decomposition_program(dtype, modulus=modulus),
        field_inverse_degree_one_program(dtype),
    )
    for program in programs:
        bound, ports = bind_product(program)
        count = max(w for word in (*ports.values(), *bound.outputs) for w in word) + 1
        Circuit(count, dtype=torch.complex128).append(
            FieldOperation(count, bound.circuit.instructions)
        ).state()


@pytest.mark.parametrize("characteristic,degree", FIELDS)
def test_typed_left_quadratic_inverse_releases_matching_source_result(
    characteristic: int, degree: int
) -> None:
    from qualtran._infra.data_types import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2Multiplication

    from flagquantum.compiler.registers.register_binding import bind_register_program

    encoding = FiniteFieldEncoding(characteristic, degree)
    dtype, modulus = encoding.register_type(), int(encoding.gf_type.irreducible_poly)
    width = int(encoding.bitsize)
    forward, _ = bind_product(
        field_multiplication_decomposition_program(dtype, modulus=modulus)
    )
    inverse = bind_register_program(
        field_multiplication_decomposition_program(
            dtype, modulus=modulus, adjoint=True
        ),
        forward.outputs,
        input_ownership=tuple((wire, False) for wire in forward.outputs[2]),
    )
    ports = dict(zip(("x", "y", "result"), forward.outputs, strict=True))
    source = GF2Multiplication(QGF(characteristic, degree)).adjoint()
    gates, returned = product_source_stream(source, inverse, ports)
    assert inverse.circuit.instructions == gates
    assert inverse.outputs == tuple(returned.values()) == forward.outputs[:2]
    assert len(inverse.events) == 1 and inverse.events[0].kind == "free"
    count = 3 * width + 2
    starts = (
        torch.arange(1 << (2 * width), dtype=torch.int64) << (width + 2)
    ).repeat_interleave(2)
    starts |= torch.arange(starts.numel()).remainder(2) << 1
    amplitudes = torch.randn(
        starts.numel(),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(1251),
    )
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << count, dtype=torch.complex128)
    state[starts] = amplitudes
    circuit = Circuit(count, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(count, forward.circuit.instructions))
    circuit.append(FieldOperation(count, inverse.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), state, atol=2e-12, rtol=0)
