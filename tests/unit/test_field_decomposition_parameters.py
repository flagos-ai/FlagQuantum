"""Numerical integer aliases project to the existing canonical quantum vocabulary."""

from dataclasses import replace

import pytest
import sympy

from flagquantum.core.ir.program.register_verifier import (
    executable_classical_register_width,
    executable_register_width,
)
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_metadata import resolve_quantum_field_storage

np = pytest.importorskip("numpy")
pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ("bool", "numpy_signed", "numpy_unsigned"))
def test_quantum_projection_canonicalizes_integral_aliases_without_mutating_encoding(
    kind: str,
) -> None:
    from qualtran._infra.data_types import QGF

    value = (
        True
        if kind == "bool"
        else np.int64(3) if kind == "numpy_signed" else np.uint64(3)
    )
    encoding = FiniteFieldEncoding(2, value)
    literal = encoding.register_type()
    assert dict(literal.parameters)["degree"] is value
    native = resolve_quantum_field_storage(literal)
    assert encoding.degree is value
    parameters = dict(native.parameters)
    assert type(parameters["degree"]) is int
    assert parameters["degree"] == int(value)
    assert type(parameters["order"]) is int
    assert executable_register_width(native) == QGF(2, value).num_qubits
    classical = encoding.register_type(classical=True)
    assert dict(classical.parameters)["degree"] is value
    with pytest.raises(ValueError):
        executable_classical_register_width(classical)
    with pytest.raises(ValueError):
        resolve_quantum_field_storage(classical)
    # Explicit malformed Core metadata remains rejected; the projection does
    # not teach Core a new boolean/integer-alias parameter kind.
    raw = replace(
        native,
        parameters=tuple(
            (name, value if name == "degree" else item)
            for name, item in native.parameters
        ),
    )
    with pytest.raises(ValueError):
        executable_register_width(raw)


def test_symbolic_quantum_projection_keeps_expression_identity() -> None:
    degree = sympy.Symbol("m", integer=True, positive=True)
    encoding = FiniteFieldEncoding(2, degree)
    descriptor = encoding.register_type()
    assert encoding.degree is degree
    assert dict(descriptor.parameters)["degree"] is degree
    with pytest.raises(ValueError):
        executable_register_width(descriptor)


@pytest.mark.parametrize("family", ("addition", "constant", "square"))
@pytest.mark.parametrize("kind", ("bool", "numpy_signed", "numpy_unsigned"))
def test_projected_alias_reaches_original_complete_quantum_decomposition(
    family: str, kind: str
) -> None:
    import torch
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.data_types import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_add_k import GF2AddK
    from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from examples.algorithms.field_addition_decomposition import (
        field_addition_catalog_program,
    )
    from examples.algorithms.field_power_decomposition import field_power_program
    from flagquantum import Circuit
    from flagquantum.compiler.registers.finite_field.field_multiplication import (
        FieldDecomposition,
        bind_field_outputs,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.unitary import get_unitary

    degree = (
        True
        if kind == "bool"
        else np.int64(3) if kind == "numpy_signed" else np.uint64(3)
    )
    encoding = FiniteFieldEncoding(2, degree)
    descriptor = encoding.register_type()
    width = executable_register_width(resolve_quantum_field_storage(descriptor))
    source: Bloq
    if family == "square":
        source = GF2Square(QGF(2, degree))
        record = field_power_program(
            "GF2Square",
            Signature.build_from_dtypes(x=descriptor),
            {"k": 1, "uncompute": False},
        )
    elif family == "addition":
        source = GF2Addition(QGF(2, degree))
        record = field_addition_catalog_program(
            "GF2Addition", Signature.build_from_dtypes(x=descriptor, y=descriptor), {}
        )
    else:
        source = GF2AddK(degree, 1)
        record = field_addition_catalog_program(
            "GF2AddK", Signature.build_from_dtypes(x=descriptor), {"k": 1}
        )
    assert record is not None
    inputs = (
        (tuple(range(width)), tuple(range(width, 2 * width)))
        if family == "addition"
        else (tuple(range(width)),)
    )
    bound = bind_register_program(record, inputs)
    gates = bind_field_outputs(
        FieldDecomposition(bound.circuit.instructions, bound.outputs, ()), inputs
    )
    n = width * len(inputs)
    actual = get_unitary(
        Circuit(n, dtype=torch.complex128).append(FieldOperation(n, gates))
    )
    expected = torch.as_tensor(
        source.decompose_bloq().flatten().tensor_contract(), dtype=torch.complex128
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
    assert bound.events == () and bound.classical_outputs == ()
