"""Close the atomic polynomial views against actual source tensor connections."""

from typing import Any

import pytest

from flagquantum.compiler.registers.finite_field.field_polynomial_component import (
    field_polynomial_component_program,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)
from flagquantum.core.ir.program.verifier import verify_program_record
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from tests.unit.source_example_protocols import source_example_protocols

np = pytest.importorskip("numpy")
pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("reverse", (False, True))
def test_actual_factory_protocols_preserve_atomic_and_codec_boundaries(
    reverse: bool,
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import (
        _gf_poly_join,
        _gf_poly_split,
    )

    example = _gf_poly_join if reverse else _gf_poly_split
    observed = source_example_protocols(example)
    assert {mode: status for mode, (status, _) in observed.items()} == {
        "make": "PASS",
        "decompose": "NA",
        "counts": "MISSING",
        "serialize": "FAIL",
        "qtyping": "NA",
    }
    rejected_type = "QGF" if reverse else "QGFPoly"
    assert f"data_types.{rejected_type}'" in observed["serialize"][1]
    with pytest.raises(DecomposeTypeError):
        example.make().decompose_bloq()


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(
    "characteristic,field_degree,degree", ((2, 3, 4), (3, 1, 2), (5, 1, 1), (3, 2, 0))
)
def test_complete_tensor_factors_match_typed_coefficient_ports(
    characteristic: int, field_degree: int, degree: int, reverse: bool
) -> None:
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import GFPolySplit
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.simulation.tensor import cbloq_to_quimb

    field = FiniteFieldEncoding(characteristic, field_degree)
    polynomial = FiniteFieldPolynomialEncoding(degree, field)
    width = int(field.bitsize)
    total = (degree + 1) * width
    source: Any = GFPolySplit(QGFPoly(degree, QGF(characteristic, field_degree)))
    if reverse:
        source = source.adjoint()
    native = field_polynomial_component_program(
        FieldPolynomialLayout(degree, width),
        field.register_type(),
        reverse=reverse,
        packed_dtype=polynomial.register_type(),
    )
    assert not verify_program_record(native)
    ports = native.functions[0].registers
    assert ports is not None
    assert tuple((p.name, p.shape, p.bitsize, p.side.name) for p in ports) == tuple(
        (p.name, p.shape, p.bitsize, p.side.name) for p in source.signature
    )
    assert source.adjoint().adjoint() == source
    network = cbloq_to_quimb(source.as_composite_bloq())
    assert len(network.tensors) == total
    expected = set()
    for bit in range(total):
        packed = ("reg", (), bit, "r" if reverse else "l")
        coefficient = (
            "reg",
            (bit // width,),
            bit % width,
            "l" if reverse else "r",
        )
        expected.add((packed, coefficient) if reverse else (coefficient, packed))
    assert {tensor.inds for tensor in network.tensors} == expected
    for tensor in network.tensors:
        np.testing.assert_array_equal(tensor.data, np.eye(2))
    # Prove the full operator as a product of exact identity factors. The
    # default 15-bit factory would otherwise allocate a 32768-square matrix.
    assert get_cost_value(source, QubitCount()) == total
    model = get_cost_value(source, QECGatesCost())
    assert model.total_t_count() == 0
    assert model.clifford == 0
    assert model.rotation == 0
