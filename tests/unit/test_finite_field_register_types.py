"""Explicit numerical-to-neutral field projection against pinned source types."""

import numpy as np
import pytest
import sympy

from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
OMITTED = object()
N = sympy.Symbol("n", integer=True, positive=True)
CASES = [
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
    (5, 1),
    (np.int64(2), np.int64(3)),
    (sympy.Integer(2), sympy.Integer(3)),
    (N, 2),
    (2, N),
    (N, N),
    (2.0, 2),
    (2, 2.0),
    (4, 2),
    (2, 0),
    (2, -1),
    (None, 2),
    (2, None),
]


def original(classical, characteristic, degree, polynomial=OMITTED):
    from qualtran import CGF, QGF

    constructor = CGF if classical else QGF
    if polynomial is OMITTED:
        return constructor(characteristic, degree)
    return constructor(characteristic, degree, polynomial)


def source_projection(classical, characteristic, degree, polynomial=OMITTED):
    source = original(classical, characteristic, degree, polynomial)
    width = source.bitsize
    parameters = (("storage", "classical"),) if classical else ()
    parameters += (
        ("characteristic", source.characteristic),
        ("degree", source.degree),
        ("order", source.order),
    )
    if source.irreducible_poly is not None:
        poly = source.irreducible_poly
        field = poly.field
        parameters += (
            ("polynomial", int(poly)),
            ("polynomial_field_characteristic", int(field.characteristic)),
            ("polynomial_field_degree", int(field.degree)),
            ("polynomial_field_primitive_element", int(field.primitive_element)),
            ("polynomial_field_modulus", int(field.irreducible_poly)),
        )
    return RegisterType(
        "finite_field", 0 if classical else width, width if classical else 0, parameters
    )


def native_projection(classical, characteristic, degree, polynomial=OMITTED):
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

    field = (
        FiniteFieldEncoding(characteristic, degree)
        if polynomial is OMITTED
        else FiniteFieldEncoding(characteristic, degree, polynomial)
    )
    return field.register_type(classical=classical)


@pytest.fixture(params=["source", "native"])
def project(request):
    return source_projection if request.param == "source" else native_projection


def outcome(call):
    try:
        value = call()
    except (
        ValueError,
        TypeError,
        RuntimeError,
        OverflowError,
        ZeroDivisionError,
    ) as error:
        return type(error), str(error)
    return (
        value.encoding,
        value.quantum_bits,
        type(value.quantum_bits),
        value.classical_bits,
        type(value.classical_bits),
        tuple(
            (name, parameter, type(parameter)) for name, parameter in value.parameters
        ),
    )


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("characteristic,degree", CASES)
@pytest.mark.parametrize("polynomial", [OMITTED, None, [3, 1, 0]])
def test_projection_values_types_and_failure_stage(
    project, classical, characteristic, degree, polynomial
):
    assert outcome(
        lambda: project(classical, characteristic, degree, polynomial)
    ) == outcome(
        lambda: source_projection(classical, characteristic, degree, polynomial)
    )


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("characteristic,degree", CASES[:11])
def test_shaped_signature_and_symbolic_classification(
    project, classical, characteristic, degree
):
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    source = original(classical, characteristic, degree)
    dtype = project(classical, characteristic, degree)
    assert dtype.is_symbolic() == source.is_symbolic()
    native = Signature((Register("field", dtype, (2, 3)),))
    reference = SourceSignature((SourceRegister("field", source, (2, 3)),))
    assert (native.n_bits(), native.n_qubits(), native.n_cbits()) == (
        reference.n_bits(),
        reference.n_qubits(),
        reference.n_cbits(),
    )
    assert all(
        type(parameter) is int
        for name, parameter in dtype.parameters
        if name.startswith("polynomial")
    )


@pytest.mark.parametrize("classical", [False, True])
def test_polynomial_and_coefficient_field_nominal_identity(project, classical):
    import galois

    fields = [
        galois.GF(2),
        galois.GF(3),
        galois.GF(7, primitive_element=3),
        galois.GF(7, primitive_element=5),
        galois.GF(2, 3, irreducible_poly=11),
        galois.GF(2, 3, irreducible_poly=13),
    ]
    polynomials = [
        None,
        *[galois.Poly.Int(3, field=field) for field in fields],
        galois.Poly.Int(5),
    ]
    # Degree one deliberately retains a polynomial ignored by gf_type.
    for first in polynomials:
        left = project(classical, 2, 1, first)
        source_left = original(classical, 2, 1, first)
        for second in polynomials:
            right = project(classical, 2, 1, second)
            if (first is None) != (second is None):
                with pytest.raises(TypeError, match="NoneType"):
                    _ = source_left == original(classical, 2, 1, second)
                assert left != right
            else:
                assert (left == right) == (
                    source_left == original(classical, 2, 1, second)
                )
    assert project(classical, 2, 3) != project(classical, 2, 3, None)
    assert project(classical, 2, 3) != project(not classical, 2, 3)


def test_projection_does_not_resolve_target_field_or_recompute_width():
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

    field = FiniteFieldEncoding(4, 2, None)
    assert field.register_type().num_bits == 4
    assert "gf_type" not in field.__dict__
    assert field.register_type().parameters == field.register_type().parameters
    assert "bitsize" in field.__dict__ and "order" in field.__dict__
