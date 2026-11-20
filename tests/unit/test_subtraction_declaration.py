"""General subtraction declarations retain original dtype and model contracts."""

from collections.abc import Hashable
from typing import Any

import pytest
import sympy

from flagquantum.compiler._component_trace.arithmetic_call import (
    ArithmeticCall,
    ArithmeticEncoding,
)
from flagquantum.compiler.registers.arithmetic.subtraction_component import (
    subtract_from_program,
    subtraction_program,
)
from flagquantum.compiler.registers.arithmetic.subtraction_declaration import (
    SubtractFromDeclaration,
    SubtractionDeclaration,
)
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    montgomery_register_type,
    opaque_register_type,
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import RegisterType
from tests.unit.test_subtraction_file_audit import _key

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def native_subtraction_declaration(
    source: Any,
) -> SubtractionDeclaration | SubtractFromDeclaration:
    """Translate source dtype objects at the optional reference boundary."""
    from qualtran import QInt, QMontgomeryUInt
    from qualtran.bloqs.arithmetic.subtraction import SubtractFrom

    def dtype(original: Any) -> RegisterType:
        if isinstance(original, QInt):
            return signed_register_type(original.bitsize)
        if isinstance(original, QMontgomeryUInt):
            return montgomery_register_type(original.bitsize, original.modulus)
        return unsigned_register_type(original.bitsize)

    return (
        SubtractFromDeclaration(dtype(source.dtype))
        if isinstance(source, SubtractFrom)
        else SubtractionDeclaration(dtype(source.a_dtype), dtype(source.b_dtype))
    )


def source_key(child: Any) -> Hashable:
    from qualtran import QInt, QMontgomeryUInt
    from qualtran.bloqs.arithmetic import Add, BitwiseNot

    if isinstance(child, (Add, BitwiseNot)):
        original = child.a_dtype if isinstance(child, Add) else child.dtype
        encoding: ArithmeticEncoding = (
            "twos_complement"
            if isinstance(original, QInt)
            else "montgomery" if isinstance(original, QMontgomeryUInt) else "unsigned"
        )
        if isinstance(child, Add):
            return ArithmeticCall(
                "add",
                (child.a_dtype.bitsize, child.b_dtype.bitsize),
                (encoding, encoding),
            )
        return ArithmeticCall("bitwise_not", (child.dtype.bitsize,), (encoding,))
    return _key(child)


@pytest.mark.parametrize("kind", ("unsigned", "signed", "montgomery"))
@pytest.mark.parametrize("left,right", ((1, 1), (2, 3), (3, 5), (2, 2)))
def test_dtype_defaults_projections_signatures_and_ordered_source_models(
    kind: str, left: int, right: int
) -> None:
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom

    constructor = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[
        kind
    ]
    for source in (
        Subtract(constructor(left)),
        Subtract(constructor(left), constructor(right)),
        SubtractFrom(constructor(right)),
    ):
        actual = native_subtraction_declaration(source)
        assert [(r.name, r.bitsize, r.side.name) for r in actual.signature] == [
            (r.name, r.bitsize, r.side.name) for r in source.signature
        ]
        assert list(actual.direct_calls().items()) == [
            (source_key(child), count) for child, count in source.bloq_counts().items()
        ]
        if isinstance(actual, SubtractionDeclaration):
            for name in ("a_dtype_as_unsigned", "b_dtype_as_unsigned"):
                projected = getattr(actual, name)
                original = getattr(source, name)
                assert projected.num_qubits == original.bitsize
                assert (projected.encoding == "montgomery") == isinstance(
                    original, QMontgomeryUInt
                )
            if source.a_dtype == source.b_dtype:
                assert actual.dtype is actual.a_dtype
            else:
                with pytest.raises(ValueError, match="same dtype"):
                    actual.dtype
                with pytest.raises(ValueError, match="same dtype"):
                    source.dtype


@pytest.mark.parametrize("adjacent", (False, True))
def test_montgomery_modulus_identity_and_source_default_object_retained(
    adjacent: bool,
) -> None:
    from qualtran import QMontgomeryUInt
    from qualtran.bloqs.arithmetic.subtraction import Subtract

    a = montgomery_register_type(4, 11)
    b = montgomery_register_type(4, 13) if adjacent else a
    native = SubtractionDeclaration(a, b)
    original = Subtract(
        QMontgomeryUInt(4, 11), QMontgomeryUInt(4, 13 if adjacent else 11)
    )
    assert native.a_dtype_as_unsigned is a and native.b_dtype_as_unsigned is b
    assert native.direct_calls() == {
        source_key(k): v for k, v in original.bloq_counts().items()
    }
    assert SubtractionDeclaration(a).b_dtype is a
    if adjacent:
        with pytest.raises(ValueError, match="same dtype"):
            native.dtype


@pytest.mark.parametrize("signed", (False, True))
@pytest.mark.parametrize("unequal", (False, True))
def test_symbolic_descriptors_keep_exact_source_direct_model_and_finite_refusal(
    signed: bool, unequal: bool
) -> None:
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom

    n = sympy.Symbol("n")
    dtype = QInt if signed else QUInt
    for source in (
        Subtract(dtype(n), dtype(n + 1 if unequal else n)),
        SubtractFrom(dtype(n)),
    ):
        actual = native_subtraction_declaration(source)
        assert actual.direct_calls() == {
            source_key(k): v for k, v in source.bloq_counts().items()
        }
        with pytest.raises(ValueError):
            actual.program()


@pytest.mark.parametrize(
    "dtype",
    (quantum_bit_type(), opaque_register_type(2), bounded_unsigned_register_type(3, 5)),
)
def test_original_type_family_rejection_is_preserved(dtype: RegisterType) -> None:
    from qualtran import BQUInt, QAny, QBit
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom

    source = {"bit": QBit(), "opaque": QAny(2), "uint": BQUInt(3, 5)}[dtype.encoding]
    for constructor in (Subtract, SubtractFrom):
        with pytest.raises(ValueError, match="Only QInt"):
            constructor(source)
    for constructor in (SubtractionDeclaration, SubtractFromDeclaration):
        with pytest.raises(ValueError, match="quantum signed"):
            constructor(dtype)


def test_finite_conversion_reuses_existing_owners_and_rejects_wider_left() -> None:
    a, b = signed_register_type(2), unsigned_register_type(3)
    assert SubtractionDeclaration(a, b).program() == subtraction_program(a, b)
    assert SubtractFromDeclaration(b).program() == subtract_from_program(b)
    with pytest.raises(ValueError, match="less than or equal"):
        SubtractionDeclaration(b, a)
