"""Numerical field metadata supports all concrete source projections independently of IR."""

from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest

from flagquantum.core.ir.program.register_verifier import executable_register_width
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.simulation.binary_field_encoding import resolve_binary_field_encoding
from flagquantum.simulation.classical_register_codec import (
    decode_register_bits,
    encode_register_bits,
)
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_metadata import resolve_finite_field_encoding

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = [
    (2, 1, "default"),
    (2, 3, "default"),
    (2, 3, "none"),
    (2, 3, "13"),
    (3, 1, "default"),
    (3, 2, "default"),
    (5, 1, "default"),
    (7, 1, "none"),
]


def field_pair(
    p: int, degree: int, mode: str, classical: bool
) -> tuple[FiniteFieldEncoding, Any]:
    import galois
    from qualtran import CGF, QGF

    constructor = CGF if classical else QGF
    if mode == "default":
        return FiniteFieldEncoding(p, degree), constructor(p, degree)
    poly = galois.Poly.Int(int(mode)) if mode.isdecimal() else None
    return FiniteFieldEncoding(p, degree, poly), constructor(p, degree, poly)


@pytest.mark.parametrize("p,degree,mode", CASES)
@pytest.mark.parametrize("classical", [False, True])
def test_complete_projection_field_class_domain_and_multiplication(
    p: int, degree: int, mode: str, classical: bool
) -> None:
    field, source = field_pair(p, degree, mode, classical)
    dtype = field.register_type(classical=classical)
    resolved = resolve_finite_field_encoding(dtype)
    assert resolved.register_type(classical=classical) == dtype
    assert "gf_type" not in resolved.__dict__
    assert resolved.gf_type is source.gf_type is field.gf_type
    np.testing.assert_array_equal(
        list(resolved.get_domain()), list(source.get_classical_domain())
    )
    elements = resolved.gf_type.elements
    np.testing.assert_array_equal(
        elements[:, None] * elements[None, :],
        source.gf_type.elements[:, None] * source.gf_type.elements[None, :],
    )
    if p == 2 and not classical:
        assert executable_register_width(dtype) == field.bitsize
        assert resolve_binary_field_encoding(dtype).register_type() == dtype
    else:
        with pytest.raises(ValueError):
            executable_register_width(dtype)
        with pytest.raises(ValueError):
            resolve_binary_field_encoding(dtype)


@pytest.mark.parametrize("p,degree,mode", CASES)
@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (0,)])
def test_all_small_binary_words_and_typed_output_errors(
    p: int, degree: int, mode: str, classical: bool, shape: tuple[int, ...]
) -> None:
    field, source = field_pair(p, degree, mode, classical)
    register = Register("x", field.register_type(classical=classical), shape)
    for word in product((0, 1), repeat=int(field.bitsize)):
        bits = np.array(word, dtype=np.uint8)
        if shape:
            bits = np.broadcast_to(bits, shape + (int(field.bitsize),))
        try:
            expected = (
                source.from_bits_array(bits)
                if shape
                else source.from_bits(list(bits.flat))
            )
        except ValueError as error:
            with pytest.raises(type(error)):
                decode_register_bits(register, bits)
            continue
        actual = decode_register_bits(register, bits)
        assert type(actual) is type(expected) is source.gf_type
        np.testing.assert_array_equal(actual, expected)
        assert isinstance(actual, np.ndarray)
        for value in (actual, np.asarray(actual)):
            try:
                encoded = source.to_bits_array(value)
            except ValueError as error:
                with pytest.raises(type(error)):
                    encode_register_bits(register, value)
            else:
                np.testing.assert_array_equal(
                    encode_register_bits(register, value), encoded
                )


@pytest.mark.parametrize("p,degree,mode", CASES)
@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("effect", [False, True])
def test_original_dense_type_loss_and_nonpower_of_two_domain_failures(
    p: int, degree: int, mode: str, classical: bool, effect: bool
) -> None:
    from qualtran import (
        Bloq,
        Side,
    )
    from qualtran import (
        Register as SourceRegister,
    )
    from qualtran import (
        Signature as SourceSignature,
    )
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
    )

    from flagquantum.core.ir.program.registers import RegisterSide
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.tensor_network.classical_action_tensor import (
        classical_action_tensor,
    )

    field, source_dtype = field_pair(p, degree, mode, classical)
    source_type: Any = type(
        "ProjectedFieldAction",
        (Bloq,),
        {
            "signature": SourceSignature(
                [
                    SourceRegister(
                        "x", source_dtype, side=Side.LEFT if effect else Side.THRU
                    )
                ]
            ),
            "on_classical_vals": lambda self, **v: {} if effect else v,
        },
    )
    source = source_type()
    signature = Signature(
        [
            Register(
                "x",
                field.register_type(classical=classical),
                side=RegisterSide.LEFT if effect else RegisterSide.THRU,
            )
        ]
    )
    try:
        expected = bloq_to_dense_via_classical_action(source)
    except ValueError as error:
        with pytest.raises(ValueError) as actual:
            classical_action_tensor(
                signature, source.call_classically, operation_name=str(source)
            )
        assert str(actual.value) == str(error)
        assert str(actual.value.__cause__) == str(error.__cause__)
    else:
        assert p == 2 and effect
        np.testing.assert_array_equal(
            classical_action_tensor(
                signature, source.call_classically, operation_name=str(source)
            ),
            expected,
        )


@pytest.mark.parametrize(
    "defect",
    [
        "encoding",
        "order",
        "degree",
        "characteristic",
        "width",
        "both_storage",
        "extra",
        "duplicate",
        "reordered",
        "symbolic",
        "partial",
        "bool",
    ],
)
def test_bad_projection_is_rejected_before_optional_import(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sympy

    import flagquantum.simulation.finite_field_metadata as owner

    dtype = FiniteFieldEncoding(3, 2).register_type()
    parameters = dtype.parameters
    changes = {
        "encoding": replace(dtype, encoding="uint"),
        "order": replace(
            dtype,
            parameters=tuple((k, 8 if k == "order" else v) for k, v in parameters),
        ),
        "degree": replace(
            dtype,
            parameters=tuple((k, 0 if k == "degree" else v) for k, v in parameters),
        ),
        "characteristic": replace(
            dtype,
            parameters=tuple(
                (k, 1 if k == "characteristic" else v) for k, v in parameters
            ),
        ),
        "width": replace(dtype, quantum_bits=3),
        "both_storage": replace(dtype, classical_bits=4),
        "extra": replace(dtype, parameters=parameters + (("unknown", 1),)),
        "duplicate": replace(dtype, parameters=parameters + (("order", 9),)),
        "reordered": replace(dtype, parameters=tuple(reversed(parameters))),
        "symbolic": replace(dtype, quantum_bits=sympy.Symbol("n")),
        "partial": replace(dtype, parameters=parameters[:3] + (("polynomial", 17),)),
        "bool": replace(
            dtype,
            parameters=tuple(
                (k, True if k == "characteristic" else v) for k, v in parameters
            ),
        ),
    }
    monkeypatch.setattr(
        owner, "import_module", lambda name: pytest.fail("optional import")
    )
    with pytest.raises(ValueError):
        resolve_finite_field_encoding(changes[defect])


def test_binary_admission_reuses_the_same_numerical_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation.binary_field_encoding as binary

    field = FiniteFieldEncoding(2, 3)
    dtype = field.register_type()
    seen: list[RegisterType] = []

    def resolve(declaration: RegisterType) -> FiniteFieldEncoding:
        seen.append(declaration)
        return field

    monkeypatch.setattr(binary, "resolve_finite_field_encoding", resolve)
    assert binary.resolve_binary_field_encoding(dtype) is field
    assert seen == [dtype]
