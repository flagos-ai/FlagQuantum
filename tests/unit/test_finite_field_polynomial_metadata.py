"""Projected polynomial declarations consume existing numerical owners."""

from dataclasses import replace
from functools import partial
from itertools import product
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from flagquantum.core.ir.program.register_verifier import executable_register_width
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_register_codec import (
    decode_register_bits,
    encode_register_bits,
)
from flagquantum.simulation.finite_field_polynomial_metadata import (
    resolve_finite_field_polynomial_encoding,
)
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor,
)
from tests.unit.test_classical_domains import assert_domain_and_table
from tests.unit.test_classical_typed_values import compare
from tests.unit.test_finite_field_polynomial_encoding import CASES, pair, same_call

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("p,m,degree", CASES)
@pytest.mark.parametrize("classical", [False, True])
def test_projection_domains_truth_table_and_independent_core_refusal(
    p: int, m: int, degree: int, classical: bool
) -> None:
    encoding, source = pair(p, m, degree, classical)
    dtype = encoding.register_type(classical=classical)
    resolved = resolve_finite_field_polynomial_encoding(dtype)
    assert resolved.register_type(classical=classical) == dtype
    assert resolved.field.gf_type is encoding.field.gf_type is source.qgf.gf_type
    assert_domain_and_table(dtype, source)
    assert (dtype.num_qubits, dtype.num_cbits) == (source.num_qubits, source.num_cbits)
    with pytest.raises(ValueError, match="unsupported executable"):
        executable_register_width(dtype)


@pytest.mark.parametrize("p,m,degree", CASES)
@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_all_binary_words_default_codecs_and_binding(
    p: int, m: int, degree: int, classical: bool, shape: tuple[int, ...]
) -> None:
    encoding, source = pair(p, m, degree, classical)
    dtype = encoding.register_type(classical=classical)
    register = Register("x", dtype, shape)
    for word in product((0, 1), repeat=int(encoding.bitsize)):
        bits = np.array(word, dtype=np.uint8)
        if shape:
            bits = np.broadcast_to(bits, shape + (len(word),))
        same_call(
            partial(decode_register_bits, register, bits),
            (
                partial(source.from_bits_array, bits)
                if shape
                else partial(source.from_bits, list(bits.flat))
            ),
        )
    for polynomial in [
        next(iter(encoding.get_domain())),
        list(encoding.get_domain())[-1],
    ]:
        values = np.empty(shape, dtype=object)
        values[...] = polynomial
        same_call(
            partial(encode_register_bits, register, values),
            partial(source.to_bits_array, values),
        )
        compare(dtype, source, shape, values if shape else polynomial)
    invalid = np.ones(shape, dtype=int) if shape else 1
    compare(dtype, source, shape, invalid)


@pytest.mark.parametrize("p,m,degree", CASES)
@pytest.mark.parametrize("classical", [False, True])
def test_dense_default_codec_keeps_source_data_or_original_failure(
    p: int, m: int, degree: int, classical: bool
) -> None:
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        _bloq_to_dense_via_classical_action,
    )

    encoding, source = pair(p, m, degree, classical)
    signature = Signature((Register("x", encoding.register_type(classical=classical)),))
    original = SimpleNamespace(
        signature=SourceSignature([SourceRegister("x", source)]),
        call_classically=lambda **values: (values["x"],),
    )
    try:
        expected = _bloq_to_dense_via_classical_action(original)
    except ValueError as error:
        with pytest.raises(ValueError) as caught:
            classical_action_tensor(
                signature, lambda **values: (values["x"],), operation_name="poly"
            )
        assert caught.value.__cause__ is not None
        assert str(caught.value.__cause__) == str(error)
    else:
        actual = classical_action_tensor(
            signature, lambda **values: (values["x"],), operation_name="poly"
        )
        np.testing.assert_array_equal(actual, expected.reshape(actual.shape))


@pytest.mark.parametrize("mode", ["none", "13"])
@pytest.mark.parametrize("classical", [False, True])
def test_coefficient_polynomial_identity_is_not_reconstructed_by_degree_only(
    mode: str, classical: bool
) -> None:
    from flagquantum.simulation.finite_field_polynomial_encoding import (
        FiniteFieldPolynomialEncoding,
    )
    from tests.unit.test_finite_field_metadata import field_pair

    field, source_field = field_pair(2, 3, mode, False)
    encoding = FiniteFieldPolynomialEncoding(1, field)
    dtype = encoding.register_type(classical=classical)
    restored = resolve_finite_field_polynomial_encoding(dtype)
    assert restored.field.register_type() == field.register_type()
    assert restored.field.gf_type is source_field.gf_type
    assert restored.register_type(classical=classical) == dtype


@pytest.mark.parametrize(
    "case",
    [
        "kind",
        "negative_degree",
        "symbolic_degree",
        "missing_degree",
        "wrong_degree_key",
        "fractional_width",
        "other_storage",
        "negative_width",
        "field_order",
        "field_width",
        "field_key",
    ],
)
def test_malformed_metadata_is_refused_before_optional_import(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sympy

    import flagquantum.simulation.finite_field_metadata as fields

    encoding, _ = pair(2, 2, 1, False)
    dtype = encoding.register_type()
    entries = dtype.parameters
    variants: dict[str, Any] = {
        "kind": {"encoding": "uint"},
        "negative_degree": {"parameters": (("polynomial_degree", -1),) + entries[1:]},
        "symbolic_degree": {
            "parameters": (("polynomial_degree", sympy.Symbol("n")),) + entries[1:]
        },
        "missing_degree": {"parameters": ()},
        "wrong_degree_key": {"parameters": (("degree", 1),) + entries[1:]},
        "fractional_width": {"quantum_bits": 4.5},
        "other_storage": {"classical_bits": 1},
        "negative_width": {"quantum_bits": -4},
        "field_order": {
            "parameters": tuple((k, 5 if k == "order" else v) for k, v in entries)
        },
        "field_width": {"quantum_bits": 6},
        "field_key": {"parameters": entries + (("unknown", 1),)},
    }
    monkeypatch.setattr(
        fields, "import_module", lambda name: pytest.fail("optional import")
    )
    with pytest.raises(ValueError):
        resolve_finite_field_polynomial_encoding(replace(dtype, **variants[case]))


def test_physical_consumer_replaces_manual_codec_with_metadata_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.unit.test_finite_field_polynomial_encoding as consumer

    original = consumer.pair

    def resolved(p: int, m: int, degree: int, classical: bool) -> Any:
        encoding, source = original(p, m, degree, classical)
        return (
            resolve_finite_field_polynomial_encoding(
                encoding.register_type(classical=classical)
            ),
            source,
        )

    monkeypatch.setattr(consumer, "pair", resolved)
    consumer.test_existing_polynomial_add_consumer_uses_native_coefficients()
