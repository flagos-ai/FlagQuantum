"""Contract for what ``CircuitIR.dtype`` may name.

An IR describes amplitudes, so the dtype it declares is a complex one. That was the
intent but nobody owned the field: ``__post_init__`` stripped a ``torch.`` prefix and
refused an empty string, and the supported set was written out inline in four
readers that did not agree. ``resolve_dtype``, which serves ``run``, also accepted
``float32`` and ``float64``; target legalization and a hybrid session accepted only
the two complex names; and ``Circuit.from_ir`` accepted whatever ``torch`` happened
to have as an attribute.

So a decoded IR declaring ``"float32"`` was legal for a run -- which then silently
used ``complex64`` while the IR went on saying ``float32`` -- illegal for target
legalization, illegal for a hybrid session, and refused by ``from_ir`` with a
message about ``complex_dtype``. ``"banana"`` reached ``Unsupported dtype 'banana'.``
one layer down, or an ``AttributeError`` about ``torch`` from the other reader. None
of the four messages said what the field may hold, because nothing did.

``str(self.dtype)`` also laundered the type: ``3`` became the string ``'3'`` and
``None`` became ``'None'``, each a dtype name nothing recognizes, stored as though
the caller had named one. A wrong type is a ``TypeError`` -- what the errors-module
boundary reserves for it, and what the neighbouring count rules in the same file
already raise -- while a string that is merely not a supported name is an
``IRValidationError``, which is a ``ValidationError``.

The file imports only ``torch``, ``pytest`` and the standard library, for the reason
the neighbouring contract files give: a ``unit`` test is collected by every CPU lane,
including ``triton-optional``, which installs ``torch`` and not much else.
"""

from __future__ import annotations

import json

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR
from flagquantum.errors import ValidationError
from flagquantum.runtime.backend_registry import resolve_dtype

pytestmark = pytest.mark.unit

# What the field may name, and what each name means to the run path. Both are written
# out here rather than read from the module, so that narrowing the field's own set
# cannot make this file agree with itself.
SUPPORTED_DTYPES = ("complex64", "complex128")
RESOLVED_COMPLEX = {"complex64": torch.complex64, "complex128": torch.complex128}

# Names that are not amplitude dtypes, for four different reasons: a real dtype, a
# dtype of the wrong width or kind, a name that is not a dtype at all, and a name that
# is only whitespace.
UNSUPPORTED_DTYPES = (
    "float32",
    "float64",
    "float16",
    "bfloat16",
    "int32",
    "bool",
    "complex32",
    "Complex64",
    "FLOAT32",
    "banana",
    " ",
    "torch.banana",
)

# Values whose type is not a string at all.
NON_STRING_DTYPES = (3, 3.5, True, None, b"complex64", bytearray(b"complex64"))


def _payload(**overrides: object) -> dict[str, object]:
    """A minimal decodable IR payload, with fields replaced."""

    payload: dict[str, object] = {
        "kind": "flagquantum.circuit_ir",
        "version": "1.0",
        "n_wires": 1,
        "dtype": "complex64",
        "shape": (),
        "instructions": (),
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_an_ir_dtype_is_one_of_the_complex_dtypes(dtype: str) -> None:
    assert CircuitIR(1, (), dtype=dtype).dtype == dtype


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_a_torch_prefixed_dtype_names_the_same_dtype(dtype: str) -> None:
    # The prefix strip is a compatibility affordance and it stays; what changed is
    # that stripping is no longer the only thing the field is held to.
    assert CircuitIR(1, (), dtype=f"torch.{dtype}").dtype == dtype


@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test_an_ir_refuses_a_name_that_is_not_an_amplitude_dtype(dtype: str) -> None:
    with pytest.raises(ValidationError) as caught:
        CircuitIR(1, (), dtype=dtype)
    message = str(caught.value)
    # The message has to say what the field may hold, which is what none of the four
    # readers' messages did -- and it has to name the value it refused, so that a
    # caller is not left comparing its input against a list.
    assert "dtype" in message
    assert repr(dtype.removeprefix("torch.")) in message
    for supported in SUPPORTED_DTYPES:
        assert supported in message


@pytest.mark.parametrize("dtype", NON_STRING_DTYPES)
def test_an_ir_dtype_that_is_not_a_string_is_a_type_error(dtype: object) -> None:
    with pytest.raises(TypeError) as caught:
        CircuitIR(1, (), dtype=dtype)  # type: ignore[arg-type]
    assert not isinstance(caught.value, ValidationError)
    assert type(dtype).__name__ in str(caught.value)


def test_an_empty_dtype_still_says_that_it_is_empty() -> None:
    # The membership rule would refuse this too, but with a message that says only
    # that the name is unsupported. The specific message is better and is kept.
    for empty in ("", "torch."):
        with pytest.raises(ValidationError, match="cannot be empty"):
            CircuitIR(1, (), dtype=empty)


@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES + NON_STRING_DTYPES)
def test_the_decode_boundary_refuses_what_the_constructor_refuses(
    dtype: object,
) -> None:
    # ``from_dict`` used to stringify, so a document declaring a JSON number reached
    # the field as ``'3'`` and the two doors disagreed about the same mistake.
    with pytest.raises((TypeError, ValidationError)):
        CircuitIR.from_dict(_payload(dtype=dtype))


@pytest.mark.parametrize("dtype", NON_STRING_DTYPES)
def test_the_decode_boundary_keeps_the_type_error_a_type_error(dtype: object) -> None:
    with pytest.raises(TypeError) as caught:
        CircuitIR.from_dict(_payload(dtype=dtype))
    assert not isinstance(caught.value, ValidationError)


def test_a_payload_without_a_dtype_is_still_refused_as_empty() -> None:
    # An absent field is not a wrongly typed one: the default is the empty string, so
    # the caller is told the field is empty rather than that ``None`` is not a string.
    without_dtype = _payload()
    del without_dtype["dtype"]
    with pytest.raises(ValidationError, match="cannot be empty"):
        CircuitIR.from_dict(without_dtype)


def test_a_payload_with_an_explicit_null_dtype_is_a_type_error() -> None:
    # ``str(None)`` used to store the name ``'None'``, which no reader recognizes.
    with pytest.raises(TypeError) as caught:
        CircuitIR.from_dict(_payload(dtype=None))
    assert not isinstance(caught.value, ValidationError)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_a_supported_dtype_survives_a_json_round_trip(dtype: str) -> None:
    ir = CircuitIR(2, (), dtype=dtype)
    assert CircuitIR.from_json(ir.to_json()).dtype == dtype
    assert json.loads(ir.to_json())["dtype"] == dtype


def test_the_default_dtype_is_complex64() -> None:
    # The control: the rule narrows what may be written into the field and does not
    # move what the field defaults to.
    assert CircuitIR(1, ()).dtype == "complex64"


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_every_dtype_the_field_accepts_is_a_torch_complex_dtype(dtype: str) -> None:
    # ``Circuit.from_ir`` and the MPS executors reach the dtype through
    # ``getattr(torch, ir.dtype)``, so a name that is not a torch attribute -- or is
    # one that is not complex -- would be accepted by the field and fail there.
    ir = CircuitIR(1, (), dtype=dtype)
    resolved = getattr(torch, ir.dtype)
    assert resolved in {torch.complex64, torch.complex128}
    assert resolved == RESOLVED_COMPLEX[ir.dtype]


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_the_run_path_reads_the_declared_dtype_as_that_same_complex_dtype(
    dtype: str,
) -> None:
    # The measured harm: an IR declaring ``float32`` ran as ``complex64`` while
    # continuing to say ``float32``. For every dtype the field now accepts, the dtype
    # a run resolves is the one the IR declares.
    ir = CircuitIR(1, (), dtype=dtype)
    _real, complex_dtype = resolve_dtype(ir.dtype)
    assert complex_dtype == RESOLVED_COMPLEX[ir.dtype]
    assert complex_dtype == getattr(torch, ir.dtype)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_the_two_readers_that_narrow_the_dtype_accept_everything_it_allows(
    dtype: str,
) -> None:
    # Target legalization and a hybrid session each name ``{"complex64",
    # "complex128"}`` inline. They are re-checks at their own boundary, so they stay;
    # the point of the field's rule is that they are no longer *narrower* than it, and
    # therefore no longer a second, silent definition of what the field holds.
    ir = CircuitIR(1, (), dtype=dtype)
    assert ir.dtype in {"complex64", "complex128"}


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_a_circuit_and_its_ir_agree_about_the_dtype(dtype: str) -> None:
    # The producer side: whatever ``to_ir()`` writes has to be something the field
    # accepts, or a circuit would build an IR it could not decode.
    circuit = fq.Circuit(1, dtype=getattr(torch, dtype)).h(0)
    ir = circuit.to_ir()
    assert ir.dtype == dtype
    assert CircuitIR.from_dict(ir.to_dict()).dtype == dtype
    assert fq.Circuit.from_ir(ir).to_ir().dtype == dtype
