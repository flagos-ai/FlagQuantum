"""Native domain composition follows the unmodified pinned dtype methods."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_domains import register_classical_domain
from flagquantum.simulation.classical_truth_table import (
    classical_truth_table,
    format_classical_truth_table,
    register_classical_truth_table,
)
from flagquantum.simulation.integer_values import integer_word_domain
from tests.unit.test_classical_register_codec import KINDS, pair
from tests.unit.test_finite_field_metadata import CASES, field_pair

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def assert_domain_and_table(dtype: RegisterType, source: Any) -> None:
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.classical_sim import (
        format_classical_truth_table as source_format,
    )
    from qualtran.simulation.classical_sim import (
        get_classical_truth_table,
    )

    register = Register("x", dtype)
    actual = list(register_classical_domain(register))
    expected = list(source.get_classical_domain())
    np.testing.assert_array_equal(actual, expected)
    assert [type(x) for x in actual] == [type(x) for x in expected]
    source_bloq = SimpleNamespace(
        signature=SourceSignature([SourceRegister("x", source)]),
        call_classically=lambda **values: (values["x"],),
    )
    table = register_classical_truth_table(
        Signature((register,)), lambda values: (values["x"],)
    )
    reference = get_classical_truth_table(source_bloq)
    assert table == reference
    assert format_classical_truth_table(*table) == source_format(*reference)


@pytest.mark.parametrize("kind", KINDS)
def test_all_concrete_word_declarations(kind: str) -> None:
    dtype, source = pair(kind)
    if kind == "opaque":
        for action in (
            lambda: register_classical_domain(Register("x", dtype)),
            source.get_classical_domain,
        ):
            with pytest.raises(TypeError, match="[Aa]mbiguous"):
                action()
        return
    assert_domain_and_table(dtype, source)


@pytest.mark.parametrize("p,degree,mode", CASES)
@pytest.mark.parametrize("classical", [False, True])
def test_fields_keep_typed_elements_and_exact_domain(
    p: int, degree: int, mode: str, classical: bool
) -> None:
    field, source = field_pair(p, degree, mode, classical)
    assert_domain_and_table(field.register_type(classical=classical), source)


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("width", [0, 1, 2, 4])
def test_integer_source_domain_including_zero_width(width: int, signed: bool) -> None:
    from qualtran import QInt, QUInt

    source = (QInt if signed else QUInt)(width)
    if width == 0 and signed:
        with pytest.raises(ValueError):
            list(source.get_classical_domain())
        with pytest.raises(ValueError):
            integer_word_domain(width, signed=signed)
    else:
        assert list(integer_word_domain(width, signed=signed)) == list(
            source.get_classical_domain()
        )


@pytest.mark.parametrize("bound", [-1, 0, 1, 5])
def test_bound_is_not_repaired_to_capacity(bound: int) -> None:
    from qualtran import BQUInt

    from flagquantum.core.ir.program.register_types import (
        bounded_unsigned_register_type,
    )

    # Diagnostic metadata retains source bounds independently of executable admission.
    dtype = bounded_unsigned_register_type(3, 1)
    dtype = replace(dtype, parameters=(("bound", bound),))
    assert_domain_and_table(dtype, BQUInt(3, bound))


@pytest.mark.parametrize("modulus", [None, -1, 0, 1, 3])
def test_montgomery_domain_keeps_source_modulus_policy(modulus: int | None) -> None:
    from qualtran import QMontgomeryUInt

    from flagquantum.core.ir.program.register_types import montgomery_register_type

    dtype = montgomery_register_type(3, 3)
    parameters = () if modulus is None else (("modulus", modulus),)
    assert_domain_and_table(
        replace(dtype, parameters=parameters), QMontgomeryUInt(3, modulus)
    )


def test_shape_refusal_precedes_opaque_domain_and_custom_domain_remains_available() -> (
    None
):
    dtype, _ = pair("opaque")
    signature = Signature((Register("x", dtype), Register("array", dtype, (2,))))
    with pytest.raises(NotImplementedError):
        register_classical_truth_table(
            signature, lambda values: pytest.fail("evaluate")
        )
    scalar = Signature((Register("x", dtype),))
    outputs: list[object] = [object()]
    table = classical_truth_table(scalar, lambda r: [-4, 7], lambda v: outputs)
    assert [inputs for inputs, _ in table[2]] == [(-4,), (7,)]
    assert all(output is outputs for _, output in table[2])


def test_unknown_encoding_and_symbolic_metadata_fail_without_fallback() -> None:
    from sympy import Symbol

    dtype, _ = pair("uint")
    with pytest.raises(ValueError, match="no native classical domain"):
        register_classical_domain(Register("x", replace(dtype, encoding="unknown")))
    with pytest.raises(ValueError, match="concrete nonnegative"):
        register_classical_domain(
            Register("x", replace(dtype, quantum_bits=Symbol("n")))
        )


def test_domain_table_drives_existing_measured_addition_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.unit.test_classical_truth_table as consumer

    # Replace the manual unsigned callback behind the same physical consumer.
    def connected(signature: Signature, domain: Any, evaluate: Any) -> Any:
        return register_classical_truth_table(signature, evaluate)

    monkeypatch.setattr(consumer, "classical_truth_table", connected)
    consumer.test_table_evaluator_uses_native_graph_and_complete_measured_branches()
