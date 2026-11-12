"""Symbolic metadata reaches existing owners without relaxing bit codec admission."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import sympy

from flagquantum.core.ir.program import register_types as types
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.simulation.classical_domains import register_classical_domain
from flagquantum.simulation.classical_register_codec import decode_register_bits
from flagquantum.simulation.classical_typed_values import (
    collect_typed_register_value,
)
from tests.unit.test_classical_typed_values import compare

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
KINDS = ["opaque", "uint", "signed", "bounded", "ones", "fxp", "mont"]


def pair(kind: str, mode: str, classical: bool) -> tuple[RegisterType, Any]:
    import qualtran as q

    n, b = sympy.symbols("n b", integer=True, positive=True)
    width = 3 if mode == "parameter" else n
    bound = b if mode == "parameter" else 3
    fractional = b if mode == "parameter" else 1
    native: dict[str, Any] = {
        "opaque": lambda: types.opaque_register_type(width),
        "uint": lambda: types.unsigned_register_type(width),
        "signed": lambda: types.signed_register_type(width),
        "bounded": lambda: types.bounded_unsigned_register_type(width, bound),
        "ones": lambda: types.ones_complement_register_type(width),
        "fxp": lambda: types.fixed_point_register_type(width, fractional, True),
        "mont": lambda: types.montgomery_register_type(width, bound),
    }
    source: dict[str, Any] = {
        "opaque": lambda: q.QAny(width),
        "uint": lambda: (q.CUInt if classical else q.QUInt)(width),
        "signed": lambda: (q.CInt if classical else q.QInt)(width),
        "bounded": lambda: (q.BCUInt if classical else q.BQUInt)(width, bound),
        "ones": lambda: (q.CIntOnesComp if classical else q.QIntOnesComp)(width),
        "fxp": lambda: (q.CFxp if classical else q.QFxp)(width, fractional, True),
        "mont": lambda: (q.CMontgomeryUInt if classical else q.QMontgomeryUInt)(
            width, bound
        ),
    }
    dtype = native[kind]()
    if classical and kind != "opaque":
        dtype = replace(
            dtype,
            quantum_bits=0,
            classical_bits=width,
            parameters=dtype.parameters + (("storage", "classical"),),
        )
    return dtype, source[kind]()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", ["width", "parameter"])
@pytest.mark.parametrize("classical", [False, True])
def test_symbolic_scalar_identity_and_original_numeric_validation(
    kind: str, mode: str, classical: bool
) -> None:
    dtype, source = pair(kind, mode, classical)
    for value in [sympy.Symbol("x"), sympy.Integer(1), 0, 1, -1, 0.5]:
        compare(dtype, source, (), value)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("shape", [(2,), (0,)])
def test_symbolic_width_arrays_keep_source_success_or_failure(
    kind: str, classical: bool, shape: tuple[int, ...]
) -> None:
    dtype, source = pair(kind, "width", classical)
    compare(dtype, source, shape, np.ones(shape, dtype=int))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", ["width", "parameter"])
@pytest.mark.parametrize("classical", [False, True])
def test_symbolic_domains_match_source_without_numerical_substitution(
    kind: str, mode: str, classical: bool
) -> None:
    dtype, source = pair(kind, mode, classical)
    try:
        expected = list(source.get_classical_domain())
    except (ValueError, TypeError) as error:
        with pytest.raises(type(error)):
            list(register_classical_domain(Register("x", dtype)))
    else:
        assert list(register_classical_domain(Register("x", dtype))) == expected


def test_symbolic_codec_still_refuses_and_unknown_metadata_is_not_accepted() -> None:
    from flagquantum.simulation.classical_register_metadata import (
        value_register_metadata,
    )

    dtype, _ = pair("uint", "width", False)
    with pytest.raises(ValueError, match="concrete nonnegative"):
        decode_register_bits(Register("x", dtype), np.array([0], dtype=np.uint8))
    for width in [-1, 2.0]:
        with pytest.raises(ValueError, match="nonnegative or symbolic"):
            value_register_metadata(Register("x", replace(dtype, quantum_bits=width)))
    for parameters in [(("bound", 0.5),), (("unknown", sympy.Symbol("x")),)]:
        with pytest.raises(ValueError, match="integer or symbolic"):
            value_register_metadata(
                Register("x", replace(dtype, parameters=parameters))
            )


def test_symbolic_scalar_collection_does_not_choose_array_storage() -> None:
    for kind in KINDS:
        dtype, _ = pair(kind, "width", False)
        value = object()
        assert collect_typed_register_value(Register("x", dtype), {(): value}) is value


def test_physical_consumer_and_clean_work_remain_unchanged() -> None:
    from tests.unit.test_classical_truth_table import (
        test_table_evaluator_uses_native_graph_and_complete_measured_branches,
    )

    test_table_evaluator_uses_native_graph_and_complete_measured_branches()
