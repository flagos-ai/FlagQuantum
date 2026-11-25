"""Whole duplicate source/test inventory and factory signature responsibility."""

from hashlib import sha1
from itertools import combinations_with_replacement
from pathlib import Path

import pytest

from flagquantum.compiler.registers.sorting.sorted_duplicates import (
    sorted_duplicates_signature,
)
from flagquantum.core.ir.program.register_types import (
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.simulation.classical_sorted_duplicates import (
    sorted_duplicates_classical_values,
)
from tests.unit.source_example_protocols import source_example_protocols

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "name,symbolic_length",
    (
        ("_has_duplicates", False),
        ("_has_duplicates_symb", False),
        ("_has_duplicates_symb_len", True),
    ),
)
def test_actual_factory_raw_protocols(name: str, symbolic_length: bool) -> None:
    from qualtran.bloqs.arithmetic.lists import has_duplicates

    observed = source_example_protocols(getattr(has_duplicates, name))
    assert {mode: status for mode, (status, _) in observed.items()} == {
        "make": "PASS",
        "decompose": "NA" if symbolic_length else "PASS",
        "counts": "UNVERIFIED" if symbolic_length else "PASS",
        "serialize": "FAIL" if symbolic_length else "PASS",
        "qtyping": "NA" if symbolic_length else "PASS",
    }


@pytest.mark.parametrize("length", (2, 3, 4))
@pytest.mark.parametrize("width,signed", ((2, False), (2, True), (3, False)))
def test_all_original_sorted_classical_domains(
    length: int, width: int, signed: bool
) -> None:
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates

    np = pytest.importorskip("numpy")
    dtype = QInt(width) if signed else QUInt(width)
    source = HasDuplicates(length, dtype)
    body = source.decompose_bloq()
    # The original sorts every Cartesian tuple before checking it. Each unique
    # sorted tuple below has precisely the same assertion inputs, without
    # redundantly repeating its permutations.
    for word in combinations_with_replacement(dtype.get_classical_domain(), length):
        values = np.asarray(word)
        for flag in (0, 1):
            expected = body.call_classically(xs=values.copy(), flag=flag)
            actual = sorted_duplicates_classical_values(
                {"xs": values.copy(), "flag": flag}
            )
            np.testing.assert_array_equal(actual["xs"], expected[0])
            assert actual["flag"] == expected[1]


def test_frozen_source_and_original_test_files() -> None:
    from qualtran.bloqs.arithmetic.lists import has_duplicates

    for name, blob in (
        ("has_duplicates.py", "4da8d2a6170108ec9507359b03b7c789969860f3"),
        ("has_duplicates_test.py", "0a02fa83af190c936f5e036cc7fb3c49833f3cd9"),
    ):
        raw = (Path(has_duplicates.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_all_factories_and_documented_subset() -> None:
    from qualtran.bloqs.arithmetic.lists.has_duplicates import (
        _HAS_DUPLICATES_DOC,
        _has_duplicates,
        _has_duplicates_symb,
        _has_duplicates_symb_len,
    )

    assert _HAS_DUPLICATES_DOC.examples == (_has_duplicates_symb, _has_duplicates)
    for example in (_has_duplicates, _has_duplicates_symb, _has_duplicates_symb_len):
        source = example.make()
        signature = sorted_duplicates_signature(
            source.l, unsigned_register_type(source.dtype.num_qubits)
        )
        assert [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in signature
        ] == [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in source.signature
        ]
        assert source.adjoint() is source


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("signed", [False, True])
def test_declared_symbolic_and_concrete_type_control_order(
    controlled: bool, signed: bool
) -> None:
    import sympy
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.lists import HasDuplicates

    length, width = sympy.symbols("l n", positive=True, integer=True)
    for count, bits in ((4, 3), (4, width), (length, width)):
        source_dtype = (QInt if signed else QUInt)(bits)
        native_dtype = (signed_register_type if signed else unsigned_register_type)(
            bits
        )
        source = HasDuplicates(count, source_dtype, controlled)
        signature = sorted_duplicates_signature(
            count, native_dtype, controlled=controlled
        )
        assert [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in signature
        ] == [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in source.signature
        ]
        assert signature.get_left("xs").dtype == native_dtype
        assert signature.get_left("flag").dtype.encoding == "bit"
        if controlled:
            assert signature.get_left("ctrl").dtype.encoding == "bit"
