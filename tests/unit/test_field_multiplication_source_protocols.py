"""Observe complete multiplication factory protocols without source skip policy."""

import pytest

from tests.unit.source_example_protocols import source_example_protocols

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "name,expected",
    (
        ("_gf16_multiplication", ("PASS", "PASS", "FAIL", "FAIL", "PASS")),
        (
            "_gf2_multiplication_symbolic",
            ("PASS", "NA", "UNVERIFIED", "FAIL", "NA"),
        ),
        ("_gf2_multiply_by_constant", ("PASS", "PASS", "FAIL", "FAIL", "PASS")),
        ("_gf2_poly_multiply_by_constant", ("PASS", "PASS", "FAIL", "FAIL", "PASS")),
        ("_multiplypolybyoneplusxk", ("PASS", "PASS", "PASS", "FAIL", "PASS")),
        ("_binarypolynomialmultiplication", ("PASS", "PASS", "PASS", "FAIL", "PASS")),
        ("_gf2shiftleft", ("PASS", "PASS", "FAIL", "FAIL", "UNVERIFIED")),
        ("_gf2shiftright", ("PASS", "PASS", "FAIL", "FAIL", "UNVERIFIED")),
        ("_gf2mulviakaratsuba", ("PASS", "PASS", "FAIL", "FAIL", "PASS")),
    ),
)
def test_all_actual_factory_raw_protocols(
    name: str, expected: tuple[str, str, str, str, str]
) -> None:
    from qualtran import BloqExample
    from qualtran.bloqs.gf_arithmetic import gf2_multiplication

    example = getattr(gf2_multiplication, name)
    assert isinstance(example, BloqExample)
    observed = source_example_protocols(example)
    modes = ("make", "decompose", "counts", "serialize", "qtyping")
    assert tuple(observed[mode][0] for mode in modes) == expected
    if name in {"_multiplypolybyoneplusxk", "_binarypolynomialmultiplication"}:
        assert "DeSerialization Failed" in observed["serialize"][1]
        assert "Unable to find a Bloq" in observed["serialize"][1]
    else:
        assert "not recognized" in observed["serialize"][1]
    # A failed source count or codec must remain a diagnostic, rather than
    # becoming a positive merely because the source autotester skips it.
    for mode, (status, diagnostic) in observed.items():
        if status in {"FAIL", "UNVERIFIED"}:
            assert diagnostic, (name, mode)
