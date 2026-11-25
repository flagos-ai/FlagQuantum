"""Every frozen comparison definition has an explicit native disposition."""

import ast
from hashlib import sha1
from pathlib import Path

import pytest

from tests.unit.source_example_protocols import source_example_protocols

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_frozen_comparison_definition_disposition() -> None:
    from qualtran.bloqs.arithmetic import comparison

    raw = Path(comparison.__file__).read_bytes()
    audit = (
        Path(__file__).parents[2] / "docs/development/COMPARISON_FILE_AUDIT.md"
    ).read_text()
    digest = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    assert digest in audit
    assert "bcb17bb25360ed68ce3bc084645db3e929799389" in audit
    count = 0
    for node in ast.parse(raw).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert f"`{node.name}`" in audit
            count += 1
            for method in node.body if isinstance(node, ast.ClassDef) else ():
                if isinstance(method, ast.FunctionDef):
                    assert f"`{node.name}.{method.name}`" in audit
                    count += 1
    assert count == 115


@pytest.mark.parametrize(
    "name,decompose,counts,qtyping",
    (
        ("_lt_k", "PASS", "FAIL", "PASS"),
        ("_lt_k_symb", "NA", "UNVERIFIED", "NA"),
        ("_bi_qubits_mixer", "PASS", "FAIL", "PASS"),
        ("_sq_cmp", "PASS", "FAIL", "PASS"),
        ("_leq_symb", "NA", "UNVERIFIED", "NA"),
        ("_leq", "PASS", "FAIL", "PASS"),
        ("_greater_than", "PASS", "PASS", "PASS"),
        ("_gt_k", "MISSING", "UNVERIFIED", "NA"),
        ("_equals", "PASS", "FAIL", "PASS"),
        ("_eq_k", "PASS", "FAIL", "UNVERIFIED"),
        ("_clineardepthgreaterthan_example", "PASS", "PASS", "UNVERIFIED"),
        ("_lineardepthhalfgreaterthan_small", "PASS", "FAIL", "UNVERIFIED"),
        ("_lineardepthhalflessthan_small", "PASS", "FAIL", "UNVERIFIED"),
        ("_lineardepthhalfgreaterthanequal_small", "PASS", "FAIL", "UNVERIFIED"),
        ("_lineardepthhalflessthanequal_small", "PASS", "FAIL", "UNVERIFIED"),
    ),
)
def test_all_actual_factory_raw_protocols(
    name: str, decompose: str, counts: str, qtyping: str
) -> None:
    from qualtran import BloqExample
    from qualtran.bloqs.arithmetic import comparison

    example = getattr(comparison, name)
    assert isinstance(example, BloqExample)
    observed = source_example_protocols(example)
    assert {mode: status for mode, (status, _) in observed.items()} == {
        "make": "PASS",
        "decompose": decompose,
        "counts": counts,
        "serialize": "PASS",
        "qtyping": qtyping,
    }
    if counts == "FAIL":
        assert "does not have equivalent bloq counts" in observed["counts"][1]
    if qtyping == "UNVERIFIED":
        assert "Strict type checking failed" in observed["qtyping"][1]
        assert "QDTypes are incompatible" in observed["qtyping"][1]
    if name == "_sq_cmp":
        # The original declaration also loses a control value. This failure is
        # not entirely explained by omitting zero-cost bookkeeping nodes.
        assert "And(cv1=0" in observed["counts"][1]


def test_complete_original_test_inventory_and_frozen_blob() -> None:
    from qualtran.bloqs.arithmetic import comparison

    path = Path(comparison.__file__).with_name("comparison_test.py")
    raw = path.read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "9876537e8199b0e6af0ca7bf9f595f94304146d6"
    )
    functions = {
        node.name for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)
    }
    assert len(functions) == 39
    assert functions - {name for name in functions if name.startswith("test_")} == {
        "identity_map",
        "_less_than_equal_expected_t_complexity",
    }
    audit = (
        Path(__file__).parents[2] / "docs/development/COMPARISON_FILE_AUDIT.md"
    ).read_text()
    for name in functions:
        assert f"`{name}`" in audit
