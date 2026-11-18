"""Complete locked list notebook and independent native append/model accounts."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import pytest

from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_reference_and_native_list_notebooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython.display
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import LinearDepthHalfLessThan
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt import MultiControlX
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.symbolics import HasLength

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent / "ftqc-reference-env/qualtran/bloqs/arithmetic/lists/lists.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "2251b288d0a1fe782e98e730058157c251bc5914"
    )
    original = json.loads(raw)
    native = json.loads((root / "examples/notebooks/register_lists.ipynb").read_text())
    assert len(original["cells"]) == 21 and len(native["cells"]) == 23
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in native["cells"][:21]
    ]
    reference = execute(original)
    displays: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise AssertionError(
                "The native list notebook must use owned implementations"
            )
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        result = execute(native)
    assert len(displays) == 8
    for native_name, source_name in (
        ("symmetric_signature", "symm_diff"),
        ("has_duplicates", "has_duplicates"),
        ("has_duplicates_symb", "has_duplicates_symb"),
    ):
        signature = result[native_name]
        source = reference[source_name]
        assert [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in signature
        ] == [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in source.signature
        ]
    assert {
        child.name: count
        for child, count in result["symmetric_graph"].leaf_counts.items()
    } == {
        str(child).replace("†", "Adjoint"): count
        for child, count in reference["symm_diff_sigma"].items()
    }
    comparison = LinearDepthHalfLessThan(QUInt(reference["n"]))
    assert reference["has_duplicates_symb_sigma"] == {
        comparison: 3,
        comparison.adjoint(): 3,
        MultiControlX(HasLength(3)): 1,
        XGate(): 1,
    }
    assert result["duplicate_graph"].leaf_counts == {
        "Half less than": 3,
        "Half less than adjoint": 3,
        "Three-control X": 1,
        "X": 1,
    }
    source_model = reference["symm_diff"]
    assert get_cost_value(source_model, QECGatesCost()).cswap == 96
    assert (
        get_cost_value(source_model, QubitCount())
        == estimate_component_resources(result["symm_diff"]).peak_logical_qubits
        == 55
    )
    counts = result["measured_primitives"]
    assert counts["t"] + counts["tdg"] == 6044
    assert counts["cx"] == 14953
    assert counts["measure"] == counts["reset"] == 797
    assert result["work_width"] == 113
