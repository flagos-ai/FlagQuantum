"""Complete controlled-adder notebook migration keeps source failure evidence."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import pytest

from tests.unit.test_addition_file_audit import execute
from tests.unit.test_controlled_addition_resource_width import source_key

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_source_and_native_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    import IPython.display
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.resource_counting.generalizers import ignore_split_join

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/arithmetic/controlled_addition.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "d4235893521075c7a25d1a00f586dd39e2cd86f2"
    )
    original = json.loads(raw)
    native = json.loads(
        (root / "examples/notebooks/register_controlled_addition.ipynb").read_text()
    )
    assert len(original["cells"]) == 11 and len(native["cells"]) == 13
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in native["cells"][:11]
    ]
    displays: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    reference = execute(original)
    assert reference["cadd_small"] == CAdd(QUInt(3))
    assert reference["cadd_large"] == CAdd(QUInt(1000))
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise AssertionError("The native notebook must use owned implementations")
        return original_import(name, globals, locals, fromlist, level)

    displays.clear()
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        result = execute(native)
    assert len(displays) == 5
    for name in ("cadd_small", "cadd_large"):
        call, signature = result[name]
        source = reference[name]
        assert call == source_key(source)
        assert [(r.name, r.bitsize, r.side.name) for r in signature] == [
            (r.name, r.bitsize, r.side.name) for r in source.signature
        ]
        assert [r.dtype.encoding for r in signature] == ["bit", "uint", "uint"]
    source = reference["cadd_small"]
    _, source_counts = source.call_graph(max_depth=1, generalizer=ignore_split_join)
    assert result["cadd_graph"].leaf_counts == {
        source_key(child): count for child, count in source_counts.items()
    }
    counts = result["measured_primitives"]
    assert counts["t"] + counts["tdg"] == 20
    assert counts["measure"] == counts["reset"] == 5
    assert result["modeled_resources"].logical_qubits == 12
