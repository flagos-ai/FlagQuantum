"""Complete original index notebook and independent native execution accounts."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import pytest

from flagquantum.compiler._component_trace.resource_generalization import FixedTCall
from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_original_and_native_notebooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython.display
    from qualtran.bloqs.arithmetic.conversions import ToContiguousIndex
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.resource_counting.generalizers import ignore_split_join

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/arithmetic/conversions/contiguous_index.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "104352322042258aaed807779f33bf8611053c77"
    )
    original = json.loads(raw)
    native = json.loads(
        (root / "examples/notebooks/triangular_register_index.ipynb").read_text()
    )
    assert len(original["cells"]) == 10 and len(native["cells"]) == 12
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in native["cells"][:10]
    ]
    displays: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    reference = execute(original)
    source = reference["to_contg_index"]
    assert source == ToContiguousIndex(bitsize=4, s_bitsize=8)
    _, sigma = source.call_graph(max_depth=1, generalizer=ignore_split_join)
    assert sigma == {Toffoli(): 19}
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
    signature = result["index_signature"]
    assert [(r.name, r.bitsize, r.side.name) for r in signature] == [
        (r.name, r.bitsize, r.side.name) for r in source.signature
    ]
    assert all(r.dtype.encoding == "uint" for r in signature)
    assert result["index_graph"].leaf_counts == {FixedTCall("toffoli"): 19}
    assert result["to_contg_index"].gates.toffoli == 19
    assert result["to_contg_index"].qubits == 16
    counts = result["measured_primitives"]
    assert counts["t"] + counts["tdg"] == 76
    assert counts["cx"] == 226 and counts["measure"] == 19
    assert result["work_width"] == 22
    assert counts["measure"] == counts["reset"]
    assert counts.get("ccx", 0) == 0
