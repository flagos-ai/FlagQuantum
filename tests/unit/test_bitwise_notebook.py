"""The complete bitwise teaching asset runs on owned metadata and circuits."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import pytest

from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_source_and_native_notebooks(monkeypatch: pytest.MonkeyPatch) -> None:
    import IPython.display
    from qualtran import QAny, QUInt
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot, Xor

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent / "ftqc-reference-env/qualtran/bloqs/arithmetic/bitwise.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "1057f0d643ffeb5c7c9f507f5ba1043913c5aa1a"
    )
    original = json.loads(raw)
    native = json.loads(
        (root / "examples/notebooks/register_bitwise.ipynb").read_text()
    )
    assert len(original["cells"]) == 20 and len(native["cells"]) == 22
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in native["cells"][:20]
    ]
    rendered: list[object] = []
    monkeypatch.setattr(IPython.display, "display", rendered.append)
    reference = execute(original)
    assert reference["xor"] == Xor(QAny(4))
    assert reference["bitwise_not"] == BitwiseNot(QUInt(4))
    rendered.clear()
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

    with monkeypatch.context() as native_patch:
        native_patch.setattr(builtins, "__import__", native_import)
        result = execute(native)
    assert len(rendered) == 7
    assert result["measured_primitives"] == {"xor": {"cx": 4}, "complement": {"x": 4}}
    for name in ("xor", "xor_symbolic"):
        _, signature = result[name]
        assert [r.name for r in signature] == ["x", "y"]
        assert all(r.dtype.encoding == "opaque" for r in signature)
    for name in ("bitwise_not", "bitwise_not_symbolic"):
        _, signature = result[name]
        assert [r.name for r in signature] == ["x"]
        assert signature[0].dtype.encoding == "uint"
    assert list(result["xor_graph"].leaf_counts.values()) == [4]
    assert list(result["bitwise_not_graph"].leaf_counts.values()) == [1]
