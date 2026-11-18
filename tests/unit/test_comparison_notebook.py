"""Complete comparison teaching asset, original diagrams and owned execution."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import pytest

from examples.algorithms.comparison_catalog import comparison_examples
from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_locked_and_native_comparison_notebooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython.display
    from qualtran import Bloq, QAny, QBit, QInt, QUInt
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent / "ftqc-reference-env/qualtran/bloqs/arithmetic/comparison.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "7dfc96afc9eb98246acbb7a949d1f5b6325888b7"
    )
    original = json.loads(raw)
    native = json.loads(
        (root / "examples/notebooks/register_comparison.ipynb").read_text()
    )
    assert len(original["cells"]) == 108 and len(native["cells"]) == 110
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in native["cells"][:108]
    ]
    reference = execute(original)
    examples = comparison_examples()
    assert [name for name, _, _ in examples] == [
        name for name, value in reference.items() if isinstance(value, Bloq)
    ]
    encodings = {QAny: "opaque", QBit: "bit", QInt: "signed", QUInt: "uint"}
    for name, signature, calls in examples:
        source = reference[name]
        assert [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name, r.dtype.encoding)
            for r in signature
        ] == [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name, encodings[type(r.dtype)])
            for r in source.signature
        ]
        assert calls == {
            repr(child): count
            for child, count in reference.get(name + "_sigma", {}).items()
        }
    # Preserve distinct same-named forward/adjoint mixer and comparator nodes.
    leq_calls = dict((name, calls) for name, _, calls in examples)["leq"]
    assert (
        leq_calls["BiQubitsMixer(is_adjoint=False)"]
        == leq_calls["BiQubitsMixer(is_adjoint=True)"]
        == 3
    )
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
                "Native comparison teaching must use owned implementations"
            )
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        result = execute(native)
    assert len(displays) == 41
    for name, _, calls in examples:
        assert result[name][2] == calls
        if name + "_sigma" in reference:
            assert result[name + "_sigma"] == calls
    assert result["less_than_model"].gates.total_t_count() == 32
    assert (
        result["less_than_model"].gates.total_t_count()
        == get_cost_value(reference["lt_k"], QECGatesCost()).total_t_count()
    )
    assert result["less_than_model"].peak_logical_qubits == get_cost_value(
        reference["lt_k"], QubitCount()
    )
    assert result["half_model"].and_bloq == 3
    assert result["half_model"].logical_qubits == 12
