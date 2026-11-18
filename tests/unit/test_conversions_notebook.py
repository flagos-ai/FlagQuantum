"""All conversion teaching declarations run on native metadata and circuits."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import pytest

from flagquantum.compiler._component_trace.resource_generalization import (
    FanoutCall,
    FixedTCall,
)
from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_original_and_native_notebooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython.display
    from qualtran import QFxp, QInt, QIntOnesComp, QUInt
    from qualtran.bloqs.arithmetic.conversions import SignExtend
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.bloqs.mcmt import MultiTargetCNOT
    from qualtran.resource_counting.generalizers import ignore_split_join

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/arithmetic/conversions/conversions.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "9b9f5f1a659905b2fc9f7e6dfd7e7bc1a0cb4783"
    )
    original = json.loads(raw)
    native = json.loads(
        (root / "examples/notebooks/register_conversions.ipynb").read_text()
    )
    assert len(original["cells"]) == 34 and len(native["cells"]) == 36
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in native["cells"][:34]
    ]
    displays: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    reference = execute(original)
    assert reference["sign_extend"] == SignExtend(QInt(8), QInt(16))
    fxp = reference["sign_extend_fxp"]
    assert fxp == SignExtend(QFxp(8, 4, signed=True), QFxp(16, 4, signed=True))
    # Fixed-point classical values use scaled integer storage, as upstream does.
    assert fxp.call_classically(x=-16) == (-16,)
    assert fxp.decompose_bloq().signature == fxp.signature
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
    assert len(displays) == 13
    for name in (
        "signed_to_twos",
        "to_contg_index",
        "sign_extend",
        "sign_extend_fxp",
        "sign_truncate",
    ):
        source, example = reference[name], result[name]
        assert [(r.name, r.bitsize, r.side.name) for r in example.signature] == [
            (r.name, r.bitsize, r.side.name) for r in source.signature
        ]
        encodings = {
            QInt: "signed",
            QIntOnesComp: "ones_complement",
            QUInt: "uint",
            QFxp: "fixed_point",
        }
        assert [r.dtype.encoding for r in example.signature] == [
            encodings[type(r.dtype)] for r in source.signature
        ]
        if name == "sign_extend_fxp":
            for r in example.signature:
                assert dict(r.dtype.parameters) == {
                    "fractional_width": 4,
                    "signed": True,
                }
        _, source_counts = source.call_graph(max_depth=1, generalizer=ignore_split_join)
        if name in {"signed_to_twos", "to_contg_index"}:
            assert example.child == FixedTCall("toffoli")
            assert source_counts == {Toffoli(): example.multiplicity}
        else:
            assert example.child == FanoutCall(8)
            assert source_counts == {MultiTargetCNOT(8): example.multiplicity}
    assert result["signed_to_twos"].multiplicity == 8
    assert result["to_contg_index"].multiplicity == 19
    assert result["measured_primitives"] == {"cx": 15}
    for name in ("signed_to_twos", "to_contg_index", "sign_extend", "sign_truncate"):
        example = result[name]
        assert result[name + "_graph"].leaf_counts == {
            example.child: example.multiplicity
        }
