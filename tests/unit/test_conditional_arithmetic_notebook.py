"""The complete conditional arithmetic notebook retains source topics and calls."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest
from sympy import Symbol

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    PredicateControlledCall,
)
from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_original_and_native_notebooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython.display
    from qualtran import CtrlSpec, QInt, QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.resource_counting.generalizers import ignore_split_join

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/arithmetic/controlled_add_or_subtract.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "97da1cd23d35b8535e5da1b9a4e644ffcd16c869"
    )
    original = json.loads(raw)
    native = json.loads(
        (root / "examples/notebooks/register_conditional_arithmetic.ipynb").read_text()
    )
    assert len(original["cells"]) == 12 and len(native["cells"]) == 14
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in native["cells"][:12]
    ]
    displays: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    reference = execute(original)
    n = Symbol("n")
    names = (
        "ctrl_add_or_sub_unsigned",
        "ctrl_add_or_sub_signed",
        "ctrl_add_or_sub_signed_symb",
    )
    sources = (
        ControlledAddOrSubtract(QUInt(8)),
        ControlledAddOrSubtract(QInt(8)),
        ControlledAddOrSubtract(QInt(n)),
    )
    assert tuple(reference[name] for name in names) == sources
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
    for name, source in zip(names, sources, strict=True):
        example = result[name]
        assert tuple(r.name for r in example.signature) == ("ctrl", "a", "b")
        assert tuple(r.bitsize for r in example.signature) == tuple(
            r.bitsize for r in source.signature
        )
        assert example.signed == isinstance(source.a_dtype, QInt)
        _, sigma = source.call_graph(max_depth=1, generalizer=ignore_split_join)
        assert sigma == {
            XGate(): 2,
            BitwiseNot(source.b_dtype).controlled(CtrlSpec()): 2,
            Add(source.a_dtype, source.b_dtype): 1,
        }
        encoding: Literal["twos_complement", "unsigned"] = (
            "twos_complement" if example.signed else "unsigned"
        )
        assert example.direct_calls() == {
            CliffordCall("x", 1): 2,
            PredicateControlledCall(
                ArithmeticCall("bitwise_not", (example.width,), (encoding,)), (1,)
            ): 2,
            ArithmeticCall("add", (example.width, example.width), (encoding,) * 2): 1,
        }
    assert sorted(result["conditional_graph"].leaf_counts.values()) == [1, 2, 2]
    counts = result["measured_primitives"]
    assert counts["t"] + counts["tdg"] == 28
    assert counts["cx"] == 97 and counts["x"] == 2
    assert counts["measure"] == counts["reset"] == 7
    model = result["modeled_resources"]
    assert model.logical_qubits == 24
