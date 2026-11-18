"""Execute every original and native instance-loading notebook cell."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import pytest
from qualtran.resource_counting import QECGatesCost, get_cost_value

from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_original_native_cells_graphs_costs_and_physical_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython.display

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/optimization/k_xor_sat/load_kxor_instance.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "9a08818e673ccc5621e4a077ca1fab1f840a4075"
    )
    source = json.loads(raw)
    native = json.loads(
        (root / "examples/notebooks/kxor_instance_loading.ipynb").read_text()
    )
    assert len(source["cells"]) == 11 and len(native["cells"]) == 12
    assert [c["cell_type"] for c in native["cells"][:11]] == [
        c["cell_type"] for c in source["cells"]
    ]
    assert [c["metadata"]["frozen_source_cell"] for c in native["cells"][:11]] == list(
        range(11)
    )
    displays: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    original = execute(source)
    displays.clear()
    source_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "openfermion"}:
            raise AssertionError(
                "Native notebook must use domain-owned implementations"
            )
        return source_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        migrated = execute(native)
    assert len(displays) == 7
    svg = [item for item in displays if isinstance(item, IPython.display.SVG)]
    assert len(svg) == 3
    assert all("<svg" in item.data for item in svg)
    for name in ("load_scopes", "load_scopes_symb"):
        a, b = migrated[name], original[name]
        assert [(r.name, r.bitsize, r.side.name) for r in a.signature] == [
            (r.name, r.bitsize, r.side.name) for r in b.signature
        ]
        assert a.inst.n == b.inst.n and a.inst.k == b.inst.k and a.inst.m == b.inst.m
        if not a.inst.is_symbolic():
            assert a.inst.batched_scopes == b.inst.batched_scopes
    ((qrom, count),) = original["load_scopes_symb_sigma"].items()
    assert migrated["load_scopes_symb_sigma"] == {"QROM": count} == {"QROM": 1}
    assert qrom.data_shape == migrated["load_scopes_symb"].qrom.shape
    formula = migrated["load_scopes_symb_model"]
    expected = get_cost_value(original["load_scopes_symb"], QECGatesCost())
    for name in (
        "and_bloq",
        "clifford",
        "measurement",
        "t",
        "toffoli",
        "rotation",
        "cswap",
    ):
        assert getattr(formula, name) == getattr(expected, name)
    assert migrated["primitive_count"] == 86
    assert migrated["measurement_count"] == 3
    gates = migrated["model_categories"]
    cost = get_cost_value(original["load_scopes"], QECGatesCost())
    for name in (
        "and_bloq",
        "clifford",
        "measurement",
        "t",
        "toffoli",
        "rotation",
        "cswap",
    ):
        assert getattr(gates, name) == getattr(cost, name)
