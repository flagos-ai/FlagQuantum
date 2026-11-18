"""Execute complete frozen and native Kikuchi list/matrix notebooks."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import pytest
import sympy
from qualtran.resource_counting import QECGatesCost, get_cost_value

from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    ("suffix", "blob", "original_cells", "names", "graphs"),
    (
        (
            "list",
            "7ed33e384065876e374a0eb63c88676f09c01b0e",
            10,
            ("kikuchi_nonzero_index",),
            2,
        ),
        (
            "matrix",
            "599e88a21a42bfe9ca32ce7bc08277f7b5aa44f1",
            11,
            ("kikuchi_matrix_entry_symb", "kikuchi_matrix_entry"),
            3,
        ),
    ),
)
def test_complete_original_and_native_cells_signatures_graphs_and_models(
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    blob: str,
    original_cells: int,
    names: tuple[str, ...],
    graphs: int,
) -> None:
    import IPython.display

    root = Path(__file__).resolve().parents[2]
    filename = f"kikuchi_adjacency_{suffix}.ipynb"
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/optimization/k_xor_sat"
        / filename
    ).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    source = json.loads(raw)
    native = json.loads((root / "examples/notebooks" / filename).read_text())
    assert len(source["cells"]) == original_cells
    assert len(native["cells"]) == original_cells + 1
    assert [cell["cell_type"] for cell in native["cells"][:-1]] == [
        cell["cell_type"] for cell in source["cells"]
    ]
    assert [
        cell["metadata"]["frozen_source_cell"] for cell in native["cells"][:-1]
    ] == list(range(original_cells))
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
    svg = [item for item in displays if isinstance(item, IPython.display.SVG)]
    assert len(svg) == graphs
    assert all("<svg" in item.data for item in svg)
    for name in names:
        actual, expected = migrated[name], original[name]
        assert actual.ell == expected.ell
        assert [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in actual.signature
        ] == [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name)
            for r in expected.signature
        ]
        assert (actual.inst.n, actual.inst.m, actual.inst.k) == (
            expected.inst.n,
            expected.inst.m,
            expected.inst.k,
        )
        if not actual.inst.is_symbolic():
            assert actual.inst.batched_scopes == expected.inst.batched_scopes
        cost = get_cost_value(expected, QECGatesCost())
        model = actual.resource_gates()
        for field in (
            "t",
            "toffoli",
            "cswap",
            "and_bloq",
            "clifford",
            "rotation",
            "measurement",
        ):
            assert sympy.simplify(getattr(model, field) - getattr(cost, field)) == 0
    graph_name = names[0]
    normalized = {}
    for child, count in original[graph_name + "_sigma"].items():
        if type(child).__name__ == "Always":
            child = child.subbloq
        inverse = type(child).__name__ == "Adjoint"
        if inverse:
            child = child.subbloq
        normalized[type(child).__name__ + ("†" if inverse else "")] = count
    assert migrated[graph_name + "_sigma"] == normalized
    assert migrated["model_categories"] == migrated[names[-1]].resource_gates()
