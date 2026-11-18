"""Execute complete frozen and native textbook-QPE notebook responsibilities."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tests.unit.test_addition_file_audit import execute
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_source_notebook_and_native_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython.display
    from qualtran import Adjoint
    from qualtran.bloqs.basic_gates import CZPowGate, Hadamard, OnEach
    from qualtran.bloqs.bookkeeping import Allocate
    from qualtran.bloqs.phase_estimation import RectangularWindowState
    from qualtran.bloqs.qft.qft_text_book import QFTTextBook

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/phase_estimation/text_book_qpe.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "139af0f36d2671d02ce95abbedcbeda60b1d19ff"
    )
    original = json.loads(raw)
    migrated = json.loads(
        (root / "examples/notebooks/textbook_phase_estimation.ipynb").read_text()
    )
    assert len(original["cells"]) == 24 and len(migrated["cells"]) == 26
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in migrated["cells"][:24]
    ]
    assert [
        c["metadata"]["frozen_source_cell"] for c in migrated["cells"][:24]
    ] == list(range(24))
    displays: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    source = execute(original)
    displays.clear()
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "openfermion"}:
            raise AssertionError("Native notebook must use owned implementations")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        native = execute(migrated)
    assert len(displays) == 12
    for name in (
        "textbook_qpe_small",
        "textbook_qpe_using_m_bits",
        "textbook_qpe_from_standard_deviation_eps",
        "textbook_qpe_from_precision_and_delta",
        "rectangular_window_state_small",
        "rectangular_window_state_symbolic",
    ):
        actual, expected = native[name], source[name]
        assert [
            (r.name, r.bitsize, r.shape, r.side.name) for r in actual.signature
        ] == [(r.name, r.bitsize, r.shape, r.side.name) for r in expected.signature]
    graph = {}
    for child, count in source["textbook_qpe_small_sigma"].items():
        if isinstance(child, RectangularWindowState):
            key = ("RectangularWindow", child.bitsize)
        elif isinstance(child, CZPowGate):
            key = ("CZPow", child.exponent)
        else:
            assert isinstance(child, Adjoint) and isinstance(child.subbloq, QFTTextBook)
            key = ("TextbookQFT†", child.subbloq.bitsize)
        graph[key] = count
    assert native["textbook_qpe_graph"].leaf_counts == graph
    rectangular = {}
    for child, count in source["rectangular_window_state_small_sigma"].items():
        if isinstance(child, Allocate):
            rectangular["Allocate(phase)"] = count
        else:
            assert isinstance(child, OnEach) and isinstance(child.gate, Hadamard)
            rectangular["H"] = count * child.n
    assert native["rectangular_window_graph"].leaf_counts == rectangular
    np.testing.assert_allclose(
        native["phase_amplitudes"], native["expected"], atol=1e-13, rtol=0
    )
    assert native["primitive_counts"] == {"h": 6, "cphase": 10, "swap": 1}
    assert_source_model(native["model"], source["textbook_qpe_small"])
