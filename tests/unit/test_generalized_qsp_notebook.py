"""Execute the complete frozen/native GQSP notebooks and their append example."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import cirq
import numpy as np
import pytest

from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_original_and_native_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    import IPython.display
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent / "ftqc-reference-env/qualtran/bloqs/qsp/generalized_qsp.ipynb"
    ).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "bdc342ca7bb678eac2c98065d198b035a19bc293"
    )
    original = json.loads(raw)
    migrated = json.loads(
        (root / "examples/notebooks/generalized_signal_processing.ipynb").read_text()
    )
    assert len(original["cells"]) == 12 and len(migrated["cells"]) == 14
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in migrated["cells"][:12]
    ]
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
    assert len(displays) == 7
    for name in ("gqsp", "gqsp_with_negative_power", "gqsp_with_large_negative_power"):
        actual, expected = native[name], source[name]
        np.testing.assert_array_equal(actual.p, expected.P)
        np.testing.assert_array_equal(actual.q, expected.Q)
        assert actual.negative_power == expected.negative_power
        assert actual.precision == expected.precision == 0
        for left, right in zip(actual._phases, expected._qsp_phases, strict=True):
            np.testing.assert_array_equal(left, right)
        assert [
            (r.name, r.bitsize, r.shape, r.side.name) for r in actual.signature
        ] == [(r.name, r.bitsize, r.shape, r.side.name) for r in expected.signature]
    counts = {}
    for child, count in source["gqsp_sigma"].items():
        if isinstance(child, SU2RotationGate):
            key = "SU2Rotation"
        else:
            assert child == source["gqsp"].U.controlled(ctrl_spec=CtrlSpec(cvs=0))
            key = "OpenControlledX"
        counts[key] = count
    assert (
        native["gqsp_graph"].leaf_counts
        == counts
        == {
            "SU2Rotation": 2,
            "OpenControlledX": 1,
        }
    )
    np.testing.assert_allclose(
        native["complete_state"][:, 0],
        cirq.unitary(source["gqsp"]) @ native["initial"],
        atol=1e-13,
        rtol=0,
    )
    np.testing.assert_array_equal(native["complete_state"][:, 1], 0)
    assert native["modeled_resources"].gates.rotation == 6
    assert native["modeled_resources"].gates.clifford == 3
    assert native["modeled_resources"].peak_logical_qubits == 2
    assert not (native["primitive_counts"].keys() & {"measure", "reset"})
