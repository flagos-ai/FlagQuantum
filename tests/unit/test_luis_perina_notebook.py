"""Execute all original and native Luis–Perina notebook cells."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from flagquantum.core.value_math import SymbolicReal
from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_call_key(child: Any) -> tuple[str, SymbolicReal]:
    """Normalize external graph nodes at the reference-only SDK boundary."""
    from qualtran import Adjoint
    from qualtran.bloqs.basic_gates import CZ, CRz, Hadamard, OnEach, Ry, Rz, XGate
    from qualtran.bloqs.phase_estimation.lp_resource_state import LPRSInterimPrep
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    if isinstance(child, OnEach):
        assert isinstance(child.gate, Hadamard)
        return "OnEach(H)", child.n
    if isinstance(child, Hadamard):
        return "H", 0
    if isinstance(child, Rz):
        return "Rz", child.angle
    if isinstance(child, LPRSInterimPrep):
        return "LuisPerinaInterim", child.bitsize
    if isinstance(child, Adjoint):
        assert isinstance(child.subbloq, LPRSInterimPrep)
        return "LuisPerinaInterim†", child.subbloq.bitsize
    if isinstance(child, Ry):
        return "Ry", child.angle
    if isinstance(child, ReflectionUsingPrepare):
        assert child.global_phase == 1j
        return "ReflectionZero(i)", sum(
            r.total_bits() for r in child.selection_registers
        )
    if isinstance(child, XGate):
        return "X", 0
    if isinstance(child, CZ):
        return "CZ", 0
    assert isinstance(child, CRz)
    return "CRz", child.angle


def test_complete_source_and_native_notebooks(monkeypatch: pytest.MonkeyPatch) -> None:
    import IPython.display

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/phase_estimation/lp_resource_state.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "5e4568579be1cbfec1c0ab614fb9bc98c9641590"
    )
    original = json.loads(raw)
    migrated = json.loads(
        (root / "examples/notebooks/luis_perina_phase_window.ipynb").read_text()
    )
    assert len(original["cells"]) == 19 and len(migrated["cells"]) == 21
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in migrated["cells"][:19]
    ]
    assert [
        c["metadata"]["frozen_source_cell"] for c in migrated["cells"][:19]
    ] == list(range(19))
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
    assert len(displays) == 9
    for name in (
        "lprs_interim_prep",
        "lp_resource_state_small",
        "lp_resource_state_symbolic",
    ):
        actual, expected = native[name], source[name]
        assert actual.bitsize == expected.bitsize
        assert [
            (r.name, r.bitsize, r.shape, r.side.name) for r in actual.signature
        ] == [(r.name, r.bitsize, r.shape, r.side.name) for r in expected.signature]
    interim = {
        source_call_key(child): count
        for child, count in source["lprs_interim_prep_sigma"].items()
    }
    actual_interim = {
        child: count for (_, child), count in native["lprs_interim_direct"].items()
    }
    assert actual_interim == interim
    state_calls = {
        source_call_key(child): count
        for child, count in source["lp_resource_state_small_sigma"].items()
    }
    assert native["lp_resource_graph"].leaf_counts == state_calls
    np.testing.assert_allclose(
        native["prepared"][:, 0], native["expected"], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(native["prepared"][:, 1:], 0, atol=1e-12, rtol=0)
    assert (native["work"], native["slots"]) == (6, 3)
    assert native["primitive_counts"]["measure"] == 3
    assert native["declared_qec"] == {
        "and_bloq": 3,
        "clifford": 49,
        "rotation": 18,
        "measurement": 3,
    }
