"""Execute the complete physical-cost notebook through native domain owners."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

np = pytest.importorskip("numpy")
pytest.importorskip("qualtran.surface_code")
plt = pytest.importorskip("matplotlib.pyplot")


def test_all_native_notebook_cells_and_original_scenario_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.test_addition_file_audit import execute

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/surface_code/physical_cost_model.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "5cb5070d0ea667fcb51b1fe972258d8761f8240d"
    )
    original = json.loads(raw)
    native = json.loads(
        (root / "examples/notebooks/physical_cost_model.ipynb").read_text()
    )
    assert len(original["cells"]) == 21
    assert len(native["cells"]) == 22
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in native["cells"][:-1]
    ]
    assert [c["metadata"]["frozen_source_cell"] for c in native["cells"][:-1]] == list(
        range(21)
    )
    plt.close("all")
    source_state = execute(original)
    source_points = np.array(plt.gca().collections[0].get_offsets())
    plt.close("all")
    original_import = builtins.__import__

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
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        native_state = execute(native)
    for name in ("duration_hr_estimates", "n_physical_qubits_estimates"):
        assert len(native_state[name]) == len(source_state[name]) == 48
        np.testing.assert_array_equal(native_state[name], source_state[name])
    np.testing.assert_array_equal(plt.gca().collections[0].get_offsets(), source_points)
    for key in ("cost_model", "cost_model_2"):
        actual, expected = native_state[key], source_state[key]
        assert actual.n_cycles(native_state["alg"]) == expected.n_cycles(
            source_state["alg"]
        )
        assert actual.n_physical_qubits(native_state["alg"]) == expected.n_phys_qubits(
            source_state["alg"]
        )
        assert actual.error(native_state["alg"]) == pytest.approx(
            expected.error(source_state["alg"]), rel=0, abs=8e-7
        )
    for key in ("simple_data_block", "fast_data_block"):
        assert native_state[key].physical_qubits(100) == source_state[
            key
        ].n_physical_qubits(100)
    assert native_state["ccz_factory"].n_cycles(
        native_state["gate_counts"], native_state["err_model"]
    ) == source_state["ccz_factory"].n_cycles(
        source_state["gate_counts"], source_state["err_model"]
    )
    plt.close("all")
