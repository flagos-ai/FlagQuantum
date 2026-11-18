"""Complete ideal-tensor comparison and separate physical Kaiser child."""

import builtins
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_original_and_native_cells_tensors_samples_and_physical_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython.display
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/phase_estimation/kaiser_window_state.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "c108729f0a6c6bf738d1cca65c91677e17380750"
    )
    original = json.loads(raw)
    migrated = json.loads(
        (root / "examples/notebooks/kaiser_phase_window.ipynb").read_text()
    )
    assert len(original["cells"]) == 16 and len(migrated["cells"]) == 18
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in migrated["cells"][:16]
    ]
    assert [
        c["metadata"]["frozen_source_cell"] for c in migrated["cells"][:16]
    ] == list(range(16))
    displays: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        np.random.seed(73)
        source = execute(original)
        plt.close("all")
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
        assert plt.gca().get_yscale() == "log"
        assert len(plt.gca().lines) == 3
        assert len(plt.gca().collections) == 1
        assert native["theta"] == source["theta"] == 0.51234
        assert native["m"] == source["m"] == 6
        assert native["n_samples"] == source["n_samples"] == 100_000
        for name in (
            "kaiser_window_state_small",
            "kaiser_window_state_symbolic",
            "kaiser_window_state",
        ):
            actual, expected = native[name], source[name]
            assert actual.bitsize == expected.bitsize and actual.alpha == expected.alpha
            assert [(r.name, r.bitsize, r.side.name) for r in actual.signature] == [
                (r.name, r.bitsize, r.side.name) for r in expected.signature
            ]
        ((child, count),) = source["kaiser_window_state_small_sigma"].items()
        assert count == 1 and child.phase_bitsize == 5
        np.testing.assert_array_equal(
            native["kaiser_window_state_small"].source_preparation_coefficients,
            child.state_coefficients,
        )
        assert native["kaiser_window_graph"].leaf_counts == {
            ("RotationPreparation", 32, 5): 1
        }
        for operation_name, sample_name in (
            ("qpe_textbook", "thetas_textbook"),
            ("qpe_sinstate", "thetas_sinstate"),
            ("qpe_kaiser_state", "thetas_kaiser_state"),
        ):
            expected_tensor = source["construct_composite_boq"](
                source[operation_name]
            ).tensor_contract()
            circuit = native["construct_composite_boq"](native[operation_name])
            actual = circuit.state().detach().numpy().reshape(64, 2)
            np.testing.assert_allclose(
                actual[:, 1], expected_tensor, atol=2e-13, rtol=0
            )
            np.testing.assert_allclose(actual[:, 0], 0, atol=1e-14, rtol=0)
            assert len(circuit.to_ir().instructions) == 87
            probabilities = np.abs(expected_tensor) ** 2
            for samples in (source[sample_name], native[sample_name]):
                assert (
                    samples.shape == (100_000,)
                    and 0 <= samples.min() <= samples.max() < 64
                )
                counts = np.bincount(samples, minlength=64) / 100_000
                assert np.all(
                    np.abs(counts - probabilities)
                    <= 6 * np.sqrt(probabilities * (1 - probabilities) / 100_000)
                    + 2 / 100_000
                )
        thetas = np.array([0.25, 0.5, 0.75])
        assert native["holevo_variance"](thetas) == source["holevo_variance"](thetas)
        np.testing.assert_allclose(
            native["physical_joint"][:, 1:], 0, atol=1e-12, rtol=0
        )
        assert native["ideal_tensor_error"] == pytest.approx(
            0.2561492360899099, abs=1e-13
        )
        assert sum(native["primitive_counts"].values()) == 158
        assert native["primitive_counts"]["measure"] == 7
        assert (native["work"], native["slots"]) == (5, 2)
        model = native["declared_model"].logical
        assert {key: value for key, value in asdict(model.gates).items() if value} == {
            "and_bloq": 4,
            "clifford": 26,
            "measurement": 4,
        }
        assert model.peak_logical_qubits == 9
    finally:
        plt.close("all")
