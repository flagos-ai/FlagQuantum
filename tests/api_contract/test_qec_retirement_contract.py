"""Prepare the approved 0.4.0 break without activating it or refreshing fixtures."""

from __future__ import annotations

import base64
import importlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

import flagquantum.qec as qec
from tools import check_qec_retirement as gate

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = gate.load_contract()
RETIRED = tuple(CONTRACT["module_replacements"])


def test_contract_limits_the_authorized_break_and_records_single_class_amendment() -> (
    None
):
    assert CONTRACT["status"] == "approved_contract_before_implementation"
    assert CONTRACT["current_phase"] == "prepare"
    assert CONTRACT["target_version"] == "0.4.0"
    assert CONTRACT["proposal"]["pull_request"] == 757
    assert CONTRACT["proposal"]["head"] == "ed0b572657c8249c1d7fe605e8d0b05cdd74fc7b"
    assert len(RETIRED) == len(set(RETIRED)) == 23
    assert all(name.rsplit(".", 1)[0] == "flagquantum.qec" for name in RETIRED)
    assert {name.rsplit(".", 1)[1] for name in RETIRED} == {
        "belief_propagation",
        "bicycle",
        "bposd",
        "chunks",
        "circuit",
        "context",
        "css_code",
        "decoding_graph",
        "dem_alternatives",
        "dem_circuit",
        "dem_construction",
        "dem_matrices",
        "dem_text",
        "floquet",
        "matching",
        "qldpc",
        "reed_muller",
        "registry",
        "repetition",
        "sampling",
        "sliding_window",
        "subsystem",
        "surface",
    }
    surface = CONTRACT["rotated_surface"]
    assert surface["decision"] == "single_registered_class"
    assert (
        surface["authority"]
        == "flagquantum.qec.codes.surface.rotated.RotatedSurfaceCode"
    )
    assert surface["old_class_identity_guarantee"] is False
    assert surface["numerical_equivalence_required"] is True
    assert CONTRACT["approval"]["rotated_surface_amendment"]
    assert CONTRACT["preserved"]["ordered_root_export_count"] == 108
    assert CONTRACT["preserved"]["shared_leaves"] == [
        "gf2",
        "pauli",
        "logical",
        "noise",
        "types",
    ]
    assert (
        CONTRACT["verification"]["fixture_regeneration_skip_xfail_or_threshold_changes"]
        is False
    )


def test_all_frozen_fixture_bytes_are_protected() -> None:
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests/qec/fixtures").iterdir()
        if path.is_file()
    }
    assert set(CONTRACT["frozen_fixture_sha256"]) == actual
    assert gate.fixture_errors() == []


def test_fixture_gate_rejects_a_missing_or_changed_artifact(tmp_path: Path) -> None:
    assert len(gate.fixture_errors(tmp_path)) == len(CONTRACT["frozen_fixture_sha256"])
    first = next(iter(CONTRACT["frozen_fixture_sha256"]))
    path = tmp_path / first
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed fixture")
    assert first + ": frozen fixture changed or missing" in gate.fixture_errors(
        tmp_path
    )


def test_current_implementation_passes_preparation_but_not_retirement() -> None:
    assert gate.check("prepare") == []
    errors = gate.check("retired")
    for name in RETIRED:
        assert f"{name}: retired module is still importable" in errors
    assert "RotatedSurfaceCode: duplicate implementation remains" in errors
    assert any("expected one definition" in error for error in errors)
    with pytest.raises(ValueError, match="phase must be"):
        gate.check("automatic")


def test_gate_rejects_reordered_root_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qec, "__all__", tuple(reversed(qec.__all__)))
    assert "flagquantum.qec: ordered 108 root exports changed" in gate.check("prepare")


@pytest.mark.parametrize("name", RETIRED)
def test_retired_import_policy_accepts_only_exact_module_absence(name: str) -> None:
    # This import-boundary fake proves the future policy, not migrated runtime code.
    with patch.object(
        gate, "import_module", side_effect=ModuleNotFoundError(name=name)
    ):
        assert gate.import_errors((name,), absent=True) == []
        assert gate.import_errors((name,), absent=False)
    with patch.object(gate, "import_module", return_value=ModuleType(name)):
        assert gate.import_errors((name,), absent=True) == [
            f"{name}: retired module is still importable"
        ]
    with patch.object(
        gate, "import_module", side_effect=ModuleNotFoundError(name="torch")
    ):
        assert gate.import_errors((name,), absent=True) == [
            f"{name}: unexpected missing module torch"
        ]


def test_root_workflow_and_registered_class_remain_authoritative() -> None:
    from flagquantum.qec import codes
    from flagquantum.qec.codes.surface import RotatedSurfaceCode as Surface
    from flagquantum.qec.codes.surface.rotated import RotatedSurfaceCode as Owner

    baseline = json.loads(
        (ROOT / CONTRACT["preserved"]["root_api_fixture"]).read_text()
    )
    assert list(qec.__all__) == baseline["exports"]
    assert qec.RotatedSurfaceCode is codes.RotatedSurfaceCode is Surface is Owner
    assert list(qec.code_names()) == baseline["code_names"]
    assert list(qec.decoder_names()) == baseline["decoder_names"]
    for distance in (3, 5, 7):
        code = qec.get_code("rotated_surface", distance=distance)
        assert type(code) is Owner
        restored = pickle.loads(pickle.dumps(code, protocol=4))
        assert type(restored) is Owner
        assert restored == code
        assert restored.checks == code.checks
        assert restored.logical_observables == code.logical_observables
        assert type(code).__module__ == "flagquantum.qec.codes"
    model = qec.DetectorErrorModel.from_stim_text(
        "error(0.125) D0 L0\ndetector D0\nlogical_observable L0\n"
    )
    assert qec.get_decoder(qec.AUTHORITY_NAME, model).decode((0,)).observables == (0,)
    sample = qec.DetectorErrorModel(
        2, 1, (qec.DemError(0.125, (0,), (0,)), qec.DemError(0.25, (0, 1), ()))
    ).dem_sampling(shots=16, seed=7)
    assert sample.detectors.tolist() == baseline["seeded_dem"]["detectors"]
    assert sample.observables.tolist() == baseline["seeded_dem"]["observables"]


@pytest.mark.parametrize("distance", (3, 5, 7))
def test_single_class_decision_preserves_both_current_records_numerically(
    distance: int,
) -> None:
    from flagquantum.qec.surface import RotatedSurfaceCode as Historical

    registered = qec.get_code("rotated_surface", distance=distance)
    historical = Historical(distance)
    assert type(registered) is not type(historical)
    for name in (
        "checks",
        "stabilizers",
        "logical_observables",
        "data_qubits",
        "ancilla_qubits",
        "num_data_qubits",
        "num_ancilla_qubits",
        "num_ancilla_x_qubits",
        "num_ancilla_z_qubits",
    ):
        assert getattr(registered, name) == getattr(historical, name), name
    noise = qec.PhenomenologicalNoise(data_flip=0.125, measurement_flip=0.25)
    assert (
        qec.DetectorErrorModel.from_code(
            registered, noise=noise, num_rounds=2
        ).to_stim_text()
        == qec.DetectorErrorModel.from_code(
            historical, noise=noise, num_rounds=2
        ).to_stim_text()
    )


def test_plugins_use_the_same_authorities_and_keep_duplicate_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.qec import codes
    from flagquantum.qec.decoders import registry

    monkeypatch.setattr(codes, "_CODES", dict(codes._CODES))
    monkeypatch.setattr(registry, "_DECODERS", dict(registry._DECODERS))
    for name in ("register_code", "get_code", "code_names"):
        assert getattr(qec, name) is getattr(codes, name)
    for name in ("register_decoder", "get_decoder", "decoder_names"):
        assert getattr(qec, name) is getattr(registry, name)
    assert (
        qec.register_code("retirement_probe")(qec.RotatedSurfaceCode)
        is qec.RotatedSurfaceCode
    )
    assert (
        type(codes.get_code("retirement_probe", distance=3)) is qec.RotatedSurfaceCode
    )
    with pytest.raises(ValueError, match="already"):
        codes.register_code("retirement_probe")(qec.SteaneCode)
    qec.register_decoder("retirement_probe")(qec.MinimumWeightMatchingDecoder)
    model = qec.DetectorErrorModel(1, 1, (qec.DemError(0.125, (0,), (0,)),))
    assert registry.get_decoder("retirement_probe", model).decode((0,)).observables == (
        0,
    )
    with pytest.raises(ValueError, match="already"):
        registry.register_decoder("retirement_probe")(qec.MinimumWeightMatchingDecoder)


def test_surviving_and_retiring_pickle_lookups_are_classified_without_rewriting_fixtures() -> (
    None
):
    baseline = json.loads(
        (ROOT / CONTRACT["preserved"]["root_api_fixture"]).read_text()
    )
    retiring, surviving = [], []
    for name, encoded in baseline["pickles"].items():
        module, attribute = name.rsplit(".", 1)
        value = pickle.loads(base64.b64decode(encoded))
        assert type(value) is getattr(importlib.import_module(module), attribute)
        (retiring if module in RETIRED else surviving).append(name)
    assert retiring == [
        "flagquantum.qec.surface.ZxxzSurfaceCode",
        "flagquantum.qec.circuit.MeasurementRef",
        "flagquantum.qec.surface.RotatedSurfaceCode",
    ]
    assert len(surviving) == 10
    assert "flagquantum.qec.surface.RotatedSurfaceCode" in retiring
    assert "flagquantum.qec.codes.RotatedSurfaceCode" in surviving
    assert gate._pickle_errors(baseline, set(RETIRED), active=False) == []
    errors = gate._pickle_errors(baseline, set(RETIRED), active=True)
    assert errors == [
        f"{name}: retired pickle lookup still succeeds" for name in retiring
    ]


@pytest.mark.integration
def test_phase_commands_are_bounded_and_retirement_does_not_claim_completion() -> None:
    for phase, returncode in (("prepare", 0), ("retired", 1)):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/check_qec_retirement.py"),
                "--phase",
                phase,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == returncode, result.stdout + result.stderr
        assert (
            "passed: prepare, 23 modules, 108 exports"
            if phase == "prepare"
            else "retired module is still importable"
        ) in result.stdout
