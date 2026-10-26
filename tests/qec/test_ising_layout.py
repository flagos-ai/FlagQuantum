"""Analytic measurement-record fixtures, never physical samples or GPU receipts."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from examples.qec import compare_ising as comparison
from examples.qec import diagnose_predecoder as archive
from examples.qec import ising_layout as layout
from flagquantum.qec.decoders._ai.spacetime import SpacetimePredecoder
from flagquantum.qec.dem.model import DemError, DetectorErrorModel

pytestmark = pytest.mark.integration


def _fixture() -> tuple[Any, SimpleNamespace]:
    stim = pytest.importorskip("stim")
    coordinates = [(2, 2), (2, 6), (4, 0), (4, 4), (0, 2), (4, 2), (2, 4), (6, 4)]
    supports = [
        (0, 1, 3, 4),
        (2, 5),
        (3, 6),
        (4, 5, 7, 8),
        (0, 1),
        (3, 4, 6, 7),
        (1, 2, 4, 5),
        (7, 8),
    ]
    code = SimpleNamespace(
        data_qubits=tuple(range(9)),
        xcheck_qubits=(9, 10, 11, 12),
        zcheck_qubits=(13, 14, 15, 16),
        code_dict={
            "data": {
                q: {"coord": [1 + 2 * (q // 3), 1 + 2 * (q % 3)]} for q in range(9)
            },
            "syndrome_X": {
                q: {"coord": list(coordinates[q - 9])} for q in range(9, 13)
            },
            "syndrome_Z": {
                q: {"coord": list(coordinates[q - 9])} for q in range(13, 17)
            },
        },
        hx=torch.tensor([[int(q in s) for q in range(9)] for s in supports[:4]]),
        hz=torch.tensor([[int(q in s) for q in range(9)] for s in supports[4:]]),
    )
    # Absolute records: first X=0..3/Z=4..7, next X=8..11/Z=12..15,
    # terminal data=16..24. The parity oracle is stated independently below.
    parities = [
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 8),
        (1, 9),
        (2, 10),
        (3, 11),
        (4, 12),
        (5, 13),
        (6, 14),
        (7, 15),
        (8, 16, 17, 19, 20),
        (9, 18, 21),
        (10, 19, 22),
        (11, 20, 21, 23, 24),
    ]
    circuit = stim.Circuit()
    for _ in range(2):
        circuit.append("MRX", [9, 10, 11, 12])
        circuit.append("MR", [13, 14, 15, 16])
    circuit.append("MX", list(range(9)))
    for parity in parities:
        circuit.append("DETECTOR", [stim.target_rec(p - 25) for p in parity])
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-9)], 0)
    return circuit, code


def _replace(circuit: Any, index: int, text: str) -> Any:
    stim = pytest.importorskip("stim")
    return stim.Circuit(
        "\n".join(text if i == index else str(op) for i, op in enumerate(circuit))
    )


def test_records_determine_kind_time_and_terminal_support_in_original_order() -> None:
    circuit, code = _fixture()
    shape, sites, kinds = layout.resolve_layout(circuit, code, distance=3, rounds=2)
    assert shape == (3, 4, 4)
    assert sites == (5, 13, 2, 10, 21, 29, 18, 26, 20, 22, 25, 27, 37, 45, 34, 42)
    assert kinds == (1,) * 8 + (2,) * 4 + (1,) * 4
    assert [sum(site // 16 == t for site in sites) for t in range(3)] == [4, 8, 4]
    dem = DetectorErrorModel(16, 1, (DemError(0.01, (0, 5, 12), (0,)),))
    model = SpacetimePredecoder.from_dem(
        dem, grid_shape=shape, sites=sites, kinds=kinds
    )
    layout.verify_encoding(model)
    assert torch.equal(model.h, dem.detector_error_matrix().float())
    assert torch.equal(model.logical, dem.observables_flips_matrix().float())


@pytest.mark.parametrize(
    "index,text",
    [
        (9, "DETECTOR rec[-13] rec[-21]"),  # Put Z instead of X in the first bulk row.
        (
            17,
            "DETECTOR rec[-17] rec[-9] rec[-8] rec[-6]",
        ),  # Missing terminal data term.
        (5, "DETECTOR rec[-26]"),
        (5, "DETECTOR rec[-25] rec[-25]"),
        (0, "MRX 10 9 11 12"),
        (0, "MR 9 10 11 12"),
        (4, "MX !0 1 2 3 4 5 6 7 8"),
        (4, "MY 0 1 2 3 4 5 6 7 8"),
    ],
)
def test_wrong_record_identity_order_basis_or_terminal_is_refused(
    index: int, text: str
) -> None:
    circuit, code = _fixture()
    with pytest.raises(ValueError):
        layout.resolve_layout(
            _replace(circuit, index, text), code, distance=3, rounds=2
        )


def test_missing_and_permuted_detector_rows_are_refused() -> None:
    circuit, code = _fixture()
    for changed in [_replace(circuit, 5, ""), _replace(circuit, 5, str(circuit[6]))]:
        with pytest.raises(ValueError, match="detector"):
            layout.resolve_layout(changed, code, distance=3, rounds=2)


@pytest.mark.parametrize(
    "coordinate", [[True, 2], [2.0, 2], [-2, 2], [1, 2], [2, 8], [2]]
)
def test_invalid_coordinate_values_fail_before_coercion(coordinate: list[int]) -> None:
    circuit, code = _fixture()
    code.code_dict["syndrome_X"][9]["coord"] = coordinate
    with pytest.raises(ValueError):
        layout.resolve_layout(circuit, code, distance=3, rounds=2)


def test_support_and_coordinate_identity_are_checked_independently() -> None:
    circuit, code = _fixture()
    bad = copy.deepcopy(code)
    bad.hx[0, 8] = 1
    with pytest.raises(ValueError, match="support"):
        layout.resolve_layout(circuit, bad, distance=3, rounds=2)
    bad = copy.deepcopy(code)
    bad.code_dict["syndrome_Z"][13]["coord"] = [2, 2]
    with pytest.raises(ValueError, match="colliding"):
        layout.resolve_layout(circuit, bad, distance=3, rounds=2)
    bad = copy.deepcopy(code)
    bad.code_dict["data"][0]["coord"] = [1, 3]
    with pytest.raises(ValueError, match="complete square"):
        layout.resolve_layout(circuit, bad, distance=3, rounds=2)
    bad = copy.deepcopy(code)
    bad.xcheck_qubits = (9, 9, 11, 12)
    with pytest.raises(ValueError, match="identities"):
        layout.resolve_layout(circuit, bad, distance=3, rounds=2)


@pytest.mark.parametrize("distance,rounds", [(2, 2), (3, 1), (True, 2), (3, 2.0)])
def test_unsupported_dimensions_refused(distance: int, rounds: int) -> None:
    circuit, code = _fixture()
    with pytest.raises(ValueError):
        layout.resolve_layout(circuit, code, distance=distance, rounds=rounds)


def test_all_onehots_are_bounded_and_no_sample_or_forward_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dem = DetectorErrorModel(75, 1, (DemError(0.01, (0, 2, 74), (0,)),))
    model = SpacetimePredecoder.from_dem(
        dem, grid_shape=(3, 5, 5), sites=tuple(range(75)), kinds=(1,) * 75
    )
    original = model.encode_input
    rows: list[torch.Tensor] = []

    def encode(bits: torch.Tensor) -> torch.Tensor:
        assert len(bits) <= 32
        rows.append(bits.clone())
        return original(bits)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("no sampler or model forward belongs in layout checks")

    monkeypatch.setattr(model, "encode_input", encode)
    monkeypatch.setattr(model, "sample", forbidden)
    monkeypatch.setattr(model, "forward", forbidden)
    layout.verify_encoding(model)
    actual = torch.cat(rows)
    assert torch.equal(
        actual, torch.cat((torch.zeros(1, 75), torch.ones(1, 75), torch.eye(75)))
    )

    def broken(bits: torch.Tensor) -> torch.Tensor:
        encoded = original(bits)
        encoded[:, 0].zero_()
        return encoded

    monkeypatch.setattr(model, "encode_input", broken)
    with pytest.raises(ValueError, match="roundtrip"):
        layout.verify_encoding(model)


def test_wrong_source_refused_before_vendor_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source does not match"):
        layout.build_model(tmp_path)


def test_cached_vendor_origin_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reference(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace()

    foreign = ModuleType("qec")
    foreign.__file__ = str(tmp_path / "other" / "qec.py")
    monkeypatch.setitem(sys.modules, "qec", foreign)
    monkeypatch.setattr(comparison, "reference", reference)
    with pytest.raises(ValueError, match="module origin"):
        layout.build_model(tmp_path)


def test_wrong_dem_anchor_stops_before_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Guard isolation only: these substituted objects are not a pinned-source receipt.
    circuit, code = _fixture()
    shape, sites, kinds = layout.resolve_layout(circuit, code, distance=3, rounds=2)
    dem = DetectorErrorModel(16, 1, (DemError(0.01, (0, 5, 12), (0,)),))
    (tmp_path / "results.json").write_text(
        json.dumps({"circuit_sha256": "wrong", "training_dem_sha256": "wrong"})
    )

    def reference(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(circuit=circuit)

    def vendor(name: str) -> SimpleNamespace:
        return SimpleNamespace(SurfaceCode=lambda *args, **kwargs: code)

    def mapping(
        *args: object, **kwargs: object
    ) -> tuple[tuple[int, int, int], tuple[int, ...], tuple[int, ...]]:
        return shape, sites, kinds

    def training_dem(value: object) -> DetectorErrorModel:
        return dem

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("bad anchor must stop before constructing a model")

    monkeypatch.setattr(comparison, "reference", reference)
    monkeypatch.setattr(importlib, "import_module", vendor)
    monkeypatch.setattr(layout, "resolve_layout", mapping)
    monkeypatch.setattr(comparison, "training_dem", training_dem)
    monkeypatch.setattr(archive, "ARCHIVE", tmp_path)
    monkeypatch.setattr(SpacetimePredecoder, "from_dem", forbidden)
    with pytest.raises(ValueError, match="archived cell"):
        layout.build_model(tmp_path)
