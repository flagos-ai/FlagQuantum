"""Check source smoke logic and failure isolation; real installs run in package CI."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec.adapters.pymatching_graph import PyMatchingGraphDecoder
from flagquantum.qec.decoders.matching_graph import MatchingEdge, MatchingGraph
from tools.check_qec_installed_profiles import _run
from tools.qec_installed_smoke import PROFILES, _exercise, _exercise_matching_graph

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("profile", PROFILES)
def test_public_workflow_in_the_available_source_profile(profile: str) -> None:
    available = tuple(
        importlib.util.find_spec(name) is not None for name in ("stim", "pymatching")
    )
    if available != PROFILES[profile]:
        pytest.skip("this source environment does not have that real SDK configuration")
    graph_report = _exercise_matching_graph(profile)
    assert (graph_report is not None) == available[1]
    records = _exercise(profile)
    assert len(records) == (4 if available[0] else 0)
    for record in records:
        assert record["shots"] == 35
        assert record["observables"] == 1


def test_installer_child_cannot_import_through_caller_python_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        monkeypatch.setenv(name, "/unrelated/source")
    log = tmp_path / "child.log"
    _run(
        [
            sys.executable,
            "-c",
            "import json,os; print(json.dumps([os.getenv(x) for x in "
            "('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE')]))",
        ],
        cwd=tmp_path,
        log=log,
        seconds=10,
    )
    assert json.loads(log.read_text()) == [None, None, None]


def test_failed_install_is_not_reclassified_as_a_success(tmp_path: Path) -> None:
    log = tmp_path / "failed.log"
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run(
            [sys.executable, "-c", "print('preserved failure'); raise SystemExit(13)"],
            cwd=tmp_path,
            log=log,
            seconds=10,
        )
    assert error.value.returncode == 13
    assert "preserved failure" in log.read_text()


def test_dependency_warning_does_not_corrupt_the_json_report(tmp_path: Path) -> None:
    report = tmp_path / "smoke.json"
    _run(
        [
            sys.executable,
            "-c",
            "import json,sys; print('dependency warning', file=sys.stderr); "
            "print(json.dumps({'profile': 'core'}))",
        ],
        cwd=tmp_path,
        log=report,
        seconds=10,
    )
    assert json.loads(report.read_text()) == {"profile": "core"}
    assert "dependency warning" in report.with_suffix(".stderr.log").read_text()


def test_hanging_child_is_bounded_and_its_log_is_retained(tmp_path: Path) -> None:
    log = tmp_path / "timed-out.log"
    with pytest.raises(subprocess.TimeoutExpired):
        _run(
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(30)",
            ],
            cwd=tmp_path,
            log=log,
            seconds=1,
        )
    assert "started" in log.read_text()


@pytest.fixture
def graph_decoder_fake() -> tuple[PyMatchingGraphDecoder, Mock]:
    decoder = object.__new__(PyMatchingGraphDecoder)
    sdk = Mock()
    object.__setattr__(
        decoder, "graph", MatchingGraph(2, 1, (MatchingEdge((0, 1), -2.0, (0,)),))
    )
    object.__setattr__(decoder, "_matching", sdk)
    object.__setattr__(decoder, "_record_matching", sdk)
    return decoder, sdk


def test_graph_solution_keeps_both_weight_readings(
    graph_decoder_fake: tuple[PyMatchingGraphDecoder, Mock]
) -> None:
    decoder, sdk = graph_decoder_fake
    raw = torch.tensor([1, 1], dtype=torch.uint8)
    sdk.decode.return_value = (raw, -2.00000001)
    solution = decoder.decode((1, 0))
    assert solution.fault_ids == (0,) and solution.edge_indices == (0,)
    assert solution.weight == -2.0 and solution.sdk_weight == -2.00000001
    raw.zero_()
    assert solution.edge_indices == (0,)
    sdk.decode.assert_called_once_with([1, 1], return_weight=True)


@pytest.mark.parametrize(
    "raw",
    [
        [torch.tensor([1, 1], dtype=torch.uint8), -2.0],
        (torch.tensor([1, 1], dtype=torch.uint8),),
        (torch.tensor([1.0, 1.0]), -2.0),
        (torch.tensor([0, 1], dtype=torch.uint8), -2.0),
        (torch.tensor([1, 0], dtype=torch.uint8), -2.0),
        (torch.tensor([1, 1], dtype=torch.uint8), float("nan")),
        (torch.tensor([1, 1], dtype=torch.uint8), True),
        (torch.tensor([[1, 1]], dtype=torch.uint8), -2.0),
        (object(), -2.0),
    ],
)
def test_graph_invalid_solution_fails_atomically(
    graph_decoder_fake: tuple[PyMatchingGraphDecoder, Mock], raw: object
) -> None:
    decoder, sdk = graph_decoder_fake
    sdk.decode.return_value = raw
    with pytest.raises(CapabilityError):
        decoder.decode((0, 1))


def test_graph_empty_syndrome_is_not_assumed_zero_cost() -> None:
    decoder = object.__new__(PyMatchingGraphDecoder)
    graph = MatchingGraph(
        3,
        0,
        (
            MatchingEdge((0, 1), -1.0),
            MatchingEdge((1, 2), -1.0),
            MatchingEdge((0, 2), -1.0),
        ),
    )
    sdk = Mock()
    sdk.decode.return_value = (torch.ones(3, dtype=torch.uint8), -3.0)
    object.__setattr__(decoder, "graph", graph)
    object.__setattr__(decoder, "_record_matching", sdk)
    assert decoder.decode(()).weight == -3.0
    sdk.decode.assert_called_once()


def test_graph_rejects_invalid_events_before_sdk(
    graph_decoder_fake: tuple[PyMatchingGraphDecoder, Mock]
) -> None:
    decoder, sdk = graph_decoder_fake
    with pytest.raises(ValueError):
        decoder.decode((0, 0))
    with pytest.raises(ValueError):
        decoder.decode((2,))
    with pytest.raises(TypeError):
        decoder.decode((True,))
    with pytest.raises(CapabilityError):
        decoder.decode_to_matched_detectors((0, 1))
    sdk.decode.assert_not_called()
    sdk.decode_to_matched_dets_array.assert_not_called()
    error = ValueError("unreachable")
    sdk.decode.side_effect = error
    with pytest.raises(CapabilityError) as caught:
        decoder.decode((0, 1))
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    "packed_in,packed_out", [(False, False), (False, True), (True, False), (True, True)]
)
def test_graph_batch_owns_storage(
    graph_decoder_fake: tuple[PyMatchingGraphDecoder, Mock],
    monkeypatch: pytest.MonkeyPatch,
    packed_in: bool,
    packed_out: bool,
) -> None:
    decoder, sdk = graph_decoder_fake
    monkeypatch.setattr(torch.Tensor, "numpy", lambda self: self)
    bits = torch.ones((1, 1), dtype=torch.uint8)
    costs = torch.tensor([-2.0], dtype=torch.float64)

    def predict(
        buffer: torch.Tensor, **flags: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert flags == dict(
            return_weights=True,
            bit_packed_shots=packed_in,
            bit_packed_predictions=packed_out,
        )
        buffer.zero_()
        return bits, costs

    sdk.decode_batch.side_effect = predict
    events = torch.tensor([[3]] if packed_in else [[1, 1]], dtype=torch.uint8)
    before = events.clone()
    result, weights = decoder.decode_batch(
        events, bit_packed_shots=packed_in, bit_packed_predictions=packed_out
    )
    assert torch.equal(events, before)
    bits.zero_()
    costs.zero_()
    assert result.tolist() == [[1]] and weights.tolist() == [-2.0]
    sdk.decode_batch.assert_called_once()


def test_graph_missing_sdk_has_existing_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.qec import MatchingDependencyError
    from flagquantum.qec.adapters import pymatching_graph

    def missing() -> SimpleNamespace:
        raise MatchingDependencyError("controlled missing SDK")

    monkeypatch.setattr(pymatching_graph, "_pymatching", missing)
    with pytest.raises(MatchingDependencyError):
        PyMatchingGraphDecoder(MatchingGraph(1, 0, (MatchingEdge((0,), 1.0),)))
