"""Check source smoke logic and failure isolation; real installs run in package CI."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec import PyMatchingDecoder
from flagquantum.qec.dem.graph import DecodingGraph
from tools.check_qec_installed_profiles import _run
from tools.qec_installed_smoke import PROFILES, _exercise, _exercise_packed_predictions

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("profile", PROFILES)
def test_public_workflow_in_the_available_source_profile(profile: str) -> None:
    available = tuple(
        importlib.util.find_spec(name) is not None for name in ("stim", "pymatching")
    )
    if available != PROFILES[profile]:
        pytest.skip("this source environment does not have that real SDK configuration")
    packed = _exercise_packed_predictions(profile)
    assert packed == ([[0], [1], [2], [4], [7]] if available[1] else None)
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


def _packed_fake(
    monkeypatch: pytest.MonkeyPatch, observables: int = 9
) -> tuple[PyMatchingDecoder, Mock]:
    decoder = object.__new__(PyMatchingDecoder)
    sdk = Mock()
    object.__setattr__(decoder, "graph", DecodingGraph(9, observables))
    object.__setattr__(decoder, "_matching", sdk)
    # A controlled boundary stand-in keeps validation independent of NumPy/SDK.
    monkeypatch.setattr(torch.Tensor, "numpy", lambda self: self)
    return decoder, sdk


def test_packed_owns_input_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    decoder, sdk = _packed_fake(monkeypatch)

    def predict(
        buffer: torch.Tensor, *, bit_packed_shots: bool, bit_packed_predictions: bool
    ) -> torch.Tensor:
        assert bit_packed_shots and bit_packed_predictions
        buffer.fill_(0)
        return buffer

    sdk.decode_batch.side_effect = predict
    backing = torch.tensor([[1, 99, 1, 99], [2, 99, 0, 99]], dtype=torch.uint8)
    events = backing[:, ::2]
    before = backing.clone()
    result = decoder.decode_packed_batch(events)
    assert torch.equal(backing, before)
    assert result.tolist() == [[0, 0], [0, 0]]
    assert result.data_ptr() != sdk.decode_batch.call_args.args[0].data_ptr()
    sdk.decode_batch.assert_called_once()


@pytest.mark.parametrize(
    "events,exception",
    [
        (None, TypeError),
        (torch.zeros((1, 2), dtype=torch.bool), TypeError),
        (torch.zeros((1, 2), dtype=torch.int64), TypeError),
        (torch.zeros(2, dtype=torch.uint8), ValueError),
        (torch.zeros((1, 3), dtype=torch.uint8), ValueError),
        (torch.tensor([[0, 2]], dtype=torch.uint8), ValueError),
        (torch.empty((1, 2), dtype=torch.uint8, device="meta"), CapabilityError),
    ],
)
def test_packed_refuses_before_sdk(
    monkeypatch: pytest.MonkeyPatch, events: torch.Tensor, exception: type[Exception]
) -> None:
    decoder, sdk = _packed_fake(monkeypatch)
    with pytest.raises(exception):
        decoder.decode_packed_batch(events)
    sdk.decode_batch.assert_not_called()


@pytest.mark.parametrize(
    "answer",
    [
        torch.zeros((1, 2), dtype=torch.int64),
        torch.zeros((1, 1), dtype=torch.uint8),
        torch.tensor([[0, 2]], dtype=torch.uint8),
        torch.empty((1, 2), dtype=torch.uint8, device="meta"),
        object(),
    ],
)
def test_packed_rejects_malformed_sdk(
    monkeypatch: pytest.MonkeyPatch, answer: object
) -> None:
    decoder, sdk = _packed_fake(monkeypatch)
    sdk.decode_batch.return_value = answer
    with pytest.raises(CapabilityError):
        decoder.decode_packed_batch(torch.zeros((1, 2), dtype=torch.uint8))


def test_packed_empty_and_zero_observables(monkeypatch: pytest.MonkeyPatch) -> None:
    decoder, sdk = _packed_fake(monkeypatch, observables=0)
    assert decoder.decode_packed_batch(
        torch.empty((0, 2), dtype=torch.uint8)
    ).shape == (0, 0)
    sdk.decode_batch.assert_not_called()
    sdk.decode_batch.return_value = torch.empty((2, 0), dtype=torch.uint8)
    assert decoder.decode_packed_batch(
        torch.zeros((2, 2), dtype=torch.uint8)
    ).shape == (2, 0)
    sdk.decode_batch.assert_called_once()


def test_packed_sdk_error_retains_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    decoder, sdk = _packed_fake(monkeypatch)
    error = ValueError("no perfect matching")
    sdk.decode_batch.side_effect = error
    with pytest.raises(CapabilityError) as caught:
        decoder.decode_packed_batch(torch.zeros((1, 2), dtype=torch.uint8))
    assert caught.value.__cause__ is error
