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
from tools.qec_installed_smoke import PROFILES, _exercise, _exercise_sdk_weights

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("profile", PROFILES)
def test_public_workflow_in_the_available_source_profile(profile: str) -> None:
    available = tuple(
        importlib.util.find_spec(name) is not None for name in ("stim", "pymatching")
    )
    if available != PROFILES[profile]:
        pytest.skip("this source environment does not have that real SDK configuration")
    weighted = _exercise_sdk_weights(profile)
    assert (weighted is not None) == available[1]
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
def weighted_fake(monkeypatch: pytest.MonkeyPatch) -> tuple[PyMatchingDecoder, Mock]:
    decoder = object.__new__(PyMatchingDecoder)
    sdk = Mock()
    object.__setattr__(decoder, "graph", DecodingGraph(9, 9))
    object.__setattr__(decoder, "_matching", sdk)
    monkeypatch.setattr(torch.Tensor, "numpy", lambda self: self)
    return decoder, sdk


@pytest.mark.parametrize(
    "packed_in,packed_out", [(False, False), (False, True), (True, False), (True, True)]
)
def test_weighted_pair_is_owned_and_single_call(
    weighted_fake: tuple[PyMatchingDecoder, Mock], packed_in: bool, packed_out: bool
) -> None:
    decoder, sdk = weighted_fake
    columns = 2 if packed_out else 9
    bits = torch.zeros((2, columns), dtype=torch.uint8)
    weights = torch.tensor([0.0, 1.0], dtype=torch.float64)

    def predict(
        buffer: torch.Tensor, **options: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert options == dict(
            return_weights=True,
            bit_packed_shots=packed_in,
            bit_packed_predictions=packed_out,
        )
        buffer.zero_()
        return bits, weights

    sdk.decode_batch.side_effect = predict
    backing = torch.ones((2, 4 if packed_in else 18), dtype=torch.uint8)
    events = backing[:, ::2]
    before = backing.clone()
    output, costs = decoder.decode_batch_with_weights(
        events, bit_packed_shots=packed_in, bit_packed_predictions=packed_out
    )
    assert torch.equal(before, backing)
    bits.fill_(1)
    weights.fill_(-1)
    assert not bool(output.any()) and costs.tolist() == [0.0, 1.0]
    sdk.decode_batch.assert_called_once()


@pytest.mark.parametrize(
    "case",
    [
        "pair",
        "arity",
        "bits_shape",
        "bits_dtype",
        "bits_padding",
        "weights_shape",
        "weights_dtype",
        "nan",
        "negative",
        "device",
    ],
)
def test_weighted_malformed_pair_is_atomic(
    weighted_fake: tuple[PyMatchingDecoder, Mock], case: str
) -> None:
    decoder, sdk = weighted_fake
    bits = torch.zeros((1, 2), dtype=torch.uint8)
    costs = torch.zeros(1, dtype=torch.float64)
    raw: object = (bits, costs)
    if case == "pair":
        raw = [bits, costs]
    elif case == "arity":
        raw = (bits,)
    elif case == "bits_shape":
        raw = (bits[:, :1], costs)
    elif case == "bits_dtype":
        raw = (bits.bool(), costs)
    elif case == "bits_padding":
        bits[0, 1] = 2
    elif case == "weights_shape":
        raw = (bits, costs.reshape(1, 1))
    elif case == "weights_dtype":
        raw = (bits, costs.float())
    elif case == "nan":
        costs.fill_(float("nan"))
    elif case == "negative":
        costs.fill_(-1)
    elif case == "device":
        raw = (bits, torch.empty(1, dtype=torch.float64, device="meta"))
    sdk.decode_batch.return_value = raw
    with pytest.raises(CapabilityError):
        decoder.decode_batch_with_weights(
            torch.zeros((1, 2), dtype=torch.uint8),
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )


def test_weighted_empty_and_failure_semantics(
    weighted_fake: tuple[PyMatchingDecoder, Mock]
) -> None:
    decoder, sdk = weighted_fake
    bits, costs = decoder.decode_batch_with_weights(
        torch.empty((0, 9), dtype=torch.uint8)
    )
    assert bits.shape == (0, 9) and costs.shape == (0,)
    sdk.decode_batch.assert_not_called()
    with pytest.raises(ValueError):
        decoder.decode_batch_with_weights(
            torch.tensor([[0, 2]], dtype=torch.uint8), bit_packed_shots=True
        )
    sdk.decode_batch.assert_not_called()
    error = ValueError("unreachable")
    sdk.decode_batch.side_effect = error
    with pytest.raises(CapabilityError) as caught:
        decoder.decode_batch_with_weights(torch.zeros((1, 9), dtype=torch.uint8))
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("flag", [0, 1, None, "true"])
def test_weighted_flags_are_strict_bools(
    weighted_fake: tuple[PyMatchingDecoder, Mock], flag: bool
) -> None:
    decoder, sdk = weighted_fake
    with pytest.raises(TypeError, match="bool"):
        decoder.decode_batch_with_weights(
            torch.zeros((1, 9), dtype=torch.uint8), bit_packed_shots=flag
        )
    with pytest.raises(TypeError, match="bool"):
        decoder.decode_batch_with_weights(
            torch.zeros((1, 9), dtype=torch.uint8), bit_packed_predictions=flag
        )
    sdk.decode_batch.assert_not_called()


@pytest.mark.parametrize("packed", [False, True])
def test_weighted_empty_layout_shapes(
    weighted_fake: tuple[PyMatchingDecoder, Mock], packed: bool
) -> None:
    decoder, sdk = weighted_fake
    bits, weights = decoder.decode_batch_with_weights(
        torch.empty((0, 2 if packed else 9), dtype=torch.uint8),
        bit_packed_shots=packed,
        bit_packed_predictions=packed,
    )
    assert bits.shape == (0, 2 if packed else 9)
    assert bits.dtype == (torch.uint8 if packed else torch.bool)
    assert weights.shape == (0,) and weights.dtype == torch.float64
    sdk.decode_batch.assert_not_called()
