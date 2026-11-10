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
from flagquantum.qec import PyMatchingDecoder
from flagquantum.qec.adapters.pymatching_graph import PyMatchingGraphDecoder
from flagquantum.qec.decoders.matching_graph import MatchingEdge, MatchingGraph
from flagquantum.qec.dem.graph import DecodingGraph
from tools.check_qec_installed_profiles import _run
from tools.qec_installed_smoke import (
    PROFILES,
    _embedded_profile,
    _exercise,
    _exercise_compiled_measurement_sampler,
    _exercise_decoder_records,
    _exercise_matched_detectors,
    _exercise_matching_graph,
    _exercise_packed_predictions,
    _exercise_sdk_weights,
    _exercise_stim_conversion,
    _exercise_stim_experiment,
    _exercise_terminal_conversion,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("profile", PROFILES)
def test_public_workflow_in_the_available_source_profile(profile: str) -> None:
    available = tuple(
        importlib.util.find_spec(name) is not None for name in ("stim", "pymatching")
    )
    if available != PROFILES[profile]:
        pytest.skip("this source environment does not have that real SDK configuration")
    assert _exercise_packed_predictions(profile) == (
        [[0], [1], [2], [4], [7]] if available[1] else None
    )
    assert (_exercise_sdk_weights(profile) is not None) == available[1]
    assert _exercise_matched_detectors(profile) == (
        [[0, None], [1, 2]] if available[1] else None
    )
    assert (_exercise_stim_conversion(profile) is not None) == available[0]
    assert (_exercise_terminal_conversion(profile) is not None) == available[0]
    assert (_exercise_compiled_measurement_sampler(profile) is not None) == (
        available[0] or _embedded_profile()
    )
    records = _exercise(profile)
    assert len(records) == (4 if available[0] else 0)
    for record in records:
        assert record["shots"] == 35
        assert record["observables"] == 1
    decoders = _exercise_decoder_records(profile)
    assert len(decoders) == (5 if available[1] else 4)
    experiment = _exercise_stim_experiment(profile)
    if not available[0]:
        assert experiment is None
    else:
        assert experiment is not None
        assert experiment["shots"] == 35
        assert experiment["logical_failures"] == 0
        assert experiment["observable_failures"] == [0, 0]
        assert experiment["sampling_source"] == "stim_circuit"
        assert experiment["reference_ones_verified"] is True
        assert [record["entrance"] for record in experiment["memory_experiments"]] == [
            "repetition_z",
            "surface_z",
            "surface_x",
        ]
        assert all(
            record["shots"] == 35 and record["sampling_source"] == "memory_circuit"
            for record in experiment["memory_experiments"]
        )
    graph_report = _exercise_matching_graph(profile)
    assert (graph_report is not None) == available[1]


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
    weighted_fake: tuple[PyMatchingDecoder, Mock],
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


@pytest.fixture
def matched_fake() -> tuple[PyMatchingDecoder, Mock]:
    decoder = object.__new__(PyMatchingDecoder)
    sdk = Mock()
    object.__setattr__(decoder, "graph", DecodingGraph(3, 1))
    object.__setattr__(decoder, "_matching", sdk)
    return decoder, sdk


def test_matched_pairs_cover_events_and_own_values(
    matched_fake: tuple[PyMatchingDecoder, Mock],
) -> None:
    decoder, sdk = matched_fake
    raw = torch.tensor([[2, 1], [0, -1]], dtype=torch.int64)
    sdk.decode_to_matched_dets_array.return_value = raw
    pairs = decoder.decode_to_matched_detectors(iter((2, 0, 1)))
    assert pairs == ((0, None), (1, 2))
    raw.fill_(-1)
    assert pairs == ((0, None), (1, 2))
    sdk.decode_to_matched_dets_array.assert_called_once_with([1, 1, 1])


@pytest.mark.parametrize(
    "raw",
    [
        torch.tensor([0, -1]),
        torch.tensor([[0.0, -1.0]]),
        torch.tensor([[True, False]]),
        torch.tensor([[-1, 0]]),
        torch.tensor([[0, 0], [1, 2]]),
        torch.tensor([[0, -2], [1, 2]]),
        torch.tensor([[0, -1], [1, 3]]),
        torch.tensor([[0, -1]]),
        torch.tensor([[0, -1], [0, -1], [1, 2]]),
        torch.empty((2, 2), dtype=torch.int64, device="meta"),
        object(),
    ],
)
def test_matched_malformed_result_is_atomic(
    matched_fake: tuple[PyMatchingDecoder, Mock], raw: object
) -> None:
    decoder, sdk = matched_fake
    sdk.decode_to_matched_dets_array.return_value = raw
    with pytest.raises(CapabilityError):
        decoder.decode_to_matched_detectors((0, 1, 2))


def test_matched_validation_and_failure(
    matched_fake: tuple[PyMatchingDecoder, Mock],
) -> None:
    decoder, sdk = matched_fake
    assert decoder.decode_to_matched_detectors(()) == ()
    with pytest.raises(ValueError):
        decoder.decode_to_matched_detectors((0, 0))
    with pytest.raises(ValueError):
        decoder.decode_to_matched_detectors((3,))
    with pytest.raises(TypeError):
        decoder.decode_to_matched_detectors((True,))
    sdk.decode_to_matched_dets_array.assert_not_called()
    error = ValueError("no matching")
    sdk.decode_to_matched_dets_array.side_effect = error
    with pytest.raises(CapabilityError) as caught:
        decoder.decode_to_matched_detectors((0,))
    assert caught.value.__cause__ is error


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
    graph_decoder_fake: tuple[PyMatchingGraphDecoder, Mock],
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
    graph_decoder_fake: tuple[PyMatchingGraphDecoder, Mock],
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
    from flagquantum.qec.adapters import pymatching as owner

    def missing() -> SimpleNamespace:
        raise MatchingDependencyError("controlled missing SDK")

    monkeypatch.setattr(owner, "_pymatching", missing)
    with pytest.raises(MatchingDependencyError):
        PyMatchingGraphDecoder(MatchingGraph(1, 0, (MatchingEdge((0,), 1.0),)))


@pytest.mark.parametrize("profile", ("core", "stim", "pymatching", "both"))
def test_native_detector_installed_profile(profile: str) -> None:
    from tools.qec_installed_smoke import _exercise_native_detector_execution

    if _embedded_profile() or profile in ("stim", "both"):
        if not _embedded_profile():
            pytest.importorskip("stim")
        result = _exercise_native_detector_execution(profile)
        assert result == {
            "packed_detector_observable": [[3], [3]],
            "mechanisms": 1,
            "detector_file_sizes": {
                "01": 192,
                "b8": 64,
                "r8": 192,
                "hits": 256,
                "dets": 704,
                "ptb64": 16,
            },
        }
    else:
        assert _exercise_native_detector_execution(profile) is None


@pytest.mark.parametrize("embedded", (False, True))
def test_core_sampler_success_requires_the_explicit_embedded_profile(
    monkeypatch: pytest.MonkeyPatch,
    embedded: bool,
) -> None:
    from flagquantum.dynamic import DynamicCircuit
    from tools import qec_installed_smoke

    class Sampler:
        def sample(self, shots: int, *, bit_packed: bool) -> torch.Tensor:
            return torch.ones((shots, 1), dtype=torch.uint8)

        def sample_write(self, shots: int, *, filepath: Path, format: str) -> None:
            data = {
                "01": b"10\n" * shots,
                "b8": b"\x01" * shots,
                "r8": b"\x00\x01" * shots,
                "hits": b"0\n" * shots,
                "dets": b"shot M0\n" * shots,
                "ptb64": (b"\xff" * 8 + b"\x00" * 8) * (shots // 64),
            }
            filepath.write_bytes(data[format])

    monkeypatch.setattr(qec_installed_smoke, "_embedded_profile", lambda: embedded)
    monkeypatch.setattr(
        DynamicCircuit, "compile_sampler", lambda self, **kwargs: Sampler()
    )
    if embedded:
        assert _exercise_compiled_measurement_sampler("core") == {
            "packed_samples": [[1], [1]],
            "b8_hex": "0101",
            "file_format_sizes": {
                "01": 192,
                "b8": 64,
                "r8": 128,
                "hits": 128,
                "dets": 512,
                "ptb64": 16,
            },
        }
    else:
        with pytest.raises(AssertionError, match="ignored missing Stim"):
            _exercise_compiled_measurement_sampler("core")


@pytest.mark.parametrize("profile", ("core", "stim", "pymatching", "both"))
def test_measurement_file_conversion_installed_profile(profile: str) -> None:
    from tools.qec_installed_smoke import _exercise_measurement_file_conversion

    if _embedded_profile() or profile in ("stim", "both"):
        if not _embedded_profile():
            pytest.importorskip("stim")
        result = _exercise_measurement_file_conversion(profile)
        assert result == {
            "shots_per_case": 64,
            "input_formats": ["01", "b8", "r8", "hits", "dets", "ptb64"],
            "output_formats": ["01", "b8", "r8", "hits", "dets"],
            "byte_cases": 30,
            "occurrence_columns": 3,
            "appended_and_separate": True,
        }
    else:
        assert _exercise_measurement_file_conversion(profile) is None


def test_file_conversion_profile_failure_is_not_silently_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum import Circuit
    from tools import qec_installed_smoke

    monkeypatch.setattr(qec_installed_smoke, "_embedded_profile", lambda: True)

    def failed(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected converter failure")

    monkeypatch.setattr(Circuit, "compile_m2d_converter", failed)
    with pytest.raises(RuntimeError, match="injected converter failure"):
        qec_installed_smoke._exercise_measurement_file_conversion("core")
