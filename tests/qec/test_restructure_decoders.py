"""Decoder and adapter relocation against trusted pre-006 behavior."""

from __future__ import annotations

import dataclasses
import importlib
import json
import pickle
from pathlib import Path

import pytest
import torch

import flagquantum.qec as qec
from flagquantum.errors import CapabilityError

pytestmark = pytest.mark.unit
_FIXTURES = Path(__file__).with_name("fixtures")


def _plain(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "values": value.tolist()}
    if dataclasses.is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _resolve(path: str) -> object:
    parts = path.split(".")
    value = importlib.import_module(".".join(parts[:3]))
    for part in parts[3:]:
        value = getattr(value, part)
    return value


def _model() -> qec.DetectorErrorModel:
    return qec.DetectorErrorModel(
        2,
        1,
        (
            qec.DemError(0.125, (0,), (0,)),
            qec.DemError(0.25, (0, 1)),
            qec.DemError(0.0625, (1,)),
        ),
    )


def _history() -> tuple[qec.SyndromeRound, ...]:
    return (
        qec.SyndromeRound(
            0, (1, 1), (qec.DetectionEvent(0, 0), qec.DetectionEvent(0, 1))
        ),
        qec.SyndromeRound(1, (1, 1)),
    )


def _stream_spec() -> qec.DemChunksSpec:
    model = qec.DetectorErrorModel(
        4,
        1,
        (
            qec.DemError(0.125, (0,), (0,)),
            qec.DemError(0.0625, (1,)),
            qec.DemError(0.125, (2,)),
            qec.DemError(0.0625, (3,)),
            qec.DemError(0.03125, (1, 2)),
        ),
    )
    return qec.DemChunksSpec(qec.ChunkLayout(model, (1, 1, 1, 1)), window=2)


def _scenarios() -> dict[str, object]:
    model = _model()
    graph = qec.DecodingGraph.from_detector_error_model(model)
    sources = {"model": model, "text": model.to_stim_text(), "graph": graph}
    result: dict[str, object] = {}
    for name in (
        qec.AUTHORITY_NAME,
        qec.BELIEF_PROPAGATION_NAME,
        qec.BELIEF_PROPAGATION_OSD_NAME,
        qec.SLIDING_WINDOW_NAME,
    ):
        for carrier, source in sources.items():
            key = name + ":" + carrier
            try:
                decoder = qec.get_decoder(name, source)
            except CapabilityError as error:
                result[key] = [type(error).__name__, str(error)]
                continue
            result[key] = [
                _plain(decoder.decode(syndrome))
                for syndrome in ((), (0,), (1,), (0, 1))
            ]
    hypergraph = qec.DetectorErrorModel(
        3,
        1,
        (
            qec.DemError(0.125, (0, 1, 2), (0,)),
            qec.DemError(0.25, (0,)),
            qec.DemError(0.0625, (1, 2)),
        ),
    )
    for name in (qec.BELIEF_PROPAGATION_NAME, qec.BELIEF_PROPAGATION_OSD_NAME):
        decoder = qec.get_decoder(name, hypergraph)
        result[name + ":hyperedges"] = [
            _plain(decoder.decode(syndrome))
            for syndrome in ((), (0,), (1, 2), (0, 1, 2))
        ]
    history = _history()
    result["repetition:lookup"] = _plain(qec.RepetitionLookupDecoder().decode(history))
    result["repetition:streaming"] = _plain(
        qec.RepetitionStreamingLookupDecoder().decode_round(history)
    )
    result["repetition:temporal"] = _plain(
        qec.RepetitionTemporalDecoder().decode_round(history)
    )
    stream = qec.SlidingWindowDecoder(_stream_spec(), qec.BELIEF_PROPAGATION_NAME)
    result["stream"] = [
        _plain(stream.decode_round(events)) for events in ((0,), (), (), ())
    ]
    decorator = qec.register_decoder("metadata_probe_without_registration")
    result["decorator"] = [decorator.__module__, decorator.__qualname__]
    return result


def test_old_decoder_global_pickles_and_typevar_keep_exact_identities() -> None:
    # Repository-owned pre-move fixture, never an external pickle.
    restored = pickle.loads((_FIXTURES / "restructure_006_globals.pkl").read_bytes())
    for path, value in restored.items():
        assert value is _resolve(path), path
        assert pickle.loads(pickle.dumps(value, protocol=4)) is value, path


def test_decoder_metadata_and_historical_namespaces_remain_identical() -> None:
    expected = json.loads((_FIXTURES / "restructure_006_metadata.json").read_text())
    for path, metadata in expected["definitions"].items():
        value = _resolve(path)
        if isinstance(value, property):
            value = value.fget
        name = getattr(value, "__qualname__", getattr(value, "__name__", None))
        assert [value.__module__, name] == metadata, path
    for old, names in expected["namespaces"].items():
        module = importlib.import_module("flagquantum.qec." + old)
        for name in names:
            assert hasattr(module, name), (old, name)


def test_decoder_numerical_carrier_and_streaming_baseline() -> None:
    expected = json.loads((_FIXTURES / "restructure_006_results.json").read_text())
    expected.pop("adapter")
    assert _scenarios() == expected


def _adapter_scenario() -> dict[str, object]:
    decoder = qec.get_decoder(qec.CROSS_CHECK_NAME, _model())
    result = {
        "results": [
            _plain(decoder.decode(syndrome)) for syndrome in ((), (0,), (1,), (0, 1))
        ],
    }
    try:
        restored = pickle.loads(pickle.dumps(decoder))
    except TypeError as error:
        result["pickle"] = type(error).__name__
    else:
        result["pickle"] = type(restored).__module__
    return result


def test_adapter_results_and_existing_pickle_capability() -> None:
    from flagquantum.qec.adapters import MatchingDependencyError

    if importlib.util.find_spec("pymatching") is None:
        with pytest.raises(MatchingDependencyError, match="optional dependency"):
            qec.get_decoder(qec.CROSS_CHECK_NAME, _model())
        return
    expected = json.loads((_FIXTURES / "restructure_006_results.json").read_text())
    assert _adapter_scenario() == expected["adapter"]


def test_old_decoder_instances_resume_with_the_same_classes_and_results() -> None:
    restored = pickle.loads((_FIXTURES / "restructure_006_instances.pkl").read_bytes())
    expected = json.loads((_FIXTURES / "restructure_006_results.json").read_text())
    for path, value in restored.items():
        assert type(value) is _resolve(path), path
        assert type(pickle.loads(pickle.dumps(value))) is type(value), path
        name = type(value).__name__
        if name == "SlidingWindowDecoder":
            assert [_plain(value.decode_round(())) for _ in range(3)] == expected[
                "stream"
            ][1:]
        elif name == "RepetitionLookupDecoder":
            assert _plain(value.decode(_history())) == expected["repetition:lookup"]
        elif name in ("RepetitionStreamingLookupDecoder", "RepetitionTemporalDecoder"):
            key = (
                "streaming"
                if name == "RepetitionStreamingLookupDecoder"
                else "temporal"
            )
            assert (
                _plain(value.decode_round(_history())) == expected["repetition:" + key]
            )
        elif name != "MatchingDependencyError":
            original = qec.get_decoder(
                {
                    "MinimumWeightMatchingDecoder": qec.AUTHORITY_NAME,
                    "BeliefPropagationDecoder": qec.BELIEF_PROPAGATION_NAME,
                    "BeliefPropagationOsdDecoder": qec.BELIEF_PROPAGATION_OSD_NAME,
                    "SlidingWindowMatchingDecoder": qec.SLIDING_WINDOW_NAME,
                }[name],
                _model(),
            )
            assert _plain(value.decode((0,))) == _plain(original.decode((0,)))
