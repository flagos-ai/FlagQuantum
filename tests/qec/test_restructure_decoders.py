"""Decoder and adapter relocation against trusted pre-006 behavior."""

from __future__ import annotations

import dataclasses
import importlib
import json
import pickle
from pathlib import Path

import pytest
import torch
from _retirement import (
    assert_metadata,
    assert_namespace,
    assert_retired_payload,
    resolve,
)

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
    return resolve(path)


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


def test_retired_decoder_global_pickles_refuse_and_current_globals_roundtrip() -> None:
    assert_retired_payload("restructure_006_globals.pkl")
    metadata = json.loads((_FIXTURES / "restructure_006_metadata.json").read_text())
    for path in metadata["definitions"]:
        if len(path.split(".")) == 4:
            value = _resolve(path)
            assert pickle.loads(pickle.dumps(value, protocol=4)) is value, path


def test_decoder_metadata_and_surviving_namespaces_follow_the_retirement_boundary() -> (
    None
):
    expected = json.loads((_FIXTURES / "restructure_006_metadata.json").read_text())
    for path, metadata in expected["definitions"].items():
        assert_metadata(path, metadata[0], metadata[1])
    for old, names in expected["namespaces"].items():
        assert_namespace(old, names)


def test_decoder_numerical_carrier_and_streaming_baseline() -> None:
    expected = json.loads((_FIXTURES / "restructure_006_results.json").read_text())
    expected.pop("adapter")
    actual = _scenarios()
    old_module, qualname = expected.pop("decorator")
    assert old_module == "flagquantum.qec.registry"
    assert actual.pop("decorator") == ["flagquantum.qec.decoders.registry", qualname]
    assert actual == expected


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


def test_decoder_checkpoints_resume_with_the_same_classes_and_results() -> None:
    assert_retired_payload("restructure_006_instances.pkl")
    evidence = (
        Path(__file__).with_name("retirement_fixtures") / "surviving_instances.pkl"
    )
    restored = pickle.loads(evidence.read_bytes())["restructure_006_instances.pkl"]
    expected = json.loads((_FIXTURES / "restructure_006_results.json").read_text())
    for path, value in restored.items():
        assert type(value) is _resolve(path), path
        assert type(pickle.loads(pickle.dumps(value))) is type(value), path
        name = type(value).__name__
        if name == "RepetitionLookupDecoder":
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
        else:
            assert name == "MatchingDependencyError"
    for name in (
        qec.AUTHORITY_NAME,
        qec.BELIEF_PROPAGATION_NAME,
        qec.BELIEF_PROPAGATION_OSD_NAME,
        qec.SLIDING_WINDOW_NAME,
    ):
        original = qec.get_decoder(name, _model())
        current = pickle.loads(pickle.dumps(original, protocol=4))
        assert type(current) is type(original)
        for syndrome in ((), (0,), (1,), (0, 1)):
            assert _plain(current.decode(syndrome)) == _plain(original.decode(syndrome))
    stream = qec.SlidingWindowDecoder(_stream_spec(), qec.BELIEF_PROPAGATION_NAME)
    assert _plain(stream.decode_round((0,))) == expected["stream"][0]
    resumed = pickle.loads(pickle.dumps(stream, protocol=4))
    assert type(resumed) is type(stream)
    assert [_plain(resumed.decode_round(())) for _ in range(3)] == expected["stream"][
        1:
    ]


def test_historical_decoder_imports_are_canonical_aliases() -> None:
    owners = {
        "decoders": "decoders.repetition",
        "registry": "decoders.registry",
        "matching": "decoders.matching",
        "belief_propagation": "decoders.belief_propagation",
        "bposd": "decoders.bposd",
        "sliding_window": "decoders.sliding_window",
        "adapters": "adapters.pymatching",
    }
    expected = json.loads((_FIXTURES / "restructure_006_metadata.json").read_text())
    for path in expected["definitions"]:
        parts = path.split(".")
        if len(parts) != 4:
            continue
        owner = importlib.import_module("flagquantum.qec." + owners[parts[2]])
        assert _resolve(path) is getattr(owner, parts[3]), path


def test_one_registry_serves_root_and_canonical_plugin_imports() -> None:
    from flagquantum.qec.decoders import registry as owner

    qec.decoder_names()
    table = owner._DECODERS
    assert qec.register_decoder is owner.register_decoder
    original = dict(table)
    try:
        assert (
            owner.register_decoder("package_identity_probe")(
                qec.MinimumWeightMatchingDecoder
            )
            is qec.MinimumWeightMatchingDecoder
        )
        for get in (qec.get_decoder, owner.get_decoder):
            assert (
                type(get("package_identity_probe", _model()))
                is qec.MinimumWeightMatchingDecoder
            )
        with pytest.raises(ValueError, match="already registered"):
            owner.register_decoder("package_identity_probe")
    finally:
        table.clear()
        table.update(original)
