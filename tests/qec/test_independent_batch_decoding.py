"""Complete records, independent shots and unchanged decoder refusals.

Core lanes exercise all four built-in algorithms. SDK lanes additionally
parameterize the real PyMatching adapter; its absence is checked separately.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from typing import Any

import pytest
import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec import DetectorErrorModel, get_decoder, register_decoder
from flagquantum.qec.adapters.pymatching import MatchingDependencyError
from flagquantum.qec.decoders.batch import decode_batch
from flagquantum.qec.decoders.belief_propagation import BeliefPropagationDecoder
from flagquantum.qec.decoders.bposd import BeliefPropagationOsdDecoder
from flagquantum.qec.decoders.matching import MinimumWeightMatchingDecoder
from flagquantum.qec.decoders.sliding_window import SlidingWindowDecoder
from flagquantum.qec.dem import DemError
from flagquantum.qec.dem.chunks import ChunkLayout, DemChunksSpec

pytestmark = pytest.mark.unit

_CORE_NAMES = (
    "minimum_weight_matching",
    "sliding_window_matching",
    "belief_propagation",
    "belief_propagation_osd",
)
_NAMES = _CORE_NAMES + (
    ("pymatching",) if importlib.util.find_spec("pymatching") is not None else ()
)


def _model() -> DetectorErrorModel:
    return DetectorErrorModel.from_stim_text(
        "error(0.125) D0 L0\nerror(0.2) D1\ndetector D0\ndetector D1\n"
        "logical_observable L0\n"
    )


@pytest.fixture(params=_NAMES)
def decoder(request: pytest.FixtureRequest) -> Any:
    return get_decoder(request.param, _model())


def test_full_records_match_direct_calls_and_an_independent_answer(
    decoder: Any,
) -> None:
    shots = ((), (0,), (1,), (1, 0))
    expected = tuple(decoder.decode(row) for row in shots)
    actual = decode_batch(decoder, shots)
    assert actual == expected
    assert tuple(result.observables for result in actual) == ((), (0,), (), (0,))
    assert tuple(type(result) for result in actual) == tuple(
        type(result) for result in expected
    )
    assert actual[1].weight > 0
    if hasattr(actual[1], "error_edges"):
        assert actual[1].error_edges
    else:
        assert actual[1].mechanisms == (0,)
        assert isinstance(actual[1].converged, bool)
        assert actual[1].iterations >= 1


def test_empty_and_single_shot_batches(decoder: Any) -> None:
    assert decode_batch(decoder, ()) == ()
    assert decode_batch(decoder, ((0,),)) == (decoder.decode((0,)),)


def test_inputs_are_consumed_once_and_caller_lists_remain_independent(
    decoder: Any,
) -> None:
    visits: list[tuple[str, int]] = []
    source = [[], [0], [1]]

    def row(index: int) -> Iterable[int]:
        visits.append(("row", index))
        yield from source[index]

    def shots() -> Iterable[Iterable[int]]:
        for index in range(len(source)):
            visits.append(("outer", index))
            yield row(index)

    result = decode_batch(decoder, shots())
    assert result == tuple(decoder.decode(item) for item in source)
    assert visits == [
        ("outer", 0),
        ("row", 0),
        ("outer", 1),
        ("row", 1),
        ("outer", 2),
        ("row", 2),
    ]
    assert source == [[], [0], [1]]
    source[1].append(1)
    assert result[1] == decoder.decode((0,))


@pytest.mark.parametrize("invalid", [(True,), (0.5,), (-1,), (2,), "0"])
def test_invalid_late_row_causes_no_decode_calls(
    decoder: Any, monkeypatch: pytest.MonkeyPatch, invalid: Any
) -> None:
    calls: list[object] = []

    def unexpected(self: Any, events: object) -> Any:
        calls.append(events)
        raise AssertionError("decoding began before complete input validation")

    monkeypatch.setattr(type(decoder), "decode", unexpected)
    with pytest.raises((TypeError, ValueError)):
        decode_batch(decoder, ((), invalid))
    assert calls == []


def test_duplicate_policy_is_the_single_shot_policy(decoder: Any) -> None:
    if type(decoder) is BeliefPropagationDecoder:
        assert decode_batch(decoder, ((0, 0),)) == (decoder.decode((0,)),)
    else:
        with pytest.raises(ValueError, match="same detector twice"):
            decode_batch(decoder, ((), (0, 0)))


@pytest.mark.parametrize("dense", [torch.zeros((0, 2)), (torch.tensor([0, 1]),)])
def test_dense_carriers_have_explicit_guidance(decoder: Any, dense: Any) -> None:
    with pytest.raises(TypeError, match="PyMatchingDecoder.decode_batch"):
        decode_batch(decoder, dense)


def test_failed_input_iterator_does_not_begin_decoding(
    decoder: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: Any) -> Any:
        raise AssertionError("decode called before input iterator completed")

    def broken() -> Iterable[Iterable[int]]:
        yield ()
        raise RuntimeError("input iterator failed")

    monkeypatch.setattr(type(decoder), "decode", unexpected)
    with pytest.raises(RuntimeError, match="input iterator failed"):
        decode_batch(decoder, broken())


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("perturbation", [0.0, 1e-14])
def test_tied_and_near_tied_cycles_retain_each_decoders_own_records(
    name: str, perturbation: float
) -> None:
    model = DetectorErrorModel(
        num_detectors=4,
        num_observables=1,
        errors=(
            DemError(probability=0.125, detectors=(0, 1), observables=(0,)),
            DemError(probability=0.125, detectors=(1, 2)),
            DemError(probability=0.125, detectors=(2, 3)),
            DemError(probability=0.125 + perturbation, detectors=(0, 3)),
        ),
    )
    decoder = get_decoder(name, model)
    shots = ((), (0, 1, 2, 3), (0, 2))
    assert decode_batch(decoder, shots) == tuple(decoder.decode(row) for row in shots)


def test_subclasses_and_unknown_plugins_are_refused_even_for_empty_input() -> None:
    class Derived(MinimumWeightMatchingDecoder):
        pass

    class Plugin:
        def decode(self, events: Iterable[int]) -> Any:
            raise AssertionError("plugin was called")

    for unsupported in (
        Derived(get_decoder("minimum_weight_matching", _model()).graph),
        Plugin(),
    ):
        for shots in ((), ((0,),)):
            with pytest.raises(CapabilityError, match="certified concrete"):
                decode_batch(unsupported, shots)
    for invalid in (object(), None, MinimumWeightMatchingDecoder):
        with pytest.raises(TypeError, match="decoder instance"):
            decode_batch(invalid, ())


def test_registered_single_shot_plugin_keeps_its_existing_route() -> None:
    class Plugin:
        @classmethod
        def from_detector_error_model(cls, model: DetectorErrorModel) -> Plugin:
            return cls()

        @classmethod
        def from_decoding_graph(cls, graph: Any) -> Plugin:
            return cls()

        def decode(self, events: Iterable[int]) -> Any:
            return get_decoder("belief_propagation", _model()).decode(events)

    try:
        register_decoder("minimum_weight_matching", replace=True)(Plugin)
        plugin = get_decoder("minimum_weight_matching", _model())
        assert plugin.decode((0,)).observables == (0,)
        with pytest.raises(CapabilityError):
            decode_batch(plugin, ((0,),))
        assert plugin.decode((0,)).observables == (0,)
    finally:
        register_decoder("minimum_weight_matching", replace=True)(
            MinimumWeightMatchingDecoder
        )


def test_partially_consumed_stateful_stream_is_untouched() -> None:
    model = DetectorErrorModel(
        num_detectors=4,
        num_observables=0,
        errors=tuple(
            DemError(probability=0.1, detectors=(index, index + 1))
            for index in range(3)
        ),
    )
    stream = SlidingWindowDecoder(
        DemChunksSpec(layout=ChunkLayout(model=model, widths=(1, 1, 1, 1)), window=2),
        "belief_propagation",
    )
    stream.decode_round((0,))
    before = (
        dict(stream._buffer),
        list(stream._mods),
        set(stream._faults),
        stream._next_layer,
        stream._next_window,
    )
    for shots in ((), ((0,),)):
        with pytest.raises(CapabilityError):
            decode_batch(stream, shots)
    assert (
        stream._buffer,
        stream._mods,
        stream._faults,
        stream._next_layer,
        stream._next_window,
    ) == before


@pytest.mark.parametrize("name", _NAMES)
def test_algorithmic_failure_propagates_without_a_partial_result(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    model = DetectorErrorModel(
        num_detectors=2,
        num_observables=0,
        errors=(DemError(probability=0.1, detectors=(0, 1)),),
    )
    decoder = get_decoder(name, model)
    original = type(decoder).decode
    calls: list[tuple[int, ...]] = []

    def tracked(self: Any, events: Iterable[int]) -> Any:
        row = tuple(events)
        calls.append(row)
        return original(self, row)

    monkeypatch.setattr(type(decoder), "decode", tracked)
    with pytest.raises(CapabilityError):
        decode_batch(decoder, ((), (0,)))
    assert calls == [(), (0,)]


def test_defect_budget_and_hyperedge_domains_are_preserved() -> None:
    matcher = MinimumWeightMatchingDecoder.from_detector_error_model(
        _model(), max_defects=1
    )
    with pytest.raises(CapabilityError, match="at most"):
        decode_batch(matcher, ((), (0, 1)))
    model = DetectorErrorModel(
        num_detectors=3,
        num_observables=1,
        errors=(DemError(probability=0.1, detectors=(0, 1, 2), observables=(0,)),),
    )
    for cls in (BeliefPropagationDecoder, BeliefPropagationOsdDecoder):
        decoder = cls.from_detector_error_model(model)
        assert decode_batch(decoder, ((), (0, 1, 2))) == (
            decoder.decode(()),
            decoder.decode((0, 1, 2)),
        )
    with pytest.raises(CapabilityError):
        MinimumWeightMatchingDecoder.from_detector_error_model(model)


def test_nonconvergence_record_and_disabled_fallback_are_preserved() -> None:
    model = DetectorErrorModel(
        num_detectors=3,
        num_observables=2,
        errors=(
            DemError(probability=0.1, detectors=(0, 1, 2), observables=(0,)),
            DemError(probability=0.05, detectors=(0, 1), observables=(1,)),
            DemError(probability=0.05, detectors=(1, 2), observables=(0,)),
            DemError(probability=0.05, detectors=(0, 2), observables=(0,)),
        ),
    )
    decoder = BeliefPropagationDecoder(model, max_iterations=1)
    result = decode_batch(decoder, ((0, 1, 2),))[0]
    assert result == decoder.decode((0, 1, 2))
    assert result.converged is False and result.iterations == 1
    refusing = BeliefPropagationDecoder(model, max_iterations=1, osd=False)
    with pytest.raises(CapabilityError, match="did not settle"):
        decode_batch(refusing, ((), (0, 1, 2)))


def test_exclusive_model_is_not_reinterpreted_as_independent_faults() -> None:
    model = DetectorErrorModel(
        num_detectors=2,
        num_observables=1,
        errors=(
            DemError(probability=0.1, detectors=(0,), error_id=7),
            DemError(probability=0.1, detectors=(1,), error_id=7),
        ),
    )
    ids = model.error_ids
    with pytest.raises(CapabilityError, match="alternatives"):
        BeliefPropagationDecoder.from_detector_error_model(model)
    assert model.error_ids == ids


def test_pymatching_full_results_do_not_call_the_logical_only_batch_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("pymatching") is None:
        with pytest.raises(MatchingDependencyError):
            get_decoder("pymatching", _model())
        return
    decoder = get_decoder("pymatching", _model())
    expected = decoder.decode((0,))

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("logical-only Tensor output cannot supply full records")

    monkeypatch.setattr(type(decoder), "decode_batch", unexpected)
    assert decode_batch(decoder, ((0,),)) == (expected,)


def test_optional_adapter_is_not_installed_or_selected_by_batch_import() -> None:
    if importlib.util.find_spec("pymatching") is None:
        with pytest.raises(MatchingDependencyError):
            get_decoder("pymatching", _model())
    else:
        decoder = get_decoder("pymatching", _model())
        assert decode_batch(decoder, ((0,),)) == (decoder.decode((0,)),)
