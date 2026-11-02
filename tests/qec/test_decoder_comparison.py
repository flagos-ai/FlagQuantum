"""Shared-input decoder decisions against an independent finite DEM oracle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import pytest
import torch

from examples.qec.decoder_comparison import CASES
from flagquantum.errors import CapabilityError
from flagquantum.qec import (
    BeliefPropagationOsdDecodeResult,
    DetectorErrorModel,
    MatchingDecodeResult,
    PyMatchingDecoder,
    get_decoder,
)

pytestmark = pytest.mark.unit
NAMES = ("minimum_weight_matching", "belief_propagation_osd", "pymatching")
# Existing BPOSD-reference and PyMatching selection bounds; no production change.
WEIGHT_ATOL = 1e-9
PYMATCHING_ROUNDING = 2**-23


@dataclass(frozen=True)
class _Explanation:
    indices: tuple[int, ...]
    syndrome: int
    observable: int
    probability: Fraction
    weight: float


def _effect(model: DetectorErrorModel, indices: tuple[int, ...]) -> tuple[int, int]:
    detectors = observables = 0
    for index in indices:
        for detector in model.errors[index].detectors:
            detectors ^= 1 << detector
        for observable in model.errors[index].observables:
            observables ^= 1 << observable
    return detectors, observables


def _enumerate(model: DetectorErrorModel) -> tuple[_Explanation, ...]:
    """Enumerate the independent law using exact original-float probabilities."""
    assert model.num_detectors <= 4 and len(model.errors) <= 5
    rows = []
    for mask in range(1 << len(model.errors)):
        indices = tuple(i for i in range(len(model.errors)) if mask >> i & 1)
        probability = Fraction(1)
        for i, error in enumerate(model.errors):
            p = Fraction.from_float(error.probability)
            probability *= p if mask >> i & 1 else 1 - p
        detectors, observables = _effect(model, indices)
        weight = math.fsum(
            math.log((1 - model.errors[i].probability) / model.errors[i].probability)
            for i in indices
        )
        rows.append(_Explanation(indices, detectors, observables, probability, weight))
    assert sum(row.probability for row in rows) == 1
    return tuple(rows)


def _indices(
    model: DetectorErrorModel,
    result: MatchingDecodeResult | BeliefPropagationOsdDecodeResult,
) -> tuple[int, ...]:
    if isinstance(result, BeliefPropagationOsdDecodeResult):
        return result.mechanisms
    selected = []
    for edge in result.error_edges:
        matches = [
            i
            for i, error in enumerate(model.errors)
            if error.detectors
            == tuple(d for d in edge.detectors if d < model.num_detectors)
            and error.observables == edge.observables
            and error.probability == edge.probability
        ]
        assert len(matches) == 1
        selected.append(matches[0])
    assert len(set(selected)) == len(selected)
    return tuple(sorted(selected))


def _check_analytic_case(
    name: str, support: tuple[_Explanation, ...], mask: int
) -> None:
    if not support:
        return
    best = max(row.probability for row in support)
    minima = tuple(row for row in support if row.probability == best)
    if name == "unique_boundary":
        assert len(support) == 1
    if name == "unique_triangle":
        assert len(minima) == 1
    if name == "equal_cycle" and mask == 5:
        assert {row.observable for row in minima} == {0, 1}
    if name == "near_cycle" and mask == 5:
        assert len(minima) == 1
        assert len({row.probability for row in support}) == 2
    if name == "objective_diamond" and mask == 1:
        assert len(minima) == 1 and minima[0].indices == (0,)
        assert minima[0].observable == 1
        total = sum(row.probability for row in support)
        zero = sum(row.probability for row in support if row.observable == 0)
        one = sum(row.probability for row in support if row.observable == 1)
        assert (zero / total, one / total) == (Fraction(63, 104), Fraction(41, 104))
        assert zero > one  # Logical-class MAP differs from the best configuration.


@pytest.mark.parametrize("name,source,example_syndrome", CASES)
def test_shared_models_against_exact_independent_law(
    name: str, source: str, example_syndrome: tuple[int, ...]
) -> None:
    pytest.importorskip("pymatching")
    model = DetectorErrorModel.from_stim_text(source)
    before = model.to_stim_text()
    rows = _enumerate(model)
    decoders = tuple((key, get_decoder(key, model)) for key in NAMES)
    assert any(row.syndrome == sum(1 << d for d in example_syndrome) for row in rows)
    for mask in range(1 << model.num_detectors):
        syndrome = tuple(i for i in range(model.num_detectors) if mask >> i & 1)
        support = tuple(row for row in rows if row.syndrome == mask)
        _check_analytic_case(name, support, mask)
        for key, decoder in decoders:
            if not support:
                with pytest.raises(CapabilityError):
                    decoder.decode(syndrome)
                continue
            result = decoder.decode(syndrome)
            assert isinstance(
                result, (MatchingDecodeResult, BeliefPropagationOsdDecodeResult)
            )
            indices = _indices(model, result)
            chosen = next(row for row in support if row.indices == indices)
            assert result.observables == tuple(
                i for i in range(model.num_observables) if chosen.observable >> i & 1
            ), (name, key, syndrome, result, chosen)
            assert result.weight == pytest.approx(chosen.weight, abs=WEIGHT_ATOL, rel=0)
            optimum = max(row.probability for row in support)
            best_weight = min(row.weight for row in support)
            assert chosen.weight >= best_weight - WEIGHT_ATOL
            if isinstance(result, BeliefPropagationOsdDecodeResult):
                assert (
                    isinstance(result.converged, bool) and 0 <= result.iterations <= 30
                )
                # A valid BP+OSD explanation need not minimize configuration cost.
                continue
            band = WEIGHT_ATOL
            if key == "pymatching":
                band = (
                    len(model.errors) * PYMATCHING_ROUNDING * max(1, abs(result.weight))
                )
            assert abs(result.weight - best_weight) <= band
            best_labels = {
                row.observable for row in support if row.probability == optimum
            }
            alternatives = [
                row.weight for row in support if row.observable not in best_labels
            ]
            if len(best_labels) == 1 and (
                not alternatives or min(alternatives) - best_weight > band
            ):
                assert chosen.observable in best_labels
            if name == "near_cycle" and mask == 5 and key == "pymatching":
                assert len(best_labels) == 1
                assert 0 < max(row.weight for row in support) - best_weight < band
    assert model.to_stim_text() == before


@pytest.mark.parametrize(
    "source",
    [
        "error(0.125) D0\ndetector D0\n",
        "error(0.125) D0 L0\nerror(0.25) D1 L1\n"
        "detector D0\ndetector D1\nlogical_observable L0\nlogical_observable L1\n",
    ],
)
def test_private_edge_labels_do_not_change_logical_batch_or_instance(
    source: str,
) -> None:
    pytest.importorskip("pymatching")
    model = DetectorErrorModel.from_stim_text(source)
    decoder = get_decoder("pymatching", model)
    assert isinstance(decoder, PyMatchingDecoder)
    stored_graph = decoder._matching
    fields = tuple(vars(decoder))
    assert stored_graph.num_fault_ids == model.num_observables
    result = decoder.decode(tuple(range(model.num_detectors)))
    assert result.observables == tuple(range(model.num_observables))
    assert _effect(model, _indices(model, result)) == (
        (1 << model.num_detectors) - 1,
        (1 << model.num_observables) - 1,
    )
    batch = decoder.decode_batch(torch.ones((1, model.num_detectors), dtype=torch.bool))
    assert batch.shape == (1, model.num_observables) and bool(batch.all())
    assert decoder._matching is stored_graph
    assert stored_graph.num_fault_ids == model.num_observables
    assert tuple(vars(decoder)) == fields


@pytest.mark.parametrize(
    "source,syndrome,refused,reason",
    [
        (
            "error(0.125) D0 D1 D2 L0\ndetector D0\ndetector D1\ndetector D2\nlogical_observable L0\n",
            (0, 1, 2),
            ("minimum_weight_matching", "pymatching"),
            "hyperedge",
        ),
        (
            "error(0.125) D0 L0\nerror(0.25) D0\ndetector D0\nlogical_observable L0\n",
            (0,),
            ("pymatching",),
            "one edge per detector pair",
        ),
    ],
)
def test_refusal_domains_do_not_change_the_shared_input(
    source: str, syndrome: tuple[int, ...], refused: tuple[str, ...], reason: str
) -> None:
    pytest.importorskip("pymatching")
    model = DetectorErrorModel.from_stim_text(source)
    before = model.to_stim_text()
    for name in NAMES:
        if name in refused:
            with pytest.raises(CapabilityError, match=reason):
                get_decoder(name, model)
            continue
        result = get_decoder(name, model).decode(syndrome)
        assert isinstance(
            result, (MatchingDecodeResult, BeliefPropagationOsdDecodeResult)
        )
        detectors, observables = _effect(model, _indices(model, result))
        assert detectors == sum(1 << i for i in syndrome)
        assert result.observables == tuple(
            i for i in range(model.num_observables) if observables >> i & 1
        )
    assert model.to_stim_text() == before
