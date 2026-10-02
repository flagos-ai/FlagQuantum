"""Pin the arithmetic of the stim text interchange against stim itself.

The interchange has two writers and one reader. This package writes a
probability with ``repr``, so the text carries the shortest decimal that reads
back as the identical double; stim writes sixteen significant digits, which is
not always enough for that. The difference is a property of the two writers and
it is measured here rather than asserted, because the documents quote a digit
count and a drift bound that a reader has to be able to check.

Four things are pinned, all deterministic:

1. stim's writer emits exactly ``%.16g`` for every probability it prints, so
   sixteen significant digits is a measured format and not an estimate.
2. This package's writer loses no value over the sweep while stim's writer
   loses about a quarter of them, so the sweep discriminates between the two
   writers rather than merely passing on both.
3. The drift stim introduces is bounded by one part in ``10**15``, a bound that
   follows from the format and is confirmed against the sweep's worst case.
4. A model that leaves through this writer and returns through stim keeps its
   shape and its detector rates inside that bound, so the bound is attached to
   the interchange rather than to a single probability.

What this file does and does not prove
--------------------------------------

It proves the digit count, the direction of the loss, and the size of the drift
a stim round trip introduces. Each is read off stim's own writer, so a stim
release that changes its precision fails here instead of silently invalidating
the documents that quote it.

It does **not** prove that the drift is harmless. A detector rate that moves by
one part in ``10**15`` moves a matching weight by the same relative amount, and
whether that can reorder two competing explanations is the question
``tests/qec/test_adapters_pymatching.py`` answers for the cross-check; here the
drift is only characterised and bounded.
"""

from __future__ import annotations

import random

import pytest

from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit

stim = pytest.importorskip("stim")

# The sweep is drawn from a fixed seed so the pinned worst case is reproducible.
_SWEEP_SEED = 20261002
_SWEEP_DRAWS = 20000

# The format stim prints in, as a relative bound: sixteen significant decimal
# digits round to within half a unit in the last of them, and reading that
# decimal back as a double adds at most one binary unit in the last place, which
# is under ``2**-51`` of the value. The two together stay under one part in
# 10**15.
_DRIFT_BOUND = 1e-15


def _probability_text(model_text: str) -> str:
    """Return the probability stim printed on the first ``error`` line."""

    line = next(
        entry for entry in model_text.splitlines() if entry.startswith("error(")
    )
    return line[len("error(") : line.index(")")]


def _swept_probabilities() -> list[float]:
    """Return the fixed sweep of probabilities both writers are compared on."""

    rng = random.Random(_SWEEP_SEED)
    values = [rng.random() for _ in range(_SWEEP_DRAWS)]
    return [value for value in values if value != 0.0]


def _one_mechanism_model(probability: float) -> DetectorErrorModel:
    """Return a single-mechanism model whose only rate is ``probability``."""

    return DetectorErrorModel(
        num_detectors=1,
        num_observables=1,
        errors=(DemError(probability=probability, detectors=(0,), observables=(0,)),),
    )


def _stim_reprint(probability: float) -> str:
    """Return the probability stim prints after reading what this writer wrote."""

    text = _one_mechanism_model(probability).to_stim_text()
    return _probability_text(str(stim.DetectorErrorModel(text)))


def test_stim_prints_exactly_sixteen_significant_digits() -> None:
    """stim's writer is ``%.16g`` over the sweep, so the format is measured."""

    for probability in _swept_probabilities():
        assert _stim_reprint(probability) == f"{probability:.16g}"


def test_our_writer_is_exact_where_stims_writer_is_not() -> None:
    """This writer round trips every value; stim's loses a discriminating share."""

    probabilities = _swept_probabilities()
    ours_lost = 0
    stim_lost = 0
    for probability in probabilities:
        if float(
            _probability_text(_one_mechanism_model(probability).to_stim_text())
        ) != (probability):
            ours_lost += 1
        if float(_stim_reprint(probability)) != probability:
            stim_lost += 1

    assert ours_lost == 0
    # A sweep where stim's writer happened to be exact would not discriminate,
    # so the share is asserted rather than assumed.
    assert stim_lost > len(probabilities) // 10


def test_the_documented_precision_and_drift_are_the_ones_stim_shows() -> None:
    """Sixteen digits bound the drift by ``_DRIFT_BOUND``, worst case included."""

    worst = 0.0
    for probability in _swept_probabilities():
        printed = float(_stim_reprint(probability))
        worst = max(worst, abs(printed - probability) / probability)

    assert worst < _DRIFT_BOUND
    # The documents quote a drift near this figure, so a sweep that stopped
    # exercising the format would be visible rather than silently permissive.
    assert worst > 1e-16


def test_a_model_that_round_trips_through_stim_keeps_its_shape_and_rates() -> None:
    """The drift is a property of the interchange, not of one probability."""

    probabilities = _swept_probabilities()[:64]
    model = DetectorErrorModel(
        num_detectors=2,
        num_observables=1,
        errors=tuple(
            DemError(
                probability=probability,
                detectors=(index % 2,),
                observables=(0,) if index % 3 == 0 else (),
            )
            for index, probability in enumerate(probabilities)
        ),
    )

    returned = DetectorErrorModel.from_stim_text(
        str(stim.DetectorErrorModel(model.to_stim_text()))
    )
    assert returned.num_detectors == model.num_detectors
    assert returned.num_observables == model.num_observables
    assert returned.num_errors == model.num_errors

    for mine, theirs in zip(
        model.detector_rates().tolist(),
        returned.detector_rates().tolist(),
        strict=True,
    ):
        assert abs(theirs - mine) <= _DRIFT_BOUND * max(mine, 1.0)
    for mine, theirs in zip(
        model.observable_rates().tolist(),
        returned.observable_rates().tolist(),
        strict=True,
    ):
        assert abs(theirs - mine) <= _DRIFT_BOUND * max(mine, 1.0)
