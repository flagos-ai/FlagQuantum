"""Pin the arithmetic of the stim text interchange against stim itself.

The interchange has two writers. This package writes a probability with
``repr``, so the text carries the shortest decimal that reads back as the
identical double. Stim writes it at ``std::setprecision(std::numeric_limits<
long double>::digits10 + 1)``, so the number of digits is a property of the
platform's ``long double`` rather than a constant of stim: nineteen significant
digits where that is the x86 80-bit extended type, which is the Linux CI
platform, and sixteen where it is a double, which is arm64 macOS. Seventeen
significant digits is what a double needs to round trip, so stim's writer is
exact at nineteen digits and lossy at sixteen, while this package's writer is
exact on either platform.

Everything below is measured rather than asserted, because the documents quote
the width and the drift that follows from it and a reader has to be able to
check them. The width is read off stim's own output first, and the sweep is then
held against that width, so this file discriminates on either platform. The
widths this repository's test platforms are known to show are pinned so that a
stim release which changed its precision fails here; a platform with no pinned
measurement still has every consequence of its own width checked.

Four things are pinned, all deterministic:

1. Stim's writer emits one fixed ``g`` width over the sweep, and that width is
   the one the probe reads, so the format is measured rather than estimated.
2. This package's writer loses no value over the sweep. Whether stim's writer
   loses one follows from its width: at nineteen digits none of the swept
   probabilities came back from stim's own reprint as a different value, and at
   sixteen about a quarter of them did.
3. Where that width is under seventeen digits the drift is bounded by one part
   in ``10**15`` -- sixteen significant decimal digits round to within half a
   unit in the last of them, and reading that decimal back as a double adds at
   most one binary unit in the last place, which is under ``2**-51`` of the
   value -- and the sweep's worst case is ``5.4e-16``, inside the bound. Where
   the width is nineteen digits the drift is exactly zero, because nothing the
   printer was given was rounded away.
4. A model that leaves through this writer and returns through stim keeps its
   shape and its detector rates inside that bound, so the bound is attached to
   the interchange rather than to a single probability.

What this file does and does not prove
--------------------------------------

It proves the printed width, the direction of the loss, and the size of the
drift a stim round trip introduces. Each is read off stim's own writer, so a
stim release or a platform whose width is not the one this file expects fails
here instead of silently invalidating the documents that quote the figure.

It does **not** prove that the drift is harmless. A detector rate that moves by
one part in ``10**15`` moves a matching weight by the same relative amount, and
whether that can reorder two competing explanations is the question
``tests/qec/test_adapters_pymatching.py`` answers for the cross-check; here the
drift is only characterised and bounded.

It also does not prove that stim could not print more digits: the width is
stim's own choice of stream precision, and a platform whose ``long double`` is
wider than a double gets a longer decimal for free rather than by design.
"""

from __future__ import annotations

import platform
import random
import sys

import pytest

from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit

stim = pytest.importorskip("stim")

# The sweep is drawn from a fixed seed so the pinned figures are reproducible.
_SWEEP_SEED = 20261002
_SWEEP_DRAWS = 20000

# Seventeen significant decimal digits name every double uniquely, so a printer
# that emits at least that many cannot lose one and a printer that emits fewer
# can. This is ``std::numeric_limits<double>::max_digits10``.
_DOUBLE_ROUND_TRIP_DIGITS = 17

# The format's relative bound where it is narrower than that: sixteen
# significant decimal digits round to within half a unit in the last of them,
# and reading that decimal back as a double adds at most one binary unit in the
# last place, which is under ``2**-51`` of the value. The two stay under one
# part in 10**15.
_DRIFT_BOUND = 1e-15

# A probability whose ``repr`` needs the seventeen digits a double needs and
# whose exact expansion needs more than thirty, so the width stim printed shows
# in the digits themselves rather than being hidden by a value a shorter decimal
# can also name.
_WIDTH_PROBE = 0.16098361967034136
_WIDTH_SEARCH_LIMIT = 40

# The widths stim's printer is known to show on the platforms this repository
# runs its tests on. The width follows the platform's ``long double``, so it is
# a property of the platform rather than of stim's version, and the two figures
# here are measurements of stim 1.16.0: nineteen digits on the Linux x86-64
# runners, where ``long double`` is the 80-bit extended type, and sixteen on
# arm64 macOS, where it is a double. A platform that is not listed is not
# pinned, because there is no measurement of it to pin; the width it shows is
# still read, and every consequence below is still checked against that width.
_PINNED_WIDTHS = {
    ("linux", "x86_64"): 19,
    ("darwin", "arm64"): 16,
}


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


def _stim_width() -> int:
    """Return the number of significant digits stim's printer emits here.

    The width is read off stim's own output rather than assumed.
    ``_WIDTH_PROBE`` needs seventeen digits to be named and more than thirty to
    be written out in full, so the largest precision whose ``g`` format
    reproduces stim's text for it is the precision stim used: a narrower format
    would have rounded the probe away, and a wider one would have written digits
    stim did not print.
    """

    printed = _stim_reprint(_WIDTH_PROBE)
    widths = [
        width
        for width in range(1, _WIDTH_SEARCH_LIMIT + 1)
        if f"{_WIDTH_PROBE:.{width}g}" == printed
    ]
    assert widths, (
        f"stim printed {printed!r} for {_WIDTH_PROBE!r}, which is not the text of "
        f"any fixed g width up to {_WIDTH_SEARCH_LIMIT} significant digits"
    )
    return max(widths)


def test_the_printed_width_is_the_one_this_platform_shows() -> None:
    """A stim release or a platform that moved the width fails here, not later."""

    key = (sys.platform, platform.machine())
    expected = _PINNED_WIDTHS.get(key)
    if expected is None:
        pytest.skip(f"no stim width is pinned for {key[0]}/{key[1]}")

    assert _stim_width() == expected, (
        f"stim 1.16.0 prints {expected} significant digits on {key[0]}/{key[1]}, "
        f"where long double has digits10 = {expected - 1}; a different width means "
        "stim's stream precision moved and the documents quoting that width are stale"
    )


def test_stim_prints_one_fixed_width_over_the_sweep() -> None:
    """Stim's writer is one ``g`` format at the measured width, sweep included."""

    width = _stim_width()
    for probability in _swept_probabilities():
        assert _stim_reprint(probability) == f"{probability:.{width}g}"


def test_our_writer_is_exact_and_stims_follows_its_width() -> None:
    """This writer round trips every value; stim's does exactly when its width allows."""

    probabilities = _swept_probabilities()
    width = _stim_width()
    ours_lost = 0
    stim_lost = 0
    for probability in probabilities:
        if (
            float(_probability_text(_one_mechanism_model(probability).to_stim_text()))
            != probability
        ):
            ours_lost += 1
        if float(_stim_reprint(probability)) != probability:
            stim_lost += 1

    assert ours_lost == 0
    if width >= _DOUBLE_ROUND_TRIP_DIGITS:
        # Seventeen digits name every double, so a printer at least that wide
        # cannot report a value other than the one it was given.
        assert stim_lost == 0
    else:
        # A sweep where stim's writer happened to be exact would not
        # discriminate, so the share is asserted rather than assumed.
        assert stim_lost > len(probabilities) // 10


def test_the_drift_follows_the_width_stim_prints() -> None:
    """Under seventeen digits the drift is bounded by ``_DRIFT_BOUND`` and measured."""

    width = _stim_width()
    worst = 0.0
    for probability in _swept_probabilities():
        printed = float(_stim_reprint(probability))
        worst = max(worst, abs(printed - probability) / probability)

    if width >= _DOUBLE_ROUND_TRIP_DIGITS:
        # Nothing the printer was given was rounded away, so a stim round trip
        # moved no probability.
        assert worst == 0.0
    else:
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
