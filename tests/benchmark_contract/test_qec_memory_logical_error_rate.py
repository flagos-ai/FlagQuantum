"""Logical-error-rate evidence for the memory route on the stabilizer engine.

The gate this file answers is the one the detector error model's own statistics
cannot: whether a decoding claim built out of a FlagQuantum memory experiment
reproduces the reference implementation's logical error rate. It measures a
curve rather than one number, so the direction of the curve is evidence too.

What is compared
----------------

One grid, both sides on the same experiment: a repetition code and a rotated
surface patch at distances 3, 5 and 7, at three physical error rates. Our side
lowers the memory circuit, samples it on the stabilizer engine, derives the
detector error model for the same circuit and the same noise record, and decodes
with PyMatching over that model's graph. The reference side transcribes the same
memory circuit gate for gate into a stim circuit, asks stim for that circuit's
own detector error model, and samples and decodes it the same way. Both sides
therefore see the same gates, the same noise locations and one decoding
algorithm, so a disagreement is a disagreement about the sampler or about the
derived model.

The claim is on aggregate rates and never on per-shot labels. Both models carry
mechanisms of equal weight, so a minimum-weight matcher has no reason to prefer
one of two explanations of a syndrome and the two are free to name different
observables at the same weight. Measured at distance three: the repetition code
disagreed on the observable of 38 shots in twenty thousand at one percent noise
and 91 in twenty thousand at two, and the surface patch of 46 shots in five
thousand at two percent and 14 in one thousand at distance five. The weight
itself agreed on every shot of every configuration, with a largest gap of
exactly zero (``tests/qec/test_adapters_pymatching.py`` owns the weight claim,
which is the one that is exact).

What the grid shows and what it does not
----------------------------------------

Every rate decreases from distance 3 to 5 to 7 at every rate measured, on both
codes, which the sweep below asserts. That is a decreasing curve and nothing
more: three distances at three rates estimate no threshold, and no logical
suppression, no fault-tolerance and no performance claim follows from them. A
threshold estimate needs many more distances and a fit, and this file
deliberately does not fit one.

Where the width comes from
--------------------------

The curve runs to a distance-7 patch, which is 97 wires and 336 detectors. The
derived construction route reads a signature off the circuit's layouts and
executes nothing, which is what makes that width reachable; the route that
forced each signature through an execution could not build past a distance-3
patch, whose 17 wires are the last the allocator survives. The derivation is
held against that execution location by location in
``tests/qec/test_dem_from_memory_circuit.py``, which is also where the
distance-7 build is asserted to be a property of the derivation rather than a
lucky sample. The in-tree matcher is not on this path: it enumerates the ways to
pair a syndrome's defective detectors and refuses more than twenty of them
outright, and it is exact rather than fast below that, so the reference decoder
is what decodes both sides and the in-tree matcher's ceiling is a separate owned
gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import pytest

from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    StabilizerCode,
    build_memory_circuit,
    sample_memory_circuit,
)

# Stim is an optional distribution and PyMatching is optional as well. The
# reference side needs both, so the file skips rather than failing: the core
# environment installs neither and still runs the tier. NumPy arrives with
# PyMatching and the counting below is written in it, so it is imported under the
# same guard.
pytest.importorskip("stim")
pytest.importorskip("pymatching")

import numpy as np
import pymatching
import stim

from tests.qec.test_dem_stim_reference_rates import _transcribe

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.integration]

# The shot count the tolerance is stated at. Four times the sibling file's is not
# needed: the reference comparison there is over detector pairs, whose rates go
# down with the square of the mechanism rate, while a logical error rate is one
# number per configuration, and 200,000 shots resolve a rate of two per million.
_SHOTS = 200_000
_SEED = 20260214
_SIGMA = 5.0

# Three rates below the points where the two codes cross, so the curve is a
# suppression curve on both families rather than a curve through a threshold.
_PARAMETER_SWEEP = (0.005, 0.01, 0.02)
_DISTANCES = (3, 5, 7)

# The reference decoder and sampler, and the graph our model defines. The model
# is merged first: a split fault is a hyperedge the graph refuses, and the graph
# states what a decoder may match a syndrome against rather than what a circuit
# contains.
_FAMILIES = {
    "repetition": RepetitionCode,
    "surface": RotatedSurfaceCode,
}


@dataclass(frozen=True)
class _Point:
    """One measured configuration of the grid.

    ``failures`` and ``reference_failures`` are kept beside the rates so the
    tolerance is a sampling bound on two counts rather than on two rounded
    numbers, and ``detectors`` and ``errors`` describe the model the point
    decodes with.
    """

    failures: int
    reference_failures: int
    shots: int
    detectors: int
    errors: int
    wires: int

    @property
    def rate(self) -> float:
        return self.failures / self.shots

    @property
    def reference_rate(self) -> float:
        return self.reference_failures / self.shots


def _code(family: str, distance: int) -> StabilizerCode:
    return _FAMILIES[family](distance=distance)


@lru_cache(maxsize=None)
def _measure(family: str, distance: int, probability: float) -> _Point:
    """Return both sides of one configuration, sampled from one seed.

    The two sides share the seed because they draw from different streams: the
    seed fixes each sampler's own sequence, so the comparison is reproducible
    without pretending the two draw correlated samples.
    """

    code = _code(family, distance)
    memory = build_memory_circuit(code, rounds=distance)
    noise = PhenomenologicalNoise(data_flip=probability, measurement_flip=probability)
    model = DetectorErrorModel.from_memory_circuit(
        memory, noise=noise
    ).merge_duplicate_mechanisms()
    reference_circuit = _transcribe(memory, probability=probability)
    reference_matching = pymatching.Matching.from_detector_error_model(
        reference_circuit.detector_error_model(decompose_errors=False)
    )
    ours_matching = pymatching.Matching.from_detector_error_model(
        stim.DetectorErrorModel(model.to_stim_text())
    )

    reference_detectors, reference_observables = (
        reference_circuit.compile_detector_sampler(seed=_SEED).sample(
            _SHOTS, separate_observables=True
        )
    )
    sampled = sample_memory_circuit(memory, noise=noise, shots=_SHOTS, seed=_SEED)
    ours_detectors = sampled.detectors.numpy().astype(bool)
    ours_observables = sampled.observables.numpy().astype(bool)

    failures = int(
        np.count_nonzero(
            np.any(
                ours_matching.decode_batch(ours_detectors) != ours_observables, axis=1
            )
        )
    )
    reference_failures = int(
        np.count_nonzero(
            np.any(
                reference_matching.decode_batch(reference_detectors)
                != reference_observables,
                axis=1,
            )
        )
    )
    return _Point(
        failures=failures,
        reference_failures=reference_failures,
        shots=_SHOTS,
        detectors=model.num_detectors,
        errors=model.num_errors,
        wires=len(code.data_wires) + len(code.ancilla_wires),
    )


def _tolerance(point: _Point) -> float:
    """Return the five-standard-error bound on the difference of two rates.

    The two sides are independent samples, so the difference's variance is the
    sum of theirs, and the bound is five standard errors of it. A rate this small
    is resolved only to whole events, so the variance is floored at the width of
    one unresolved event on each side; without the floor a configuration where
    neither side failed would be compared against a tolerance of exactly zero.
    The bound is also asserted below the standard error of a rate of one half,
    which no sampled rate at this shot count can exceed: a tolerance above it
    would no longer be a sampling error.
    """

    rates = (point.rate, point.reference_rate)
    variance = sum(rate * (1.0 - rate) for rate in rates) / point.shots
    variance = max(variance, 2.0 / point.shots**2)
    tolerance = _SIGMA * math.sqrt(variance)
    assert tolerance <= _SIGMA * math.sqrt(0.25 / point.shots), tolerance
    # A configuration where neither side failed is the one the floor above is for,
    # and the assertion it produces there is ``0 <= 0``. That is not a comparison,
    # so the bound is required to be a positive number of events on both sides
    # rather than an exact match that no independent sample could fail.
    assert tolerance > 0.0, tolerance
    return tolerance


@pytest.mark.parametrize("probability", _PARAMETER_SWEEP)
@pytest.mark.parametrize("distance", _DISTANCES)
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_the_logical_error_rate_agrees_with_the_reference(
    family: str, distance: int, probability: float
) -> None:
    """One grid point, both toolchains, one decoding algorithm.

    The assertion is on the rate and not on the labels, for the reason the module
    docstring states: both models carry mechanisms of equal weight, so a
    minimum-weight matcher may name different observables for the same syndrome
    at the same weight, and the labels are a tie-breaking artefact rather than a
    disagreement about the experiment.
    """

    point = _measure(family, distance, probability)
    tolerance = _tolerance(point)
    assert abs(point.rate - point.reference_rate) <= tolerance, (
        f"{family} distance {distance} at {probability}: "
        f"{point.failures}/{point.shots} failures here against "
        f"{point.reference_failures}/{point.shots} in the reference, a "
        f"difference of {abs(point.rate - point.reference_rate):.6f} against a "
        f"tolerance of {tolerance:.6f}"
    )


@pytest.mark.parametrize("probability", _PARAMETER_SWEEP)
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_the_rate_decreases_with_distance(family: str, probability: float) -> None:
    """The curve is a suppression curve over the three distances measured.

    Three points are enough to state that the rate falls and not enough to
    estimate where it would stop falling, so this asserts monotonicity and
    nothing about a threshold. The reference side is asserted with ours, because
    a curve that only one side shows is a disagreement about the code rather than
    about the toolchain.
    """

    points = [_measure(family, distance, probability) for distance in _DISTANCES]
    rates = [point.rate for point in points]
    reference = [point.reference_rate for point in points]
    # The sweep has to be three different patches, because a repeated distance
    # satisfies both monotonicity assertions trivially and would leave the
    # suppression claim resting on one width.
    widths = [point.detectors for point in points]
    assert len(set(widths)) == len(_DISTANCES), (family, probability, widths)
    assert widths == sorted(widths), (family, probability, widths)
    assert rates == sorted(rates, reverse=True), (family, probability, rates)
    assert reference == sorted(reference, reverse=True), (
        family,
        probability,
        reference,
    )


@pytest.mark.parametrize("distance", _DISTANCES)
def test_the_surface_patch_curve_widens_the_patch_it_models(distance: int) -> None:
    """A rotated surface patch at distance d is d-squared data wires and one
    ancilla per check, and the derived model reaches the width.

    The detector count is asserted against the layout's own rule rather than
    against a remembered integer: a check the preparation state pins keeps a
    boundary detector and every other check keeps one detector per adjacent round
    pair, so ``protected * (rounds + 1) + (total - protected) * (rounds - 1)``
    with ``rounds = distance``. The wire counts are what make the distance-7
    patch unreachable through an execution, which is the property the sibling
    suite pins by asserting that the model builds at all.
    """

    code = RotatedSurfaceCode(distance=distance)
    point = _measure("surface", distance, _PARAMETER_SWEEP[0])
    checks = code.checks
    protected = len([check for check in checks if not check.stabilizer.x_wires])
    expected = protected * (distance + 1) + (len(checks) - protected) * (distance - 1)
    assert point.detectors == expected
    assert point.wires == len(code.data_wires) + len(code.ancilla_wires)
    # A check per data qubit's worth of ancillas: distance**2 data wires and
    # distance**2 - 1 ancillas, which is 17 wires at distance 3, 49 at 5 and 97 at
    # 7. The executed route allocated the whole amplitude vector and stopped
    # between 17 and 31 wires.
    assert point.wires == distance**2 + distance**2 - 1
    assert point.errors > 0
