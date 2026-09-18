"""Coverage for the forced-error signature engine."""

from __future__ import annotations

import dataclasses

import pytest

from flagquantum.qec.circuit import MemoryCircuit, build_memory_circuit
from flagquantum.qec.codes import RepetitionCode
from flagquantum.qec.dem import (
    _forced_signature,
    _inject_data_flip,
    _inject_measurement_flip,
)

pytestmark = pytest.mark.integration


def _built(distance: int) -> MemoryCircuit:
    return build_memory_circuit(RepetitionCode(distance=distance), rounds=distance)


def test_data_flip_at_round_zero_wire_zero() -> None:
    built = _built(3)
    source = _inject_data_flip(built, round_index=0, wire=0)
    detectors, observables = _forced_signature(built, source)
    assert detectors == (0,)
    assert observables == (0,)


def test_data_flip_on_the_middle_wire_touches_two_checks() -> None:
    built = _built(3)
    source = _inject_data_flip(built, round_index=0, wire=1)
    detectors, _ = _forced_signature(built, source)
    assert detectors == (0, 1)


def test_data_flip_in_the_last_round_touches_only_that_round() -> None:
    built = _built(3)
    source = _inject_data_flip(built, round_index=2, wire=0)
    detectors, observables = _forced_signature(built, source)
    assert detectors == (4,)
    assert observables == (0,)


def test_measurement_flip_flips_its_round_and_the_next_round() -> None:
    """A flipped syndrome bit is compared against both of its neighbours."""

    built = _built(3)
    source = _inject_measurement_flip(built, round_index=0, ancilla_wire=3)
    detectors, observables = _forced_signature(built, source)
    assert detectors == (0, 2)
    assert observables == ()


def test_measurement_flip_in_the_last_round_reaches_the_terminal_boundary() -> None:
    built = _built(3)
    source = _inject_measurement_flip(built, round_index=2, ancilla_wire=3)
    detectors, observables = _forced_signature(built, source)
    assert detectors == (4, 6)
    assert observables == ()


def test_measurement_flip_on_the_last_check() -> None:
    built = _built(3)
    source = _inject_measurement_flip(built, round_index=0, ancilla_wire=4)
    detectors, _ = _forced_signature(built, source)
    assert detectors == (1, 3)


def test_scaled_distance_produces_the_expected_detector_count() -> None:
    built = _built(5)
    source = _inject_data_flip(built, round_index=0, wire=0)
    detectors, observables = _forced_signature(built, source)
    assert len(built.detectors.detectors) == 24
    assert detectors == (0,)
    assert observables == (0,)


def test_data_injection_anchor_must_be_unique() -> None:
    built = _built(3)
    mangled = dataclasses.replace(
        built,
        source=built.source.replace("    for round_index in range(rounds):\n", ""),
    )
    with pytest.raises(ValueError, match="round loop"):
        _inject_data_flip(mangled, round_index=0, wire=0)


def test_measurement_injection_anchor_must_be_unique() -> None:
    built = _built(3)
    mangled = dataclasses.replace(
        built, source=built.source.replace("        last = qp.measure(wires=3)\n", "")
    )
    with pytest.raises(ValueError, match="measurement"):
        _inject_measurement_flip(mangled, round_index=0, ancilla_wire=3)


def test_injection_rejects_a_round_outside_the_configured_range() -> None:
    """A mechanism outside the configured rounds is rejected, not silently ignored."""

    built = _built(3)
    with pytest.raises(ValueError, match="round"):
        _inject_data_flip(built, round_index=3, wire=0)


def test_injection_rejects_an_undeclared_wire() -> None:
    built = _built(3)
    with pytest.raises(ValueError, match="declared"):
        _inject_data_flip(built, round_index=0, wire=99)
