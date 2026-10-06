"""Coverage for the forced-error signature engine."""

from __future__ import annotations

import dataclasses

import pytest

from flagquantum.compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from flagquantum.qec.circuit import (
    MeasurementRef,
    MemoryCircuit,
    build_memory_circuit,
)
from flagquantum.qec.codes import CodeCheck, RepetitionCode
from flagquantum.qec.dem_construction import (
    _forced_signature,
    _inject_data_flip,
    _inject_measurement_flip,
)
from flagquantum.qec.surface import RotatedSurfaceCode
from flagquantum.runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session

pytestmark = pytest.mark.integration


def _built(distance: int) -> MemoryCircuit:
    return build_memory_circuit(RepetitionCode(distance=distance), rounds=distance)


def _surface(rounds: int = 3) -> MemoryCircuit:
    return build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=rounds)


def _classical_rows(built: MemoryCircuit, *, shots: int) -> list[list[int]]:
    """Execute the circuit's own source and return the recorded syndrome bits."""

    program = capture_source(built.source, (INDEX,))
    lowered = lower_dynamic_program(
        program,
        (built.rounds,),
        max_dynamic_measurements=built.rounds * len(built.code.checks),
    )
    execution = execute_hybrid_dynamic_session(
        lowered.circuit, shots=shots, seed=0, strategy="trajectory"
    )
    return execution.classical_bits.tolist()


def _round_detector(built: MemoryCircuit, *, round_index: int, check: CodeCheck) -> int:
    """Return the index of the round-comparison detector of one check.

    The index is looked up in the layout rather than computed from the counting
    rule, so a test that uses it is asking what the layout declares.
    """

    wanted = {MeasurementRef(round_index, check.ancilla_wire)}
    if round_index > 0:
        wanted.add(MeasurementRef(round_index - 1, check.ancilla_wire))
    for detector in built.detectors.detectors:
        if set(detector.parity) == wanted:
            return detector.index
    raise AssertionError(
        f"the layout declares no round-{round_index} detector for check "
        f"{check.index}"
    )


def _z_checks_containing(built: MemoryCircuit, wire: int) -> list[CodeCheck]:
    return [
        check
        for check in built.code.checks
        if not check.stabilizer.x_wires and wire in check.stabilizer.support
    ]


def test_an_x_type_code_yields_a_signature_where_the_register_bits_vary() -> None:
    """Determinism is required of the flip set, not of the raw register.

    An X-type ancilla is prepared in ``|+>``, so its round-zero outcome is a coin
    toss and no two shots agree bit for bit. The detectors the model records are
    nevertheless fixed: the steady-state X-type detector compares two rounds that
    the first round projected into the same eigenstate. A guard that compares the
    raw register refuses every code with an X-type check — the rotated surface
    code among them — for a randomness the model never sees.
    """

    built = _surface()
    rows = _classical_rows(built, shots=8)
    assert len({tuple(row) for row in rows}) > 1, "the probe needs varying shots"

    source = _inject_data_flip(built, round_index=1, wire=0)
    detectors, observables = _forced_signature(built, source)
    assert observables == (0,)


def test_a_surface_code_data_flip_flips_the_rounds_of_its_z_type_checks() -> None:
    """A data flip's detectors are the Z-type checks whose support holds it.

    It is read off the code's own stabilizer support and the layout's own round
    detectors, so it fails if either the adjacency or the detector grammar is
    wrong. The X-type checks are untouched: a bit flip commutes with them.
    """

    built = _surface()
    source = _inject_data_flip(built, round_index=1, wire=4)
    detectors, _ = _forced_signature(built, source)

    touching = _z_checks_containing(built, 4)
    assert len(touching) == 2, "wire 4 is the interior site of the distance-3 patch"
    assert detectors == tuple(
        sorted(_round_detector(built, round_index=1, check=check) for check in touching)
    )


def test_a_surface_code_measurement_flip_on_an_x_type_check_has_no_terminal() -> None:
    """An X-type check has no terminal detector, so its flip reaches only two.

    A measurement flip is compared against the round before it and the round
    after it. The Z-type check additionally has a terminal detector that compares
    its last syndrome with the data readout, which this check does not declare.
    """

    built = _surface()
    x_check = next(check for check in built.code.checks if check.stabilizer.x_wires)
    source = _inject_measurement_flip(
        built, round_index=1, ancilla_wire=x_check.ancilla_wire
    )
    detectors, observables = _forced_signature(built, source)

    assert detectors == tuple(
        sorted(
            (
                _round_detector(built, round_index=1, check=x_check),
                _round_detector(built, round_index=2, check=x_check),
            )
        )
    )
    assert observables == ()
    assert len(detectors) < len(built.detectors.detectors)


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


def test_a_nondeterministic_source_is_refused() -> None:
    """A mechanism whose flip set moves between shots is refused, not read.

    The Hadamard inserted before each check's measurement rotates that check's
    readout into a basis the state is not stabilized in, so the recorded bit is
    an independent coin toss. A round-zero detector is that single bit and a
    later detector XORs two independent tosses, so the flip set itself changes
    between the two seeded shots — which is the criterion, not the raw bits.
    """

    built = _built(5)
    source = built.source
    for ancilla in built.code.ancilla_wires:
        anchor = f"        last = qp.measure(wires={ancilla})\n"
        source = source.replace(anchor, f"        qp.H(wires={ancilla})\n{anchor}")
    with pytest.raises(ValueError, match="not deterministic"):
        _forced_signature(built, source)


def test_data_injection_anchor_must_be_unique() -> None:
    built = _built(3)
    mangled = dataclasses.replace(
        built,
        source=built.source.replace("    for round_index in range(rounds):\n", ""),
    )
    with pytest.raises(ValueError, match="round loop"):
        _inject_data_flip(mangled, round_index=0, wire=0)


def test_data_injection_refuses_a_repeated_round_loop_anchor() -> None:
    """A source that repeats the loop header cannot anchor a round injection."""

    built = _built(3)
    anchor = "    for round_index in range(rounds):\n"
    mangled = dataclasses.replace(
        built, source=built.source.replace(anchor, anchor * 2)
    )
    with pytest.raises(ValueError, match="found 2"):
        _inject_data_flip(mangled, round_index=0, wire=0)


def test_measurement_injection_anchor_must_be_unique() -> None:
    built = _built(3)
    mangled = dataclasses.replace(
        built, source=built.source.replace("        last = qp.measure(wires=3)\n", "")
    )
    with pytest.raises(ValueError, match="measurement"):
        _inject_measurement_flip(mangled, round_index=0, ancilla_wire=3)


def test_measurement_injection_refuses_a_repeated_measurement_anchor() -> None:
    """A repeated measure line leaves no way to tell which check was meant."""

    built = _built(3)
    anchor = "        last = qp.measure(wires=3)\n"
    mangled = dataclasses.replace(
        built, source=built.source.replace(anchor, anchor * 2)
    )
    with pytest.raises(ValueError, match="found 2"):
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
