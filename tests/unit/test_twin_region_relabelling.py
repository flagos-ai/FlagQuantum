"""The Twin region composer relabels through Core's one qubit-relabelling rule.

``flagquantum.core.qubit_mapping.remap_qubits`` owns that rule. This file measures the
consumer side of it: that ``flagquantum.twin.region_model`` reads the rule from there
instead of carrying a private copy, and that every refusal the composition can raise is
reported in the same qubit vocabulary with the composed subject named.

The refusal text is user-visible, so each one is asserted exactly rather than by
substring: a message that calls a qubit a "wire" again is a vocabulary regression, and
exact equality is what makes the regression visible. The composed objects themselves --
calibration, durations, noise and readout rules -- are measured by
``tests/test_twin_region_model.py``, which is an ``integration`` file over the same public
entry point.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import flagquantum as fq
from flagquantum.noise import (
    DeviceNoiseProfile,
    GateDuration,
    NoiseModel,
    QubitNoiseCalibration,
    ReadoutError,
    bit_flip_channel,
)
from flagquantum.twin import QPUDigitalTwin, TwinCircuitSupport

pytestmark = pytest.mark.unit

#: A readout confusion matrix that is not the identity, so a carried-through rule is
#: distinguishable from a default.
_READOUT = ReadoutError(probabilities=((0.98, 0.02), (0.03, 0.97)))


def _calibration(qubit: int) -> QubitNoiseCalibration:
    return QubitNoiseCalibration(
        wire=qubit,
        t1=40.0,
        t2=60.0,
        excited_population=0.01,
        readout_error=_READOUT,
    )


def _support(twin: QPUDigitalTwin) -> TwinCircuitSupport:
    """Declare the smallest envelope that admits the two-qubit cell program."""

    qubits = twin.snapshot.physical_qubits
    return TwinCircuitSupport(
        evidence=fq.twin.TwinEvidenceEnvelope(
            snapshot_identity=twin.snapshot.identity,
            physical_qubits=qubits,
            supported_operations=("h", "cx", "rx"),
            maximum_instruction_count=8,
            verified_circuit_identities=(
                fq.Circuit(2).h(0).cx(0, 1).to_ir().content_hash,
            ),
            evidence_identity=(f"{qubits[0]:02x}{qubits[1]:02x}" * 16)[:64],
            verified_tv_error_bound=0.04,
            estimated_tv_error_bound=0.08,
            confidence_level=0.95,
        ),
        directed_couplers=((qubits[0], qubits[1]),),
        maximum_circuit_depth=4,
    )


def _cell(
    physical_qubits: tuple[int, int],
    *,
    durations: tuple[GateDuration, ...],
    rule_qubits: tuple[int, ...] | None = None,
    readout_qubits: tuple[int, ...] | None = None,
) -> tuple[QPUDigitalTwin, TwinCircuitSupport]:
    """Build one local Twin cell whose noise is scoped in that cell's local labels.

    Every scope is written in local labels -- ``0`` and ``1`` -- so a test can read the
    regional labels off the composed model and see the relabelling happen. A device
    profile requires at least one gate duration, so the caller states them.
    """

    profile = DeviceNoiseProfile(
        qubits=(_calibration(0), _calibration(1)),
        gate_durations=durations,
        source="acme:test",
        captured_at="2026-01-01T00:00:00+00:00",
    )
    model = NoiseModel(device_profile=profile)
    if rule_qubits is not None:
        model.add(("cx",), bit_flip_channel(0.01), qubits=rule_qubits)
    if readout_qubits is not None:
        model.add_readout(readout_qubits, _READOUT)
    twin = fq.twin.from_noise_model(
        model, target="acme:research", qubits=physical_qubits
    )
    return twin, _support(twin)


def _two_cell_region():
    """Two cells overlapping on qubit 13, so the second cell's labels must move."""

    return fq.twin.compose_region_twin(
        [
            _cell((12, 13), durations=(GateDuration("cx", 200.0, (0, 1)),)),
            _cell((13, 14), durations=(GateDuration("h", 20.0, (0,)),)),
        ]
    )


def test_a_region_relabels_a_cells_local_qubits_onto_the_region() -> None:
    """The second cell's local qubit 0 is the region's qubit 1, not its qubit 0."""

    model = _two_cell_region()

    assert model.region.physical_qubits == (12, 13, 14)
    assert sorted(
        (duration.gate_name, duration.wires)
        for duration in model.twin.noise_model.device_profile.gate_durations
    ) == [("cx", (0, 1)), ("h", (1,))]


def test_a_region_relabels_gate_noise_and_readout_rules_the_same_way() -> None:
    model = fq.twin.compose_region_twin(
        [
            _cell((12, 13), durations=(GateDuration("cx", 200.0, (0, 1)),)),
            _cell(
                (13, 14),
                durations=(GateDuration("cx", 200.0, (0, 1)),),
                rule_qubits=(0, 1),
                readout_qubits=(0,),
            ),
        ]
    )

    assert [tuple(rule.wires) for rule in model.twin.noise_model.rules] == [(1, 2)]
    assert [tuple(rule.wires) for rule in model.twin.noise_model.readout_rules] == [
        (1,)
    ]


def test_an_unscoped_gate_duration_stays_unscoped() -> None:
    """``None`` means "every qubit", which no relabelling may turn into a scope."""

    model = fq.twin.compose_region_twin(
        [_cell((12, 13), durations=(GateDuration("h", 20.0, None),))]
    )

    assert [
        (duration.gate_name, duration.wires)
        for duration in model.twin.noise_model.device_profile.gate_durations
    ] == [("h", None)]


def test_a_gate_duration_outside_the_cell_is_refused_by_name() -> None:
    with pytest.raises(ValueError) as error:
        fq.twin.compose_region_twin(
            [_cell((12, 13), durations=(GateDuration("h", 20.0, (9,)),))]
        )

    assert str(error.value) == (
        "Twin device profile references qubit 9, which the qubit map does not name"
    )


def test_a_gate_noise_rule_outside_the_cell_is_refused_by_name() -> None:
    with pytest.raises(ValueError) as error:
        fq.twin.compose_region_twin(
            [
                _cell(
                    (12, 13),
                    durations=(GateDuration("cx", 200.0, (0, 1)),),
                    rule_qubits=(0, 9),
                )
            ]
        )

    assert str(error.value) == (
        "Twin noise rule references qubit 9, which the qubit map does not name"
    )


def test_a_readout_rule_outside_the_cell_is_refused_by_name() -> None:
    with pytest.raises(ValueError) as error:
        fq.twin.compose_region_twin(
            [
                _cell(
                    (12, 13),
                    durations=(GateDuration("cx", 200.0, (0, 1)),),
                    readout_qubits=(9,),
                )
            ]
        )

    assert str(error.value) == (
        "Twin readout rule references qubit 9, which the qubit map does not name"
    )


def test_a_qubit_calibration_outside_the_cell_is_refused_by_name() -> None:
    """A calibration is relabelled by the same rule, so it refuses in the same words.

    The calibration is the first relabelling the composition performs, so this is the
    refusal a directly built twin reaches before any duration or rule is read. A twin
    assembled by ``from_noise_model`` cannot reach it -- that entry point already
    requires every profile label to be one of the cell's qubits -- so the cell here is
    assembled the way `fq.twin.QPUDigitalTwin` also allows: a matching snapshot over a
    profile that carries one calibration more than the cell has qubits.
    """

    twin, support = _cell((12, 13), durations=(GateDuration("cx", 200.0, (0, 1)),))
    profile = twin.noise_model.device_profile
    assert profile is not None
    wider = DeviceNoiseProfile(
        qubits=(*profile.qubits, _calibration(9)),
        gate_durations=profile.gate_durations,
        source=profile.source,
        captured_at=profile.captured_at,
    )
    wider_model = NoiseModel(device_profile=wider)
    directly_built = QPUDigitalTwin(
        snapshot=replace(
            twin.snapshot,
            calibration_identity=wider.identity,
            noise_model_identity=wider_model.identity,
        ),
        noise_model=wider_model,
    )
    wider_support = replace(
        support,
        evidence=replace(
            support.evidence, snapshot_identity=directly_built.snapshot.identity
        ),
    )

    with pytest.raises(ValueError) as error:
        fq.twin.compose_region_twin([(directly_built, wider_support)])

    assert str(error.value) == (
        "Twin device profile references qubit 9, which the qubit map does not name"
    )
