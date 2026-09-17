"""Code-driven memory-circuit source with detection and observable layouts.

The source is a bounded hybrid-compiler program that runs the configured number
of syndrome-extraction rounds and returns the final check measurement. Detector
and observable identity is derived from the code, so a caller never asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from .codes import StabilizerCode
from .pauli import Pauli

_FUNCTION_NAME = "memory_experiment"


@dataclass(frozen=True)
class MeasurementRef:
    """One measurement location, by round and wire.

    ``round_index`` is ``None`` for the terminal data readout that follows the
    final syndrome round. That readout is not an explicit measurement in the
    emitted program: it is the runtime's final sample of the wire, the same
    convention the frozen repetition profile uses. The record is not ordered,
    because an optional round index has no total order.
    """

    round_index: int | None
    wire: int

    def __post_init__(self) -> None:
        if self.round_index is not None:
            if isinstance(self.round_index, bool) or not isinstance(
                self.round_index, Integral
            ):
                raise TypeError("measurement round index must be an integer or None")
            if self.round_index < 0:
                raise ValueError("measurement round index must be non-negative")
        if isinstance(self.wire, bool) or not isinstance(self.wire, Integral):
            raise TypeError("measurement wire must be an integer")
        if self.wire < 0:
            raise ValueError("measurement wire must be non-negative")


@dataclass(frozen=True)
class Detector:
    """One measurement parity that is deterministic in the noiseless circuit."""

    index: int
    parity: tuple[MeasurementRef, ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("detector index must be non-negative")
        if not self.parity:
            raise ValueError("detector must reference at least one measurement")
        for reference in self.parity:
            if not isinstance(reference, MeasurementRef):
                raise TypeError(
                    "detector parity entries must be MeasurementRef records"
                )
        if len(set(self.parity)) != len(self.parity):
            raise ValueError("a detector cannot repeat a measurement reference")


@dataclass(frozen=True)
class DetectorLayout:
    """Dense, ordered detectors for one configured memory experiment."""

    detectors: tuple[Detector, ...]

    def __post_init__(self) -> None:
        if not self.detectors:
            raise ValueError("detector layout requires at least one detector")
        expected = tuple(range(len(self.detectors)))
        if tuple(item.index for item in self.detectors) != expected:
            raise ValueError("detector layout must be dense and ordered from zero")

    def __len__(self) -> int:
        return len(self.detectors)


@dataclass(frozen=True)
class LogicalObservable:
    """One declared logical operator and the readout parity that realizes it."""

    index: int
    pauli: Pauli
    measurement_parity: tuple[MeasurementRef, ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("observable index must be non-negative")
        if not isinstance(self.pauli, Pauli):
            raise TypeError("observable operator must be a Pauli operator")
        if self.pauli.is_identity:
            raise ValueError("observable operator must not be the identity")
        if self.pauli.x_wires:
            raise ValueError("observable readout supports Z-type operators only")
        if not self.measurement_parity:
            raise ValueError("observable must reference at least one measurement")
        if any(
            not isinstance(reference, MeasurementRef)
            for reference in self.measurement_parity
        ):
            raise TypeError("observable parity entries must be MeasurementRef records")
        if any(
            reference.round_index is not None for reference in self.measurement_parity
        ):
            raise ValueError(
                "observable readout must reference terminal data readouts only"
            )
        wires = tuple(sorted(reference.wire for reference in self.measurement_parity))
        if wires != self.pauli.support:
            raise ValueError(
                "observable readout must measure exactly the observable's support"
            )


@dataclass(frozen=True)
class ObservableLayout:
    """Dense, ordered logical observables for one configured memory experiment."""

    observables: tuple[LogicalObservable, ...]

    def __post_init__(self) -> None:
        if not self.observables:
            raise ValueError("observable layout requires at least one observable")
        expected = tuple(range(len(self.observables)))
        if tuple(item.index for item in self.observables) != expected:
            raise ValueError("observable layout must be dense and ordered from zero")

    def __len__(self) -> int:
        return len(self.observables)


@dataclass(frozen=True)
class MemoryCircuit:
    """Code-driven memory-experiment source with its detection layouts."""

    code: StabilizerCode
    rounds: int
    source: str
    detectors: DetectorLayout
    observables: ObservableLayout

    def __post_init__(self) -> None:
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, Integral):
            raise TypeError("memory rounds must be an integer")
        if self.rounds <= 0:
            raise ValueError("memory rounds must be positive")
        if not isinstance(self.code, StabilizerCode):
            raise TypeError("code must implement the StabilizerCode protocol")
        if not self.source.strip():
            raise ValueError("memory circuit source must not be empty")
        expected = len(self.code.checks) * (self.rounds + 1)
        if len(self.detectors) != expected:
            raise ValueError(
                "memory circuit detector count must match len(checks) * (rounds + 1)"
            )
        self._validate_layout()

    def _validate_layout(self) -> None:
        """Tie every detector and observable reference to what this code declares."""

        ancilla_wires = set(self.code.ancilla_wires)
        data_wires = set(self.code.data_wires)
        for detector in self.detectors.detectors:
            for reference in detector.parity:
                if reference.round_index is None:
                    if reference.wire not in data_wires:
                        raise ValueError(
                            "detector terminal readout must reference a declared "
                            "data wire"
                        )
                elif reference.round_index >= self.rounds:
                    raise ValueError(
                        "detector syndrome round must be inside the configured rounds"
                    )
                elif reference.wire not in ancilla_wires:
                    raise ValueError(
                        "detector syndrome measurement must reference a declared "
                        "ancilla wire"
                    )
        logical_observables = self.code.logical_observables
        if len(self.observables) != len(logical_observables):
            raise ValueError(
                "observable layout must declare one observable per code logical"
            )
        for observable in self.observables.observables:
            if observable.pauli != logical_observables[observable.index]:
                raise ValueError(
                    "observable operator must match the code's declared logical "
                    "observable"
                )
            for reference in observable.measurement_parity:
                if reference.wire not in data_wires:
                    raise ValueError(
                        "observable readout must reference a declared data wire"
                    )


def _check_source(code: StabilizerCode) -> str:
    lines = [
        f"def {_FUNCTION_NAME}(rounds):",
        "    last = False",
        "    for round_index in range(rounds):",
    ]
    for check in code.checks:
        for control, target in check.cnot_wires:
            lines.append(f"        qp.CNOT(wires=[{control}, {target}])")
        lines.append(f"        last = qp.measure(wires={check.ancilla_wire})")
        lines.append(f"        qp.reset(wires={check.ancilla_wire})")
    lines.append("    return last")
    return "\n".join(lines) + "\n"


def _detector_layout(code: StabilizerCode, *, rounds: int) -> DetectorLayout:
    checks = code.checks
    detectors: list[Detector] = []
    for round_index in range(rounds):
        for check in checks:
            parity = [MeasurementRef(round_index, check.ancilla_wire)]
            if round_index > 0:
                parity.append(MeasurementRef(round_index - 1, check.ancilla_wire))
            detectors.append(Detector(index=len(detectors), parity=tuple(parity)))
    for check in checks:
        parity = [MeasurementRef(rounds - 1, check.ancilla_wire)]
        parity.extend(MeasurementRef(None, wire) for wire in check.stabilizer.support)
        detectors.append(Detector(index=len(detectors), parity=tuple(parity)))
    return DetectorLayout(tuple(detectors))


def _observable_layout(code: StabilizerCode) -> ObservableLayout:
    return ObservableLayout(
        tuple(
            LogicalObservable(
                index=index,
                pauli=pauli,
                measurement_parity=tuple(
                    MeasurementRef(None, wire) for wire in pauli.support
                ),
            )
            for index, pauli in enumerate(code.logical_observables)
        )
    )


def build_memory_circuit(code: StabilizerCode, *, rounds: int) -> MemoryCircuit:
    """Build a code's memory-experiment source and its detection layouts.

    Detector ``r * len(checks) + c`` compares check ``c`` in round ``r`` with
    the same check in round ``r - 1``, where round ``-1`` is the known all-zero
    prior state. The last ``len(checks)`` detectors compare the final syndrome
    round with the terminal data readout. The source itself is a bounded hybrid
    compiler program; this function neither lowers nor executes it.
    """

    if not isinstance(code, StabilizerCode):
        raise TypeError("code must implement the StabilizerCode protocol")
    if not code.checks:
        raise ValueError("memory experiment requires a code with at least one check")
    if isinstance(rounds, bool) or not isinstance(rounds, Integral):
        raise TypeError("rounds must be an integer")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    configured_rounds = int(rounds)
    return MemoryCircuit(
        code=code,
        rounds=configured_rounds,
        source=_check_source(code),
        detectors=_detector_layout(code, rounds=configured_rounds),
        observables=_observable_layout(code),
    )


__all__ = (
    "Detector",
    "DetectorLayout",
    "LogicalObservable",
    "MeasurementRef",
    "MemoryCircuit",
    "ObservableLayout",
    "build_memory_circuit",
)
