"""Code-driven memory-circuit source with detection and observable layouts.

The source is a bounded hybrid-compiler program that runs the configured number
of syndrome-extraction rounds and returns the final check measurement. Detector
and observable identity is derived from the code, so a caller never asserts it.

The experiment's readout frame is part of what is derived rather than assumed.
By default every data wire is prepared and read out in the Z basis, which is the
frame the code's declared logical observables are stated in. Asked for a logical
product instead, the builder rotates exactly the wires that product carries an
``X`` factor on, at both ends of the experiment, and re-derives which checks stay
deterministic. Both frames go through the same builder and the same layouts; only
the rotation set differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from .codes import StabilizerCode
from .logical import certify_logical_product
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
    """One declared logical operator and the readout parity that realizes it.

    ``measurement_parity`` names the terminal readout of each wire in the
    operator's support. Which basis that readout used is not recorded here but on
    the ``MemoryCircuit`` that carries this observable, because it is a property
    of the experiment rather than of the operator.
    """

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
        if self.pauli.x_wires and self.pauli.z_wires:
            raise ValueError(
                "observable operator must be a pure X-type or pure Z-type "
                "operator, because the terminal readout measures each wire in a "
                "single basis and a Y factor would need a second rotation"
            )
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
    """Code-driven memory-experiment source with its detection layouts.

    ``x_readout_wires`` names the data wires this experiment prepares and reads
    out in the X basis, in ascending order. It is empty for the default frame, in
    which the whole experiment is in the Z basis and the observables are the
    code's declared Z-type logical observables. It is the support of a certified
    logical product otherwise, and that product is then the experiment's single
    observable.
    """

    code: StabilizerCode
    rounds: int
    source: str
    detectors: DetectorLayout
    observables: ObservableLayout
    x_readout_wires: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, Integral):
            raise TypeError("memory rounds must be an integer")
        if self.rounds <= 0:
            raise ValueError("memory rounds must be positive")
        if not isinstance(self.code, StabilizerCode):
            raise TypeError("code must implement the StabilizerCode protocol")
        if not self.source.strip():
            raise ValueError("memory circuit source must not be empty")
        data_wires = set(self.code.data_wires)
        for wire in self.x_readout_wires:
            if isinstance(wire, bool) or not isinstance(wire, Integral):
                raise TypeError("X readout wires must be integers")
            if wire not in data_wires:
                raise ValueError(
                    "X readout wires must be declared data wires of the code"
                )
        if tuple(sorted(set(self.x_readout_wires))) != self.x_readout_wires:
            raise ValueError(
                "X readout wires must be unique and in ascending order, because "
                "they are applied to the source in the order given"
            )
        expected = _detector_count(
            self.code, rounds=self.rounds, x_readout=self.x_readout_wires
        )
        if len(self.detectors) != expected:
            raise ValueError(
                "memory circuit detector count must match the code's checks, the "
                "readout frame and the configured rounds"
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
        if self.x_readout_wires:
            self._validate_rotated_observable()
        else:
            self._validate_declared_observables()
        for observable in self.observables.observables:
            for reference in observable.measurement_parity:
                if reference.wire not in data_wires:
                    raise ValueError(
                        "observable readout must reference a declared data wire"
                    )

    def _validate_declared_observables(self) -> None:
        """Require the code's own declared Z-type logical observables, in order."""

        readable = _z_type_logical_observables(self.code)
        if len(self.observables) != len(readable):
            raise ValueError(
                "observable layout must declare one observable per readable code "
                "logical"
            )
        for observable in self.observables.observables:
            if observable.pauli != readable[observable.index]:
                raise ValueError(
                    "observable operator must match the code's declared logical "
                    "observable"
                )

    def _validate_rotated_observable(self) -> None:
        """Require one certified logical product read out on exactly its X wires."""

        if len(self.observables) != 1:
            raise ValueError(
                "a rotated readout measures one logical product, so the observable "
                "layout must declare exactly one observable"
            )
        (observable,) = self.observables.observables
        certify_logical_product(self.code, observable.pauli)
        if observable.pauli.x_wires != self.x_readout_wires:
            raise ValueError(
                "observable X-type factor must be exactly the wires the experiment "
                "reads out in the X basis, because a wire read out in the wrong "
                "basis measures a different operator"
            )


def _z_type_logical_observables(code: StabilizerCode) -> tuple[Pauli, ...]:
    """Return the declared logical operators this Z-basis readout can measure.

    Both the initial data state and the terminal data readout are in the Z basis,
    because the initial state is all-zero and this module reads out through the
    runtime's final sample of each data wire. A declared logical operator that is
    pure Z-type is measurable under those premises and becomes one observable of
    the experiment, in the order the code declares it. A pure X-type operator is
    not measurable in this frame and is left out of the layout: reading it needs
    the readout rotated into the X basis, which is the route a caller takes by
    passing a certified logical ``product`` to :func:`build_memory_circuit`. A
    mixed operator is neither, so it is refused rather than silently dropped.

    A code that declares no Z-type logical operator at all has nothing this
    experiment can read, so it is refused as well.

    Z-type checks are deterministic in round zero and again at the terminal
    readout, so each of them contributes a detector in every round and one
    terminal detector. An X-type check is deterministic in neither place: it is
    still measured every round, because comparing two consecutive rounds of it is
    what detects the Z errors the readout is vulnerable to, but it contributes no
    round-zero and no terminal detector.
    """

    observables = code.logical_observables
    if not observables:
        raise ValueError(
            "memory-circuit source requires a Z-type logical observable, because "
            "both the initial state and the terminal data readout are in the Z "
            "basis"
        )
    selected: list[Pauli] = []
    for observable in observables:
        if observable.x_wires and observable.z_wires:
            raise ValueError(
                "memory-circuit source requires pure logical observables, because "
                "both the initial state and the terminal data readout are in the Z "
                "basis, which cannot read a mixed operator"
            )
        if not observable.x_wires:
            selected.append(observable)
    if not selected:
        raise ValueError(
            "memory-circuit source requires a Z-type logical observable, because "
            "both the initial state and the terminal data readout are in the Z "
            "basis"
        )
    return tuple(selected)


def _protected_checks(
    code: StabilizerCode, *, x_readout: tuple[int, ...]
) -> tuple[bool, ...]:
    """Return, per check, whether it is deterministic in round zero and at the end.

    A check is deterministic in the preparation state when the state pins the
    eigenvalue the check measures. A Z-type check reads a pair of data wires that
    start in ``|0>``, and an X-type check reads a pair that start in ``|+>``;
    either way the check is pinned exactly when every wire of its support was
    prepared in the basis its type needs, which is the Z basis unless the wire is
    in ``x_readout``. The same condition holds again at the terminal readout,
    because the experiment ends by measuring each data wire in that same basis.

    A check that fails the condition is still measured every round: comparing two
    consecutive rounds of it is what detects the errors the readout is vulnerable
    to. It only loses its round-zero and terminal detectors.
    """

    rotated = frozenset(x_readout)
    return tuple(
        all(
            (wire in rotated) == bool(check.stabilizer.x_wires)
            for wire in check.stabilizer.support
        )
        for check in code.checks
    )


def _detector_count(
    code: StabilizerCode, *, rounds: int, x_readout: tuple[int, ...] = ()
) -> int:
    """Return how many detectors one configured memory experiment must declare."""

    if not x_readout:
        _z_type_logical_observables(code)
    protected = sum(_protected_checks(code, x_readout=x_readout))
    total = len(code.checks)
    return protected * (rounds + 1) + (total - protected) * (rounds - 1)


def _check_source(code: StabilizerCode, *, x_readout: tuple[int, ...] = ()) -> str:
    lines = [
        f"def {_FUNCTION_NAME}(rounds):",
        "    last = False",
    ]
    lines.extend(f"    qp.H(wires={wire})" for wire in x_readout)
    lines.append("    for round_index in range(rounds):")
    for check in code.checks:
        ancilla = check.ancilla_wire
        if check.stabilizer.x_wires:
            lines.append(f"        qp.H(wires={ancilla})")
        for control, target in check.cnot_wires:
            lines.append(f"        qp.CNOT(wires=[{control}, {target}])")
        if check.stabilizer.x_wires:
            lines.append(f"        qp.H(wires={ancilla})")
        lines.append(f"        last = qp.measure(wires={ancilla})")
        lines.append(f"        qp.reset(wires={ancilla})")
    lines.extend(f"    qp.H(wires={wire})" for wire in x_readout)
    lines.append("    return last")
    return "\n".join(lines) + "\n"


def _detector_layout(
    code: StabilizerCode, *, rounds: int, x_readout: tuple[int, ...] = ()
) -> DetectorLayout:
    if not x_readout:
        _z_type_logical_observables(code)
    protected = _protected_checks(code, x_readout=x_readout)
    detectors: list[Detector] = []
    for round_index in range(rounds):
        for position, check in enumerate(code.checks):
            if round_index == 0 and not protected[position]:
                continue
            parity = [MeasurementRef(round_index, check.ancilla_wire)]
            if round_index > 0:
                parity.append(MeasurementRef(round_index - 1, check.ancilla_wire))
            detectors.append(Detector(index=len(detectors), parity=tuple(parity)))
    for position, check in enumerate(code.checks):
        if not protected[position]:
            continue
        parity = [MeasurementRef(rounds - 1, check.ancilla_wire)]
        parity.extend(MeasurementRef(None, wire) for wire in check.stabilizer.support)
        detectors.append(Detector(index=len(detectors), parity=tuple(parity)))
    return DetectorLayout(tuple(detectors))


def _observable_layout(
    code: StabilizerCode, *, product: Pauli | None = None
) -> ObservableLayout:
    if product is not None:
        return ObservableLayout(
            (
                LogicalObservable(
                    index=0,
                    pauli=product,
                    measurement_parity=tuple(
                        MeasurementRef(None, wire) for wire in product.support
                    ),
                ),
            )
        )
    return ObservableLayout(
        tuple(
            LogicalObservable(
                index=index,
                pauli=pauli,
                measurement_parity=tuple(
                    MeasurementRef(None, wire) for wire in pauli.support
                ),
            )
            for index, pauli in enumerate(_z_type_logical_observables(code))
        )
    )


def build_memory_circuit(
    code: StabilizerCode, *, rounds: int, product: Pauli | None = None
) -> MemoryCircuit:
    """Build a code's memory-experiment source and its detection layouts.

    By default the experiment is in the Z basis. A Z-type check is then
    deterministic in round zero, because the initial state is the all-zero state,
    and again at the terminal readout. It therefore gets one detector per round
    comparing that round with the round before it, where round ``-1`` is the known
    prior state, plus one terminal detector comparing the final syndrome round
    with the terminal data readout. An X-type check is deterministic in neither
    place: it gets one detector for every round after the first, comparing two
    consecutive syndrome rounds, and nothing else. A code that declares no Z-type
    logical observable is refused, and a code that declares a pure X-type logical
    operator alongside a Z-type one keeps only the Z-type operator as an
    observable of this experiment.

    Passing a ``product`` moves the experiment into that product's frame instead.
    The product must be a certified logical operator of the same code, per
    :func:`flagquantum.qec.certify_logical_product`; every data wire it carries an
    ``X`` factor on is prepared in the X basis and read out in the X basis, and
    the product itself becomes the experiment's single observable. The detector
    rule is the same one stated above, applied with each wire's own basis: a check
    keeps its round-zero and terminal detectors exactly when every wire of its
    support was prepared in the basis its type needs.

    The source itself is a bounded hybrid compiler program; this function neither
    lowers nor executes it.
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
    if product is not None:
        certify_logical_product(code, product)
    x_readout = product.x_wires if product is not None else ()
    if _detector_count(code, rounds=configured_rounds, x_readout=x_readout) == 0:
        raise ValueError(
            f"a {configured_rounds}-round memory experiment of this code in this "
            "readout frame has no deterministic detector, because no check is "
            "pinned by the preparation state and comparing two syndrome rounds is "
            "the only comparison left; increase the round count"
        )
    return MemoryCircuit(
        code=code,
        rounds=configured_rounds,
        source=_check_source(code, x_readout=x_readout),
        detectors=_detector_layout(code, rounds=configured_rounds, x_readout=x_readout),
        observables=_observable_layout(code, product=product),
        x_readout_wires=x_readout,
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
