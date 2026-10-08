"""Code-driven memory-circuit source with detection and observable layouts.

The source is a bounded hybrid-compiler program that runs the configured number
of syndrome-extraction rounds and returns the final check measurement. Detector
and observable identity is derived from the code, so a caller never asserts it,
and the measurement handles the experiment records are derived from the code the
same way, so a recorded bit is named by a handle rather than by a column index a
caller has to know.

A memory experiment is read in one basis, and the experiment states which. The
data qubits are prepared in that basis' state, the checks of that basis are
deterministic in round zero and again against the terminal readout, and the
logical observable read is the code's declared one in the same basis. Upstream
derives the basis from the preparation kernel it is handed; here it is the
``readout_basis`` field of the experiment, checked against the code's declared
logical observables rather than assumed from them, because the Steane code
declares one of each and a record alone does not say which experiment is meant.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

from .codes import CodeCheck, StabilizerCode, ancilla_bands
from .pauli import Pauli

_FUNCTION_NAME = "memory_experiment"

# The basis a memory experiment is prepared, checked and read out in. Lowercase
# because that is how the two components of a decoder context are already named.
ReadoutBasis = Literal["x", "z"]
_READOUT_BASES: tuple[ReadoutBasis, ...] = ("x", "z")


def _checked_basis(basis: object) -> ReadoutBasis:
    """Return ``basis`` once it names a basis this layer can read out in."""

    if not isinstance(basis, str):
        raise TypeError("readout basis must be a string")
    if basis not in _READOUT_BASES:
        raise ValueError(
            f"readout basis must be one of {list(_READOUT_BASES)}, not {basis!r}"
        )
    return basis


def _carried_by(check: CodeCheck, basis: ReadoutBasis) -> bool:
    """Return whether ``check``'s ancilla readout is in ``basis``.

    A check's stabilizer is pure X or pure Z -- :class:`~flagquantum.qec.CodeCheck`
    refuses a mixed one -- so a check is carried by exactly one basis, and a
    detector whose parity reads that check's ancilla belongs to that basis.
    """

    return bool(check.stabilizer.x_qubits) == (basis == "x")


def _readout_basis_of(pauli: Pauli) -> ReadoutBasis | None:
    """Return the basis ``pauli`` is measured in, or ``None`` when it is mixed."""

    if pauli.x_qubits and pauli.z_qubits:
        return None
    return "x" if pauli.x_qubits else "z"


def _readout_observables(
    code: StabilizerCode, basis: ReadoutBasis
) -> tuple[Pauli, ...]:
    """Return the code's declared logical observables that ``basis`` can read.

    The observable an experiment reads is the one measured in its readout basis:
    a Z-basis experiment reads the code's Z-type logical operator and an X-basis
    experiment the X-type one, which is upstream's ``get_observables_z`` and
    ``get_observables_x``. A code that declares no observable in the requested
    basis has no memory experiment here, and that is refused where the basis is
    chosen rather than answered by dropping the observable and building an
    experiment with no logical qubit.
    """

    observables = tuple(
        observable
        for observable in code.logical_observables
        if _readout_basis_of(observable) == basis
    )
    if not observables:
        name = basis.upper()
        article = "an" if basis == "x" else "a"
        raise ValueError(
            f"memory-circuit source requires {article} {name}-type logical "
            f"observable, because an experiment read out in the {name} basis "
            f"prepares the data qubits in that basis and reads them out in it, "
            f"and no logical observable this code declares is {name}-type"
        )
    return observables


@dataclass(frozen=True)
class MeasurementRef:
    """One measurement location, by round and qubit.

    ``round_index`` is ``None`` for the terminal data readout that follows the
    final syndrome round. That readout is not an explicit measurement in the
    emitted program: it is the runtime's final sample of that qubit, the same
    convention the frozen repetition profile uses. The record is not ordered,
    because an optional round index has no total order.
    """

    round_index: int | None
    qubit: int

    def __post_init__(self) -> None:
        if self.round_index is not None:
            if isinstance(self.round_index, bool) or not isinstance(
                self.round_index, Integral
            ):
                raise TypeError("measurement round index must be an integer or None")
            if self.round_index < 0:
                raise ValueError("measurement round index must be non-negative")
        if isinstance(self.qubit, bool) or not isinstance(self.qubit, Integral):
            raise TypeError("measurement qubit must be an integer")
        if self.qubit < 0:
            raise ValueError("measurement qubit must be non-negative")


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

    The operator is measured in the experiment's readout basis, so it must be
    pure X or pure Z: a mixed operator would be measured under one basis'
    premises over half its support, and the parity that realizes it would not be
    the parity of the terminal readouts it names.
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
        if self.pauli.x_qubits and self.pauli.z_qubits:
            raise ValueError(
                "observable readout reads one basis at a time, so its operator "
                "must be pure X or pure Z rather than mixed"
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
        qubits = tuple(sorted(reference.qubit for reference in self.measurement_parity))
        if qubits != self.pauli.support:
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

    ``readout_basis`` is the basis the experiment prepares, checks and reads out
    in, and it is part of the record rather than derived at use, because the code
    alone does not name one experiment: the Steane code declares a logical
    observable in each basis, so the same code and round count describe a Z
    memory and an X memory and only the record says which was built. Everything
    the basis decides -- the preparation and terminal rotation in ``source``, the
    terminal detectors, and which of the code's observables the layout reads --
    is derived from this one field, so no two fields of the record can describe
    two different experiments.
    """

    code: StabilizerCode
    rounds: int
    source: str
    detectors: DetectorLayout
    observables: ObservableLayout
    readout_basis: ReadoutBasis = "z"

    def __post_init__(self) -> None:
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, Integral):
            raise TypeError("memory rounds must be an integer")
        if self.rounds <= 0:
            raise ValueError("memory rounds must be positive")
        if not isinstance(self.code, StabilizerCode):
            raise TypeError("code must implement the StabilizerCode protocol")
        _checked_basis(self.readout_basis)
        if not self.source.strip():
            raise ValueError("memory circuit source must not be empty")
        expected = _detector_count(
            self.code, rounds=self.rounds, readout_basis=self.readout_basis
        )
        if len(self.detectors) != expected:
            raise ValueError(
                "memory circuit detector count must match the code's checks and "
                "the configured rounds"
            )
        self._validate_layout()

    @property
    def measurement_refs(self) -> tuple[MeasurementRef, ...]:
        """Every measurement location this configured experiment records.

        The vector is the handles the circuit itself declares rather than the
        subset its layouts happen to reference, and it is read off the code and
        the round count, so it needs no lowering: one handle per check per
        syndrome round -- the source measures every check once every round -- in
        round-major order, then one handle per data qubit for the terminal
        readout. A position in this vector is what names a recorded bit.

        The two sets need not coincide in either direction, which is why the
        vector is stated here rather than collected from the layouts. A one-round
        experiment measures the checks the readout basis cannot compare -- on a
        patch, its X-type checks under an all-zero preparation, or its Z-type
        checks under a ``|+>`` one -- and no detector names them, because such a
        check is deterministic neither in round zero nor at the terminal readout,
        so those bits are recorded and referenced by nothing; a longer experiment
        does name every handle, because each round after the first compares
        against the round before it. Reading a handle by name does not depend on
        which of the two holds.
        """

        refs = [
            MeasurementRef(round_index, check.ancilla_qubit)
            for round_index in range(self.rounds)
            for check in self.code.checks
        ]
        refs.extend(MeasurementRef(None, wire) for wire in self.code.data_qubits)
        return tuple(refs)

    def _validate_layout(self) -> None:
        """Tie every detector and observable reference to what this code declares."""

        ancilla_qubits = set(self.code.ancilla_qubits)
        data_qubits = set(self.code.data_qubits)
        for detector in self.detectors.detectors:
            for reference in detector.parity:
                if reference.round_index is None:
                    if reference.qubit not in data_qubits:
                        raise ValueError(
                            "detector terminal readout must reference a declared "
                            "data qubit"
                        )
                elif reference.round_index >= self.rounds:
                    raise ValueError(
                        "detector syndrome round must be inside the configured rounds"
                    )
                elif reference.qubit not in ancilla_qubits:
                    raise ValueError(
                        "detector syndrome measurement must reference a declared "
                        "ancilla qubit"
                    )
        readout_observables = _readout_observables(self.code, self.readout_basis)
        if len(self.observables) != len(readout_observables):
            raise ValueError(
                "observable layout must declare one observable per code logical "
                f"the experiment's {self.readout_basis.upper()} readout basis can "
                "measure"
            )
        for observable in self.observables.observables:
            if observable.pauli != readout_observables[observable.index]:
                raise ValueError(
                    "observable operator must match the code's declared logical "
                    "observable"
                )
            for reference in observable.measurement_parity:
                if reference.qubit not in data_qubits:
                    raise ValueError(
                        "observable readout must reference a declared data qubit"
                    )


def _require_ancilla_bands(code: StabilizerCode) -> None:
    """Refuse a record whose per-basis ancilla counts contradict its checks.

    The two bands are a function of the checks, and
    :func:`~flagquantum.qec.codes.ancilla_bands` derives them everywhere they are
    read. A record that also reports the two counts is stating them a second
    time, so this is where the two statements are compared: a count that does not
    match the band its own checks define would otherwise be read by whoever asks
    the record and ignored by whoever reads the checks, and the two readers would
    disagree silently. A record that declares ancillas measuring neither basis --
    a flag or an idle ancilla -- keeps them out of both bands and is unaffected,
    because this compares the counts against the bands and not against the total.
    """

    x_qubits, z_qubits = ancilla_bands(code.checks)
    stated = (code.num_ancilla_x_qubits, code.num_ancilla_z_qubits)
    derived = (len(x_qubits), len(z_qubits))
    if stated != derived:
        raise ValueError(
            "code reports "
            f"{stated[0]} X-type and {stated[1]} Z-type ancilla qubits, but its "
            f"checks measure {derived[0]} X-type and {derived[1]} Z-type "
            "stabilizers, so the two bands it states and the two bands its checks "
            "define disagree"
        )


def _detector_count(
    code: StabilizerCode, *, rounds: int, readout_basis: ReadoutBasis
) -> int:
    """Return how many detectors one configured memory experiment must declare.

    A check the readout basis carries is deterministic in round zero and again
    against the terminal readout, so it declares one detector per round against
    its predecessor plus a terminal one; a check the other basis carries is
    deterministic only against the round before it. The count is therefore
    ``protected * (rounds + 1) + other * (rounds - 1)``, where ``protected``
    counts the checks measuring the readout basis, and the two bases are the same
    formula read over the two check bands.
    """

    _readout_observables(code, readout_basis)
    protected = sum(1 for check in code.checks if _carried_by(check, readout_basis))
    total = len(code.checks)
    return protected * (rounds + 1) + (total - protected) * (rounds - 1)


def _basis_rotation(basis: ReadoutBasis) -> str:
    """Return the single-qubit gate that rotates a data qubit into ``basis``.

    A Z-basis experiment prepares the all-zero state the runtime starts in and
    reads out through the runtime's final sample, so it needs no rotation at all;
    an X-basis one is the same experiment conjugated by ``H`` on every data qubit,
    which is why upstream's X memory is its Z memory with the preparation kernel
    exchanged rather than a second circuit.
    """

    return "" if basis == "z" else "H"


def _check_source(code: StabilizerCode, *, readout_basis: ReadoutBasis) -> str:
    """Return the memory-experiment source in ``readout_basis``.

    Every check is prepared, measured and reset the same way it is in the Z
    source, because only the *data* basis changes the experiment: the loop always
    leaves each ancilla's Z-basis readout equal to its check's eigenvalue. What
    the basis adds is the rotation of the data qubits -- into the basis state
    before the first round and back before the terminal sample -- so the
    preparation and the readout are the basis the checks of that basis are
    deterministic in.
    """

    rotation = _basis_rotation(readout_basis)
    rotate = (
        []
        if not rotation
        else [f"    qp.{rotation}(wires={qubit})" for qubit in code.data_qubits]
    )
    lines = [
        f"def {_FUNCTION_NAME}(rounds):",
        *rotate,
        "    last = False",
        "    for round_index in range(rounds):",
    ]
    for check in code.checks:
        ancilla = check.ancilla_qubit
        if check.stabilizer.x_qubits:
            lines.append(f"        qp.H(wires={ancilla})")
        for control, target in check.cnot_qubits:
            lines.append(f"        qp.CNOT(wires=[{control}, {target}])")
        if check.stabilizer.x_qubits:
            lines.append(f"        qp.H(wires={ancilla})")
        lines.append(f"        last = qp.measure(wires={ancilla})")
        lines.append(f"        qp.reset(wires={ancilla})")
    lines.extend(rotate)
    lines.append("    return last")
    return "\n".join(lines) + "\n"


def _detector_layout(
    code: StabilizerCode, *, rounds: int, readout_basis: ReadoutBasis
) -> DetectorLayout:
    _readout_observables(code, readout_basis)
    detectors: list[Detector] = []
    for round_index in range(rounds):
        for check in code.checks:
            if round_index == 0 and not _carried_by(check, readout_basis):
                continue
            parity = [MeasurementRef(round_index, check.ancilla_qubit)]
            if round_index > 0:
                parity.append(MeasurementRef(round_index - 1, check.ancilla_qubit))
            detectors.append(Detector(index=len(detectors), parity=tuple(parity)))
    for check in code.checks:
        if not _carried_by(check, readout_basis):
            continue
        parity = [MeasurementRef(rounds - 1, check.ancilla_qubit)]
        parity.extend(MeasurementRef(None, qubit) for qubit in check.stabilizer.support)
        detectors.append(Detector(index=len(detectors), parity=tuple(parity)))
    return DetectorLayout(tuple(detectors))


def _observable_layout(
    code: StabilizerCode, *, readout_basis: ReadoutBasis
) -> ObservableLayout:
    return ObservableLayout(
        tuple(
            LogicalObservable(
                index=index,
                pauli=pauli,
                measurement_parity=tuple(
                    MeasurementRef(None, qubit) for qubit in pauli.support
                ),
            )
            for index, pauli in enumerate(_readout_observables(code, readout_basis))
        )
    )


def build_memory_circuit(
    code: StabilizerCode, *, rounds: int, readout_basis: ReadoutBasis = "z"
) -> MemoryCircuit:
    """Build a code's memory-experiment source and its detection layouts.

    The experiment is read in ``readout_basis``: the data qubits are prepared in
    that basis' state, and its checks are the ones deterministic in round zero and
    again at the terminal readout. Such a check therefore gets one detector per
    round comparing that round with the round before it, where round ``-1`` is the
    known preparation, plus one terminal detector comparing the final syndrome
    round with the terminal data readout. A check of the other basis is
    deterministic in neither place: it gets one detector for every round after the
    first, comparing two consecutive syndrome rounds, and nothing else. Which
    basis that is is stated by ``readout_basis`` rather than inferred from the
    code, because the Steane code declares a logical observable in each basis and
    the same code and round count describe two different experiments.

    The code must declare a logical observable in the requested basis, which is
    the operator the terminal readout realizes; a code that declares none is
    refused.

    The source itself is a bounded hybrid compiler program; this function neither
    lowers nor executes it.

    Args:
        code: The stabilizer code record to build an experiment for.
        rounds: Positive number of syndrome-extraction rounds.
        readout_basis: ``"z"`` or ``"x"``, the basis the experiment is prepared
            and read out in.

    Returns:
        A `~flagquantum.qec.MemoryCircuit` whose layouts describe the experiment
        ``readout_basis`` configures.

    Raises:
        TypeError: ``code`` does not implement the stabilizer-code protocol, or
            ``rounds`` is not an integer, or ``readout_basis`` is not a string.
        ValueError: ``rounds`` is not positive, ``readout_basis`` names no basis,
            or the code declares no logical observable in that basis.

    Examples:
        >>> import flagquantum.qec as qec
        >>> memory = qec.build_memory_circuit(
        ...     qec.RotatedSurfaceCode(distance=3), rounds=3
        ... )
        >>> len(memory.detectors), len(memory.observables), memory.readout_basis
        (24, 1, 'z')
    """

    if not isinstance(code, StabilizerCode):
        raise TypeError("code must implement the StabilizerCode protocol")
    if not code.checks:
        raise ValueError("memory experiment requires a code with at least one check")
    _require_ancilla_bands(code)
    basis = _checked_basis(readout_basis)
    _readout_observables(code, basis)
    if isinstance(rounds, bool) or not isinstance(rounds, Integral):
        raise TypeError("rounds must be an integer")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    configured_rounds = int(rounds)
    return MemoryCircuit(
        code=code,
        rounds=configured_rounds,
        source=_check_source(code, readout_basis=basis),
        detectors=_detector_layout(code, rounds=configured_rounds, readout_basis=basis),
        observables=_observable_layout(code, readout_basis=basis),
        readout_basis=basis,
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
