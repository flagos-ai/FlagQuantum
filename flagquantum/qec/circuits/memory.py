"""Code-driven memory-circuit source with detection and observable layouts.

The source is a bounded hybrid-compiler program that runs the configured number
of syndrome-extraction rounds and returns the final check measurement. Detector
and observable identity is derived from the code, so a caller never asserts it,
and the measurement handles the experiment records are derived from the code the
same way, so a recorded bit is named by a handle rather than by a column index a
caller has to know.

Which qubits are prepared and read out in the X basis is part of what is derived
rather than assumed, and it is the record's one piece of authored frame
information. By default no qubit is rotated, so the whole experiment is in the Z
basis, which is the frame the code's declared logical observables are stated in,
and the experiment reads the code's declared Z-type logical operator. Asked for a
certified logical ``product`` instead, the builder rotates exactly the qubits
that product carries an ``X`` factor on, at both ends of the experiment, and the
product itself becomes the experiment's single observable. Asked for the whole
experiment in the X basis, it rotates every data qubit and reads the code's
declared X-type logical operator. All three routes go through the same builder,
the same detector rule and the same observable layout: only the set of qubits
rotated into the X basis differs. Upstream derives that set from the preparation
kernel it is handed; here it is derived from the request and stored on the
record, because the code alone does not name one experiment -- the Steane code
declares a logical observable in each basis, so the same code and round count
describe a Z memory and an X memory and only the record says which was built.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

from ..codes import StabilizerCode, ancilla_band_counts
from ..logical import certify_logical_product
from ..pauli import Pauli

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


def _readout_basis_of(pauli: Pauli) -> ReadoutBasis | None:
    """Return the basis ``pauli`` is measured in, or ``None`` when it is mixed."""

    if pauli.x_qubits and pauli.z_qubits:
        return None
    return "x" if pauli.x_qubits else "z"


def _declared_readout(code: StabilizerCode, basis: ReadoutBasis) -> tuple[Pauli, ...]:
    """Return the code's declared logical observables ``basis`` can measure."""

    return tuple(
        observable
        for observable in code.logical_observables
        if _readout_basis_of(observable) == basis
    )


def _readout_observables(
    code: StabilizerCode, basis: ReadoutBasis
) -> tuple[Pauli, ...]:
    """Return the code's declared logical observables that ``basis`` can read.

    The observable an experiment reads is the one measured in its readout basis:
    a Z-basis experiment reads the code's Z-type logical operator and an X-basis
    experiment the X-type one, which is upstream's ``get_observables_z`` and
    ``get_observables_x``. A code that declares no observable in the requested
    basis has no memory experiment here, and that is refused where the frame is
    chosen rather than answered by dropping the observable and building an
    experiment with no logical qubit.
    """

    observables = _declared_readout(code, basis)
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


def _pinned_checks(
    code: StabilizerCode, *, x_readout: tuple[int, ...]
) -> tuple[bool, ...]:
    """Return, per check, whether it is deterministic in round zero and at the end.

    A check is deterministic in the preparation state when the state pins the
    eigenvalue the check measures. Each qubit of the check's support is prepared
    in the Z basis unless it is in ``x_readout``, so the check is pinned exactly
    when every qubit carrying an ``X`` factor was prepared in the X basis and
    every qubit carrying a ``Z`` factor was prepared in the Z basis. With an empty
    frame that is "a pure Z-type check" and with every data qubit rotated it is
    "a pure X-type check"; a frame that rotates a subset needs a *pattern* of
    rotations rather than all or none. The same condition holds again at the
    terminal readout, because the experiment ends by measuring each data qubit in
    that same basis.

    A check that fails the condition is still measured every round: comparing two
    consecutive rounds of it is what detects the errors the readout is vulnerable
    to. It only loses its round-zero and terminal detectors.
    """

    rotated = frozenset(x_readout)
    return tuple(
        all(
            (qubit in rotated) == (qubit in check.stabilizer.x_qubits)
            for qubit in check.stabilizer.support
        )
        for check in code.checks
    )


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
                "must be pure X-type or pure Z-type rather than mixed"
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

    ``x_readout_qubits`` is the set of data qubits this experiment prepares and
    reads out in the X basis, in ascending order, and it is the record's one
    authored statement of the experiment's frame. Everything the frame decides --
    the preparation and terminal rotation in ``source``, which checks are
    deterministic in round zero and again at the terminal readout, and which of
    the code's observables the layout reads -- is derived from it, so no two
    fields of the record can describe two different experiments.

    It is empty for the default frame, in which the whole experiment is in the Z
    basis and the observables are the code's declared Z-type logical observables.
    It is every data qubit for an X-basis experiment, whose observables are the
    code's declared X-type logical observables. It is the support of a certified
    logical product otherwise, and that product is then the experiment's single
    observable. ``readout_basis`` names which of the two pure frames that is, and
    is derived from this field rather than stored beside it.
    """

    code: StabilizerCode
    rounds: int
    source: str
    detectors: DetectorLayout
    observables: ObservableLayout
    x_readout_qubits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, Integral):
            raise TypeError("memory rounds must be an integer")
        if self.rounds <= 0:
            raise ValueError("memory rounds must be positive")
        if not isinstance(self.code, StabilizerCode):
            raise TypeError("code must implement the StabilizerCode protocol")
        if not self.source.strip():
            raise ValueError("memory circuit source must not be empty")
        data_qubits = set(self.code.data_qubits)
        for qubit in self.x_readout_qubits:
            if isinstance(qubit, bool) or not isinstance(qubit, Integral):
                raise TypeError("X readout qubits must be integers")
            if qubit not in data_qubits:
                raise ValueError(
                    "X readout qubits must be declared data qubits of the code"
                )
        if tuple(sorted(set(self.x_readout_qubits))) != self.x_readout_qubits:
            raise ValueError(
                "X readout qubits must be unique and in ascending order, because "
                "they are applied to the source in the order given"
            )
        expected = _detector_count(
            self.code, rounds=self.rounds, x_readout=self.x_readout_qubits
        )
        if len(self.detectors) != expected:
            raise ValueError(
                "memory circuit detector count must match the code's checks, the "
                "readout frame and the configured rounds"
            )
        self._validate_layout()

    @property
    def readout_basis(self) -> ReadoutBasis:
        """Return the basis this experiment's logical observable is read in."""

        return "x" if self.x_readout_qubits else "z"

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
        experiment measures the checks the readout frame cannot compare -- on a
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
        refs.extend(MeasurementRef(None, qubit) for qubit in self.code.data_qubits)
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
        if not self.x_readout_qubits:
            self._validate_declared_observables()
        else:
            declared = _declared_readout(self.code, "x")
            stated = tuple(item.pauli for item in self.observables.observables)
            if not declared or stated != declared:
                self._validate_rotated_observable()
        for observable in self.observables.observables:
            for reference in observable.measurement_parity:
                if reference.qubit not in data_qubits:
                    raise ValueError(
                        "observable readout must reference a declared data qubit"
                    )

    def _validate_declared_observables(self) -> None:
        """Require the code's own declared logical observables, in order."""

        readable = _readout_observables(self.code, "z")
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
        """Require one certified logical product read out on exactly its X qubits."""

        if len(self.observables) != 1:
            raise ValueError(
                "a rotated readout measures one logical product, so the observable "
                "layout must declare exactly one observable"
            )
        (observable,) = self.observables.observables
        certify_logical_product(self.code, observable.pauli)
        if observable.pauli.x_qubits != self.x_readout_qubits:
            raise ValueError(
                "observable X-type factor must be exactly the wires the experiment "
                "reads out in the X basis, because a qubit read out in the wrong "
                "basis measures a different operator"
            )


def _require_ancilla_bands(code: StabilizerCode) -> None:
    """Refuse a record whose per-basis ancilla counts contradict its checks.

    The two counts are a function of the checks, and
    :func:`~flagquantum.qec.codes.ancilla_band_counts` reads them off the checks
    everywhere they are consumed. A record that also reports the two counts is
    stating them a second time, so this is where the two statements are compared:
    a count that does not match what its own checks measure would otherwise be
    read by whoever asks the record and ignored by whoever reads the checks, and
    the two readers would disagree silently. A record that declares ancillas
    measuring neither basis -- a flag or an idle ancilla -- keeps them out of both
    counts and is unaffected, because this compares counts and not the total. A
    record whose checks measure both bases is compared the same way, because a
    check that carries both factors counts once for each; the band *sets* are what
    a mixed check leaves undefined, and this guard reads the counts rather than
    the sets for exactly that reason.
    """

    stated = (code.num_ancilla_x_qubits, code.num_ancilla_z_qubits)
    derived = ancilla_band_counts(code.checks)
    if stated != derived:
        raise ValueError(
            "code reports "
            f"{stated[0]} X-type and {stated[1]} Z-type ancilla qubits, but its "
            f"checks measure {derived[0]} X-type and {derived[1]} Z-type "
            "stabilizers, so the two bands it states and the two bands its checks "
            "define disagree"
        )


def _detector_count(
    code: StabilizerCode, *, rounds: int, x_readout: tuple[int, ...] = ()
) -> int:
    """Return how many detectors one configured memory experiment must declare.

    A check the frame pins is deterministic in round zero and again against the
    terminal readout, so it declares one detector per round against its
    predecessor plus a terminal one; a check the frame does not pin is
    deterministic only against the round before it. The count is therefore
    ``pinned * (rounds + 1) + other * (rounds - 1)``. An empty frame reads the
    code's declared Z-type logical observable, so a code that declares none is
    refused here rather than answered with a layout that reads nothing.
    """

    if not x_readout:
        _readout_observables(code, "z")
    pinned = sum(_pinned_checks(code, x_readout=x_readout))
    total = len(code.checks)
    return pinned * (rounds + 1) + (total - pinned) * (rounds - 1)


def _check_source(code: StabilizerCode, *, x_readout: tuple[int, ...] = ()) -> str:
    """Return the memory-experiment source whose frame is ``x_readout``.

    Every check is prepared, measured and reset the same way whatever the frame,
    because only the *data* qubits' basis changes the experiment: the loop always
    leaves each ancilla's Z-basis readout equal to its check's eigenvalue. What
    the frame adds is the rotation of the data qubits -- into the frame's basis
    state before the first round and back before the terminal sample -- so the
    preparation and the readout are the basis the checks the frame pins are
    deterministic in.
    """

    rotate = [f"    qp.H(wires={qubit})" for qubit in x_readout]
    lines = [
        f"def {_FUNCTION_NAME}(rounds):",
        *rotate,
        "    last = False",
        "    for round_index in range(rounds):",
    ]
    for check in code.checks:
        ancilla = check.ancilla_qubit
        # ``cnot_qubits`` states the Z-factor pairs first and the X-factor pairs
        # after them, so the split is the Z run's length and the ``H`` pair wraps
        # the X run alone -- never the Z run, which would conjugate the check
        # into a different operator. A pure Z-type check has an empty X run and
        # gets no ``H`` at all; a pure X-type one has an empty Z run and is the
        # wrapped gadget alone. Those are the two shapes this emission already
        # produced, and the mixed check uses both runs through the same ancilla.
        split = len(check.stabilizer.z_qubits)
        for control, target in check.cnot_qubits[:split]:
            lines.append(f"        qp.CNOT(wires=[{control}, {target}])")
        if check.stabilizer.x_qubits:
            lines.append(f"        qp.H(wires={ancilla})")
        for control, target in check.cnot_qubits[split:]:
            lines.append(f"        qp.CNOT(wires=[{control}, {target}])")
        if check.stabilizer.x_qubits:
            lines.append(f"        qp.H(wires={ancilla})")
        lines.append(f"        last = qp.measure(wires={ancilla})")
        lines.append(f"        qp.reset(wires={ancilla})")
    lines.extend(rotate)
    lines.append("    return last")
    return "\n".join(lines) + "\n"


def _detector_layout(
    code: StabilizerCode, *, rounds: int, x_readout: tuple[int, ...] = ()
) -> DetectorLayout:
    if not x_readout:
        _readout_observables(code, "z")
    pinned = _pinned_checks(code, x_readout=x_readout)
    detectors: list[Detector] = []
    for round_index in range(rounds):
        for position, check in enumerate(code.checks):
            if round_index == 0 and not pinned[position]:
                continue
            parity = [MeasurementRef(round_index, check.ancilla_qubit)]
            if round_index > 0:
                parity.append(MeasurementRef(round_index - 1, check.ancilla_qubit))
            detectors.append(Detector(index=len(detectors), parity=tuple(parity)))
    for position, check in enumerate(code.checks):
        if not pinned[position]:
            continue
        parity = [MeasurementRef(rounds - 1, check.ancilla_qubit)]
        parity.extend(MeasurementRef(None, qubit) for qubit in check.stabilizer.support)
        detectors.append(Detector(index=len(detectors), parity=tuple(parity)))
    return DetectorLayout(tuple(detectors))


def _observable_layout(
    code: StabilizerCode,
    *,
    x_readout: tuple[int, ...] = (),
    product: Pauli | None = None,
) -> ObservableLayout:
    """Return the observables one frame reads, in the order the code declares them.

    A frame that rotates a proper subset of the data qubits is the frame of the
    certified ``product`` that named it, and that product is the experiment's one
    observable. An empty frame reads the code's declared Z-type logical
    observables and a frame over every data qubit reads its declared X-type ones,
    because those are the declared operators the frame's terminal readout can
    measure.
    """

    if product is not None and x_readout:
        return ObservableLayout(
            (
                LogicalObservable(
                    index=0,
                    pauli=product,
                    measurement_parity=tuple(
                        MeasurementRef(None, qubit) for qubit in product.support
                    ),
                ),
            )
        )
    basis: ReadoutBasis = "x" if x_readout else "z"
    return ObservableLayout(
        tuple(
            LogicalObservable(
                index=index,
                pauli=pauli,
                measurement_parity=tuple(
                    MeasurementRef(None, qubit) for qubit in pauli.support
                ),
            )
            for index, pauli in enumerate(_readout_observables(code, basis))
        )
    )


def build_memory_circuit(
    code: StabilizerCode,
    *,
    rounds: int,
    product: Pauli | None = None,
    readout_basis: ReadoutBasis = "z",
) -> MemoryCircuit:
    """Build a code's memory-experiment source and its detection layouts.

    The experiment is read in one frame. By default no data qubit is rotated, so
    the whole experiment is in the Z basis: the code's Z-type checks are
    deterministic in round zero, because the initial state is the all-zero state,
    and again at the terminal readout. Such a check therefore gets one detector
    per round comparing that round with the round before it, where round ``-1``
    is the known preparation, plus one terminal detector comparing the final
    syndrome round with the terminal data readout. A check the frame does not pin
    is deterministic in neither place: it gets one detector for every round after
    the first, comparing two consecutive syndrome rounds, and nothing else.

    Passing a ``product`` moves the experiment into that product's frame instead.
    The product must be a certified logical operator of the same code, per
    :func:`flagquantum.qec.certify_logical_product`; every data qubit it carries
    an ``X`` factor on is prepared in the X basis and read out in the X basis, and
    the product itself becomes the experiment's single observable. Asking for
    ``readout_basis="x"`` rotates every data qubit and reads the code's declared
    X-type logical operator. The detector rule is the same one stated above,
    applied with each qubit's own basis: a check keeps its round-zero and
    terminal detectors exactly when every qubit of its support was prepared in
    the basis its factor needs.

    The code must declare a logical observable the frame can read, which is the
    operator the terminal readout realizes; a code that declares none is refused.

    The source itself is a bounded hybrid compiler program; this function neither
    lowers nor executes it.

    Args:
        code: The stabilizer code record to build an experiment for.
        rounds: Positive number of syndrome-extraction rounds.
        product: A certified logical operator whose X support is the frame, or
            ``None`` to select the frame by ``readout_basis``.
        readout_basis: ``"z"`` or ``"x"``, the basis a frame selected by name is
            prepared and read out in.

    Returns:
        A `~flagquantum.qec.MemoryCircuit` whose layouts describe the experiment
        the requested frame configures.

    Raises:
        TypeError: ``code`` does not implement the stabilizer-code protocol, or
            ``rounds`` is not an integer, or ``product`` is neither a Pauli
            operator nor ``None``, or ``readout_basis`` is not a string.
        ValueError: ``rounds`` is not positive, ``readout_basis`` names no basis,
            both ``product`` and an explicit ``readout_basis`` are stated, the
            product is not a certified logical operator, or the code declares no
            logical observable the frame can read.

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
    if isinstance(rounds, bool) or not isinstance(rounds, Integral):
        raise TypeError("rounds must be an integer")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    configured_rounds = int(rounds)
    if product is not None:
        if not isinstance(product, Pauli):
            raise TypeError("product must be a Pauli operator or None")
        if readout_basis != "z":
            raise ValueError(
                "product and an explicit readout_basis were both stated, and they "
                "are two statements of the one readout frame; state the frame once"
            )
        certify_logical_product(code, product)
        x_readout = product.x_qubits
    else:
        basis = _checked_basis(readout_basis)
        x_readout = () if basis == "z" else tuple(code.data_qubits)
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
        observables=_observable_layout(code, x_readout=x_readout, product=product),
        x_readout_qubits=x_readout,
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
