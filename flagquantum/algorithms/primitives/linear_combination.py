"""The Pauli linear-combination block encoding, and the walk step it admits.

The spectral encoding in :mod:`flagquantum.algorithms.primitives.block_encoding` is built
from a dense matrix: it diagonalises the operator, emits one dense gate, and extracts the
matrix over its subnormalisation exactly. This module is the second implementation behind
the same two protocols, and it is the one an operator's own algebra reaches: it takes a
:class:`~flagquantum.algorithms.core.Hamiltonian`, a weighted sum of Pauli products, and
builds the encoding from the coefficients and the Pauli words alone. No matrix is formed
and nothing is diagonalised, so the subnormalisation is a number the caller's data already
determines rather than one an eigendecomposition has to measure.

**The construction.** With ``H = sum_j c_j P_j`` over ``m`` terms, write
``alpha = sum_j |c_j|``. The encoding uses an *index* register of
``k = max(1, ceil(log2 m))`` wires to name a term, and with ``Q = sum_j |j><j| (x) P_j``
the selection operator and ``Sigma`` the diagonal that carries the sign of each
coefficient, the unitary is

    U = prepare^dagger . Q . Sigma . prepare

where ``prepare`` maps ``|0...0>`` to ``sum_j sqrt(|c_j| / alpha) |j>`` over the ``m``
occupied index states and to zero on every index above ``m - 1``. On the flagged block
``Q`` and ``Sigma`` reduce to ``sum_j c_j P_j / alpha = H / alpha``, so the flagged block
of ``U`` is ``H`` over ``alpha`` and the encoding is a block encoding at that factor.
``sum_j |c_j|`` is at least the spectral norm of ``H``, since ``P_j`` is unitary and the
triangle inequality gives ``|H| <= sum_j |c_j|``, so every eigenvalue of ``H / alpha`` lies
in ``[-1, 1]``, which is the condition the definition needs.

That factor is deliberately **not** the Frobenius norm
:func:`~flagquantum.algorithms.primitives.block_encoding.subnormalisation` would pick, and
the two are not even ordered: the Frobenius norm of a Pauli product on ``n`` wires is
``2 ** (n / 2)``, so a one-wire sum can have a Frobenius norm above the sum of its
coefficient magnitudes. The spectral norm is the one the definition needs and it is the one
this factor dominates, and no matrix is formed to measure either.

**The sign is its own operator, and that is a measured constraint rather than a style
choice.** The obvious construction folds the sign of ``c_j`` into the prepared amplitude.
It cannot: :func:`~flagquantum.algorithms.primitives.state_preparation.arbitrary_state`
realises the magnitudes it is handed up to *one* phase for the whole vector, so it carries
no relative sign between two amplitudes. Asked for ``[1, -2]`` it returns the normalised
magnitudes with both entries sharing a phase, and the negative entry comes back positive.
So the preparation is over magnitudes only -- one ``arbitrary_state`` on the index register
-- and the sign is emitted as an explicit diagonal ``Sigma``, which flips the phase of
exactly the index states whose coefficient is negative.

**Two registers, and the reflection is on the first of them.** The flag register is
``num_ancilla = k + max(k - 2, 0)`` consecutive wires: the ``k`` index wires, followed by
the ``max(k - 2, 0)`` ladder ancillas that
:func:`~flagquantum.algorithms.primitives.oracle.append_multi_controlled_x` needs to build
a multi-controlled X with ``k`` controls. Every one of those ladder gates is uncomputed
before the gate returns, so the ladder wires enter and leave ``|0>`` and carry no
amplitude: the flag is the index register alone, and the walk step's reflection is
``2|0><0| - I`` on those ``k`` wires. The same layout is what a consumer written against
the protocol already builds, because ``ancilla`` names the *first* of the flag register's
wires and the register runs upward from it, which is the reading
:class:`~flagquantum.algorithms.primitives.block_encoding.BlockEncoding` documents.

**The walk step.** The phase flip composed with the encoding, ``W = R U`` for ``R`` the
reflection on the index register, is a qubitization: on the two-dimensional subspace
``span{|0>|v_j>, U|0>|v_j>}`` for an eigenvector ``|v_j>`` of ``H`` with eigenvalue
``lambda_j``, ``W`` is a rotation by ``arccos(lambda_j / alpha)``. So each
``lambda_j / alpha`` appears as the cosine of an eigenvalue of one walk step, twice. The
claim is stated that way and not as an equality of the two spectra, because the walk has
further eigenvalues: the ``2**k - m`` index states the preparation leaves empty are still
states of the register, and on those ``W`` acts without an encoded eigenvalue behind it.
An equality of whole spectra would be false for every ``m`` that is not a power of two.

**The adjoint walk step needs no daggered gate, but it does need a reversed select.**
``prepare`` is the one piece of ``U`` that is not self-adjoint, and ``unprepare`` is its exact
adjoint, so no gate has to be daggered anywhere. ``Sigma`` is a product of multi-controlled
phase flips on the index register, which commute, so it is its own adjoint as a product and
not only term by term. ``Q`` is not: each term's own factor is a Pauli product on the
operator's wires controlled by one index state and is therefore Hermitian, but the terms
share the ladder ancillas the controlled gates are built from, so the product of them is not
its own reverse. Measured at ``k = 3`` the two orders differ by ``2.0`` in their largest
entry, and they agree only to ``4.7e-17`` on the ladder-zero subspace, which is where the
encoding is defined. Emitting the terms forward and calling the result ``U^dagger`` leaves a
residual of ``1.66`` on the six-term sum the unit tests use -- not a rounding error. The
adjoint therefore emits the select's terms
in reverse order and leaves everything else as it is, and then the two compose to the
identity entrywise, which the unit tests assert against the dense unitary.

**What is deliberately absent, and is owned rather than silent.** The preparation is a
dense classical precomputation over ``2**k`` amplitudes and the select is ``m``
multi-controlled Paulis, one per term, so this is a demonstration-scale encoding and not a
gate-efficient synthesis: there is no amplitude amplification, no uniform-preparation
trick, and no claim about the gate count of the family. A **complex** coefficient is
refused by name rather than reduced to its real part, and the reason is the one measured
above: the preparation primitive carries no relative phase, so a phase-carrying amplitude
cannot be prepared and the sum would not be the operator the caller named. There is no
``alpha`` parameter, because the factor is the sum of the coefficient magnitudes and is
determined by the data; a caller who wants a larger subnormalisation scales the
Hamiltonian, which scales the encoded block by the same factor. And nothing here is
exported through the Stable Core, so no public API change follows from this module.

**The register parameter is spelled ``qubits``, and the flag register names its first
wire.** This package is migrating every public ``wire``-named parameter to its qubit-named
spelling, and ``tools/check_qubit_vocabulary.py`` fails closed when a new public parameter
grows the ledger. Nothing in this module adds to it: the register is ``qubits``, the flag
register is named by its first wire, and ``contracts/qubit-vocabulary-contract.toml``
owns the ledger.

This unit carries no performance, capacity, or hardware claim of its own, and it does not
select a runtime.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from ...circuit import Circuit
from ...errors import CapabilityError
from .block_encoding import _FLAG_REFLECTION
from .oracle import append_multi_controlled_x
from .state_preparation import arbitrary_state

if TYPE_CHECKING:
    from ..core import Hamiltonian

__all__ = [
    "LinearCombinationEncoding",
]

# The basis change that turns a Pauli axis into the axis the multi-controlled X already
# carries: ``Y = s X s^dagger``, so a ``s^dagger`` before the gate and an ``s`` after it
# conjugate the controlled-X into a controlled-Y. ``X`` needs neither and is absent from
# the table, and ``Z`` is conjugated by ``h`` on both sides, which is its own inverse.
_BASIS_CHANGE = {
    "y": ("sdg", "s"),
    "z": ("h", "h"),
}


def _zero_positions(index: int, width: int) -> tuple[int, ...]:
    """Return the bit positions of ``index`` that hold a zero, most significant first.

    The index register holds the term number in binary with the most significant bit on
    the lowest-addressed wire, which is the convention :class:`~flagquantum.circuit.Circuit`
    uses for a basis state. A bit position is returned when the corresponding wire is
    ``|0>``, so that flipping exactly those wires moves ``|index>`` onto the all-ones
    state a multi-controlled gate fires on.

    Args:
        index: The term number, below ``2 ** width``.
        width: The number of index wires.

    Returns:
        The positions, in ascending wire order.
    """

    return tuple(
        position
        for position in range(width)
        if not (index >> (width - 1 - position)) & 1
    )


def _append_flips(circuit: Circuit, wires: Sequence[int]) -> None:
    """Append an ``x`` to each wire, in place.

    Every use of this helper is its own inverse, so the same call both moves a register
    onto the state a controlled gate fires on and moves it back, and there is one spelling
    of the pair rather than two that could drift apart.
    """

    for wire in wires:
        circuit.gate("x", wire)


def _append_multi_controlled_z(
    circuit: Circuit, controls: Sequence[int], target: int, ancillas: Sequence[int]
) -> None:
    """Append a ``z`` on ``target`` exactly when every wire in ``controls`` is set.

    ``H X H = Z``, so a phase flip on the target is the multi-controlled X it already
    carries, conjugated by ``h`` on the target alone. With no controls the conjugation is
    not needed and the gate is a bare ``z``, which is the ``k == 1`` case of every caller
    here.
    """

    if not controls:
        circuit.gate(_FLAG_REFLECTION, target)
        return
    circuit.gate("h", target)
    append_multi_controlled_x(
        circuit,
        controls,
        target,
        ancillas=ancillas[: max(len(controls) - 2, 0)],
    )
    circuit.gate("h", target)


def _real_weight(coefficient: Any, *, index: int) -> float:
    """Return the coefficient as a finite real number, or refuse it by name.

    The subnormalisation is a sum of magnitudes over these coefficients, so a coefficient
    that is not a number, not finite, or not real has no sum to contribute and is refused
    where it is read rather than reduced to something that looks like one. The refusal is
    a ``CapabilityError`` for a complex coefficient, because that is a construction this
    module does not have rather than a value that cannot exist, and a ``ValueError`` for
    the rest.
    """

    if isinstance(coefficient, bool) or isinstance(
        coefficient, (str, bytes, bytearray)
    ):
        raise ValueError(
            f"term {index} has coefficient {coefficient!r}, which is not a number; a "
            "coefficient is the weight the term carries in the sum being encoded"
        )
    if isinstance(coefficient, torch.Tensor):
        if coefficient.numel() != 1:
            raise ValueError(
                f"term {index} has a coefficient holding {coefficient.numel()} values, "
                "and a Hamiltonian term carries one"
            )
        if coefficient.is_complex():
            raise CapabilityError(
                f"term {index} has a complex coefficient, and this encoding prepares "
                "magnitudes only: its preparation primitive realises the amplitudes it "
                "is handed up to one phase for the whole vector, so it cannot carry the "
                "relative phase a complex coefficient names"
            )
        value = float(coefficient.detach().to(torch.float64).reshape(()))
    elif isinstance(coefficient, complex):
        if coefficient.imag != 0.0:
            raise CapabilityError(
                f"term {index} has coefficient {coefficient!r}, and this encoding "
                "prepares magnitudes only: its preparation primitive realises the "
                "amplitudes it is handed up to one phase for the whole vector, so it "
                "cannot carry the relative phase a complex coefficient names"
            )
        value = coefficient.real
    else:
        try:
            value = float(coefficient)
        except (TypeError, ValueError):
            raise ValueError(
                f"term {index} has coefficient {coefficient!r}, which is not a number; a "
                "coefficient is the weight the term carries in the sum being encoded"
            ) from None
    if not math.isfinite(value):
        raise ValueError(
            f"term {index} has coefficient {value}, which is not finite; the "
            "subnormalisation is a sum of magnitudes, and a sum with a value that is not "
            "a number in it is not one"
        )
    return value


def _require_terms(
    hamiltonian: Any,
) -> tuple[tuple[float, tuple[tuple[int, str], ...]], ...]:
    """Return the validated ``(coefficient, ops)`` table of a Hamiltonian, or refuse it.

    The encoding is built from ``H = sum_j c_j P_j``, so this is where every property the
    construction depends on is established once: there is at least one term, every
    coefficient is a finite real number, and every term names Pauli axes on wires that the
    encoding knows how to apply. The table is the reading the value object caches; the
    Hamiltonian stays the one place the sum is written down.
    """

    terms = getattr(hamiltonian, "terms", None)
    if terms is None:
        raise ValueError(
            f"the encoding is built from a Hamiltonian, and "
            f"{type(hamiltonian).__name__} has no terms; a Pauli sum is what the "
            "construction is a linear combination of"
        )
    table = []
    for index, term in enumerate(terms):
        ops = getattr(term, "ops", None)
        if ops is None:
            raise ValueError(
                f"term {index} does not carry Pauli operators; a Hamiltonian term is a "
                "coefficient and the Pauli product it weights"
            )
        operators = []
        for wire, axis in ops:
            if axis == "i":
                continue
            if axis not in ("x", "y", "z"):
                raise ValueError(
                    f"term {index} names Pauli axis {axis!r} on wire {wire}; this "
                    "encoding applies X, Y and Z, and an axis outside them names an "
                    "operator the circuit layer does not carry"
                )
            if isinstance(wire, bool) or not isinstance(wire, int) or wire < 0:
                raise ValueError(
                    f"term {index} names wire {wire!r}, which is not a wire index; the "
                    "encoding places each Pauli product on the register the caller gives "
                    "it"
                )
            operators.append((wire, axis))
        table.append((_real_weight(term.coefficient, index=index), tuple(operators)))
    if not table:
        raise ValueError(
            "the Hamiltonian carries no terms, and this encoding is a linear combination "
            "of at least one: a sum with nothing in it has no coefficient to prepare"
        )
    return tuple(table)


@dataclass(frozen=True, eq=False)
class LinearCombinationEncoding:
    """A block encoding of a Pauli sum, by linear combination of unitaries.

    The flagged block of the circuit :meth:`append_apply` emits is ``H / alpha`` for
    ``alpha`` the sum of the coefficient magnitudes, which is at least the spectral norm of
    ``H`` and so satisfies the definition of a block encoding at that factor. The same
    object satisfies :class:`~flagquantum.algorithms.primitives.block_encoding.WalkEncoding`,
    because the construction is a qubitization: see the module docstring for the walk
    identity and for what it does and does not say.

    The factor is computed from the coefficients and is not a parameter. A caller who wants
    a larger one scales the Hamiltonian, which scales the encoded block by the same factor.

    Attributes:
        hamiltonian: The weighted sum of Pauli products being encoded. It is read once,
            when the encoding is built, and every derived number is checked against it
            then; the object is held so that a reader can see which sum a given encoding
            belongs to, and it is not read again while a circuit is being emitted.
    """

    hamiltonian: Hamiltonian
    # The validated table ``_require_terms`` returned, one ``(coefficient, ops)`` pair per
    # term. It is the reading the append helpers use, so a circuit is emitted from numbers
    # that were checked when the encoding was built rather than from an object a caller can
    # still hold and change.
    _terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...] = field(
        init=False, repr=False, compare=False
    )
    # ``sum_j |c_j|``, the subnormalisation. It is at least the spectral norm of ``H``, so
    # every eigenvalue of the encoded matrix lies in ``[-1, 1]``.
    alpha: float = field(init=False)
    # The number of wires the index register needs to name one of ``m`` terms.
    index_width: int = field(init=False)
    # The number of wires the encoded operator acts on, read off the Pauli words rather
    # than from a declared width, so the two cannot disagree.
    num_system: int = field(init=False)
    # The preparation and its inverse, built once on wires ``0..k-1`` and composed onto the
    # caller's index register at append time. Building them once is what keeps the value
    # object independent of where in a circuit the encoding is placed.
    _prepare: Circuit = field(init=False, repr=False, compare=False)
    _unprepare: Circuit = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        table = _require_terms(self.hamiltonian)
        magnitudes = [abs(coefficient) for coefficient, _ in table]
        alpha = math.fsum(magnitudes)
        if alpha == 0.0:
            raise ValueError(
                "every coefficient of the Hamiltonian is zero, so the subnormalisation "
                "is zero and there is no block to normalise; a sum whose terms all "
                "cancel their own weight is not an operator to encode"
            )
        index_width = max(1, (len(table) - 1).bit_length())
        num_system = (
            max(max((wire for wire, _ in ops), default=-1) for _, ops in table) + 1
        )
        amplitudes = [math.sqrt(magnitude / alpha) for magnitude in magnitudes]
        amplitudes += [0.0] * ((1 << index_width) - len(table))
        prepare = arbitrary_state(
            torch.tensor(amplitudes, dtype=torch.complex128),
            wires=list(range(index_width)),
        )
        object.__setattr__(self, "_terms", table)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "index_width", index_width)
        object.__setattr__(self, "num_system", num_system)
        object.__setattr__(self, "_prepare", prepare)
        object.__setattr__(self, "_unprepare", prepare.adjoint())

    @property
    def num_ancilla(self) -> int:
        """The number of wires in the flag register: the index wires and the ladder's.

        The flag register is ``num_ancilla`` consecutive wires starting at the ``ancilla``
        a caller names, and the encoded block is the one on which every one of them is
        ``|0>``. The last ``max(index_width - 2, 0)`` of them are the ladder ancillas the
        multi-controlled X is built from; they are uncomputed before every gate returns
        and carry no amplitude of their own.
        """

        return self.index_width + max(self.index_width - 2, 0)

    @property
    def terms(self) -> tuple[tuple[float, tuple[tuple[int, str], ...]], ...]:
        """The encoded sum's ``(coefficient, ops)`` pairs, in the order they were given.

        Each pair is a finite real coefficient and a tuple of ``(wire, axis)`` operators
        with the identity dropped, which is the reading
        :class:`~flagquantum.algorithms.core.HamiltonianTerm` normalises to. It is the
        table the circuit is emitted from.
        """

        return self._terms

    def append_apply(
        self, circuit: Circuit, *, ancilla: int, qubits: Sequence[int]
    ) -> None:
        """Append the encoding itself, as ``prepare^dagger Q Sigma prepare``.

        Args:
            circuit: The circuit to append to, which must be wide enough to hold the flag
                register and the encoded operator's own register.
            ancilla: The first wire of the flag register, which runs upward for
                :attr:`num_ancilla` wires.
            qubits: The :attr:`num_system` wires the encoded operator acts on, disjoint
                from the flag register.

        Raises:
            ValueError: If ``ancilla`` is not a wire index; if the register is not
                :attr:`num_system` wires; if a wire appears twice; if the flag register
                overlaps the register; or if the circuit is too narrow to hold both.
        """

        index, scratch = _require_registers(self, ancilla, qubits, circuit=circuit)
        _append_lcu(circuit, self, index, scratch, qubits, adjoint=False)

    def append_walk_step(
        self, circuit: Circuit, *, ancilla: int, qubits: Sequence[int]
    ) -> None:
        """Append one qubitization walk step, which is the encoding then the reflection.

        The step is ``W = R U`` for ``R`` the reflection ``2|0><0| - I`` on the index
        register. Its eigenvalues are the phases whose cosines are, twice each, the
        eigenvalues of the encoded operator over ``alpha``; the module docstring records
        which further eigenvalues it has and why an equality of whole spectra would be
        false.

        Args:
            circuit: The circuit to append to.
            ancilla: The first wire of the flag register.
            qubits: The wires the encoded operator acts on.

        Raises:
            ValueError: As :meth:`append_apply`.
        """

        index, scratch = _require_registers(self, ancilla, qubits, circuit=circuit)
        _append_lcu(circuit, self, index, scratch, qubits, adjoint=False)
        _append_reflection(circuit, index, scratch)

    def append_adjoint_walk_step(
        self, circuit: Circuit, *, ancilla: int, qubits: Sequence[int]
    ) -> None:
        """Append the walk step's inverse, which is ``U^dagger R``.

        ``prepare`` is the only piece of ``U`` that is not self-adjoint, so the inverse of
        ``R U`` is the reflection followed by the same pieces with the select's terms in
        reverse rather than a second sequence of daggered gates. Appending this after
        :meth:`append_walk_step` on the same wires returns the circuit to where it was, which
        the unit tests assert entrywise against the dense unitary as well.

        Args:
            circuit: The circuit to append to.
            ancilla: The first wire of the flag register.
            qubits: The wires the encoded operator acts on.

        Raises:
            ValueError: As :meth:`append_apply`.
        """

        index, scratch = _require_registers(self, ancilla, qubits, circuit=circuit)
        _append_reflection(circuit, index, scratch)
        _append_lcu(circuit, self, index, scratch, qubits, adjoint=True)


def _require_registers(
    encoding: LinearCombinationEncoding,
    ancilla: Any,
    qubits: Any,
    *,
    circuit: Circuit,
) -> tuple[list[int], list[int]]:
    """Check the two registers, and return the index and ladder wires, or refuse.

    Every condition is checked before a single gate is emitted, so a caller whose wires are
    wrong keeps an empty circuit rather than half an encoding. The flag register is
    derived here once, which is why the append methods share this instead of each
    computing ``range`` and hoping the three agree.
    """

    if isinstance(ancilla, bool) or not isinstance(ancilla, int):
        raise ValueError(
            f"ancilla must be the first wire of the flag register, got {ancilla!r}; the "
            f"register is {encoding.num_ancilla} consecutive wires and this argument "
            "names where it starts"
        )
    wires = tuple(qubits)
    width = len(wires)
    if width != encoding.num_system:
        raise ValueError(
            f"the register must be {encoding.num_system} wire(s) for this encoding, "
            f"got {width}"
        )
    if len(set(wires)) != width:
        raise ValueError(
            f"the register must be distinct, got {list(wires)}; an operator acts on each "
            "wire once"
        )
    flag = list(range(ancilla, ancilla + encoding.num_ancilla))
    overlap = sorted(set(flag) & set(wires))
    if overlap:
        raise ValueError(
            f"the flag register {flag} overlaps the operator's own wires {list(wires)} at "
            f"{overlap}; the flag is not part of the register the encoded operator acts on"
        )
    highest = max(flag[-1], *wires)
    if highest >= circuit.n_wires:
        raise ValueError(
            f"the circuit has {circuit.n_wires} wire(s) and this encoding reaches wire "
            f"{highest}; build the circuit on at least {highest + 1} wires"
        )
    return flag[: encoding.index_width], flag[encoding.index_width :]


def _append_lcu(
    circuit: Circuit,
    encoding: LinearCombinationEncoding,
    index: Sequence[int],
    scratch: Sequence[int],
    qubits: Sequence[int],
    *,
    adjoint: bool,
) -> None:
    """Append ``prepare^dagger Q Sigma prepare``, with the select's term order reversed.

    A circuit composes its instructions in reverse, so the sequence emitted for the product
    ``A B C D`` is ``D`` first and ``A`` last: the preparation is emitted first and its
    inverse last. ``prepare`` is the only piece here that is not self-adjoint, and
    ``unprepare`` is its exact adjoint, so the adjoint of the whole product is
    ``prepare^dagger Sigma Q^dagger prepare``. ``Sigma`` is its own adjoint and commutes with
    ``Q`` -- measured, their commutator is ``1.1e-16`` -- so neither the order between the two
    nor the signs themselves change between the two directions. ``Q`` is not its own adjoint:
    each term's factor is Hermitian, but the terms share the ladder ancillas, so the product
    is not the product reversed. That reversal is the only difference between the two
    directions, and it is what makes ``U^dagger`` exact as an operator rather than exact on
    the flagged subspace alone.
    """

    circuit.compose(encoding._prepare, qubits=list(index))
    _append_signs(circuit, encoding, index, scratch)
    _append_select(circuit, encoding, index, scratch, qubits, adjoint=adjoint)
    circuit.compose(encoding._unprepare, qubits=list(index))


def _append_signs(
    circuit: Circuit,
    encoding: LinearCombinationEncoding,
    index: Sequence[int],
    scratch: Sequence[int],
) -> None:
    """Append the diagonal that carries the sign of every negative coefficient.

    One multi-controlled phase flip per negative term, each conditioned on the index
    register holding that term's number. A term with a positive coefficient contributes
    nothing, so the gate count follows the data rather than the term count alone. Every gate
    here is diagonal in the index register, so the product is its own adjoint and the term
    order is free: the adjoint emits exactly these gates.
    """

    for position in range(len(encoding.terms)):
        coefficient = encoding.terms[position][0]
        if coefficient >= 0.0:
            continue
        flips = [index[bit] for bit in _zero_positions(position, encoding.index_width)]
        _append_flips(circuit, flips)
        _append_multi_controlled_z(circuit, index[:-1], index[-1], scratch)
        _append_flips(circuit, flips)


def _append_select(
    circuit: Circuit,
    encoding: LinearCombinationEncoding,
    index: Sequence[int],
    scratch: Sequence[int],
    qubits: Sequence[int],
    *,
    adjoint: bool,
) -> None:
    """Append the selection ``sum_j |j><j| (x) P_j``, one term at a time.

    The index register is moved onto the all-ones state a multi-controlled gate fires on,
    the term's Pauli product is applied to the operator's register through one
    multi-controlled X per non-identity axis, and the index register is moved back. Each
    term's factor is Hermitian, but the terms share the ladder ancillas the multi-controlled
    gates are built from, so the product is not its own reverse even though the blocks act on
    disjoint index states. The emission order is the caller's term order forward and its
    reverse for the adjoint, which is what makes the second direction the exact adjoint; the
    two orders agree on the ladder-zero subspace, where the encoding is defined, and the
    module docstring records the measured residual of the reversal being skipped.
    """

    controls = max(encoding.index_width - 2, 0)
    order = list(range(len(encoding.terms)))
    if adjoint:
        order.reverse()
    for position in order:
        ops = encoding.terms[position][1]
        flips = [index[bit] for bit in _zero_positions(position, encoding.index_width)]
        _append_flips(circuit, flips)
        for wire, axis in ops:
            target = qubits[wire]
            change = _BASIS_CHANGE.get(axis)
            if change is not None:
                circuit.gate(change[0], target)
            append_multi_controlled_x(
                circuit, index, target, ancillas=scratch[:controls]
            )
            if change is not None:
                circuit.gate(change[1], target)
        _append_flips(circuit, flips)


def _append_reflection(
    circuit: Circuit, index: Sequence[int], scratch: Sequence[int]
) -> None:
    """Append the reflection ``2|0><0| - I`` on the index register.

    ``X C_kZ X`` is ``I - 2|0><0|`` -- the negative of the reflection the walk is defined
    with -- and the two differ by a global phase of ``-1``, which is exactly what a walk
    step's cosine spectrum is sensitive to, so the phase is emitted rather than dropped.
    Four gates on one index wire compose to ``-I``, since ``X Z X Z = -(Z Z) = -I``. One
    index wire needs none of that: the reflection on a single wire is ``Z`` itself, which
    is the gate the spectral encoding's single flag wire carries.
    """

    if len(index) == 1:
        circuit.gate(_FLAG_REFLECTION, index[0])
        return
    _append_flips(circuit, index)
    _append_multi_controlled_z(circuit, index[:-1], index[-1], scratch)
    _append_flips(circuit, index)
    circuit.gate(_FLAG_REFLECTION, index[0])
    circuit.gate("x", index[0])
    circuit.gate(_FLAG_REFLECTION, index[0])
    circuit.gate("x", index[0])
