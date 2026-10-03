"""Trotterized time evolution of a Pauli-sum Hamiltonian.

:func:`trotter_circuit` turns a weighted Pauli sum into the circuit a product
formula applies, so a time-evolution workload becomes an ordinary
:class:`flagquantum.Circuit` that the existing compiler, runtime, and gradient
paths already handle.  The primitive underneath it,
:func:`pauli_exponential_circuit`, is the exact circuit for ``exp(-i * theta * P)``
of one Pauli word ``P``.

The product formula is an approximation and the returned circuit says so by
construction.  ``order=1`` is the Lie-Trotter product applied once per step, whose
defect from ``exp(-i * t * H)`` is second order in the step length; ``order=2`` is
the symmetric (Strang) product, whose defect is third order.  The tests measure
both orders against ``torch.matrix_exp`` rather than asserting the rate here.
Nothing in this module bounds the error for a caller's Hamiltonian: a bound needs
a commutator norm, and a commutator norm belongs to the caller.

Two limits are deliberate and stated rather than worked around.  A term that is a
multiple of the identity exponentiates to a global phase, and no gate in this
repository applies one, so such a term is refused instead of dropped.  And a
product formula's value depends on the order its terms are declared in, so the
declared order is preserved rather than sorted into a canonical one.

``exp_pauli`` is not here.  CUDA-Q's ``exp_pauli`` applies ``exp(-i * theta * P)``
as one opaque instruction whose decomposition the compiler owns; this module emits
the decomposition directly, as a basis change and a CX ladder, so the result is an
ordinary circuit the existing passes can inspect, route for a target, and
differentiate through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..errors import CapabilityError
from ..simulation.pauli import finite_rotation_angle, pauli_word_operators

if TYPE_CHECKING:
    from ..circuit import Circuit
    from .core import Hamiltonian

TROTTER_ORDERS = (1, 2)
# The product-formula orders this module builds.
#
# Order 1 composes the term exponentials forward once per step; order 2 composes them
# forward and then backward over half a step each, which is the symmetric composition.
# Higher orders -- the Yoshida and Suzuki recursions -- are not built: each costs more
# exponentials per step than the accuracy it buys, and neither is needed until a caller
# measures that the second order is not accurate enough.

_BASIS_CHANGE = {
    "X": (("h",), ("h",)),
    "Y": (("sdg", "h"), ("h", "s")),
    "Z": ((), ()),
}
# How each Pauli axis is turned into the axis a rotation about ``z`` acts on.
#
# A ``z`` rotation is diagonal only in the computational basis, so the other two axes
# are read in a basis where their own operator is the diagonal one: ``H`` maps ``X`` to
# ``Z``, and ``sdg`` followed by ``H`` maps ``Y`` to ``Z``. The forward and backward
# sequences are separate because ``H`` is its own inverse while ``sdg`` is undone by
# ``s``. ``z`` needs neither, which is what keeps a diagonal term free of basis gates.


def _real_coefficient(coefficient: Any, *, index: int) -> float | torch.Tensor:
    """Return the coefficient a unitary time step needs, or refuse it.

    ``exp(-i * h * c * P)`` is unitary only when ``c`` is real, so a coefficient with a
    nonzero imaginary part describes a non-Hermitian generator and is refused by name
    instead of being silently reduced to its real part. A tensor coefficient is returned
    unchanged, because it is the one form whose gradient can still flow into the
    circuit's rotation angles.
    """

    if isinstance(coefficient, torch.Tensor):
        if coefficient.numel() != 1:
            raise ValueError(
                f"term {index} has a coefficient holding {coefficient.numel()} "
                "values, and a Hamiltonian term carries one"
            )
        if coefficient.is_complex():
            raise CapabilityError(
                f"term {index} has a complex coefficient, and a complex coefficient "
                "makes the generator non-Hermitian, so the time step it names is not "
                "unitary"
            )
        return coefficient
    if isinstance(coefficient, bool) or isinstance(
        coefficient, (str, bytes, bytearray)
    ):
        raise ValueError(
            f"term {index} has coefficient {coefficient!r}, which is not a number"
        )
    try:
        value = complex(coefficient)
    except (TypeError, ValueError):
        raise ValueError(
            f"term {index} has coefficient {coefficient!r}, which is not a number"
        ) from None
    if value.imag != 0.0:
        raise CapabilityError(
            f"term {index} has coefficient {value!r}, whose imaginary part makes the "
            "generator non-Hermitian and the time step it names not unitary"
        )
    return value.real


def _require_steps(steps: Any) -> int:
    """Read a step count, refusing anything that does not denote a positive count."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    return steps


def _require_order(order: Any) -> int:
    """Read a product-formula order, refusing one this module does not build."""

    if (
        isinstance(order, bool)
        or not isinstance(order, int)
        or order not in TROTTER_ORDERS
    ):
        raise ValueError(
            f"order must be one of {', '.join(str(item) for item in TROTTER_ORDERS)}, "
            f"got {order!r}"
        )
    return order


def _supported_operators(
    word: str,
    targets: list[int] | tuple[int, ...],
    width: int,
) -> list[tuple[int, str]]:
    """Read a Pauli word into the supported factors a rotation about ``z`` can use."""

    operators = [
        (wire, axis)
        for wire, axis in pauli_word_operators(word, targets)
        if axis != "I"
    ]
    # A negative target never reaches this line: ``pauli_word_operators`` refuses one
    # before returning, so the only wire this check can catch is one past the register.
    outside = sorted({wire for wire, _ in operators if wire >= width})
    if outside:
        raise ValueError(
            f"Pauli word {word!r} addresses wire(s) {outside} outside a {width}-wire "
            "circuit"
        )
    if not operators:
        raise CapabilityError(
            f"Pauli word {word!r} has no support, so its exponential is a global "
            "phase exp(-i * theta); FlagQuantum has no gate that applies a global "
            "phase, so the operator has no circuit here"
        )
    return operators


def _append_exponential(
    circuit: Circuit,
    angle: float | torch.Tensor,
    operators: list[tuple[int, str]],
) -> None:
    """Append ``exp(-i * angle * P)`` for an already-read product ``P``."""

    for wire, axis in operators:
        for name in _BASIS_CHANGE[axis][0]:
            circuit.gate(name, (wire,))
    anchor = operators[0][0]
    # Every CX shares ``anchor`` as its target and its controls are distinct, so the
    # ladder's gates commute and the same sequence undoes it.
    for wire, _ in operators[1:]:
        circuit.gate("cx", (wire, anchor))
    circuit.gate("rz", (anchor,), theta=2.0 * angle)
    for wire, _ in operators[1:]:
        circuit.gate("cx", (wire, anchor))
    for wire, axis in operators:
        for name in _BASIS_CHANGE[axis][1]:
            circuit.gate(name, (wire,))


def pauli_exponential_circuit(
    theta: float,
    word: str,
    targets: list[int] | tuple[int, ...],
    n_qubits: int,
    *,
    dtype: torch.dtype | None = None,
) -> Circuit:
    """Return the circuit whose unitary is ``exp(-i * theta * P)``.

    ``word`` and ``targets`` are read positionally, exactly as
    :func:`flagquantum.simulation.pauli.exponential_pauli_operator` reads them, so
    character ``i`` of the word acts on ``targets[i]`` and together they name the
    product ``P``.  An ``I`` character consumes its target and contributes no support,
    so an all-identity word names the identity, whose exponential is a global phase
    rather than a circuit; that input is refused.

    The circuit is a basis change into the axis a ``z`` rotation is diagonal on, a CX
    ladder onto the first supported wire, one ``rz`` of twice the angle, and the ladder
    and the basis change undone.  It is exact for every angle, up to one global phase,
    which this repository's gate set cannot express.

    Args:
        theta: The rotation angle. A bool, a string, an infinite value, and a NaN are
            refused before anything is built.
        word: One character per entry of ``targets``, drawn from ``X``, ``Y``, ``Z``,
            and ``I``.
        targets: The wires the word acts on, in the order the word is written.
        n_qubits: The width of the returned circuit.
        dtype: The circuit's complex dtype. The default is the runtime configuration's,
            which is complex64 unless configured otherwise.

    Returns:
        A :class:`flagquantum.Circuit` on ``n_qubits`` wires.

    Raises:
        ValueError: ``word`` and ``targets`` disagree, a character is not a Pauli label,
            a target repeats or falls outside ``n_qubits``, or ``theta`` is not a finite
            real number.
        CapabilityError: ``word`` holds no support, so the requested operator is a
            global phase.

    Examples:
        >>> from flagquantum.algorithms.trotter import pauli_exponential_circuit
        >>> circuit = pauli_exponential_circuit(0.5, "ZZ", (0, 1), 2)
        >>> [instruction.name for instruction in circuit.to_ir().instructions]
        ['cx', 'rz', 'cx']
    """

    angle = finite_rotation_angle(theta)
    width = int(n_qubits)
    if width < 1:
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits!r}")
    operators = _supported_operators(word, targets, width)

    from ..circuit import Circuit

    circuit = Circuit(width, dtype=dtype)
    _append_exponential(circuit, angle, operators)
    return circuit


def _prepare_terms(
    hamiltonian: Hamiltonian,
    width: int,
) -> list[tuple[float | torch.Tensor, list[tuple[int, str]]]]:
    """Read every term into the coefficient and product one time step needs.

    Every refusal a term can raise happens here, before a circuit exists, so a
    Hamiltonian whose terms cannot be evolved is refused without a half-built circuit
    and without a register-size error standing in for the real reason.
    """

    prepared: list[tuple[float | torch.Tensor, list[tuple[int, str]]]] = []
    for index, term in enumerate(hamiltonian.terms):
        if not term.ops:
            raise CapabilityError(
                f"term {index} is a multiple of the identity, whose exponential is a "
                "global phase rather than a circuit; FlagQuantum has no gate that "
                "applies one, and no measurement distinguishes it, so drop the term"
            )
        coefficient = _real_coefficient(term.coefficient, index=index)
        word = "".join(axis.upper() for _, axis in term.ops)
        targets = tuple(wire for wire, _ in term.ops)
        prepared.append((coefficient, _supported_operators(word, targets, width)))
    return prepared


def _append_term(
    circuit: Circuit,
    coefficient: float | torch.Tensor,
    operators: list[tuple[int, str]],
    weight: float | torch.Tensor,
) -> None:
    """Append one prepared term's exponential of ``weight`` times its coefficient."""

    _append_exponential(circuit, weight * coefficient, operators)


def trotter_circuit(
    hamiltonian: Hamiltonian,
    time: float,
    *,
    steps: int = 1,
    order: int = 2,
    n_qubits: int | None = None,
    dtype: torch.dtype | None = None,
) -> Circuit:
    """Return the circuit that applies a product formula to a Pauli sum.

    The Hamiltonian is read as ``H = sum_j c_j P_j`` and the returned circuit applies
    the product of the term exponentials once per step, so the whole circuit
    approximates ``exp(-i * time * H)``.  ``order=1`` composes the terms forward over
    the whole step; ``order=2`` composes them forward and then backward over half a
    step each, which cancels the leading defect.  How far the returned circuit is from
    the exact evolution is not bounded here: a bound needs a commutator norm, which is
    the caller's to supply.

    The terms are applied in the order the Hamiltonian declares them.  A product
    formula's value depends on that order, so it is preserved rather than sorted, and
    two Hamiltonians holding the same terms in a different order produce different
    circuits.

    Args:
        hamiltonian: The Hamiltonian to evolve, as
            :class:`flagquantum.algorithms.Hamiltonian`.
        time: The evolution time, a finite real number.
        steps: How many steps to divide ``time`` into, a positive integer.
        order: The product-formula order, one of :data:`TROTTER_ORDERS`.
        n_qubits: The width of the returned circuit. The default is the narrowest
            register the Hamiltonian's terms fit in, which is ``hamiltonian.n_wires``.
        dtype: The circuit's complex dtype. The default is the runtime configuration's,
            which is complex64 unless configured otherwise.

    Returns:
        A :class:`flagquantum.Circuit` approximating ``exp(-i * time * hamiltonian)``.

    Raises:
        ValueError: ``time`` is not a finite real number, ``steps`` is not a positive
            integer, ``order`` is not a supported order, ``n_qubits`` is narrower than a
            term needs, or a term's coefficient is not a real number.
        CapabilityError: A term is a multiple of the identity, or its coefficient has a
            nonzero imaginary part.

    Examples:
        One first-order step of a single ``Z`` term is exactly ``RZ(2 * time)``:

        >>> from flagquantum.algorithms import Hamiltonian, pauli_term
        >>> from flagquantum.algorithms.trotter import trotter_circuit
        >>> hamiltonian = Hamiltonian([pauli_term(1.0, "Z", 0)])
        >>> circuit = trotter_circuit(hamiltonian, 0.25, order=1)
        >>> [instruction.name for instruction in circuit.to_ir().instructions]
        ['rz']
    """

    time_value = finite_rotation_angle(time)
    step_count = _require_steps(steps)
    step_order = _require_order(order)
    required = int(hamiltonian.n_wires)
    width = required if n_qubits is None else int(n_qubits)
    prepared = _prepare_terms(hamiltonian, width)

    from ..circuit import Circuit

    circuit = Circuit(width, dtype=dtype)
    step_time = time_value / step_count
    for _ in range(step_count):
        if step_order == 1:
            for coefficient, operators in prepared:
                _append_term(circuit, coefficient, operators, step_time)
            continue
        half = step_time / 2.0
        for coefficient, operators in prepared:
            _append_term(circuit, coefficient, operators, half)
        # The symmetric composition repeats the same term exponentials in reverse, so
        # the leading defect cancels and the step is accurate to the third order.
        for coefficient, operators in reversed(prepared):
            _append_term(circuit, coefficient, operators, half)
    return circuit


__all__ = (
    "TROTTER_ORDERS",
    "pauli_exponential_circuit",
    "trotter_circuit",
)
