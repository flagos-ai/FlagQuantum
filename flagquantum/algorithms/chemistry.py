"""Chemistry ansatz construction for FlagQuantum.

This module owns the ansatz half of a chemistry workload: which excitations exist
for a given number of electrons and spin orbitals, the fermionic generator each
excitation exponentially evolves, the exact circuit that realizes that
exponential, and the Trotterized product that is UCCSD. It composes the
fermionic algebra in :mod:`flagquantum.observables.fermion` with the public
:class:`flagquantum.Circuit`; the circuit is the only thing it returns, and every
identity it relies on is checked against that fermionic algebra in the tests
rather than asserted here.

This module does not read molecular integrals, drive a chemistry package, or
solve a Hartree-Fock problem. A caller supplies the integrals or the Hamiltonian;
FlagQuantum supplies the state-preparation circuits.

The gate sequences are the ones CUDA-Q's ``singleExcitation``,
``doubleExcitation``, ``uccsd`` and ``hwe`` kernels apply, so a workload built
around them transfers. Two details differ deliberately:

* Every excitation here requires its occupied indices to sit below its virtual
  ones, which is the only arrangement the UCCSD excitation lists produce.
  CUDA-Q accepts the opposite order and emits a circuit whose conditional
  ladders are empty, so it silently becomes a local rotation instead of an
  excitation; FlagQuantum refuses that input.
* :func:`uccsd_excitations` fills the lowest ``n_electrons`` spin orbitals for
  every electron count. CUDA-Q's ``get_uccsd_excitations`` divides by two before
  rounding up, so it agrees here whenever the electron count is even and places
  an occupied index above a virtual one when it is odd.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Union

import torch

from ..circuit import Circuit
from ..core.runtime_config import get_runtime_config
from ..observables.fermion import FermionOperator, annihilate, create

__all__ = (
    "UCCSDExcitations",
    "coupler_hardware_efficient_ansatz",
    "coupler_hardware_efficient_parameter_count",
    "double_excitation",
    "excitation_operator",
    "single_excitation",
    "uccsd_ansatz",
    "uccsd_excitations",
    "uccsd_factors",
)

Angle = Union[float, torch.Tensor]
"""A rotation angle: a Python number or a scalar tensor that keeps its graph."""

_HALF_PI = math.pi / 2.0
"""The basis-change angle both excitation sequences use."""


def _count(subject: Any, noun: str, *, minimum: int = 0) -> int:
    """Return ``subject`` as an integer count, or refuse it by name."""

    if isinstance(subject, (bool, str)):
        raise ValueError(f"{noun} must be an integer, got {subject!r}")
    try:
        value = operator.index(subject)
    except TypeError:
        raise ValueError(f"{noun} must be an integer, got {subject!r}") from None
    if value < minimum:
        raise ValueError(f"{noun} must be at least {minimum}, got {value}")
    return value


def _angle(subject: Any, noun: str) -> Angle:
    """Return ``subject`` as an angle, or refuse it by name."""

    if isinstance(subject, torch.Tensor):
        return subject
    if isinstance(subject, (bool, str)):
        raise ValueError(f"{noun} must be a real number, got {subject!r}")
    try:
        value = float(subject)
    except (TypeError, ValueError):
        raise ValueError(f"{noun} must be a real number, got {subject!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{noun} must be finite, got {subject!r}")
    return value


def _index(subject: Any, *, n_qubits: int | None, noun: str) -> int:
    """Return one spin-orbital index, or refuse it by name.

    ``n_qubits`` is ``None`` when the caller has no register to bound the index
    against, as :func:`excitation_operator` has not.
    """

    if isinstance(subject, bool):
        raise ValueError(f"{noun} must be an integer, got {subject!r}")
    try:
        value = operator.index(subject)
    except TypeError:
        raise ValueError(f"{noun} must be an integer, got {subject!r}") from None
    if value < 0:
        raise ValueError(f"{noun} must be non-negative, got {value}")
    if n_qubits is not None and value >= n_qubits:
        raise ValueError(
            f"{noun} {value} is outside the {n_qubits} spin orbital(s) declared"
        )
    return value


def _excitation_indices(
    occupied: Sequence[int],
    virtual: Sequence[int],
    *,
    n_qubits: int | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate one excitation's spin-orbital indices and return them.

    ``n_qubits`` is ``None`` when the caller has no register to bound the
    indices against, as :func:`excitation_operator` has not, and is already a
    validated width when it is not: both circuit builders read their own width
    before they delegate, so the register is not re-validated here.
    """

    occupied_indices = tuple(
        _index(index, n_qubits=n_qubits, noun="occupied index") for index in occupied
    )
    virtual_indices = tuple(
        _index(index, n_qubits=n_qubits, noun="virtual index") for index in virtual
    )
    if len(occupied_indices) != len(virtual_indices):
        raise ValueError(
            "an excitation moves as many electrons as it fills, got "
            f"{len(occupied_indices)} occupied and {len(virtual_indices)} virtual"
        )
    if len(occupied_indices) not in (1, 2):
        raise ValueError(
            "an excitation acts on one or two electrons, got "
            f"{len(occupied_indices)}"
        )
    for noun, indices in (
        ("occupied indices", occupied_indices),
        ("virtual indices", virtual_indices),
    ):
        if len(set(indices)) != len(indices):
            raise ValueError(f"{noun} must be distinct, got {indices!r}")
    if max(occupied_indices) >= min(virtual_indices):
        raise ValueError(
            "an excitation moves electrons from the occupied spin orbitals below "
            f"the virtual ones, got occupied {occupied_indices!r} and virtual "
            f"{virtual_indices!r}"
        )
    return occupied_indices, virtual_indices


def _parameters(
    parameters: torch.Tensor | Sequence[float],
    expected: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the parameters as a flat float32 tensor of the expected length."""

    values = torch.as_tensor(parameters, dtype=torch.float32, device=device).reshape(-1)
    if values.numel() != expected:
        raise ValueError(f"Expected {expected} parameters, got {values.numel()}.")
    return values


def _circuit(
    n_qubits: int,
    dtype: torch.dtype | None,
    device: torch.device | str | None,
) -> Circuit:
    """Return a fresh circuit, defaulting to the runtime configuration."""

    config = get_runtime_config()
    return Circuit(
        n_qubits,
        device=config.device if device is None else device,
        dtype=getattr(torch, config.complex_dtype) if dtype is None else dtype,
    )


def _product(factors: Sequence[int], *, creating: bool) -> FermionOperator:
    """Return ``a_i^dagger a_j^dagger ...`` or ``a_i a_j ...`` in the given order."""

    operator: FermionOperator | None = None
    for index in factors:
        factor = create(index) if creating else annihilate(index)
        operator = factor if operator is None else operator * factor
    if operator is None:
        raise ValueError("an excitation needs at least one spin orbital")
    return operator


def excitation_operator(
    occupied: Sequence[int],
    virtual: Sequence[int],
) -> FermionOperator:
    """Return the anti-Hermitian generator one excitation evolves.

    For one occupied and one virtual spin orbital the generator is
    ``T = a_v^dagger a_o - a_o^dagger a_v``; for two of each it is
    ``T = a_s^dagger a_r^dagger a_q a_p - a_p^dagger a_q^dagger a_r a_s``, with
    each collection read in the order it is given. Both circuit builders are the
    exponential of the same generator:

    ``single_excitation(...) == double_excitation(...) == exp(i * (theta / 2) * H)``
    for ``H = jordan_wigner(1j * T)``, which is Hermitian because ``T`` is
    anti-Hermitian. Order matters: an excitation operator changes sign when the
    two indices of one collection are exchanged, and the circuit builders take
    that sign from the order they are given, exactly as CUDA-Q's
    ``doubleExcitation`` does. The same spin-orbital order must therefore reach
    both the operator and the circuit for their energies to agree.

    This is the operator pool a UCCSD or ADAPT-VQE workflow screens gradients
    against; :func:`uccsd_excitations` enumerates the indices, this function
    turns one of them into an operator.

    Args:
        occupied: One or two occupied spin orbitals, below every virtual index.
        virtual: One or two virtual spin orbitals.

    Returns:
        The :class:`~flagquantum.observables.fermion.FermionOperator` ``T``.

    Raises:
        ValueError: If the two collections have different sizes, if either has
            more than two indices or repeats one, or if an occupied index is not
            below every virtual index.

    Examples:
        >>> from flagquantum.algorithms.chemistry import excitation_operator
        >>> from flagquantum.observables.fermion import jordan_wigner
        >>> generator = excitation_operator((0,), (1,))
        >>> [(term.coefficient, term.factors) for term in jordan_wigner(1j * generator, n_modes=2).terms]
        [(0.5, ((0, 'x'), (1, 'y'))), (-0.5, ((0, 'y'), (1, 'x')))]
    """

    occupied_indices, virtual_indices = _excitation_indices(
        occupied, virtual, n_qubits=None
    )
    excitation = _product(virtual_indices, creating=True) * _product(
        occupied_indices, creating=False
    )
    deexcitation = _product(occupied_indices, creating=True) * _product(
        virtual_indices, creating=False
    )
    return excitation - deexcitation


@dataclass(frozen=True, slots=True)
class UCCSDExcitations:
    """The single and double excitations of one electron count.

    The five collections are in the order a Trotterized UCCSD product applies
    them, and every index is a spin orbital. ``occupied_alpha`` and
    ``occupied_beta`` are the spin orbitals the reference determinant fills.
    """

    n_electrons: int
    n_qubits: int
    occupied_alpha: tuple[int, ...]
    occupied_beta: tuple[int, ...]
    virtual_alpha: tuple[int, ...]
    virtual_beta: tuple[int, ...]
    singles_alpha: tuple[tuple[int, int], ...] = ()
    singles_beta: tuple[tuple[int, int], ...] = ()
    doubles_mixed: tuple[tuple[int, int, int, int], ...] = ()
    doubles_alpha: tuple[tuple[int, int, int, int], ...] = ()
    doubles_beta: tuple[tuple[int, int, int, int], ...] = ()

    @property
    def reference_occupation(self) -> tuple[int, ...]:
        """Return the spin orbitals the reference determinant fills."""

        return tuple(range(self.n_electrons))

    @property
    def factors(self) -> tuple[tuple[int, ...], ...]:
        """Return every excitation in the order a UCCSD product applies them."""

        return (
            *self.singles_alpha,
            *self.singles_beta,
            *self.doubles_mixed,
            *self.doubles_alpha,
            *self.doubles_beta,
        )

    @property
    def parameter_count(self) -> int:
        """Return the number of angles one UCCSD product over these factors needs."""

        return len(self.factors)


def uccsd_excitations(n_electrons: int, n_qubits: int) -> UCCSDExcitations:
    """Return the UCCSD excitations of ``n_electrons`` in ``n_qubits`` spin orbitals.

    The spin orbitals are paired: index ``2s`` is the alpha spin orbital of
    spatial orbital ``s`` and index ``2s + 1`` is its beta partner. The reference
    determinant fills the lowest ``n_electrons`` spin orbitals, which is the
    occupation a Hartree-Fock calculation for that electron count produces.

    The product is in the order CUDA-Q's ``uccsd`` kernel uses: alpha singles,
    beta singles, mixed-spin doubles, same-spin alpha doubles, same-spin beta
    doubles. The mixed-spin doubles are ordered
    ``(occupied_alpha, occupied_beta, virtual_beta, virtual_alpha)``.

    Args:
        n_electrons: The number of electrons, at most ``n_qubits``.
        n_qubits: The number of spin orbitals, even because every spatial
            orbital carries both spins.

    Returns:
        The :class:`UCCSDExcitations` whose ``parameter_count`` angles
        :func:`uccsd_factors` consumes.

    Raises:
        ValueError: If either count is not an integer, if ``n_qubits`` is odd, or
            if ``n_electrons`` is greater than ``n_qubits``.

    Examples:
        >>> from flagquantum.algorithms.chemistry import uccsd_excitations
        >>> excitations = uccsd_excitations(2, 4)
        >>> excitations.occupied_alpha, excitations.virtual_beta
        ((0,), (3,))
        >>> excitations.factors
        ((0, 2), (1, 3), (0, 1, 3, 2))
        >>> excitations.parameter_count
        3
    """

    width = _count(n_qubits, "n_qubits")
    if width % 2 != 0:
        raise ValueError(
            f"n_qubits must be even, got {width}, because every spatial orbital "
            "carries an alpha and a beta spin orbital"
        )
    electrons = _count(n_electrons, "n_electrons")
    if electrons > width:
        raise ValueError(
            f"n_electrons must be at most the {width} spin orbital(s), got {electrons}"
        )

    occupied_alpha = tuple(index for index in range(electrons) if index % 2 == 0)
    occupied_beta = tuple(index for index in range(electrons) if index % 2 == 1)
    virtual_alpha = tuple(index for index in range(electrons, width) if index % 2 == 0)
    virtual_beta = tuple(index for index in range(electrons, width) if index % 2 == 1)

    singles_alpha = tuple(
        (occupied, virtual) for occupied in occupied_alpha for virtual in virtual_alpha
    )
    singles_beta = tuple(
        (occupied, virtual) for occupied in occupied_beta for virtual in virtual_beta
    )
    doubles_mixed = tuple(
        (alpha_occupied, beta_occupied, beta_virtual, alpha_virtual)
        for alpha_occupied in occupied_alpha
        for beta_occupied in occupied_beta
        for beta_virtual in virtual_beta
        for alpha_virtual in virtual_alpha
    )
    doubles_alpha = tuple(
        (i, j, a, b)
        for i, j in combinations(occupied_alpha, 2)
        for a, b in combinations(virtual_alpha, 2)
    )
    doubles_beta = tuple(
        (i, j, a, b)
        for i, j in combinations(occupied_beta, 2)
        for a, b in combinations(virtual_beta, 2)
    )
    return UCCSDExcitations(
        n_electrons=electrons,
        n_qubits=width,
        occupied_alpha=occupied_alpha,
        occupied_beta=occupied_beta,
        virtual_alpha=virtual_alpha,
        virtual_beta=virtual_beta,
        singles_alpha=singles_alpha,
        singles_beta=singles_beta,
        doubles_mixed=doubles_mixed,
        doubles_alpha=doubles_alpha,
        doubles_beta=doubles_beta,
    )


def _single_excitation_sequence(
    occupied: int,
    virtual: int,
) -> tuple[tuple[str, int, float], ...]:
    """Return the single-excitation gate sequence as readable data.

    Each entry is ``(opcode, wire, argument)``: ``"h"`` ignores the argument,
    ``"rx"`` rotates by it, ``"cx"`` reads it as the target wire, and ``"rz"``
    multiplies it by the caller's half-angle. The sequence is the CUDA-Q
    decomposition, whose two halves use opposite signs because one is ``Y X``
    and the other is ``-X Y``.
    """

    sequence: list[tuple[str, int, float]] = []

    def ladder() -> None:
        for index in range(occupied, virtual):
            sequence.append(("cx", index, float(index + 1)))

    def reverse_ladder() -> None:
        for index in range(virtual, occupied, -1):
            sequence.append(("cx", index - 1, float(index)))

    sequence.append(("rx", occupied, _HALF_PI))
    sequence.append(("h", virtual, 0.0))
    ladder()
    sequence.append(("rz", virtual, 1.0))
    reverse_ladder()
    sequence.append(("h", virtual, 0.0))
    sequence.append(("rx", occupied, -_HALF_PI))

    sequence.append(("h", occupied, 0.0))
    sequence.append(("rx", virtual, _HALF_PI))
    ladder()
    sequence.append(("rz", virtual, -1.0))
    reverse_ladder()
    sequence.append(("rx", virtual, -_HALF_PI))
    sequence.append(("h", occupied, 0.0))
    return tuple(sequence)


def single_excitation(
    n_qubits: int,
    occupied: int,
    virtual: int,
    theta: Angle,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Circuit:
    """Return the circuit that rotates one electron between two spin orbitals.

    The circuit implements ``exp(i * (theta / 2) * H)`` for the ``H`` of
    ``jordan_wigner(1j * excitation_operator((occupied,), (virtual,)))``, which
    moves population between the two spin orbitals rather than accumulating a
    complex phase. It is four basis changes, two conditional ladders, and two
    half-angle ``rz`` rotations.

    Args:
        n_qubits: The number of spin orbitals the circuit declares.
        occupied: The spin orbital the electron leaves.
        virtual: The spin orbital the electron enters.
        theta: The rotation angle, a real number or a scalar tensor.
        dtype: The circuit's complex dtype. The default is the runtime
            configuration's.
        device: The circuit's device. The default is the runtime configuration's.

    Returns:
        The :class:`flagquantum.Circuit` realizing the rotation.

    Raises:
        ValueError: If ``occupied`` and ``virtual`` are equal, out of range, or
            not integers, or if ``theta`` is not a finite real number.

    Examples:
        >>> from flagquantum.algorithms.chemistry import single_excitation
        >>> circuit = single_excitation(2, 0, 1, 0.7)
        >>> circuit.analysis().two_qubit_gates
        4
    """

    width = _count(n_qubits, "n_qubits", minimum=1)
    occupied_indices, virtual_indices = _excitation_indices(
        (occupied,), (virtual,), n_qubits=width
    )
    angle = _angle(theta, "theta")

    circuit = _circuit(width, dtype, device)
    _apply_single(circuit, occupied_indices[0], virtual_indices[0], angle)
    return circuit


def _double_excitation_sequence(
    first_occupied: int,
    second_occupied: int,
    first_virtual: int,
    second_virtual: int,
) -> tuple[tuple[str, int, float], ...]:
    """Return the double-excitation gate sequence as readable data.

    The entries have the same meaning as in :func:`_single_excitation_sequence`.
    Eight blocks of basis changes and conditional ladders carry a quarter of the
    two-electron rotation each, so every ``"rz"`` angle is an eighth of the
    caller's; the four blocks that rotate by ``-1`` cancel the sign of the
    exchange they undo.
    """

    sequence: list[tuple[str, int, float]] = []
    i_occ, j_occ = first_occupied, second_occupied
    a_virt, b_virt = first_virtual, second_virtual

    def h(wire: int) -> None:
        sequence.append(("h", wire, 0.0))

    def rx(wire: int, angle: float) -> None:
        sequence.append(("rx", wire, angle))

    def ladder(low: int, high: int) -> None:
        for index in range(low, high):
            sequence.append(("cx", index, float(index + 1)))

    def reverse_ladder(low: int, high: int) -> None:
        for index in range(high, low, -1):
            sequence.append(("cx", index - 1, float(index)))

    def junction() -> None:
        sequence.append(("cx", j_occ, float(a_virt)))

    def rotate(coefficient: float) -> None:
        sequence.append(("rz", b_virt, coefficient))

    # The eight blocks below are the CUDA-Q sequence in its own order. Four of
    # them rotate by a positive eighth of the angle and four by a negative one,
    # which is the exchange sign each block undoes.
    h(i_occ)
    h(j_occ)
    h(a_virt)
    rx(b_virt, _HALF_PI)
    ladder(i_occ, j_occ)
    junction()
    ladder(a_virt, b_virt)
    rotate(1.0)
    reverse_ladder(a_virt, b_virt)
    junction()
    rx(b_virt, -_HALF_PI)
    h(a_virt)

    rx(a_virt, _HALF_PI)
    h(b_virt)
    junction()
    ladder(a_virt, b_virt)
    rotate(1.0)
    reverse_ladder(a_virt, b_virt)
    junction()
    reverse_ladder(i_occ, j_occ)
    rx(a_virt, -_HALF_PI)
    h(j_occ)

    rx(j_occ, _HALF_PI)
    h(a_virt)
    ladder(i_occ, j_occ)
    junction()
    ladder(a_virt, b_virt)
    rotate(-1.0)
    reverse_ladder(a_virt, b_virt)
    junction()
    h(b_virt)
    h(a_virt)

    rx(a_virt, _HALF_PI)
    rx(b_virt, _HALF_PI)
    junction()
    ladder(a_virt, b_virt)
    rotate(1.0)
    reverse_ladder(a_virt, b_virt)
    junction()
    reverse_ladder(i_occ, j_occ)
    rx(j_occ, -_HALF_PI)
    h(i_occ)

    rx(i_occ, _HALF_PI)
    h(j_occ)
    ladder(i_occ, j_occ)
    junction()
    ladder(a_virt, b_virt)
    rotate(1.0)
    reverse_ladder(a_virt, b_virt)
    junction()
    rx(b_virt, -_HALF_PI)
    rx(a_virt, -_HALF_PI)
    h(a_virt)
    h(b_virt)

    junction()
    ladder(a_virt, b_virt)
    rotate(-1.0)
    reverse_ladder(a_virt, b_virt)
    junction()
    reverse_ladder(i_occ, j_occ)
    h(b_virt)
    h(j_occ)

    rx(j_occ, _HALF_PI)
    rx(b_virt, _HALF_PI)
    ladder(i_occ, j_occ)
    junction()
    ladder(a_virt, b_virt)
    rotate(-1.0)
    reverse_ladder(a_virt, b_virt)
    junction()
    rx(b_virt, -_HALF_PI)
    h(a_virt)

    rx(a_virt, _HALF_PI)
    h(b_virt)
    junction()
    ladder(a_virt, b_virt)
    rotate(-1.0)
    reverse_ladder(a_virt, b_virt)
    junction()
    reverse_ladder(i_occ, j_occ)
    h(b_virt)
    rx(a_virt, -_HALF_PI)
    rx(j_occ, -_HALF_PI)
    rx(i_occ, -_HALF_PI)

    return tuple(sequence)


def double_excitation(
    n_qubits: int,
    occupied: Sequence[int],
    virtual: Sequence[int],
    theta: Angle,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Circuit:
    """Return the circuit that rotates two electrons between two spin-orbital pairs.

    The circuit implements ``exp(i * (theta / 2) * H)`` for the ``H`` of
    ``jordan_wigner(1j * excitation_operator(occupied, virtual))``. Carrying the
    rotation across two occupied spin orbitals, the junction between the pairs,
    and two virtual spin orbitals takes eight blocks of conditional ladders and
    basis changes, each contributing an eighth of ``theta``.

    Args:
        n_qubits: The number of spin orbitals the circuit declares.
        occupied: The two spin orbitals the electrons leave.
        virtual: The two spin orbitals the electrons enter.
        theta: The rotation angle, a real number or a scalar tensor.
        dtype: The circuit's complex dtype. The default is the runtime
            configuration's.
        device: The circuit's device. The default is the runtime configuration's.

    Returns:
        The :class:`flagquantum.Circuit` realizing the rotation.

    Raises:
        ValueError: If either pair is not two distinct in-range indices, if an
            occupied index is not below every virtual index, or if ``theta`` is
            not a finite real number.

    Examples:
        >>> from flagquantum.algorithms.chemistry import double_excitation
        >>> circuit = double_excitation(4, (0, 1), (2, 3), 0.7)
        >>> circuit.analysis().two_qubit_gates
        40
    """

    width = _count(n_qubits, "n_qubits", minimum=1)
    occupied_indices, virtual_indices = _excitation_indices(
        tuple(occupied), tuple(virtual), n_qubits=width
    )
    angle = _angle(theta, "theta")

    circuit = _circuit(width, dtype, device)
    _apply_double(circuit, occupied_indices, virtual_indices, angle)
    return circuit


def uccsd_factors(
    n_electrons: int,
    n_qubits: int,
    parameters: torch.Tensor | Sequence[float],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Circuit:
    """Return the Trotterized UCCSD product, without preparing a reference state.

    This is the product CUDA-Q's ``uccsd`` kernel applies: each factor's
    excitation gate in :attr:`UCCSDExcitations.factors` order, with the next
    angle. The circuit acts on whatever state the caller's register already
    holds, so a caller that wants the Hartree-Fock reference either prepares that
    occupation itself or uses :func:`uccsd_ansatz`.

    Args:
        n_electrons: The number of electrons, at most ``n_qubits``.
        n_qubits: The number of spin orbitals, which must be even.
        parameters: The angles, one per factor, in
            :attr:`UCCSDExcitations.factors` order.
        dtype: The circuit's complex dtype. The default is the runtime
            configuration's.
        device: The circuit's device. The default is the runtime configuration's.

    Returns:
        The :class:`flagquantum.Circuit` realizing the product.

    Raises:
        ValueError: If the counts are refused by :func:`uccsd_excitations` or if
            ``parameters`` does not have ``parameter_count`` entries.

    Examples:
        >>> from flagquantum.algorithms.chemistry import uccsd_factors
        >>> circuit = uccsd_factors(2, 4, [0.0, 0.0, 0.0])
        >>> float(circuit.probabilities().reshape(-1)[0]) > 0.999
        True
    """

    excitations = uccsd_excitations(n_electrons, n_qubits)
    values = _parameters(parameters, excitations.parameter_count, _device(device))
    circuit = _circuit(excitations.n_qubits, dtype, device)
    _apply_factors(circuit, excitations.factors, values)
    return circuit


def _apply_factors(
    circuit: Circuit,
    factors: Sequence[Sequence[int]],
    values: torch.Tensor,
) -> None:
    """Apply each excitation factor to ``circuit`` in order."""

    for cursor, factor in enumerate(factors):
        if len(factor) == 2:
            _apply_single(circuit, factor[0], factor[1], values[cursor])
        else:
            _apply_double(circuit, factor[:2], factor[2:], values[cursor])


def _apply_sequence(
    circuit: Circuit,
    sequence: Sequence[tuple[str, int, float]],
    *,
    rotation_scale: Angle,
) -> None:
    """Apply a gate sequence whose ``"rz"`` entries carry a multiple of the angle."""

    for opcode, wire, argument in sequence:
        if opcode == "h":
            circuit.gate("h", (wire,))
        elif opcode == "rx":
            circuit.gate("rx", (wire,), theta=argument)
        elif opcode == "cx":
            circuit.gate("cx", (wire, int(argument)))
        else:
            circuit.gate("rz", (wire,), theta=rotation_scale * argument)


def _apply_single(
    circuit: Circuit,
    occupied: int,
    virtual: int,
    angle: Angle,
) -> None:
    """Apply one already-validated single excitation to ``circuit``."""

    _apply_sequence(
        circuit,
        _single_excitation_sequence(occupied, virtual),
        rotation_scale=0.5 * angle,
    )


def _apply_double(
    circuit: Circuit,
    occupied: Sequence[int],
    virtual: Sequence[int],
    angle: Angle,
) -> None:
    """Apply one already-validated double excitation to ``circuit``."""

    sign = 1.0 if (occupied[0] < occupied[1]) == (virtual[0] < virtual[1]) else -1.0
    _apply_sequence(
        circuit,
        _double_excitation_sequence(*sorted(occupied), *sorted(virtual)),
        rotation_scale=0.125 * angle * sign,
    )


def uccsd_ansatz(
    n_qubits: int,
    n_electrons: int,
    parameters: torch.Tensor | Sequence[float],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Circuit:
    """Return a complete UCCSD state-preparation circuit.

    The circuit fills the reference determinant with an ``x`` on each of the
    lowest ``n_electrons`` spin orbitals and then applies :func:`uccsd_factors`,
    so its output state is the UCCSD state rather than the product alone.

    Args:
        n_qubits: The number of spin orbitals, which must be even.
        n_electrons: The number of electrons, at most ``n_qubits``.
        parameters: The angles, one per factor.
        dtype: The circuit's complex dtype. The default is the runtime
            configuration's.
        device: The circuit's device. The default is the runtime configuration's.

    Returns:
        The :class:`flagquantum.Circuit` preparing the UCCSD state.

    Raises:
        ValueError: If the counts are refused by :func:`uccsd_excitations` or if
            ``parameters`` does not have ``parameter_count`` entries.

    Examples:
        >>> from flagquantum.algorithms.chemistry import uccsd_ansatz
        >>> circuit = uccsd_ansatz(4, 2, [0.0, 0.0, 0.0])
        >>> float(circuit.probabilities().reshape(-1)[0b1100]) > 0.999
        True
    """

    excitations = uccsd_excitations(n_electrons, n_qubits)
    values = _parameters(parameters, excitations.parameter_count, _device(device))
    circuit = _circuit(excitations.n_qubits, dtype, device)
    for spin_orbital in excitations.reference_occupation:
        circuit.gate("x", (spin_orbital,))
    _apply_factors(circuit, excitations.factors, values)
    return circuit


def coupler_hardware_efficient_parameter_count(n_qubits: int, layers: int) -> int:
    """Return the number of angles the coupler hardware-efficient ansatz needs.

    The count is ``2 * n_qubits * (1 + layers)``: one rotation block before the
    first layer and one after each layer's couplers.

    Raises:
        ValueError: If either count is not an integer of at least ``0``.
    """

    return 2 * _count(n_qubits, "n_qubits") * (1 + _count(layers, "layers"))


def _coupler_pair(coupler: Any, *, n_qubits: int) -> tuple[int, int]:
    """Return one coupler as a pair of distinct in-range qubits, or refuse it."""

    if isinstance(coupler, (str, bytes)) or not isinstance(coupler, Iterable):
        raise ValueError(f"a coupler must be a pair of qubits, got {coupler!r}")
    pair = tuple(coupler)
    if len(pair) != 2:
        raise ValueError(f"a coupler must be a pair of qubits, got {pair!r}")
    control = _index(pair[0], n_qubits=n_qubits, noun="coupler control")
    target = _index(pair[1], n_qubits=n_qubits, noun="coupler target")
    if control == target:
        raise ValueError(f"a coupler must join two qubits, got {control} twice")
    return control, target


def coupler_hardware_efficient_ansatz(
    n_qubits: int,
    layers: int,
    parameters: torch.Tensor | Sequence[float],
    *,
    couplers: Iterable[tuple[int, int]] | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Circuit:
    """Return the hardware-efficient ansatz of Kandala et al. with named couplers.

    Every qubit rotates by ``ry`` then ``rz`` once before the first layer and
    once after each layer's couplers, so ``layers`` layers consume
    ``2 * n_qubits * (1 + layers)`` angles in that order. This is the ansatz
    CUDA-Q's ``hwe`` kernel builds, including its leading rotation block and its
    caller-supplied coupler list.

    :func:`flagquantum.algorithms.hardware_efficient_ansatz` is a different
    circuit: it has no leading rotation block, so it consumes
    ``2 * n_qubits * layers`` angles, and it selects its own entanglers from
    ``"linear"``, ``"circular"`` or ``"none"``. A parameter vector for one is not
    a parameter vector for the other.

    Args:
        n_qubits: The number of qubits the ansatz acts on.
        layers: The number of coupler layers.
        parameters: The angles, in the order the rotations consume them.
        couplers: The ``(control, target)`` pairs, distinct and in range. The
            default is the nearest-neighbour chain ``(q, q + 1)``, which is empty
            for fewer than two qubits.
        dtype: The circuit's complex dtype. The default is the runtime
            configuration's.
        device: The circuit's device. The default is the runtime configuration's.

    Returns:
        The :class:`flagquantum.Circuit` realizing the ansatz.

    Raises:
        ValueError: If a count is not an integer of at least ``0``, if a coupler
            is not a distinct in-range pair, or if ``parameters`` does not have
            the expected number of entries.

    Examples:
        >>> from flagquantum.algorithms.chemistry import (
        ...     coupler_hardware_efficient_ansatz,
        ...     coupler_hardware_efficient_parameter_count,
        ... )
        >>> coupler_hardware_efficient_parameter_count(2, 1)
        8
        >>> circuit = coupler_hardware_efficient_ansatz(2, 1, [0.0] * 8)
        >>> circuit.analysis().two_qubit_gates
        1
    """

    width = _count(n_qubits, "n_qubits")
    depth = _count(layers, "layers")
    expected = coupler_hardware_efficient_parameter_count(width, depth)
    values = _parameters(parameters, expected, _device(device))
    if couplers is None:
        pairs = tuple((qubit, qubit + 1) for qubit in range(width - 1))
    else:
        pairs = tuple(_coupler_pair(coupler, n_qubits=width) for coupler in couplers)
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"couplers must be distinct, got {pairs!r}")

    circuit = _circuit(width, dtype, device)
    cursor = 0
    for qubit in range(width):
        circuit.gate("ry", (qubit,), theta=values[cursor])
        circuit.gate("rz", (qubit,), theta=values[cursor + 1])
        cursor += 2
    for _ in range(depth):
        for control, target in pairs:
            circuit.gate("cx", (control, target))
        for qubit in range(width):
            circuit.gate("ry", (qubit,), theta=values[cursor])
            circuit.gate("rz", (qubit,), theta=values[cursor + 1])
            cursor += 2
    return circuit


def _device(device: torch.device | str | None) -> torch.device | str:
    """Return the caller's device or the runtime configuration's."""

    return get_runtime_config().device if device is None else device
