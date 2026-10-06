"""State preparation from a classical amplitude vector.

The uniform case puts every qubit in an equal superposition with Hadamard gates. The arbitrary
case builds a circuit that carries ``|0...0>`` onto a requested amplitude vector using
uniformly controlled rotations, following Möttönen, Vartiainen, Bergholm, and Salomaa,
"Transformation of quantum states using uniformly controlled rotations", *Quantum Information
and Computation* **5**, 467 (2005), arXiv:quant-ph/0407010.

The premise this module rests on runs against a speedup, and it is worth stating plainly.
Computing the rotation angles requires a classical pass over all ``2**n`` amplitudes *and* a
``2**n`` by ``2**n`` linear solve, so **the input is already exponential in size**. The module
demonstrates that a state can be prepared efficiently *given* its amplitudes; it does not show
that preparing a state is cheaper than the classical description of one. The preparation is
exact only to the working precision of that solve.

This unit is demonstration scale. It makes no performance, capacity, or hardware claim, and it
does not select a runtime.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ...circuit import Circuit

__all__ = ["append_arbitrary_state", "arbitrary_state", "uniform_state"]

# The relative phases are solved as one square system whose last column is a global phase
# offset, so the unknown count is 2**n - 1 ladder angles plus that offset.
_RY_GATE = "ry"
_RZ_GATE = "rz"


def uniform_state(n_qubits: int) -> Circuit:
    """Build the uniform superposition on ``n_qubits`` qubits.

    Args:
        n_qubits: The number of qubits; must be at least one.

    Returns:
        A circuit of ``n_qubits`` Hadamard gates, whose amplitudes are all ``1/sqrt(2**n)``.

    Raises:
        ValueError: If ``n_qubits`` is not positive.
    """
    if n_qubits < 1:
        raise ValueError(f"uniform state needs at least one qubit, got {n_qubits}")
    circuit = Circuit(n_qubits)
    for qubit in range(n_qubits):
        circuit.gate("h", qubit)
    return circuit


def arbitrary_state(
    amplitudes: torch.Tensor, *, qubits: Sequence[int] | None = None
) -> Circuit:
    """Build a circuit that prepares ``amplitudes`` from ``|0...0>``.

    The circuit is built on a fresh register: with ``qubits`` given, a circuit large enough to
    hold the highest qubit, and with ``qubits`` omitted, a circuit on qubits ``0..n-1`` where
    ``n`` is the base-two logarithm of the amplitude count. Qubit ``0`` is the most significant
    bit, so ``circuit.state().reshape(-1)[k]`` is the amplitude of basis state ``|k>``.

    The cost is the caveat that matters here. Computing the rotation angles takes a classical
    pass over all ``2**n`` amplitudes and a ``2**n`` by ``2**n`` linear solve, so the input is
    already exponential in size: this prepares a state efficiently *given* its amplitudes, and
    it does not show that preparing a state is cheaper than its classical description.

    The amplitudes are read as classical data, not as a differentiable input. Their values fix
    the rotation angles outright, and each angle is emitted as a plain float, so a tensor that
    requires a gradient receives none: the vector is detached on entry and no autograd edge
    reaches the circuit. ``requires_grad`` on ``amplitudes`` is therefore accepted and ignored.

    Args:
        amplitudes: The amplitude vector, one-dimensional, of power-of-two length at least
            two, and not the zero vector. Any norm is accepted; the vector is normalised here.
            Read as classical data: see above.
        qubits: The qubits to prepare, most significant first; defaults to
            ``range(n)`` with ``n = log2(len(amplitudes))``.

    Returns:
        A circuit preparing the normalised amplitudes on ``qubits``.

    Raises:
        ValueError: If ``amplitudes`` is not a one-dimensional tensor of power-of-two length
            at least two; if it is the zero vector; if ``qubits`` does not carry one qubit per
            amplitude index; or if it repeats a qubit or holds a negative or non-integer one.
    """
    data, n_qubits = _prepared_amplitudes(amplitudes)
    ordered = _resolve_qubits(qubits, n_qubits)
    circuit = Circuit(max(ordered) + 1)
    _append_state_preparation(circuit, data, ordered)
    return circuit


def append_arbitrary_state(
    circuit: Circuit, amplitudes: torch.Tensor, qubits: Sequence[int]
) -> None:
    """Append a state-preparation circuit for ``amplitudes`` to ``circuit`` in place.

    The unitaries commute with nothing the caller has already emitted in general, so this
    appends the preparation to the end of the circuit rather than to its start.

    Computing the rotation angles takes a classical pass over all ``2**n`` amplitudes and a
    ``2**n`` by ``2**n`` linear solve, so the input is already exponential in size.

    As in :func:`arbitrary_state`, the amplitudes are read as classical data and detached on
    entry, so no gradient flows back to the caller's tensor.

    Args:
        circuit: The circuit to extend.
        amplitudes: The amplitude vector, validated as in :func:`arbitrary_state`, and read as
            classical data there.
        qubits: The qubits to prepare, most significant first.

    Raises:
        ValueError: If ``amplitudes`` or ``qubits`` fail the validation described in
            :func:`arbitrary_state`.
    """
    data, n_qubits = _prepared_amplitudes(amplitudes)
    ordered = _resolve_qubits(qubits, n_qubits)
    _append_state_preparation(circuit, data, ordered)


def _prepared_amplitudes(amplitudes: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Validate an amplitude vector and normalise it onto a CPU ``complex64`` tensor.

    The angle computation is a classical precomputation, so it runs on the CPU regardless of
    the device the caller holds the vector on.

    Args:
        amplitudes: The candidate amplitude vector.

    Returns:
        The normalised vector and the register width ``n`` with ``2**n`` entries.

    Raises:
        ValueError: If the vector is not a one-dimensional tensor, does not have a
            power-of-two length of at least two, or is the zero vector.
    """
    if not isinstance(amplitudes, torch.Tensor):
        raise ValueError(
            f"amplitudes must be a torch.Tensor, got {type(amplitudes).__name__}"
        )
    if amplitudes.dim() != 1:
        raise ValueError(
            "amplitudes must be one-dimensional, got shape "
            f"{tuple(int(size) for size in amplitudes.shape)}"
        )
    length = int(amplitudes.shape[0])
    if length < 2:
        raise ValueError(f"amplitudes needs at least two entries, got {length}")
    if length & (length - 1):
        raise ValueError(
            f"amplitudes length must be a power of two, got {length}; a register of "
            "n qubits carries 2**n amplitudes"
        )
    data = amplitudes.to(device="cpu", dtype=torch.complex64)
    # The amplitudes are read as classical data: the angles below are solved from their values
    # and emitted as plain floats, so no gradient can flow back to the caller's vector. The
    # `detach` states that once, here, rather than leaving every later scalar conversion to
    # raise a PyTorch warning about a `requires_grad` tensor being read as a number.
    norm = float(data.detach().norm())
    if norm == 0.0:
        raise ValueError(
            "amplitudes must not be the zero vector; nothing can be normalised"
        )
    n_qubits = length.bit_length() - 1
    return data.detach() / norm, n_qubits


def _resolve_qubits(qubits: Sequence[int] | None, n_qubits: int) -> list[int]:
    """Validate the qubits a preparation targets and return them as a list.

    Args:
        qubits: The requested qubits, or ``None`` for the leading register.
        n_qubits: The number of qubits the amplitudes require.

    Returns:
        The qubits in the order the caller gave them.

    Raises:
        ValueError: If the count does not match, or a qubit repeats, is negative, or is not an
            integer.
    """
    if qubits is None:
        return list(range(n_qubits))
    ordered = list(qubits)
    if len(ordered) != n_qubits:
        raise ValueError(
            f"the amplitudes describe {n_qubits} qubits, got {len(ordered)} qubits"
        )
    if len(set(ordered)) != len(ordered):
        raise ValueError(
            "qubits must be distinct; a repeated qubit would drive the same qubit twice"
        )
    for qubit in ordered:
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise ValueError(f"qubits must be integers, got {qubit!r}")
        if qubit < 0:
            raise ValueError(f"qubits must be non-negative, got {qubit}")
    return ordered


def _append_state_preparation(
    circuit: Circuit, data: torch.Tensor, qubits: Sequence[int]
) -> None:
    """Append the two-pass preparation of ``data`` on ``qubits`` to ``circuit``.

    The magnitude pass emits one uniformly controlled ``ry`` per level, which leaves the
    register in the all-positive vector of the target magnitudes. The phase pass then emits
    one uniformly controlled ``rz`` per level, which is diagonal in the computational basis
    and so adds the relative phases without touching a magnitude. The phase ladders must come
    after every magnitude ladder, because they correct the phases of the state those ladders
    produced.

    Args:
        circuit: The circuit to extend.
        data: The normalised amplitude vector, one entry per basis state.
        qubits: The qubits to prepare, most significant first.
    """
    n_qubits = len(qubits)
    for level in range(n_qubits):
        _append_uniformly_controlled_rotation(
            circuit,
            gate=_RY_GATE,
            target=qubits[level],
            controls=qubits[:level],
            theta=_magnitude_angles(data, n_qubits, level),
        )
    for level, angles in enumerate(_phase_angles(data, n_qubits)):
        _append_uniformly_controlled_rotation(
            circuit,
            gate=_RZ_GATE,
            target=qubits[level],
            controls=qubits[:level],
            theta=angles,
        )


def _magnitude_angles(data: torch.Tensor, n_qubits: int, level: int) -> list[float]:
    """Return the ``ry`` angle of every branch of one level of the magnitude pass.

    Level ``q`` splits the register into ``2**q`` blocks of ``2**(n-q)`` amplitudes. Inside a
    block the qubit ``q`` bit picks the half, and ``2*atan2`` of the two half-norms is the
    rotation that gives the halves their relative weight.

    Args:
        data: The normalised amplitude vector.
        n_qubits: The register width.
        level: The level ``q``, from zero upward.

    Returns:
        One angle per branch of the level's control register, in branch order.
    """
    block = 2 ** (n_qubits - level)
    half = block // 2
    angles: list[float] = []
    for branch in range(2**level):
        start = branch * block
        low = data[start : start + half]
        high = data[start + half : start + block]
        angles.append(2 * math.atan2(_block_norm(high), _block_norm(low)))
    return angles


def _block_norm(block: torch.Tensor) -> float:
    """Return the euclidean norm of a slice of the amplitude vector."""
    return float(block.abs().pow(2).sum().sqrt())


def _phase_angles(data: torch.Tensor, n_qubits: int) -> list[list[float]]:
    """Return the ``rz`` angle of every branch of every level of the phase pass.

    Each uniformly controlled ``rz`` at level ``q`` adds ``0.5 * eps_q(k) * gamma[q][p]`` to
    the phase of basis state ``k``, where ``p`` is the top ``q`` bits of ``k`` and ``eps_q(k)``
    is ``+1`` when bit ``q`` of ``k`` is set. The all-zeros state is not exempt: every ladder
    contributes its ``-gamma/2`` term there, so the phases are solvable only up to one additive
    constant. The system therefore carries the ``2**n - 1`` ladder angles *plus* one global
    offset column and is solved jointly, which is what makes the relative phase of ``|0...0>``
    come out right.

    Args:
        data: The normalised amplitude vector.
        n_qubits: The register width.

    Returns:
        One list of angles per level, in branch order.
    """
    size = 2**n_qubits
    column_of: list[list[int]] = []
    columns = 0
    for level in range(n_qubits):
        level_columns = []
        for _branch in range(2**level):
            level_columns.append(columns)
            columns += 1
        column_of.append(level_columns)
    # ``columns`` is now 2**n - 1; the last column is the global offset, so the system is
    # square and every target phase vector is reachable up to that one additive constant.
    matrix = torch.zeros((size, size), dtype=torch.float64)
    rhs = torch.angle(data).to(torch.float64)
    for state in range(size):
        for level in range(n_qubits):
            sign = 1.0 if (state >> (n_qubits - 1 - level)) & 1 else -1.0
            matrix[state, column_of[level][state >> (n_qubits - level)]] += 0.5 * sign
        matrix[state, size - 1] = 1.0
    solution = torch.linalg.solve(matrix, rhs)
    return [
        [float(solution[column]) for column in level_columns]
        for level_columns in column_of
    ]


def _append_uniformly_controlled_rotation(
    circuit: Circuit,
    *,
    gate: str,
    target: int,
    controls: Sequence[int],
    theta: Sequence[float],
) -> None:
    """Append a uniformly controlled rotation of ``target`` to ``circuit``.

    The effective rotation in the branch of the control register is ``theta[branch]``, so the
    emitted angles are the Walsh transform of ``theta`` in Gray-code order. Consecutive
    branches differ in one control bit, so each step carries one CNOT on that bit and the walk
    closes with one more CNOT; leaving the closing CNOT out makes the walk open, which leaves
    an odd number of CNOTs on each control qubit and breaks the block structure.

    Args:
        circuit: The circuit to extend.
        gate: ``"ry"`` or ``"rz"``, the rotation to control.
        target: The qubit the rotation acts on.
        controls: The control qubits, most significant first.
        theta: One effective angle per branch of ``controls``.
    """
    n_controls = len(controls)
    if n_controls == 0:
        circuit.gate(gate, target, theta=theta[0])
        return
    alpha = _walsh(theta, n_controls)
    previous = 0
    for step in range(2**n_controls):
        current = step ^ (step >> 1)
        if step > 0:
            bit = (current ^ previous).bit_length() - 1
            circuit.gate("cx", (controls[n_controls - 1 - bit], target))
        circuit.gate(gate, target, theta=alpha[current])
        previous = current
    closing = previous.bit_length() - 1
    circuit.gate("cx", (controls[n_controls - 1 - closing], target))


def _walsh(theta: Sequence[float], n_controls: int) -> list[float]:
    """Return the Walsh transform of ``theta``, normalised by ``2**n_controls``.

    Args:
        theta: The effective angle of each branch of the control register.
        n_controls: The number of control qubits.

    Returns:
        ``alpha`` with ``alpha[p] = 2**-k * sum_q (-1)**popcount(p & q) * theta[q]``.
    """
    scale = 2.0**n_controls
    alpha: list[float] = []
    for index in range(2**n_controls):
        total = 0.0
        for branch in range(2**n_controls):
            sign = -1.0 if bin(index & branch).count("1") % 2 else 1.0
            total += sign * theta[branch]
        alpha.append(total / scale)
    return alpha
