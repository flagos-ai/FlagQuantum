"""State preparation from a classical amplitude vector.

The uniform case puts every wire in an equal superposition with Hadamard gates. The arbitrary
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


def uniform_state(n_wires: int) -> Circuit:
    """Build the uniform superposition on ``n_wires`` wires.

    Args:
        n_wires: The number of wires; must be at least one.

    Returns:
        A circuit of ``n_wires`` Hadamard gates, whose amplitudes are all ``1/sqrt(2**n)``.

    Raises:
        ValueError: If ``n_wires`` is not positive.
    """
    if n_wires < 1:
        raise ValueError(f"uniform state needs at least one wire, got {n_wires}")
    circuit = Circuit(n_wires)
    for wire in range(n_wires):
        circuit.gate("h", wire)
    return circuit


def arbitrary_state(
    amplitudes: torch.Tensor, *, wires: Sequence[int] | None = None
) -> Circuit:
    """Build a circuit that prepares ``amplitudes`` from ``|0...0>``.

    The circuit is built on a fresh register: with ``wires`` given, a circuit large enough to
    hold the highest wire, and with ``wires`` omitted, a circuit on wires ``0..n-1`` where
    ``n`` is the base-two logarithm of the amplitude count. Wire ``0`` is the most significant
    bit, so ``circuit.state().reshape(-1)[k]`` is the amplitude of basis state ``|k>``.

    The cost is the caveat that matters here. Computing the rotation angles takes a classical
    pass over all ``2**n`` amplitudes and a ``2**n`` by ``2**n`` linear solve, so the input is
    already exponential in size: this prepares a state efficiently *given* its amplitudes, and
    it does not show that preparing a state is cheaper than its classical description.

    Args:
        amplitudes: The amplitude vector, one-dimensional, of power-of-two length at least
            two, and not the zero vector. Any norm is accepted; the vector is normalised here.
        wires: The wires to prepare, least significant register order; defaults to
            ``range(n)`` with ``n = log2(len(amplitudes))``.

    Returns:
        A circuit preparing the normalised amplitudes on ``wires``.

    Raises:
        ValueError: If ``amplitudes`` is not a one-dimensional tensor of power-of-two length
            at least two; if it is the zero vector; if ``wires`` does not carry one wire per
            amplitude index; or if it repeats a wire or holds a negative or non-integer one.
    """
    data, n_wires = _prepared_amplitudes(amplitudes)
    ordered = _resolve_wires(wires, n_wires)
    circuit = Circuit(max(ordered) + 1)
    _append_state_preparation(circuit, data, ordered)
    return circuit


def append_arbitrary_state(
    circuit: Circuit, amplitudes: torch.Tensor, wires: Sequence[int]
) -> None:
    """Append a state-preparation circuit for ``amplitudes`` to ``circuit`` in place.

    The unitaries commute with nothing the caller has already emitted in general, so this
    appends the preparation to the end of the circuit rather than to its start.

    Computing the rotation angles takes a classical pass over all ``2**n`` amplitudes and a
    ``2**n`` by ``2**n`` linear solve, so the input is already exponential in size.

    Args:
        circuit: The circuit to extend.
        amplitudes: The amplitude vector, validated as in :func:`arbitrary_state`.
        wires: The wires to prepare, least significant register order.

    Raises:
        ValueError: If ``amplitudes`` or ``wires`` fail the validation described in
            :func:`arbitrary_state`.
    """
    data, n_wires = _prepared_amplitudes(amplitudes)
    ordered = _resolve_wires(wires, n_wires)
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
            "n wires carries 2**n amplitudes"
        )
    data = amplitudes.to(device="cpu", dtype=torch.complex64)
    norm = float(data.norm())
    if norm == 0.0:
        raise ValueError(
            "amplitudes must not be the zero vector; nothing can be normalised"
        )
    n_wires = length.bit_length() - 1
    return data / norm, n_wires


def _resolve_wires(wires: Sequence[int] | None, n_wires: int) -> list[int]:
    """Validate the wires a preparation targets and return them as a list.

    Args:
        wires: The requested wires, or ``None`` for the leading register.
        n_wires: The number of wires the amplitudes require.

    Returns:
        The wires in the order the caller gave them.

    Raises:
        ValueError: If the count does not match, or a wire repeats, is negative, or is not an
            integer.
    """
    if wires is None:
        return list(range(n_wires))
    ordered = list(wires)
    if len(ordered) != n_wires:
        raise ValueError(
            f"the amplitudes describe {n_wires} wires, got {len(ordered)} wires"
        )
    if len(set(ordered)) != len(ordered):
        raise ValueError(
            "wires must be distinct; a repeated wire would drive the same qubit twice"
        )
    for wire in ordered:
        if isinstance(wire, bool) or not isinstance(wire, int):
            raise ValueError(f"wires must be integers, got {wire!r}")
        if wire < 0:
            raise ValueError(f"wires must be non-negative, got {wire}")
    return ordered


def _append_state_preparation(
    circuit: Circuit, data: torch.Tensor, wires: Sequence[int]
) -> None:
    """Append the two-pass preparation of ``data`` on ``wires`` to ``circuit``.

    The magnitude pass emits one uniformly controlled ``ry`` per level, which leaves the
    register in the all-positive vector of the target magnitudes. The phase pass then emits
    one uniformly controlled ``rz`` per level, which is diagonal in the computational basis
    and so adds the relative phases without touching a magnitude. The phase ladders must come
    after every magnitude ladder, because they correct the phases of the state those ladders
    produced.

    Args:
        circuit: The circuit to extend.
        data: The normalised amplitude vector, one entry per basis state.
        wires: The wires to prepare, most significant first.
    """
    n_wires = len(wires)
    for level in range(n_wires):
        _append_uniformly_controlled_rotation(
            circuit,
            gate=_RY_GATE,
            target=wires[level],
            controls=wires[:level],
            theta=_magnitude_angles(data, n_wires, level),
        )
    for level, angles in enumerate(_phase_angles(data, n_wires)):
        _append_uniformly_controlled_rotation(
            circuit,
            gate=_RZ_GATE,
            target=wires[level],
            controls=wires[:level],
            theta=angles,
        )


def _magnitude_angles(data: torch.Tensor, n_wires: int, level: int) -> list[float]:
    """Return the ``ry`` angle of every branch of one level of the magnitude pass.

    Level ``q`` splits the register into ``2**q`` blocks of ``2**(n-q)`` amplitudes. Inside a
    block the wire ``q`` bit picks the half, and ``2*atan2`` of the two half-norms is the
    rotation that gives the halves their relative weight.

    Args:
        data: The normalised amplitude vector.
        n_wires: The register width.
        level: The level ``q``, from zero upward.

    Returns:
        One angle per branch of the level's control register, in branch order.
    """
    block = 2 ** (n_wires - level)
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


def _phase_angles(data: torch.Tensor, n_wires: int) -> list[list[float]]:
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
        n_wires: The register width.

    Returns:
        One list of angles per level, in branch order.
    """
    size = 2**n_wires
    column_of: list[list[int]] = []
    columns = 0
    for level in range(n_wires):
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
        for level in range(n_wires):
            sign = 1.0 if (state >> (n_wires - 1 - level)) & 1 else -1.0
            matrix[state, column_of[level][state >> (n_wires - level)]] += 0.5 * sign
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
    an odd number of CNOTs on each control wire and breaks the block structure.

    Args:
        circuit: The circuit to extend.
        gate: ``"ry"`` or ``"rz"``, the rotation to control.
        target: The wire the rotation acts on.
        controls: The control wires, most significant first.
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
        n_controls: The number of control wires.

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
