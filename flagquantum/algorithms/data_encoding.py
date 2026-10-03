"""Encoding a classical feature vector onto a quantum state.

Two encodings are provided, and they answer different questions. **Amplitude
encoding** spends the whole ``2**n``-dimensional state vector on the data: one
feature vector of length ``d`` becomes the amplitudes of ``n = ceil(log2 d)``
qubits. **Angular encoding** spends only ``n`` angles on ``n`` qubits, one
parameterised rotation per qubit, and produces a product state whose dimension is
still exponential while the data it carries is linear.

That asymmetry is the whole trade-off and it is not a defect of either method.
Angular encoding is cheap to build and to differentiate, and a repeated
interleaving of it with a variational block is what a data-reuploading feature
map is; amplitude encoding packs the data far more densely but needs a
state-preparation circuit whose gate count grows with the data it encodes, so
the classical input is already exponential in the qubit count. Neither encoding
is a speedup claim, and this module makes none.

The amplitude case delegates the preparation to
:func:`~flagquantum.algorithms.primitives.arbitrary_state` rather than
reimplementing the uniformly controlled rotation ladder, so there is exactly one
state-preparation implementation in the repository. What this module adds on top
of it is the classical-data front end: padding a non-power-of-two feature vector
to the next power of two, normalising it, and the refusals a data encoder owes
its caller.

**Wire order.** Wire ``0`` is the most significant bit, so
``circuit.state().reshape(-1)[k]`` is the amplitude of the basis state whose
bits read qubit ``0`` to qubit ``n-1`` from left to right. This is the ordering
:mod:`~flagquantum.algorithms.primitives.state_preparation` already documents,
and it is fixed here rather than made configurable: a second index convention
would be a second source of truth for what a basis state is.

Both entry points return or extend a :class:`~flagquantum.circuit.Circuit`. They
do not execute it, do not select a runtime, and do not carry a simulator object:
the caller decides how the prepared state is read.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..circuit import Circuit
from .primitives.state_preparation import arbitrary_state

__all__ = ["amplitude_encode", "angular_encode", "append_angular_encode"]

# CUDA-Q's ``angular_encode`` names the rotation by its Pauli axis and resolves it
# to the gate, so the mapping is kept keyed by the axis rather than by the gate.
_ROTATION_GATES: dict[str, str] = {"X": "rx", "Y": "ry", "Z": "rz"}
_DEFAULT_ROTATION = "Y"


def amplitude_encode(
    amplitudes: Any,
    *,
    pad: complex | float = 0.0,
    qubits: Sequence[int] | None = None,
) -> Circuit:
    """Prepare a circuit whose state vector is ``amplitudes``.

    The feature vector is first padded to the next power of two with ``pad``,
    then normalised by its euclidean norm, then prepared by uniformly controlled
    rotations. Padding happens **before** normalisation, so the pad value takes
    part in the norm exactly as a real feature would.

    Args:
        amplitudes: The feature vector, one-dimensional. Any sequence of real or
            complex numbers, or any object :func:`torch.as_tensor` can read, such
            as a :class:`torch.Tensor`. A tensor on any device is accepted and
            moved to the CPU, because the rotation angles are a classical
            precomputation.
        pad: The value appended to reach a power-of-two length. Ignored when the
            length already is one.
        qubits: The qubits to prepare, most significant first; defaults to
            ``range(n)`` with ``n`` the power-of-two exponent of the padded
            length.

    Returns:
        A circuit of ``max(qubits) + 1`` qubits whose state prepares the normalised
        amplitudes on ``qubits``.

    Raises:
        TypeError: If ``amplitudes`` is a string, bytes, or bytearray, or an
            object :func:`torch.as_tensor` cannot read; or if ``pad`` is not a
            real number.
        ValueError: If ``amplitudes`` is not one-dimensional, is empty, is not
            finite, has zero norm once padded, describes fewer than two amplitude
            entries, or if ``qubits`` does not carry one qubit per amplitude index,
            repeats a qubit, or holds a negative or non-integer one.

    Examples:
        >>> import flagquantum as fq
        >>> circuit = fq.algorithms.data_encoding.amplitude_encode([1.0, 1.0])
        >>> [round(abs(complex(a)), 6) for a in circuit.state().reshape(-1)]
        [0.707107, 0.707107]
    """
    data = _amplitude_vector(amplitudes, pad)
    # The registers are resolved here rather than left to the preparation primitive,
    # so that a mismatch is refused in the canonical vocabulary this module speaks.
    # One amplitude per basis state means one register per bit of the index, not one
    # register per amplitude.
    n_qubits = int(data.numel()).bit_length() - 1
    ordered = _resolve_qubits(qubits, n_qubits, noun="amplitudes")
    return arbitrary_state(data, wires=ordered)


def angular_encode(
    features: Any,
    *,
    qubits: Sequence[int] | None = None,
    rotation: str = _DEFAULT_ROTATION,
) -> Circuit:
    """Prepare a product state carrying one feature per qubit as a rotation.

    One :func:`~flagquantum.circuit.Circuit.rx`, ``ry`` or ``rz`` gate is emitted
    per qubit, so the prepared state is the product of the individual rotations
    applied to ``|0...0>``.

    The angle is handed to the circuit unchanged, as the tensor the caller
    supplied where one was supplied. That is deliberate: an angle converted to a
    Python ``float`` would be detached from the graph and the encoding would stop
    being differentiable with respect to the features.

    Args:
        features: The angle vector, one-dimensional and one entry per qubit. A
            :class:`torch.Tensor` or any sequence of real numbers.
        qubits: The qubits to rotate, one per feature; defaults to ``range(n)``
            with ``n`` the feature count.
        rotation: The rotation axis, ``'X'``, ``'Y'``, or ``'Z'``, matched
            case-insensitively. Defaults to ``'Y'``.

    Returns:
        A circuit of ``max(qubits) + 1`` qubits carrying one rotation per feature.

    Raises:
        TypeError: If ``features`` is a string, bytes, or bytearray, or an object
            :func:`torch.as_tensor` cannot read; or if ``features`` is not real.
        ValueError: If ``features`` is not one-dimensional, is empty, is not
            finite, if ``rotation`` is not one of the three axes, or if ``qubits``
            does not carry one qubit per feature, repeats a qubit, or holds a
            negative or non-integer one.

    Examples:
        >>> import flagquantum as fq
        >>> circuit = fq.algorithms.data_encoding.angular_encode([0.0, 0.0])
        >>> [complex(a) for a in circuit.state().reshape(-1)]
        [(1+0j), 0j, 0j, 0j]
    """
    ordered, values = _rotation_angles(features, qubits)
    gate = _rotation_gate(rotation)
    circuit = Circuit(max(ordered) + 1)
    for qubit, angle in zip(ordered, values, strict=True):
        circuit.gate(gate, qubit, theta=angle)
    return circuit


def append_angular_encode(
    circuit: Circuit,
    features: Any,
    qubits: Sequence[int],
    *,
    rotation: str = _DEFAULT_ROTATION,
) -> None:
    """Append one rotation per qubit carrying ``features`` to ``circuit`` in place.

    This is the form a feature map needs, because a data-reuploading block is an
    encoding interleaved with a variational circuit rather than an encoding that
    stands alone.

    Args:
        circuit: The circuit to extend.
        features: The angle vector, validated as in :func:`angular_encode`.
        qubits: The qubits to rotate, one per feature, in feature order.
        rotation: The rotation axis, validated as in :func:`angular_encode`.

    Raises:
        TypeError: If ``features`` fails the validation described in
            :func:`angular_encode`.
        ValueError: If ``features``, ``qubits``, or ``rotation`` fail the
            validation described in :func:`angular_encode`.
    """
    declared, values = _rotation_angles(features, qubits)
    gate = _rotation_gate(rotation)
    for qubit, angle in zip(declared, values, strict=True):
        circuit.gate(gate, qubit, theta=angle)


def _rotation_gate(rotation: str) -> str:
    """Return the gate name for a rotation axis.

    Args:
        rotation: The axis name, matched case-insensitively.

    Returns:
        ``"rx"``, ``"ry"``, or ``"rz"``.

    Raises:
        ValueError: If the name is not a string, or is not one of the three axes.
    """
    if not isinstance(rotation, str):
        raise ValueError(f"rotation must be a string, got {type(rotation).__name__}")
    key = rotation.upper()
    gate = _ROTATION_GATES.get(key)
    if gate is None:
        raise ValueError(
            f"unsupported rotation {rotation!r}; expected 'X', 'Y', or 'Z'"
        )
    return gate


def _rotation_angles(
    features: Any, qubits: Sequence[int] | None
) -> tuple[list[int], torch.Tensor]:
    """Validate a feature vector and the qubits it rotates.

    Args:
        features: The candidate angle vector.
        qubits: The requested qubits, or ``None`` for the leading register.

    Returns:
        ``(ordered, values)``: the qubits in the caller's order, and the angles as
        a real tensor of the same length.

    Raises:
        TypeError: If the vector cannot be read as a real tensor.
        ValueError: If the vector is not one-dimensional, is empty, is not
            finite, or if the qubits fail their validation.
    """
    values = _as_tensor(features, noun="features")
    if values.dim() != 1:
        raise ValueError(
            "features must be one-dimensional, got shape "
            f"{tuple(int(size) for size in values.shape)}"
        )
    if values.shape[0] == 0:
        raise ValueError("features must not be empty")
    if values.is_complex():
        raise TypeError(
            "features must be real numbers, got a complex tensor; a rotation "
            "angle has no imaginary part"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("features must be finite")
    ordered = _resolve_qubits(qubits, int(values.shape[0]), noun="features")
    return ordered, values


def _amplitude_vector(amplitudes: Any, pad: complex | float) -> torch.Tensor:
    """Validate a feature vector, pad it, and normalise it.

    The padding comes first and the normalisation second, so the pad value is
    part of the vector the norm is taken over; padding after normalising would
    leave the result unnormalised.

    Args:
        amplitudes: The candidate feature vector.
        pad: The value appended to reach a power-of-two length.

    Returns:
        The normalised vector, one entry per basis state.

    Raises:
        TypeError: If the vector or the pad value has an unsupported type.
        ValueError: If the vector is not one-dimensional, is empty, is not
            finite, is the zero vector once padded, or describes fewer than two
            amplitude entries.
    """
    values = _as_tensor(amplitudes, noun="amplitudes")
    if values.dim() != 1:
        raise ValueError(
            "amplitudes must be one-dimensional, got shape "
            f"{tuple(int(size) for size in values.shape)}"
        )
    if values.shape[0] == 0:
        raise ValueError("amplitudes must not be empty")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("amplitudes must be finite")
    if isinstance(pad, bool) or not isinstance(pad, (int, float, complex)):
        raise TypeError(f"pad must be a real number, got {type(pad).__name__}")
    if isinstance(pad, complex):
        if pad.imag != 0.0:
            raise TypeError(f"pad must be a real number, got the complex value {pad!r}")
        pad = pad.real
    if pad != pad or pad in (float("inf"), float("-inf")):
        raise ValueError(f"pad must be finite, got {pad!r}")

    data = values.to(dtype=torch.complex64)
    size = int(data.shape[0])
    target = 1 << (size - 1).bit_length()
    if target != size:
        data = torch.cat([data, torch.full((target - size,), pad, dtype=data.dtype)])
    if target < 2:
        raise ValueError(
            "amplitudes needs at least two entries once padded, got 1; a single "
            "amplitude describes a register of no qubits, which a circuit cannot "
            "carry"
        )
    norm = float(data.norm())
    if norm == 0.0:
        raise ValueError(
            "amplitudes must not be the zero vector; nothing can be normalised"
        )
    return data / norm


def _as_tensor(values: Any, *, noun: str) -> torch.Tensor:
    """Read a feature vector as a flat-agnostic CPU tensor.

    ``str``, ``bytes`` and ``bytearray`` are refused before the conversion:
    ``torch.as_tensor(b"01")`` reads the bytes as a buffer and would accept a
    value that is plainly not a feature vector.

    Args:
        values: The candidate vector.
        noun: The parameter name, for the failure message.

    Returns:
        The vector as a tensor, on the CPU, with its element type preserved.

    Raises:
        TypeError: If the value has an unsupported type.
    """
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(
            f"{noun} must be a sequence of numbers, got {type(values).__name__}"
        )
    if isinstance(values, torch.Tensor):
        return values.to(device="cpu")
    try:
        return torch.as_tensor(values, device="cpu")
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{noun} must be a torch.Tensor, an array, or a sequence of real or "
            f"complex numbers, got {type(values).__name__}"
        ) from error


def _resolve_qubits(
    qubits: Sequence[int] | None, count: int, *, noun: str
) -> list[int]:
    """Validate the qubits an encoding targets and return them as a list.

    Args:
        qubits: The requested qubits, or ``None`` for the leading register.
        count: The number of qubits the features require.
        noun: The parameter name the count is stated against.

    Returns:
        The qubits in the order the caller gave them.

    Raises:
        ValueError: If the count does not match, or a qubit repeats, is negative,
            or is not an integer.
    """
    if qubits is None:
        return list(range(count))
    ordered = list(qubits)
    if len(ordered) != count:
        raise ValueError(f"the {noun} require {count} qubits, got {len(ordered)}")
    if len(set(ordered)) != len(ordered):
        raise ValueError("qubits must be distinct and appear once each")
    for qubit in ordered:
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise ValueError(f"qubits must be integers, got {qubit!r}")
        if qubit < 0:
            raise ValueError(f"qubits must be non-negative, got {qubit}")
    return ordered
