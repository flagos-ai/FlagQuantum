"""Dense Pauli-product numerics shared by algorithm workflows."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence

import torch

from ..errors import ValidationError
from .matrices import GATE_MAT_DICT
from .statevector.operations import _apply_matrix

PauliProduct = Sequence[tuple[int, str]]

_PAULI_WORD_CHARACTERS = frozenset({"I", "X", "Y", "Z"})
"""The four characters a Pauli word is written with, in CUDA-Q's own spelling."""


def infer_n_wires_from_dense_state(state: torch.Tensor) -> int:
    """Infer a qubit count from a dense statevector or density matrix."""

    dimension = int(state.shape[-1])
    n_wires = dimension.bit_length() - 1
    if dimension <= 0 or 2**n_wires != dimension:
        raise ValueError("State dimension must be a power of two.")
    return n_wires


def pauli_product_operator(
    operators: PauliProduct,
    n_wires: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Materialize one dense Pauli-product operator."""

    by_wire = {int(wire): str(name).lower() for wire, name in operators}
    if any(wire < 0 or wire >= n_wires for wire in by_wire):
        raise ValueError("Pauli operator wire index out of range")
    result = torch.ones(1, 1, dtype=dtype, device=device)
    for wire in range(int(n_wires)):
        matrix = GATE_MAT_DICT[by_wire.get(wire, "i")]
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("Pauli products require fixed gate matrices.")
        matrix = matrix.to(
            device=device,
            dtype=dtype,
        )
        result = torch.kron(result, matrix)
    return result


def pauli_product_statevector_expectation(
    state: torch.Tensor,
    operators: PauliProduct,
    n_wires: int,
) -> torch.Tensor:
    """Evaluate one Pauli product on a dense statevector batch."""

    batch = state.reshape(1, -1) if state.ndim == 1 else state
    transformed = batch
    for wire, name in operators:
        matrix = GATE_MAT_DICT[str(name).lower()]
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("Pauli products require fixed gate matrices.")
        matrix = matrix.to(
            device=batch.device,
            dtype=batch.dtype,
        )
        transformed = _apply_matrix(transformed, matrix, (int(wire),), n_wires)
    return torch.real((torch.conj(batch) * transformed).sum(dim=-1))


def pauli_product_density_expectation(
    density: torch.Tensor,
    operators: PauliProduct,
    n_wires: int,
) -> torch.Tensor:
    """Evaluate one Pauli product on a dense density-matrix batch."""

    batch = density.reshape(1, *density.shape) if density.ndim == 2 else density
    operator = pauli_product_operator(
        operators,
        n_wires,
        dtype=batch.dtype,
        device=batch.device,
    )
    values = torch.einsum("bij,ji->b", batch, operator)
    return torch.real(values)


def _finite_angle(theta: float) -> float:
    """Read one rotation angle, refusing a value that does not denote a number.

    ``bool`` and ``str`` are refused although ``float`` reads both: a flag is not an
    angle, and the text of a number is a mistyped value rather than the number.  An
    infinite or NaN angle would be carried into a silently wrong unitary, so it is
    refused here rather than at the arithmetic.
    """

    if isinstance(theta, (bool, str)):
        raise ValidationError(f"theta must be a real number, got {theta!r}")
    try:
        angle = float(theta)
    except (TypeError, ValueError):
        raise ValidationError(f"theta must be a real number, got {theta!r}") from None
    if not math.isfinite(angle):
        raise ValidationError(f"theta must be finite, got {theta!r}")
    return angle


def _pauli_word_targets(targets: Sequence[int]) -> tuple[int, ...]:
    """Read the targets of a Pauli word one label at a time.

    A label is read through the interpreter's own ``__index__`` protocol, which admits
    NumPy integers and zero-dimensional integer tensors while refusing floats and
    strings.  ``bool`` is refused although it satisfies ``operator.index``: a flag is
    not a wire, and reading it as one would address an operator nobody asked for.
    """

    labels: list[int] = []
    for target in targets:
        if isinstance(target, bool):
            raise ValidationError(
                f"Pauli word target must be an integer, got {target!r}"
            )
        try:
            labels.append(operator.index(target))
        except TypeError:
            raise ValidationError(
                f"Pauli word target must be an integer, got {target!r}"
            ) from None
    if any(label < 0 for label in labels):
        raise ValidationError(
            f"Pauli word targets must be non-negative, got {tuple(labels)}"
        )
    return tuple(labels)


def _pauli_word_operators(
    word: str,
    targets: Sequence[int],
) -> tuple[tuple[int, str], ...]:
    """Read a Pauli word and its targets into the product form the kernels take.

    The word is read the way CUDA-Q reads it: one character per target, in that order,
    so ``"YX"`` beside ``(0, 2)`` is the product ``Y`` on wire 0 times ``X`` on wire 2.
    An ``I`` consumes a target and contributes no support, which is what lets a caller
    align a word with a wire list without dropping the wires it does not act on.  A
    word and a target sequence of different lengths, a character outside ``X``, ``Y``,
    ``Z`` and ``I``, and a repeated target are each refused, because none of them
    denotes a Pauli product.
    """

    if not isinstance(word, str):
        raise ValidationError(f"Pauli word must be a string, got {word!r}")
    labels = _pauli_word_targets(targets)
    if len(word) != len(labels):
        raise ValidationError(
            f"Pauli word {word!r} has {len(word)} characters for {len(labels)} "
            "targets, and a word character acts on exactly one target"
        )
    unknown = sorted(set(word) - _PAULI_WORD_CHARACTERS)
    if unknown:
        raise ValidationError(
            f"Pauli word {word!r} contains {unknown}, and every character must be "
            "one of X, Y, Z, or I"
        )
    if len(set(labels)) != len(labels):
        raise ValidationError(f"Pauli word targets must be unique, got {labels}")
    return tuple(zip(labels, word, strict=True))


def exponential_pauli_operator(
    theta: float,
    word: str,
    targets: Sequence[int],
    n_qubits: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Materialize the dense unitary ``exp(-i * theta * P)`` of one Pauli word.

    ``word`` and ``targets`` are read positionally, so character ``i`` of the word acts
    on ``targets[i]`` and together they name the product ``P`` that is exponentiated.
    The product is placed on ``n_qubits`` qubits, and an ``I`` character contributes no
    support while still consuming its target.

    A Pauli product squares to the identity, so the exponential is evaluated in closed
    form rather than by exponentiating a matrix:
    ``exp(-i * theta * P) = cos(theta) * I - i * sin(theta) * P``.  The expression is
    exact for every angle, and an all-identity word therefore returns the global phase
    ``exp(-i * theta) * I`` it denotes rather than the identity.

    Examples:
        >>> import torch
        >>> from flagquantum.simulation.pauli import exponential_pauli_operator
        >>> step = exponential_pauli_operator(
        ...     1.0, "XX", (0, 1), 2, dtype=torch.complex128, device="cpu"
        ... )
        >>> float(step[0, 0].real)
        0.5403023058681398
    """

    angle = _finite_angle(theta)
    operators = _pauli_word_operators(word, targets)
    identity = pauli_product_operator((), n_qubits, dtype=dtype, device=device)
    product = pauli_product_operator(operators, n_qubits, dtype=dtype, device=device)
    return math.cos(angle) * identity - 1j * math.sin(angle) * product


__all__ = (
    "exponential_pauli_operator",
    "infer_n_wires_from_dense_state",
    "pauli_product_density_expectation",
    "pauli_product_operator",
    "pauli_product_statevector_expectation",
)
