"""Qubit labels, and their relabelling when one program is placed onto another.

Two things are written once here because more than one entry point needs them.

Reading a label: ``int(value)`` accepted ``0.5`` as qubit ``0``, ``"0"`` as qubit
``0`` and ``True`` as qubit ``1``, so a mistyped label selected a different qubit
instead of failing. Composition and the circuit gate API both read labels, so the
rule belongs to neither of them.

Relabelling: composition takes a program built on its own qubits and continues it
on the qubits of a target. The map is total over the source width -- a source qubit
with no image is refused rather than mapped to itself, because an identity image
turns a short or mistyped map into a silently misplaced gate on a circuit that still
runs.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from ..errors import ValidationError

__all__ = ["normalize_qubits", "qubit_label", "qubit_map_from", "remap_qubits"]


def qubit_label(owner: str, value: Any) -> int:
    """Read one qubit label without quietly rewriting the caller's value.

    Only values that already denote an integer are accepted, through the
    interpreter's own ``__index__`` protocol, which admits NumPy integers and
    zero-dimensional torch integer tensors while refusing floats and strings.
    ``bool`` is refused although it satisfies ``operator.index``: it is a flag
    rather than a label. The range of the label is not this function's business --
    the width of the program it refers to decides that.
    """

    if isinstance(value, bool):
        raise TypeError(f"{owner} qubit must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{owner} qubit must be an integer, got {value!r}") from None


def normalize_qubits(qubits: Iterable[int] | int, *, owner: str) -> tuple[int, ...]:
    """Read the qubit arguments of one instruction, one label at a time.

    A single label and a sequence of labels are both accepted, and the order matters:
    a label is tried first, so a bare zero-dimensional tensor is read as the label it
    denotes instead of being iterated. ``str`` is excluded from the sequence form,
    because ``"01"`` is a mistyped label rather than two labels.
    """

    if isinstance(qubits, (str, bytes)):
        return (qubit_label(owner, qubits),)
    try:
        return (qubit_label(owner, qubits),)
    except TypeError as scalar_error:
        if not hasattr(qubits, "__iter__"):
            raise
        try:
            items = tuple(cast("Iterable[int]", qubits))
        except TypeError:
            # An iterable that refuses to be iterated as a sequence -- a
            # zero-dimensional tensor is the case that occurs -- is a bad label, not a
            # bad sequence, so the scalar refusal is the true one.
            raise scalar_error from None
        return tuple(qubit_label(owner, item) for item in items)


def qubit_map_from(
    local_qubits: int,
    *,
    qubits: Iterable[int] | int | None = None,
    qubit_map: Mapping[int, int] | None = None,
    owner: str = "composition",
) -> tuple[int, ...]:
    """Return the target qubit of every local qubit of one program.

    A single target qubit is read as a length-one sequence, so a one-qubit program
    accepts ``qubits=3`` as well as ``qubits=(3,)``. With neither argument given the
    map is the identity, which is the common case of continuing a program on the same
    qubits of a wider circuit.

    Args:
        local_qubits: Width of the program being placed.
        qubits: Target qubit of each local qubit, in local order.
        qubit_map: Target qubit of each local qubit, keyed by local qubit.
        owner: Name of the entry point, used in refusal messages.

    Returns:
        A tuple whose ``index``-th entry is the target qubit of local qubit ``index``.

    Raises:
        TypeError: If both ``qubits`` and ``qubit_map`` are given, if ``qubit_map`` is
            not a mapping, or if a label is not an integer.
        ValidationError: If the map does not name every local qubit, names a local
            qubit that does not exist, or names one target qubit twice.
    """

    if qubits is not None and qubit_map is not None:
        raise TypeError(f"{owner} accepts either qubits or qubit_map, not both")
    if qubit_map is not None:
        ordered = _map_from_mapping(local_qubits, qubit_map, owner=f"{owner} qubit_map")
    elif qubits is not None:
        ordered = normalize_qubits(qubits, owner=owner)
        if len(ordered) != local_qubits:
            raise ValidationError(
                f"{owner} qubits must name all {local_qubits} qubit(s) of the "
                f"composed program, got {len(ordered)}"
            )
    else:
        ordered = tuple(range(local_qubits))
    repeated = _first_repeat(ordered)
    if repeated is not None:
        raise ValidationError(
            f"{owner} cannot place two local qubits on qubit {repeated}: {ordered}"
        )
    return ordered


def remap_qubits(
    wires: Sequence[int],
    mapping: Mapping[int, int] | Sequence[int],
    *,
    owner: str = "composition",
) -> tuple[int, ...]:
    """Relabel ``wires`` through ``mapping``.

    Args:
        wires: Qubit labels to relabel.
        mapping: Either a mapping from source label to target label, or a sequence
            whose ``index``-th entry is the target of source label ``index``.
        owner: Name of the caller, used in refusal messages.

    Returns:
        The target of every label in ``wires``, in the original order.

    Raises:
        ValueError: If a label has no target, or a sequence mapping is indexed by a
            label it does not cover. A negative index is refused rather than read from
            the end of the sequence.
    """

    if isinstance(mapping, Mapping):
        targets = []
        for wire in wires:
            if wire not in mapping:
                raise ValueError(
                    f"{owner} references qubit {wire}, which the qubit map does not name"
                )
            targets.append(mapping[wire])
        return tuple(targets)
    ordered = tuple(mapping)
    targets = []
    for wire in wires:
        if wire < 0 or wire >= len(ordered):
            raise ValueError(
                f"{owner} references qubit {wire}, which a qubit map of "
                f"{len(ordered)} qubit(s) does not name"
            )
        targets.append(ordered[wire])
    return tuple(targets)


def _map_from_mapping(
    local_qubits: int,
    qubit_map: Mapping[Any, Any],
    *,
    owner: str,
) -> tuple[int, ...]:
    """Read an explicit local-to-target mapping as a dense target sequence."""

    if not isinstance(qubit_map, Mapping):
        raise TypeError(f"{owner} must be a mapping, got {type(qubit_map).__name__}")
    read: dict[int, int] = {}
    for key, value in qubit_map.items():
        local = qubit_label(f"{owner} source", key)
        if local < 0 or local >= local_qubits:
            raise ValidationError(
                f"{owner} names local qubit {local}, which a {local_qubits}-qubit "
                "program does not have"
            )
        read[local] = qubit_label(f"{owner} target", value)
    missing = tuple(local for local in range(local_qubits) if local not in read)
    if missing:
        raise ValidationError(f"{owner} must name every local qubit; missing {missing}")
    return tuple(read[local] for local in range(local_qubits))


def _first_repeat(values: Sequence[int]) -> int | None:
    """Return the first label that appears twice, or ``None`` if all are distinct."""

    seen: set[int] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
