"""Public Circuit building blocks for the full reversible compilation."""

from collections.abc import Callable, Sequence

import flagquantum as fq
from flagquantum.algorithms.primitives.oracle import (
    append_comparator,
    append_multi_controlled_x,
)

Leaf = Callable[[fq.Circuit, int], None]


def mcx(
    circuit: fq.Circuit, controls: Sequence[int], target: int, scratch: Sequence[int]
) -> None:
    """Append a clean-ancilla saturation ladder, including zero controls."""
    if controls:
        append_multi_controlled_x(
            circuit, controls, target, ancillas=scratch[: max(0, len(controls) - 2)]
        )
    else:
        circuit.x(target)


def positive_phase_flip(
    circuit: fq.Circuit, controls: Sequence[int], target: int, scratch: Sequence[int]
) -> None:
    circuit.h(target)
    mcx(circuit, controls, target, scratch)
    circuit.h(target)


def xor_polynomial(
    values: Sequence[int],
    input_wires: Sequence[int],
    output_wires: Sequence[int],
    scratch: Sequence[int],
    circuit: fq.Circuit,
) -> None:
    """Compile a Boolean word table's algebraic normal form."""
    coefficients = list(values)
    for bit in range(len(input_wires)):
        mask = 1 << bit
        for word in range(len(coefficients)):
            if word & mask:
                coefficients[word] ^= coefficients[word ^ mask]
    for word, outputs in enumerate(coefficients):
        if not outputs:
            continue
        controls = [wire for bit, wire in enumerate(input_wires) if word >> bit & 1]
        for bit, out in enumerate(output_wires):
            if outputs >> bit & 1:
                mcx(circuit, controls, out, scratch)


def less_than_xor(
    circuit: fq.Circuit,
    left: Sequence[int],
    right: Sequence[int],
    target: int,
    work: Sequence[int],
) -> None:
    """XOR an unsigned less-than comparison with the native clean comparator."""
    append_comparator(
        circuit,
        lhs=right[::-1],
        rhs=left[::-1],
        target=target,
        equality=work[: len(left) + 1],
        scratch=work[len(left) + 1],
    )


def table_walk(
    circuit: fq.Circuit,
    address: Sequence[int],
    rows: Sequence[Leaf],
    scratch: Sequence[int],
    control: int | None = None,
) -> None:
    """Emit the source's asymmetric binary traversal and clean its branch flags."""
    if not rows:
        raise ValueError("a table walk needs at least one row")

    def visit(active: int, depth: int, entries: Sequence[Leaf]) -> None:
        if len(entries) == 1:
            entries[0](circuit, active)
            return
        count = 1 << ((len(entries) - 1).bit_length() - 1)
        wire, branch = address[depth], scratch[depth]
        circuit.x(wire).ccx(active, wire, branch).x(wire)
        visit(branch, depth + 1, entries[:count])
        circuit.cx(active, branch)
        visit(branch, depth + 1, entries[count:])
        circuit.ccx(active, wire, branch)

    if control is not None:
        visit(control, 0, rows)
    else:
        count = 1 << ((len(rows) - 1).bit_length() - 1)
        root = address[0]
        circuit.x(root)
        visit(root, 1, rows[:count])
        circuit.x(root)
        visit(root, 1, rows[count:])


def qrom_rows(
    values: Sequence[Sequence[bool]], targets: Sequence[int]
) -> tuple[Leaf, ...]:
    def leaf(row: Sequence[bool]) -> Leaf:
        selected = tuple(w for w, flag in zip(targets, row, strict=True) if flag)

        def apply(circuit: fq.Circuit, enable: int) -> None:
            for wire in selected:
                circuit.cx(enable, wire)

        return apply

    return tuple(leaf(row) for row in values)


def _borrowed_saturation(
    circuit: fq.Circuit, controls: Sequence[int], target: int, borrowed: Sequence[int]
) -> None:
    first, second, *tail = controls
    ladder = tuple(
        (borrowed[i], bit, borrowed[i + 1]) for i, bit in enumerate(tail[:-1])
    )
    for _ in range(2):
        circuit.ccx(first, second, borrowed[0])
        for step in ladder:
            circuit.ccx(*step)
        circuit.ccx(borrowed[len(controls) - 3], tail[-1], target)
        for step in reversed(ladder):
            circuit.ccx(*step)


def wide_phase_flip(
    circuit: fq.Circuit, controls: Sequence[int], target: int, flags: tuple[int, int]
) -> None:
    """Implement one wide controlled Z with two clean flags."""
    mid = len(controls) // 2
    left, right = controls[:mid], controls[mid:]
    one, two = flags
    _borrowed_saturation(circuit, left, one, right)
    _borrowed_saturation(circuit, right, two, left)
    circuit.h(target).ccx(one, two, target).h(target)
    _borrowed_saturation(circuit, right, two, left)
    _borrowed_saturation(circuit, left, one, right)
