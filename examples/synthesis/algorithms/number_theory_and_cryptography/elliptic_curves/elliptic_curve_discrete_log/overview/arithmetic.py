"""Bounded reversible arithmetic for the notebook's field F_7."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from flagquantum import Circuit

Word = tuple[int, ...]
Gate = tuple[Word, int]
Predicate = tuple[tuple[int, int], ...]
MODULUS = 7
WIDTH = 3
ROUNDS = 2 * WIDTH


@dataclass
class ReversibleBlock:
    """Keep polarity-aware controls until lowering to native reversible gates."""

    gates: list[tuple[Predicate, int]] = field(default_factory=list)

    def flip(self, target: int, when: Predicate = ()) -> None:
        self.gates.append((when, target))

    def extend(
        self, block: "ReversibleBlock", when: Predicate = (), *, inverse: bool = False
    ) -> None:
        gates = reversed(block.gates) if inverse else iter(block.gates)
        self.gates.extend((when + controls, target) for controls, target in gates)

    def lower(self, scratch: Word) -> list[Gate]:
        """Use a clean Toffoli ladder and restore it after each controlled flip."""
        result: list[Gate] = []
        for conditions, target in self.gates:
            controls = tuple(wire for wire, _ in conditions)
            if len(set((*controls, target, *scratch))) != len(controls) + 1 + len(
                scratch
            ):
                raise ValueError("reversible gate wires overlap")
            negations = [((), wire) for wire, bit in conditions if bit == 0]
            result.extend(negations)
            if len(controls) <= 2:
                result.append((controls, target))
            else:
                needed = len(controls) - 2
                if len(scratch) < needed:
                    raise ValueError("insufficient Toffoli ladder workspace")
                ladder = [(controls[:2], scratch[0])]
                for i in range(1, needed):
                    ladder.append(((scratch[i - 1], controls[i + 1]), scratch[i]))
                result.extend(ladder)
                result.append(((scratch[needed - 1], controls[-1]), target))
                result.extend(reversed(ladder))
            result.extend(reversed(negations))
        return result


def equal(word: Word, value: int) -> Predicate:
    return tuple((wire, (value >> i) & 1) for i, wire in enumerate(word))


def swap(block: ReversibleBlock, first: int, second: int) -> None:
    block.flip(second, ((first, 1),))
    block.flip(first, ((second, 1),))
    block.flip(second, ((first, 1),))


def rotate(block: ReversibleBlock, word: Word, *, right: bool = False) -> None:
    positions = range(len(word) - 1) if right else range(len(word) - 2, -1, -1)
    for i in positions:
        swap(block, word[i], word[i + 1])


def transpose(block: ReversibleBlock, word: Word, start: int, end: int) -> None:
    """Connect two basis words by a Gray path, then restore its intermediates."""
    path: list[tuple[Predicate, int]] = []
    for bit, wire in enumerate(word):
        if (start ^ end) & (1 << bit):
            controls = tuple(entry for entry in equal(word, start) if entry[0] != wire)
            path.append((controls, wire))
            start ^= 1 << bit
    block.gates.extend(path)
    block.gates.extend(reversed(path[:-1]))


def permute(block: ReversibleBlock, word: Word, image: Sequence[int]) -> None:
    visited: set[int] = set()
    for start in range(len(image)):
        if start in visited:
            continue
        visited.add(start)
        end = image[start]
        while end != start:
            transpose(block, word, start, end)
            visited.add(end)
            end = image[end]


def translate(word: Word, increment: int, modulus: int = MODULUS) -> ReversibleBlock:
    """Add a fixed increment, with identity action on the extra field encoding."""
    block = ReversibleBlock()
    permute(
        block,
        word,
        [
            (value + increment) % modulus if value < modulus else value
            for value in range(8)
        ],
    )
    return block


def accumulate(
    block: ReversibleBlock,
    source: Word,
    target: Word,
    *,
    scale: int = 1,
    modulus: int = MODULUS,
) -> None:
    for i, wire in enumerate(source):
        block.extend(translate(target, scale << i, modulus), ((wire, 1),))


def product(
    target: Word, first: Word, second: Word, *, scale: int = 1
) -> ReversibleBlock:
    """Accumulate the product with controlled binary partial products."""
    block = ReversibleBlock()
    for i, left in enumerate(first):
        for j, right in enumerate(second):
            when = ((left, 1),) if left == right else ((left, 1), (right, 1))
            block.extend(translate(target, scale << (i + j)), when)
    return block


def xor_function(
    block: ReversibleBlock,
    source: Word,
    target: Word,
    function: Callable[[int], int],
) -> None:
    for value in range(1 << len(source)):
        output = function(value)
        for i, wire in enumerate(target):
            if (output >> i) & 1:
                block.flip(wire, equal(source, value))


def circuit_from_gates(width: int, gates: Iterable[Gate]) -> Circuit:
    circuit = Circuit(width)
    for controls, target in gates:
        circuit.gate(("x", "cx", "ccx")[len(controls)], (*controls, target))
    return circuit
