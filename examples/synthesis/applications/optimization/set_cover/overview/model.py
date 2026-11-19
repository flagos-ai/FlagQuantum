"""Build the original eight-subset Set Cover instance and full slack costs."""

from dataclasses import dataclass

import torch

SETS = (
    (1, 2, 3, 4),
    (2, 3, 4, 5),
    (6, 7),
    (8, 9, 10),
    (1, 6, 8),
    (3, 7, 9),
    (4, 7, 10),
    (2, 5, 8),
)
PENALTY = 10
DECISIONS = 8
PHYSICAL_WIDTH = 23
NAMES = ("x", *(f"s{i}" for i in range(10)))


@dataclass(frozen=True)
class Model:
    sets: tuple[tuple[int, ...], ...]
    coverers: tuple[tuple[int, ...], ...]
    widths: tuple[int, ...]

    def covers(self, word: int) -> bool:
        return all(any(word & (1 << i) for i in group) for group in self.coverers)


def source_model() -> Model:
    coverers = tuple(
        tuple(i for i, subset in enumerate(SETS) if item in subset)
        for item in range(1, 11)
    )
    return Model(SETS, coverers, (DECISIONS, *(len(group) - 1 for group in coverers)))


def costs(model: Model) -> torch.Tensor:
    """Enumerate all joint words in register declaration order in bounded blocks."""
    if sum(model.widths) != PHYSICAL_WIDTH:
        raise ValueError("set cover requires its full twenty-three-bit encoding")
    values = torch.empty(1 << PHYSICAL_WIDTH, dtype=torch.int16)
    block = 1 << 18
    for start in range(0, values.numel(), block):
        stop = min(start + block, values.numel())
        words = torch.arange(start, stop, dtype=torch.int32)
        selected = tuple(
            (words >> (PHYSICAL_WIDTH - DECISIONS + bit)) & 1
            for bit in range(DECISIONS)
        )
        total = words.new_zeros(words.shape)
        for bit_values in selected:
            total += bit_values
        offset = PHYSICAL_WIDTH - DECISIONS
        for group, width in zip(model.coverers, model.widths[1:], strict=True):
            offset -= width
            residual = words.new_ones(words.shape)
            for bit in range(width):
                residual += (words >> (offset + bit)) & 1
            for decision in group:
                residual -= selected[decision]
            total += PENALTY * residual.square()
        assert offset == 0
        values[start:stop] = total
    return values


def exact_solutions(model: Model) -> tuple[int, ...]:
    feasible = [word for word in range(1 << DECISIONS) if model.covers(word)]
    optimum = min(word.bit_count() for word in feasible)
    return tuple(word for word in feasible if word.bit_count() == optimum)
