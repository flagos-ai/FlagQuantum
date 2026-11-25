"""Exact predicate toggles using restored borrowed storage."""

from collections.abc import Sequence

import flagquantum as fq


def dirty_ladder(
    circuit: fq.Circuit, controls: Sequence[int], target: int, borrowed: Sequence[int]
) -> None:
    count = len(controls)
    if count == 1:
        circuit.cx(controls[0], target)
        return
    if count == 2:
        circuit.ccx(controls[0], controls[1], target)
        return
    if len(borrowed) < count - 2:
        raise ValueError("The dirty ladder requires one borrowed bit per internal rung")
    for _ in range(2):
        circuit.ccx(controls[-1], borrowed[count - 3], target)
        for index in range(count - 2, 1, -1):
            circuit.ccx(controls[index], borrowed[index - 2], borrowed[index - 1])
        circuit.ccx(controls[0], controls[1], borrowed[0])
        for index in range(2, count - 1):
            circuit.ccx(controls[index], borrowed[index - 2], borrowed[index - 1])


def controlled_x(
    circuit: fq.Circuit, controls: Sequence[int], target: int, borrowed: int
) -> None:
    labels = (*controls, target, borrowed)
    if len(set(labels)) != len(labels) or not controls:
        raise ValueError(
            "Use nonempty distinct controls and disjoint target and borrowed bit"
        )
    if len(controls) <= 2:
        dirty_ladder(circuit, controls, target, ())
        return
    split = (len(controls) + 1) // 2
    first, second = controls[:split], controls[split:]
    for _ in range(2):
        dirty_ladder(circuit, first, borrowed, second)
        dirty_ladder(circuit, (borrowed, *second), target, first)
