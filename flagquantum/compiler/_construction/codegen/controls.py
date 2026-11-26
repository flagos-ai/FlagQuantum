"""Emit bounded controlled gates and modular permutations with clean scratch."""

from flagquantum.compiler.frontends._structured.model import Operation

from ....core._composition import controlled_instruction, inverted_instruction
from ....core.controlled import _clean_mcx_instructions
from ....core.ir import Instruction
from ..mcx import one_clean_mcx
from ..semantics._phase_polynomial import MAX_PHASE_DEGREE
from ..verification.profile import _fail


def _array_controlled_x(
    controls: tuple[int, ...],
    target: int,
    workspace_start: int,
    n_qubits: int,
    op: Operation,
) -> tuple[tuple[Instruction, ...], int]:
    """Reuse clean ladders and share their scratch across conjunction groups."""
    count = len(controls)
    scratch_width = max(count - 2, 0)
    group_size = count
    flags_width = 0
    # A flag holds each group result while its ladder scratch can be reused.
    # Use the direct ladder on a tie to avoid extra compute/uncompute.
    for size in range(2, count):
        flag_count = (count + size - 1) // size
        scratch = max(size - 2, flag_count - 2)
        if flag_count + scratch < flags_width + scratch_width:
            group_size, flags_width, scratch_width = size, flag_count, scratch

    workspace_width = flags_width + scratch_width
    if workspace_start + workspace_width > n_qubits:
        # Keep the existing smaller grouped implementation under tighter caps.
        grouped_width = count // 3
        if grouped_width and workspace_start + grouped_width <= n_qubits:
            return _bounded_control_group(
                Instruction("x", (target,)), controls, workspace_start, n_qubits
            )
        if count <= MAX_PHASE_DEGREE:
            return controlled_instruction(Instruction("x", (target,)), controls), 0
        if workspace_start < n_qubits:
            return one_clean_mcx(controls, target, workspace_start), 1
        _fail("lowering.workspace", "array control exceeds clean workspace", op)
    flags = tuple(range(workspace_start, workspace_start + flags_width))
    scratch_wires = tuple(
        range(workspace_start + flags_width, workspace_start + workspace_width)
    )
    gates: list[Instruction] = []

    def append(source: tuple[int, ...], destination: int) -> None:
        gates.extend(
            _clean_mcx_instructions(
                source, destination, scratch_wires[: max(len(source) - 2, 0)]
            )
        )

    if flags:
        groups = tuple(
            controls[i : i + group_size] for i in range(0, count, group_size)
        )
        for group, flag in zip(groups, flags, strict=True):
            append(group, flag)
        append(flags, target)
        for group, flag in zip(reversed(groups), reversed(flags), strict=True):
            append(group, flag)
    else:
        append(controls, target)
    return tuple(gates), workspace_width


def _bounded_control_group(
    gate: Instruction,
    controls: tuple[int, ...],
    position: int,
    n_qubits: int,
) -> tuple[tuple[Instruction, ...], int]:
    # Group verified wide masks without allocating a full control ladder.
    group_count = len(controls) // 3
    if not group_count or position + group_count > n_qubits:
        return controlled_instruction(gate, controls), 0
    flags = tuple(range(position, position + group_count))
    prefix = tuple(
        entry
        for ordinal, flag in enumerate(flags)
        for entry in controlled_instruction(
            Instruction("x", (flag,)), controls[3 * ordinal : 3 * (ordinal + 1)]
        )
    )
    middle = controlled_instruction(gate, (*flags, *controls[3 * group_count :]))
    return (
        *prefix,
        *middle,
        *(inverted_instruction(entry) for entry in reversed(prefix)),
    ), group_count


def _modular_exp_instructions(
    target: tuple[int, ...],
    exponent: tuple[int, ...],
    workspace_start: int,
    n_qubits: int,
    op: Operation,
) -> tuple[tuple[Instruction, ...], int]:
    """Compose controlled modular permutations using adjacent basis swaps."""
    modulus, base = op.attributes["modulus"], op.attributes["base"]
    gates: list[Instruction] = []
    workspace = 0

    def swap(control: int, start: int, end: int) -> None:
        nonlocal workspace
        path = [start]
        while path[-1] != end:
            difference = path[-1] ^ end
            path.append(path[-1] ^ (difference & -difference))
        edges = list(zip(path[:-1], path[1:], strict=True))
        for left, right in (*edges, *reversed(edges[:-1])):
            bit = (left ^ right).bit_length() - 1
            wire = target[-1 - bit]
            controls = (control, *(q for q in target if q != wire))
            flips = tuple(
                Instruction("x", (q,))
                for index, q in enumerate(reversed(target))
                if q != wire and not left & (1 << index)
            )
            middle, used = _array_controlled_x(
                controls, wire, workspace_start, n_qubits, op
            )
            workspace = max(workspace, used)
            gates.extend((*flips, *middle, *flips))

    for index, control in enumerate(reversed(exponent)):
        multiplier = pow(base, 1 << index, modulus)
        seen: set[int] = set()
        for first in range(modulus):
            if first in seen:
                continue
            seen.add(first)
            current = first * multiplier % modulus
            while current != first:
                swap(control, first, current)
                seen.add(current)
                current = current * multiplier % modulus
    return tuple(gates), workspace
