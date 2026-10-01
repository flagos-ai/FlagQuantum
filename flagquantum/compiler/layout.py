"""Logical-to-physical wire layouts for compiled programs.

A routing pass moves logical wires onto physical wires. :class:`Layout` is that
movement as a value, :func:`apply_layout` places a program on one, and
:func:`final_layout` reads the one a routed program ended on.

Routing in this package always walks every logical wire back onto its own
physical wire before returning, which is what makes a routed program's output
layout the identity and keeps a measurement naming a logical wire correct. That
walk is an appended SWAP phase and it is the largest single part of a routed
program's SWAP count, so :func:`remove_layout_restore` removes it for a caller
that reads results per physical wire and reorders them with
:meth:`Layout.to_logical_order`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir

__all__ = (
    "Layout",
    "apply_layout",
    "final_layout",
    "remove_layout_restore",
)


@dataclass(frozen=True)
class Layout:
    """A complete assignment of logical wires to physical wires.

    ``logical_to_physical[logical]`` is the physical wire that carries
    ``logical``. Every routing pass in this package places a circuit onto exactly
    as many physical wires as the circuit has logical wires, so the assignment is
    always a permutation of ``range(n_wires)``. A device with idle physical wires
    has no ``Layout`` here: :func:`apply_layout` refuses a layout that does not
    cover the circuit rather than leaving wires unplaced.

    A physical SWAP is not an exchange of two entries of this permutation. It
    exchanges the wires the two *positions* currently hold, so
    :meth:`apply_swaps` is the operation to use; indexing the tuple by a physical
    wire name silently reorders the wrong wires.

    Examples:
        Read a two-wire program's per-wire outcomes in logical-wire order after
        routing placed logical 0 on physical 1:

        >>> layout = Layout((1, 0))
        >>> layout.physical_of(0), layout.logical_of(1)
        (1, 0)
        >>> layout.to_logical_order("ab")
        ('b', 'a')
    """

    logical_to_physical: tuple[int, ...]

    def __init__(self, logical_to_physical: Iterable[int]) -> None:
        placement = tuple(logical_to_physical)
        if any(type(item) is not int for item in placement):
            raise ValueError(
                f"layout must be a tuple of integer physical wires, got {placement!r}"
            )
        if not placement:
            raise ValueError("layout must place at least one logical wire")
        if sorted(placement) != list(range(len(placement))):
            raise ValueError(
                f"layout must be a permutation of the circuit wires, got {placement!r}"
            )
        object.__setattr__(self, "logical_to_physical", placement)

    def __str__(self) -> str:
        return f"Layout({self.logical_to_physical})"

    @property
    def n_wires(self) -> int:
        """Return the number of logical wires, which is also the physical count."""
        return len(self.logical_to_physical)

    @property
    def physical_to_logical(self) -> tuple[int, ...]:
        """Return the inverse assignment; index it by physical wire."""
        inverse = [0] * len(self.logical_to_physical)
        for logical, physical in enumerate(self.logical_to_physical):
            inverse[physical] = logical
        return tuple(inverse)

    def physical_of(self, logical: int) -> int:
        """Return the physical wire that carries ``logical``."""
        if not 0 <= logical < self.n_wires:
            raise ValueError(f"logical wire {logical} is outside the layout")
        return self.logical_to_physical[logical]

    def logical_of(self, physical: int) -> int:
        """Return the logical wire that ``physical`` carries."""
        if not 0 <= physical < self.n_wires:
            raise ValueError(f"physical wire {physical} is outside the layout")
        return self.physical_to_logical[physical]

    def map_wires(self, wires: Iterable[int]) -> tuple[int, ...]:
        """Return ``wires`` read as logical wires, expressed as physical wires."""
        return tuple(self.physical_of(wire) for wire in wires)

    def apply_swaps(self, swaps: Iterable[tuple[int, int]]) -> Layout:
        """Return the layout left behind by SWAPs on physical wire pairs.

        Raises:
            ValueError: if a SWAP names a wire outside the layout.
        """

        logical_to_physical = list(self.logical_to_physical)
        for left, right in swaps:
            if not 0 <= left < self.n_wires or not 0 <= right < self.n_wires:
                raise ValueError(
                    f"SWAP wire pair ({left}, {right}) is outside a "
                    f"{self.n_wires}-wire layout"
                )
            if left == right:
                continue
            left_logical = logical_to_physical.index(left)
            right_logical = logical_to_physical.index(right)
            logical_to_physical[left_logical] = right
            logical_to_physical[right_logical] = left
        return Layout(logical_to_physical)

    def to_logical_order(self, values: Sequence[Any]) -> tuple[Any, ...]:
        """Return one value per logical wire, from a sequence indexed by physical wire.

        A program routed without its layout restore reports results indexed by
        physical wire; ``values`` is one entry per physical wire and the result is
        the same entries read in logical-wire order.

        Raises:
            ValueError: if ``values`` does not have one entry per physical wire.
        """

        ordered = tuple(values)
        if len(ordered) != self.n_wires:
            raise ValueError(
                "to_logical_order needs one value per physical wire; got "
                f"{len(ordered)} for {self.n_wires} wires"
            )
        return tuple(ordered[physical] for physical in self.logical_to_physical)


def apply_layout(circuit_or_ir: Any, layout: Layout) -> CircuitIR:
    """Place a program on ``layout``, relabelling every wire it names.

    Instructions, observables, and measurements move together, so the result
    computes the same thing under different wire labels and keeps ``n_wires``.
    This is the relabelling a caller needs to hand a program to a pass that
    expects it to start on a specific layout.

    The applied assignment is recorded under ``metadata["layout"]``, because it is
    the only record of the wire labels the program arrived with.

    Raises:
        TypeError: if ``layout`` is not a :class:`Layout`.
        ValueError: if the layout does not cover exactly the circuit's wires.
    """

    ir = ensure_circuit_ir(circuit_or_ir)
    if not isinstance(layout, Layout):
        raise TypeError("layout must be a Layout")
    if layout.n_wires != ir.n_wires:
        raise ValueError(
            f"layout covers {layout.n_wires} physical wires but the circuit has "
            f"{ir.n_wires} logical wires; a layout must cover exactly the circuit"
        )
    metadata = dict(ir.metadata)
    metadata["layout"] = {
        "n_wires": layout.n_wires,
        "logical_to_physical": layout.logical_to_physical,
        "physical_to_logical": layout.physical_to_logical,
    }
    return replace(
        ir,
        instructions=tuple(
            Instruction(
                instruction.name,
                layout.map_wires(instruction.wires),
                params=instruction.params,
                matrix=instruction.matrix,
                metadata=instruction.metadata,
            )
            for instruction in ir.instructions
        ),
        observables=tuple(
            replace(node, wires=layout.map_wires(node.wires)) for node in ir.observables
        ),
        measurements=tuple(
            replace(node, wires=layout.map_wires(node.wires))
            for node in ir.measurements
        ),
        metadata=metadata,
    )


def _reported_layout(routing: dict[str, Any], key: str, n_wires: int) -> Layout:
    placement = routing.get(key)
    if not isinstance(placement, (list, tuple)):
        raise ValueError(f"routing metadata does not report {key}")
    if len(placement) != n_wires:
        raise ValueError(
            f"routing metadata reports {key} for {len(placement)} wires, but the "
            f"circuit has {n_wires}"
        )
    try:
        return Layout(tuple(placement))
    except ValueError as error:
        raise ValueError(
            f"routing metadata reports an invalid {key}: {error}"
        ) from error


def final_layout(circuit_or_ir: Any) -> Layout:
    """Return the layout a routed program ends on, from its routing metadata.

    Raises:
        ValueError: if the program has no routing metadata, or if the layout that
            metadata reports is not a complete assignment of the circuit's wires.
    """

    ir = ensure_circuit_ir(circuit_or_ir)
    routing = ir.metadata.get("routing")
    if not isinstance(routing, dict):
        raise ValueError("program has no routing metadata; route it first")
    return _reported_layout(routing, "final_logical_to_physical", ir.n_wires)


def _routing_swap_phases(ir: CircuitIR) -> list[tuple[int, str, tuple[int, int]]]:
    """Return ``(index, phase, wires)`` for every routed SWAP, validating each one."""

    marked: list[tuple[int, str, tuple[int, int]]] = []
    for index, instruction in enumerate(ir.instructions):
        phase = instruction.metadata.get("routing_phase")
        if phase is None:
            continue
        if instruction.name != "swap" or len(instruction.wires) != 2:
            raise ValueError(f"routed instruction {index} has invalid SWAP evidence")
        if instruction.metadata.get("source_instruction_index") is None:
            raise ValueError(f"routed instruction {index} has no source instruction")
        marked.append((index, str(phase), instruction.wires))
    return marked


def remove_layout_restore(circuit_or_ir: Any) -> CircuitIR:
    """Leave a routed program on the layout its routing pass reached.

    Routing appends a ``final_restore`` phase that walks every logical wire back
    onto its own physical wire, which is what makes a routed program's output
    layout the identity and lets an observable or a measurement keep naming a
    logical wire. Those trailing SWAPs carry no routing information. A caller that
    reads results per physical wire and reorders them with
    :meth:`Layout.to_logical_order` computes the same thing for fewer SWAPs, and
    the reported output layout says which physical wire carries each logical wire.

    Observables and measurements move onto that layout, because a routed program's
    instructions are on physical wires and the output layout is no longer the
    identity to absorb the difference.

    The result is a program to run, not a routing result to hand back to target
    legalization or to the deployment routing evidence: both require a routed
    program to end on the identity layout and refuse this one by design.

    A strategy that restores after every gate rather than once at the end, such
    as ``restore_after_each_gate``, has no appended restore to remove and the
    program is returned unchanged. A program with no routing metadata is
    rejected, because there is no layout to report and no evidence to check.

    Routing SWAP counts in the metadata are rewritten to describe the program
    returned, and ``removed_layout_restore_swap_count`` records the saving.

    Raises:
        ValueError: if the program has no routing metadata, if a SWAP on the
            program contradicts the layout its metadata reports, or if a
            ``final_restore`` phase is not an appended tail of the program.
    """

    ir = ensure_circuit_ir(circuit_or_ir)
    routing = ir.metadata.get("routing")
    if not isinstance(routing, dict):
        raise ValueError(
            "program has no routing metadata; route it before removing its layout "
            "restore"
        )
    if routing.get("mapping_restored") is not True:
        return ir
    marked = _routing_swap_phases(ir)
    restore_indexes = {index for index, phase, _ in marked if phase == "final_restore"}
    if not restore_indexes:
        return ir
    forward_swaps = [
        wires for index, _, wires in marked if index not in restore_indexes
    ]
    placed_before_restore = _reported_layout(
        routing, "initial_logical_to_physical", ir.n_wires
    ).apply_swaps(forward_swaps)
    reported = _reported_layout(routing, "pre_restore_logical_to_physical", ir.n_wires)
    if placed_before_restore != reported:
        raise ValueError(
            "routing metadata does not agree with the SWAP evidence: replaying the "
            "program's non-restore SWAPs from the initial layout reaches "
            f"{placed_before_restore}, not the reported pre-restore layout {reported}"
        )
    first_restore = min(restore_indexes)
    if restore_indexes != set(range(first_restore, len(ir.instructions))):
        raise ValueError(
            "the final_restore phase is interleaved with the program rather than "
            "appended to it, so it cannot be removed as a group"
        )
    kept = list(ir.instructions[:first_restore])
    inserted_swap_count = sum(
        1
        for instruction in kept
        if instruction.metadata.get("routing_phase") is not None
    )
    updated = dict(routing) | {
        "final_logical_to_physical": reported.logical_to_physical,
        "mapping_restored": False,
        "inserted_swap_count": inserted_swap_count,
        "planned_inserted_swap_count": inserted_swap_count,
        "removed_layout_restore_swap_count": len(restore_indexes),
    }
    if updated.get("post_optimization_inserted_swap_count") is not None:
        updated["post_optimization_inserted_swap_count"] = inserted_swap_count
    metadata = dict(ir.metadata) | {"routing": updated}
    return replace(
        ir,
        instructions=tuple(kept),
        observables=tuple(
            replace(node, wires=reported.map_wires(node.wires))
            for node in ir.observables
        ),
        measurements=tuple(
            replace(node, wires=reported.map_wires(node.wires))
            for node in ir.measurements
        ),
        metadata=metadata,
    )
