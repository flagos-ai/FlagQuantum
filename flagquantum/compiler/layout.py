"""Logical-to-physical wire layouts for compiled programs.

A routing pass moves logical wires onto physical wires. :class:`Layout` is that
movement as a value, :func:`apply_layout` places a program on one, and
:func:`final_layout` reads the one a routed program ended on.

Routing in this package always restores before returning: every logical wire is
walked back onto the physical slot the route declared as its result, which is its
own wire when no workspace was allocated. That is what makes a routed program's
output layout the identity and keeps a measurement naming a logical wire correct.
The restore is an appended SWAP phase and it is the largest single part of a routed
program's SWAP count, so :func:`remove_layout_restore` removes it for a caller that
reads results per physical wire and reorders them with
:meth:`Layout.to_logical_order`.

A route that allocates physical workspace widens the program to the device width,
so a placement there leaves slots idle. :class:`Layout` carries that case too:
``physical_slot_count`` is that width and ``physical_to_logical`` reports ``None``
for a slot that holds no logical wire. ``None`` for an idle slot is the occupancy
convention :mod:`flagquantum.core._compilation_evidence_v3` already records its own
allocated routing evidence with. A route that never allocates stays on the wires
the program owns, and reports no idle slots even on a wider device.
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
    """An assignment of logical wires to physical wires.

    ``logical_to_physical[logical]`` is the physical wire that carries
    ``logical``, and no two logical wires share one. A routing pass that places a
    circuit onto exactly as many physical wires as the circuit has logical wires
    produces a permutation of ``range(n_wires)``, and that is the default: leave
    ``physical_slot_count`` out and the assignment must cover every wire.

    A device wider than the program is the other case. There the program owns
    fewer wires than the device has slots, so the assignment injects the logical
    wires into a subset of the slots and ``physical_slot_count`` is the device
    width. Slots that hold no logical wire are idle, ``physical_to_logical``
    reports ``None`` for them, and :func:`apply_layout` refuses such a layout
    because a program cannot name a wire outside its own width.

    A physical SWAP is not an exchange of two entries of the assignment. It
    exchanges the wires the two *positions* currently hold, so
    :meth:`apply_swaps` is the operation to use; indexing the tuple by a physical
    wire name silently reorders the wrong wires. A SWAP against an idle slot is
    how an allocated route moves a logical wire into its workspace.

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
    physical_slot_count: int

    def __init__(
        self, logical_to_physical: Iterable[int], physical_slot_count: int | None = None
    ) -> None:
        placement = tuple(logical_to_physical)
        if any(type(item) is not int for item in placement):
            raise ValueError(
                f"layout must be a tuple of integer physical wires, got {placement!r}"
            )
        if not placement:
            raise ValueError("layout must place at least one logical wire")
        if physical_slot_count is None:
            slots = len(placement)
            if sorted(placement) != list(range(slots)):
                raise ValueError(
                    "layout must be a permutation of the circuit wires, got "
                    f"{placement!r}"
                )
        else:
            if type(physical_slot_count) is not int:
                raise ValueError(
                    "layout physical_slot_count must be an integer, got "
                    f"{physical_slot_count!r}"
                )
            slots = physical_slot_count
            if slots < len(placement):
                raise ValueError(
                    f"layout places {len(placement)} logical wires but has only "
                    f"{slots} physical slots"
                )
            if len(set(placement)) != len(placement):
                raise ValueError(
                    "layout must place every logical wire on its own physical wire, "
                    f"got {placement!r}"
                )
            if any(item < 0 or item >= slots for item in placement):
                raise ValueError(
                    f"layout places a wire outside its {slots} physical slots, got "
                    f"{placement!r}"
                )
        object.__setattr__(self, "logical_to_physical", placement)
        object.__setattr__(self, "physical_slot_count", slots)

    def __str__(self) -> str:
        if self.has_idle_slots:
            return f"Layout({self.logical_to_physical}, {self.physical_slot_count})"
        return f"Layout({self.logical_to_physical})"

    @property
    def n_wires(self) -> int:
        """Return the number of logical wires this layout places."""
        return len(self.logical_to_physical)

    @property
    def has_idle_slots(self) -> bool:
        """Return whether some physical slot holds no logical wire."""
        return self.n_wires < self.physical_slot_count

    @property
    def physical_to_logical(self) -> tuple[int | None, ...]:
        """Return the occupancy; index it by physical slot, ``None`` if idle."""
        inverse: list[int | None] = [None] * self.physical_slot_count
        for logical, physical in enumerate(self.logical_to_physical):
            inverse[physical] = logical
        return tuple(inverse)

    def physical_of(self, logical: int) -> int:
        """Return the physical wire that carries ``logical``."""
        if not 0 <= logical < self.n_wires:
            raise ValueError(f"logical wire {logical} is outside the layout")
        return self.logical_to_physical[logical]

    def logical_of(self, physical: int) -> int:
        """Return the logical wire that ``physical`` carries.

        Raises:
            ValueError: if ``physical`` is outside the device or holds no wire.
        """
        if not 0 <= physical < self.physical_slot_count:
            raise ValueError(f"physical wire {physical} is outside the layout")
        occupied = self.physical_to_logical[physical]
        if occupied is None:
            raise ValueError(f"physical wire {physical} holds no logical wire")
        return occupied

    def map_wires(self, wires: Iterable[int]) -> tuple[int, ...]:
        """Return ``wires`` read as logical wires, expressed as physical wires."""
        return tuple(self.physical_of(wire) for wire in wires)

    def apply_swaps(self, swaps: Iterable[tuple[int, int]]) -> Layout:
        """Return the layout left behind by SWAPs on physical wire pairs.

        Raises:
            ValueError: if a SWAP names a slot outside the device.
        """

        occupancy = list(self.physical_to_logical)
        for left, right in swaps:
            if not 0 <= left < self.physical_slot_count or not (
                0 <= right < self.physical_slot_count
            ):
                raise ValueError(
                    f"SWAP wire pair ({left}, {right}) is outside a "
                    f"{self.physical_slot_count}-wire layout"
                )
            if left == right:
                continue
            occupancy[left], occupancy[right] = occupancy[right], occupancy[left]
        placement = [0] * self.n_wires
        for physical, logical in enumerate(occupancy):
            if logical is not None:
                placement[logical] = physical
        return Layout(placement, self.physical_slot_count)

    def to_logical_order(self, values: Sequence[Any]) -> tuple[Any, ...]:
        """Return one value per logical wire, from a sequence indexed by physical wire.

        A program routed without its layout restore reports results indexed by
        physical wire; ``values`` is one entry per physical slot of the device and
        the result is the same entries read in logical-wire order. An allocated
        route therefore drops the entries of its idle slots.

        Raises:
            ValueError: if ``values`` does not have one entry per physical slot.
        """

        ordered = tuple(values)
        if len(ordered) != self.physical_slot_count:
            raise ValueError(
                "to_logical_order needs one value per physical wire; got "
                f"{len(ordered)} for {self.physical_slot_count} wires"
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
        ValueError: if the layout does not cover exactly the circuit's wires, or if
            it leaves physical slots idle, which a program cannot name.
    """

    ir = ensure_circuit_ir(circuit_or_ir)
    if not isinstance(layout, Layout):
        raise TypeError("layout must be a Layout")
    if layout.n_wires != ir.n_wires or layout.has_idle_slots:
        raise ValueError(
            f"layout places {layout.n_wires} logical wires on "
            f"{layout.physical_slot_count} physical slots but the circuit has "
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


def _reported_wire_counts(routing: dict[str, Any], n_wires: int) -> tuple[int, int]:
    """Return the ``(logical, physical)`` widths the routing metadata reports.

    A route that allocates physical workspace widens the program to the device
    width, so the number of logical wires has to come from the metadata rather
    than from the routed program's own width. The reported physical width has to
    be the width the routed program actually has: the program is what the width
    describes, so a disagreement is stale or tampered metadata rather than a
    layout, and reading it as a layout would name slots that do not exist.
    """

    logical = routing.get("logical_wire_count", n_wires)
    if type(logical) is not int or not 0 < logical <= n_wires:
        raise ValueError(
            f"routing metadata reports logical_wire_count as {logical!r}, which "
            f"does not fit a {n_wires}-wire program"
        )
    slots = routing.get("physical_slot_count", n_wires)
    if slots != n_wires:
        raise ValueError(
            f"routing metadata reports physical_slot_count as {slots!r}, but the "
            f"routed program has {n_wires} physical wires"
        )
    return logical, slots


def _reported_layout(routing: dict[str, Any], key: str, n_wires: int) -> Layout:
    logical, slots = _reported_wire_counts(routing, n_wires)
    placement = routing.get(key)
    if not isinstance(placement, (list, tuple)):
        raise ValueError(f"routing metadata does not report {key}")
    if len(placement) != logical:
        raise ValueError(
            f"routing metadata reports {key} for {len(placement)} wires, but the "
            f"circuit has {logical}"
        )
    try:
        return Layout(tuple(placement), slots)
    except ValueError as error:
        raise ValueError(
            f"routing metadata reports an invalid {key}: {error}"
        ) from error


def final_layout(circuit_or_ir: Any) -> Layout:
    """Return the layout a routed program ends on, from its routing metadata.

    A route that allocates physical workspace widens the program to the device
    width, and the layout it reports places one logical wire per entry and leaves
    the rest of that width idle, so ``physical_slot_count`` is the device width and
    :attr:`Layout.has_idle_slots` is true.

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
    onto the physical wire its routing pass declared as its result slot, which is
    what makes a routed program's output layout the identity when no workspace was
    allocated and lets an observable or a measurement keep naming a logical wire.
    Those trailing SWAPs carry no routing information. A caller that reads results
    per physical wire and reorders them with :meth:`Layout.to_logical_order`
    computes the same thing for fewer SWAPs, and the reported output layout says
    which physical wire carries each logical wire.

    Observables and measurements move onto that layout, because a routed program's
    instructions are on physical wires and the output layout is no longer the
    identity to absorb the difference. A routed program names, in each output node,
    the physical slot the output layout gives the logical wire, so removing the
    restore renames those slots rather than the logical wires they carry.

    The result is a program to run against a local simulator or a provider that
    accepts per-physical-wire results, not a routing result to hand back as
    deployment evidence. ``flagquantum.deployment.routing_evidence`` requires a
    deployment routing plan to end restored and rejects this one by design, and the
    deployment entry points treat a program whose ``mapping_restored`` is not true
    as not routed and route it again, which spends back the SWAPs removed here.

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
    initial = _reported_layout(routing, "initial_logical_to_physical", ir.n_wires)
    placed_before_restore = initial.apply_swaps(forward_swaps)
    reported = _reported_layout(routing, "pre_restore_logical_to_physical", ir.n_wires)
    if placed_before_restore != reported:
        raise ValueError(
            "routing metadata does not agree with the SWAP evidence: replaying the "
            "program's non-restore SWAPs from the initial layout reaches "
            f"{placed_before_restore}, not the reported pre-restore layout {reported}"
        )
    # The routed program names, in each output node, the physical slot the restore
    # left that output on, which is the routed output layout. The reduced program
    # leaves the same outputs on the pre-restore layout, so a rename follows the
    # logical wire a slot carries rather than the slot number.
    output = _reported_layout(routing, "final_logical_to_physical", ir.n_wires)
    logical_of_slot = {
        physical: logical for logical, physical in enumerate(output.logical_to_physical)
    }

    def rename(wires: tuple[int, ...]) -> tuple[int, ...]:
        renamed: list[int] = []
        for wire in wires:
            logical = logical_of_slot.get(wire)
            if logical is None:
                raise ValueError(
                    f"routed output names physical wire {wire}, which carries no "
                    f"logical wire on the layout the route reports ({output})"
                )
            renamed.append(reported.logical_to_physical[logical])
        return tuple(renamed)

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
    placement = reported.logical_to_physical
    updated = dict(routing) | {
        "final_logical_to_physical": placement,
        "mapping_restored": False,
        "inserted_swap_count": inserted_swap_count,
        "planned_inserted_swap_count": inserted_swap_count,
        "removed_layout_restore_swap_count": len(restore_indexes),
    }
    # Every statement of the output layout has to move together, or the metadata
    # would carry two answers to one question.
    if "logical_result_physical_slots" in updated:
        updated["logical_result_physical_slots"] = placement
    if "final_physical_to_logical" in updated:
        updated["final_physical_to_logical"] = reported.physical_to_logical
    if updated.get("post_optimization_inserted_swap_count") is not None:
        updated["post_optimization_inserted_swap_count"] = inserted_swap_count
    metadata = dict(ir.metadata) | {"routing": updated}
    return replace(
        ir,
        instructions=tuple(kept),
        observables=tuple(
            replace(node, wires=rename(node.wires)) for node in ir.observables
        ),
        measurements=tuple(
            replace(node, wires=rename(node.wires)) for node in ir.measurements
        ),
        metadata=metadata,
    )
