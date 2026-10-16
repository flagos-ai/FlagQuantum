"""Logical-to-physical qubit layouts for compiled programs.

A routing pass moves logical qubits onto physical qubits. :class:`Layout` is that
movement as a value, :func:`apply_layout` places a program on one, and
:func:`final_layout` reads the one a routed program ended on.

Routing in this package always restores before returning: every logical qubit is
walked back onto the physical slot the route declared as its result, which is its
own qubit when no workspace was allocated. That is what makes a routed program's
output layout the identity and keeps a measurement naming a logical qubit correct.
The restore is an appended SWAP phase and it is the largest single part of a routed
program's SWAP count, so :func:`remove_layout_restore` removes it for a caller that
reads results per physical qubit and reorders them with
:meth:`Layout.to_logical_order`.

A route that allocates physical workspace widens the program to the device width,
so a placement there leaves slots idle. :class:`Layout` carries that case too:
``physical_slot_count`` is that width and ``physical_to_logical`` reports ``None``
for a slot that holds no logical qubit. ``None`` for an idle slot is the occupancy
convention :mod:`flagquantum.core._compilation_evidence_v3` already records its own
allocated routing evidence with. A route that never allocates stays on the qubits
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
    """An assignment of logical qubits to physical qubits.

    ``logical_to_physical[logical]`` is the physical qubit that carries
    ``logical``, and no two logical qubits share one. A routing pass that places a
    circuit onto exactly as many physical qubits as the circuit has logical qubits
    produces a permutation of ``range(n_qubits)``, and that is the default: leave
    ``physical_slot_count`` out and the assignment must cover every qubit.

    A device wider than the program is the other case. There the program owns
    fewer qubits than the device has slots, so the assignment injects the logical
    qubits into a subset of the slots and ``physical_slot_count`` is the device
    width. Slots that hold no logical qubit are idle, ``physical_to_logical``
    reports ``None`` for them, and :func:`apply_layout` refuses such a layout
    because a program cannot name a qubit outside its own width.

    A physical SWAP is not an exchange of two entries of the assignment. It
    exchanges the qubits the two *positions* currently hold, so
    :meth:`apply_swaps` is the operation to use; indexing the tuple by a physical
    qubit name silently reorders the wrong qubits. A SWAP against an idle slot is
    how an allocated route moves a logical qubit into its workspace.

    Examples:
        Read a two-qubit program's per-qubit outcomes in logical-qubit order after
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
                f"layout must be a tuple of integer physical qubits, got {placement!r}"
            )
        if not placement:
            raise ValueError("layout must place at least one logical qubit")
        if physical_slot_count is None:
            slots = len(placement)
            if sorted(placement) != list(range(slots)):
                raise ValueError(
                    "layout must be a permutation of the circuit qubits, got "
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
                    f"layout places {len(placement)} logical qubits but has only "
                    f"{slots} physical slots"
                )
            if len(set(placement)) != len(placement):
                raise ValueError(
                    "layout must place every logical qubit on its own physical qubit, "
                    f"got {placement!r}"
                )
            if any(item < 0 or item >= slots for item in placement):
                raise ValueError(
                    f"layout places a qubit outside its {slots} physical slots, got "
                    f"{placement!r}"
                )
        object.__setattr__(self, "logical_to_physical", placement)
        object.__setattr__(self, "physical_slot_count", slots)

    def __str__(self) -> str:
        if self.has_idle_slots:
            return f"Layout({self.logical_to_physical}, {self.physical_slot_count})"
        return f"Layout({self.logical_to_physical})"

    @property
    def n_qubits(self) -> int:
        """Return the number of logical qubits this layout places."""
        return len(self.logical_to_physical)

    @property
    def has_idle_slots(self) -> bool:
        """Return whether some physical slot holds no logical qubit."""
        return self.n_qubits < self.physical_slot_count

    @property
    def physical_to_logical(self) -> tuple[int | None, ...]:
        """Return the occupancy; index it by physical slot, ``None`` if idle."""
        inverse: list[int | None] = [None] * self.physical_slot_count
        for logical, physical in enumerate(self.logical_to_physical):
            inverse[physical] = logical
        return tuple(inverse)

    def physical_of(self, logical: int) -> int:
        """Return the physical qubit that carries ``logical``."""
        if not 0 <= logical < self.n_qubits:
            raise ValueError(f"logical qubit {logical} is outside the layout")
        return self.logical_to_physical[logical]

    def logical_of(self, physical: int) -> int:
        """Return the logical qubit that ``physical`` carries.

        Raises:
            ValueError: if ``physical`` is outside the device or holds no qubit.
        """
        if not 0 <= physical < self.physical_slot_count:
            raise ValueError(f"physical qubit {physical} is outside the layout")
        occupied = self.physical_to_logical[physical]
        if occupied is None:
            raise ValueError(f"physical qubit {physical} holds no logical qubit")
        return occupied

    def map_qubits(self, qubits: Iterable[int]) -> tuple[int, ...]:
        """Return ``qubits`` read as logical qubits, expressed as physical qubits."""
        return tuple(self.physical_of(qubit) for qubit in qubits)

    def apply_swaps(self, swaps: Iterable[tuple[int, int]]) -> Layout:
        """Return the layout left behind by SWAPs on physical qubit pairs.

        Raises:
            ValueError: if a SWAP names a slot outside the device.
        """

        occupancy = list(self.physical_to_logical)
        for left, right in swaps:
            if not 0 <= left < self.physical_slot_count or not (
                0 <= right < self.physical_slot_count
            ):
                raise ValueError(
                    f"SWAP qubit pair ({left}, {right}) is outside a "
                    f"{self.physical_slot_count}-qubit layout"
                )
            if left == right:
                continue
            occupancy[left], occupancy[right] = occupancy[right], occupancy[left]
        placement = [0] * self.n_qubits
        for physical, logical in enumerate(occupancy):
            if logical is not None:
                placement[logical] = physical
        return Layout(placement, self.physical_slot_count)

    def to_logical_order(self, values: Sequence[Any]) -> tuple[Any, ...]:
        """Return one value per logical qubit, from a sequence indexed by physical qubit.

        A program routed without its layout restore reports results indexed by
        physical qubit; ``values`` is one entry per physical slot of the device and
        the result is the same entries read in logical-qubit order. An allocated
        route therefore drops the entries of its idle slots.

        Raises:
            ValueError: if ``values`` does not have one entry per physical slot.
        """

        ordered = tuple(values)
        if len(ordered) != self.physical_slot_count:
            raise ValueError(
                "to_logical_order needs one value per physical qubit; got "
                f"{len(ordered)} for {self.physical_slot_count} qubits"
            )
        return tuple(ordered[physical] for physical in self.logical_to_physical)


def apply_layout(circuit_or_ir: Any, layout: Layout) -> CircuitIR:
    """Place a program on ``layout``, relabelling every qubit it names.

    Instructions, observables, and measurements move together, so the result
    computes the same thing under different qubit labels and keeps ``n_qubits``.
    This is the relabelling a caller needs to hand a program to a pass that
    expects it to start on a specific layout.

    The applied assignment is recorded under ``metadata["layout"]``, because it is
    the only record of the qubit labels the program arrived with.

    Raises:
        TypeError: if ``layout`` is not a :class:`Layout`.
        ValueError: if the layout does not cover exactly the circuit's qubits, or if
            it leaves physical slots idle, which a program cannot name.
    """

    ir = ensure_circuit_ir(circuit_or_ir)
    if not isinstance(layout, Layout):
        raise TypeError("layout must be a Layout")
    if layout.n_qubits != ir.n_wires or layout.has_idle_slots:
        raise ValueError(
            f"layout places {layout.n_qubits} logical qubits on "
            f"{layout.physical_slot_count} physical slots but the circuit has "
            f"{ir.n_wires} logical qubits; a layout must cover exactly the circuit"
        )
    metadata = dict(ir.metadata)
    metadata["layout"] = {
        "n_wires": layout.n_qubits,
        "logical_to_physical": layout.logical_to_physical,
        "physical_to_logical": layout.physical_to_logical,
    }
    return replace(
        ir,
        instructions=tuple(
            Instruction(
                instruction.name,
                layout.map_qubits(instruction.wires),
                params=instruction.params,
                matrix=instruction.matrix,
                metadata=instruction.metadata,
            )
            for instruction in ir.instructions
        ),
        observables=tuple(
            replace(node, wires=layout.map_qubits(node.wires))
            for node in ir.observables
        ),
        measurements=tuple(
            replace(node, wires=layout.map_qubits(node.wires))
            for node in ir.measurements
        ),
        metadata=metadata,
    )


def _reported_qubit_counts(routing: dict[str, Any], n_qubits: int) -> tuple[int, int]:
    """Return the ``(logical, physical)`` widths the routing metadata reports.

    A route that allocates physical workspace widens the program to the device
    width, so the number of logical qubits has to come from the metadata rather
    than from the routed program's own width. The reported physical width has to
    be the width the routed program actually has: the program is what the width
    describes, so a disagreement is stale or tampered metadata rather than a
    layout, and reading it as a layout would name slots that do not exist.
    """

    logical = routing.get("logical_wire_count", n_qubits)
    if type(logical) is not int or not 0 < logical <= n_qubits:
        raise ValueError(
            f"routing metadata reports logical_qubit_count as {logical!r}, which "
            f"does not fit a {n_qubits}-qubit program"
        )
    slots = routing.get("physical_slot_count", n_qubits)
    if slots != n_qubits:
        raise ValueError(
            f"routing metadata reports physical_slot_count as {slots!r}, but the "
            f"routed program has {n_qubits} physical qubits"
        )
    return logical, slots


def _reported_layout(routing: dict[str, Any], key: str, n_qubits: int) -> Layout:
    logical, slots = _reported_qubit_counts(routing, n_qubits)
    placement = routing.get(key)
    if not isinstance(placement, (list, tuple)):
        raise ValueError(f"routing metadata does not report {key}")
    if len(placement) != logical:
        raise ValueError(
            f"routing metadata reports {key} for {len(placement)} qubits, but the "
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
    width, and the layout it reports places one logical qubit per entry and leaves
    the rest of that width idle, so ``physical_slot_count`` is the device width and
    :attr:`Layout.has_idle_slots` is true.

    Raises:
        ValueError: if the program has no routing metadata, or if the layout that
            metadata reports is not a complete assignment of the circuit's qubits.
    """

    ir = ensure_circuit_ir(circuit_or_ir)
    routing = ir.metadata.get("routing")
    if not isinstance(routing, dict):
        raise ValueError("program has no routing metadata; route it first")
    return _reported_layout(routing, "final_logical_to_physical", ir.n_wires)


def _routing_swap_phases(ir: CircuitIR) -> list[tuple[int, str, tuple[int, int]]]:
    """Return ``(index, phase, qubits)`` for every routed SWAP, validating each one."""

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

    Routing appends a ``final_restore`` phase that walks every logical qubit back
    onto the physical qubit its routing pass declared as its result slot, which is
    what makes a routed program's output layout the identity when no workspace was
    allocated and lets an observable or a measurement keep naming a logical qubit.
    Those trailing SWAPs carry no routing information. A caller that reads results
    per physical qubit and reorders them with :meth:`Layout.to_logical_order`
    computes the same thing for fewer SWAPs, and the reported output layout says
    which physical qubit carries each logical qubit.

    Observables and measurements move onto that layout, because a routed program's
    instructions are on physical qubits and the output layout is no longer the
    identity to absorb the difference. A routed program names, in each output node,
    the physical slot the output layout gives the logical qubit, so removing the
    restore renames those slots rather than the logical qubits they carry.

    The result is a program to run against a local simulator or a provider that
    accepts per-physical-qubit results, not a routing result to hand back as
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
        qubits for index, _, qubits in marked if index not in restore_indexes
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
    # logical qubit a slot carries rather than the slot number.
    output = _reported_layout(routing, "final_logical_to_physical", ir.n_wires)
    logical_of_slot = {
        physical: logical for logical, physical in enumerate(output.logical_to_physical)
    }

    def rename(qubits: tuple[int, ...]) -> tuple[int, ...]:
        renamed: list[int] = []
        for qubit in qubits:
            logical = logical_of_slot.get(qubit)
            if logical is None:
                raise ValueError(
                    f"routed output names physical qubit {qubit}, which carries no "
                    f"logical qubit on the layout the route reports ({output})"
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
