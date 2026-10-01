"""Initial-placement planning for a program on a coupling map.

Routing has two halves: deciding which physical wires a program's logical wires
start on, and inserting SWAPs when a two-wire operation lands on wires the device
does not join. This module is the first half. It answers "where should this
program start" before any SWAP is considered.

A placement here is ``logical_to_physical[logical]``, one physical slot per
logical wire, with distinct slots drawn from ``range(coupling_map.n_wires)``. It
is deliberately *not* a :class:`~flagquantum.compiler.layout.Layout`: a device
wider than the program leaves slots idle, and ``Layout`` requires a complete
permutation of the program's own wires.

The distinction carries the whole contract. Every routed program in this package
restores its logical output order, so a placement that starts off the identity
survives that restore only on a
:class:`~flagquantum.compiler.directed_topology.DirectedCouplingMap`, whose
physical workspace owns the idle slots and is cleaned by the inverse routing
SWAPs. A plain :class:`~flagquantum.compiler.routing.CouplingMap` refuses
``initial_layout`` outright, because routing one there may not reach through a
wire the program does not own. A caller therefore pairs a placement from this
module with ``route_to_directed_topology(..., initial_layout=placement)`` or with
``legalize_circuit_topology(..., initial_layout=placement)`` on a directed
coupling map.

``plan_trivial_layout`` places logical wire ``i`` on physical wire ``i``.
``plan_dense_layout`` places the program on a densely connected window of
physical wires. On a device no wider than the program there is exactly one window
of the required width, so both functions return the identity, and
``plan_dense_layout`` is safe to ask unconditionally.

The window search is the one Qiskit's ``DenseLayout`` runs before it reorders its
result: walk breadth first from every physical wire in turn, keep the window with
the most device connections inside it, and let the lowest-index winning window
win a tie. Two measured caveats:

* The search is a heuristic over breadth-first prefixes and not an exact
  optimum. Checking every connected subset of the required width over 55
  device/width combinations found the same answer in 45 and a better window in
  10, each time by exactly one internal connection. The 4-wire windows of a 4x4
  grid are the smallest case: the search returns ``(0, 1, 2, 4)`` with 3 internal
  connections, where ``(5, 6, 9, 10)`` has 4.
* Qiskit then applies a reverse-Cuthill-McKee permutation that SciPy computes.
  This module returns the window in ascending physical order instead, because
  ascending order is what collapses to the identity when the window is the whole
  device, and because the permutation measured no better here: over 20 seeded
  programs on each of eight devices it cost 3946 inserted SWAPs against 3842 for
  ascending order, losing on three devices, tying two, and winning three.
"""

from __future__ import annotations

from typing import Any

from ..core.ir import ensure_circuit_ir
from ..errors import CompilationError
from .directed_topology import DirectedCouplingMap
from .routing import CouplingMap

__all__ = (
    "LayoutPlanningError",
    "plan_dense_layout",
    "plan_trivial_layout",
)


class LayoutPlanningError(CompilationError):
    """Raised when no initial placement satisfies the device."""


def _validated_device(
    coupling_map: object,
    n_wires: int,
) -> CouplingMap | DirectedCouplingMap:
    """Return the coupling map, refusing one the program cannot fit on."""

    if not isinstance(coupling_map, (CouplingMap, DirectedCouplingMap)):
        raise TypeError("coupling_map must be a Compiler coupling map")
    if coupling_map.n_wires < n_wires:
        raise LayoutPlanningError(
            f"coupling map has {coupling_map.n_wires} physical wires but the "
            f"program has {n_wires} logical wires; a placement needs at least as "
            "many physical wires as logical wires"
        )
    return coupling_map


def _weak_adjacency(
    coupling_map: CouplingMap | DirectedCouplingMap,
) -> tuple[tuple[int, ...], ...]:
    """Return ascending undirected neighbours for every physical wire.

    A directed coupling map is read as weak connectivity, because the placement
    decides which wires a program may occupy and not which way a two-wire
    operation may point.
    """

    neighbours: list[set[int]] = [set() for _ in range(coupling_map.n_wires)]
    for left, right in coupling_map.edges:
        neighbours[left].add(right)
        neighbours[right].add(left)
    return tuple(tuple(sorted(entry)) for entry in neighbours)


def _breadth_first_window(
    neighbours: tuple[tuple[int, ...], ...],
    start: int,
    width: int,
) -> tuple[int, ...] | None:
    """Return the first ``width`` wires a breadth-first walk from ``start`` reaches.

    The walk completes a level before it starts the next one, so the window is a
    connected set of wires rather than a path. ``None`` means ``start``'s
    connected component holds fewer than ``width`` wires.
    """

    window: list[int] = []
    seen = {start}
    level = [start]
    while level:
        following: list[int] = []
        for wire in level:
            window.append(wire)
            if len(window) == width:
                return tuple(window)
            for peer in neighbours[wire]:
                if peer not in seen:
                    seen.add(peer)
                    following.append(peer)
        level = following
    return None


def _densest_window(
    neighbours: tuple[tuple[int, ...], ...],
    width: int,
) -> tuple[int, ...]:
    """Return the breadth-first window with the most connections inside it."""

    best: tuple[int, ...] | None = None
    best_connections = -1
    for start in range(len(neighbours)):
        window = _breadth_first_window(neighbours, start, width)
        if window is None:
            continue
        inside = set(window)
        connections = sum(
            1 for wire in window for peer in neighbours[wire] if peer in inside
        )
        if connections > best_connections:
            best, best_connections = window, connections
    if best is None:
        raise LayoutPlanningError(
            f"the device has no connected window of {width} physical wires; "
            "every connected component is narrower than the program"
        )
    return best


def plan_trivial_layout(
    circuit_or_ir: Any,
    coupling_map: CouplingMap | DirectedCouplingMap,
) -> tuple[int, ...]:
    """Place logical wire ``i`` on physical wire ``i``.

    This is the placement that assumes the device's lowest-numbered wires are as
    good as any other set, which is true only on a device whose wires are
    interchangeable. It is the placement every router in this package already
    starts from, and the one a device exactly as wide as the program is stuck
    with.

    Raises:
        TypeError: if ``coupling_map`` is not a Compiler coupling map.
        LayoutPlanningError: if the device has fewer wires than the program.
    """

    ir = ensure_circuit_ir(circuit_or_ir)
    _validated_device(coupling_map, ir.n_wires)
    return tuple(range(ir.n_wires))


def plan_dense_layout(
    circuit_or_ir: Any,
    coupling_map: CouplingMap | DirectedCouplingMap,
) -> tuple[int, ...]:
    """Place the program on a densely connected window of the device.

    The placement starts every logical wire on the window and returns the window
    in ascending physical order, so logical wire ``i`` lands on the ``i``-th
    lowest wire of the window. A device no wider than the program has one window
    of the required width, and this returns the identity there, which is why the
    function is safe to ask before knowing whether the device is wider.

    The window is the best one a breadth-first walk from each physical wire
    finds, which is not always the exact optimum; see the module docstring for
    the measured shortfall.

    The result is a placement, not a
    :class:`~flagquantum.compiler.layout.Layout`; see the module docstring for the
    coupling map a caller must route on to keep it.

    Raises:
        TypeError: if ``coupling_map`` is not a Compiler coupling map.
        LayoutPlanningError: if the device has fewer wires than the program, or if
            every connected component of the device is narrower than the program.
    """

    ir = ensure_circuit_ir(circuit_or_ir)
    device = _validated_device(coupling_map, ir.n_wires)
    if device.n_wires == ir.n_wires:
        return tuple(range(ir.n_wires))
    window = _densest_window(_weak_adjacency(device), ir.n_wires)
    return tuple(sorted(window))
