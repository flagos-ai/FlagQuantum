"""Conformance of initial-placement planning against one shared contract.

Routing has two halves: where a program starts, and which SWAPs that costs.
Before this pass the second half was reachable and the first was not, so every
router started on physical wires ``0..n-1`` whatever the device looked like. On
a device whose lowest wires are a poor or unusable set that is not a placement,
it is an assumption, and the routers failed closed with "No coupling path"
against a device that had a perfectly good connected window further up.

The contract these tests pin is that a placement is a total injection of the
program's logical wires into the device's physical wires, that the dense pass
finds the best connected window, and that a placement only reaches the compiler
through a directed coupling map, whose physical workspace owns the idle slots.
"""

from __future__ import annotations

import inspect
import random
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import (
    CouplingMap,
    plan_dense_layout,
    plan_trivial_layout,
    route_to_topology,
)
from flagquantum.compiler.directed_topology import (
    DirectedCouplingMap,
    route_to_directed_topology,
)
from flagquantum.compiler.layout_planning import LayoutPlanningError
from flagquantum.compiler.topology_legalization import (
    TopologyLegalizationError,
    legalize_circuit_topology,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.target_capabilities import (
    CapabilityScope,
    TargetCapabilitySnapshot,
    TargetIdentity,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

pytestmark = pytest.mark.unit

MANIFEST = Path(__file__).resolve().parents[3] / "capability-maturity.toml"

# The lowest four wires are a bare two-link chain, so wire 3 is reachable from
# nothing below it, while wires 4..7 are a complete graph. A four-wire program
# has no legal placement on the low half and an ideal one on the high half.
_COMPLETE_WINDOW_DEVICE = CouplingMap(
    8, [(0, 1), (1, 2), *combinations((4, 5, 6, 7), 2)]
)
_COMPLETE_WINDOW = (4, 5, 6, 7)

# The same shape with the low half still connected, so the trivial placement is
# legal as well as bad and the two placements can be compared on one program.
_CONNECTED_LOW_HALF_DEVICE = CouplingMap(
    8, [(0, 1), (1, 2), (2, 3), *combinations((4, 5, 6, 7), 2)]
)

# A device with a connected window wider than four wires, so the injection
# contract is exercised on a placement that cannot be the identity.
_COMPLETE_WINDOW_DEVICE_11 = CouplingMap(
    11, [(0, 1), (1, 2), *combinations((5, 6, 7, 8, 9, 10), 2)]
)


def _snapshot() -> TargetCapabilitySnapshot:
    now = datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc)
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="layout-planning-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="layout-planning-environment",
        ),
        scope=CapabilityScope(device_ids=("placement:0",)),
        captured_at=now.isoformat(),
        valid_until=(now + timedelta(hours=1)).isoformat(),
        facts=(),
        evidence_refs=(),
    )


def _weakly_directed(coupling_map: CouplingMap) -> DirectedCouplingMap:
    """Return the coupling map with both directions of every edge.

    Placement planning decides which wires a program may occupy, not which way a
    two-wire operation may point, so a placement is exercised on a directed
    device that is symmetric.
    """

    return DirectedCouplingMap(
        coupling_map.n_wires,
        [
            *coupling_map.edges,
            *((right, left) for left, right in coupling_map.edges),
        ],
    )


def _complete_program(n_wires: int) -> CircuitIR:
    """Return the program that interacts every pair of its wires once."""

    circuit = fq.Circuit(n_wires)
    for left, right in combinations(range(n_wires), 2):
        circuit = circuit.cx(left, right)
    return circuit.to_ir()


def _random_program(seed: int, n_wires: int, gate_count: int) -> CircuitIR:
    rng = random.Random(seed)
    instructions = []
    for _ in range(gate_count):
        left, right = rng.sample(range(n_wires), 2)
        instructions.append(Instruction("cx", (left, right)))
        instructions.append(Instruction("ry", (left,), params={"theta": 0.37}))
    return CircuitIR(n_wires, tuple(instructions))


def _swap_count(program: CircuitIR, coupling_map: CouplingMap, layout: tuple[int, ...]):
    routed = route_to_directed_topology(
        program,
        _weakly_directed(coupling_map),
        initial_layout=layout,
    )
    return routed.metadata["routing"]["inserted_swap_count"]


def _placed_branch(
    state: torch.Tensor,
    layout: tuple[int, ...],
    n_physical: int,
) -> torch.Tensor:
    """Return the state the placed wires carry, reordered to logical wire order.

    The workspace router cleans up after itself, so every wire outside the
    placement is back to ``|0>``. Setting those wires to ``0`` therefore selects
    the branch the program ran in, and the remaining axes are reordered because
    the state's basis order is the physical wire order, not the layout's.
    """

    tensor = state.reshape(-1, *(2,) * n_physical)
    idle = [wire for wire in range(n_physical) if wire not in layout]
    for wire in sorted(idle, reverse=True):
        tensor = tensor.select(1 + wire, 0)
    ascending = sorted(layout)
    order = [ascending.index(slot) for slot in layout]
    return tensor.permute(0, *(1 + position for position in order))


def test_trivial_layout_places_every_logical_wire_on_its_own_index() -> None:
    program = _complete_program(4)

    assert plan_trivial_layout(program, _COMPLETE_WINDOW_DEVICE) == (0, 1, 2, 3)
    assert plan_trivial_layout(program, CouplingMap.grid(4, 4)) == (0, 1, 2, 3)

    # The trivial placement is the identity on any device, so it never consults
    # connectivity and never depends on the program's contents.
    assert plan_trivial_layout(
        _random_program(seed=3, n_wires=4, gate_count=8), _COMPLETE_WINDOW_DEVICE
    ) == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "coupling_map",
    (CouplingMap.line(5), CouplingMap.ring(5), CouplingMap.grid(2, 3)),
    ids=("line", "ring", "grid"),
)
def test_dense_layout_collapses_to_the_identity_on_a_device_no_wider_than_the_program(
    coupling_map: CouplingMap,
) -> None:
    program = _random_program(seed=5, n_wires=5, gate_count=20)

    # One window of the required width exists, so both passes answer it and the
    # dense pass is safe to ask before knowing whether the device is wider.
    assert plan_dense_layout(program, coupling_map) == tuple(range(5))
    assert plan_trivial_layout(program, coupling_map) == tuple(range(5))


def test_dense_layout_finds_the_connected_window_the_trivial_placement_misses() -> None:
    program = _complete_program(4)

    dense = plan_dense_layout(program, _COMPLETE_WINDOW_DEVICE)

    assert dense == _COMPLETE_WINDOW
    # The window is returned in ascending physical order and not reordered by
    # graph structure, so the result collapses to the identity when the window
    # is the whole device.
    assert list(dense) == sorted(dense)
    assert plan_trivial_layout(program, _COMPLETE_WINDOW_DEVICE) == (0, 1, 2, 3)


def _internal_connections(device: CouplingMap, window: tuple[int, ...]) -> int:
    inside = set(window)
    return (
        sum(1 for wire in window for peer in device.neighbors(wire) if peer in inside)
        // 2
    )


def test_dense_layout_prefers_the_window_with_more_internal_connections() -> None:
    # The complete window higher up the device has more internal connections
    # than the chain below it, so a program that fits in either must be sent up.
    program = _complete_program(4)
    device = _CONNECTED_LOW_HALF_DEVICE

    dense = plan_dense_layout(program, device)
    trivial = plan_trivial_layout(program, device)

    assert dense == _COMPLETE_WINDOW
    assert trivial == (0, 1, 2, 3)
    assert _internal_connections(device, dense) > _internal_connections(device, trivial)
    assert _swap_count(program, device, dense) < _swap_count(program, device, trivial)


def test_dense_layout_records_that_the_search_is_a_heuristic_not_an_optimum() -> None:
    # A 4-wire program on a 4x4 grid is the smallest case where the breadth-first
    # search is provably not optimal: it returns a 3-connection window while a
    # 4-connection one exists. Pinned here so the documented shortfall stays
    # attached to the behaviour it describes rather than to a claim of exactness.
    program = _complete_program(4)
    device = CouplingMap.grid(4, 4)
    window = plan_dense_layout(program, device)

    assert window == (0, 1, 2, 4)
    assert sorted(window) == list(window)
    assert _internal_connections(device, window) == 3
    assert _internal_connections(device, (5, 6, 9, 10)) == 4
    # Both windows hold the program, so this is a quality gap and not a legal
    # placement the search failed to reach.
    assert _swap_count(program, device, window) > 0
    assert _swap_count(program, device, (5, 6, 9, 10)) > 0


@pytest.mark.parametrize(
    ("coupling_map", "n_wires"),
    (
        (_COMPLETE_WINDOW_DEVICE, 3),
        (_COMPLETE_WINDOW_DEVICE, 4),
        (_COMPLETE_WINDOW_DEVICE_11, 6),
        (CouplingMap.grid(4, 4), 4),
        (CouplingMap.grid(4, 4), 9),
        (CouplingMap.grid(5, 5), 12),
        (CouplingMap.line(9), 4),
        (CouplingMap.ring(9), 4),
    ),
)
def test_a_placement_injects_every_logical_wire_into_the_device(
    coupling_map: CouplingMap, n_wires: int
) -> None:
    program = _random_program(seed=n_wires, n_wires=n_wires, gate_count=3 * n_wires)

    for layout in (
        plan_trivial_layout(program, coupling_map),
        plan_dense_layout(program, coupling_map),
    ):
        assert isinstance(layout, tuple)
        assert len(layout) == n_wires
        assert len(set(layout)) == n_wires
        assert all(
            isinstance(slot, int) and 0 <= slot < coupling_map.n_wires
            for slot in layout
        )


def test_a_placement_depends_on_the_device_and_not_on_the_program_contents() -> None:
    device = _COMPLETE_WINDOW_DEVICE
    first = _random_program(seed=11, n_wires=4, gate_count=12)
    second = _random_program(seed=12, n_wires=4, gate_count=4)

    assert plan_dense_layout(first, device) == plan_dense_layout(second, device)
    assert plan_dense_layout(first, device) == plan_dense_layout(second, device)
    assert plan_trivial_layout(first, device) == plan_trivial_layout(second, device)


def test_dense_layout_removes_the_swaps_the_trivial_placement_pays() -> None:
    program = _complete_program(4)
    device = _CONNECTED_LOW_HALF_DEVICE

    trivial = plan_trivial_layout(program, device)
    dense = plan_dense_layout(program, device)

    assert trivial == (0, 1, 2, 3)
    assert dense == _COMPLETE_WINDOW
    # The program is a complete graph, so the complete window holds it with no
    # SWAP at all and the chain below pays for every pair it cannot reach.
    assert _swap_count(program, device, dense) == 0
    assert _swap_count(program, device, trivial) == 12


def test_dense_layout_beats_the_trivial_placement_across_wider_devices() -> None:
    devices = (
        (_CONNECTED_LOW_HALF_DEVICE, 4),
        (
            CouplingMap(
                8, [(0, 1), (1, 2), (2, 3), (3, 4), *combinations((4, 5, 6, 7), 2)]
            ),
            6,
        ),
        (CouplingMap.grid(4, 4), 5),
        (CouplingMap.grid(4, 4), 8),
        (CouplingMap.grid(5, 5), 9),
    )

    trivial_total = 0
    dense_total = 0
    losses = 0
    ties = 0
    for index, (device, n_wires) in enumerate(devices):
        device_trivial = 0
        device_dense = 0
        for seed in range(4):
            program = _random_program(
                seed=100 * index + seed, n_wires=n_wires, gate_count=3 * n_wires
            )
            trivial = _swap_count(program, device, plan_trivial_layout(program, device))
            dense = _swap_count(program, device, plan_dense_layout(program, device))
            if dense > trivial:
                losses += 1
            elif dense == trivial:
                ties += 1
            device_trivial += trivial
            device_dense += dense
        # The pass maximises device connectivity inside the window and never
        # estimates routing cost, so an individual program can prefer a sparser
        # window. What it must not do is cost more on a device as a whole.
        assert device_dense <= device_trivial, (device.edges, n_wires)
        trivial_total += device_trivial
        dense_total += device_dense

    # Choosing the window is only worth its search on a device wider than the
    # program; on an equal-width device the two placements coincide, so this is
    # the whole of the saving the pass can claim over the placement every router
    # already used.
    assert dense_total < trivial_total
    assert dense_total > 0, "the fixture must contain programs that need SWAPs"
    # Recorded so the lossy cases stay visible rather than averaged away.
    assert (losses, ties) == (2, 1)


def test_dense_layout_routes_a_program_the_trivial_placement_cannot_route() -> None:
    program = _complete_program(4)
    device = _COMPLETE_WINDOW_DEVICE

    # Every plain strategy routes only over the wires the program owns, so on
    # this device it fails closed on a pair the low half cannot connect even
    # though the device can hold the whole program without a single SWAP.
    for strategy in (
        "restore_after_each_gate",
        "persistent_layout",
        "sabre",
        "sabre_layout",
    ):
        with pytest.raises(ValueError, match="No coupling path"):
            route_to_topology(program, device, strategy=strategy)

    assert _swap_count(program, device, plan_dense_layout(program, device)) == 0


def test_a_placement_preserves_the_program_state_on_the_placed_wires() -> None:
    program = _random_program(seed=21, n_wires=4, gate_count=16)
    device = _CONNECTED_LOW_HALF_DEVICE
    source = fq.Circuit.from_ir(program).state()

    for layout in (
        plan_trivial_layout(program, device),
        plan_dense_layout(program, device),
    ):
        routed = route_to_directed_topology(
            program,
            _weakly_directed(device),
            initial_layout=layout,
        )
        branch = _placed_branch(
            fq.Circuit.from_ir(routed).state(), layout, device.n_wires
        )

        # The idle wires must be in |0> for the projection to mean anything, so
        # the branch has to carry the whole state.
        assert torch.allclose(branch.norm(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(branch.reshape(source.shape), source, atol=1e-6)


def test_a_placement_reaches_the_compiler_only_through_a_directed_coupling_map() -> (
    None
):
    program = _complete_program(4)

    # A plain coupling map routes only over wires the program owns, so it can
    # neither hold a placement that leaves idle slots nor advertise a parameter
    # it would have to reinterpret.
    assert "initial_layout" not in inspect.signature(route_to_topology).parameters
    with pytest.raises(
        TopologyLegalizationError, match="requires a DirectedCouplingMap"
    ):
        legalize_circuit_topology(
            program,
            coupling_map=_CONNECTED_LOW_HALF_DEVICE,
            snapshot=_snapshot(),
            initial_layout=(4, 5, 6, 7),
        )


def test_a_placement_fails_closed_on_a_device_the_program_cannot_occupy() -> None:
    program = _complete_program(4)

    with pytest.raises(LayoutPlanningError, match="needs at least as many"):
        plan_trivial_layout(program, CouplingMap.line(3))
    with pytest.raises(LayoutPlanningError, match="needs at least as many"):
        plan_dense_layout(program, CouplingMap.line(3))

    # A device wide enough overall but split into components narrower than the
    # program cannot hold it: no window keeps every wire mutually reachable.
    split = CouplingMap(8, [(0, 1), (2, 3), (4, 5), (6, 7)])
    with pytest.raises(LayoutPlanningError, match="no connected window"):
        plan_dense_layout(program, split)

    with pytest.raises(TypeError, match="coupling map"):
        plan_dense_layout(program, [(0, 1)])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="coupling map"):
        plan_trivial_layout(program, [(0, 1)])  # type: ignore[arg-type]


def test_the_manifest_names_the_placement_entry_points() -> None:
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    limitations = data["capabilities"]["program_compilation"]["limitations"]
    public_apis = data["capabilities"]["program_compilation"]["public_apis"]

    for name in ("plan_trivial_layout", "plan_dense_layout"):
        assert (
            name in limitations
        ), f"capability-maturity.toml does not state what {name} does"
        assert (
            f"flagquantum.compiler.{name}" in public_apis
        ), f"capability-maturity.toml does not list flagquantum.compiler.{name}"

    # The window search is a heuristic, and the capability statement is the one
    # place a reader looks to find that out. A claim of exactness here would be
    # the kind of stale prose #325 had to correct.
    lowered = limitations.lower()
    for phrase in ("heuristic", "exact optimum", "internal connection"):
        assert phrase in lowered, (
            "capability-maturity.toml does not state that the placement search is "
            f"a heuristic; {phrase!r} is missing"
        )
