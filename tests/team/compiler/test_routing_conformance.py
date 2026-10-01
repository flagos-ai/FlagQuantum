"""Conformance of the routing boundary against one shared contract.

``route_to_topology`` is the single entry point the compiler, the dynamic-circuit
runtime, the topology legalizer, and the deployment evidence builder use to reach
a router. The strategies behind it are separate implementations of the same task,
so replacing or adding one is acceptable only while it satisfies the same
postconditions. This module states those postconditions once and drives every
strategy through all of them, rather than asserting a different subset per
strategy.

The contract has five parts:

* the routed program keeps every two-wire operation on a device edge, emits no
  wire outside the circuit, leaves the final measurements alone, and computes the
  source program's state;
* the routing metadata restores the logical output layout;
* every source instruction survives exactly once and keeps its source order
  against every instruction it shares a wire with;
* the routing metadata carries the routing plan schema and counts what it emits;
* routing is deterministic, and reusing a device changes only its cache telemetry.

The order and determinism parts are what make a routed program usable by a
consumer that is not a statevector simulation: a mid-circuit operation only
observes the same classical history when the wire order is preserved, and a
cached plan is only reusable when the same input gives the same program.

The checks take the router as an argument so the last two tests can run an
independent router, and the compiler in front of it, through the identical
assertions. That is the routing boundary being proved by replacement rather than
by the existence of an interface.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import torch

import flagquantum as fq
import flagquantum.compiler.pipeline as compiler_pipeline
import flagquantum.compiler.topology_legalization as topology_legalization
import flagquantum.runtime.dynamic.routing as dynamic_routing
from flagquantum.compiler import CouplingMap, compile, route_to_topology
from flagquantum.compiler.topology_legalization import legalize_circuit_topology
from flagquantum.core.ir import (
    CircuitIR,
    Instruction,
    MeasurementNode,
    ensure_circuit_ir,
)
from flagquantum.core.target_capabilities import (
    CapabilityScope,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from flagquantum.dynamic import DynamicCircuit

pytestmark = pytest.mark.unit

ROUTING_STRATEGIES = (
    "restore_after_each_gate",
    "persistent_layout",
    "sabre",
    "sabre_layout",
)

# Each device is exercised at its full width and at a narrower width. The
# narrower widths are prefixes whose induced subgraph is still connected, so a
# two-wire operation between any two used wires stays routable.
#
# A device is stored as its wire count and edges rather than as a built
# ``CouplingMap`` because a ``CouplingMap`` carries the mutable shortest-path
# cache it was measured with. Every case builds its own device so that no test
# depends on the order in which the cases ran; the one test that wants a warm
# device asks for one explicitly.
_TOPOLOGIES = (
    ("line4", 4, CouplingMap.line(4).edges, (2, 4)),
    ("ring5", 5, CouplingMap.ring(5).edges, (3, 5)),
    ("line6", 6, CouplingMap.line(6).edges, (3, 6)),
    ("grid3x3", 9, CouplingMap.grid(3, 3).edges, (6, 9)),
    ("line9", 9, CouplingMap.line(9).edges, (5, 9)),
)

_ROTATIONS = ("rx", "ry", "rz")
_TWO_WIRE = ("cx", "cz")

_SCHEMA: dict[str, type] = {
    "schema": str,
    "strategy": str,
    "coupling_n_wires": int,
    "coupling_edges": tuple,
    "initial_logical_to_physical": tuple,
    "pre_restore_logical_to_physical": tuple,
    "final_logical_to_physical": tuple,
    "mapping_restored": bool,
    "direction_semantics": str,
    "topology_gate_count": int,
    "routed_gate_count": int,
    "inserted_swap_count": int,
    "planned_inserted_swap_count": int,
    "skipped_channel_count": int,
    "persistent_layout_supported": bool,
    "path_cache": dict,
}


@dataclass(frozen=True)
class _Case:
    """One seeded program on one device width, routed by one strategy."""

    topology: str
    strategy: str
    seed: int
    gate_count: int
    program: CircuitIR
    n_wires: int
    edges: tuple[tuple[int, int], ...]

    @property
    def label(self) -> str:
        return (
            f"{self.topology}/n={self.program.n_wires}/{self.strategy}"
            f"/seed={self.seed}/gates={self.gate_count}"
        )

    def device(self) -> CouplingMap:
        """Return a device that no earlier case has routed on."""

        return CouplingMap(self.n_wires, self.edges)


def _seeded_program(seed: int, n_wires: int, gate_count: int) -> CircuitIR:
    """Return a deterministic mixed single- and two-wire program.

    Two-wire operands are drawn from every wire pair, so most programs start with
    operations the device cannot execute directly.
    """

    rng = random.Random(seed)
    instructions: list[Instruction] = []
    for _ in range(gate_count):
        if rng.random() < 0.35:
            instructions.append(
                Instruction(
                    rng.choice(_ROTATIONS),
                    (rng.randrange(n_wires),),
                    params={"theta": rng.uniform(-3.0, 3.0)},
                )
            )
        else:
            left, right = rng.sample(range(n_wires), 2)
            instructions.append(Instruction(rng.choice(_TWO_WIRE), (left, right)))
    return CircuitIR(
        n_wires,
        tuple(instructions),
        dtype="complex128",
        measurements=(MeasurementNode("sample", tuple(range(n_wires)), shots=4),),
    )


def _cases() -> Iterator[_Case]:
    for topology, n_wires, edges, widths in _TOPOLOGIES:
        for program_wires in widths:
            for seed in range(3):
                for gate_count in (2 * program_wires, 6 * program_wires):
                    program = _seeded_program(seed, program_wires, gate_count)
                    for strategy in ROUTING_STRATEGIES:
                        yield _Case(
                            topology=topology,
                            strategy=strategy,
                            seed=seed,
                            gate_count=gate_count,
                            program=program,
                            n_wires=n_wires,
                            edges=edges,
                        )


def _state(program: CircuitIR) -> torch.Tensor:
    return fq.Circuit.from_ir(program).state()


def _is_inserted_swap(instruction: Instruction) -> bool:
    """Return whether the router inserted this SWAP rather than the program."""

    return instruction.metadata.get("routing_phase") is not None


def _emitted_source_indices(routed: CircuitIR) -> list[int | None]:
    """Return the source index each emitted non-SWAP operation carries.

    Routing remaps wires, so a routed operation cannot be identified by its name
    and wire pair: the stamp is the only thing that ties it back to the source
    program. An operation that arrives without one is reported as ``None``.
    """

    indices: list[int | None] = []
    for instruction in routed:
        if _is_inserted_swap(instruction):
            continue
        index = instruction.metadata.get("source_instruction_index")
        indices.append(index if isinstance(index, int) else None)
    return indices


def _source_positions(routed: CircuitIR) -> dict[int, int]:
    """Map each source instruction index onto its position in the routed program."""

    positions: dict[int, int] = {}
    for position, index in enumerate(_emitted_source_indices(routed)):
        if index is not None:
            positions.setdefault(index, position)
    return positions


# A router takes the program, the device, and the requested strategy. Every check
# below is written against this signature so the shipped entry point, a
# replacement router, and the compiler in front of either can be measured the
# same way.
_Router = Callable[[CircuitIR, CouplingMap, str], CircuitIR]


def _shipped_router(
    program: CircuitIR, device: CouplingMap, strategy: str
) -> CircuitIR:
    return route_to_topology(program, device, strategy=strategy)


def _device_legality_failures(router: _Router) -> list[str]:
    failures: list[str] = []
    for case in _cases():
        device = case.device()
        routed = router(case.program, device, case.strategy)
        if routed.n_wires != case.program.n_wires:
            failures.append(
                f"{case.label}: n_wires {routed.n_wires} != {case.program.n_wires}"
            )
        for instruction in routed:
            if any(wire >= case.program.n_wires for wire in instruction.wires):
                failures.append(f"{case.label}: wire outside the circuit {instruction}")
                break
            if len(instruction.wires) == 2 and not device.has_edge(*instruction.wires):
                failures.append(f"{case.label}: nonlocal operation {instruction}")
                break
        if routed.measurements != case.program.measurements:
            failures.append(f"{case.label}: the final measurements changed")
        if not torch.allclose(_state(routed), _state(case.program), atol=1e-6, rtol=0):
            failures.append(f"{case.label}: the routed state differs from the source")
    return failures


def _layout_restoration_failures(router: _Router) -> list[str]:
    failures: list[str] = []
    for case in _cases():
        routed = router(case.program, case.device(), case.strategy)
        routing = routed.metadata["routing"]
        expected = tuple(range(case.program.n_wires))
        initial = tuple(routing.get("initial_logical_to_physical", ()))
        final = tuple(routing.get("final_logical_to_physical", ()))
        if sorted(initial) != list(expected):
            failures.append(
                f"{case.label}: initial layout {initial} is not a permutation"
            )
        if final != expected:
            failures.append(f"{case.label}: final layout {final} is not the identity")
        if routing.get("mapping_restored") is not True:
            failures.append(
                f"{case.label}: mapping_restored is {routing.get('mapping_restored')!r}"
            )
        if routing.get("direction_semantics") != "logical_wire_order_preserved":
            failures.append(
                f"{case.label}: direction_semantics "
                f"{routing.get('direction_semantics')!r}"
            )
    return failures


def _source_order_failures(router: _Router) -> list[str]:
    """Report missing, duplicated, or wire-crossing source operations.

    Operations on disjoint wires may be reordered, because they commute and the
    SABRE strategies do reorder them. Operations that share a wire may not, which
    is what a consumer reading a mid-circuit result depends on.
    """

    failures: list[str] = []
    for case in _cases():
        routed = router(case.program, case.device(), case.strategy)
        indices = _emitted_source_indices(routed)
        unattributed = indices.count(None)
        if unattributed:
            failures.append(
                f"{case.label}: {unattributed} emitted operation(s) carry no "
                "source index"
            )
        emitted = sorted(index for index in indices if index is not None)
        expected = list(range(len(case.program.instructions)))
        if emitted != expected:
            failures.append(
                f"{case.label}: emitted source indices do not match the source "
                f"program exactly ({len(emitted)} emitted, {len(expected)} expected)"
            )
            continue
        positions = _source_positions(routed)
        for left_index in range(len(expected)):
            for right_index in range(left_index + 1, len(expected)):
                left = case.program.instructions[left_index]
                right = case.program.instructions[right_index]
                if not set(left.wires) & set(right.wires):
                    continue
                if positions[left_index] > positions[right_index]:
                    failures.append(
                        f"{case.label}: {left.name}{left.wires} and "
                        f"{right.name}{right.wires} share a wire but were emitted "
                        "in the opposite order"
                    )
    return failures


def _schema_failures(router: _Router) -> list[str]:
    failures: list[str] = []
    for case in _cases():
        routed = router(case.program, case.device(), case.strategy)
        routing = routed.metadata["routing"]
        for key, expected in _SCHEMA.items():
            if key not in routing:
                failures.append(f"{case.label}: routing metadata has no {key!r}")
            elif type(routing[key]) is not expected:
                failures.append(
                    f"{case.label}: {key!r} is {type(routing[key]).__name__}"
                )
        if routing.get("schema") != "flagquantum_routing_plan_v1":
            failures.append(f"{case.label}: schema {routing.get('schema')!r}")
        if routing.get("strategy") != case.strategy:
            failures.append(
                f"{case.label}: strategy echoed as {routing.get('strategy')!r}"
            )
        if routing.get("coupling_n_wires") != case.n_wires:
            failures.append(
                f"{case.label}: coupling_n_wires {routing.get('coupling_n_wires')!r}"
            )
        emitted = sum(1 for instruction in routed if _is_inserted_swap(instruction))
        if routing.get("inserted_swap_count") != emitted:
            failures.append(
                f"{case.label}: inserted_swap_count "
                f"{routing.get('inserted_swap_count')!r} counts {emitted} emitted SWAPs"
            )
    return failures


def _purity_failures(router: _Router) -> list[str]:
    """Two equally cold devices must produce byte-identical plans.

    This is what lets a plan be cached, compared between machines, or replayed
    from an evidence record.
    """

    failures: list[str] = []
    for case in _cases():
        first = router(case.program, case.device(), case.strategy)
        second = router(case.program, case.device(), case.strategy)
        if first.to_json() != second.to_json():
            failures.append(f"{case.label}: two runs produced different programs")
    return failures


def _warm_cache_failures(router: _Router) -> list[str]:
    """A warm device must not change the plan, only the reported cache counters.

    ``path_cache`` reports the counters a run observed, so a reused device
    legitimately reports hits the first run could not. Everything a consumer acts
    on has to stay identical. A strategy that resolves no path, or a program whose
    two-wire operations already sit on device edges, leaves the cache empty and
    then even the telemetry repeats; only a run that filled the cache is required
    to show the reuse.
    """

    warmed = 0
    failures: list[str] = []
    for case in _cases():
        device = case.device()
        first = router(case.program, device, case.strategy)
        second = router(case.program, device, case.strategy)
        if [(item.name, item.wires) for item in first] != [
            (item.name, item.wires) for item in second
        ]:
            failures.append(f"{case.label}: a warm device changed the program")
        first_metadata = dict(first.metadata["routing"])
        second_metadata = dict(second.metadata["routing"])
        differing = {
            key
            for key in first_metadata
            if first_metadata[key] != second_metadata.get(key)
        }
        if differing - {"path_cache"}:
            failures.append(
                f"{case.label}: a warm device changed "
                f"{sorted(differing - {'path_cache'})}"
            )
        if first_metadata["path_cache"]["after"]["size"]:
            warmed += 1
            if (
                second_metadata["path_cache"]["before"]
                != first_metadata["path_cache"]["after"]
            ):
                failures.append(
                    f"{case.label}: the second run did not observe the first "
                    f"run's cache: {second_metadata['path_cache']['before']}"
                )
    if not warmed:
        failures.append(
            "no case filled the device path cache, so warm reuse is untested"
        )
    return failures


def test_the_matrix_covers_every_strategy_and_a_narrower_program() -> None:
    """Guard the coverage the other checks rely on.

    A device wider than the program is the shape that reached an unguarded
    device-wide neighbour lookup in an earlier revision, so the matrix must keep
    exercising it for every strategy.
    """

    cases = list(_cases())
    assert {case.strategy for case in cases} == set(ROUTING_STRATEGIES)
    narrower = [case for case in cases if case.n_wires > case.program.n_wires]
    assert {case.strategy for case in narrower} == set(ROUTING_STRATEGIES)
    assert len(cases) == 240


def test_routed_program_is_device_legal_and_preserves_the_source_state() -> None:
    assert not _device_legality_failures(_shipped_router), "\n".join(
        _device_legality_failures(_shipped_router)
    )


def test_routing_metadata_restores_the_logical_output_layout() -> None:
    assert not _layout_restoration_failures(_shipped_router), "\n".join(
        _layout_restoration_failures(_shipped_router)
    )


def test_routing_keeps_every_source_instruction_once_in_dependency_order() -> None:
    assert not _source_order_failures(_shipped_router), "\n".join(
        _source_order_failures(_shipped_router)
    )


def test_routing_metadata_carries_the_routing_plan_schema() -> None:
    assert not _schema_failures(_shipped_router), "\n".join(
        _schema_failures(_shipped_router)
    )


def test_a_routing_run_is_a_pure_function_of_its_request() -> None:
    assert not _purity_failures(_shipped_router), "\n".join(
        _purity_failures(_shipped_router)
    )


def test_reusing_a_device_changes_only_its_cache_telemetry() -> None:
    assert not _warm_cache_failures(_shipped_router), "\n".join(
        _warm_cache_failures(_shipped_router)
    )


def _replacement_swap(
    left: int,
    right: int,
    source: Instruction,
    source_index: int,
    strategy: str,
    phase: str,
) -> Instruction:
    return Instruction(
        name="swap",
        wires=(left, right),
        metadata={
            "routed_from": source.name,
            "routing_strategy": strategy,
            "routing_phase": phase,
            "logical_wires": source.wires,
            "source_instruction_index": source_index,
        },
    )


def _replacement_router(
    program: CircuitIR, device: CouplingMap, strategy: str
) -> CircuitIR:
    """Route by walking the right operand towards the left one.

    This is the replacement half of the boundary proof, and it is deliberately a
    different implementation from every shipped strategy: where
    ``restore_after_each_gate`` moves the left operand to the far end of the path
    and executes there, this moves the right operand to the near end and
    executes there. The two therefore choose different physical edges and
    different SWAP positions while owing the same postconditions.
    """

    ir = ensure_circuit_ir(program)
    cache_before = device.path_cache_info()
    emitted: list[Instruction] = []
    topology_gate_count = 0
    inserted_swap_count = 0
    skipped_channel_count = 0
    for source_index, instruction in enumerate(ir):
        wires = instruction.wires
        path: tuple[int, ...] | None = None
        if len(wires) == 2 and instruction.metadata.get("is_channel"):
            skipped_channel_count += 1
        elif len(wires) == 2:
            path = device.shortest_path(*wires)
            topology_gate_count += 1
            for step in range(len(path) - 1, 1, -1):
                emitted.append(
                    _replacement_swap(
                        path[step - 1],
                        path[step],
                        instruction,
                        source_index,
                        strategy,
                        "forward",
                    )
                )
                inserted_swap_count += 1
            wires = (path[0], path[1])
        emitted.append(
            Instruction(
                name=instruction.name,
                wires=wires,
                params=instruction.params,
                matrix=instruction.matrix,
                metadata=dict(instruction.metadata)
                | {
                    "routed": wires != instruction.wires,
                    "logical_wires": instruction.wires,
                    "routing_strategy": strategy,
                    "source_instruction_index": source_index,
                },
            )
        )
        if path is not None:
            for step in range(2, len(path)):
                emitted.append(
                    _replacement_swap(
                        path[step - 1],
                        path[step],
                        instruction,
                        source_index,
                        strategy,
                        "gate_restore",
                    )
                )
                inserted_swap_count += 1
    cache_after = device.path_cache_info()
    metadata = dict(ir.metadata)
    metadata["routing"] = {
        "schema": "flagquantum_routing_plan_v1",
        "strategy": strategy,
        "coupling_n_wires": device.n_wires,
        "coupling_edges": device.edges,
        "initial_logical_to_physical": tuple(range(ir.n_wires)),
        "pre_restore_logical_to_physical": tuple(range(ir.n_wires)),
        "final_logical_to_physical": tuple(range(ir.n_wires)),
        "mapping_restored": True,
        "direction_semantics": "logical_wire_order_preserved",
        "topology_gate_count": topology_gate_count,
        "routed_gate_count": topology_gate_count,
        "inserted_swap_count": inserted_swap_count,
        "planned_inserted_swap_count": inserted_swap_count,
        "skipped_channel_count": skipped_channel_count,
        "persistent_layout_supported": True,
        "path_cache": {
            "before": cache_before,
            "after": cache_after,
            "delta": {
                key: cache_after[key] - cache_before[key]
                for key in ("hits", "misses", "evictions")
            },
        },
    }
    return replace(ir, instructions=tuple(emitted), metadata=metadata)


def test_a_replacement_router_satisfies_the_same_contract() -> None:
    """An independent router passes every check the shipped strategies pass."""

    failures = (
        _device_legality_failures(_replacement_router)
        + _layout_restoration_failures(_replacement_router)
        + _source_order_failures(_replacement_router)
        + _schema_failures(_replacement_router)
        + _purity_failures(_replacement_router)
        + _warm_cache_failures(_replacement_router)
    )
    assert not failures, "\n".join(failures)


def test_the_replacement_router_really_routes_differently() -> None:
    """Guard the replacement test against a router that equals the shipped one.

    A replacement that produced the same programs as a shipped strategy would
    prove nothing about the boundary, so the two are required to disagree. They
    agree exactly when no two-wire operation needed routing, and disagree on
    every case that did.
    """

    differing = 0
    routed_cases = 0
    for case in _cases():
        if case.strategy != "restore_after_each_gate":
            continue
        shipped = _shipped_router(case.program, case.device(), case.strategy)
        replacement = _replacement_router(case.program, case.device(), case.strategy)
        if not any(_is_inserted_swap(item) for item in shipped):
            continue
        routed_cases += 1
        if shipped.to_json() != replacement.to_json():
            differing += 1
    assert routed_cases > 0
    assert differing == routed_cases


def _monkeypatch_consumers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every routing consumer reach the replacement router."""

    def replacement_route(
        circuit_or_ir: Any,
        coupling_map: Any,
        *,
        strategy: str = "restore_after_each_gate",
    ) -> CircuitIR:
        program = ensure_circuit_ir(circuit_or_ir)
        device = (
            coupling_map
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(program.n_wires, coupling_map)
        )
        return _replacement_router(program, device, strategy)

    monkeypatch.setattr(compiler_pipeline, "route_to_topology", replacement_route)
    monkeypatch.setattr(topology_legalization, "route_to_topology", replacement_route)
    monkeypatch.setattr(dynamic_routing, "route_to_topology", replacement_route)


def test_the_compiler_accepts_a_replacement_router_without_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the whole matrix through ``compile`` over the replacement router."""

    _monkeypatch_consumers(monkeypatch)

    def compiled(program: CircuitIR, device: CouplingMap, strategy: str) -> CircuitIR:
        return compile(
            program,
            coupling_map=device,
            routing_strategy=strategy,
            optimize=False,
        )

    failures = (
        _device_legality_failures(compiled)
        + _layout_restoration_failures(compiled)
        + _source_order_failures(compiled)
        + _schema_failures(compiled)
    )
    assert not failures, "\n".join(failures)


def _snapshot() -> TargetCapabilitySnapshot:
    now = datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc)
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="routing-conformance-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="routing-conformance-environment",
        ),
        scope=CapabilityScope(device_ids=("topology:0",)),
        captured_at=now.isoformat(),
        valid_until=(now + timedelta(hours=1)).isoformat(),
        facts=(),
        evidence_refs=(),
    )


def test_the_topology_legalizer_accepts_a_replacement_router_without_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legalizer's own postcondition must hold over the replacement router."""

    _monkeypatch_consumers(monkeypatch)
    case = next(
        case
        for case in _cases()
        if case.strategy == "restore_after_each_gate"
        and case.topology == "grid3x3"
        and case.gate_count == 6 * case.program.n_wires
    )
    result = legalize_circuit_topology(
        case.program,
        coupling_map=CouplingMap(case.n_wires, case.edges),
        snapshot=_snapshot(),
        strategy=case.strategy,
    )
    assert result.final_logical_to_physical == tuple(range(case.program.n_wires))
    assert result.inserted_swap_count > 0
    assert result.routed_instruction_count > result.source_instruction_count


def test_the_dynamic_runtime_accepts_a_replacement_router_without_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dynamic-circuit router must accept the replacement unchanged."""

    _monkeypatch_consumers(monkeypatch)
    circuit = DynamicCircuit(4)
    circuit.h(0)
    circuit.cx(0, 3)
    routed = dynamic_routing.route_dynamic_circuit(circuit, CouplingMap.line(4))

    routed_ir = routed.to_ir()
    routing = routed_ir.metadata["routing"]
    assert routing["strategy"] == "restore_after_each_gate"
    assert routing["mapping_restored"] is True
    assert routing["final_logical_to_physical"] == (0, 1, 2, 3)
    assert routing["dynamic_boundary_mapping_policy"] == "identity_restored_per_gate"
    device = CouplingMap.line(4)
    for instruction in routed_ir:
        if len(instruction.wires) == 2:
            assert device.has_edge(*instruction.wires), instruction
