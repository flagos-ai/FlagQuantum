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
  against every instruction it conflicts with, on a wire or on a classical bit;
* the routing metadata carries the routing plan schema and counts what it emits;
* routing is deterministic, and reusing a device changes only its cache telemetry.

The order and determinism parts are what make a routed program usable by a
consumer that is not a statevector simulation: a mid-circuit operation only
observes the same classical history when the wire order is preserved, and a
cached plan is only reusable when the same input gives the same program.

A wire is not the only channel through which order is observable. A measurement
writes a classical bit that no wire records, and a conditional reads that bit, so
a conditional and the measurement it depends on are ordered even when they touch
disjoint wires. Those cases are stated separately from the seeded wire-only grid,
because they need a program whose outcome depends on the classical channel rather
than a device on which an operation is nonlocal.

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
from flagquantum.compiler.layout import Layout
from flagquantum.compiler.sabre import plan_sabre_swaps
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
from flagquantum.runtime.dynamic import run_dynamic

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


def _classical_accesses(instruction: Instruction) -> tuple[int | None, frozenset[int]]:
    """Return the classical bit an instruction writes and the bits it reads.

    This is the test's own reading of the metadata spellings the cases build. It
    is deliberately not the planner's reader: a contract checked with the
    implementation's own helper only proves that the helper is self-consistent.
    """

    written = instruction.metadata.get("classical_bit")
    raw_clauses: Any = instruction.metadata.get("condition_clauses")
    if raw_clauses is None:
        raw_clauses = (instruction.metadata.get("conditions") or (),)
    read = {int(bit) for clause in raw_clauses or () for bit, _value in clause}
    return (None if written is None else int(written)), frozenset(read)


def _conflicts(left: Instruction, right: Instruction) -> str | None:
    """Return how two source instructions are ordered, or ``None`` if free.

    Two operations conflict when they touch a shared wire, or when they touch a
    shared classical bit and at least one of them writes it. Two readers of one
    bit conflict with nothing: neither can observe the other.
    """

    shared_wires = set(left.wires) & set(right.wires)
    if shared_wires:
        return f"share wire(s) {sorted(shared_wires)}"
    left_write, left_read = _classical_accesses(left)
    right_write, right_read = _classical_accesses(right)
    left_bits = set(left_read) | ({left_write} if left_write is not None else set())
    right_bits = set(right_read) | ({right_write} if right_write is not None else set())
    writers = {bit for bit in (left_write, right_write) if bit is not None}
    shared_bits = writers & left_bits & right_bits
    if shared_bits:
        return f"share classical bit(s) {sorted(shared_bits)}"
    return None


def _source_order_failures(router: _Router) -> list[str]:
    """Report missing, duplicated, or reordered conflicting source operations.

    Operations that conflict on nothing may be reordered, because they commute and
    the SABRE strategies do reorder them. Operations that share a wire, or that
    share a classical bit one of them writes, may not: that order is what a
    consumer reading a mid-circuit result depends on.
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
                conflict = _conflicts(left, right)
                if conflict is None:
                    continue
                if positions[left_index] > positions[right_index]:
                    failures.append(
                        f"{case.label}: {left.name}{left.wires} and "
                        f"{right.name}{right.wires} {conflict} but were emitted "
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


# The seeded grid above contains no classical state: its programs carry gates and
# final measurements, so every ordering it exercises is a wire ordering. The cases
# below are hand-written instead, because the property they state is about the
# classical channel rather than about a device. Each one is a program whose
# observable outcome changes when a conditional crosses the measurement that sets
# its condition bit, so routing it in the wrong order produces either a different
# result or a program the runtime refuses to execute at all.


def _measurement(qubit: int, bit: int) -> Instruction:
    return Instruction(
        "measure",
        (qubit,),
        metadata={"is_dynamic": True, "classical_bit": bit},
    )


def _conditional(
    name: str, wires: tuple[int, ...], conditions: tuple[tuple[int, int], ...]
) -> Instruction:
    return Instruction(
        name, wires, metadata={"is_dynamic": True, "conditions": conditions}
    )


def _classical_programs() -> dict[str, CircuitIR]:
    """Return the classical-order cases, keyed by what makes each one distinct."""

    return {
        # The conditional is two wires away from the qubit it is conditioned on, so
        # no wire orders it against its measurement on any strategy.
        "conditional_two_wires_from_its_measurement": CircuitIR(
            3,
            (
                Instruction("x", (0,)),
                Instruction("x", (1,)),
                _measurement(1, 0),
                _conditional("cx", (0, 2), ((0, 1),)),
            ),
            dtype="complex128",
        ),
        # The same shape on one wire, where the strategies disagree about the plan.
        "conditional_one_wire_from_its_measurement": CircuitIR(
            3,
            (
                Instruction("x", (0,)),
                Instruction("x", (1,)),
                _measurement(1, 0),
                _conditional("x", (2,), ((0, 1),)),
            ),
            dtype="complex128",
        ),
        # The conditional sits between two writes of the bit it reads, so it owes
        # order to the write it observes and to the write that replaces the value.
        "conditional_between_two_writes_of_its_bit": CircuitIR(
            3,
            (
                Instruction("x", (0,)),
                _measurement(0, 0),
                _conditional("x", (1,), ((0, 1),)),
                _measurement(1, 0),
            ),
            dtype="complex128",
        ),
        # Two writes of one bit with no read between them, so the second must keep
        # its source place even though nothing observes the value it replaces.
        "two_writes_of_one_bit": CircuitIR(
            3,
            (
                Instruction("x", (0,)),
                _measurement(0, 0),
                _measurement(1, 0),
                _conditional("x", (2,), ((0, 1),)),
            ),
            dtype="complex128",
        ),
        # One conditional reading two bits, each written by a different measurement.
        "conditional_reads_two_bits": CircuitIR(
            3,
            (
                Instruction("x", (0,)),
                Instruction("x", (1,)),
                _measurement(0, 0),
                _measurement(1, 1),
                _conditional("x", (2,), ((0, 1), (1, 1))),
            ),
            dtype="complex128",
        ),
        # Nothing reads these bits, so no classical order constrains this program
        # and a router that added one anyway would pay for swaps it does not owe.
        "measurements_nothing_reads": CircuitIR(
            3,
            (
                Instruction("cx", (0, 2)),
                _measurement(0, 0),
                _measurement(1, 1),
                _measurement(2, 2),
            ),
            dtype="complex128",
        ),
    }


def _classical_order_failures(router: _Router) -> list[str]:
    """Report conditional/measurement pairs a router emitted out of source order."""

    failures: list[str] = []
    for label, program in _classical_programs().items():
        for strategy in ROUTING_STRATEGIES:
            routed = router(program, CouplingMap(3, ((0, 1), (1, 2))), strategy)
            positions = _source_positions(routed)
            for left_index in range(len(program.instructions)):
                for right_index in range(left_index + 1, len(program.instructions)):
                    left = program.instructions[left_index]
                    right = program.instructions[right_index]
                    conflict = _conflicts(left, right)
                    if conflict is None:
                        continue
                    if left_index not in positions or right_index not in positions:
                        failures.append(
                            f"{label}/{strategy}: {left.name}{left.wires} or "
                            f"{right.name}{right.wires} is missing from the output"
                        )
                    elif positions[left_index] > positions[right_index]:
                        failures.append(
                            f"{label}/{strategy}: {left.name}{left.wires} and "
                            f"{right.name}{right.wires} {conflict} but were emitted "
                            "in the opposite order"
                        )
    return failures


def _executed_bits(program: CircuitIR, shots: int = 64, seed: int = 11):
    """Return the shot-by-bit result the dynamic runtime produces for a program."""

    result = run_dynamic(DynamicCircuit.from_ir(program), shots=shots, seed=seed)
    samples = result.final_samples
    if not isinstance(samples, torch.Tensor):
        samples = torch.as_tensor(samples)
    return samples.reshape(shots, -1)


def _classical_execution_failures(router: _Router) -> list[str]:
    """Report routed programs the runtime refuses, or that read out differently.

    Execution is the oracle the order contract is a proxy for: a router that
    preserves every conflicting pair in source order cannot produce a program the
    runtime rejects, and one that does not is caught here rather than by a user.
    """

    failures: list[str] = []
    for label, program in _classical_programs().items():
        try:
            baseline = _executed_bits(program)
        except Exception as error:
            failures.append(f"{label}: the source program itself failed: {error!r}")
            continue
        for strategy in ROUTING_STRATEGIES:
            routed = router(program, CouplingMap(3, ((0, 1), (1, 2))), strategy)
            try:
                observed = _executed_bits(routed)
            except Exception as error:
                failures.append(
                    f"{label}/{strategy}: the routed program failed: {error}"
                )
                continue
            if not torch.equal(observed, baseline):
                failures.append(
                    f"{label}/{strategy}: the routed program reads out "
                    f"{observed.tolist()[0]} where the source reads out "
                    f"{baseline.tolist()[0]}"
                )
    return failures


def test_routing_keeps_a_conditional_behind_the_measurement_it_reads() -> None:
    failures = _classical_order_failures(_shipped_router)
    assert not failures, "\n".join(failures)


def test_a_routed_conditional_program_runs_and_reads_out_like_its_source() -> None:
    failures = _classical_execution_failures(_shipped_router)
    assert not failures, "\n".join(failures)


def test_the_sabre_planner_refuses_conditions_it_cannot_order_on() -> None:
    """A condition stated both ways is not a condition any layer can read.

    The planner reads the condition metadata itself rather than through the
    runtime's condition vocabulary, so it has to refuse the same malformed shapes
    that vocabulary refuses rather than plan against an ambiguous read set.
    """

    program = CircuitIR(
        2,
        (
            _measurement(0, 0),
            Instruction(
                "x",
                (1,),
                metadata={
                    "is_dynamic": True,
                    "conditions": ((0, 1),),
                    "condition_clauses": (((0, 1),),),
                },
            ),
        ),
        dtype="complex128",
    )
    with pytest.raises(ValueError, match="cannot define both"):
        plan_sabre_swaps(program, CouplingMap.line(2))


def test_the_sabre_planner_leaves_a_read_of_an_unwritten_bit_alone() -> None:
    """A conditional on a bit nothing writes keeps the order the source gave it.

    This is not a licence to accept the program: reading a bit no measurement
    writes is a caller error and the runtime refuses it. The planner must not
    refuse it here, because the layout search plans the *reversed* program, where
    the writes necessarily follow the reads. Refusing would turn the layout search
    into a failure on every program that carries a conditional.
    """

    program = CircuitIR(
        2,
        (
            Instruction("x", (0,)),
            _conditional("x", (1,), ((0, 1),)),
        ),
        dtype="complex128",
    )
    plan = plan_sabre_swaps(program, CouplingMap.line(2))
    assert plan.sequence == (0, 1)


def test_the_classical_matrix_covers_every_strategy() -> None:
    """Guard the coverage the two checks above depend on.

    Five of the six programs read a classical bit, and every one of those must
    state at least one conflicting pair, or the checks above pass without ever
    having an order to preserve.
    """

    programs = _classical_programs()
    assert len(programs) == 6
    reading = {
        label: program
        for label, program in programs.items()
        if any(_classical_accesses(item)[1] for item in program.instructions)
    }
    assert len(reading) == 5
    for label, program in reading.items():
        conflicts = [
            (left_index, right_index)
            for left_index in range(len(program.instructions))
            for right_index in range(left_index + 1, len(program.instructions))
            if _conflicts(
                program.instructions[left_index], program.instructions[right_index]
            )
        ]
        assert conflicts, f"{label} declares a read but no conflicting pair"
    assert not _classical_order_failures(_replacement_router)


# ``compile`` optimizes the routed program a second time, and those passes are
# allowed to delete an inserted SWAP: a forward SWAP and its matching restore SWAP
# on the same physical wires cancel, as does a final restore another one undid.
# They must not, however, leave a plan whose SWAP evidence no longer describes the
# program. No optimization pass knows it is editing a routing artifact, so the
# property has to be asserted from the outside, over the plan a consumer reads.


def _optimizing_compiler(
    program: CircuitIR, device: CouplingMap, strategy: str
) -> CircuitIR:
    """The shipped compile entry point with the post-routing optimization on."""

    return compile(
        program, coupling_map=device, routing_strategy=strategy, optimize=True
    )


def _optimization_evidence_failures(router: _Router) -> tuple[list[str], int]:
    """Report plans the post-routing optimization left unable to describe itself.

    Also returns how many cases lost an inserted SWAP, so the caller can require
    that the check ran against the removals it exists for rather than passing
    because nothing was ever removed.
    """

    failures: list[str] = []
    removed_in = 0
    for case in _cases():
        optimized = router(case.program, case.device(), case.strategy)
        routing = optimized.metadata.get("routing")
        if not isinstance(routing, dict):
            failures.append(f"{case.label}: the optimized program lost its plan")
            continue
        surviving = [
            (str(instruction.metadata.get("routing_phase")), instruction.wires)
            for instruction in optimized
            if _is_inserted_swap(instruction)
        ]
        planned = routing.get("inserted_swap_count")
        retained = routing.get("post_optimization_inserted_swap_count")
        if retained != len(surviving):
            failures.append(
                f"{case.label}: the plan reports {retained!r} surviving inserted "
                f"SWAPs but the program carries {len(surviving)}"
            )
        if routing.get("post_optimization_instruction_count") != len(optimized):
            failures.append(
                f"{case.label}: the plan reports "
                f"{routing.get('post_optimization_instruction_count')!r} instructions "
                f"but the program carries {len(optimized)}"
            )
        if not isinstance(planned, int) or not isinstance(retained, int):
            continue
        if retained > planned:
            failures.append(
                f"{case.label}: optimization raised the inserted SWAP count "
                f"{planned} -> {retained}"
            )
        elif retained < planned:
            removed_in += 1
        try:
            initial = Layout(tuple(routing["initial_logical_to_physical"]))
            placed = initial.apply_swaps(
                wires for phase, wires in surviving if phase != "final_restore"
            )
            reported = Layout(tuple(routing["pre_restore_logical_to_physical"]))
            ended = initial.apply_swaps(wires for _, wires in surviving)
            final = Layout(tuple(routing["final_logical_to_physical"]))
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"{case.label}: the plan cannot be replayed: {error}")
            continue
        if placed != reported:
            failures.append(
                f"{case.label}: the surviving non-restore SWAPs reach {placed} "
                f"rather than the reported pre-restore layout {reported}"
            )
        if ended != final or routing.get("mapping_restored") is not True:
            failures.append(
                f"{case.label}: the surviving SWAPs reach {ended} rather than the "
                f"reported output layout {final}"
            )
    return failures, removed_in


def test_the_optimized_compile_keeps_the_routing_plan_replayable() -> None:
    """The post-routing optimization may remove a routing SWAP, but not the plan."""

    evidence_failures, removed_in = _optimization_evidence_failures(
        _optimizing_compiler
    )
    failures = _device_legality_failures(_optimizing_compiler) + evidence_failures
    assert not failures, "\n".join(failures)
    assert removed_in > 0, (
        "no case lost an inserted SWAP to the post-routing optimization, so this "
        "check never ran against the removals it exists for"
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
        "coupling_n_wires": device.n_qubits,
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
