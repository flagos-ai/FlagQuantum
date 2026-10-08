"""Physical locality of multi-wire instructions across every routing strategy.

A three-or-more-wire instruction can only stay in a routed program when the
device carries the couplings its operands actually interact over, because no
router in this package synthesizes one onto physical couplings and a strategy
cannot insert SWAPs for it. Every strategy therefore has to refuse the rest, and
the refusal has to be observable rather than a physically unrealizable program.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import flagquantum as fq
from flagquantum.compiler import compile as compile_circuit
from flagquantum.compiler.directed_topology import (
    DirectedCouplingMap,
    route_to_directed_topology,
)
from flagquantum.compiler.operand_semantics import _MULTI_WIRE_OPERAND_PAIRS
from flagquantum.compiler.routing import (
    ROUTING_STRATEGIES,
    CouplingMap,
    route_to_topology,
)
from flagquantum.compiler.topology_legalization import (
    TopologyLegalizationError,
    legalize_circuit_topology,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.core.target_capabilities import (
    CapabilityScope,
    TargetCapabilitySnapshot,
    TargetIdentity,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc)

# A device that carries the triangle 0-1-2 plus the tail 2-3-4. ``ccx`` on
# ``(0, 1, 2)`` conjugates on wire 2, which both remaining wires reach, so it is
# executable; ``(0, 2, 4)`` and the layouts below are not.
_TRIANGLE_AND_TAIL = CouplingMap(5, ((0, 1), (1, 2), (0, 2), (2, 3), (3, 4)))
_DIRECTED_TRIANGLE_AND_TAIL = DirectedCouplingMap(
    5,
    (
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (0, 2),
        (2, 0),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
    ),
)


def _snapshot() -> TargetCapabilitySnapshot:
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="multi-wire-locality-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="multi-wire-locality-environment",
        ),
        scope=CapabilityScope(device_ids=("topology:0",)),
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=(),
        evidence_refs=(),
    )


def _missing_couplings(
    coupling: CouplingMap, opcode: str, wires: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    """Return the couplings the device lacks for one multi-wire instruction."""

    return tuple(
        (wires[left], wires[right])
        for left, right in _MULTI_WIRE_OPERAND_PAIRS[opcode]
        if not coupling.has_edge(wires[left], wires[right])
    )


def _multi_wire_instructions(program) -> list[tuple[str, tuple[int, ...]]]:
    """Return every non-inserted instruction that touches three or more wires."""

    return [
        (item.name, item.wires)
        for item in program
        if item.metadata.get("routing_phase") is None and len(item.wires) >= 3
    ]


def test_every_core_multi_wire_opcode_has_a_connectivity_rule() -> None:
    """The locality table is total over Core's three-or-more-wire operands."""

    core_multi_wire = {
        schema.opcode for schema in OPERATOR_SCHEMAS.values() if schema.arity >= 3
    }
    assert core_multi_wire, "Core declares no multi-wire opcode; the check is vacuous"
    assert core_multi_wire <= set(_MULTI_WIRE_OPERAND_PAIRS), (
        "Core multi-wire opcodes without a routing connectivity rule: "
        f"{sorted(core_multi_wire - set(_MULTI_WIRE_OPERAND_PAIRS))}"
    )


def test_connectivity_rules_are_two_wire_interactions_of_their_opcode() -> None:
    """A rule names real operand pairs of the opcode it belongs to."""

    for opcode, pairs in _MULTI_WIRE_OPERAND_PAIRS.items():
        arity = OPERATOR_SCHEMAS[opcode].arity
        assert pairs, opcode
        assert all(
            left != right and 0 <= left < arity and 0 <= right < arity
            for left, right in pairs
        ), opcode


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_device_local_multi_wire_instruction_is_emitted_verbatim(strategy: str) -> None:
    """Non-vacuity anchor: a device-local ``ccx`` must still route unchanged."""

    source = fq.Circuit(5).x(0).ccx(0, 1, 2)
    routed = route_to_topology(source, _TRIANGLE_AND_TAIL, strategy=strategy)

    assert routed.metadata["routing"]["inserted_swap_count"] == 0
    emitted = _multi_wire_instructions(routed)
    assert emitted == [("ccx", (0, 1, 2))]
    assert _missing_couplings(_TRIANGLE_AND_TAIL, *emitted[0]) == ()


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_off_device_multi_wire_instruction_fails_closed(strategy: str) -> None:
    """A multi-wire instruction the device cannot host must be refused."""

    source = fq.Circuit(5).x(0).ccx(0, 2, 4)
    with pytest.raises(ValueError, match="multi-wire instruction"):
        route_to_topology(source, _TRIANGLE_AND_TAIL, strategy=strategy)


def test_a_multi_wire_instruction_needs_its_real_interactions_not_a_chain() -> None:
    """A chain of couplings is not enough: ``ccx`` conjugates on its target."""

    line = CouplingMap.line(3)
    assert line.edges == ((0, 1), (1, 2))
    # (0, 1, 2) is a chain, but ccx also conjugates wire 0 on wire 2, and the
    # line does not carry that coupling.
    with pytest.raises(ValueError, match=r"requires physical couplings \(\(0, 2\),\)"):
        route_to_topology(fq.Circuit(3).ccx(0, 1, 2), line, strategy="sabre")


def test_a_cswap_rule_is_about_its_control_not_its_operand_order() -> None:
    """``cswap`` exchanges under its first operand, so the controls must couple."""

    line = CouplingMap.line(3)
    # cswap(1, 0, 2) exchanges wires 0 and 2 under the control on wire 1, and
    # wire 1 is coupled to both, so the device can host it.
    routed = route_to_topology(fq.Circuit(3).cswap(1, 0, 2), line, strategy="sabre")
    assert _multi_wire_instructions(routed) == [("cswap", (1, 0, 2))]
    # cswap(0, 1, 2) exchanges under the control on wire 0, and wire 0 is not
    # coupled to wire 2, so the same device cannot host it.
    with pytest.raises(ValueError, match=r"requires physical couplings \(\(0, 2\),\)"):
        route_to_topology(fq.Circuit(3).cswap(0, 1, 2), line, strategy="sabre")


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_multi_wire_instruction_without_a_rule_fails_closed(strategy: str) -> None:
    """An instruction the table does not describe is refused, not assumed local."""

    source = CircuitIR(
        5,
        (
            Instruction("cx", (3, 4)),
            Instruction(
                "test_three_wire_channel", (0, 1, 2), metadata={"is_channel": True}
            ),
        ),
        dtype="complex128",
    )
    with pytest.raises(ValueError, match="no verified physical connectivity rule"):
        route_to_topology(source, _TRIANGLE_AND_TAIL, strategy=strategy)


@pytest.mark.parametrize("strategy", ("persistent_layout", "sabre", "sabre_layout"))
def test_layout_that_moves_a_multi_wire_instruction_off_device_fails_closed(
    strategy: str,
) -> None:
    """A layout choice must not silently relocate a wide gate off the device."""

    source = fq.Circuit(5).cx(1, 4).ccx(0, 1, 2)
    with pytest.raises(ValueError, match="multi-wire instruction"):
        route_to_topology(source, _TRIANGLE_AND_TAIL, strategy=strategy)


def test_directed_routing_refuses_off_device_multi_wire_instruction() -> None:
    source = fq.Circuit(5).x(0).ccx(0, 2, 4)
    with pytest.raises(ValueError, match="multi-wire instruction"):
        route_to_directed_topology(source.to_ir(), _DIRECTED_TRIANGLE_AND_TAIL)


def test_directed_routing_keeps_a_device_local_multi_wire_instruction() -> None:
    """Non-vacuity anchor for the directed path."""

    source = fq.Circuit(5).x(0).ccx(0, 1, 2)
    routed = route_to_directed_topology(source.to_ir(), _DIRECTED_TRIANGLE_AND_TAIL)
    assert _multi_wire_instructions(routed) == [("ccx", (0, 1, 2))]


def test_compiler_compile_refuses_off_device_multi_wire_instruction() -> None:
    """The user-facing entry must not emit a program the device cannot run."""

    source = fq.Circuit(5).x(0).ccx(0, 2, 4)
    with pytest.raises(ValueError, match="multi-wire instruction"):
        compile_circuit(source, coupling_map=_TRIANGLE_AND_TAIL)


def test_topology_legalization_refuses_off_device_multi_wire_instruction() -> None:
    source = fq.Circuit(5).x(0).ccx(0, 2, 4).to_ir()
    with pytest.raises(TopologyLegalizationError, match="multi-wire instruction"):
        legalize_circuit_topology(
            source,
            coupling_map=_TRIANGLE_AND_TAIL,
            snapshot=_snapshot(),
        )


def test_two_wire_routing_behaviour_is_unchanged() -> None:
    """The two-wire contract keeps inserting SWAPs instead of refusing."""

    source = fq.Circuit(5).cz(0, 4)
    routed = route_to_topology(source, CouplingMap.line(5), strategy="sabre")
    assert routed.metadata["routing"]["inserted_swap_count"] > 0
    assert _multi_wire_instructions(routed) == []
