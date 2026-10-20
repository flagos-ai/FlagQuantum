"""Route a program onto a device by what its links cost, not by how far apart they are.

This is the Compiler domain's weighted-routing golden path. It answers, in order:
what a device states about a link, what a route costs on such a device, what the
plan reports about that cost, that the routed program still computes what the
source computed, and which strategies are refused because they would not read the
cost at all.

The device is a line of four qubits with one chord, so the two qubits this program
has to join are two hops apart whichever way a route goes. A hop count therefore
has nothing to say about which route to take; the calibration does. One route
crosses the chord, whose links are dirty, and the other does not.

Run it with::

    python -m examples.weighted_routing
"""

from __future__ import annotations

import torch

import flagquantum as fq
from flagquantum.compiler import CouplingMap, LinkErrorRates
from flagquantum.compiler.routing import (
    LINK_ERROR_ROUTING_STRATEGIES,
    estimate_routing_cost,
    route_to_topology,
    select_routing_strategy,
)
from flagquantum.core.ir import CircuitIR

#: The CPU statevector path. `complex128` with no fallback makes the comparison
#: below exact, so the state check cannot pass by taking a different backend.
OPTIONS = fq.ExecutionOptions(
    mode="statevector",
    backend="pytorch",
    device="cpu",
    precision="complex128",
    allow_backend_fallback=False,
)

#: The two-qubit error probability of each link, as a calibration would report it.
#: The chord 0-3 and the link 0-1 are the dirty ones; the far side of the line is
#: twenty times cleaner.
CALIBRATION = {
    (0, 1): 0.02,
    (0, 3): 0.02,
    (1, 2): 0.001,
    (2, 3): 0.001,
}

#: A device that states only which qubits it couples.
CONNECTIVITY_ONLY = CouplingMap(4, ((0, 1), (1, 2), (2, 3), (0, 3)))

#: The same couplings with the measured error of each link beside them.
CALIBRATED = LinkErrorRates(CONNECTIVITY_ONLY, CALIBRATION)


def _opcodes(program: CircuitIR) -> list[tuple[str, tuple[int, ...]]]:
    return [(item.name, item.wires) for item in program.instructions]


def main() -> None:
    circuit = fq.Circuit(n_qubits=4, dtype=torch.complex128).h(1).cx(1, 3)
    reference = circuit.state()

    # A connectivity-only device answers one question: which qubits are joined by
    # a hop. The two routes between qubits 1 and 3 are both two hops, so the
    # router takes the one a breadth-first search meets first and has no way to
    # prefer the other.
    hop_routed = route_to_topology(circuit, CONNECTIVITY_ONLY)
    hop_plan = hop_routed.metadata["routing"]

    # A calibrated device answers a second question: what each hop costs. The
    # route through qubit 0 crosses two links priced at 0.02; the route through
    # qubit 2 crosses two links priced at 0.001, so it is the cheaper route even
    # though it is exactly as long.
    routed = route_to_topology(circuit, CALIBRATED)
    plan = routed.metadata["routing"]

    assert _opcodes(hop_routed) == [
        ("h", (1,)),
        ("swap", (1, 0)),
        ("cx", (0, 3)),
        ("swap", (1, 0)),
    ]
    assert _opcodes(routed) == [
        ("h", (1,)),
        ("swap", (1, 2)),
        ("cx", (2, 3)),
        ("swap", (1, 2)),
    ]

    # Every count the router reported before the calibration existed is unchanged:
    # this is not a program that got smaller, it is the same program on cheaper
    # links. That is why a SWAP count could not have chosen it.
    assert plan["inserted_swap_count"] == hop_plan["inserted_swap_count"] == 2
    assert plan["topology_gate_count"] == hop_plan["topology_gate_count"]

    # What changes is the fault total: the two SWAPs and the gate, each on the link
    # it lands on. The hop-count route would have paid fifteen times as much.
    assert hop_plan["expected_link_faults"] is None
    assert plan["expected_link_faults"] == 0.003
    avoided = CALIBRATED.route_error(CONNECTIVITY_ONLY.shortest_path(1, 3))
    assert round(avoided, 6) == 0.04

    # The estimate is the plan: the price a caller is quoted is the price of the
    # program it gets, taken before the program is emitted.
    estimate = estimate_routing_cost(circuit, CALIBRATED)
    assert estimate.expected_link_faults == plan["expected_link_faults"]

    # The routed program still computes what the source computed, on the exact
    # CPU statevector path with fallback disabled.
    torch.testing.assert_close(
        fq.run(routed, options=OPTIONS).state, reference, atol=1e-12, rtol=0
    )
    compiled = circuit.compile(coupling_map=CALIBRATED)
    torch.testing.assert_close(
        fq.run(compiled, options=OPTIONS).state, reference, atol=1e-12, rtol=0
    )

    # Choosing between strategies is a choice between their fault totals, so a
    # calibrated device ranks only the strategies that read the calibration. A
    # planner that searches for its own placements would route by hop count and
    # report a price the calibration had no part in, so it is refused rather than
    # ranked.
    selection = select_routing_strategy(circuit, CALIBRATED)
    assert tuple(selection.candidates) == LINK_ERROR_ROUTING_STRATEGIES
    refusals: list[str] = []
    for strategy in ("sabre", "sabre_layout"):
        try:
            route_to_topology(circuit, CALIBRATED, strategy=strategy)
        except ValueError as error:
            refusals.append(str(error))
    assert len(refusals) == 2

    print("FlagQuantum weighted routing check passed")
    print(
        f"  link error rates: {[(left, right, rate) for (left, right), rate in CALIBRATION.items()]}"
    )
    print(
        f"  routes between qubits 1 and 3: {CONNECTIVITY_ONLY.shortest_path(1, 3)} and {CALIBRATED.lowest_error_path(1, 3)}"
    )
    print(f"  connecting couplings only: {_opcodes(hop_routed)}")
    print(f"  connecting those couplings with their error: {_opcodes(routed)}")
    print(
        f"  inserted swaps: {plan['inserted_swap_count']} on both, so a count cannot choose"
    )
    print(
        f"  expected link faults: {plan['expected_link_faults']} against {round(avoided, 6)} avoided"
    )
    print(f"  estimate before routing: {estimate.summary()['expected_link_faults']}")
    print(
        f"  automatic choice: {selection.summary()['selected_strategy']} ({selection.summary()['objective']})"
    )
    print(f"  refused: {refusals[0]}")


if __name__ == "__main__":
    main()
