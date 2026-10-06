"""Reconcile the placement-in-routing-plan proposal against the running code.

Most obstacles asserted here are refusals that exist today. Asserting a refusal
is what stops the proposal from describing the repository as it would like it to
be: when an obstacle is closed, the assertion fails and the proposal has to be
updated in the same commit. The contract is a candidate with
``implementation.authorized: false``, so that is the intended reading.

Obstacle 5 is the exception: it was a defect inside the existing v1 contract
rather than a missing capability, so it was fixed without the authorization the
proposal asks for, and the test now asserts the *closed* state together with the
two ways the fix could have gone too far. Its companion assertion, the one that
pins what re-routing an already-routed program still does, is deliberately left
as a refusal because deciding it is the proposal's business and not the fix's.

Two of the obstacles sit behind each other: the strategy clause refuses before
the permutation clauses are reached. Removing one clause to see what the *next*
refusal is, and asserting what it says, is what attributes a refusal to a clause
rather than to the first thing that happened to fire. ``_validator_without_one_clause``
does that, in a module built from the validator's own source, so the experiment
cannot drift from the code it is testing.
"""

from __future__ import annotations

import inspect
import json
import types
from pathlib import Path
from typing import Any

import pytest

import flagquantum as fq
from flagquantum.compiler import CouplingMap
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler.directed_topology import (
    DirectedCouplingMap,
    route_to_directed_topology,
)
from flagquantum.compiler.layout import remove_layout_restore
from flagquantum.compiler.layout_planning import plan_dense_layout
from flagquantum.compiler.routing import ROUTING_STRATEGIES
from flagquantum.deployment.cloud import CloudBackendProfile, create_deployment_package
from flagquantum.deployment.routing_evidence import (
    DeploymentRoutingEvidenceError,
    same_undirected_device,
    validate_deployment_routing_plan,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic import (
    create_dynamic_deployment_package,
    route_dynamic_circuit,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "contracts" / "routing-plan-v2-candidate.json"
PROPOSAL = ROOT / "docs" / "development" / "API_CHANGE_PROPOSAL_068_ROUTING_PLAN_V2.md"

LINE3 = ((0, 1), (1, 2))
LINE4 = ((0, 1), (1, 2), (2, 3))
GRID3X3 = (
    (0, 1),
    (1, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 4),
    (4, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 7),
    (7, 8),
)
GRID6 = ((0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5))

STRATEGY_CLAUSE = 'strategy not in {"restore_after_each_gate", "persistent_layout"}'
SCHEMA_CLAUSE = 'plan.get("schema") != "flagquantum_routing_plan_v1"'
STRATEGY_CLAUSE_WIDENED = (
    'strategy not in {"restore_after_each_gate", "persistent_layout", '
    '"sabre", "sabre_layout"}'
)


def _candidate() -> dict[str, Any]:
    return json.loads(CANDIDATE.read_text(encoding="utf-8"))


def _program(n_wires: int = 4) -> fq.Circuit:
    """A program whose interactions span the device it is routed onto."""

    circuit = fq.Circuit(n_wires)
    circuit.cx(0, n_wires - 1)
    if n_wires > 3:
        circuit.cx(1, n_wires - 2)
    circuit.rx(0, 0.25)
    circuit.cx(0, n_wires - 2)
    return circuit


def _validator_without_one_clause(**replacements: str) -> types.ModuleType:
    """Build the deployment validator with one literal clause replaced.

    The point is to attribute a refusal to a clause. Removing the clause that
    fires first and asking what refuses next is the only way to see whether the
    second obstacle is real or merely hidden behind the first.
    """

    source = inspect.getsource(
        __import__(
            "flagquantum.deployment.routing_evidence", fromlist=["routing_evidence"]
        )
    )
    for original, replacement in replacements.items():
        literal = {"strategy": STRATEGY_CLAUSE, "schema": SCHEMA_CLAUSE}[original]
        assert source.count(literal) == 1, (
            f"the validator no longer contains {literal!r} exactly once; the "
            f"obstacle this test attributes is closed or rewritten, so this "
            f"proposal and its contract need updating"
        )
        source = source.replace(literal, replacement)

    module = types.ModuleType("flagquantum.deployment._candidate_shadow")
    module.__package__ = "flagquantum.deployment"
    module.__file__ = "<routing-plan-v2-candidate shadow>"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _verdict(
    validate: Any, error: type[Exception], plan: Any, coupling: CouplingMap
) -> str:
    try:
        validate(plan, n_qubits=coupling.n_qubits, coupling_map=coupling)
    except error as failure:
        return str(failure)
    return "accepted"


def test_candidate_records_a_proposal_and_authorizes_nothing() -> None:
    candidate = _candidate()

    assert candidate["status"] == "candidate"
    assert candidate["proposal"] == (
        "docs/development/API_CHANGE_PROPOSAL_068_ROUTING_PLAN_V2.md"
    )
    assert candidate["approval_token"] == (
        "approve API_CHANGE_PROPOSAL_068_ROUTING_PLAN_V2"
    )
    assert candidate["implementation"]["authorized"] is False
    assert PROPOSAL.is_file()
    for flag in (
        "public_api_change",
        "stable_core_change",
        "root_manifest_change",
        "default_path_change",
        "capability_level_change",
    ):
        assert candidate[flag] is False, flag


def test_obstacle_1_the_strategy_vocabulary_is_a_copied_literal() -> None:
    candidate = _candidate()
    assert (
        len(ROUTING_STRATEGIES)
        == (
            candidate["obstacles"]["strategy_vocabulary_is_a_copied_literal"][
                "authority_members"
            ]
        )
        == 4
    )

    coupling = CouplingMap(4, LINE4)
    for strategy in ROUTING_STRATEGIES:
        routed = compile_program(
            _program(), coupling_map=coupling, routing_strategy=strategy
        )
        verdict = _verdict(
            validate_deployment_routing_plan,
            DeploymentRoutingEvidenceError,
            routed.metadata["routing"],
            coupling,
        )
        if strategy in {"restore_after_each_gate", "persistent_layout"}:
            assert verdict == "accepted", strategy
        else:
            assert verdict == "unsupported routing strategy", strategy

    # Widening only the strategy literal is what shows the cost is the literal
    # and not something about the plans those two strategies produce.
    widened = _validator_without_one_clause(strategy=STRATEGY_CLAUSE_WIDENED)
    for strategy in ROUTING_STRATEGIES:
        routed = compile_program(
            _program(), coupling_map=coupling, routing_strategy=strategy
        )
        verdict = _verdict(
            widened.validate_deployment_routing_plan,
            widened.DeploymentRoutingEvidenceError,
            routed.metadata["routing"],
            coupling,
        )
        if strategy == "sabre_layout":
            assert verdict == "routing initial permutation must be identity", (
                "widening the strategy vocabulary should expose obstacle 2 for "
                "sabre_layout and nothing else"
            )
        else:
            assert verdict == "accepted", strategy


def test_the_shadow_validator_is_the_real_validator_until_it_is_changed() -> None:
    """An instrument that does not take effect reads exactly like a finding.

    The clause-removal experiment above is only evidence if the module built from
    the validator's own source agrees with the validator whenever nothing was
    replaced. Otherwise a shadow refusal could be an artifact of how the shadow
    was built rather than a property of the contract.
    """

    shadow = _validator_without_one_clause()
    coupling = CouplingMap(4, LINE4)
    for strategy in ROUTING_STRATEGIES:
        plan = dict(
            compile_program(
                _program(), coupling_map=coupling, routing_strategy=strategy
            ).metadata["routing"]
        )
        assert _verdict(
            shadow.validate_deployment_routing_plan,
            shadow.DeploymentRoutingEvidenceError,
            plan,
            coupling,
        ) == _verdict(
            validate_deployment_routing_plan,
            DeploymentRoutingEvidenceError,
            plan,
            coupling,
        ), strategy

    # And the comparison is not vacuous: the same shadow, widened, disagrees with
    # the real validator on exactly the plan the real validator refuses.
    widened = _validator_without_one_clause(strategy=STRATEGY_CLAUSE_WIDENED)
    sabre = compile_program(
        _program(), coupling_map=coupling, routing_strategy="sabre"
    ).metadata["routing"]
    assert (
        _verdict(
            validate_deployment_routing_plan,
            DeploymentRoutingEvidenceError,
            sabre,
            coupling,
        )
        == "unsupported routing strategy"
    )
    assert (
        _verdict(
            widened.validate_deployment_routing_plan,
            widened.DeploymentRoutingEvidenceError,
            sabre,
            coupling,
        )
        == "accepted"
    )


def test_obstacle_2_a_non_identity_placement_has_nowhere_to_be_reported() -> None:
    coupling = CouplingMap(4, LINE4)
    routed = compile_program(
        _program(), coupling_map=coupling, routing_strategy="sabre_layout"
    )
    plan = routed.metadata["routing"]

    # A real strategy, on a real device, plans a real placement.
    assert plan["schema"] == "flagquantum_routing_plan_v1"
    assert tuple(plan["initial_logical_to_physical"]) != tuple(range(4))
    assert tuple(plan["final_logical_to_physical"]) == tuple(range(4))
    assert plan["mapping_restored"] is True

    widened = _validator_without_one_clause(strategy=STRATEGY_CLAUSE_WIDENED)
    assert (
        _verdict(
            widened.validate_deployment_routing_plan,
            widened.DeploymentRoutingEvidenceError,
            plan,
            coupling,
        )
        == "routing initial permutation must be identity"
    )

    # The public planner reaches the same clause, one obstacle further back.
    grid = CouplingMap(6, GRID6)
    placement = plan_dense_layout(_program(), grid)
    assert placement != tuple(range(4)), (
        "plan_dense_layout returned the identity, so this device cannot "
        "demonstrate obstacle 2"
    )
    directed = route_to_directed_topology(
        _program(), DirectedCouplingMap(6, GRID6), initial_layout=placement
    )
    assert (
        tuple(directed.metadata["routing"]["initial_logical_to_physical"]) == placement
    )
    assert (
        _verdict(
            validate_deployment_routing_plan,
            DeploymentRoutingEvidenceError,
            directed.metadata["routing"],
            grid,
        )
        == "unsupported routing plan schema"
    )  # obstacle 4 still fires first
    widened_all = _validator_without_one_clause(
        strategy=STRATEGY_CLAUSE_WIDENED,
        schema='plan.get("schema") not in {"flagquantum_routing_plan_v1", '
        '"flagquantum_directed_routing_plan_v1", "flagquantum_directed_routing_plan_v2"}',
    )
    assert (
        _verdict(
            widened_all.validate_deployment_routing_plan,
            widened_all.DeploymentRoutingEvidenceError,
            directed.metadata["routing"],
            grid,
        )
        == "routing initial permutation must be identity"
    )


def test_obstacle_3_removing_the_restore_produces_a_refused_program() -> None:
    candidate = _candidate()
    coupling = CouplingMap(4, LINE4)
    routed = compile_program(
        _program(), coupling_map=coupling, routing_strategy="persistent_layout"
    )
    assert (
        _verdict(
            validate_deployment_routing_plan,
            DeploymentRoutingEvidenceError,
            routed.metadata["routing"],
            coupling,
        )
        == "accepted"
    )

    reduced = remove_layout_restore(routed)
    reduced_plan = reduced.metadata["routing"]
    assert reduced_plan["mapping_restored"] is False
    assert tuple(reduced_plan["final_logical_to_physical"]) != tuple(range(4))
    assert (
        _verdict(
            validate_deployment_routing_plan,
            DeploymentRoutingEvidenceError,
            reduced_plan,
            coupling,
        )
        == candidate["obstacles"]["final_permutation_must_be_identity"][
            "measured_refusal"
        ]
    )
    assert reduced_plan["inserted_swap_count"] < (
        routed.metadata["routing"]["inserted_swap_count"]
    ), "the reduced program has to be cheaper or obstacle 3 costs nothing"

    # restore_after_each_gate interleaves its restores, so the transformation is
    # the identity on it and obstacle 3 costs that strategy nothing.
    interleaved = compile_program(
        _program(), coupling_map=coupling, routing_strategy="restore_after_each_gate"
    )
    unchanged = remove_layout_restore(interleaved)
    assert unchanged.metadata["routing"]["mapping_restored"] is True
    assert (
        _verdict(
            validate_deployment_routing_plan,
            DeploymentRoutingEvidenceError,
            unchanged.metadata["routing"],
            coupling,
        )
        == "accepted"
    )


def test_obstacle_4_the_directed_plan_is_a_foreign_schema() -> None:
    candidate = _candidate()
    obstacle = candidate["obstacles"]["exactly_one_schema_literal_is_accepted"]
    grid = CouplingMap(6, GRID6)
    directed = route_to_directed_topology(
        _program(),
        DirectedCouplingMap(6, GRID6),
        initial_layout=plan_dense_layout(_program(), grid),
    )
    plan = directed.metadata["routing"]
    assert plan["schema"] in obstacle["observed_schemas"]
    assert plan["schema"] != "flagquantum_routing_plan_v1"
    assert (
        _verdict(
            validate_deployment_routing_plan,
            DeploymentRoutingEvidenceError,
            plan,
            grid,
        )
        == obstacle["measured_refusal"]
    )


def test_obstacle_5_a_device_is_identified_by_its_couplings_not_their_order() -> None:
    """Obstacle 5 was closed as a defect fix; this asserts the closed state.

    Round 24 pinned the refusal: two serializations of one device were two
    devices to this contract. The comparison now runs over undirected coupling
    sets in all three sites that make the decision, so the plan is accepted and
    reused. The assertions below are the closed state, the two ways the fix could
    have gone too far, and the third site that makes it a three-site fix.
    """

    candidate = _candidate()
    obstacle = candidate["obstacles"]["device_identity_is_compared_by_edge_order"]
    assert obstacle["status"].startswith("closed_by_defect_fix")
    assert obstacle["re_routing_decision_status"] == "open", (
        "the order fix removed a trigger, not the decision behind it, so the "
        "contract cannot report this chain as closed"
    )

    shifted = ((1, 2), (0, 1), (0, 3), (2, 5), (1, 4), (4, 5), (3, 4))
    assert set(shifted) == set(GRID6)
    assert shifted != GRID6
    assert shifted != tuple(sorted(shifted)), (
        "the re-ordered device happens to be its own canonical order, so this "
        "test could not tell canonicalization from its absence"
    )
    assert CouplingMap(6, shifted).edges == shifted, (
        "CouplingMap canonicalizes its edges, so the shifted sequence is no "
        "longer reachable from a real device and this test would stop being "
        "evidence that the deployment contract tolerates edge order"
    )
    canonical = CouplingMap(6, GRID6).edges

    # The comparison ignores the order of the couplings and the order of the
    # endpoints inside one coupling ...
    assert same_undirected_device(shifted, canonical)
    assert same_undirected_device(
        tuple((right, left) for left, right in shifted), canonical
    )
    assert same_undirected_device(canonical, canonical)
    # ... and still separates devices that differ by a single coupling.
    assert not same_undirected_device(GRID6[:-1], canonical)
    assert not same_undirected_device(GRID6 + ((5, 8),), canonical)
    assert not same_undirected_device(
        ((0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (5, 8)), canonical
    )

    backend = CloudBackendProfile(
        provider="local",
        name="grid6",
        n_qubits=6,
        coupling_map=CouplingMap(6, GRID6),
        is_simulator=True,
    )
    assert backend.coupling_map is not None
    routed = compile_program(
        _program(n_wires=6),
        coupling_map=CouplingMap(6, shifted),
        routing_strategy="persistent_layout",
    )
    plan = dict(routed.metadata["routing"])
    assert tuple(plan["coupling_edges"]) == shifted
    assert tuple(plan["coupling_edges"]) != backend.coupling_map.edges
    assert set(plan["coupling_edges"]) == set(backend.coupling_map.edges)

    # Site 1: the validator accepts the plan against the device it was routed on.
    assert (
        _verdict(
            validate_deployment_routing_plan,
            DeploymentRoutingEvidenceError,
            plan,
            backend.coupling_map,
        )
        == "accepted"
    )

    # The normalization runs whether or not there is a device to compare against,
    # so a field that is not a list of pairs is refused by name in both cases. It
    # used to be read only when a device was present, which made the same field
    # acceptable with one argument and refused with the other.
    with pytest.raises(DeploymentRoutingEvidenceError) as malformed:
        validate_deployment_routing_plan(
            plan | {"coupling_edges": ((0, 1, 2),)}, n_qubits=6, coupling_map=None
        )
    assert str(malformed.value) == "routing coupling edges must be pairs of qubits"

    # Site 2: the deployment entry reuses the plan instead of routing the
    # already-routed program a second time, so the plan's own counts survive and
    # the SWAP-count refusal that the second pass used to produce is not reached.
    package = create_deployment_package(routed, backend=backend)
    assert package.metadata["routing_reused"] is True
    assert package.metadata["routing_evidence"]["routing_reused"] is True
    published = package.metadata["routing_plan"]
    assert published["inserted_swap_count"] == plan["inserted_swap_count"] != 0, (
        "the plan has to carry SWAPs, or its count could not show that the "
        "second pass's zero was an accounting artifact rather than a reuse"
    )

    # Site 3: dynamic packaging makes the same decision from its own module, so
    # one comparison in one file would have left this path re-routing.
    dynamic = DynamicCircuit(3)
    dynamic.measure(0, classical_bit=0)
    dynamic.conditional("cx", (0, 2), classical_bit=0)
    dynamic_backend = CloudBackendProfile(
        provider="local",
        name="dynamic-line3",
        n_qubits=3,
        supports_openqasm=True,
        supports_dynamic_circuits=True,
        max_classical_bits=4,
        coupling_map=CouplingMap(3, LINE3),
    )
    dynamic_routed = route_dynamic_circuit(
        dynamic, CouplingMap(3, tuple(reversed(LINE3)))
    )
    dynamic_plan = dict(dynamic_routed.to_ir().metadata["routing"])
    assert tuple(dynamic_plan["coupling_edges"]) == tuple(reversed(LINE3))
    assert set(dynamic_plan["coupling_edges"]) == set(LINE3)
    assert dynamic_backend.coupling_map is not None
    dynamic_package = create_dynamic_deployment_package(
        dynamic_routed, backend=dynamic_backend, shots=32
    )
    assert dynamic_package.metadata["routing_reused"] is True
    assert (
        dynamic_package.metadata["routing_plan"]["inserted_swap_count"]
        == dynamic_plan["inserted_swap_count"]
        != 0
    ), (
        "the dynamic package has to report the plan it reused, not a second "
        "pass that planned nothing while shipping the first pass's SWAPs"
    )


def test_re_routing_onto_a_different_device_is_still_the_open_decision() -> None:
    """The order fix removed a trigger, not the decision behind it.

    A program that arrives routed onto a genuinely *different* device is routed
    again, and the second pass reports that it planned no SWAPs while shipping
    the first pass's. That is the accounting question the proposal asks to decide
    rather than take, so this test pins it: when the decision lands, this
    assertion fails and the contract has to be updated in the same commit.
    """

    candidate = _candidate()
    obstacle = candidate["obstacles"]["device_identity_is_compared_by_edge_order"]
    assert obstacle["re_routing_decision_status"] == "open", (
        "record the decision in the contract before changing the behaviour it "
        "describes"
    )
    assert any(
        "re-routing an already-routed program" in item
        for item in candidate["acceptance"]
    ), "the open decision has to stay an acceptance criterion"

    routed = compile_program(
        _program(n_wires=4),
        coupling_map=CouplingMap(4, LINE4),
        routing_strategy="persistent_layout",
    )
    plan = dict(routed.metadata["routing"])
    assert plan["inserted_swap_count"] != 0
    other_device = CloudBackendProfile(
        provider="local",
        name="ring4",
        n_qubits=4,
        coupling_map=CouplingMap.ring(4),
        is_simulator=True,
    )

    # Re-routing is the right *action* here, because the plan is for another
    # device. What is not defined is what the program's counts then mean.
    with pytest.raises(DeploymentRoutingEvidenceError) as refused:
        create_deployment_package(routed, backend=other_device)
    assert str(refused.value) == obstacle["refusals_that_follow"][1]

    second_pass = compile_program(
        routed,
        coupling_map=other_device.coupling_map,
        routing_strategy="restore_after_each_gate",
    ).metadata["routing"]
    assert second_pass["planned_inserted_swap_count"] == 0
    assert (
        second_pass["post_optimization_inserted_swap_count"]
        == plan["inserted_swap_count"]
    ), (
        "the second pass's retained count has to be the first pass's SWAPs, "
        "which is what makes the refusal an accounting artifact"
    )


def test_every_clause_that_refuses_today_is_still_a_contract() -> None:
    """A widened contract must still refuse, so every clause keeps an input."""

    coupling = CouplingMap(4, LINE4)
    wider_device = CouplingMap(4, ((0, 1), (1, 2), (2, 3), (0, 3)))
    routed = compile_program(
        _program(), coupling_map=coupling, routing_strategy="persistent_layout"
    )
    good = dict(routed.metadata["routing"])
    assert (
        _verdict(
            validate_deployment_routing_plan,
            DeploymentRoutingEvidenceError,
            good,
            coupling,
        )
        == "accepted"
    )

    cases = (
        (
            good | {"schema": "flagquantum_routing_plan_v3"},
            coupling,
            ("unsupported routing plan schema"),
        ),
        (good | {"strategy": "teleport"}, coupling, "unsupported routing strategy"),
        (
            good,
            wider_device,
            "routing coupling edges do not match the deployment backend",
        ),
        (
            good | {"coupling_edges": ((0, 1, 2),)},
            coupling,
            "routing coupling edges must be pairs of qubits",
        ),
        (
            good | {"initial_logical_to_physical": (1, 0, 2, 3)},
            coupling,
            ("routing initial permutation must be identity"),
        ),
        (
            good | {"pre_restore_logical_to_physical": (0, 1, 2)},
            coupling,
            ("routing pre-restore mapping must be a complete permutation"),
        ),
        (
            good | {"final_logical_to_physical": (0, 1, 2)},
            coupling,
            ("deployment routing must restore the final logical permutation"),
        ),
        (
            good | {"mapping_restored": False},
            coupling,
            ("deployment routing must restore the final logical permutation"),
        ),
        (
            good | {"direction_semantics": "unknown"},
            coupling,
            ("routing direction semantics are missing or unsupported"),
        ),
        (
            good | {"inserted_swap_count": good["planned_inserted_swap_count"] + 1},
            coupling,
            "inserted SWAP compatibility count must equal planned count",
        ),
    )
    for plan, device, expected in cases:
        assert (
            _verdict(
                validate_deployment_routing_plan,
                DeploymentRoutingEvidenceError,
                plan,
                device,
            )
            == expected
        ), expected


def test_candidate_names_every_obstacle_these_tests_assert() -> None:
    candidate = _candidate()

    assert set(candidate["obstacles"]) == {
        "strategy_vocabulary_is_a_copied_literal",
        "initial_permutation_must_be_identity",
        "final_permutation_must_be_identity",
        "exactly_one_schema_literal_is_accepted",
        "device_identity_is_compared_by_edge_order",
    }
    assert set(candidate["proposed_vocabulary"]["final_layout"]["values"]) == {
        "identity",
        "explicit_permutation",
        "logical_result_physical_slots",
    }
    assert len(candidate["acceptance"]) == 7
    assert candidate["non_goals"]


def test_the_grid3x3_device_is_only_used_for_width_not_for_order() -> None:
    """The two devices this module uses are declared, not incidental."""

    assert CouplingMap(4, LINE4).edges == LINE4
    assert CouplingMap(6, GRID6).n_qubits == 6
    assert CouplingMap(9, GRID3X3).n_qubits == 9
