"""Shared bounded Boolean factors preserve native phase and clean workspace."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeVar

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.logistics.facility_location import main as facility
from examples.synthesis.applications.logistics.facility_location.comparisons import (
    circuit as reference,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import CostFactor, factored_cost_circuit
from examples.synthesis.quadratic_reference import order_register_factors
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, _phase_factors
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.compiler._phase_factors import Cube
from flagquantum.compiler._phase_polynomial import Location
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "tests/fixtures/classiq_qmod/corpus/original/applications/logistics/facility_location/facility_location.qmod"
)
_Node = TypeVar("_Node")


@pytest.mark.parametrize("binding", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_facility_original_and_native(
    binding: int, objective: Literal["width", "depth"]
) -> None:
    params = reference.BINDINGS[binding]
    native = synth.synthesize(
        facility.main, classical_inputs={"params": params}, objective=objective
    )
    source = synth.synthesize(
        SOURCE, classical_inputs={"params": params}, objective=objective
    )
    compare_states(native, reference.reference_circuit(params), ("assignments",))
    compare_states(
        source, reference.source_reference_circuit(params), reference.SOURCE_FIELDS
    )
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 17
    assert native.report["workspace_qubits"] == 5


def test_every_facility_assignment_matches_independent_constraints() -> None:
    factors = reference.cost_factors()
    for basis in range(4096):
        bits = tuple((basis >> bit) & 1 for bit in range(12))
        total = sum(
            table[sum(bits[bit] << index for index, bit in enumerate(support))]
            for support, table in factors
        )
        assert total == pytest.approx(reference.cost(bits), abs=1e-12, rel=0)
    assert reference.cost((0,) * 12) == -80
    assert reference.cost((1,) * 12) == pytest.approx(-sum(reference.WEIGHTS) - 170)


def test_eight_bit_truth_reference_has_absolute_phase_without_dense_matrix() -> None:
    table = tuple(
        0.37 if bits == 0 else -0.29 if bits == 255 else 0.0 for bits in range(256)
    )
    circuit = factored_cost_circuit(8, [(tuple(range(8)), table)], (1.0, 0.0))
    actual = fq.run(circuit).statevector().flatten()
    expected = torch.exp(1j * torch.tensor(table, dtype=torch.float64)) / 16
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_replaced_factor_order_works_behind_both_unchanged_frontends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _phase_factors.preserve_factors
    counts: list[int] = []

    def replacement(
        expression: _Node,
        quantum_word: Callable[[_Node], tuple[Location, ...] | None],
        *,
        operator_of: Callable[[_Node], str],
        operands_of: Callable[[_Node], tuple[_Node, ...]],
        replace_operands: Callable[[_Node, tuple[_Node, ...]], _Node],
        make_factor: Callable[[_Node, int], _Node],
    ) -> tuple[_Node, tuple[Cube, ...]]:
        markers: dict[int, int] = {}

        def mark(node: _Node, index: int) -> _Node:
            result = make_factor(node, index)
            markers[id(result)] = index
            return result

        rewritten, factors = original(
            expression,
            quantum_word,
            operator_of=operator_of,
            operands_of=operands_of,
            replace_operands=replace_operands,
            make_factor=mark,
        )

        def remap(node: _Node) -> _Node:
            if id(node) in markers:
                return make_factor(node, len(factors) - 1 - markers[id(node)])
            return replace_operands(
                node, tuple(remap(child) for child in operands_of(node))
            )

        counts.append(len(factors))
        return remap(rewritten), tuple(reversed(factors))

    monkeypatch.setattr(_phase_factors, "preserve_factors", replacement)
    params = reference.BINDINGS[0]
    native = facility.build_program(params)
    compare_states(native, reference.reference_circuit(params), ("assignments",))
    source = synth.synthesize(SOURCE, classical_inputs={"params": params})
    compare_states(
        source, reference.source_reference_circuit(params), reference.SOURCE_FIELDS
    )
    assert counts == [3] * 10


@pytest.mark.parametrize("helper", [False, True])
@pytest.mark.parametrize("angle", [0.37, -0.29])
def test_signed_literals_views_and_coherent_spectators(
    helper: bool, angle: float
) -> None:
    # Overlapping cubes retain a coherent spectator within the existing word bound.
    expression = "(q[0]*(1-q[1])*(1-q[2])*(1-q[3])+(1-q[3])*(1-q[4])*(1-q[5])*(1-q[6])+(1-q[7])*(1-q[8])*(1-q[9])*(1-q[10])-1)**2"
    helper_source = (
        (
            "@qfunc\ndef cost(angle:float,q:QArray[QBit]):\n    phase("
            + expression
            + ",angle)\n"
        )
        if helper
        else ""
    )
    width = 12
    action = "cost(angle,q[1:12])" if helper else "phase(" + expression + ",angle)"
    entry = program_definition(
        "from flagquantum.synth import *\n"
        + helper_source
        + f"@qfunc\ndef main(angle:float,q:Output[QArray[QBit]]):\n    allocate({width},q)\n    hadamard_transform(q)\n    {action}\n"
    )
    program = synth.synthesize(entry, classical_inputs={"angle": angle})
    factors: list[CostFactor] = [((), (1.0,))]
    offset = 1 if helper else 0
    cubes = (
        ((offset, True), (offset + 1, False), (offset + 2, False), (offset + 3, False)),
        tuple((offset + bit, False) for bit in (3, 4, 5, 6)),
        tuple((offset + bit, False) for bit in (7, 8, 9, 10)),
    )
    for first in range(3):
        literals = dict(cubes[first])
        support = tuple(literals)
        selected = sum(int(literals[bit]) << i for i, bit in enumerate(support))
        factors.append(
            (
                support,
                tuple(
                    -1.0 if value == selected else 0.0
                    for value in range(1 << len(support))
                ),
            )
        )
        for second in range(first + 1, 3):
            joined = dict((*cubes[first], *cubes[second]))
            pair = tuple(joined)
            selected = sum(int(joined[bit]) << i for i, bit in enumerate(pair))
            factors.append(
                (
                    pair,
                    tuple(
                        2.0 if value == selected else 0.0
                        for value in range(1 << len(pair))
                    ),
                )
            )
    compare_states(program, factored_cost_circuit(width, factors, (angle, 0.0)), ("q",))


def test_shared_factors_across_distinct_owners() -> None:
    expression = "((1-a[0])*(1-a[1])*(1-b[0])*(1-b[1])+(1-a[2])*(1-a[3])*(1-b[2])*(1-b[3])+(1-a[4])*(1-a[5])*(1-b[4])*(1-b[5])-1)**2"
    entry = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef main(a:Output[QArray[QBit]],b:Output[QArray[QBit]]):\n    allocate(6,a)\n    allocate(6,b)\n    hadamard_transform(a)\n    hadamard_transform(b)\n    phase("
        + expression
        + ",0.37)\n"
    )
    program = synth.synthesize(entry)
    supports = [(2 * i, 2 * i + 1, 6 + 2 * i, 7 + 2 * i) for i in range(3)]
    factors: list[CostFactor] = [((), (1.0,))]
    for first in range(3):
        factors.append((supports[first], (-1.0,) + (0.0,) * 15))
        for second in range(first + 1, 3):
            factors.append(
                ((*supports[first], *supports[second]), (2.0,) + (0.0,) * 255)
            )
    compare_states(
        program,
        factored_cost_circuit(12, order_register_factors(factors, (6, 6)), (0.37, 0.0)),
        ("a", "b"),
    )


@pytest.mark.parametrize(
    "budget",
    [
        "max_workspace",
        "max_qubits",
        "max_operations",
        "max_depth",
        "memory_limit_bytes",
    ],
)
def test_factoring_respects_actual_physical_budgets(budget: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            facility.main,
            classical_inputs={"params": reference.BINDINGS[0]},
            max_workspace=0 if budget == "max_workspace" else None,
            max_qubits=16 if budget == "max_qubits" else None,
            max_operations=1 if budget == "max_operations" else None,
            max_depth=1 if budget == "max_depth" else None,
            memory_limit_bytes=1 if budget == "memory_limit_bytes" else None,
        )


def test_unfactorable_expression_keeps_monomial_budget() -> None:
    expression = "(" + "+".join(f"q[{i}]" for i in range(12)) + ")**8"
    entry = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef main(q:Output[QArray[QBit]]):\n    allocate(12,q)\n    phase("
        + expression
        + ",0.37)\n"
    )
    with pytest.raises(HybridVerificationError, match="256 monomials"):
        synth.synthesize(entry)


def test_wider_mask_conservative_lowering_refuses_memory_before_execution() -> None:
    expression = "((1-q[0])*(1-q[1])*(1-q[2])*(1-q[3])+(1-q[4])*(1-q[5])*(1-q[6])*(1-q[7])+(1-q[8])*(1-q[9])*(1-q[10])*(1-q[11])-1)**2"
    entry = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef cost(q:QArray[QBit]):\n    phase("
        + expression
        + ",0.37)\n@qfunc\ndef main(q:Output[QArray[QBit]]):\n    allocate(14,q)\n    cost(q[1:13])\n"
    )
    with pytest.raises(ValueError, match="memory budget"):
        synth.synthesize(entry, memory_limit_bytes=64 * 1024**2)


@pytest.mark.parametrize("guarded", [False, True])
def test_factoring_does_not_expand_native_predicate_boundary(guarded: bool) -> None:
    cubes = "+".join(
        "*".join(f"(1-q[{4*row+bit}])" for bit in range(4)) for row in range(3)
    )
    action = (
        "control((" + cubes + ")==1,lambda:phase(q[0],0.37))"
        if guarded
        else "phase((" + cubes + ")==1,0.37)"
    )
    entry = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef main(q:Output[QArray[QBit]]):\n    allocate(12,q)\n    "
        + action
        + "\n"
    )
    diagnostic = "eight input bits" if guarded else "use finite real arithmetic"
    with pytest.raises(HybridVerificationError, match=diagnostic):
        synth.synthesize(entry)
