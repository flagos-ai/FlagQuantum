"""Predicate-gated native phases retain shared reads and exact source programs."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.qaoa import main as qaoa
from examples.synthesis.algorithms.search_and_optimization.qaoa.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_knapsack import (
    main as knapsack,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import estimate_resources
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_polynomial_phase import definition

pytestmark = pytest.mark.integration
CORPUS = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/algorithms/search_and_optimization/QAOA"
)
ERRORS = (synth.SynthesisError, HybridVerificationError)


@pytest.mark.parametrize("case", ["qaoa", "qaoa_knapsack"])
@pytest.mark.parametrize("dataset", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_knapsack_originals_and_native_composition_match_raw_states(
    case: str, dataset: int, objective: Literal["width", "depth"]
) -> None:
    entry = qaoa.main if case == "qaoa" else knapsack.main
    params = BINDINGS[dataset]
    native = synth.synthesize(
        entry, classical_inputs={"params": params}, objective=objective, max_workspace=0
    )
    source = synth.synthesize(
        CORPUS / (case + ".qmod"),
        classical_inputs={"params": params},
        objective=objective,
    )
    reference = reference_circuit(params)
    compare_states(native, reference, ("a", "b"))
    compare_states(source, reference, ("v.a", "v.b"))
    for program in (native, source):
        assert program.report["workspace_qubits"] == 0
        assert estimate_resources(program.circuit).n_qubits == 5


@pytest.mark.parametrize(
    ("guard", "predicate"),
    [
        ("a-b<0", lambda a, b: a - b < 0),
        ("a-b<=0", lambda a, b: a - b <= 0),
        ("a-b>0", lambda a, b: a - b > 0),
        ("a-b>=0", lambda a, b: a - b >= 0),
        ("a-b==0", lambda a, b: a - b == 0),
        ("a-b!=0", lambda a, b: a - b != 0),
        ("2*a+3*b<=12", lambda a, b: 2 * a + 3 * b <= 12),
    ],
)
def test_nested_helper_phases_cover_negative_differences_and_inclusive_boundary(
    guard: str, predicate: Callable[[int, int], bool]
) -> None:
    entry = definition(
        "allocate(3,a)\nallocate(2,b)\nhadamard_transform(a)\nhadamard_transform(b)\nphase(a-0.17,0.29)\nouter(g,a,b)\nhadamard_transform(a)\nhadamard_transform(b)",
        "g:float,a:Output[QNum],b:Output[QNum]",
        "@qfunc\ndef inner(g:float,a:QNum,b:QNum):\n    control("
        + guard
        + ",lambda:phase(3+2*a-b,g))\n"
        "@qfunc\ndef outer(g:float,a:QNum,b:QNum):\n    inner(g+0.2,a,b)\n",
    )
    gamma = -0.37
    values = torch.tensor(
        [
            0.29 * (a - 0.17)
            + (gamma + 0.2) * (3 + 2 * a - b if predicate(a, b) else 0)
            for a in range(8)
            for b in range(4)
        ],
        dtype=torch.float64,
    )
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        reference.h(wire)
    reference.any(*range(5), matrix=torch.diag(torch.exp(1j * values)))
    for wire in range(5):
        reference.h(wire)
    compare_states(
        synth.synthesize(entry, classical_inputs={"g": gamma}, max_workspace=0),
        reference,
        ("a", "b"),
    )


@pytest.mark.parametrize("constant", [-0.37, 0.53])
@pytest.mark.parametrize("guard", ["q[0]", "q[0]==1"])
def test_scalar_view_guard_retains_conditional_constant_phase_with_interference(
    constant: float,
    guard: str,
) -> None:
    entry = definition(
        "allocate(3,q)\nhadamard_transform(q)\nhelper(g,q[1:3])\nhadamard_transform(q)",
        "g:float,q:Output[QArray[QBit]]",
        "@qfunc\ndef helper(g:float,q:QArray[QBit]):\n    control("
        + guard
        + ",lambda:phase("
        + str(constant)
        + "+q[0]/2-q[1],g))\n",
    )
    values = torch.tensor(
        [(-0.29 * (constant + 0.5 - ((i >> 2) & 1)) if i & 2 else 0) for i in range(8)],
        dtype=torch.float64,
    )
    reference = (
        fq.Circuit(3, dtype=torch.complex128)
        .h(0)
        .h(1)
        .h(2)
        .any(0, 1, 2, matrix=torch.diag(torch.exp(1j * values)))
        .h(0)
        .h(1)
        .h(2)
    )
    compare_states(
        synth.synthesize(entry, classical_inputs={"g": -0.29}, max_workspace=0),
        reference,
        ("q",),
    )


def test_scalar_phase_may_read_its_own_predicate_without_changing_basis_values() -> (
    None
):
    entry = definition(
        "allocate(q)\nhadamard_transform(q)\ncontrol(q==1,lambda:phase(q,3.141592653589793))\nhadamard_transform(q)",
        "q:Output[QBit]",
    )
    compare_states(
        synth.synthesize(entry, max_workspace=0),
        fq.Circuit(1, dtype=torch.complex128).x(0),
        ("q",),
    )


@pytest.mark.parametrize("expression", ["1+2", "math.pi", "g"])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_preexisting_constant_controls_preserve_resources_and_absolute_phase(
    expression: str, objective: Literal["width", "depth"]
) -> None:
    entry = definition(
        "allocate(2,a)\nhadamard_transform(a)\n"
        f"control(a<2,lambda:phase({expression},0.37))",
        "g:int,a:Output[QNum[2]]",
        "import math\n",
    )
    program = synth.synthesize(entry, classical_inputs={"g": 2}, objective=objective)
    resources = estimate_resources(program.circuit)
    # Measured on the preceding committed capture implementation, cd7d80314.
    assert (resources.n_qubits, resources.n_operations, resources.depth) == (8, 13, 11)
    assert resources.operation_counts == {"h": 2, "x": 4, "ccx": 4, "cx": 2, "phase": 1}
    assert program.report["workspace_qubits"] == 6
    constant = {"1+2": 3.0, "math.pi": torch.pi, "g": 2.0}[expression]
    values = torch.tensor([constant, constant, 0.0, 0.0], dtype=torch.float64)
    reference = (
        fq.Circuit(2, dtype=torch.complex128)
        .h(0)
        .h(1)
        .any(0, 1, matrix=torch.diag(torch.exp(0.37j * values)))
    )
    compare_states(program, reference, ("a",))


@pytest.mark.parametrize(
    ("body", "signature", "prelude", "diagnostic"),
    [
        (
            "allocate(2,b)\ncontrol(a-a==0,lambda:phase(b-0.5,0.2))",
            "a:Output[QNum[2]],b:Output[QNum[2]]",
            "",
            "initialize",
        ),
        (
            "allocate(2,a)\ncontrol(a<2,lambda:phase(b-b,0.2))",
            "a:Output[QNum[2]],b:Output[QNum[2]]",
            "",
            "initialize",
        ),
        (
            "allocate(2,a)\nallocate(2,b)\ncontrol(a/2<1,lambda:phase(b-0.5,0.2))",
            "a:Output[QNum],b:Output[QNum]",
            "",
            "integer arithmetic",
        ),
        (
            "allocate(2,a)\nallocate(2,b)\ncontrol(0.5*a<1,lambda:phase(b-0.5,0.2))",
            "a:Output[QNum],b:Output[QNum]",
            "",
            "integer arithmetic",
        ),
        (
            "allocate(2,a)\nallocate(2,b)\ncontrol(a<b<3,lambda:phase(b-0.5,0.2))",
            "a:Output[QNum],b:Output[QNum]",
            "",
            "one integer comparison",
        ),
        (
            "allocate(2,a)\ncontrol(a,lambda:phase(a-0.5,0.2))",
            "a:Output[QNum]",
            "",
            "comparison or scalar bit",
        ),
        (
            "allocate(2,a)\ncontrol(boom(),lambda:phase(a-0.5,0.2))",
            "a:Output[QNum]",
            "def boom():\n    raise AssertionError('host callback executed')\n",
            "comparison or scalar bit",
        ),
        (
            "allocate(2,a)\ncontrol(a<2,lambda:phase(boom(),0.2))",
            "a:Output[QNum]",
            "def boom():\n    raise AssertionError('host callback executed')\n",
            "finite real",
        ),
        (
            "allocate(2,a)\nhelper(a,a)",
            "a:Output[QNum]",
            "@qfunc\ndef helper(a:QNum,b:QNum):\n    control(a==0,lambda:phase(b-0.5,0.2))\n",
            "disjoint",
        ),
        (
            "allocate(a)\nhelper(a)",
            "a:Output[QBit]",
            "@qfunc\ndef helper(a:QBit):\n    control(a==0,lambda:H(a))\n",
            "phase-only control",
        ),
        (
            "allocate(a)\nwith adjoint():\n    helper(a)",
            "a:Output[QBit]",
            "@qfunc\ndef helper(a:QBit):\n    control(a==0,lambda:phase(a-0.5,0.2))\n",
            "Const-only",
        ),
        (
            "allocate(a)\nhelper(a)",
            "a:Output[QBit]",
            "@qfunc\ndef publish(a:Output[QBit]):\n    allocate(a)\n@qfunc\ndef helper(a:QBit):\n    control(a==0,lambda:publish(a))\n",
            "phase-only control",
        ),
        (
            "allocate(9,a)\ncontrol(a<3,lambda:phase(a-0.5,0.2))",
            "a:Output[QNum]",
            "",
            "eight input bits",
        ),
        (
            "allocate(8,a)\nallocate(b)\ncontrol(a==255,lambda:phase(b-0.5,0.2))",
            "a:Output[QNum],b:Output[QBit]",
            "",
            "degree eight",
        ),
        (
            "allocate(8,a)\nallocate(17,b)\ncontrol(a<1,lambda:phase(b-0.5,0.2))",
            "a:Output[QNum],b:Output[QNum]",
            "",
            "24 quantum",
        ),
    ],
)
def test_guarded_phase_refusals_preserve_semantic_and_borrow_boundaries(
    body: str, signature: str, prelude: str, diagnostic: str
) -> None:
    with pytest.raises(ERRORS, match=diagnostic):
        synth.synthesize(definition(body, signature, prelude))


@pytest.mark.parametrize("budget", ["qubits", "operations", "memory"])
def test_guarded_knapsack_enforces_actual_resource_budgets(budget: str) -> None:
    with pytest.raises((*ERRORS, PlanningError)):
        synth.synthesize(
            qaoa.main,
            classical_inputs={"params": BINDINGS[0]},
            max_qubits=4 if budget == "qubits" else 5,
            max_operations=1 if budget == "operations" else 100000,
            memory_limit_bytes=16 if budget == "memory" else 2**20,
        )


def test_guard_expression_depth_is_checked_before_recursive_expansion() -> None:
    condition = "+".join(["a", *(["1"] * 130)]) + "<200"
    entry = definition(
        "allocate(a)\ncontrol(" + condition + ",lambda:phase(a-0.5,0.2))",
        "a:Output[QBit]",
    )
    with pytest.raises(ERRORS, match="expression depth"):
        synth.synthesize(entry)
