"""Real native phases preserve sparse arithmetic, ownership and absolute amplitudes."""

from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.adapt_qaoa.comparisons.circuit import (
    BINDINGS as ADAPT_BINDINGS,
)
from examples.synthesis.applications.optimization.adapt_qaoa.comparisons.circuit import (
    reference_circuit as adapt_reference,
)
from examples.synthesis.applications.optimization.adapt_qaoa.main import main as adapt
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.combinatorial_workshop.combinatorial_qmod_workshop_for_maxcut.comparisons.circuit import (
    BINDINGS as MAXCUT_BINDINGS,
)
from examples.synthesis.tutorials.workshops.combinatorial_workshop.combinatorial_qmod_workshop_for_maxcut.comparisons.circuit import (
    reference_circuit as maxcut_reference,
)
from examples.synthesis.tutorials.workshops.combinatorial_workshop.combinatorial_qmod_workshop_for_maxcut.main import (
    main as maxcut,
)
from flagquantum import synth
from flagquantum.compiler import estimate_resources
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/corpus/original"
ERRORS = (synth.SynthesisError, HybridVerificationError)


def definition(
    body: str, signature: str = "q:Output[QArray[QBit]]", prelude: str = ""
) -> _QuantumFunction:
    return program_definition(
        "from flagquantum.synth import *\n"
        + prelude
        + f"\n@qfunc\ndef main({signature}):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("case", ["adapt", "maxcut"])
@pytest.mark.parametrize("dataset", [0, 1])
def test_complete_originals_and_native_programs_match_independent_full_states(
    objective: Literal["width", "depth"],
    case: str,
    dataset: int,
) -> None:
    if case == "adapt":
        entry, params, reference = adapt, ADAPT_BINDINGS[dataset], adapt_reference
        path = "applications/optimization/adapt_qaoa/adapt_qaoa.qmod"
    else:
        entry, params, reference = maxcut, MAXCUT_BINDINGS[dataset], maxcut_reference
        path = "tutorials/workshops/combinatorial_workshop/combinatorial_qmod_workshop_for_maxcut.qmod"
    native = synth.synthesize(
        entry, classical_inputs={"params": params}, objective=objective, max_workspace=0
    )
    original = synth.synthesize(
        CORPUS / path, classical_inputs={"params": params}, objective=objective
    )
    for program in (native, original):
        compare_states(program, reference(params), ("v",))
        assert program.report["workspace_qubits"] == 0
    assert native.report["resources"] == original.report["resources"]


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_nested_mutable_helpers_preserve_shared_reads_bit_order_and_caller_scope(
    objective: Literal["width", "depth"],
) -> None:
    entry = definition(
        "allocate(3,q)\nhadamard_transform(q)\nS(q[1])\nCX(q[0],q[2])\nouter(g,q[1:3])\nphase(g*q[0]+0.17,g)",
        "g:float,q:Output[QArray[QBit]]",
        "@qfunc\ndef inner(g:float,q:QArray[QBit]):\n    phase(-0.3+q[0]*q[0]/2+q[0]*q[1],g)\n"
        "@qfunc\ndef outer(g:float,q:QArray[QBit]):\n    inner(g+0.2,q)\n",
    )
    gamma = -0.37
    circuit = fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2).s(1).cx(2, 0)
    values = []
    for basis in range(8):
        bits = tuple((basis >> i) & 1 for i in range(3))
        values.append(
            (gamma + 0.2) * (-0.3 + bits[1] / 2 + bits[1] * bits[2])
            + gamma * (gamma * bits[0] + 0.17)
        )
    circuit.any(
        0,
        1,
        2,
        matrix=torch.diag(torch.exp(1j * torch.tensor(values, dtype=torch.float64))),
    )
    compare_states(
        synth.synthesize(
            entry, classical_inputs={"g": gamma}, objective=objective, max_workspace=0
        ),
        circuit,
        ("q",),
    )


@pytest.mark.parametrize("constant", [-0.73, 0.41])
def test_constant_phase_survives_interference_and_no_phase_alignment(
    constant: float,
) -> None:
    entry = definition(
        f"allocate(2,q)\nhadamard_transform(q)\nphase({constant}+0.25*q[0]-0.5*q[1],0.37)\nhadamard_transform(q)"
    )
    values = torch.tensor(
        [constant + 0.25 * (i & 1) - 0.5 * ((i >> 1) & 1) for i in range(4)],
        dtype=torch.float64,
    )
    reference = (
        fq.Circuit(2, dtype=torch.complex128)
        .h(0)
        .h(1)
        .any(0, 1, matrix=torch.diag(torch.exp(0.37j * values)))
        .h(0)
        .h(1)
    )
    compare_states(synth.synthesize(entry, max_workspace=0), reference, ("q",))
    without_constant = fq.run(reference).statevector() * torch.exp(
        torch.tensor(-0.37j * constant, dtype=torch.complex128)
    )
    assert (
        float(
            (fq.run(synth.synthesize(entry)).statevector() - without_constant)
            .abs()
            .max()
        )
        > 0.1
    )


def test_numeric_register_phase_uses_unsigned_bit_weights_and_finite_real_terms() -> (
    None
):
    entry = definition(
        "allocate(2,q)\nhadamard_transform(q)\nphase(-0.25*q*q+q/2+0.37,g)",
        "g:float,q:Output[QNum]",
    )
    values = torch.tensor(
        [-0.25 * x * x + x / 2 + 0.37 for x in range(4)], dtype=torch.float64
    )
    reference = (
        fq.Circuit(2, dtype=torch.complex128)
        .h(0)
        .h(1)
        .any(0, 1, matrix=torch.diag(torch.exp(-0.31j * values)))
    )
    compare_states(
        synth.synthesize(entry, classical_inputs={"g": -0.31}, max_workspace=0),
        reference,
        ("q",),
    )


@pytest.mark.parametrize(
    ("body", "signature", "prelude", "diagnostic"),
    [
        ("phase(q[0]-q[0],0.1)", "q:Output[QArray[QBit]]", "", "width|initialize"),
        ("allocate(2,q)\nphase(q[2],0.1)", "q:Output[QArray[QBit]]", "", "bounds"),
        (
            "allocate(2,q)\nphase(q[:1],0.1)",
            "q:Output[QArray[QBit]]",
            "",
            "scalar element",
        ),
        (
            "allocate(2,q)\nphase(q+0.5,0.1)",
            "q:Output[QArray[QBit]]",
            "",
            "element index",
        ),
        (
            "allocate(2,q)\nphase(q[0]/q[1],0.1)",
            "q:Output[QArray[QBit]]",
            "",
            "divisor",
        ),
        ("allocate(2,q)\nphase(q[0]/0,0.1)", "q:Output[QArray[QBit]]", "", "divisor"),
        ("allocate(2,q)\nphase(q[0]**9,0.1)", "q:Output[QArray[QBit]]", "", "0..8"),
        (
            "allocate(2,q)\nphase(q[0]**p,0.1)",
            "p:int,q:Output[QArray[QBit]]",
            "",
            "literal",
        ),
        (
            "allocate(2,q)\nphase(q[0]//2,0.1)",
            "q:Output[QArray[QBit]]",
            "",
            "literal powers",
        ),
        (
            "allocate(2,q)\nphase(q[0]<<1,0.1)",
            "q:Output[QArray[QBit]]",
            "",
            "literal powers",
        ),
        (
            "allocate(2,q)\nphase(q[0]+boom(),0.1)",
            "q:Output[QArray[QBit]]",
            "def boom():\n    raise AssertionError('host body executed')\n",
            "finite real",
        ),
        (
            "allocate(2,q)\nhelper(q,q)",
            "q:Output[QArray[QBit]]",
            "@qfunc\ndef helper(a:QArray[QBit],b:QArray[QBit]):\n    phase(a[0]-b[0],0.1)\n",
            "disjoint",
        ),
        (
            "allocate(q)\nouter(q)",
            "q:Output[QBit]",
            "@qfunc\ndef helper(a:Output[QBit]):\n    allocate(a)\n    phase(a-0.5,0.1)\n"
            "@qfunc\ndef outer(a:QBit):\n    helper(a)\n",
            "publish Outputs|writable quantum",
        ),
    ],
)
def test_phase_rejects_unsupported_values_without_host_execution(
    body: str, signature: str, prelude: str, diagnostic: str
) -> None:
    entry = definition(body, signature, prelude)
    with pytest.raises(ERRORS, match=diagnostic):
        synth.synthesize(
            entry, classical_inputs={"p": 2} if "p:int" in signature else None
        )


@pytest.mark.parametrize(
    ("count", "expression", "diagnostic"),
    [
        (25, "+".join(f"q[{i}]" for i in range(25)), "24 quantum"),
        (9, "*".join(f"q[{i}]" for i in range(9)), "degree eight"),
        (9, "*".join(f"(1+q[{i}])" for i in range(9)), "256 monomials"),
        (8, "(" + "*".join(f"(1+q[{i}])" for i in range(8)) + ")**2", "20000"),
        (2, "9007199254740992.0*q[0]*2", "coefficient"),
    ],
)
def test_sparse_expansion_budgets_refuse_before_materializing_states(
    count: int, expression: str, diagnostic: str
) -> None:
    entry = definition(f"allocate({count},q)\nphase({expression},0.1)")
    with pytest.raises(ERRORS, match=diagnostic):
        synth.synthesize(entry)


def test_twenty_four_bit_sparse_phase_has_linear_effect_count_without_state_enumeration() -> (
    None
):
    expression = "+".join(f"{i+1}.0*q[{i}]" for i in range(24))
    program = synth.synthesize(
        definition(f"allocate(24,q)\nhadamard_transform(q)\nphase({expression},0.1)"),
        max_workspace=0,
        memory_limit_bytes=2**30,
    )
    assert estimate_resources(program.circuit).n_qubits == 24
    assert estimate_resources(program.circuit).n_operations <= 48
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("budget", ["qubits", "operations", "memory"])
def test_phase_respects_actual_resource_limits(budget: str) -> None:
    with pytest.raises((*ERRORS, PlanningError)):
        synth.synthesize(
            adapt,
            classical_inputs={"params": ADAPT_BINDINGS[0]},
            max_qubits=7 if budget == "qubits" else 8,
            max_operations=1 if budget == "operations" else 100000,
            memory_limit_bytes=16 if budget == "memory" else 2**20,
        )


@pytest.mark.parametrize("expression", ["q-0.5", "0.3*q", "q/2"])
def test_real_phase_extension_does_not_enable_real_fresh_quantum_assignment(
    expression: str,
) -> None:
    entry = definition(
        f"allocate(q)\ny |= {expression}", "q:Output[QBit],y:Output[QNum]"
    )
    with pytest.raises(ERRORS):
        synth.synthesize(entry)


def test_mutable_phase_helper_does_not_escape_existing_adjoint_boundary() -> None:
    entry = definition(
        "allocate(2,q)\nhadamard_transform(q)\nwith adjoint():\n    cost(g,q)",
        "g:float,q:Output[QArray[QBit]]",
        "@qfunc\ndef cost(g:float,q:QArray[QBit]):\n    phase(-0.3+q[0]/2-q[0]*q[1],g)\n",
    )
    with pytest.raises(ERRORS, match="Const-only"):
        synth.synthesize(entry, classical_inputs={"g": -0.37})


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), -float("inf")])
def test_phase_rejects_nonfinite_bound_helper_coefficients(coefficient: float) -> None:
    entry = definition(
        "allocate(2,q)\ncost(g,q)",
        "g:float,q:Output[QArray[QBit]]",
        "@qfunc\ndef cost(g:float,q:QArray[QBit]):\n    phase(q[0]-0.3,g)\n",
    )
    with pytest.raises(ERRORS, match="finite"):
        synth.synthesize(entry, classical_inputs={"g": coefficient})


def test_phase_rejects_final_angle_overflow_after_valid_polynomial_expansion() -> None:
    entry = definition("allocate(2,q)\nphase(9007199254740992.0*q[0],2.0)")
    with pytest.raises(ERRORS, match="phase angle"):
        synth.synthesize(entry)
