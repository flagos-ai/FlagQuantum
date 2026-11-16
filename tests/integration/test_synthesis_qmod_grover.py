"""Checked phase borrowing, static Grover composition and independent originals."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.services.synthesis import _synthesize
from tools.validate_qmod_grover import FIXTURES, validate

pytestmark = pytest.mark.integration

ORACLE = "qperm mark(const q:qbit[]) {control(q[0] & q[2]) {phase(pi);}}"
MAIN = """qfunc main(r:int,output q:qbit[3]) {
 allocate(q);hadamard_transform(q);
 power(r) {grover_operator(lambda(v) {mark(v);},hadamard_transform,q);}
}"""


@pytest.mark.parametrize("case", ["3_sat_grover", "grover_max_cut"])
@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_originals_against_independent_circuit_and_analytic_amplitudes(
    case: str, iterations: int
) -> None:
    report = validate(case, iterations)
    assert report.workspace_nonzero_probability == 0
    assert report.max_abs_state_error < 1e-12


@pytest.mark.parametrize("reference", ["mark", "lambda(v) {mark(v);}"])
def test_named_and_lambda_oracles_preserve_the_full_operator(reference: str) -> None:
    source = (
        ORACLE
        + "qfunc main(output q:qbit[3]) {allocate(q);H(q[0]);RY(0.37,q[2]);grover_operator("
        + reference
        + ",hadamard_transform,q);}"
    )
    program = synth.synthesize(source, source_format="qmod")
    width = program.circuit.n_wires
    reference_circuit = fq.Circuit(width, dtype=torch.complex128).h(2).ry(0, theta=0.37)
    oracle = torch.diag(
        torch.tensor(
            [-1 if word & 5 == 5 else 1 for word in range(8)], dtype=torch.complex128
        )
    )
    diffuser = torch.full((8, 8), 0.25, dtype=torch.complex128) - torch.eye(
        8, dtype=torch.complex128
    )
    reference_circuit.any(0, 1, 2, matrix=oracle).any(0, 1, 2, matrix=diffuser)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference_circuit).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("iterations", [0, 1])
@pytest.mark.parametrize(
    "callback",
    [
        "mark(q);",
        "mark(other);",
        "mark(v[0:2]);",
        "mark({v,v});",
        "allocate(v);",
        "X(v[0]);",
        "out:qbit;",
        "phase(other[0]);",
    ],
)
def test_callbacks_reject_alias_capture_and_escape_even_at_zero_power(
    callback: str, iterations: int
) -> None:
    source = ORACLE + MAIN.replace(
        "output q:qbit[3]", "output q:qbit[3],output other:qbit[3]"
    ).replace("allocate(q);", "allocate(q);allocate(other);").replace(
        "mark(v);", callback
    )
    with pytest.raises(ValueError, match="oracle|scope|phase|callback|function level"):
        _synthesize(source, source_format="qmod", classical_inputs={"r": iterations})


@pytest.mark.parametrize(
    "ports,body",
    [
        ("const q:qbit[],output res:qbit", "phase(pi);"),
        ("const q:qbit[],const alias:qbit[]", "phase(pi);"),
        ("const q:qbit[],n:int,n:int", "phase(pi);"),
        ("const q:qbit[]", "X(q[0]);"),
        ("const q:qbit[]", "q -> escaped;"),
        ("const q:qbit[]", "mark(q);"),
    ],
)
def test_borrowed_definitions_are_checked_even_when_unused(
    ports: str, body: str
) -> None:
    source = (
        f"qfunc mark({ports}) {{{body}}} qfunc main(output q:qbit) {{allocate(q);}}"
    )
    with pytest.raises(ValueError, match="oracle|recursion|scope"):
        synth.synthesize(source, source_format="qmod")


def test_fixed_width_borrow_and_preparation_profile_are_checked() -> None:
    with pytest.raises(ValueError, match="width"):
        _synthesize(
            ORACLE.replace("qbit[]", "qbit[4]") + MAIN,
            source_format="qmod",
            classical_inputs={"r": 0},
        )
    with pytest.raises(ValueError, match="preparation"):
        _synthesize(
            ORACLE + MAIN.replace("hadamard_transform,q", "custom,q"),
            source_format="qmod",
            classical_inputs={"r": 0},
        )


@pytest.mark.parametrize("value", ["1.0", "True", "-1.0", str(1 << 256)])
def test_top_level_integer_constants_do_not_round_or_coerce(value: str) -> None:
    with pytest.raises(ValueError, match="integer|syntax|constant"):
        synth.synthesize(
            "K:int=" + value + ";qfunc main(output q:qbit) {allocate(q);}",
            source_format="qmod",
        )


def test_exact_large_and_negative_integer_constants() -> None:
    source = f"N:int={(1 << 200) + 1};K:int=-2;qfunc main(output q:qbit) {{allocate(q);if(N=={(1 << 200) + 1}) {{X(q);}}if(K < 0) {{H(q);}}}}"
    program = synth.synthesize(source, source_format="qmod")
    expected = fq.Circuit(1, dtype=torch.complex128).x(0).h(0)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(expected).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "constraint,value",
    [
        ("max_qubits", 4),
        ("max_operations", 100),
        ("max_depth", 100),
        ("memory_limit_bytes", 1024),
    ],
)
def test_actual_grover_resource_budgets(constraint: str, value: int) -> None:
    with pytest.raises(
        ValueError,
        match="budget|qubit|operation|depth|memory|limit|candidate|workspace",
    ):
        _synthesize(
            FIXTURES / "grover_max_cut.qmod",
            classical_inputs={"r": 2},
            **{constraint: value},
        )


@pytest.mark.parametrize("value", [-1, 257, True, 1.0])
def test_grover_iteration_bounds(value: object) -> None:
    with pytest.raises(ValueError, match="integer|repeat count"):
        _synthesize(ORACLE + MAIN, source_format="qmod", classical_inputs={"r": value})


def test_expansion_retains_the_global_owned_node_cap() -> None:
    source = ORACLE + MAIN.replace("power(r)", "power(256)")
    with pytest.raises(ValueError, match="4096|budget"):
        _synthesize(source, source_format="qmod", classical_inputs={"r": 0})


def test_phase_borrow_does_not_erase_the_original_predicate_order() -> None:
    source = "qperm mark(const q:qbit[]) {control(q[0] & ~q[2]) {phase(pi);}} qfunc main(output q:qbit[3]) {allocate(q);hadamard_transform(q);mark(q);}"
    program = synth.synthesize(source, source_format="qmod")
    expected = torch.tensor(
        [-1 if word & 1 and not word & 4 else 1 for word in range(8)],
        dtype=torch.complex128,
    ) / math.sqrt(8)
    torch.testing.assert_close(
        fq.run(program).statevector().reshape(-1), expected, atol=1e-12, rtol=0
    )


def test_native_qfunc_qmod_and_circuit_user_example() -> None:
    from examples.synthesis.algorithms.grover.grover.comparisons.circuit import (
        run_example,
    )

    run_example()


@pytest.mark.parametrize("iterations", [0, 1])
@pytest.mark.parametrize("index", ["3", "-1", "q[0]"])
def test_callback_index_checks_precede_zero_iteration_erasure(
    index: str, iterations: int
) -> None:
    source = ORACLE.replace("q[2]", "q[" + index + "]") + MAIN
    with pytest.raises(ValueError, match="oracle|index|expression"):
        _synthesize(source, source_format="qmod", classical_inputs={"r": iterations})


def test_nested_checked_oracle_calls_and_integer_specialization() -> None:
    source = """K:int=1;
      qperm leaf(n:int,const q:qbit[]) {control(q[0] & q[n]) {phase(pi);}}
      qperm mark(const v:qbit[3]) {leaf(K,v);}
      qfunc main(output q:qbit[3]) {allocate(q);hadamard_transform(q);mark(q);}
    """
    program = synth.synthesize(source, source_format="qmod")
    expected = torch.tensor(
        [-1 if word & 3 == 3 else 1 for word in range(8)], dtype=torch.complex128
    ) / math.sqrt(8)
    torch.testing.assert_close(
        fq.run(program).statevector().reshape(-1), expected, atol=1e-12, rtol=0
    )


def test_scratch_free_max_cut_preserves_the_complete_state() -> None:
    from tests.synthesis_workflow_helpers import check_program_equivalence

    path = FIXTURES / "grover_max_cut.qmod"
    unconstrained = _synthesize(path, classical_inputs={"r": 2})
    constrained = _synthesize(
        path, classical_inputs={"r": 2}, max_workspace=0, max_qubits=5
    )
    assert constrained.report["workspace_qubits"] == 0
    check_program_equivalence(unconstrained, constrained)
