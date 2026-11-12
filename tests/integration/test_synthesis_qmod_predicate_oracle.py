"""Predicate borrowing, marker XOR, original Grover programs and refusal boundaries."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.services.synthesis import _synthesize
from tools.validate_qmod_predicate_oracle import FIXTURES, marked_states, validate

pytestmark = pytest.mark.integration
DTYPE = torch.complex128
RECORD = "qstruct Pair {x:qnum<2>;y:qnum<2>;}"
PREDICATE = "qperm pred(const p:Pair,m:qbit) {m ^= (p.x-p.y)==2;}"


@pytest.mark.parametrize(
    "case,r",
    [
        ("grover_operator", 2),
        ("oracles_workshop", 2),
        ("grover", 0),
        ("grover", 1),
        ("grover", 2),
    ],
)
def test_unchanged_originals_against_full_state_circuits(case: str, r: int) -> None:
    report = validate(case, r)
    assert report.workspace_nonzero_probability == 0


@pytest.mark.parametrize("word", range(16))
@pytest.mark.parametrize("marker", [0, 1])
def test_direct_xor_all_inputs_and_nonzero_markers(word: int, marker: int) -> None:
    gates = "".join(f"X(p[{i}]);" for i in range(4) if word & (1 << i))
    source = (
        RECORD
        + PREDICATE
        + "qfunc main(output p:Pair,output m:qbit) {allocate(p);allocate(m);"
        + gates
        + ("X(m);" if marker else "")
        + "pred(p,m);}"
    )
    program = synth.synthesize(source, source_format="qmod")
    state = fq.run(program).statevector().reshape(-1)
    expected = torch.zeros_like(state)
    scratch = program.circuit.n_wires - 5
    x, y = word % 4, word // 4
    expected[((word << 1) | (marker ^ (x - y == 2))) << scratch] = 1
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("marker_kind", ["qbit", "qnum<1>"])
def test_coherent_marker_xor_keeps_relative_phase_and_clean_scratch(
    marker_kind: str,
) -> None:
    predicate = PREDICATE.replace("m:qbit", "m:" + marker_kind)
    source = (
        RECORD
        + predicate
        + f"qfunc main(output p:Pair,output m:{marker_kind}) {{allocate(p);allocate(m);hadamard_transform(p);hadamard_transform(m);"
        + ("S(m);" if marker_kind == "qbit" else "phase(m,pi/2);")
        + "pred(p,m);}"
    )
    # Use the supported literal coefficient for the integer marker phase.
    source = source.replace("pi/2", str(math.pi / 2))
    program = synth.synthesize(source, source_format="qmod")
    actual = fq.run(program).statevector().reshape(-1)
    expected = torch.zeros_like(actual)
    scratch = program.circuit.n_wires - 5
    for word in range(16):
        for marker in range(2):
            output = marker ^ (word % 4 - word // 4 == 2)
            expected[((word << 1) | output) << scratch] = (
                1j if marker else 1
            ) / math.sqrt(32)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_direct_record_phase_keeps_negative_difference_instead_of_wrapping() -> None:
    source = (
        RECORD
        + PREDICATE
        + "qfunc main(output p:Pair) {allocate(p);hadamard_transform(p);phase_oracle(pred,p);}"
    )
    program = synth.synthesize(source, source_format="qmod")
    state = fq.run(program).statevector().reshape(-1)
    expected = torch.zeros_like(state)
    scratch = program.circuit.n_wires - 4
    expected[:: 1 << scratch] = (
        torch.tensor([-1 if w % 4 - w // 4 == 2 else 1 for w in range(16)], dtype=DTYPE)
        / 4
    )
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


def test_original_graph_marks_greater_equal_four_not_previous_less_equal_four() -> None:
    _, marked = marked_states("grover")
    assert len(marked) == 10
    assert 0 not in marked and 6 in marked


@pytest.mark.parametrize(
    "callback",
    [
        "pred(v,v);",
        "pred(other,m);",
        "pred(v,other);",
        "pred(v[0:2],m);",
        "pred(v,m);X(m);",
        "allocate(m);",
        "out:qbit;",
        "pred(v,q);",
    ],
)
@pytest.mark.parametrize("iterations", [0, 1])
def test_predicate_callbacks_reject_capture_alias_and_escape_at_zero_rounds(
    callback: str, iterations: int
) -> None:
    source = (
        "qperm pred(const a:qbit[2],b:qbit) {b ^= a[0];} qfunc main(r:int,output q:qbit[2],output other:qbit) {allocate(q);allocate(other);power(r) {grover_operator(lambda(x) {phase_oracle(lambda(v,m) {"
        + callback
        + "},x);},hadamard_transform,q);}}"
    )
    with pytest.raises(ValueError, match="predicate|scope|function level"):
        _synthesize(source, source_format="qmod", classical_inputs={"r": iterations})


@pytest.mark.parametrize(
    "body",
    [
        "m ^= m;",
        "m ^= p.x;m ^= p.y;",
        "allocate(m);",
        "X(m);",
        "m -> escaped;",
        "H(m);",
        "m ^= outside;",
    ],
)
def test_unused_predicate_definitions_cannot_escape_or_read_marker(body: str) -> None:
    source = (
        RECORD
        + f"qperm pred(const p:Pair,m:qbit) {{{body}}} qfunc main(output q:qbit) {{allocate(q);}}"
    )
    with pytest.raises(ValueError, match="predicate|permutation"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("kind", ["qnum<2>", "qbit[1]", "qbit[2]"])
def test_marker_contract_is_scalar_and_one_bit(kind: str) -> None:
    source = (
        RECORD
        + PREDICATE.replace("m:qbit", "m:" + kind)
        + "qfunc main(output q:qbit) {allocate(q);}"
    )
    with pytest.raises(ValueError, match="predicate|oracle"):
        synth.synthesize(source, source_format="qmod")


def test_nominal_record_layout_cannot_be_substituted_by_equal_width_record() -> None:
    source = (
        RECORD
        + "qstruct Other {x:qnum<2>;y:qnum<2>;}"
        + PREDICATE
        + "qfunc main(output p:Other) {allocate(p);phase_oracle(pred,p);}"
    )
    with pytest.raises(ValueError, match="nominal record"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("divisor", ["0", "3", "p.y"])
def test_modulo_refuses_zero_non_power_two_and_quantum_divisors(divisor: str) -> None:
    source = (
        RECORD
        + f"qperm pred(const p:Pair,m:qbit) {{m ^= (p.x%{divisor})==1;}} qfunc main(output p:Pair) {{allocate(p);hadamard_transform(p);phase_oracle(pred,p);}}"
    )
    with pytest.raises(ValueError, match="modulo"):
        synth.synthesize(source, source_format="qmod")


def test_predicate_input_budget_refuses_more_than_twelve_decision_bits() -> None:
    source = "qstruct Big {x:qnum<7>;y:qnum<6>;} qperm pred(const p:Big,m:qbit) {m ^= (p.x^p.y)==1;} qfunc main(output p:Big) {allocate(p);phase_oracle(pred,p);}"
    with pytest.raises(ValueError, match="twelve input bits"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_qubits", 4),
        ("max_operations", 400),
        ("max_depth", 300),
        ("memory_limit_bytes", 1024),
    ],
)
def test_actual_original_resource_budgets_fail_closed(key: str, value: int) -> None:
    with pytest.raises(
        ValueError,
        match="budget|qubit|workspace|operation|depth|memory|limit|candidate",
    ):
        _synthesize(FIXTURES / "grover.qmod", classical_inputs={"r": 2}, **{key: value})


@pytest.mark.parametrize("expression", ["p.x", "p.x+1", "p.x & p.y"])
def test_marker_expression_must_prove_one_bit_even_in_unused_definition(
    expression: str,
) -> None:
    source = (
        RECORD
        + f"qperm pred(const p:Pair,m:qbit) {{m ^= {expression};}} qfunc main(output q:qbit) {{allocate(q);}}"
    )
    with pytest.raises(ValueError, match="one-bit Boolean"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("value", [0, 1])
def test_constant_oracles_keep_their_global_phase(value: int) -> None:
    source = (
        RECORD
        + f"qperm pred(const p:Pair,m:qbit) {{m ^= {value};}} qfunc main(output p:Pair) {{allocate(p);hadamard_transform(p);phase_oracle(pred,p);}}"
    )
    p = synth.synthesize(source, source_format="qmod")
    expected = torch.full((16,), (-1 if value else 1) / 4, dtype=DTYPE)
    torch.testing.assert_close(
        fq.run(p).statevector().reshape(-1), expected, atol=1e-12, rtol=0
    )


def test_native_qfunc_qmod_and_circuit_example() -> None:
    from examples.synthesis.algorithms.grover.predicate_oracle.comparisons.circuit import (
        run_example,
    )

    run_example()


def test_callback_formal_shadow_does_not_rename_record_field_identifiers() -> None:
    source = (
        RECORD
        + PREDICATE
        + "qfunc main(output q:Pair) {allocate(q);hadamard_transform(q);phase_oracle(lambda(x,y) {pred(x,y);},q);}"
    )
    program = synth.synthesize(source, source_format="qmod")
    expected = (
        torch.tensor([-1 if w % 4 - w // 4 == 2 else 1 for w in range(16)], dtype=DTYPE)
        / 4
    )
    torch.testing.assert_close(
        fq.run(program).statevector().reshape(-1), expected, atol=1e-12, rtol=0
    )


def test_marker_initialization_is_still_checked_by_compiler() -> None:
    source = (
        RECORD
        + PREDICATE
        + "qfunc main(output p:Pair,output m:qbit) {allocate(p);pred(p,m);}"
    )
    with pytest.raises(ValueError, match="initializ"):
        synth.synthesize(source, source_format="qmod")


def test_marker_cannot_alias_a_borrowed_array_element() -> None:
    source = "qperm pred(const q:qbit[2],m:qbit) {m ^= q[0];} qfunc main(output q:qbit[2]) {allocate(q);pred(q,q[0]);}"
    with pytest.raises(ValueError, match="predicate.alias"):
        synth.synthesize(source, source_format="qmod")


def test_unsized_marker_port_specializes_to_the_one_bit_phase_context() -> None:
    source = (
        RECORD
        + PREDICATE.replace("m:qbit", "m:qnum")
        + "qfunc main(output p:Pair) {allocate(p);hadamard_transform(p);phase_oracle(pred,p);}"
    )
    p = synth.synthesize(source, source_format="qmod")
    expected = (
        torch.tensor([-1 if w % 4 - w // 4 == 2 else 1 for w in range(16)], dtype=DTYPE)
        / 4
    )
    torch.testing.assert_close(
        fq.run(p).statevector().reshape(-1), expected, atol=1e-12, rtol=0
    )


def test_grover_callback_field_name_is_not_a_captured_variable() -> None:
    from tools.validate_qmod_predicate_oracle import reference_circuit

    source = (
        RECORD
        + PREDICATE
        + "qfunc main(output q:Pair) {allocate(q);hadamard_transform(q);grover_operator(lambda(x) {phase_oracle(pred,x);},hadamard_transform,q);}"
    )
    p = synth.synthesize(source, source_format="qmod")
    expected = fq.run(
        reference_circuit("oracles_workshop", 1, p.circuit.n_wires - 4)
    ).statevector()
    torch.testing.assert_close(fq.run(p).statevector(), expected, atol=1e-12, rtol=0)


def test_scratch_free_grover_preserves_the_complete_state() -> None:
    from tests.synthesis_workflow_helpers import check_program_equivalence

    unconstrained = _synthesize(FIXTURES / "grover.qmod", classical_inputs={"r": 2})
    constrained = _synthesize(
        FIXTURES / "grover.qmod",
        classical_inputs={"r": 2},
        max_workspace=0,
        max_qubits=5,
    )
    assert constrained.report["workspace_qubits"] == 0
    check_program_equivalence(unconstrained, constrained)
