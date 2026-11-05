"""Pure Pauli control composition, angle semantics and original acceptance."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from tools.validate_qmod_controlled_evolution import FIXTURES, validate

pytestmark = pytest.mark.integration
DTYPE = torch.complex128


def operator(axes: str, coefficient: float = 0.37) -> str:
    terms = ",".join(
        f"IndexedPauli {{pauli=Pauli::{axis},index={i}}}"
        for i, axis in enumerate(axes)
        if axis != "I"
    )
    return f"SparsePauliOp {{terms=[SparsePauliTerm {{paulis=[{terms}],coefficient={coefficient}}}],num_qubits={len(axes)}}}"


@pytest.mark.parametrize("delta", [0.37, -0.81])
def test_unchanged_original_full_vectors(delta: float) -> None:
    assert validate(delta)["validation"] == "passed"


@pytest.mark.parametrize("axes", ["I", "X", "Y", "Z", "XY", "YI"])
@pytest.mark.parametrize("condition,value", [("c", 1), ("c==0", 0), ("c!=1", 0)])
def test_coherent_branches_numeric_order_and_absolute_phase(
    axes: str, condition: str, value: int
) -> None:
    n = len(axes)
    op = operator(axes)
    t = 0.43
    source = f"qfunc main(output c:qbit,output q:qbit[{n}]) {{allocate(c);allocate(q);H(c);S(c);hadamard_transform(q);S(q[0]);control({condition}) {{suzuki_trotter({op},{t},2,3,q);}} else {{exponentiate({op},-0.29,q);}}}}"
    program = synth.synthesize(source, source_format="qmod", max_workspace=0)
    matrices = {
        "I": torch.eye(2, dtype=DTYPE),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=DTYPE),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE),
        "Z": torch.diag(torch.tensor([1.0, -1.0], dtype=DTYPE)),
    }
    p = torch.ones((1, 1), dtype=DTYPE)
    for axis in reversed(axes):
        p = torch.kron(p, matrices[axis])
    u_true = torch.matrix_exp(-1j * t * 0.37 * p)
    u_false = torch.matrix_exp(1j * 0.29 * 0.37 * p)
    size = 1 << n
    matrix = torch.zeros((2 * size, 2 * size), dtype=DTYPE)
    for bit in (0, 1):
        matrix[bit * size : (bit + 1) * size, bit * size : (bit + 1) * size] = (
            u_true if bit == value else u_false
        )
    c = fq.Circuit(n + 1, dtype=DTYPE).h(0).s(0)
    for bit in range(1, n + 1):
        c.h(bit)
    c.s(n).any(*range(n + 1), matrix=matrix)
    torch.testing.assert_close(
        fq.run(program).statevector().reshape(-1),
        fq.run(c).statevector().reshape(-1),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("gate", ["PHASE", "RX", "RY", "RZ"])
@pytest.mark.parametrize(
    "expression,value",
    [
        ("-theta + 0.25", -0.12),
        ("1.2+theta*4.0", 2.68),
        ("-1.5*theta+0.2", -0.355),
        ("theta * 4.0", 1.48),
        ("pi/3", math.pi / 3),
        ("(theta+pi)/2", (0.37 + math.pi) / 2),
        ("1/(theta+1)", 1 / 1.37),
    ],
)
def test_classical_angle_arithmetic_without_quantum_workspace(
    gate: str, expression: str, value: float
) -> None:
    source = f"qfunc main(theta:real,output q:qbit) {{allocate(q);H(q);{gate}({expression},q);}}"
    p = synth.synthesize(
        source,
        source_format="qmod",
        classical_inputs={"theta": 0.37},
        max_workspace=0,
        max_qubits=1,
    )
    c = fq.Circuit(1, dtype=DTYPE).h(0)
    getattr(c, gate.lower())(0, theta=value)
    torch.testing.assert_close(
        fq.run(p).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "expression", ["theta/(theta-theta)", "q+1", "missing*4", "theta*9007199254740992"]
)
def test_invalid_angle_expressions_refuse(expression: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            f"qfunc main(theta:real,output q:qbit) {{allocate(q);PHASE({expression},q);}}",
            source_format="qmod",
            classical_inputs={"theta": 2.0},
        )


@pytest.mark.parametrize(
    "body",
    [
        f'suzuki_trotter({operator("X")},0.3,2,2,q);H(q[0]);',
        f'allocate(q);suzuki_trotter({operator("X")},0.3,2,2,q);',
    ],
)
def test_mixed_or_allocating_branches_refuse(body: str) -> None:
    with pytest.raises(ValueError, match="control.evolution|initialized"):
        synth.synthesize(
            f"qfunc main(output c:qbit,output q:qbit[1]) {{allocate(c);allocate(q);control(c) {{{body}}}}}",
            source_format="qmod",
        )


def test_nested_control_is_not_silently_overwritten() -> None:
    with pytest.raises(ValueError, match="previously uncontrolled|nested"):
        synth.synthesize(
            f'qfunc main(output a:qbit,output b:qbit,output q:qbit[1]) {{allocate(a);allocate(b);allocate(q);control(a) {{control(b) {{suzuki_trotter({operator("X")},0.3,2,2,q);}}}}}}',
            source_format="qmod",
        )


@pytest.mark.parametrize("time", ["0.0", "0.3"])
def test_whole_target_overlap_refuses_even_at_zero_time(time: str) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        synth.synthesize(
            f'qfunc main(output q:qbit[1]) {{allocate(q);control(q[0]) {{suzuki_trotter({operator("I")},{time},2,2,q);}}}}',
            source_format="qmod",
        )


@pytest.mark.parametrize("prefix", ["allocate(c);", "allocate(q);"])
def test_uninitialized_control_or_target_refuses(prefix: str) -> None:
    with pytest.raises(ValueError, match="uninitialized"):
        synth.synthesize(
            f'qfunc main(output c:qbit,output q:qbit[1]) {{{prefix}control(c) {{exponentiate({operator("X")},0.0,q);}}}}',
            source_format="qmod",
        )


def test_numeric_one_bit_control_and_bound_static_helper() -> None:
    source = f'qfunc evolve(t:real,q:qbit[]) {{suzuki_trotter({operator("Y")},t/2,2,3,q);}} qfunc main(output c:qnum<1>,output q:qbit[1]) {{allocate(c);allocate(q);hadamard_transform(c);control(c==0) {{evolve(0.8,q);}}}}'
    p = synth.synthesize(source, source_format="qmod")
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
    matrix = torch.eye(4, dtype=DTYPE)
    matrix[:2, :2] = torch.matrix_exp(-1j * 0.4 * 0.37 * y)
    c = fq.Circuit(2, dtype=DTYPE).h(0).any(0, 1, matrix=matrix)
    torch.testing.assert_close(
        fq.run(p).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_qubits", 1),
        ("max_operations", 1),
        ("max_depth", 1),
        ("memory_limit_bytes", 16),
    ],
)
def test_original_actual_resource_budgets_refuse(key: str, value: int) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            FIXTURES / "bpde.qmod", classical_inputs={"delta_e": 0.37}, **{key: value}
        )


@pytest.mark.parametrize("through_helper", [False, True])
def test_signed_control_is_not_reinterpreted_as_unsigned_across_calls(
    through_helper: bool,
) -> None:
    body = f'control(c) {{suzuki_trotter({operator("X")},0.3,2,2,q);}}'
    helper = f"qfunc evolve(c:qnum,q:qbit[]) {{{body}}}" if through_helper else ""
    call = "evolve(c,q);" if through_helper else body
    source = (
        helper
        + f"qfunc main(output c:qnum<1>,output q:qbit[1]) {{allocate(1,True,0,c);allocate(q);{call}}}"
    )
    with pytest.raises(ValueError, match="unsigned"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("time", [0.0, 0.23])
def test_qdrift_identity_read_and_fixed_single_term_control(time: float) -> None:
    source = f'qfunc main(output c:qbit,output q:qbit[1]) {{allocate(c);allocate(q);H(c);control(c==0) {{qdrift({operator("Y")},{time},5,q);}}}}'
    p = synth.synthesize(source, source_format="qmod", max_workspace=0)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
    matrix = torch.eye(4, dtype=DTYPE)
    matrix[:2, :2] = torch.matrix_exp(-1j * time * 0.37 * y)
    c = fq.Circuit(2, dtype=DTYPE).h(0).any(0, 1, matrix=matrix)
    torch.testing.assert_close(
        fq.run(p).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )


def test_indexed_control_keeps_array_numeric_order() -> None:
    source = f'qfunc main(output c:qbit[2],output q:qbit[1]) {{allocate(c);allocate(q);hadamard_transform(c);control(c[0]==0) {{exponentiate({operator("Y")},0.4,q);}}}}'
    p = synth.synthesize(source, source_format="qmod", max_workspace=0)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
    matrix = torch.eye(4, dtype=DTYPE)
    matrix[:2, :2] = torch.matrix_exp(-1j * 0.4 * 0.37 * y)
    c = fq.Circuit(3, dtype=DTYPE).h(0).h(1).any(1, 2, matrix=matrix)
    torch.testing.assert_close(
        fq.run(p).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )


def test_inverse_static_helper_reuses_checked_pauli_inverse() -> None:
    source = f'qfunc evolve(q:qbit[]) {{exponentiate({operator("X")},0.4,q);}} qfunc main(output c:qbit,output q:qbit[1]) {{allocate(c);allocate(q);H(c);control(c) {{invert {{evolve(q);}}}}}}'
    p = synth.synthesize(source, source_format="qmod")
    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    matrix = torch.eye(4, dtype=DTYPE)
    matrix[2:, 2:] = torch.matrix_exp(1j * 0.4 * 0.37 * x)
    c = fq.Circuit(2, dtype=DTYPE).h(0).any(0, 1, matrix=matrix)
    torch.testing.assert_close(
        fq.run(p).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )


def test_both_branches_consume_the_original_effect_budget() -> None:
    terms = ",".join(
        f"SparsePauliTerm {{paulis=[IndexedPauli {{pauli=Pauli::{axis},index=0}}],coefficient={weight}}}"
        for axis, weight in [("X", 0.1), ("Y", 0.2), ("Z", 0.3), ("X", 0.4), ("Y", 0.5)]
    )
    op = f"SparsePauliOp {{terms=[{terms}],num_qubits=1}}"
    source = f"qfunc main(output c:qbit,output q:qbit[1]) {{allocate(c);allocate(q);control(c) {{suzuki_trotter({op},0.4,2,256,q);}} else {{suzuki_trotter({op},0.4,2,256,q);}}}}"
    with pytest.raises(ValueError, match="budget.expansion"):
        synth.synthesize(source, source_format="qmod")


def test_controlled_qdrift_preserves_fixed_sampling_sequence() -> None:
    import random

    coefficients = [0.5, -0.3, 0.2]
    axes = ["I", "X", "Z"]
    terms = ",".join(
        f'SparsePauliTerm {{paulis=[{("IndexedPauli {pauli=Pauli::"+axis+",index=0}") if axis!="I" else ""}],coefficient={weight}}}'
        for axis, weight in zip(axes, coefficients, strict=True)
    )
    op = f"SparsePauliOp {{terms=[{terms}],num_qubits=1}}"
    text = f"qfunc main(output c:qbit,output q:qbit[1]) {{allocate(c);allocate(q);H(c);S(c);H(q[0]);control(c) {{qdrift({op},0.4,7,q);}}}}"
    matrices = [
        torch.eye(2, dtype=DTYPE),
        torch.tensor([[0, 1], [1, 0]], dtype=DTYPE),
        torch.diag(torch.tensor([1.0, -1.0], dtype=DTYPE)),
    ]
    rng = random.Random(0)
    unitary = torch.eye(2, dtype=DTYPE)
    for _ in range(7):
        sample = rng.random()
        chosen = 0 if sample < 0.5 else 1 if sample < 0.8 else 2
        sign = -1 if coefficients[chosen] < 0 else 1
        unitary = torch.matrix_exp(-1j * sign * (0.4 / 7) * matrices[chosen]) @ unitary
    matrix = torch.eye(4, dtype=DTYPE)
    matrix[2:, 2:] = unitary
    reference = fq.Circuit(2, dtype=DTYPE).h(0).s(0).h(1).any(0, 1, matrix=matrix)
    program = synth.synthesize(text, source_format="qmod")
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_branch_output_escape_refuses() -> None:
    text = f'qfunc fresh(output q:qbit[1]) {{allocate(q);exponentiate({operator("X")},0.3,q);}} qfunc main(output c:qbit,output q:qbit[1]) {{allocate(c);control(c) {{fresh(q);}}}}'
    with pytest.raises(ValueError, match="control.evolution"):
        synth.synthesize(text, source_format="qmod")


def test_helper_cannot_capture_another_quantum_register() -> None:
    text = f'qfunc evolve(q:qbit[]) {{exponentiate({operator("X")},0.3,foreign);}} qfunc main(output c:qbit,output q:qbit[1],output foreign:qbit[1]) {{allocate(c);allocate(q);allocate(foreign);control(c) {{evolve(q);}}}}'
    with pytest.raises(
        ValueError, match="expected an unsigned quantum register or view"
    ):
        synth.synthesize(text, source_format="qmod")


def test_all_u_angles_accept_classical_expressions_with_global_phase() -> None:
    text = "qfunc main(theta:real,output q:qbit) {allocate(q);H(q);U(theta*2,pi/3,-theta+0.2,pi/7,q);}"
    program = synth.synthesize(
        text, source_format="qmod", classical_inputs={"theta": 0.37}
    )
    c = fq.Circuit(1, dtype=DTYPE).h(0).u3(0, theta=0.74, phi=math.pi / 3, lbd=-0.17)
    c.x(0).phase(0, theta=math.pi / 7).x(0).phase(0, theta=math.pi / 7)
    torch.testing.assert_close(
        fq.run(program).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )


def test_positive_const_control_borrows_without_changing_storage() -> None:
    text = f'qfunc act(const c:qbit,output q:qbit[1]) {{allocate(q);control(c) {{exponentiate({operator("X")},0.4,q);}}}} qfunc main(output c:qbit,output q:qbit[1]) {{allocate(c);H(c);act(c,q);}}'
    program = synth.synthesize(text, source_format="qmod")
    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    matrix = torch.eye(4, dtype=DTYPE)
    matrix[2:, 2:] = torch.matrix_exp(-1j * 0.4 * 0.37 * x)
    c = fq.Circuit(2, dtype=DTYPE).h(0).any(0, 1, matrix=matrix)
    torch.testing.assert_close(
        fq.run(program).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )


def test_negative_const_control_keeps_existing_mutation_refusal() -> None:
    text = f'qfunc act(const c:qbit,output q:qbit[1]) {{allocate(q);control(c==0) {{exponentiate({operator("X")},0.4,q);}}}} qfunc main(output c:qbit,output q:qbit[1]) {{allocate(c);H(c);act(c,q);}}'
    with pytest.raises(ValueError, match="const|Const"):
        synth.synthesize(text, source_format="qmod")
