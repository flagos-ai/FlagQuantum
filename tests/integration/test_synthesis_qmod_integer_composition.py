"""Exact integer bindings, static composition, ordered views and original states."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from tools.validate_qmod_integer_composition import FIXTURES, validate

pytestmark = pytest.mark.integration
DTYPE = torch.complex128


@pytest.mark.parametrize("iterations", [1, 2])
def test_original_fermi_full_state(iterations: int) -> None:
    report = validate("fermi_hubbard_1D", iterations=iterations)
    assert report.qubits == 8 and report.workspace == 0


@pytest.mark.parametrize("data_case", [0, 1])
def test_original_credit_full_state(data_case: int) -> None:
    report = validate("credit_card_fraud", data_case=data_case)
    assert report.qubits == 3 and report.workspace == 0


def test_original_bb84_full_state() -> None:
    assert validate("bb84_implementation").max_abs_state_error == 0


STATIC = """qfunc prepare(bits:int[],q:qbit[]) {
  repeat(i:bits.len) {if(bits[i]==1) {H(q[i]);}}
}
qfunc main(steps:int,bits:int[3],output q:qbit[3]) {
 allocate(q);prepare(bits,q);power(steps) {X(q[1]);}
}"""


@pytest.mark.parametrize("bits", [(1, 0, 1), (0, 1, 1)])
@pytest.mark.parametrize("steps", [0, 1, 2, 256])
def test_exact_integer_arrays_and_static_loop_composition(
    bits: tuple[int, ...], steps: int
) -> None:
    program = synth.synthesize(
        STATIC, source_format="qmod", classical_inputs={"steps": steps, "bits": bits}
    )
    circuit = fq.Circuit(3, dtype=DTYPE)
    for bit, value in enumerate(bits):
        if value:
            circuit.h(2 - bit)
    for _ in range(steps):
        circuit.x(1)
    torch.testing.assert_close(
        fq.run(program).statevector(), fq.run(circuit).statevector(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("value", [True, 1.0, 1.5, float("nan"), 1 << 256, "1"])
def test_integer_ports_do_not_coerce_reals_booleans_or_overflow(value: object) -> None:
    with pytest.raises(ValueError, match="exact non-Boolean integer"):
        synth.synthesize(
            STATIC,
            source_format="qmod",
            classical_inputs={"steps": value, "bits": [1, 0, 1]},
        )


@pytest.mark.parametrize(
    "bits", [[1, True, 0], [1, 0.0, 1], [1, 0], [], [1, 0, 1, 1], "101"]
)
def test_integer_array_kind_and_length_are_checked(bits: object) -> None:
    with pytest.raises(ValueError, match="integer|elements|length"):
        synth.synthesize(
            STATIC, source_format="qmod", classical_inputs={"steps": 1, "bits": bits}
        )


@pytest.mark.parametrize("steps", [-1, 257])
def test_static_count_caps_are_independent_of_integer_range(steps: int) -> None:
    with pytest.raises(ValueError, match="0..256"):
        synth.synthesize(
            STATIC,
            source_format="qmod",
            classical_inputs={"steps": steps, "bits": [1, 0, 1]},
        )


def test_large_integers_are_compared_exactly_without_float_rounding() -> None:
    large = 1 << 200
    source = f"""qfunc main(n:int,output q:qbit) {{allocate(q);if(n-{large}==1) {{X(q);}}}}"""
    program = synth.synthesize(
        source, source_format="qmod", classical_inputs={"n": large + 1}
    )
    torch.testing.assert_close(
        fq.run(program).statevector().reshape(-1),
        torch.tensor([0, 1], dtype=DTYPE),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("condition", ["q[0]==1", "q==1"])
def test_static_if_cannot_read_a_quantum_condition(condition: str) -> None:
    source = f"""qfunc main(output q:qbit[1]) {{allocate(q);repeat(i:1) {{if({condition}) {{X(q[0]);}}}}}}"""
    with pytest.raises(ValueError, match="static|integer"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("index", [-1, 2])
def test_integer_array_bounds_are_checked(index: int) -> None:
    source = f"""qfunc main(bits:int[2],output q:qbit) {{allocate(q);if(bits[{index}]==1) {{X(q);}}}}"""
    with pytest.raises(ValueError, match="bounds|expression"):
        synth.synthesize(
            source, source_format="qmod", classical_inputs={"bits": [1, 0]}
        )


OP = "SparsePauliOp {terms=[SparsePauliTerm {paulis=[IndexedPauli {pauli=Pauli::X,index=0},IndexedPauli {pauli=Pauli::Y,index=1}],coefficient=1}],num_qubits=2}"


def test_noncontiguous_pauli_view_uses_local_bit_significance_and_no_scratch() -> None:
    source = f"""qfunc twist(q:qbit[2]) {{suzuki_trotter({OP},0.37,1,1,q);}}
    qfunc main(output q:qbit[4]) {{allocate(q);hadamard_transform(q);S(q[3]);twist({{q[0],q[3]}});}}"""
    program = synth.synthesize(source, source_format="qmod", max_workspace=0)
    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
    reference = fq.Circuit(4, dtype=DTYPE)
    for bit in range(4):
        reference.h(bit)
    reference.s(0).any(0, 3, matrix=torch.matrix_exp(-0.37j * torch.kron(y, x)))
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("target", ["{a[0],a[0]}", "{a[0],b[0]}"])
def test_pauli_view_duplicates_and_mixed_owners_are_rejected(target: str) -> None:
    source = f"""qfunc main(output a:qbit[2],output b:qbit[1]) {{allocate(a);allocate(b);suzuki_trotter({OP},0.37,1,1,{target});}}"""
    with pytest.raises(ValueError, match="distinct|owner|alias"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("expression", ["-0.37", "pi", "0.6*q[0]*q[1]"])
def test_phase_default_keeps_absolute_and_relative_phase(expression: str) -> None:
    source = f"""qfunc main(output q:qbit[2]) {{allocate(q);hadamard_transform(q);phase({expression});}}"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(2, dtype=DTYPE).h(0).h(1)
    if expression in {"-0.37", "pi"}:
        angle = -0.37 if expression == "-0.37" else math.pi
        reference.x(0).phase(0, theta=angle).x(0).phase(0, theta=angle)
    else:
        reference.cphase(0, 1, theta=0.6)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


SINGLE_X = "SparsePauliOp {terms=[SparsePauliTerm {paulis=[IndexedPauli {pauli=Pauli::X,index=0}],coefficient=1}],num_qubits=1}"
SINGLE_Z = SINGLE_X.replace("Pauli::X", "Pauli::Z")


def test_noncommuting_multi_operator_order_and_classical_expression_times() -> None:
    source = f"""qfunc model(data:real[2],q:qbit[]) {{multi_suzuki_trotter([{SINGLE_X},{SINGLE_Z}],[data[0]*0.5,-(data[1]+0.1)],1,1,q);}}
    qfunc main(data:real[2],output q:qbit[1]) {{allocate(q);H(q[0]);S(q[0]);power(1) {{model(data,q);}}}}"""
    program = synth.synthesize(
        source, source_format="qmod", classical_inputs={"data": [0.74, 0.11]}
    )
    reference = (
        fq.Circuit(1, dtype=DTYPE).h(0).s(0).rx(0, theta=0.74).rz(0, theta=-0.42)
    )
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("times", "order", "reps"),
    [("[0.1]", "1", "1"), ("[0.1,0.2]", "2", "1"), ("[0.1,0.2]", "1", "2")],
)
def test_multi_operator_arity_and_unimplemented_formulas_fail_closed(
    times: str, order: str, reps: str
) -> None:
    source = f"""qfunc main(output q:qbit[1]) {{allocate(q);multi_suzuki_trotter([{SINGLE_X},{SINGLE_Z}],{times},{order},{reps},q);}}"""
    with pytest.raises(ValueError, match="equal lengths|order 1 and one repetition"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize(
    ("constraint", "value"),
    [
        ("max_qubits", 7),
        ("max_operations", 100),
        ("max_depth", 10),
        ("memory_limit_bytes", 1024),
    ],
)
def test_original_fermi_obeys_resource_budgets(constraint: str, value: int) -> None:
    with pytest.raises(
        ValueError, match="budget|qubit|operation|depth|memory|limit|candidate"
    ):
        synth.synthesize(
            FIXTURES / "fermi_hubbard_1D.qmod",
            classical_inputs={"num_iter": 2},
            **{constraint: value},
        )


def test_static_expansion_cannot_escape_the_owned_operation_budget() -> None:
    source = """qfunc step(q:qbit[]) {power(256) {X(q[0]);}}
      qfunc main(output q:qbit[1]) {allocate(q);power(20) {step(q);}}"""
    with pytest.raises(ValueError, match="4096|budget"):
        synth.synthesize(source, source_format="qmod")


def test_native_qfunc_qmod_and_circuit_user_example() -> None:
    from examples.synthesis_integer_composition import run_example

    run_example()


def test_integer_helper_phase_and_exact_angle_conversion() -> None:
    source = """qfunc mark(n:int,q:qbit) {phase(q,n);}
      qfunc main(n:int,output q:qbit) {allocate(q);H(q);mark(n,q);}"""
    program = synth.synthesize(source, source_format="qmod", classical_inputs={"n": 2})
    expected = fq.Circuit(1, dtype=DTYPE).h(0).phase(0, theta=2.0)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(expected).statevector(),
        atol=1e-12,
        rtol=0,
    )
    with pytest.raises(ValueError, match="exact real range"):
        synth.synthesize(
            source, source_format="qmod", classical_inputs={"n": (1 << 100) + 1}
        )


def test_power_does_not_bypass_the_qperm_basis_proof() -> None:
    source = """qperm bad(n:int,q:qbit) {power(n) {H(q);}}
      qfunc main(output q:qbit) {allocate(q);}"""
    with pytest.raises(ValueError, match="basis preservation"):
        synth.synthesize(source, source_format="qmod")
