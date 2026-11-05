"""Check qperm proof boundaries, Output arrays and singleton Pauli evolution."""

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem._qmod.syntax import Parser, tokenize
from tests.integration.test_synthesis_qmod_coherent_walk import _reference
from tools.validate_qmod_permutations import FIXTURES, validate

pytestmark = pytest.mark.integration
DTYPE = torch.complex128


def test_permutation_marker_and_output_array_identity_survive_parsing() -> None:
    functions, _ = Parser(
        tokenize("qperm fill(output q:qbit[2]) {allocate(q);X(q[0]);}", "test.qmod")
    ).parse()
    assert functions[0].permutation
    assert functions[0].parameters[0].kind == "qbit[]"
    assert functions[0].parameters[0].width == 2


def test_qperm_fresh_output_array_and_diagonal_phase() -> None:
    source = """qfunc basis(q:qbit[]) {X(q[0]);RZ(0.37,q[1]);}
      qperm fill(output q:qbit[2]) {allocate(q);basis(q);}
      qfunc main(output q:qbit[]) {fill(q);}"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(2, dtype=DTYPE).x(1).rz(0, theta=0.37)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert dict(program.registers) == {"q": (0, 1)}
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    "definition",
    [
        "qperm bad(q:qbit) {H(q);}",
        "qfunc hidden(q:qbit) {H(q);} qperm bad(q:qbit) {hidden(q);}",
        "qfunc Y(q:qbit) {H(q);} qperm bad(q:qbit) {Y(q);}",
        "qperm bad(q:qbit) {invert {H(q);}}",
        "qperm bad(q:qbit) {within {H(q);} apply {Z(q);}}",
        "qperm bad(q:qbit) {control(q==1) {H(q);}}",
        "qperm bad(q:qbit) {apply_to_all(lambda(t) {H(t);},q);}",
        "qperm bad(q:qbit) {RY(0.37,q);}",
        "qperm bad(q:qbit) {drop(q);}",
    ],
)
def test_qperm_rejects_unproved_operations_even_in_unused_helpers(
    definition: str,
) -> None:
    with pytest.raises(ValueError, match="qperm.*basis preservation"):
        synth.synthesize(
            definition + " qfunc main(output q:qbit) {allocate(q);}",
            source_format="qmod",
        )


def test_qperm_call_graph_recursion_is_rejected() -> None:
    source = "qperm bad(q:qbit) {bad(q);} qfunc main(output q:qbit) {allocate(q);}"
    with pytest.raises(ValueError, match="acyclic"):
        synth.synthesize(source, source_format="qmod")


def test_callback_shadowing_cannot_bypass_the_qperm_proof() -> None:
    source = """qfunc cb(angle:real,q:qbit[]) {X(q[0]);}
      qfunc superpose(angle:real,q:qbit[]) {H(q[0]);}
      qperm invoke(cb:qfunc(real,qbit[]),q:qbit[]) {cb(0,q);}
      qfunc main(output q:qbit[1]) {allocate(q);invoke(superpose,q);}
    """
    with pytest.raises(ValueError, match="dynamic callback"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize(
    "caller",
    [
        "qfunc main(output q:qbit) {fill(q);}",
        "qfunc main(output q:qbit[3]) {fill(q);}",
        "qfunc main(output q:qbit[2]) {fill(q);fill(q);}",
        "qfunc main(output q:qbit[2]) {allocate(q);fill(q);}",
    ],
)
def test_output_array_helper_keeps_type_width_and_freshness(caller: str) -> None:
    source = "qperm fill(output q:qbit[2]) {allocate(q);X(q[0]);}" + caller
    with pytest.raises(ValueError, match="kind|width|initialized|fresh|fit"):
        synth.synthesize(source, source_format="qmod")


def test_output_arrays_cannot_alias() -> None:
    source = """qperm fill(output a:qbit[2],output b:qbit[2]) {allocate(a);allocate(b);}
      qfunc main(output q:qbit[2]) {fill(q,q);}"""
    with pytest.raises(ValueError, match="distinct|alias|disjoint|unique"):
        synth.synthesize(source, source_format="qmod")


def test_one_bit_array_output_does_not_become_a_scalar() -> None:
    source = """qperm fill(output q:qbit[1]) {allocate(q);X(q[0]);}
      qfunc main(output q:qbit) {fill(q);}"""
    with pytest.raises(ValueError, match="kind/width"):
        synth.synthesize(source, source_format="qmod")


def test_output_array_cannot_escape_through_drop() -> None:
    source = """qfunc fill(output q:qbit[2]) {allocate(q);drop(q);}
      qfunc main(output q:qbit[2]) {fill(q);}"""
    with pytest.raises(ValueError, match="drop|local|output"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("position", range(8))
@pytest.mark.parametrize("coherent", [False, True])
def test_qperm_walk_wraparound_coherence_and_clean_workspace(
    position: int, coherent: bool
) -> None:
    prepare = "".join(f"X(x[{bit}]);" for bit in range(3) if position & (1 << bit))
    if coherent:
        prepare += "H(x[2]);S(x[2]);H(coin);S(coin);"
    source = f"""qperm step(x:qnum) {{x+=1;}}
      qfunc move(coin:qbit,x:qnum) {{
        H(coin);control(coin==0) {{step(x);}}
        control(coin==1) {{invert {{step(x);}}}}
      }}
      qfunc main(output x:qbit[3],output coin:qbit) {{
        allocate(x);allocate(coin);{prepare}move(coin,x);
      }}"""
    program = synth.synthesize(source, source_format="qmod")
    state = fq.run(program).statevector()
    expected = fq.run(
        _reference(3, program.circuit.n_wires, position, coherent)
    ).statevector()
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    workspace = program.report["workspace_qubits"]
    assert float(state.reshape(-1, 1 << workspace)[:, 1:].abs().square().sum()) <= 1e-24


@pytest.mark.parametrize("case", ["molecular_energy_curve", "molecule_eigensolver"])
@pytest.mark.parametrize("parameter", [0.37, -0.81])
def test_original_molecular_qperm_arrays_against_direct_circuit(
    case: str, parameter: float
) -> None:
    report = validate(case, parameter)
    assert report.qubits == 1 and report.workspace == 0


def test_original_qperm_quantum_walk() -> None:
    report = validate("discrete_quantum_walk")
    assert report.qubits == 16 and report.workspace == 5


def test_qperm_walk_workspace_and_memory_budgets() -> None:
    with pytest.raises(ValueError, match="workspace|budget|qubit"):
        synth.synthesize(FIXTURES / "discrete_quantum_walk.qmod", max_workspace=0)
    with pytest.raises(ValueError, match="state|memory|limit"):
        synth.synthesize(
            FIXTURES / "discrete_quantum_walk.qmod", memory_limit_bytes=1024
        )


OP = "SparsePauliOp {terms=[SparsePauliTerm {paulis=[IndexedPauli {pauli=Pauli::Y,index=0}],coefficient=-0.5}],num_qubits=1}"


@pytest.mark.parametrize(
    ("operators", "times", "order", "repetitions", "diagnostic"),
    [
        (f"[{OP},{OP}]", "[0.37]", "1", "1", "equal lengths"),
        (f"[{OP}]", "[0.37,0.81]", "1", "1", "equal lengths"),
        (f"[{OP}]", "[0.37]", "3", "1", "order 1 or 2"),
        (f"[{OP}]", "[0.37]", "1", "0", "repetitions"),
    ],
)
def test_singleton_multi_suzuki_trotter_rejections(
    operators: str, times: str, order: str, repetitions: str, diagnostic: str
) -> None:
    source = f"qfunc main(output q:qbit[1]) {{allocate(q);multi_suzuki_trotter({operators},{times},{order},{repetitions},q);}}"
    with pytest.raises(ValueError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")


def test_singleton_evolution_literal_time_and_second_order_reuse() -> None:
    source = f"qfunc main(output q:qbit[1]) {{allocate(q);X(q[0]);multi_suzuki_trotter([{OP}],[-0.81],2,3,q);}}"
    program = synth.synthesize(source, source_format="qmod")
    expected = fq.Circuit(1, dtype=DTYPE).x(0).ry(0, theta=0.81)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(expected).statevector(),
        atol=1e-12,
        rtol=0,
    )
