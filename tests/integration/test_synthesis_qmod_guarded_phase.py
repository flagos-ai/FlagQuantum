"""Check conditional real phases, shared reads, slice indexing and raw originals."""

from collections.abc import Callable
from math import sqrt

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from tools.validate_qmod_knapsack import BLOBS, FIXTURES, validate

pytestmark = pytest.mark.integration


def reference(
    predicate: Callable[[int, int], bool],
    cost: Callable[[int, int], float],
    angle: float,
    *,
    interfere: bool = False,
) -> fq.Circuit:
    values = torch.tensor(
        [cost(a, b) if predicate(a, b) else 0 for b in range(4) for a in range(8)],
        dtype=torch.float64,
    )
    circuit = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        circuit.h(wire)
    circuit.any(*range(5), matrix=torch.diag(torch.exp(1j * angle * values)))
    if interfere:
        for wire in range(5):
            circuit.h(wire)
    return circuit


@pytest.mark.parametrize(
    ("guard", "predicate"),
    [
        ("v.a<3", lambda a, b: a < 3),
        ("v.a<=3", lambda a, b: a <= 3),
        ("v.a>3", lambda a, b: a > 3),
        ("v.a>=3", lambda a, b: a >= 3),
        ("v.a==3", lambda a, b: a == 3),
        ("v.a!=3", lambda a, b: a != 3),
        ("(v.a<3)&(v.b!=2)", lambda a, b: a < 3 and b != 2),
        ("(v.a==1)|(v.b==2)", lambda a, b: a == 1 or b == 2),
        ("(v.a==1)^(v.b==2)", lambda a, b: (a == 1) != (b == 2)),
        ("~(v.a==1)", lambda a, b: a != 1),
        ("2*v.a+3*v.b<=12", lambda a, b: 2 * a + 3 * b <= 12),
    ],
)
def test_shared_fields_and_integer_predicate_boundaries(
    guard: str, predicate: Callable[[int, int], bool]
) -> None:
    source = f"""qstruct State {{a:qnum<3>;b:qnum<2>;}}
      qfunc kick(v:State, alpha:real) {{control({guard}) {{phase(3+2*v.a-v.b,alpha);}}}}
      qfunc main(alpha:real, output v:State) {{
        allocate(v);hadamard_transform(v);kick(v,alpha);
      }}"""
    program = synth.synthesize(
        source, source_format="qmod", classical_inputs={"alpha": -0.37}, max_workspace=0
    )
    actual = fq.run(program).statevector().reshape(-1)
    expected = (
        fq.run(reference(predicate, lambda a, b: 3 + 2 * a - b, -0.37))
        .statevector()
        .reshape(-1)
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert program.report["workspace_qubits"] == 0
    assert program.report["resources"].n_qubits == 5
    if guard == "2*v.a+3*v.b<=12":
        assert actual[6] == pytest.approx(complex(expected[6]), abs=1e-12)
        assert complex(actual[5 + 8]) == pytest.approx(1 / sqrt(32), abs=1e-12)


def test_conditional_constant_is_a_relative_phase_with_interference() -> None:
    source = """qstruct State {a:qnum<3>;b:qnum<2>;}
      qfunc main(alpha:real,output v:State) {
        allocate(v);hadamard_transform(v);
        control(2*v.a+3*v.b<=12) {phase(1,alpha);}
        hadamard_transform(v);
      }"""
    program = synth.synthesize(
        source, source_format="qmod", classical_inputs={"alpha": 0.83}, max_workspace=0
    )
    result = fq.run(program)
    expected = (
        fq.run(
            reference(
                lambda a, b: 2 * a + 3 * b <= 12, lambda a, b: 1, 0.83, interfere=True
            )
        )
        .statevector()
        .reshape(-1)
    )
    torch.testing.assert_close(
        result.statevector().reshape(-1), expected, atol=1e-12, rtol=0
    )
    assert float(expected[0].abs().square()) < 0.99


def test_scalar_pi_phase_can_share_its_predicate_register() -> None:
    source = """qfunc main(output q:qbit) {
      allocate(q);H(q);control(q==1) {phase(q,pi);} H(q);
    }"""
    program = synth.synthesize(source, source_format="qmod", max_workspace=0)
    assert dict(
        program.probabilities(fq.run(program), registers=("q",))
    ) == pytest.approx({(1,): 1})


def test_slice_element_binding_and_loop_rotation() -> None:
    source = """qfunc rotate(theta:real,q:qbit) {RX(theta,q);}
      qfunc main(params:real[4], output q:qbit) {
        allocate(q);rotate(params[1:3][0],q);
        repeat(i:2) {RX(params[1:3][i],q);}
      }"""
    params = [0.19, -0.37, 0.81, -0.11]
    program = synth.synthesize(
        source, source_format="qmod", classical_inputs={"params": params}
    )
    expected = fq.Circuit(1, dtype=torch.complex128).rx(
        0, theta=2 * params[1] + params[2]
    )
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(expected).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "slice_expression", ["params[1:3][2]", "params[2:2][0]", "params[1:5][0]"]
)
def test_slice_bounds_do_not_escape_to_the_original_array(
    slice_expression: str,
) -> None:
    source = f"qfunc main(params:real[4],output q:qbit) {{allocate(q);RX({slice_expression},q);}}"
    with pytest.raises(ValueError, match="bounds|empty"):
        synth.synthesize(
            source,
            source_format="qmod",
            classical_inputs={"params": [0.1, 0.2, 0.3, 0.4]},
        )


@pytest.mark.parametrize("guard", ["v.a==0", "(v.a-v.a)==0"])
def test_predicate_eight_bit_budget_survives_cancellation(guard: str) -> None:
    source = f"""qstruct State {{a:qnum<9>;}}
      qfunc main(alpha:real,output v:State) {{allocate(v);control({guard}) {{phase(v.a+1,alpha);}}}}"""
    with pytest.raises(ValueError, match="eight input bits"):
        synth.synthesize(source, source_format="qmod", classical_inputs={"alpha": 0.37})


def test_predicate_rejects_noninteger_classical_threshold() -> None:
    source = """qstruct State {a:qnum<2>;}
      qfunc main(alpha:real,limit:real,output v:State) {
        allocate(v);control(v.a<limit) {phase(v.a,alpha);}
      }"""
    with pytest.raises(ValueError, match="integer arithmetic"):
        synth.synthesize(
            source, source_format="qmod", classical_inputs={"alpha": 0.37, "limit": 0.5}
        )


def test_guard_times_cost_preserves_phase_degree_budget() -> None:
    source = """qfunc main(alpha:real,output x:qnum<8>,output y:qbit) {
      allocate(x);allocate(y);control(x==255) {phase(y,alpha);}
    }"""
    with pytest.raises(ValueError, match="degree eight"):
        synth.synthesize(source, source_format="qmod", classical_inputs={"alpha": 0.37})


def test_cancelled_cost_still_checks_initialization() -> None:
    source = """qstruct State {a:qnum<2>;}
      qfunc main(alpha:real,output v:State) {control(v.a==0) {phase(0,alpha);}}
    """
    with pytest.raises(ValueError, match="uninitialized"):
        synth.synthesize(source, source_format="qmod", classical_inputs={"alpha": 0.37})


def test_integer_predicate_normalization_preserves_large_cancellation() -> None:
    source = """qfunc main(alpha:real,output x:qnum<1>) {
      allocate(x);hadamard_transform(x);
      control(((9007199254740991*x-9007199254740991)*(2*x+1))>=0) {
        phase(1,alpha);
      }
    }"""
    program = synth.synthesize(
        source, source_format="qmod", classical_inputs={"alpha": 0.37}
    )
    expected = fq.Circuit(1, dtype=torch.complex128).h(0).phase(0, theta=0.37)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(expected).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("limits", [{"max_qubits": 4}, {"memory_limit_bytes": 1}])
def test_original_guarded_phase_keeps_resource_admission(
    limits: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="qubit|budget|width|state|limit"):
        synth.synthesize(
            FIXTURES / "qaoa.qmod", classical_inputs={"params": [0.01] * 6}, **limits
        )


@pytest.mark.parametrize("case", tuple(BLOBS))
@pytest.mark.parametrize("scale", [0.003, -0.005])
def test_unchanged_knapsack_against_independent_dense_circuit(
    case: str, scale: float
) -> None:
    report = validate(case, scale)
    assert report.max_abs_state_error < 1e-12
    assert report.max_abs_probability_error < 1e-12
    assert report.output_qubits_cover_state and report.workspace == 0
