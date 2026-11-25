"""Verify bounded real phase programs with independent classical costs."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.ecosystem import InteropConversionError
from flagquantum.services.synthesis import _synthesize

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128
CUT_EDGES = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))
CUBE_EDGES = ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7))


def _cut(bits):
    return sum(bits[first] != bits[second] for first, second in CUT_EDGES)


def _adaptive_cost(bits):
    negative_cross = (
        (0.7525, 0, 4),
        (0.8004, 0, 5),
        (0.1924, 2, 4),
        (0.1581, 3, 6),
        (0.9165, 3, 7),
    )
    positive_cross = ((0.9382, 1, 6), (0.2143, 1, 7), (0.9182, 2, 5))
    return (
        2 * sum(bits[first] * bits[second] for first, second in CUBE_EDGES)
        - sum(
            weight * bits[first] * bits[second]
            for weight, first, second in negative_cross
        )
        - 0.1 * sum(bits)
        - 3.4127
        + sum(
            weight * bits[first] * bits[second]
            for weight, first, second in positive_cross
        )
    )


@pytest.mark.parametrize("scale", [0.1, -0.073])
@pytest.mark.parametrize(
    ("name", "blob", "width", "count"),
    [
        ("adapt_qaoa", "a54cf3838bc16711a186884559042fab41d175d9", 8, 2),
        (
            "combinatorial_qmod_workshop_for_maxcut",
            "9524efd66f32bfd436a2f5c529bdc82300c34768",
            5,
            8,
        ),
        ("qaoa_max_cut", "443e8e8117b21375b608640c5cd54b5d34b720f1", 5, 8),
    ],
)
def test_original_qaoa_against_classical_cost_circuit(name, blob, width, count, scale):
    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    params = [scale * (index + 1) * (-1 if index % 2 else 1) for index in range(count)]
    program = _synthesize(source, classical_inputs={"params": params})
    reference = fq.Circuit(width, dtype=DTYPE)
    for wire in range(width):
        reference.h(wire)
    if name == "adapt_qaoa":
        costs = [
            _adaptive_cost(tuple((value >> bit) & 1 for bit in range(width)))
            for value in range(1 << width)
        ]
        layers = [(params[0], 2 * params[1])]
    else:
        factor = 1 / 6 if name == "qaoa_max_cut" else -1
        costs = [
            factor * _cut(tuple((value >> bit) & 1 for bit in range(width)))
            for value in range(1 << width)
        ]
        layers = (
            list(zip(params[:4], params[4:], strict=True))
            if name == "qaoa_max_cut"
            else list(zip(params[::2], params[1::2], strict=True))
        )
    values = torch.tensor(costs, dtype=torch.float64)
    for gamma, beta in layers:
        reference.any(*range(width), matrix=torch.diag(torch.exp(1j * gamma * values)))
        for wire in range(width):
            reference.rx(wire, theta=beta)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


def test_real_phase_preserves_constant_global_phase_and_boolean_powers():
    source = """qfunc layer(gamma:real,q:qbit[]) {
        phase(3.25 + ((q[0] + q[1])**2)/2 - q[0]*q[0], gamma);
      }
      qfunc main(params:real[1],output q:qbit[2]) {
        allocate(q); hadamard_transform(q); layer(params[0],q);
      }"""
    gamma = -0.31
    program = _synthesize(
        source, source_format="qmod", classical_inputs={"params": [gamma]}
    )
    costs = []
    for value in range(4):
        low, high = value & 1, value >> 1
        costs.append(3.25 + (low + high) ** 2 / 2 - low)
    reference = fq.Circuit(2, dtype=DTYPE).h(0).h(1)
    reference.any(
        0,
        1,
        matrix=torch.diag(
            torch.exp(1j * gamma * torch.tensor(costs, dtype=torch.float64))
        ),
    )
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("expression", "width", "diagnostic"),
    [
        ("q[0]/(q[1]+1)", 2, "divisor must be a nonzero classical"),
        ("q[0]/0", 2, "divisor must be a nonzero classical"),
        ("q[2]", 2, "bit index is out of bounds"),
        (
            "+".join(f"q[{bit}]" for bit in range(25)),
            25,
            "exceeds 24 quantum input bits",
        ),
        ("*".join(f"q[{bit}]" for bit in range(9)), 9, "monomial exceeds degree eight"),
        ("(9007199254740992*q[0])**8", 1, "coefficient exceeds finite"),
    ],
)
def test_phase_domain_and_budget_rejections(expression, width, diagnostic):
    source = (
        f"qfunc layer(gamma:real,q:qbit[]) {{phase({expression},gamma);}}"
        f"qfunc main(params:real[1],output q:qbit[{width}]) {{allocate(q); layer(params[0],q);}}"
    )
    with pytest.raises(InteropConversionError, match=diagnostic):
        _synthesize(source, source_format="qmod", classical_inputs={"params": [0.2]})


def test_cancelled_polynomial_still_checks_initialization():
    source = """qfunc layer(g:real,q:qbit[]) {phase(q[0]-q[0],g);}
      qfunc main(p:real[1],output q:qbit[1]) {layer(p[0],q); allocate(q);}"""
    with pytest.raises(InteropConversionError, match="initialized qubits"):
        _synthesize(source, source_format="qmod", classical_inputs={"p": [0.1]})


def test_callback_signature_and_recursion_are_checked():
    for helper, diagnostic in (
        ("qfunc cost(q:qbit[]) {H(q[0]);}", "named"),
        ("qfunc cost(g:real,q:qbit[]) {cost(g,q);}", "acyclic"),
    ):
        source = """
          qfunc run_cost(f:qfunc(real,qbit[]),g:real,q:qbit[]) {f(g,q);}
          qfunc main(p:real[1],output q:qbit[1]) {
            allocate(q); run_cost(cost,p[0],q);
          }"""
        source = helper + source
        with pytest.raises(InteropConversionError, match=diagnostic):
            _synthesize(source, source_format="qmod", classical_inputs={"p": [0.2]})


def test_map_callback_rejects_shadowing():
    source = """qfunc layer(beta:real,q:qbit[]) {
        apply_to_all(lambda(beta) {RX(beta,beta);},q);
      }
      qfunc main(p:real[1],output q:qbit[1]) {allocate(q); layer(p[0],q);}"""
    with pytest.raises(InteropConversionError, match="cannot shadow"):
        _synthesize(source, source_format="qmod", classical_inputs={"p": [0.1]})
