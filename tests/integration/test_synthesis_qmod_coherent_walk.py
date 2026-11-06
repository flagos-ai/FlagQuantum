"""Verify coherent reversible helpers against Fourier shifts and permutations."""

import hashlib
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


def _fourier(circuit, width, inverse=False):
    if inverse:
        for wire in range(width // 2):
            circuit.swap(wire, width - 1 - wire)
        for target in reversed(range(width)):
            for control in reversed(range(target + 1, width)):
                circuit.cphase(
                    control, target, theta=-math.pi / 2 ** (control - target)
                )
            circuit.h(target)
    else:
        for target in range(width):
            circuit.h(target)
            for control in range(target + 1, width):
                circuit.cphase(control, target, theta=math.pi / 2 ** (control - target))
        for wire in range(width // 2):
            circuit.swap(wire, width - 1 - wire)


def _reference(width, total, position=0, coherent=False):
    circuit = fq.Circuit(total, dtype=DTYPE)
    for wire in range(width):
        if position & (1 << (width - 1 - wire)):
            circuit.x(wire)
    if coherent:
        circuit.h(0).s(0).h(width).s(width)
    circuit.h(width)
    _fourier(circuit, width)
    for wire in range(width):
        angle = math.tau / 2 ** (wire + 1)
        circuit.phase(wire, theta=angle)
        circuit.cphase(width, wire, theta=-2 * angle)
    _fourier(circuit, width, inverse=True)
    return circuit


@pytest.mark.parametrize(
    ("name", "blob", "width", "workspace"),
    [
        (
            "classiq_discrete_quantum_walk",
            "fa16163c96db9462af9e9e1b21cee28811103776",
            6,
            3,
        ),
        (
            "discrete_quantum_walk_circle",
            "899de47fed6104329c0afa8ef0bb5b606dc23af1",
            11,
            5,
        ),
    ],
)
def test_original_walk_full_state_and_clean_workspace(name, blob, width, workspace):
    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    program = synth.synthesize(source)
    state = fq.run(program).statevector()
    reference = _reference(width, program.circuit.n_wires)
    torch.testing.assert_close(
        state, fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    expected = torch.zeros_like(state)
    expected[0, (1 << (workspace + 1))] = 2**-0.5
    expected[0, (((1 << width) - 1) * 2 + 1) << workspace] = 2**-0.5
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    assert program.report["workspace_qubits"] == workspace
    assert torch.count_nonzero(state.reshape(-1, 1 << workspace)[:, 1:]) == 0


@pytest.mark.parametrize("position", [1, 7])
@pytest.mark.parametrize("coherent", [False, True])
def test_nonzero_positions_coherence_and_wraparound(position, coherent):
    source = (FIXTURES / "discrete_quantum_walk_circle.qmod").read_text()
    source = source.replace("qnum<11>", "qnum<3>")
    preparation = "".join(f"X(x[{bit}]);" for bit in range(3) if position & (1 << bit))
    if coherent:
        # x[2] is the highest-order position bit.
        preparation += "H(x[2]); S(x[2]); H(coin); S(coin);"
    source = (
        "qfunc prepare(x:qbit[],coin:qbit) {"
        + preparation
        + "}"
        + source.replace(
            "single_step_walk(coin, x);", "prepare(x,coin); single_step_walk(coin, x);"
        )
    )
    program = synth.synthesize(source, source_format="qmod")
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(
            _reference(3, program.circuit.n_wires, position, coherent)
        ).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_control_rejects_aliases_across_helper_boundary():
    source = """qfunc flip(x:qbit[]) {X(x[0]);}
        qfunc main(output coin:qbit) {allocate(coin); control(coin==1) {flip({coin});}}"""
    with pytest.raises(InteropConversionError, match="distinct qubit locations"):
        synth.synthesize(source, source_format="qmod")


def test_walk_workspace_budget_is_enforced():
    with pytest.raises(ValueError, match="workspace|budget"):
        synth.synthesize(
            FIXTURES / "classiq_discrete_quantum_walk.qmod", max_workspace=2
        )


def test_nested_inverse_of_helper_preserves_gate_phase():
    source = """qfunc step(x:qbit[]) {S(x[0]); X(x[0]);}
      qfunc main(output q:qbit[]) {
        allocate(1,q); H(q[0]); invert {invert {step(q);}}
      }"""
    reference = fq.Circuit(1, dtype=DTYPE).h(0).s(0).x(0)
    program = synth.synthesize(source, source_format="qmod")
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("allocate(x); control(x[0:2]) {X(x[1]);}", "distinct qubit locations"),
        ("control(x[0:2]) {X(x[2]);}", "initialized qubits"),
    ],
)
def test_slice_control_checks_storage(body, diagnostic):
    source = f"qfunc main(output x:qbit[3]) {{ {body} }}"
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")


def test_controlled_hadamard_helper_preserves_coherent_phase():
    source = """qfunc step(x:qbit[]) {H(x[0]);}
      qfunc main(output x:qbit[1],output coin:qbit) {
        allocate(x); allocate(coin); H(coin); S(coin);
        control(coin==1) {step(x);}
      }"""
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
    matrix = torch.block_diag(torch.eye(2, dtype=DTYPE), hadamard)
    reference = fq.Circuit(2, dtype=DTYPE).h(1).s(1).any(1, 0, matrix=matrix)
    program = synth.synthesize(source, source_format="qmod")
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_controlled_s_helper_preserves_coherent_phase():
    source = """qfunc step(x:qbit[]) {S(x[0]);}
      qfunc main(output x:qbit[1],output coin:qbit) {
        allocate(x); allocate(coin); H(x[0]); H(coin);
        control(coin==1) {step(x);}
      }"""
    reference = fq.Circuit(2, dtype=DTYPE).h(0).h(1).cphase(1, 0, theta=math.pi / 2)
    program = synth.synthesize(source, source_format="qmod")
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_controlled_helper_refuses_non_admitted_reversible_gates():
    source = """qfunc step(x:qbit[]) {SX(x[0]);}
      qfunc main(output x:qbit[1],output coin:qbit) {
        allocate(x); allocate(coin); control(coin==1) {step(x);}
      }"""
    with pytest.raises(InteropConversionError, match="reversible X"):
        synth.synthesize(source, source_format="qmod")
