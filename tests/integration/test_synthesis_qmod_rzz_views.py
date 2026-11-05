"""Check Qmod static RZZ views and their parameterized original circuits."""

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


@pytest.mark.parametrize("scale", [1, -0.63])
@pytest.mark.parametrize(
    ("name", "blob", "weight_count"),
    [
        (
            "hybrid_qnn_for_subset_majority",
            "7c90600dc38a1edc6b0ef236d90e28707253c1d4",
            3,
        ),
        ("qgan_bars_and_strips", "3b30db5c0cf90075e8a0fa9b0307a57e6903b9b6", 8),
    ],
)
def test_original_against_independent_circuit(name, blob, weight_count, scale):
    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    data = [0.2 * scale, -0.3 * scale, 0.4 * scale, -0.5 * scale]
    weights = [0.1 * scale * (index + 1) for index in range(weight_count)]
    program = synth.synthesize(
        source, classical_inputs={"input_vec": data, "weight": weights}
    )
    if weight_count == 3:
        reference = fq.Circuit(2, dtype=torch.complex128)
        reference.rx(1, theta=math.pi * data[0]).rz(1, theta=math.pi * data[1])
        reference.rx(0, theta=math.pi * data[2]).rz(0, theta=math.pi * data[3])
        reference.ry(1, theta=weights[0]).ry(0, theta=weights[1])
        reference.rzz(1, 0, theta=weights[2])
    else:
        reference = fq.Circuit(4, dtype=torch.complex128)
        for index, wire in enumerate((3, 2, 1, 0)):
            reference.rx(wire, theta=math.pi * data[index])
            reference.rz(wire, theta=math.pi * data[index])
            reference.ry(wire, theta=weights[index])
        reference.rzz(3, 2, theta=weights[4])
        reference.rzz(2, 1, theta=weights[5])
        reference.rzz(1, 0, theta=weights[6])
        # The original static branch acts on its first two logical wires.
        reference.rzz(3, 2, theta=weights[7])
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("target", "diagnostic"),
    [
        ("q[1:4]", "slice is out of bounds"),
        ("q[2:2]", "slice is out of bounds or empty"),
        ("q[0:1]", "requires a two-qubit array or view"),
        ("{q[1],q[1]}", "requires distinct qubit locations"),
    ],
)
def test_rzz_view_rejections(target, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            f"qfunc main(output q:qbit[3]) {{ allocate(q); RZZ(1,{target}); }}",
            source_format="qmod",
        )


def test_reordered_view_and_false_static_branch():
    source = """
      qfunc layer(q:qbit[]) {
        if (q.len > 2) { Z(q[0]); }
        RZZ(1,q[0:2]);
      }
      qfunc main(output q:qbit[3]) {
        allocate(q); hadamard_transform(q); layer({q[2],q[0]});
      }
    """
    reference = fq.Circuit(3, dtype=torch.complex128).h(0).h(1).h(2)
    reference.rzz(0, 2, theta=1)
    program = synth.synthesize(source, source_format="qmod")
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("condition", "gate", "diagnostic"),
    [
        ("q == 1", "X(q[0]);", "expected a static integer or loop index"),
        ("q.len > 2", "allocate(q);", "static if bodies admit only gates"),
    ],
)
def test_static_if_rejects_quantum_conditions_and_non_gate_bodies(
    condition, gate, diagnostic
):
    source = (
        f"qfunc main(output q:qbit[2]) {{allocate(q); if ({condition}) {{ {gate} }} }}"
    )
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
