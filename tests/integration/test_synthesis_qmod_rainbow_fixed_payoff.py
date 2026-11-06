"""Independent state and amplitude-amplification Circuits for the Rainbow models."""

import ast
import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

DTYPE = torch.complex128
pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


def prepared_state(source: str, method: str) -> torch.Tensor:
    tables = [
        ast.literal_eval(table)
        for table in re.findall(r"inplace_prepare_state\(\s*(\[.*?\])", source, re.S)
    ]
    assert len(tables) == 2 and all(len(table) == 4 for table in tables)
    assert "res = max(x1 * 1.0, ((x1 * 0.75) + (x2 * 0.75)) + (-1.25));" in source
    width = 5
    fraction_bits = 2
    quantum_words = 1 << width
    rate = 0.434093883854633 / (1 << fraction_bits)
    result = torch.zeros(1 << 10, dtype=DTYPE)
    norm = math.sqrt(math.fsum(tables[0]) * math.fsum(tables[1]))
    price_scale = 193.97
    shift = -0.740780267709267
    strike = 190.0
    if method == "direct":
        max_value = (quantum_words - 1) / (1 << fraction_bits)
        low_probability = 13.784048752090222**2 / max(
            price_scale * math.exp(0.434093883854633 * max_value + shift),
            strike,
        )
    else:
        assert method == "integration"
        c = price_scale * math.exp(shift)
        correction = math.exp(-rate)
        low_probability = (strike - c * correction) / (
            c * math.expm1(quantum_words * rate) * correction
        )
    exponential = [math.exp(rate * word) for word in range(quantum_words)]
    exponential_norm = math.fsum(exponential)

    for a in range(4):
        for b in range(4):
            score = max(Fraction(a), Fraction(3 * (a + b) - 5, 4))
            raw_score = int(score * (1 << fraction_bits))
            scale = math.sqrt(tables[0][a] * tables[1][b]) / norm
            source_word = a + 4 * b
            if score <= Fraction(3, 2):
                result[source_word] = scale * math.sqrt(1 - low_probability)
                result[512 + source_word] = scale * math.sqrt(low_probability)
            else:
                for auxiliary in range(quantum_words):
                    if method == "integration":
                        magnitude = math.sqrt(exponential[auxiliary] / exponential_norm)
                        indicator = int(auxiliary <= raw_score)
                    else:
                        magnitude = 1.0
                        for bit in range(width):
                            value = auxiliary >> bit & 1
                            if raw_score >> bit & 1:
                                if value:
                                    magnitude = 0.0
                                    break
                            else:
                                zero = math.exp(-rate * (1 << bit) / 2)
                                magnitude *= math.sqrt(1 - zero**2) if value else zero
                        indicator = int(auxiliary == 0)
                    result[512 * indicator + 16 * auxiliary + source_word] = (
                        scale * magnitude
                    )
    return result


def reference(source: str, method: str, rounds: int) -> fq.Circuit:
    state = prepared_state(source, method)
    identity = torch.eye(len(state), dtype=DTYPE)
    initial = identity[:, 0] - state
    prep = identity - 2 * torch.outer(initial, initial.conj()) / torch.vdot(
        initial, initial
    )
    reflection = 2 * torch.outer(state, state.conj()) - identity
    circuit = fq.Circuit(10, dtype=DTYPE).any(*range(10), matrix=prep)
    for _ in range(rounds):
        circuit.z(0).any(*range(10), matrix=reflection)
    return circuit


def _original(method: str) -> Path:
    manifest = json.loads((_FIXTURES / "rainbow_fixed_payoff_sources.json").read_text())
    raw_license = (_FIXTURES / manifest["license_file"]).read_bytes()
    assert hashlib.sha256(raw_license).hexdigest() == manifest["license_sha256"]
    [row] = [
        item
        for item in manifest["fixtures"]
        if item["file"] == f"rainbow_options_{method}_method.qmod"
    ]
    path = _FIXTURES / row["file"]
    raw = path.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    return path


@pytest.mark.parametrize("method", ["direct", "integration"])
@pytest.mark.parametrize("rounds", [0, 1, 2])
def test_original_payoff_complex_state_and_workspace(method: str, rounds: int) -> None:
    path = _original(method)
    program = synth.synthesize(
        path,
        classical_inputs={"k": rounds},
        max_qubits=14,
        max_workspace=4,
        max_operations=60_000,
    )
    assert dict(program.registers) == {"indicator": (0,)}
    assert program.report["workspace_qubits"] == 4
    state = fq.run(program).statevector().reshape(1 << 10, 1 << 4)
    expected = (
        fq.run(reference(path.read_text(), method, rounds)).statevector().flatten()
    )
    torch.testing.assert_close(state[:, 0], expected, atol=1e-10, rtol=0)
    assert torch.linalg.vector_norm(state[:, 1:]) < 1e-10


PROGRAM = """
qperm compute(const a:qnum, const b:qnum, output score:qnum) {
    score = max(a, (0.5 * b) - 0.25);
}
qfunc apply_payoff(const score:qnum, target:qbit) {
    control (score > 0.75) {
        RY(score.size + 0.125 * score.fraction_digits, target);
    } else {
        target ^= score == 0;
    }
}
qfunc compute_then_pay(a:qnum<2>, b:qnum<2>, target:qbit) {
    score:qnum;
    compute(a,b,score);
    apply_payoff(score,target);
}
qfunc main(output a:qnum<2>, output b:qnum<2>, output target:qbit) {
    allocate(a); allocate(b); allocate(target);
    hadamard_transform(a); hadamard_transform(b); X(target); H(target); S(target);
    compute_then_pay(a,b,target);
}
"""


def test_inferred_layout_and_nonzero_target_keep_coherent_phase() -> None:
    program = synth.synthesize(PROGRAM, source_format="qmod")
    assert dict(program.registers) == {"a": (0, 1), "b": (2, 3), "target": (4,)}
    circuit = fq.Circuit(5, dtype=DTYPE)
    for wire in range(4):
        circuit.h(wire)
    circuit.x(4).h(4).s(4)
    angle = 5 + 0.125 * 2
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    ry = torch.tensor([[c, -s], [s, c]], dtype=DTYPE)
    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    identity = torch.eye(2, dtype=DTYPE)
    blocks = []
    for a in range(4):
        for b in range(4):
            score = max(Fraction(a), Fraction(2 * b - 1, 4))
            blocks.append(
                ry if score > Fraction(3, 4) else x if score == 0 else identity
            )
    circuit.any(*range(5), matrix=torch.block_diag(*blocks))
    state = fq.run(program).statevector()
    workspace = program.report["workspace_qubits"]
    state = state.reshape(32, 1 << workspace)
    torch.testing.assert_close(
        state[:, 0],
        fq.run(circuit).statevector().flatten(),
        atol=1e-11,
        rtol=0,
    )
    assert torch.linalg.vector_norm(state[:, 1:]) < 1e-11


@pytest.mark.parametrize(
    ("old", "new", "diagnostic"),
    [
        ("0.5 * b", "0.1 * b", "dyadic"),
        ("compute(a,b,score)", "compute(a,a,score)", "disjoint"),
        ("const score:qnum", "score:qnum", "incompatible"),
        (
            "RY(score.size + 0.125 * score.fraction_digits, target)",
            "RY(0.2, score[0])",
            "writable",
        ),
        ("target ^= score == 0", "allocate(target)", "reversibly"),
    ],
)
def test_payoff_rejects_unchecked_inputs(old: str, new: str, diagnostic: str) -> None:
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(PROGRAM.replace(old, new), source_format="qmod")


def test_original_rejects_workspace_budget_below_measured() -> None:
    with pytest.raises(ValueError, match="budget|workspace"):
        synth.synthesize(
            _original("direct"), classical_inputs={"k": 0}, max_workspace=3
        )
