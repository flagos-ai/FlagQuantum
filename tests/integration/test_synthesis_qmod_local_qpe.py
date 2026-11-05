"""Compare original Grover phase estimation with a dense recurrence."""

import hashlib
import json
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


def grover_qpe_reference() -> torch.Tensor:
    # The physical data basis puts the one-bit marker before the four-bit word.
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
    word_transform = hadamard
    for _ in range(3):
        word_transform = torch.kron(word_transform, hadamard)
    prepare = torch.kron(torch.eye(2, dtype=DTYPE), word_transform)
    marked = {2, 3, 5, 7, 11, 13}
    word_oracle = torch.eye(32, dtype=DTYPE)[
        [basis ^ (16 if basis % 16 in marked else 0) for basis in range(32)]
    ]
    prepare = word_oracle @ prepare
    zero_reflection = torch.eye(32, dtype=DTYPE)
    zero_reflection[0, 0] = -1
    marker_phase = torch.diag(torch.tensor([1] * 16 + [-1] * 16, dtype=DTYPE))
    grover = -prepare @ zero_reflection @ prepare.mH @ marker_phase
    state = torch.eye(32, dtype=DTYPE)[:, 0]
    powers = []
    for _ in range(32):
        powers.append(state)
        state = grover @ state
    return torch.fft.fft(torch.stack(powers), dim=0, norm="ortho") / math.sqrt(32)


def test_original_grover_qpe_matches_all_amplitudes() -> None:
    manifest = json.loads((FIXTURES / "local_qpe_sources.json").read_text())
    original = manifest["fixtures"][0]
    path = FIXTURES / original["file"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == original["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == original["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )

    program = synth.synthesize(path)
    workspace = program.report["workspace_qubits"]
    assert isinstance(workspace, int)
    state = fq.run(program).statevector().reshape(32, 32, 1 << workspace)
    expected = torch.zeros_like(state)
    expected[:, :, 0] = grover_qpe_reference()
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    assert dict(program.registers) == {"phase": tuple(range(5))}


def test_static_integer_array_equality_matches_control_on_asymmetric_word() -> None:
    source = """
        qfunc mark(words:int[], x:qnum, target:qbit) {
            repeat (i: words.len) { control (x == words[i]) { X(target); } }
        }
        qfunc main(output x:qnum<2>, output target:qbit) {
            allocate(x); allocate(target);
            hadamard_transform(x); H(target); S(target);
            mark([1], x, target);
        }
    """
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(3, dtype=DTYPE).h(0).h(1).h(2).s(2).x(0).ccx(0, 1, 2).x(0)
    torch.testing.assert_close(
        fq.run(program).statevector(), reference.state(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "body, message",
    [
        ("H(local);", "local references"),
        ("allocate(local); drop(local); H(local);", "dropped register"),
        ("allocate(local); drop(local); allocate(local);", "dropped register"),
        ("drop(result);", "requires a local register"),
    ],
)
def test_local_scalar_lifetime_is_checked(body: str, message: str) -> None:
    with pytest.raises(InteropConversionError, match=message):
        synth.synthesize(
            "qfunc main(output result:qbit) { local:qbit; allocate(result); "
            + body
            + "}",
            source_format="qmod",
        )
