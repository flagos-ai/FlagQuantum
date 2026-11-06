"""Verify both unchanged circle walks against a cyclic permutation."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.services.synthesis import _synthesize

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128
POSITIONS = 128


def original(name: str) -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "circle_walk_sources.json").read_text())
    row = next(row for row in manifest["fixtures"] if row["file"] == f"{name}.qmod")
    path = FIXTURES / row["file"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    return path, row


def reference(steps: int, balanced: bool) -> fq.Circuit:
    circuit = fq.Circuit(8, dtype=DTYPE)
    if balanced:
        circuit.h(7).s(7)
    shift = torch.zeros((2 * POSITIONS, 2 * POSITIONS), dtype=DTYPE)
    for position in range(POSITIONS):
        for coin in (0, 1):
            destination = (position + (1 if coin == 0 else -1)) % POSITIONS
            shift[2 * destination + coin, 2 * position + coin] = 1
    for _ in range(steps):
        circuit.h(7)
        circuit.any(*range(8), matrix=shift)
    return circuit


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
@pytest.mark.parametrize("balanced", [False, True])
def test_original_circle_walk_state_and_wrap(steps: int, balanced: bool) -> None:
    name = "quantum_walk_circle_balanced_coin" if balanced else "quantum_walk_circle"
    path, _ = original(name)
    program = _synthesize(path, classical_inputs={"t": steps})
    result = fq.run(program)
    assert program.report["qmod_signed_registers"] == ("x",)
    actual = result.statevector().reshape(2 * POSITIONS, -1)
    expected = fq.run(reference(steps, balanced)).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )
    if steps == 1:
        assert dict(
            program.probabilities(result, registers=("x", "coin"))
        ) == pytest.approx({(1, 0): 0.5, (127, 1): 0.5})


@pytest.mark.parametrize("steps", [-1, 257])
def test_circle_walk_repeat_budget(steps: int) -> None:
    source, _ = original("quantum_walk_circle")
    with pytest.raises(InteropConversionError, match="repeat count must be in 0..256"):
        _synthesize(source, classical_inputs={"t": steps})


def test_signed_literal_updates_refuse_conflicting_allocation() -> None:
    source = "qfunc main(output x:qnum<3,SIGNED,0>) {allocate(3,UNSIGNED,0,x);}"
    with pytest.raises(InteropConversionError, match="signedness must agree"):
        synth.synthesize(source, source_format="qmod")
