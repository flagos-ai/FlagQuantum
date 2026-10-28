"""Pinned source integrity and handwritten fakes for the existing IR boundary.

These fakes do not capture Python or parse Qmod. Real frontend conformance and
original-file execution are gates of the later implementation PRs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "synth"
_REGISTER_QUBITS = (2, 0)


def test_original_sources_and_license_match_the_pinned_manifest() -> None:
    manifest = json.loads((_FIXTURES / "manifest.json").read_text())
    assert len(manifest["fixtures"]) == 3
    for record in manifest["fixtures"]:
        payload = (_FIXTURES / record["file"]).read_bytes()
        header = b"blob " + str(len(payload)).encode("ascii") + b"\0"
        assert hashlib.sha1(header + payload).hexdigest() == record["git_blob_sha1"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
    license_bytes = (_FIXTURES / manifest["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == manifest["license_sha256"]


def _circuit_boundary_fake(value: int) -> CircuitIR:
    circuit = fq.Circuit(3, dtype=torch.complex128)
    for position, qubit in enumerate(_REGISTER_QUBITS):
        if value & (1 << (1 - position)):
            circuit.x(qubit)
    return circuit.to_ir()


def _ir_boundary_fake(value: int) -> CircuitIR:
    # Spell the same preparation independently, using the IR rather than Circuit.
    instructions = []
    if value >= 2:
        instructions.append(Instruction(name="x", wires=(2,)))
    if value % 2:
        instructions.append(Instruction(name="x", wires=(0,)))
    return CircuitIR(3, tuple(instructions), dtype="complex128")


@pytest.mark.parametrize("producer", [_circuit_boundary_fake, _ir_boundary_fake])
@pytest.mark.parametrize("value", range(4))
def test_replaceable_producers_preserve_layout_and_zero_workspace(
    producer: Callable[[int], CircuitIR], value: int
) -> None:
    program = replace(
        producer(value),
        metadata={
            "synth": {
                "version": 1,
                "registers": {"q": {"role": "output", "qubits": [2, 0], "width": 2}},
                "workspace_qubits": [1],
                "source": {
                    "frontend": "boundary_fake",
                    "origin": "test_synth_contract",
                    "sha256": hashlib.sha256(b"handwritten boundary fake").hexdigest(),
                },
            }
        },
    )
    restored = CircuitIR.from_json(program.to_json())
    assert restored.to_json() == program.to_json()
    result = fq.run(
        restored, options=fq.ExecutionOptions(mode="statevector", device="cpu")
    )
    actual = result.statevector().reshape(-1)

    # q=2*bit(qubit2)+bit(qubit0), with physical qubit0 the MSB axis.
    expected_index = (value % 2) * 4 + value // 2
    expected = torch.zeros(8, dtype=torch.complex128)
    expected[expected_index] = 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    dirty_workspace = [index for index in range(8) if index & 2]
    assert torch.sum(torch.abs(actual[dirty_workspace]) ** 2).item() == 0


def test_original_x_behavior_has_an_existing_runtime_reference() -> None:
    program = fq.Circuit(1, dtype=torch.complex128).x(0).to_ir()
    result = fq.run(
        program, options=fq.ExecutionOptions(mode="statevector", device="cpu")
    )
    expected = torch.tensor([0, 1], dtype=torch.complex128)
    torch.testing.assert_close(result.statevector().reshape(-1), expected)
