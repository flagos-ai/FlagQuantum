"""Compare the original zero-entry walk to an independent six-wire Circuit."""

import hashlib
import json
import math
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.simulation.mps.marginal import marginal_probabilities
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
_DTYPE = torch.complex128


def _original() -> Path:
    manifest = json.loads((_FIXTURES / "tensor_zero_walk_sources.json").read_text())
    [entry] = manifest["fixtures"]
    source = _FIXTURES / entry["file"]
    data = source.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        == entry["git_blob_sha1"]
    )
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert (
        hashlib.sha256((_FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    return source


def _weights(source: str) -> list[float]:
    [body] = re.findall(r"inplace_prepare_amplitudes\(\[([^]]+)\]", source)
    values = [float(value) for value in body.split(",")]
    assert len(values) == 16
    return values


def _independent_circuit(weights: list[float]) -> fq.Circuit:
    """Use analytic dense preparation/QFT matrices on the four active clock wires."""
    basis = torch.arange(16, dtype=torch.float64)
    inverse = torch.exp(-2j * math.pi * basis[:, None] * basis[None, :] / 16) / 4
    target = torch.tensor(weights, dtype=_DTYPE)
    reflected = torch.eye(16, dtype=_DTYPE)[0] - target
    preparation = torch.eye(16, dtype=_DTYPE) - (
        2
        * reflected[:, None]
        * reflected[None, :].conj()
        / reflected.abs().square().sum()
    )
    return (
        fq.Circuit(6, dtype=_DTYPE)
        .any(0, 1, 2, 3, matrix=preparation)
        .any(0, 1, 2, 3, matrix=inverse)
        .h(4)
        .h(5)
    )


@pytest.mark.parametrize("alternate", [False, True])
def test_tensor_walk_exact_mps_and_named_readout(alternate: bool) -> None:
    original = _original()
    text = original.read_text()
    if alternate:
        numbers = list(range(1, 17))
        scale = math.hypot(*numbers)
        weights = [number / scale for number in numbers]
        text = re.sub(
            r"(?<=inplace_prepare_amplitudes\(\[)[^]]+",
            ", ".join(map(repr, weights)),
            text,
        )
        program = synth.synthesize(
            text, source_format="qmod", max_qubits=83, max_workspace=0
        )
    else:
        weights = _weights(text)
        program = synth.synthesize(original, max_qubits=83, max_workspace=0)
    assert program.report["resources"].n_qubits == 83
    assert program.report["resources"].used_qubits == 6
    assert program.execution.state_mode == "mps"
    assert program.execution.state_bytes == 42496
    assert program.report["workspace_qubits"] == 0
    result = fq.run(program)
    actual = result.native()
    assert isinstance(actual, MPSState)
    assert actual.max_bond == 4 and not actual.truncation_records
    assert sum(site.numel() * site.element_size() for site in actual.tensors) == 3168
    active = (
        *program.registers["t"],
        *program.registers["psi.plus1"],
        *program.registers["psi.plus2"],
    )
    words = []
    for index in range(64):
        bits = ["0"] * 83
        for wire, bit in zip(active, f"{index:06b}", strict=True):
            bits[wire] = bit
        words.append("".join(bits))
    reference = fq.run(_independent_circuit(weights)).statevector()
    torch.testing.assert_close(actual.amplitudes(words), reference, rtol=0, atol=1e-12)
    assert actual.amplitudes(words).abs().square().sum().item() == pytest.approx(
        1, abs=1e-12
    )
    dist = program.probabilities(result, registers=("t", "ell.mu", "psi.plus2"))
    assert sum(dist.values()) == pytest.approx(1, abs=1e-12)
    expected = reference.reshape(16, 2, 2).abs().square().sum(dim=1)
    for t in range(16):
        for sign in range(2):
            assert dist.get((t, 0, sign), 0.0) == pytest.approx(
                float(expected[t, sign]), abs=1e-12
            )
    assert all(value == 0 for _, value, _ in dist)


def test_tensor_walk_refuses_changed_marker_and_dirty_controls() -> None:
    original = _original()
    source = original.read_text()
    with pytest.raises(InteropConversionError, match="not orthogonal"):
        synth.synthesize(
            source.replace("aux2 = mu <= nu;", "aux2 = mu < nu;"), source_format="qmod"
        )
    with pytest.raises(InteropConversionError, match="work must be zero"):
        synth.synthesize(
            source.replace("free(aux289);", "X(aux289); free(aux289);"),
            source_format="qmod",
        )
    with pytest.raises(ValueError, match="memory|budget"):
        synth.synthesize(original, memory_limit_bytes=1024)


def test_mps_marginal_preserves_requested_wire_order() -> None:
    from flagquantum.simulation.mps.entrypoints import run_mps

    circuit = fq.Circuit(4, dtype=_DTYPE).h(0).cx(0, 2).x(1)
    mps = run_mps(circuit, dtype=_DTYPE)
    direct = marginal_probabilities(mps, (2, 1, 0))
    expected = circuit.state().reshape(1, 2, 2, 2, 2).abs().square().sum(-1)
    torch.testing.assert_close(
        direct, expected.permute(0, 3, 2, 1).reshape(1, 8), atol=1e-12, rtol=0
    )
