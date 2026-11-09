"""Check pinned multi-controlled X sources and coherent all-ones control."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
DTYPE = torch.complex128


@pytest.mark.parametrize(
    ("name", "digest", "controls", "workspace"),
    [
        ("mcx_example.qmod", "822d889318cefe44a4eb90420102d0a65f6224a4", 7, 4),
        ("hardware_aware_mcx.qmod", "75c0e84394ef5370d7210d6a68c2b31f01537e20", 15, 6),
    ],
)
def test_original_array_control_matches_zero_state_circuit(
    name, digest, controls, workspace
):
    source = FIXTURES / name
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    program = synth.synthesize(source, memory_limit_bytes=512 * 1024**2)
    assert tuple(map(len, program.registers.values())) == (controls, 1)
    assert program.report["workspace_qubits"] == workspace
    # The source initializes every control to zero, so an empty Circuit is the oracle.
    reference = fq.Circuit(controls + 1 + workspace, dtype=DTYPE)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


SOURCE = """qfunc main(output c:qbit[], output t:qbit) {
    allocate(7, c);
    allocate(t);
    hadamard_transform(c);
    S(c[0]);
    control(c) { X(t); }
}"""


def test_array_control_is_coherent_and_uncomputes_group_flags():
    program = synth.synthesize(SOURCE, source_format="qmod", max_workspace=4)
    reference = fq.Circuit(12, dtype=DTYPE)
    for bit in range(7):
        reference.h(bit)
    reference.s(6)
    matrix = torch.eye(256, dtype=DTYPE)
    matrix[-2:, -2:] = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    reference.any(*range(8), matrix=matrix)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_array_control_respects_workspace_limit():
    with pytest.raises(ValueError, match="workspace|budget"):
        synth.synthesize(SOURCE, source_format="qmod", max_workspace=3)


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("allocate(2,c); allocate(t); control(c) { X(c); }", "disjoint"),
        ("allocate(2,c); control(c) { X(t); }", "initialized"),
        ("allocate(2,c); allocate(t); control(c) { H(t); }", "single scalar X"),
    ],
)
def test_array_control_refuses_unsupported_targets(body, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            f"qfunc main(output c:qbit[], output t:qbit) {{ {body} }}",
            source_format="qmod",
        )
