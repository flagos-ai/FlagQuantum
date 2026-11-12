"""Check the pinned Bell preparation example and its coherent sign."""

import hashlib
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from tests.synthesis_workflow_helpers import check_state

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/prepare_bell_state.qmod"
)


def test_pinned_bell_original() -> None:
    raw = SOURCE.read_bytes()
    blob = f"blob {len(raw)}\0".encode() + raw
    assert hashlib.sha1(blob).hexdigest() == "a848085bc035d413768249b115b2707152833a0d"
    program = synth.synthesize(SOURCE)
    amplitude = 1 / math.sqrt(2)
    check_state(program, fq.run(program), {(1,): amplitude, (2,): amplitude})
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("state", ["0", "3", "4", "1+0"])
def test_unverified_bell_variant_is_refused(state: str) -> None:
    source = SOURCE.read_text().replace("state(2,", f"state({state},")
    with pytest.raises(InteropConversionError, match="literal Bell states 1 and 2"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize(
    ("name", "digest"),
    [
        ("classical_shadows.qmod", "ea21facd0aec2d550ae8eed1e9e9deaebf30e796"),
        (
            "classical_shadow_tomography.qmod",
            "ec20f481553b87e3ab7e7712bb620bc2ea99b425",
        ),
    ],
)
def test_bell_one_originals_against_independent_circuit(name: str, digest: str) -> None:
    source = SOURCE.with_name(name)
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    # Prepare Phi- by placing the phase before the entangling gate.
    reference = fq.Circuit(2, dtype=torch.complex128)
    reference.h(0).z(0).cx(0, 1)
    if name == "classical_shadow_tomography.qmod":
        # Qmod array indices 0 and 1 map to Circuit wires 1 and 0.
        reference.s(1).h(1).s(0).s(0).s(1).h(0)
        reference.cx(1, 0).cx(1, 0).s(1).h(0).s(1).h(1).s(1).s(1)
        reference.cx(0, 1).h(0).h(0).s(1).s(1).s(1)
    program = synth.synthesize(source)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0
