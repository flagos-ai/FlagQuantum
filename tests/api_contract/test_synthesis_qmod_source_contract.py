"""Retained source integrity and an existing-consumer replacement seam.

The recorded model is a test producer, not a native Qmod parser. The numerical
probe deliberately uses two bits and never counts as an original-file success.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import lower_arithmetic
from flagquantum.compiler._construction.profile import uint_type, verify_construction
from flagquantum.compiler._hybrid.model import (
    INDEX,
    Block,
    HybridProgram,
    Operation,
    Region,
    SourceLocation,
    Value,
    ValueId,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError

CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
UPSTREAM = "2bfd835e73b6882fcc515a725098b04766c238e7"
ORIGINALS = {
    "X.qmod": (
        "functions/qmod_library_reference/qmod_core_library/standard_gates/X.qmod",
        "51df4959dfff80b44df7e288f82778349c10f48d",
        "36814f1f719f430310703d4ab137879477539dd0d490d043ae5dbabf1532f825",
    ),
    "synthesis_tutorial.qmod": (
        "tutorials/basic_tutorials/the_classiq_tutorial/synthesis_tutorial.qmod",
        "7e9df09937d8360be71688762c4d22f309faa443",
        "4dbff47c68a449c918d30cdb63d8398de7084f02617e604d82c24191bcbfea8b",
    ),
    "multiplication_2vars_example.qmod": (
        "functions/function_usage_examples/arithmetic/multiplication/"
        "multiplication_2vars_example.qmod",
        "2282dd8e5587da689c795f04cac2eaf7a1f2a569",
        "5322769f60fad0608e6c2446fb7b04d85828a277e00f401d1baed3b562016233",
    ),
}
LICENSE_DIGEST = "241f68c9f35357f97e8a2a195e447d4f80098c140f0b3b3c770a11eed96f58e0"


@pytest.mark.unit
def test_original_sources_and_license_keep_pinned_upstream_identities() -> None:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    assert manifest["repository"] == "https://github.com/Classiq/classiq-library"
    assert manifest["commit"] == UPSTREAM
    assert len(manifest["fixtures"]) == len(ORIGINALS)
    assert {
        row["file"]: (row["upstream_path"], row["git_blob_sha1"], row["sha256"])
        for row in manifest["fixtures"]
    } == ORIGINALS
    for name, (_, git_blob, sha256) in ORIGINALS.items():
        payload = (CORPUS / name).read_bytes()
        blob = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
        assert hashlib.sha1(blob).hexdigest() == git_blob
        assert hashlib.sha256(payload).hexdigest() == sha256
    assert manifest["license_file"] == "LICENSE.txt"
    assert manifest["license_sha256"] == LICENSE_DIGEST
    assert hashlib.sha256((CORPUS / "LICENSE.txt").read_bytes()).hexdigest() == (
        LICENSE_DIGEST
    )


@pytest.mark.integration
def test_retained_verifier_runs_offline_without_framework_imports() -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(CORPUS / "verify_sources.py")],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert all(f"Verified {name}" in result.stdout for name in ORIGINALS)
    assert "7 digest checks passed" in result.stdout


@pytest.mark.integration
@pytest.mark.parametrize("filename", (*ORIGINALS, "LICENSE.txt"))
def test_source_or_license_edit_is_refused_by_the_retained_verifier(
    filename: str, tmp_path: Path
) -> None:
    root = tmp_path / "corpus"
    shutil.copytree(CORPUS, root)
    source = root / filename
    source.write_bytes(source.read_bytes() + b"\n")
    result = subprocess.run(
        [sys.executable, "-I", str(root / "verify_sources.py")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "mismatch" in result.stderr.lower()


def _expression(x: int) -> int:
    return x * x + 1


def _recorded_expression() -> HybridProgram:
    """Supply owned records directly, without parsing or executing source text."""
    x = Value(ValueId("recorded", 0), uint_type(2))
    square = Value(ValueId("recorded", 1), uint_type(4))
    one = Value(ValueId("recorded", 2), INDEX)
    output = Value(ValueId("recorded", 3), uint_type(4))
    location = SourceLocation("<recorded-two-bit-probe>", 1, 1)
    return HybridProgram(
        "recorded",
        Region(
            (
                Block(
                    (x,),
                    (
                        Operation(
                            "qint.multiply", (x, x), (square,), location=location
                        ),
                        Operation(
                            "arith.constant",
                            results=(one,),
                            attributes={"value": 1},
                            location=location,
                        ),
                        Operation(
                            "qint.add", (square, one), (output,), location=location
                        ),
                        Operation("qint.return", (output,), location=location),
                    ),
                ),
            )
        ),
    )


@pytest.mark.integration
@pytest.mark.parametrize("producer", ["python", "recorded"])
@pytest.mark.parametrize("scheduling", ["retain", "recompute"])
def test_same_arithmetic_consumer_accepts_an_independent_frontend_producer(
    producer: str, scheduling: Literal["retain", "recompute"]
) -> None:
    graph = (
        capture(_expression, {"x": 2})
        if producer == "python"
        else _recorded_expression()
    )
    # Only the producer changes; verification, ports, lowering and Runtime do not.
    assert verify_construction((graph,), entry=graph.name) is graph
    x = graph.body.blocks[0].arguments[0]
    lowered = lower_arithmetic(
        (graph,),
        {x.id: (0, 1)},
        (2, 3, 4, 5),
        entry=graph.name,
        n_qubits=18,
        workspace=tuple(range(6, 18)),
        workspace_state="fresh_zero",
        scheduling=scheduling,
    )
    circuit = fq.Circuit(18, dtype=torch.complex128)
    circuit.h(0).h(1).s(0).x(4).x(5).compose(lowered)
    options = fq.ExecutionOptions(
        mode="statevector",
        backend="torch",
        device="cpu",
        precision="complex128",
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )
    observed = fq.run(circuit, options=options).statevector().reshape(-1)
    expected = torch.zeros_like(observed)
    for value in range(4):
        index = ((value << 4) | (3 ^ (value * value + 1))) << 12
        expected[index] = (
            torch.tensor(1j if value >= 2 else 1, dtype=expected.dtype) / 2
        )
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    assert float(observed.reshape(64, 4096)[:, 1:].abs().square().sum()) <= 1e-12


@pytest.mark.unit
def test_existing_arithmetic_profile_does_not_silently_admit_register_effects() -> None:
    graph = _recorded_expression()
    body = graph.body.blocks[0]
    extended = HybridProgram(
        graph.name,
        Region((Block(body.arguments, (Operation("quantum.x"), *body.operations)),)),
    )
    with pytest.raises(HybridVerificationError, match="operation"):
        verify_construction((extended,), entry=extended.name)
