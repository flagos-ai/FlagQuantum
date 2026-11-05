"""Check the original QSVM encoding and its coherent inverse."""

import hashlib
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
SOURCE = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/qsvm.qmod"
DATA1 = (0.27, -0.43)
DATA2 = (-0.61, 0.18)


@pytest.mark.parametrize("data2", [DATA1, DATA2])
def test_original_qsvm_against_circuit(data2: tuple[float, float]) -> None:
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "21d0c54b64a84b211c8a2238152e6d37a06ce778"
    )
    program = synth.synthesize(
        SOURCE, classical_inputs={"data1": DATA1, "data2": data2}
    )
    reference = fq.Circuit(1, dtype=torch.complex128)
    reference.rx(0, theta=math.pi * DATA1[0])
    reference.rz(0, theta=math.pi * DATA1[1])
    reference.rz(0, theta=-math.pi * data2[1])
    reference.rx(0, theta=-math.pi * data2[0])
    actual = fq.run(program).statevector()
    torch.testing.assert_close(
        actual, fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    if data2 == DATA1:
        torch.testing.assert_close(
            actual.flatten(),
            torch.tensor([1, 0], dtype=torch.complex128),
            atol=1e-12,
            rtol=0,
        )
    assert program.report["workspace_qubits"] == 0


def test_inplace_encoding_on_a_nonzero_input():
    program = synth.synthesize(
        """qfunc main(data:real[2], output q:qnum<1>) {
            q=1; inplace_encode_on_bloch(data,q);
        }""",
        source_format="qmod",
        classical_inputs={"data": DATA1},
    )
    reference = fq.Circuit(1, dtype=torch.complex128).x(0)
    reference.rx(0, theta=math.pi * DATA1[0]).rz(0, theta=math.pi * DATA1[1])
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("replacement", "diagnostic"),
    [
        ("qba: qnum<2>", "encoding output width"),
        ("encode_on_bloch(data2", "one in-place Bloch"),
        ("allocate(qba);", "uninitialized"),
    ],
)
def test_bloch_inverse_rejects_out_of_profile_calls(replacement, diagnostic):
    source = SOURCE.read_text()
    if replacement == "allocate(qba);":
        source = source.replace(replacement, "", 1)
    elif replacement.startswith("qba"):
        source = source.replace("qba: qnum<1>", replacement, 1)
    else:
        source = source.replace("inplace_encode_on_bloch(data2", replacement, 1)
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            source,
            source_format="qmod",
            classical_inputs={"data1": DATA1, "data2": DATA2},
        )
