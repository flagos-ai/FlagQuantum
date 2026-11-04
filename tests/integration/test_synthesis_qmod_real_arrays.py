"""Check fixed classical arrays against independent encoding circuits."""

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


@pytest.mark.parametrize(
    ("name", "blob", "parameter", "values"),
    [
        (
            "encode_in_angle",
            "3510209fabde4432f3fc39c20d6f481cd52112a6",
            "data",
            [0.2, -0.3, 0.4, 0.5],
        ),
        (
            "encode_on_bloch",
            "81e9d4a6c7abdec44c3905945b7402abdd72c106",
            "data",
            [0.2, -0.3, 0.4, 0.5, 0.6, -0.7, 0.8],
        ),
        (
            "execution_tutorial_part2",
            "c53f7fdf9290e4f172fa99878a6bf1a2cf061244",
            "angles",
            [0.2, -0.3],
        ),
    ],
)
def test_real_array_original_against_circuit(name, blob, parameter, values):
    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    program = synth.synthesize(source, classical_inputs={parameter: values})
    if name == "execution_tutorial_part2":
        reference = fq.Circuit(2, dtype=torch.complex128)
        reference.rx(0, theta=0.2).ry(0, theta=-0.3).cx(0, 1)
    else:
        reference = fq.Circuit(4, dtype=torch.complex128)
        if name == "encode_in_angle":
            for wire, value in enumerate(reversed(values)):
                reference.ry(wire, theta=math.pi * value)
        else:
            reference.rx(3, theta=0.2 * math.pi).rz(3, theta=-0.3 * math.pi)
            reference.rx(2, theta=0.4 * math.pi).rz(2, theta=0.5 * math.pi)
            reference.rx(1, theta=0.6 * math.pi).rz(1, theta=-0.7 * math.pi)
            reference.rx(0, theta=1.6 * math.pi)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("values", "diagnostic"),
    [
        ([0.1], "exactly 2"),
        ([True, 0], "finite real"),
        ([0, float("inf")], "finite real"),
    ],
)
def test_real_array_binding_errors(values, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            FIXTURES / "execution_tutorial_part2.qmod",
            classical_inputs={"angles": values},
        )


@pytest.mark.parametrize(
    ("index", "diagnostic"), [("2", "out of bounds"), ("0+1", "integer literal")]
)
def test_real_array_index_errors(index, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(
            f"qfunc main(data:real[2],output q:qbit) {{ allocate(q); RY(data[{index}],q); }}",
            source_format="qmod",
            classical_inputs={"data": (0.1, 0.2)},
        )
