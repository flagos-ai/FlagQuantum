"""Verify the original Grover-mixer QAOA ansatz with diagonal references."""

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
PARAMETERS = (
    (0.13, 0.47, -0.39, 0.11, 0.9, -0.29),
    (-0.32, 0.81, 0.24, -0.77, 0.36, 0.15),
)


def original() -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "gm_qaoa_sources.json").read_text())
    [row] = manifest["fixtures"]
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


def reference(params: tuple[float, ...]) -> fq.Circuit:
    preparation = fq.Circuit(4, dtype=DTYPE).x(3)
    for bit, angle in enumerate((2.0943951023932, 1.91063323624902, 1.5707963267949)):
        low, high = 3 - bit, 2 - bit
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        rotation = torch.tensor([[c, -s], [s, c]], dtype=DTYPE)
        controlled = torch.block_diag(torch.eye(2, dtype=DTYPE), rotation)
        preparation.cx(high, low).any(low, high, matrix=controlled).cx(high, low)
    inverse = preparation.adjoint()
    circuit = fq.Circuit(4, dtype=DTYPE).compose(preparation)
    energies = torch.tensor(
        [
            sum(
                weight for bit, weight in enumerate((1, 2, -1, 12)) if word & (1 << bit)
            )
            for word in range(16)
        ],
        dtype=DTYPE,
    )
    for gamma, beta in zip(params[:3], params[3:], strict=True):
        circuit.any(*range(4), matrix=torch.diag(torch.exp(1j * gamma * energies)))
        rotation = torch.ones(16, dtype=DTYPE)
        angle = 0.318309886183791 * beta
        rotation[0] = complex(math.cos(angle / 2), -math.sin(angle / 2))
        rotation[8] = rotation[0].conj()
        circuit.compose(inverse).any(*range(4), matrix=torch.diag(rotation)).compose(
            preparation
        )
    return circuit


@pytest.mark.parametrize("params", PARAMETERS)
def test_original_gm_qaoa_full_state_and_clean_workspace(
    params: tuple[float, ...],
) -> None:
    path, _ = original()
    program = synth.synthesize(path, classical_inputs={"params": params})
    actual = fq.run(program).statevector().reshape(16, -1)
    expected = fq.run(reference(params)).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )


def test_indexed_rz_refuses_alias() -> None:
    source = """
        qfunc main(output x:qnum<1>) {
            allocate(x);
            control (x == 1) {RZ(0.4,x);}
        }
    """
    with pytest.raises(InteropConversionError, match="distinct qubit locations"):
        synth.synthesize(source, source_format="qmod")


def test_indexed_rz_preserves_relative_phases_against_inactive_words() -> None:
    source = """
        qfunc main(output word:qnum<2>,output target:qbit) {
            allocate(word); allocate(target);
            hadamard_transform(word); H(target);
            control (word == 2) {RZ(0.9,target);}
        }
    """
    program = synth.synthesize(source, source_format="qmod")
    actual = fq.run(program).statevector().reshape(8, -1)
    expected = torch.ones(8, dtype=DTYPE) / math.sqrt(8)
    expected[4] *= complex(math.cos(0.45), -math.sin(0.45))
    expected[5] *= complex(math.cos(0.45), math.sin(0.45))
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )
