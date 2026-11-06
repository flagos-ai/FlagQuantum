"""Verify the original network walk and coherent real preparations."""

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


def original() -> tuple[Path, dict[str, str]]:
    manifest = json.loads((FIXTURES / "network_walk_sources.json").read_text())
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


def reference() -> fq.Circuit:
    adjacency = ((1, 2, 3), (0, 2), (0, 1), (0,))
    eye = torch.eye(4, dtype=DTYPE)
    initial = []
    reflections = []
    for neighbors in adjacency:
        state = eye[:, list(neighbors)].sum(dim=1) / math.sqrt(len(neighbors))
        delta = eye[:, 0] - state
        initial.append(
            eye
            if len(neighbors) == 1
            else eye - 2 * torch.outer(delta, delta) / delta.dot(delta)
        )
        reflections.append(2 * torch.outer(state, state) - eye)
    circuit = fq.Circuit(4, dtype=DTYPE).h(0).h(1)
    circuit.any(0, 1, 2, 3, matrix=torch.block_diag(*initial))
    reflected = torch.block_diag(*reflections)
    for _ in range(3):
        circuit.any(0, 1, 2, 3, matrix=reflected).swap(0, 2).swap(1, 3)
    return circuit


def test_original_complex_network_walk_full_state_and_marginal() -> None:
    path, _ = original()
    program = synth.synthesize(path)
    result = fq.run(program)
    expected = fq.run(reference()).statevector().flatten()
    actual = result.statevector().reshape(16, -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-12, rtol=0
    )
    marginal = expected.abs().square().reshape(4, 4).sum(dim=-1)
    reported = dict(program.probabilities(result, registers=("x",)))
    assert reported == pytest.approx({(i,): float(p) for i, p in enumerate(marginal)})


@pytest.mark.parametrize("initial", range(4))
def test_controlled_real_preparation_matches_every_target_column(initial: int) -> None:
    source = """
        qfunc main(output control_word:qnum<2>,output target:qbit[2]) {
            allocate(control_word); allocate(target);
            hadamard_transform(control_word);
            INITIAL
            control (control_word == 2) {
                inplace_prepare_state([0.125,0.25,0.5,0.125],0,target);
            }
        }
    """
    gates = "".join(f"X(target[{bit}]);" for bit in range(2) if initial & (1 << bit))
    program = synth.synthesize(source.replace("INITIAL", gates), source_format="qmod")

    def rotation(lo: float, hi: float) -> torch.Tensor:
        c = math.sqrt(lo / (lo + hi))
        s = math.sqrt(hi / (lo + hi))
        return torch.tensor([[c, -s], [s, c]], dtype=DTYPE)

    preparation = torch.block_diag(
        rotation(0.125, 0.25), rotation(0.5, 0.125)
    ) @ torch.kron(rotation(0.375, 0.625), torch.eye(2, dtype=DTYPE))
    reference = fq.Circuit(4, dtype=DTYPE).h(0).h(1)
    for bit in range(2):
        if initial & (1 << bit):
            reference.x(3 - bit)
    reference.any(
        0,
        1,
        2,
        3,
        matrix=torch.block_diag(
            torch.eye(8, dtype=DTYPE), preparation, torch.eye(4, dtype=DTYPE)
        ),
    )
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("action", "diagnostic"),
    [
        ("multiswap(x,x);", "equally sized disjoint views"),
        (
            "control (x == 1) {inplace_prepare_state([0.125,0.25,0.5,0.125],0,x);}",
            "selected preparation needs at most eight distinct bits",
        ),
        (
            "control (x == 4) {inplace_prepare_state([0.125,0.25,0.5,0.125],0,y);}",
            "equality must fit",
        ),
    ],
)
def test_guarded_preparation_and_swap_refuse_invalid_views(
    action: str, diagnostic: str
) -> None:
    source = (
        "qfunc main(output x:qnum<2>,output y:qnum<2>) {allocate(x);allocate(y);"
        + action
        + "}"
    )
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")
