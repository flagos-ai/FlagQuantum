"""Independent branch Circuit for the pinned two-fractional-bit autocallable."""

import hashlib
import json
import math
from itertools import product
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.ecosystem import InteropConversionError
from flagquantum.services.synthesis import _synthesize

pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"

DTYPE = torch.complex128
DISTRIBUTION = (
    0.065634503010069,
    0.434365496989931,
    0.434365496989931,
    0.065634503010069,
)


def discount_word(grid: int, *, shift: float = 0.1274075) -> int:
    slope = 0.238268134860707 * 1.5
    intercept = shift - 3 * 0.238268134860707
    return math.trunc(intercept * 4) + math.trunc(slope * 4) * grid


def branch_circuit(steps: tuple[int, ...], *, shift: float = 0.1274075) -> fq.Circuit:
    increments = [discount_word(grid, shift=shift) for grid in steps]
    a, b, c = increments[0], sum(increments[:2]), sum(increments)
    assert -8 <= min(a, b, c) <= max(a, b, c) < 8
    circuit = fq.Circuit(5, dtype=DTYPE)
    if a > 0:
        return circuit.ry(4, 0.875426634895785)
    if b > 0:
        return circuit.ry(4, 0.89966916778117)
    if min(a, b, c) >= -1 or c >= 0:
        return circuit.ry(4, 0.818547457426368)
    target = c + 8
    for bit in range(4):
        circuit.ry(3 - bit, 2 * math.atan(math.exp(0.125 * (1 << bit))))
    matrix = torch.eye(32, dtype=DTYPE)
    for word in range(target + 1):
        lo = word << 1
        matrix[[lo, lo + 1]] = matrix[[lo + 1, lo]]
    return circuit.any(*range(5), matrix=matrix)


def state(*, shift: float = 0.1274075) -> torch.Tensor:
    rows = []
    for steps in product(range(4), repeat=3):
        weight = math.sqrt(math.prod(DISTRIBUTION[i] for i in steps))
        branch = fq.run(branch_circuit(steps, shift=shift)).statevector().flatten()
        rows.append(weight * branch.reshape(16, 2))
    return torch.cat(rows)


def amplify(prepared: torch.Tensor, rounds: int) -> torch.Tensor:
    original = prepared.clone()
    good = float(prepared[:, 1].abs().square().sum())
    theta = math.asin(math.sqrt(good))
    original[:, 0] *= math.cos((2 * rounds + 1) * theta) / math.cos(theta)
    original[:, 1] *= math.sin((2 * rounds + 1) * theta) / math.sin(theta)
    return original


def _original() -> Path:
    manifest = json.loads(
        (_FIXTURES / "autocallable_fixed_preparation_sources.json").read_text()
    )
    [source] = manifest["fixtures"]
    for row in (source, manifest["companion_options"]):
        raw = (_FIXTURES / row["file"]).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            == row["git_blob_sha1"]
        )
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    license_bytes = (_FIXTURES / manifest["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == manifest["license_sha256"]
    return _FIXTURES / source["file"]


@pytest.mark.parametrize(
    ("rounds", "changed"), [(0, False), (1, False), (2, False), (2, True)]
)
def test_autocallable_full_purification(rounds: int, changed: bool) -> None:
    original = _original()
    shift = 0.22 if changed else 0.1274075
    bounds = dict(max_qubits=11, max_workspace=0, max_operations=4200)
    inputs = {"k": rounds}
    if changed:
        source = original.read_text().replace("0.1274075 +", "0.22 +")
        program = _synthesize(
            source,
            source_format="qmod",
            qmod_machine_precision=2,
            classical_inputs=inputs,
            **bounds,
        )
    else:
        program = _synthesize(original, classical_inputs=inputs, **bounds)
    assert program.registers == {"indicator": (0,)}
    assert program.report["workspace_qubits"] == 0
    result = fq.run(program)
    actual = result.statevector().reshape(2, 1024)
    logical = amplify(state(shift=shift), rounds)
    expected = logical.reshape(4, 4, 4, 16, 2).permute(4, 3, 2, 1, 0).reshape(2, 1024)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-11)
    density = actual @ actual.mH
    torch.testing.assert_close(density, expected @ expected.mH, rtol=0, atol=1e-11)
    if not changed and rounds == 0:
        assert 0.119158741 < density[1, 1].real < 0.1197666
    if rounds == 2:
        assert density[0, 1].real < 0
    probabilities = program.probabilities(result, registers=("indicator",))
    assert probabilities[(1,)] == pytest.approx(float(density[1, 1].real), abs=1e-11)


def test_autocallable_checks_precision_aliases_and_budgets() -> None:
    original = _original()
    inputs = {"k": 1}
    with pytest.raises(ValueError, match="budget|operation"):
        _synthesize(original, classical_inputs=inputs, max_operations=100)
    source = original.read_text()
    with pytest.raises(
        InteropConversionError,
        match="actual and formal formats|declared machine precision",
    ):
        _synthesize(
            source,
            source_format="qmod",
            qmod_machine_precision=3,
            classical_inputs=inputs,
        )
    wrong_oracle = source.replace("Z(est_reg[est_reg.len - 1]);", "Z(est_reg[0]);")
    with pytest.raises(InteropConversionError, match="must mark the indicator"):
        _synthesize(
            wrong_oracle,
            source_format="qmod",
            qmod_machine_precision=2,
            classical_inputs=inputs,
        )
    with pytest.raises(InteropConversionError, match="at most four rounds"):
        _synthesize(original, classical_inputs={"k": 5})
