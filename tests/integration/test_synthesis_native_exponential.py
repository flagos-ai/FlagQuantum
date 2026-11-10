"""Partial exponential preparation preserves the approved quantum computation."""

import math
import re
from pathlib import Path
from typing import Annotated, TypedDict

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.autocallable_options.partial_exponential_state_preparation import (
    main as native,
)
from examples.synthesis.applications.finance.autocallable_options.partial_exponential_state_preparation.comparisons.circuit import (
    input_circuit,
    reference_circuit,
    shift_matrix,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem.contracts import InteropConversionError
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    R,
    adjoint,
    allocate,
    hadamard_transform,
    inplace_prepare_amplitudes,
    phase,
    qfunc,
)

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
RELATIVE = "applications/finance/autocallable_options/partial_exponential_state_preparation.qmod"
FIXTURES = ROOT / "tests/fixtures/classiq_qmod/corpus"


class Limits(TypedDict, total=False):
    max_qubits: int
    max_workspace: int
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


def overlap(rate: float) -> float:
    weights = [math.exp(-rate * word) for word in range(32)]
    return math.sqrt(math.fsum(weights[21:]) / math.fsum(weights))


def state(program: synth.Program, size: int) -> torch.Tensor:
    raw = fq.run(program).statevector()
    assert isinstance(raw, torch.Tensor)
    viewed = raw.reshape(size, -1)
    assert float(viewed[:, 1:].abs().square().sum()) < 1e-22
    return viewed[:, 0]


@pytest.mark.parametrize("rate", [-0.25, -0.5, -0.75])
def test_full_native_approved_and_independent_circuit(rate: float) -> None:
    success = overlap(rate)
    program = native.build_program(rate, success)
    actual = state(program, 64).reshape(32, 2)
    expected = fq.run(reference_circuit(rate, success)).statevector().reshape(32, 2)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    assert float(actual[:, 1].abs().square().sum()) < 1e-22
    # The retained marked interval [21,31] shifts to [3,13].
    assert (
        float(torch.cat((actual[:3, 0], actual[14:, 0])).abs().square().sum()) < 1e-22
    )
    source = (FIXTURES / "approved" / RELATIVE).read_text()
    source = re.sub(
        r"exact_amplitude_amplification\([0-9.]+,",
        f"exact_amplitude_amplification({success!r},",
        source,
    )
    source = source.replace(
        "prepare_exponential_state(-0.5,", f"prepare_exponential_state({rate!r},"
    )
    approved = synth.synthesize(source, source_format="qmod")
    torch.testing.assert_close(state(approved, 32), expected[:, 0], atol=1e-11, rtol=0)


def test_original_refusal_and_default_model() -> None:
    original = (FIXTURES / "original" / RELATIVE).read_text()
    approved = (FIXTURES / "approved" / RELATIVE).read_text()
    assert "x >= 21" in original and "x += -18" in original
    assert (
        "allocate(5, x)" in original and "prepare_exponential_state(-0.5," in original
    )
    assert original.replace("0.996625424765961", "0.99795457844373747") == approved
    with pytest.raises(InteropConversionError, match="supplied amplitude.*disagrees"):
        synth.synthesize(FIXTURES / "original" / RELATIVE)
    with pytest.raises(ValueError, match="success amplitude"):
        native.build_program(success_amplitude=0.996625424765961)
    assert abs(overlap(-0.5) - native.SUCCESS_AMPLITUDE) < 1e-14


@pytest.mark.parametrize(
    "width,rate", [(1, -0.5), (5, 0.0), (5, -0.5), (5, 0.75), (8, -0.25)]
)
def test_exponential_input_against_product_odds(width: int, rate: float) -> None:
    amplitudes = native.exponential_amplitudes(rate, width)
    expected = fq.run(input_circuit(rate, width)).statevector().flatten()
    torch.testing.assert_close(
        torch.tensor(amplitudes, dtype=torch.complex128), expected, atol=1e-14, rtol=0
    )
    assert abs(math.fsum(a * a for a in amplitudes) - 1) < 1e-14


@qfunc
def restore(amplitudes: list[float], phi: float, x: Output[Annotated[QNum, 5]]) -> None:
    allocate(x)
    hadamard_transform(x)
    phase(x, phi)
    native.prepare(amplitudes, x)
    with adjoint():
        native.prepare(amplitudes, x)


@pytest.mark.parametrize("phi", [0.37, -0.61])
def test_preparation_inverse_on_complex_nonzero_input(phi: float) -> None:
    program = synth.synthesize(
        restore,
        classical_inputs={
            "amplitudes": native.exponential_amplitudes(-0.5, 5),
            "phi": phi,
        },
    )
    expected = torch.exp(1j * phi * torch.arange(32, dtype=torch.float64)) / math.sqrt(
        32
    )
    torch.testing.assert_close(state(program, 32), expected, atol=1e-11, rtol=0)


@qfunc
def shifted(
    amplitudes: list[float],
    phi: float,
    x: Output[Annotated[QNum, 5]],
    spectator: Output[QBit],
) -> None:
    allocate(x)
    allocate(spectator)
    inplace_prepare_amplitudes(amplitudes, 0, x)
    phase(x, phi)
    R(0.83, math.pi / 2, spectator)
    phase(spectator, -phi)
    native.shift(x)


@pytest.mark.parametrize("phi", [0.23, -0.47])
def test_modular_wrap_all_amplitudes_and_spectator(phi: float) -> None:
    amplitudes = [
        (word + 1) / math.sqrt(sum(i * i for i in range(1, 33))) for word in range(32)
    ]
    p = synth.synthesize(
        shifted, classical_inputs={"amplitudes": amplitudes, "phi": phi}
    )
    values = torch.tensor(amplitudes, dtype=torch.complex128) * torch.exp(
        1j * phi * torch.arange(32, dtype=torch.float64)
    )
    spectator = torch.tensor(
        [
            math.cos(0.83 / 2),
            math.sin(0.83 / 2) * complex(math.cos(phi), -math.sin(phi)),
        ],
        dtype=torch.complex128,
    )
    expected = torch.kron(shift_matrix() @ values, spectator)
    torch.testing.assert_close(state(p, 64), expected, atol=1e-11, rtol=0)


@qfunc
def aliased(x: Output[QBit]) -> None:
    allocate(x)
    native.reflect_zero(x, x)


def test_cross_call_alias_refuses() -> None:
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(aliased)


@pytest.mark.parametrize(
    "rate,width",
    [
        (float("nan"), 5),
        (float("inf"), 5),
        (True, 5),
        (17, 5),
        (-0.5, 0),
        (-0.5, 9),
        (-0.5, True),
    ],
)
def test_invalid_input_profile_refuses(rate: float, width: int) -> None:
    with pytest.raises(ValueError, match="rate|width"):
        native.exponential_amplitudes(rate, width)


@pytest.mark.parametrize("success", [float("nan"), float("inf"), True, 0.0, 0.9])
def test_invalid_overlap_refuses(success: float) -> None:
    with pytest.raises(ValueError, match="amplitude"):
        native.build_program(success_amplitude=success)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_qubits": 7},
        {"max_workspace": 1},
        {"max_operations": 2398},
        {"max_depth": 2045},
        {"memory_limit_bytes": 1024},
    ],
)
def test_resource_budgets_refuse(limits: Limits) -> None:
    with pytest.raises(
        (PlanningError, HybridVerificationError),
        match="budget|limit|requires|exceed|infeasible",
    ):
        synth.synthesize(
            native.main, classical_inputs=native.classical_inputs(), **limits
        )
