"""Contracts for the native CPU disjoint CX/CZ matching path."""

from __future__ import annotations

import pytest
import torch

from flagquantum import Circuit
from flagquantum.core import Instruction
from flagquantum.simulation.native_cpu import native_cpu_clifford_matching_available
from flagquantum.simulation.statevector.clifford_matching import (
    fuse_native_disjoint_clifford_matchings,
)
from flagquantum.simulation.statevector.operations import _compile_statevector_program
from flagquantum.simulation.statevector.program import (
    _StatevectorCliffordMatchingStep,
    _StatevectorGateStep,
)

pytestmark = pytest.mark.unit


def _matching_circuit(inputs: torch.Tensor) -> Circuit:
    circuit = Circuit(8, bsz=inputs.shape[0], dtype=inputs.dtype, inputs=inputs)
    circuit.cx(0, 1)
    circuit.cz(2, 3)
    circuit.cx(4, 5)
    circuit.cz(6, 7)
    return circuit


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_native_clifford_matching_matches_rollback_and_preserves_input(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(3527)
    inputs = (
        torch.randn((3, 256), generator=generator)
        + 1j * torch.randn((3, 256), generator=generator)
    ).to(dtype)
    original = inputs.clone()
    circuit = _matching_circuit(inputs)

    monkeypatch.setenv("FQ_CPU_NATIVE_CLIFFORD_MATCHING", "0")
    expected = circuit.state(refresh=True)
    assert circuit._last_statevector_runtime["statevector_apply_count"] == 4
    assert (
        circuit._last_statevector_runtime["native_cpu_clifford_matching_regions"] == 0
    )

    monkeypatch.setenv("FQ_CPU_NATIVE_CLIFFORD_MATCHING", "1")
    actual = circuit.state(refresh=True)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(inputs, original, atol=0, rtol=0)
    if native_cpu_clifford_matching_available():
        assert circuit._last_statevector_runtime["statevector_apply_count"] == 1
        assert (
            circuit._last_statevector_runtime["native_cpu_clifford_matching_regions"]
            == 1
        )


def test_native_clifford_matching_does_not_bypass_autograd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = torch.randn((2, 256), dtype=torch.complex128, requires_grad=True)
    circuit = _matching_circuit(inputs)
    monkeypatch.setenv("FQ_CPU_NATIVE_CLIFFORD_MATCHING", "1")

    output = circuit.state(refresh=True)
    output.real.sum().backward()

    assert inputs.grad is not None
    assert (
        circuit._last_statevector_runtime["native_cpu_clifford_matching_regions"] == 0
    )


def test_mixed_disjoint_matching_compiles_to_one_step() -> None:
    instructions = tuple(
        Instruction(name, wires)
        for name, wires in (
            ("cx", (0, 1)),
            ("cz", (2, 3)),
            ("cx", (4, 5)),
            ("cz", (6, 7)),
        )
    )
    program = _compile_statevector_program(
        instructions,
        8,
        enable_triton_loop=False,
        enable_cpu_native_clifford_matching=True,
    )

    assert program == (
        _StatevectorCliffordMatchingStep(
            controls=(0, 4),
            targets=(1, 5),
            cz_edges=((2, 3), (6, 7)),
        ),
    )


def test_matching_stops_at_an_overlapping_wire() -> None:
    layout = ((), ())
    program = tuple(
        _StatevectorGateStep(Instruction(name, wires), layout)
        for name, wires in (
            ("cx", (0, 1)),
            ("cz", (2, 3)),
            ("cx", (3, 4)),
            ("cz", (5, 6)),
            ("cz", (7, 8)),
        )
    )

    fused = fuse_native_disjoint_clifford_matchings(program)

    assert fused[0] == _StatevectorCliffordMatchingStep(
        controls=(0,), targets=(1,), cz_edges=((2, 3),)
    )
    assert fused[1] == _StatevectorCliffordMatchingStep(
        controls=(3,), targets=(4,), cz_edges=((5, 6), (7, 8))
    )


def test_pure_cz_matching_is_left_for_the_cz_graph_compiler() -> None:
    layout = ((), ())
    program = tuple(
        _StatevectorGateStep(Instruction("cz", wires), layout)
        for wires in ((0, 1), (2, 3))
    )

    fused = fuse_native_disjoint_clifford_matchings(program)

    assert fused == list(program)
