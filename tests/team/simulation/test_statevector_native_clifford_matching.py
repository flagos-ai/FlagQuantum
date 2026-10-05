"""Contracts for the native CPU disjoint CX/CZ matching path."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

import flagquantum.simulation.statevector.clifford_matching as clifford_matching
from flagquantum import Circuit
from flagquantum.core import Instruction
from flagquantum.simulation.native_cpu import native_cpu_clifford_matching_available
from flagquantum.simulation.statevector.clifford_matching import (
    _encode_clifford_phase_mapping,
    fuse_native_disjoint_clifford_matchings,
)
from flagquantum.simulation.statevector.operations import (
    _compile_statevector_program,
    _cx_sequence_permutation_index,
)
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


def _repeated_matching_circuit(inputs: torch.Tensor) -> Circuit:
    circuit = Circuit(8, bsz=inputs.shape[0], dtype=inputs.dtype, inputs=inputs)
    for _ in range(3):
        circuit.cx(0, 1).cz(2, 3).cx(4, 5).cz(6, 7)
        circuit.h(0).h(2)
    return circuit


def test_native_scalar_matching_has_a_wide_state_and_rollback_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clifford_matching,
        "native_cpu_clifford_matching_available",
        lambda: True,
    )
    instructions = (Instruction("cx", (0, 1)), Instruction("cz", (2, 3)))
    narrow = torch.zeros((1, 2**15), dtype=torch.complex128)
    wide = torch.zeros((1, 2**16), dtype=torch.complex128)

    assert not clifford_matching.native_clifford_matching_compile_enabled(
        instructions, None, narrow, batch_size=1
    )
    assert clifford_matching.native_clifford_matching_compile_enabled(
        instructions, None, wide, batch_size=1
    )

    monkeypatch.setenv("FQ_CPU_NATIVE_SCALAR_CLIFFORD_MATCHING", "0")
    assert not clifford_matching.native_clifford_matching_compile_enabled(
        instructions, None, wide, batch_size=1
    )


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


def test_native_clifford_matching_reuses_owned_output_with_exact_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_clifford_matching_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    generator = torch.Generator().manual_seed(7719)
    inputs = (
        torch.randn((3, 256), generator=generator)
        + 1j * torch.randn((3, 256), generator=generator)
    ).to(torch.complex128)
    circuit = _repeated_matching_circuit(inputs)

    monkeypatch.setenv("FQ_CPU_CLIFFORD_MATCHING_OUTPUT_REUSE", "0")
    expected = circuit.state(refresh=True)

    requested_outputs: list[torch.Tensor | None] = []
    implementation = clifford_matching.fused_clifford_matching_out

    def record_output(
        state: torch.Tensor,
        cx_mapping: torch.Tensor,
        cz_edges: Sequence[tuple[int, int]] | None,
        n_qubits: int,
        *,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        requested_outputs.append(output)
        return implementation(state, cx_mapping, cz_edges, n_qubits, output=output)

    monkeypatch.setattr(clifford_matching, "fused_clifford_matching_out", record_output)
    monkeypatch.setenv("FQ_CPU_CLIFFORD_MATCHING_OUTPUT_REUSE", "1")
    actual = circuit.state(refresh=True)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(requested_outputs) == 3
    assert requested_outputs[:2] == [None, None]
    assert requested_outputs[2] is not None


def test_native_clifford_phase_map_has_independent_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not native_cpu_clifford_matching_available():
        pytest.skip("native CPU extension is not built in this source checkout")
    generator = torch.Generator().manual_seed(4513)
    inputs = (
        torch.randn((2, 256), generator=generator)
        + 1j * torch.randn((2, 256), generator=generator)
    ).to(torch.complex128)
    circuit = _matching_circuit(inputs)
    monkeypatch.setenv("FQ_CPU_NATIVE_CLIFFORD_MATCHING", "1")

    monkeypatch.setenv("FQ_CPU_NATIVE_CLIFFORD_PHASE_MAP", "0")
    expected = circuit.state(refresh=True)
    monkeypatch.setenv("FQ_CPU_NATIVE_CLIFFORD_PHASE_MAP", "1")
    actual = circuit.state(refresh=True)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (
        circuit._last_statevector_runtime["native_cpu_clifford_matching_regions"] == 1
    )


def test_clifford_phase_map_encodes_sign_and_keeps_wide_mapping_compact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = _StatevectorCliffordMatchingStep(
        controls=(0, 4), targets=(1, 5), cz_edges=((2, 3), (6, 7))
    )
    mapping = _cx_sequence_permutation_index(
        step.controls,
        step.targets,
        8,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    )
    monkeypatch.setenv("FQ_CPU_NATIVE_CLIFFORD_PHASE_MAP", "1")

    encoded, remaining_edges = _encode_clifford_phase_mapping(mapping, step, 8)

    assert remaining_edges is None
    assert torch.any(encoded < 0)
    torch.testing.assert_close(
        torch.where(encoded < 0, -encoded - 1, encoded), mapping, atol=0, rtol=0
    )

    compact = torch.arange(22, dtype=torch.int64)
    wide_encoded, wide_edges = _encode_clifford_phase_mapping(compact, step, 22)
    assert wide_encoded is compact
    assert wide_edges == step.cz_edges


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
