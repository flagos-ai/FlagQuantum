from __future__ import annotations

import random

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.statevector.clifford_matching as clifford_matching
import flagquantum.simulation.statevector.local as statevector_local
import flagquantum.simulation.statevector.product_state as product_state
from flagquantum.core.ir import Instruction
from flagquantum.simulation.statevector.operations import _compile_statevector_program
from flagquantum.simulation.statevector.product_state import (
    product_state_execution_is_beneficial,
)
from flagquantum.simulation.statevector.program import (
    _StatevectorCliffordMatchingStep,
    _StatevectorCXSequenceStep,
    _StatevectorGateStep,
)

pytestmark = pytest.mark.unit


def _random_clifford(n_wires: int, dtype: torch.dtype) -> fq.Circuit:
    circuit = fq.Circuit(n_wires, dtype=dtype)
    generator = random.Random(7319 + 10_007 * n_wires)
    for _ in range(4):
        for wire in range(n_wires):
            getattr(circuit, generator.choice(("h", "s", "x")))(wire)
        wires = list(range(n_wires))
        generator.shuffle(wires)
        for index in range(0, n_wires - 1, 2):
            getattr(circuit, generator.choice(("cx", "cz")))(
                wires[index], wires[index + 1]
            )
    return circuit


def _program(circuit: fq.Circuit):
    return _compile_statevector_program(
        circuit._instructions,
        circuit.n_qubits,
        enable_triton_loop=False,
    )


def test_static_clifford_plan_uses_the_measured_product_state_cost_bound() -> None:
    circuit = _random_clifford(18, torch.complex128)

    assert product_state_execution_is_beneficial(_program(circuit), 18)


def test_parameterized_plan_keeps_the_conservative_cost_bound() -> None:
    circuit = _random_clifford(18, torch.complex128)
    instructions = list(circuit._instructions)
    instructions[0] = Instruction("ry", instructions[0].wires, params={"theta": 0.2})
    program = _compile_statevector_program(
        tuple(instructions),
        circuit.n_qubits,
        enable_triton_loop=False,
    )

    assert not product_state_execution_is_beneficial(program, 18)


def test_custom_matrix_named_like_a_clifford_gate_keeps_conservative_routing() -> None:
    circuit = _random_clifford(18, torch.complex128)
    instructions = list(circuit._instructions)
    instructions[0] = Instruction(
        "h",
        instructions[0].wires,
        matrix=torch.eye(2, dtype=torch.complex128),
    )
    program = _compile_statevector_program(
        tuple(instructions),
        circuit.n_qubits,
        enable_triton_loop=False,
    )

    assert not product_state_execution_is_beneficial(program, 18)


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_static_clifford_product_state_matches_dense_rollback(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    monkeypatch.setenv("FQ_CPU_PRODUCT_STATE_EXECUTION", "0")
    dense_circuit = _random_clifford(18, dtype)
    dense = dense_circuit.state(refresh=True)

    monkeypatch.setenv("FQ_CPU_PRODUCT_STATE_EXECUTION", "1")
    product_circuit = _random_clifford(18, dtype)
    product = product_circuit.state(refresh=True)

    tolerance = 2e-6 if dtype == torch.complex64 else 1e-12
    assert torch.allclose(product, dense, atol=tolerance, rtol=tolerance)
    assert any(
        key[0] == "cpu_product_state" for key in product_circuit._backend_programs
    )
    assert not any(key[0] == "statevector" for key in product_circuit._backend_programs)


def test_fixed_clifford_kernel_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_PRODUCT_STATE_FIXED_CLIFFORD", "0")

    def unexpected_fixed_kernel(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("fixed Clifford kernel must respect its rollback switch")

    monkeypatch.setattr(
        product_state,
        "_apply_fixed_clifford_gate",
        unexpected_fixed_kernel,
    )

    state = _random_clifford(18, torch.complex128).state(refresh=True)

    assert state.shape == (1, 2**18)


def test_product_state_routes_h_and_s_through_fixed_clifford_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: set[str] = set()
    apply_fixed = product_state._apply_fixed_clifford_gate

    def record_fixed_kernel(
        state: torch.Tensor,
        name: str,
        qubit: int,
        n_qubits: int,
    ) -> torch.Tensor:
        observed.add(name)
        return apply_fixed(state, name, qubit, n_qubits)

    monkeypatch.setattr(
        product_state,
        "_apply_fixed_clifford_gate",
        record_fixed_kernel,
    )

    _random_clifford(18, torch.complex128).state(refresh=True)

    assert {"h", "s"} <= observed


def test_disjoint_mixed_clifford_matching_batches_each_gate_kind() -> None:
    instructions = (
        Instruction("cx", (0, 1)),
        Instruction("cz", (2, 3)),
        Instruction("cx", (4, 5)),
        Instruction("cz", (6, 7)),
    )

    program = _compile_statevector_program(
        instructions,
        8,
        enable_triton_loop=False,
        enable_cpu_disjoint_clifford_matching=True,
    )

    assert [
        step.instruction.name
        for step in program[:2]
        if isinstance(step, _StatevectorGateStep)
    ] == ["cz", "cz"]
    assert program[2] == _StatevectorCXSequenceStep(
        controls=(0, 4),
        targets=(1, 5),
    )


def test_product_state_compiles_mixed_clifford_matchings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        statevector_local, "native_clifford_matching_enabled", lambda: True
    )
    circuit = _random_clifford(18, torch.complex128)

    circuit.state(refresh=True)
    program = next(
        program
        for key, program in circuit._backend_programs.items()
        if key[0] == "cpu_product_state"
    )

    assert (
        sum(isinstance(step, _StatevectorCliffordMatchingStep) for step in program) == 4
    )


def test_wide_product_component_attempts_native_mixed_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    widths: list[int] = []

    def record_native(
        step: _StatevectorCliffordMatchingStep,
        state: torch.Tensor,
        *,
        n_qubits: int,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        del step, kwargs
        widths.append(n_qubits)
        return state.clone(), None

    monkeypatch.setattr(
        product_state,
        "apply_native_clifford_matching",
        record_native,
    )
    monkeypatch.setattr(
        statevector_local, "native_clifford_matching_enabled", lambda: True
    )

    circuit = _random_clifford(18, torch.complex128)
    circuit.state(refresh=True)

    assert widths
    assert min(widths) >= 16
    assert max(widths) == 18
    assert circuit._last_statevector_runtime["native_cpu_clifford_matching_regions"]


def test_native_matching_switch_restores_the_previous_product_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clifford_matching,
        "native_cpu_clifford_matching_available",
        lambda: True,
    )
    monkeypatch.setenv("FQ_CPU_NATIVE_CLIFFORD_MATCHING", "0")
    circuit = _random_clifford(18, torch.complex128)

    circuit.state(refresh=True)
    program = next(
        program
        for key, program in circuit._backend_programs.items()
        if key[0] == "cpu_product_state"
    )

    assert not any(
        isinstance(step, _StatevectorCliffordMatchingStep) for step in program
    )
    assert sum(isinstance(step, _StatevectorCXSequenceStep) for step in program) == 4


def test_native_matching_switch_isolated_in_product_program_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clifford_matching,
        "native_cpu_clifford_matching_available",
        lambda: True,
    )
    monkeypatch.setattr(
        product_state,
        "apply_native_clifford_matching",
        lambda *args, **kwargs: (None, None),
    )
    circuit = _random_clifford(18, torch.complex128)

    monkeypatch.setenv("FQ_CPU_NATIVE_CLIFFORD_MATCHING", "1")
    enabled = circuit.state(refresh=True).clone()
    monkeypatch.setenv("FQ_CPU_NATIVE_CLIFFORD_MATCHING", "0")
    rollback = circuit.state(refresh=True).clone()

    product_programs = {
        key: program
        for key, program in circuit._backend_programs.items()
        if key[0] == "cpu_product_state"
    }
    assert len(product_programs) == 2
    assert any(
        any(isinstance(step, _StatevectorCliffordMatchingStep) for step in program)
        for program in product_programs.values()
    )
    assert any(
        not any(isinstance(step, _StatevectorCliffordMatchingStep) for step in program)
        for program in product_programs.values()
    )
    assert torch.equal(enabled, rollback)


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_batched_fixed_clifford_layer_matches_sequential_kernels(
    dtype: torch.dtype,
) -> None:
    state = torch.randn(1, 64, dtype=dtype)
    component = product_state._ProductComponent(tuple(range(6)), state)
    gates = (("h", 0), ("s", 1), ("x", 2), ("s", 4), ("x", 5))
    expected = state
    for name, wire in gates:
        if name == "x":
            expected = product_state._apply_fixed_permutation(
                expected, name, (wire,), 6
            )
        else:
            expected = product_state._apply_fixed_clifford_gate(expected, name, wire, 6)

    actual = product_state._apply_fixed_clifford_layer(component, gates)

    assert torch.equal(actual, expected)


def test_wide_product_component_uses_native_static_clifford_with_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = torch.randn(1, 2**16, dtype=torch.complex128)
    component = product_state._ProductComponent(tuple(range(16)), state)
    gates = tuple(("h" if wire % 3 == 0 else "s", wire) for wire in range(16))
    observed: list[tuple[torch.dtype, torch.dtype, int]] = []

    def record_native(
        output: torch.Tensor,
        gate_codes: torch.Tensor,
        qubits: torch.Tensor,
        *,
        n_qubits: int,
    ) -> bool:
        observed.append((gate_codes.dtype, qubits.dtype, n_qubits))
        return False

    monkeypatch.setattr(product_state, "fused_static_clifford_layer_", record_native)
    product_state._apply_fixed_clifford_layer(component, gates)

    assert observed == [(torch.int8, torch.int64, 16)]

    monkeypatch.setenv("FQ_CPU_PRODUCT_STATE_NATIVE_STATIC_CLIFFORD", "0")
    observed.clear()
    product_state._apply_fixed_clifford_layer(component, gates)

    assert observed == []


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_native_product_static_clifford_matches_exact_rollback(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    circuit = _random_clifford(18, dtype)
    monkeypatch.setenv("FQ_CPU_PRODUCT_STATE_NATIVE_STATIC_CLIFFORD", "0")
    expected = circuit.state(refresh=True)
    monkeypatch.setenv("FQ_CPU_PRODUCT_STATE_NATIVE_STATIC_CLIFFORD", "1")
    actual = circuit.state(refresh=True)

    tolerance = 2e-6 if dtype == torch.complex64 else 2e-14
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_clifford_matching_rollback_disables_layer_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_PRODUCT_STATE_CLIFFORD_MATCHING", "0")

    def unexpected_layer(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("Clifford matching must respect its rollback switch")

    monkeypatch.setattr(
        product_state,
        "_apply_fixed_clifford_layer",
        unexpected_layer,
    )

    state = _random_clifford(18, torch.complex128).state(refresh=True)

    assert state.shape == (1, 2**18)
