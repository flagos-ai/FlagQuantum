from __future__ import annotations

import random

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.statevector.product_state as product_state
from flagquantum.core.ir import Instruction
from flagquantum.simulation.statevector.operations import _compile_statevector_program
from flagquantum.simulation.statevector.product_state import (
    product_state_execution_is_beneficial,
)
from flagquantum.simulation.statevector.program import (
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
        circuit.n_wires,
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
        circuit.n_wires,
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
        circuit.n_wires,
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
        wire: int,
        n_wires: int,
    ) -> torch.Tensor:
        observed.add(name)
        return apply_fixed(state, name, wire, n_wires)

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
