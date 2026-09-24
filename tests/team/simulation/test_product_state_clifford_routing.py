from __future__ import annotations

import random

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import Instruction
from flagquantum.simulation.statevector.operations import _compile_statevector_program
from flagquantum.simulation.statevector.product_state import (
    product_state_execution_is_beneficial,
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
