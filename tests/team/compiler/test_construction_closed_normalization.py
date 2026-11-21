"""Closed composition reuses exact rewrites before consuming later gate budgets."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.codegen.registers import (
    schedule_register_program,
)
from flagquantum.compiler._construction.frontend.program import capture_program
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.unit


def test_closed_helpers_remove_inverse_rotations_without_losing_global_sign() -> None:
    entry = program_definition(
        "from flagquantum.synth import phase, RZ, within, apply\n"
        "@qfunc\ndef main(c:Output[QNum[1]],q:Output[QNum[1]]):\n"
        "    allocate(c)\n"
        "    allocate(q)\n"
        "    hadamard_transform(q)\n"
        "    for i in range(31):\n"
        "        with within():\n"
        "            flag:QNum[1] = c == 0\n"
        "        with apply():\n"
        "            phase(q,0.19)\n"
        "            phase(q,-0.19)\n"
        "        RZ(2*3.141592653589793,q)\n"
    )
    program, functions = capture_program(entry)
    candidate = schedule_register_program(
        program, functions, n_qubits=4, max_operations=64
    )
    state = fq.run(candidate.circuit).statevector().reshape(4, -1)
    assert torch.count_nonzero(state[:, 1:]) == 0
    state = state[:, 0]
    expected = torch.tensor(
        [-1 / math.sqrt(2), -1 / math.sqrt(2), 0, 0], dtype=torch.complex128
    )
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    assert len(candidate.circuit.instructions) <= 32
