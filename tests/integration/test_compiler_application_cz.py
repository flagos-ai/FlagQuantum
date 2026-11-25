"""Full-operator application checks for the private native CZ experiment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch

from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.main import (
    build_program as build_maxcut,
)
from examples.synthesis.applications.physical_systems.ising_model.main import (
    build_program as build_ising,
)
from examples.synthesis.functions.qft.main import build_program as build_qft
from flagquantum import synth
from flagquantum.compiler import CouplingMap, compile
from flagquantum.compiler.lowering.basis_translation import translate
from flagquantum.compiler.optimization._cz_blocks import resynthesize_cz_blocks
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.test_compilation_target_cz_resynthesis import _operator

pytestmark = pytest.mark.integration
_NATIVE_BASIS = frozenset(("rx", "rz", "cz"))


def _native_baseline(program: CircuitIR) -> CircuitIR:
    instructions: list[Instruction] = []
    for instruction in program.instructions:
        if instruction.name in _NATIVE_BASIS:
            instructions.append(instruction)
            continue
        leaves = translate(
            instruction,
            can_run=lambda item: item.name in _NATIVE_BASIS,
            z_rotation="rz",
            pulse_opcode="rx",
        )
        assert leaves is not None
        assert all(item.name in _NATIVE_BASIS for item in leaves)
        instructions.extend(leaves)
    return replace(program, instructions=tuple(instructions))


@pytest.mark.parametrize(
    "builder,previous_counts",
    [(build_qft, (156, 324)), (build_maxcut, (462, 798)), (build_ising, (609, 1239))],
)
@pytest.mark.parametrize("line_topology", [False, True], ids=["no-topology", "line"])
def test_application_preserves_complete_operator_and_target_constraints(
    builder: Callable[[], synth.Program],
    previous_counts: tuple[int, int],
    line_topology: bool,
) -> None:
    source = builder().circuit
    routed = compile(
        source,
        coupling_map=CouplingMap.line(source.n_wires) if line_topology else None,
    )
    baseline = _native_baseline(routed)
    # Parent-revision counts retain the CX-mediated decomposition overhead.
    assert len(baseline.instructions) < previous_counts[int(line_topology)]
    result = resynthesize_cz_blocks(baseline)
    expected, actual = _operator(source), _operator(result)
    phase = torch.trace(actual.conj().T @ expected) / expected.shape[0]
    assert float(abs(abs(phase) - 1)) <= 1e-10
    torch.testing.assert_close(expected, phase * actual, atol=1e-10, rtol=0)
    assert all(item.name in _NATIVE_BASIS for item in result.instructions)
    assert sum(item.name == "cz" for item in result.instructions) <= sum(
        item.name == "cz" for item in baseline.instructions
    )
    assert result.metadata == baseline.metadata
    if line_topology:
        assert result.metadata["routing"]["mapping_restored"] is True
        assert all(
            len(item.wires) != 2 or abs(item.wires[0] - item.wires[1]) == 1
            for item in result.instructions
        )
