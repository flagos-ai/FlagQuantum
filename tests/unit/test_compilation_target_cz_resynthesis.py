"""Independent operator checks for the opt-in, target-CZ IR transformation."""

from __future__ import annotations

import random

import pytest
import torch

from flagquantum.compiler import compile
from flagquantum.compiler.optimization import _cz_blocks
from flagquantum.compiler.optimization._cz_blocks import resynthesize_cz_blocks
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter, ParameterExpression
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit


def _operator(program: CircuitIR) -> torch.Tensor:
    """Embed simulation matrices by explicit basis-bit indexing, not compiler math."""
    size = 2**program.n_wires
    result = torch.eye(size, dtype=torch.complex128)
    for instruction in program.instructions:
        width = 2 ** len(instruction.wires)
        local = gate_matrix(
            instruction, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
        ).reshape(-1, width, width)[0]
        embedded = torch.zeros((size, size), dtype=torch.complex128)
        for column in range(size):
            local_column = 0
            for wire in instruction.wires:
                local_column = (local_column << 1) | (
                    (column >> (program.n_wires - wire - 1)) & 1
                )
            for local_row in range(width):
                row = column
                for position, wire in enumerate(instruction.wires):
                    bit = (local_row >> (len(instruction.wires) - position - 1)) & 1
                    mask = 1 << (program.n_wires - wire - 1)
                    row = (row & ~mask) | (bit * mask)
                embedded[row, column] = local[local_row, local_column]
        result = embedded @ result
    return result


def _assert_equivalent(source: CircuitIR, result: CircuitIR) -> None:
    expected, actual = _operator(source), _operator(result)
    phase = torch.trace(actual.conj().T @ expected) / expected.shape[0]
    assert float(abs(abs(phase) - 1)) < 1e-10
    torch.testing.assert_close(expected, phase * actual, atol=1e-10, rtol=0)


@pytest.mark.parametrize("pair", [(0, 1), (1, 0)])
def test_swap_and_reversed_cx_reduce_cost_without_absorbing_layout(
    pair: tuple[int, int],
) -> None:
    source = CircuitIR(
        2,
        (Instruction("swap", pair), Instruction("cx", pair[::-1])),
        metadata={"final_layout": (0, 1)},
    )
    result = resynthesize_cz_blocks(source)
    assert sum(item.name == "cz" for item in result.instructions) == 2
    assert result.metadata == source.metadata
    assert all(item.name in {"rz", "rx", "cz"} for item in result.instructions)
    _assert_equivalent(source, result)
    assert resynthesize_cz_blocks(result) is result


@pytest.mark.parametrize("seed", range(12))
def test_numeric_blocks_with_reversed_wires_and_spectators(seed: int) -> None:
    rng = random.Random(seed)
    instructions: list[Instruction] = []
    for _ in range(5):
        instructions.extend(
            (
                Instruction("cx", rng.choice(((0, 2), (2, 0)))),
                Instruction("ry", (rng.choice((0, 2)),), {"theta": rng.uniform(-3, 3)}),
                Instruction("rz", (1,), {"theta": rng.uniform(-3, 3)}),
            )
        )
    source = CircuitIR(3, tuple(instructions))
    result = resynthesize_cz_blocks(source)
    assert sum(item.name == "cz" for item in result.instructions) <= 3
    assert [item for item in result.instructions if item.wires == (1,)] == [
        item for item in source.instructions if item.wires == (1,)
    ]
    _assert_equivalent(source, result)


def test_shared_wire_closes_blocks_before_different_pair() -> None:
    source = CircuitIR(
        3,
        (
            Instruction("swap", (0, 1)),
            Instruction("cx", (1, 0)),
            Instruction("cx", (1, 2)),
            Instruction("swap", (0, 1)),
            Instruction("cx", (0, 1)),
        ),
    )
    _assert_equivalent(source, resynthesize_cz_blocks(source))


def test_disjoint_blocks_can_be_interleaved() -> None:
    source = CircuitIR(
        4,
        (
            Instruction("swap", (0, 2)),
            Instruction("swap", (1, 3)),
            Instruction("cx", (2, 0)),
            Instruction("cx", (3, 1)),
        ),
    )
    result = resynthesize_cz_blocks(source)
    assert sum(item.name == "cz" for item in result.instructions) == 4
    _assert_equivalent(source, result)


@pytest.mark.parametrize(
    "boundary",
    [
        Instruction("barrier", (0, 1), metadata={"is_dynamic": True}),
        Instruction("reset", (0,), metadata={"is_dynamic": True}),
    ],
)
def test_ordering_and_nonunitary_boundaries_are_not_crossed(
    boundary: Instruction,
) -> None:
    source = CircuitIR(
        2,
        (Instruction("swap", (0, 1)), boundary, Instruction("cx", (1, 0))),
    )
    assert resynthesize_cz_blocks(source) is source


@pytest.mark.parametrize("trainable", [False, True])
def test_tensor_parameter_and_its_gradient_survive_block_boundaries(
    trainable: bool,
) -> None:
    theta = torch.tensor([0.2, 0.4] if not trainable else 0.2, requires_grad=trainable)
    held = Instruction("rx", (0,), {"theta": theta})
    source = CircuitIR(
        2, (Instruction("swap", (0, 1)), held, Instruction("cx", (1, 0)))
    )
    result = resynthesize_cz_blocks(source)
    assert any(item is held for item in result.instructions)
    assert held.params["theta"] is theta
    if trainable:
        _operator(result).real.sum().backward()
        assert theta.grad is not None
        assert torch.isfinite(theta.grad)


def test_routing_evidence_and_custom_matrix_are_preserved() -> None:
    routing = Instruction("swap", (0, 1), metadata={"routing_phase": "forward"})
    custom = Instruction("cx", (1, 0), matrix=torch.eye(4, dtype=torch.complex128))
    source = CircuitIR(2, (routing, custom, Instruction("cx", (0, 1))))
    assert resynthesize_cz_blocks(source) is source


@pytest.mark.parametrize("angle", [Parameter("theta"), 2 * Parameter("theta") + 0.1])
def test_symbolic_angles_survive_optimization_on_both_sides(
    angle: Parameter | ParameterExpression,
) -> None:
    held = Instruction("rx", (0,), {"theta": angle})
    pair = (Instruction("swap", (0, 1)), Instruction("cx", (1, 0)))
    source = CircuitIR(2, (*pair, held, *pair))
    result = resynthesize_cz_blocks(source)
    assert any(item is held for item in result.instructions)
    assert sum(item.name == "cz" for item in result.instructions) == 4
    from flagquantum import Circuit

    _assert_equivalent(
        Circuit.from_ir(source).bind_parameters({"theta": 0.37}).to_ir(),
        Circuit.from_ir(result).bind_parameters({"theta": 0.37}).to_ir(),
    )


def test_incorrect_synthesis_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cz_blocks, "synthesize_two_qubit", lambda *args, **kwargs: ())
    source = CircuitIR(2, (Instruction("swap", (0, 1)), Instruction("cx", (1, 0))))
    assert resynthesize_cz_blocks(source) is source


def test_excessive_instruction_growth_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tuple(Instruction("x", (0,)) for _ in range(256))
    monkeypatch.setattr(
        _cz_blocks, "synthesize_two_qubit", lambda *args, **kwargs: candidate
    )
    source = CircuitIR(2, (Instruction("swap", (0, 1)), Instruction("cx", (1, 0))))
    assert resynthesize_cz_blocks(source) is source


def test_long_blocks_are_bounded_and_semantically_equivalent() -> None:
    source = CircuitIR(
        2,
        tuple(
            item
            for _ in range(100)
            for item in (Instruction("swap", (0, 1)), Instruction("cx", (1, 0)))
        ),
    )
    result = resynthesize_cz_blocks(source)
    assert len(result.instructions) < len(source.instructions)
    _assert_equivalent(source, result)


def test_existing_explicit_pass_extension_runs_end_to_end() -> None:
    source = CircuitIR(2, (Instruction("swap", (0, 1)), Instruction("cx", (1, 0))))
    result = compile(
        source, optimization_level=0, passes={"experiment.cz": resynthesize_cz_blocks}
    )
    assert "experiment.cz" in result.metadata["admitted_passes"]
    assert sum(item.name == "cz" for item in result.instructions) == 2
    _assert_equivalent(source, result)
