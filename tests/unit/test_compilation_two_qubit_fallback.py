"""Keep narrower folds without solving a rejected block twice."""

from unittest.mock import patch

import pytest
import torch

from flagquantum.compiler.optimization import two_qubit_optimization as optimization
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.test_compilation_target_cz_resynthesis import _operator

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("pair", [(0, 1), (1, 0)])
@pytest.mark.parametrize("scalar_tensor", [False, True])
def test_rejected_entangling_block_is_solved_once(
    pair: tuple[int, int], scalar_tensor: bool
) -> None:
    def angle(value: float) -> float | torch.Tensor:
        return torch.tensor(value, dtype=torch.float64) if scalar_tensor else value

    source = CircuitIR(
        2,
        (
            Instruction("rxx", pair, {"theta": angle(0.17)}),
            Instruction("rzz", pair, {"theta": angle(0.23)}),
        ),
    )
    with patch.object(
        optimization, "_replacement", wraps=optimization._replacement
    ) as solver:
        result = optimization.collapse_two_qubit_blocks(source)
    assert solver.call_count == 1
    assert result.instructions == source.instructions
    torch.testing.assert_close(_operator(result), _operator(source), atol=1e-12, rtol=0)


def test_local_member_keeps_the_narrower_cancellation_opportunities() -> None:
    source = CircuitIR(
        2,
        (
            Instruction(
                "rzz", (0, 1), {"theta": torch.tensor(0.17, dtype=torch.float64)}
            ),
            Instruction(
                "rzz", (0, 1), {"theta": torch.tensor(-0.17, dtype=torch.float64)}
            ),
            Instruction("h", (0,)),
            Instruction(
                "rxx", (0, 1), {"theta": torch.tensor(0.23, dtype=torch.float64)}
            ),
            Instruction(
                "rxx", (0, 1), {"theta": torch.tensor(-0.23, dtype=torch.float64)}
            ),
        ),
    )
    with patch.object(
        optimization, "_replacement", wraps=optimization._replacement
    ) as solver:
        result = optimization.collapse_two_qubit_blocks(source)
    assert solver.call_count == 3
    assert [item.name for item in result.instructions] == ["h"]
    torch.testing.assert_close(_operator(result), _operator(source), atol=1e-12, rtol=0)


def test_declined_matrix_is_read_again_on_the_next_invocation() -> None:
    matrix = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex128))
    source = CircuitIR(2, (Instruction("unitary", (0, 1), matrix=matrix),))
    with patch.object(
        optimization, "_replacement", wraps=optimization._replacement
    ) as solver:
        assert (
            optimization.collapse_two_qubit_blocks(source).instructions
            == source.instructions
        )
        matrix.copy_(torch.eye(4, dtype=torch.complex128))
        assert not optimization.collapse_two_qubit_blocks(source).instructions
    assert solver.call_count == 2
