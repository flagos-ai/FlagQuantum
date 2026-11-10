"""Owned X measurement matches the original joint spectator instrument."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_measurement import measure_x_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("wire", [0, 1])
def test_owned_x_measurement_preserves_original_joint_branch_density(wire: int) -> None:
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    bound = bind_register_program(
        ProgramRecord((measure_x_word_function("main"),)), ((wire,),), clean=(2,)
    )
    assert Counter(g.name for g in bound.circuit.instructions) == {"h": 1, "measure": 1}
    assert bound.outputs == ((),)
    assert bound.classical_outputs == ((0,),)
    state = torch.zeros(8, dtype=torch.complex128)
    state[::2] = torch.randn(
        4, dtype=torch.complex128, generator=torch.Generator().manual_seed(108430)
    )
    state /= state.norm()
    tensor, _ = MeasureX().my_tensors({"q": "q"}, {"c": "c"})
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for outcome in (0, 1):
        branch = (
            forced_branch(state, bound.circuit.instructions, (outcome,))
            .numpy()
            .reshape(2, 2, 2)
        )
        assert np.count_nonzero(branch[..., 1]) == 0
        source = np.moveaxis(
            np.tensordot(
                tensor.data[:, :, outcome].T,
                state.numpy().reshape(2, 2, 2),
                axes=([1], [wire]),
            ),
            0,
            wire,
        )
        restored = np.moveaxis(np.tensordot(h, branch, axes=([1], [wire])), 0, wire)
        np.testing.assert_allclose(restored, source, rtol=0, atol=3e-16)
        observed = np.moveaxis(branch, wire, 0).reshape(2, -1)
        expected = np.moveaxis(source, wire, 0).reshape(2, -1)
        np.testing.assert_allclose(
            np.einsum("ki,kj->ij", observed, observed.conj()),
            np.einsum("ki,kj->ij", expected, expected.conj()),
            rtol=0,
            atol=3e-16,
        )
    model = get_cost_value(MeasureX(), QECGatesCost())
    assert model.measurement == 1 and model.total_t_count() == 0
