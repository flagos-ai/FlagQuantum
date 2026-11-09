"""Original raw predicate callbacks and all coherent ladder erasure branches."""

from itertools import product

import numpy as np
import pytest

from flagquantum.simulation.classical_and import (
    and_classical_values,
    multi_and_classical_values,
)
from tests.unit.test_classical_comparison import compare_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("values", list(product((0, 1), repeat=2)))
@pytest.mark.parametrize("uncompute", [False, True])
@pytest.mark.parametrize("target", [None, 0, 1, 2, np.uint8(1), np.array([0, 1])])
@pytest.mark.parametrize(
    "controls",
    [
        np.array(x)
        for x in [(0, 0), (0, 1), (1, 0), (1, 1), (), (1,), (1, 1, 1), ((0, 1), (1, 0))]
    ],
)
def test_original_and_callback_types_failures_and_ignored_forward_target(
    values, uncompute, target, controls
):
    from qualtran.bloqs.mcmt.and_bloq import And

    source = And(*values, uncompute=uncompute)
    compare_calls(
        lambda c, t: tuple(source.on_classical_vals(ctrl=c, target=t).values()),
        lambda c, t: tuple(
            and_classical_values(c, values, target=t, uncompute=uncompute).values()
        ),
        (controls, target),
    )
    if not uncompute and controls.ndim == 1:
        assert and_classical_values(controls, values)["ctrl"] is controls


@pytest.mark.parametrize("values", [(0, 1, 0), (1, 1, 1), (1, 0, 1, 0)])
@pytest.mark.parametrize(
    "controls",
    [
        np.array(x)
        for x in [(0, 1, 0), (1, 1, 1), (), (1,), (0, 1, 0, 1), ((1, 0, 1), (0, 1, 0))]
    ],
)
def test_original_multi_and_broadcast_shape_prefix_and_scalar_dtype(values, controls):
    from qualtran.bloqs.mcmt.and_bloq import MultiAnd

    source = MultiAnd(values)
    compare_calls(
        lambda c: tuple(source.on_classical_vals(ctrl=c).values()),
        lambda c: tuple(multi_and_classical_values(c, values).values()),
        (controls,),
    )


@pytest.mark.parametrize("values", list(product((0, 1), repeat=3)))
def test_complex_prefix_outputs_and_all_erasure_histories(values):
    import torch

    from flagquantum.compiler.registers.control.multi_control import multi_and_ir
    from tests.unit.measurement_checks import forced_branch

    grid = torch.arange(1, 17, dtype=torch.float64)
    amplitudes = torch.complex(grid, 3 * grid - 5)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for data, spectator in product(range(8), (0, 1)):
        controls = np.array([(data >> i) & 1 for i in (2, 1, 0)], dtype=np.uint8)
        output = multi_and_classical_values(controls, values)
        prefix = (int(output["junk"][0]) << 1) | int(output["target"])
        initial[(data << 3) | spectator] = amplitudes[2 * data + spectator]
        expected[(data << 3) | (prefix << 1) | spectator] = amplitudes[
            2 * data + spectator
        ]
        assert output["ctrl"] is controls
    forward = multi_and_ir((0, 1, 2), values, (3, 4))
    erase = multi_and_ir(
        (0, 1, 2), values, (3, 4), classical_bits=(0, 1), uncompute=True
    )
    torch.testing.assert_close(
        forced_branch(initial, forward, ()), expected, rtol=0, atol=2e-12
    )
    for history in product((0, 1), repeat=2):
        actual = forced_branch(initial, forward + erase, history)
        torch.testing.assert_close(actual, initial / 2, rtol=0, atol=2e-12)
        residual = actual.reshape(8, 4, 2)[:, 1:, :]
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), rtol=0, atol=2e-12
        )


@pytest.mark.parametrize("values", [(0, 1, 0), (1, 1, 1), (1, 0, 1, 0)])
def test_source_model_and_existing_native_ladder_cost(values):
    from qualtran.bloqs.mcmt.and_bloq import MultiAnd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler.registers.control.multi_control import multi_and_ir

    n = len(values)
    controls, work = tuple(range(n)), tuple(range(n, 2 * n - 1))
    forward = multi_and_ir(controls, values, work)
    erase = multi_and_ir(
        controls, values, work, classical_bits=tuple(range(n - 1)), uncompute=True
    )
    source = MultiAnd(values)
    model = get_cost_value(source, QECGatesCost())
    reverse = get_cost_value(source.adjoint(), QECGatesCost())
    assert (
        sum(g.name in ("t", "tdg") for g in forward)
        == model.total_t_count()
        == 4 * (n - 1)
    )
    assert sum(g.name == "measure" for g in erase) == reverse.measurement == n - 1
    assert model.measurement == reverse.total_t_count() == 0
