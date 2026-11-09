"""Direct QVR source rotation order, precision, phases and edge cases."""

from collections import Counter
from math import pi, tau
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.phase.qvr_zpow import qvr_zpow_ir, qvr_zpow_schedule
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "m,f,gamma,epsilon,signed",
    [
        (1, 1, 0.5, 0.01, False),
        (4, 4, 0.1, 0.5, False),
        (5, 2, -0.375, 0.05, False),
        (8, 8, 0.625, 0.03, False),
        (5, 0, 0.125, 0.1, False),
        (4, 2, 0.001, 0.1, False),
        (4, 4, 0.0001, 0.5, False),
        (4, 1, 0.0001, 0.5, True),
    ],
)
def test_direct_qvr_source_schedule_and_full_complex_action(
    m: int, f: int, gamma: float, epsilon: float, signed: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QFxp, Register
    from qualtran.bloqs.basic_gates.rotation import ZPowGate
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrZPow
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = QvrZPow(Register("cost", QFxp(m, f, signed=signed)), gamma, epsilon)
    # Permuted physical binding plus an untouched coherent spectator.
    cost = tuple(reversed(range(m)))
    ports = {"cost": cost}
    grid = torch.arange(1 << (m + 1), dtype=torch.int64)
    weights = grid.to(torch.float64) + 1
    state = torch.complex(weights, weights * (weights % 7 - 3))
    state /= state.norm()
    params = dict(
        cost_fraction_bits=f, coefficient=gamma, epsilon=epsilon, signed=signed
    )
    refs = []
    for adjoint in (False, True):
        original = source.adjoint() if adjoint else source
        expected_terms = []
        exact_angles = torch.zeros_like(weights)

        def leaf(
            bloq: Any,
            inputs: SourcePorts,
            desired: SourcePorts,
            *,
            terms=expected_terms,
            angles: torch.Tensor = exact_angles,
        ) -> SourceResult | None:
            if not isinstance(bloq, ZPowGate):
                return None
            (wire,) = inputs["q"]
            assert not desired or desired == inputs
            terms.append((wire, bloq.exponent, bloq.eps))
            theta = pi * bloq.exponent
            angles.add_(theta * ((grid >> (m - wire)) & 1).to(torch.float64))
            return (Instruction("phase", (wire,), params={"theta": theta}),), inputs

        expected_gates, output = walk_source_graph(original, ports, ports, leaf)
        assert output == ports
        schedule = qvr_zpow_schedule(cost, **params, adjoint=adjoint)
        gates = qvr_zpow_ir(cost, **params, adjoint=adjoint)
        assert schedule == tuple(expected_terms)
        assert gates == expected_gates
        assert Counter(g.name for g in gates) == Counter({"phase": len(schedule)})
        refs.append(gates)
        torch.testing.assert_close(
            forced_branch(state, gates, ()),
            state * torch.exp(1j * exact_angles),
            atol=2e-12,
            rtol=0,
        )
        model = get_cost_value(original, QECGatesCost())
        assert model.measurement == 0
        if not adjoint:
            prototype = ZPowGate(
                exponent=gamma / (2**source.num_frac_rotations),
                eps=epsilon / source.num_rotations,
            )
            assert (
                model
                == get_cost_value(prototype, QECGatesCost()) * source.num_rotations
            )
            if (m, f, gamma) == (5, 0, 0.125):
                leaves = [
                    get_cost_value(ZPowGate(exponent, eps=tol), QECGatesCost())
                    for _, exponent, tol in schedule
                ]
                assert model.rotation == 5
                assert sum(cost.rotation for cost in leaves) == 1
                assert sum(cost.t for cost in leaves) == 1
                assert sum(cost.clifford for cost in leaves) == 3
        assert get_cost_value(original, QubitCount()) == m

    torch.testing.assert_close(
        forced_branch(state, (*refs[0], *refs[1]), ()), state, atol=2e-12, rtol=0
    )
    if not signed and source.num_rotations > 0:
        phase = torch.zeros_like(weights)
        for wire, exponent, _ in qvr_zpow_schedule(cost, **params):
            phase.add_(pi * exponent * ((grid >> (m - wire)) & 1).to(torch.float64))
        word = sum(
            ((grid >> (m - wire)) & 1) << i for i, wire in enumerate(reversed(cost))
        )
        ideal = torch.exp(1j * tau * gamma * word.to(torch.float64) / 2**f)
        assert (
            float(torch.max(torch.abs(torch.exp(1j * phase) - ideal)))
            <= epsilon + 1e-12
        )


@pytest.mark.ftqc_reference
def test_pinned_qvr_rotation_count_and_signed_index_boundaries() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QFxp, Register
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrZPow
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    signed = QvrZPow(Register("cost", QFxp(5, 2, signed=True)), 0.375, 0.05)
    with pytest.raises(IndexError, match="out of bounds"):
        signed.decompose_bloq()
    with pytest.raises(ValueError, match="beyond the signed"):
        qvr_zpow_schedule(
            (0, 1, 2, 3, 4),
            cost_fraction_bits=2,
            coefficient=0.375,
            epsilon=0.05,
            signed=True,
        )
    for gamma, source_error, message in (
        (0, ValueError, "nonzero"),
        (0.05, ZeroDivisionError, "zero rotations"),
    ):
        with pytest.raises(source_error):
            QvrZPow.from_bitsize(4, gamma, 0.5).decompose_bloq()
        with pytest.raises(ValueError, match=message):
            qvr_zpow_schedule(
                (0, 1, 2, 3), cost_fraction_bits=4, coefficient=gamma, epsilon=0.5
            )

    tiny = QvrZPow.from_bitsize(4, 0.0001, 0.5)
    assert tiny.num_rotations == get_cost_value(tiny, QECGatesCost()).rotation == -9
    assert (
        qvr_zpow_ir((0, 1, 2, 3), cost_fraction_bits=4, coefficient=0.0001, epsilon=0.5)
        == ()
    )


def test_direct_qvr_validation() -> None:
    args = dict(cost_fraction_bits=2, coefficient=0.5, epsilon=0.25)
    with pytest.raises(ValueError, match="fraction width"):
        qvr_zpow_schedule((0,), **args)
    with pytest.raises(ValueError, match="mode flags"):
        qvr_zpow_schedule((0, 1), **args, signed=1)
    with pytest.raises(ValueError, match="signed cost format"):
        qvr_zpow_schedule((0, 1), **args, signed=True)
    with pytest.raises(ValueError, match="positive epsilon"):
        qvr_zpow_schedule((0, 1), **{**args, "epsilon": 0.0})
    with pytest.raises(ValueError, match="ratio"):
        qvr_zpow_schedule((0, 1), **{**args, "epsilon": 5e-324})


def test_append_direct_qvr_example() -> None:
    from examples.algorithms.register_qvr_zpow import main

    main()
