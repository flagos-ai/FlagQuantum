"""Pinned Z-power angle quantization, load graph and exact kickback phase."""

from cmath import exp
from collections import Counter
from itertools import product
from math import sqrt, tau
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.zpow_phase_gradient import (
    zpow_phase_gradient_constant,
    zpow_via_phase_gradient_ir,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "n,exponent,quantized",
    [
        (1, 0.5, 0),
        (2, 1.25, 2),
        (4, 0.375, 3),
        (4, 0.37, 2),
        (4, 2.0, 0),
        (4, 3.0, 8),
        (8, 1.99, 254),
    ],
)
def test_zpow_source_graph_resources_and_kickback(
    n: int, exponent: float, quantized: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
    from qualtran.bloqs.rotations.zpow_via_phase_gradient import (
        ZPowConstViaPhaseGradient,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    q = 0
    gradient = tuple(range(1, n + 1))
    angle = tuple(range(n + 1, 2 * n + 1))
    carries = tuple(range(2 * n + 1, 3 * n))
    bits = tuple(range(40, 40 + len(carries)))
    source = ZPowConstViaPhaseGradient(exponent, n)
    assert source._load_bloq.k == zpow_phase_gradient_constant(exponent, n) == quantized
    ports = {"q": (q,), "phase_grad": gradient}
    for adjoint in (False, True):
        original = source.adjoint() if adjoint else source
        allocated = freed = False
        loads = 0

        def leaf(
            bloq: Any,
            inputs: SourcePorts,
            desired: SourcePorts,
            *,
            inverse: bool = adjoint,
        ) -> SourceResult | None:
            nonlocal allocated, freed, loads
            if isinstance(bloq, Allocate):
                assert not allocated and not bloq.dirty
                allocated = True
                return (), {"reg": angle}
            if isinstance(bloq, Free):
                assert not freed and inputs["reg"] == angle
                freed = True
                return (), {}
            if bloq == source._load_bloq.controlled():
                assert inputs == {"ctrl": (q,), "x": angle}
                loads += 1
                return reference_arithmetic_gates(bloq, inputs, (), {}), inputs
            if isinstance(bloq, AddIntoPhaseGrad):
                assert inputs == {"x": angle, "phase_grad": gradient}
                assert (
                    bloq.x_bitsize,
                    bloq.phase_bitsize,
                    bloq.right_shift,
                    bloq.sign,
                ) == (n, n, 0, -1 if inverse else 1)
                add = Add(QUInt(n), QUInt(n))
                if inverse:
                    add = add.adjoint()
                return (
                    reference_arithmetic_gates(
                        add,
                        {"a": angle, "b": gradient},
                        carries,
                        dict(zip(carries, bits, strict=True)),
                    ),
                    inputs,
                )
            return None

        expected, outputs = walk_source_graph(original, ports, ports, leaf)
        assert outputs == ports and allocated and freed and loads == 2
        gates = zpow_via_phase_gradient_ir(
            q, gradient, angle, carries, bits, exponent=exponent, adjoint=adjoint
        )
        assert_source(gates, expected)
        counts = Counter(g.name for g in gates)
        events = n - 1
        assert counts["t"] + counts["tdg"] == 4 * events
        assert counts["measure"] == counts["reset"] == events
        graph = original.decompose_bloq()
        phase_sign = -1 if adjoint else 1
        basis, mapped = [], []
        prepared, kicked = {}, {}
        amps = (0.6j, 0.8)
        for value, y in product((0, 1), range(1 << n)):
            before = ((value << n) | y) << (2 * n - 1)
            updated = (y + phase_sign * value * quantized) % (1 << n)
            assert graph.on_classical_vals(q=value, phase_grad=y) == {
                "q": value,
                "phase_grad": updated,
            }
            after = ((value << n) | updated) << (2 * n - 1)
            basis.append(tuple(before >> i & 1 for i in reversed(range(3 * n))))
            mapped.append(tuple(after >> i & 1 for i in reversed(range(3 * n))))
            prepared[before] = amps[value] * exp(-1j * tau * y / 2**n) / sqrt(2**n)
            kicked[before] = prepared[before] * exp(
                1j * tau * phase_sign * value * quantized / 2**n
            )
        assert certified_and_channel_map(gates, tuple(basis)) == (tuple(mapped), events)
        if n <= 4:
            initial = torch.zeros(1 << (3 * n), dtype=torch.complex128)
            result = torch.zeros_like(initial)
            initial[list(prepared)] = torch.tensor(
                list(prepared.values()), dtype=torch.complex128
            )
            result[list(kicked)] = torch.tensor(
                list(kicked.values()), dtype=torch.complex128
            )
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(initial, gates, history),
                    result * 2 ** (-events / 2),
                    atol=2e-12,
                    rtol=0,
                )
        else:
            for seed in (3, 41):
                actual = execute_sparse_register(
                    CircuitIR(3 * n, gates), prepared, seed=seed
                )
                assert all(
                    abs(actual.get(k, 0) - kicked.get(k, 0)) < 2e-12
                    for k in actual.keys() | kicked.keys()
                )

    model = get_cost_value(source, QECGatesCost())
    assert model.toffoli == n - 2 and model.measurement == 0
    assert get_cost_value(source, QubitCount()) == max(4, 2 * n + 1)


def test_zpow_source_negative_exponent_and_layout_boundaries() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.rotations.zpow_via_phase_gradient import (
        ZPowConstViaPhaseGradient,
    )

    with pytest.raises(ValueError, match="unsigned"):
        ZPowConstViaPhaseGradient(-0.25, 4).decompose_bloq()
    with pytest.raises(ValueError, match="nonnegative"):
        zpow_phase_gradient_constant(-0.25, 4)
    with pytest.raises(ValueError, match="positive"):
        zpow_phase_gradient_constant(0.25, 0)
    with pytest.raises(ValueError, match="adjoint"):
        zpow_via_phase_gradient_ir(0, (1,), (2,), (), (), exponent=0.25, adjoint=1)
    with pytest.raises(ValueError, match="angle wire"):
        zpow_via_phase_gradient_ir(0, (1,), (), (), (), exponent=0.25)
