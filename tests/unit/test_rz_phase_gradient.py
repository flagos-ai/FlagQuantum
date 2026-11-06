"""Verify the original cast/adder graph, phase signs and clean histories."""

from cmath import exp
from collections import Counter
from itertools import product
from math import sqrt, tau
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.rz_phase_gradient import rz_via_phase_gradient_ir
from flagquantum.core.ir import CircuitIR
from tests.unit.algorithms.primitives.multiplexers.test_thc_qrom import _execute_sparse
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("m,n", [(1, 1), (2, 3), (3, 3), (4, 5), (8, 8)])
def test_rz_source_graph_angle_encoding_and_complex_phase(m: int, n: int) -> None:
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran import QFxp, QUInt
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.bloqs.bookkeeping import Cast
    from qualtran.bloqs.rotations.rz_via_phase_gradient import RzViaPhaseGradient
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    angle = tuple(range(1, 1 + m))
    gradient = tuple(range(1 + m, 1 + m + n))
    work = tuple(range(1 + m + n, m + 2 * n))
    bits = tuple(range(40, 40 + n - 1))
    width = m + 2 * n
    source = RzViaPhaseGradient(QFxp(m, m), QFxp(n, n))
    ports = {"q": (0,), "angle": angle, "phase_grad": gradient}

    for adjoint in (False, True):
        original = source.adjoint() if adjoint else source
        casts, adders = [], []

        def leaf(
            bloq: Any,
            inputs: SourcePorts,
            desired: SourcePorts,
            *,
            inverse: bool = adjoint,
            cast_ports=casts,
            adder_ports=adders,
        ) -> SourceResult | None:
            if isinstance(bloq, Cast):
                (reg,) = bloq.signature.lefts()
                qubits = np.array([cirq.LineQubit(w) for w in inputs["reg"]]).reshape(
                    reg.bitsize
                )
                operation, output = bloq.as_cirq_op(qubit_manager=None, reg=qubits)
                assert operation is None
                assert tuple(w.x for w in output["reg"]) == inputs["reg"]
                cast_ports.append((bloq.inp_dtype, bloq.out_dtype))
                return (), inputs
            if isinstance(bloq, ControlledAddOrSubtract):
                assert bloq.add_when_ctrl_is_on is not inverse
                assert bloq.a_dtype == QUInt(m) and bloq.b_dtype == QUInt(n)
                adder_ports.append(inputs)
                return (
                    reference_arithmetic_gates(
                        bloq,
                        inputs,
                        work,
                        dict(zip(work, bits, strict=True)),
                        preserve_components=True,
                    ),
                    inputs,
                )
            return None

        expected, outputs = walk_source_graph(original, ports, ports, leaf)
        assert outputs == ports
        assert Counter(casts) == Counter(
            (
                (QFxp(m, m), QUInt(m)),
                (QFxp(n, n), QUInt(n)),
                (QUInt(m), QFxp(m, m)),
                (QUInt(n), QFxp(n, n)),
            )
        )
        assert adders == [{"ctrl": (0,), "a": angle, "b": gradient}]
        gates = rz_via_phase_gradient_ir(
            0, angle, gradient, work, bits, adjoint=adjoint
        )
        assert_source(gates, expected)

        model = get_cost_value(original, QECGatesCost())
        counts = Counter(g.name for g in gates)
        assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (n - 1)
        assert counts["measure"] == counts["reset"] == model.measurement == n - 1
        assert get_cost_value(original, QubitCount()) == width

        graph = original.decompose_bloq()
        angles = range(1 << m) if m <= 4 else (0, 1, 127, 128, 255)
        basis, mapped = [], []
        prepared, kicked = {}, {}
        scale = sqrt(2 * len(angles) * 2**n)
        for q, a, y in product((0, 1), angles, range(1 << n)):
            outputs = graph.on_classical_vals(q=q, angle=a, phase_grad=y)
            direction = 1 if bool(q) != adjoint else -1
            updated = (y + direction * a) % (1 << n)
            assert outputs == {"q": q, "angle": a, "phase_grad": updated}
            before = (((q << m) | a) << n | y) << (n - 1)
            after = (((q << m) | a) << n | updated) << (n - 1)
            basis.append(tuple(before >> i & 1 for i in reversed(range(width))))
            mapped.append(tuple(after >> i & 1 for i in reversed(range(width))))
            prepared[before] = exp(1j * (a - q - tau * y / 2**n)) / scale
            kicked[before] = prepared[before] * exp(1j * tau * direction * a / 2**n)
        assert certified_and_channel_map(gates, tuple(basis)) == (tuple(mapped), n - 1)

        if n <= 3:
            initial = torch.zeros(1 << width, dtype=torch.complex128)
            final = torch.zeros_like(initial)
            initial[list(prepared)] = torch.tensor(
                list(prepared.values()), dtype=torch.complex128
            )
            final[list(kicked)] = torch.tensor(
                list(kicked.values()), dtype=torch.complex128
            )
            for history in product((0, 1), repeat=n - 1):
                torch.testing.assert_close(
                    forced_branch(initial, gates, history),
                    final * 2 ** (-(n - 1) / 2),
                    atol=2e-12,
                    rtol=0,
                )
        else:
            for seed in (3, 41):
                actual = _execute_sparse(CircuitIR(width, gates), prepared, seed=seed)
                assert all(
                    abs(actual.get(k, 0) - kicked.get(k, 0)) < 2e-12
                    for k in actual.keys() | kicked.keys()
                )


def test_rz_phase_gradient_layout_validation() -> None:
    with pytest.raises(ValueError, match="adjoint"):
        rz_via_phase_gradient_ir(0, (1,), (2,), (), (), adjoint=1)
    with pytest.raises(ValueError, match="no wider"):
        rz_via_phase_gradient_ir(0, (1, 2), (3,), (), ())


def test_append_gradient_rotation_example() -> None:
    from examples.algorithms.register_phase_gradient_rotations import main

    main()
