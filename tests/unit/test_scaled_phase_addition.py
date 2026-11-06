"""Compare coefficient schedules, shared adder streams and coherent cleanup."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.phase_addition import phase_add_workspace_size
from flagquantum.compiler.registers.scaled_phase_addition import (
    scaled_phase_add_ir,
    scaled_phase_add_terms,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit

CASES = [
    (2, 2, 2, 0.5, 2, 2, False),
    (3, 3, 4, 0.75, 3, 3, False),
    (5, 2, 3, 1.875, 5, 4, False),
    (5, 3, 5, -2.75, 5, 3, True),
    (3, 2, 4, 3.25, 5, 3, True),
    (4, 4, 2, 0.875, 4, 4, False),
    (3, 3, 4, 0.0, 3, 3, False),
    (4, 2, 3, 0.99, 4, 4, False),
    (3, 3, 4, -1.25, 3, 3, False),
    (4, 0, 3, -0.375, 4, 3, False),
    (8, 8, 8, -0.375, 8, 8, False),
]


@pytest.mark.parametrize("m,f,n,gamma,k,kf,dtype_failure", CASES)
def test_scaled_source_schedule_and_complex_roundtrip(
    m: int, f: int, n: int, gamma: float, k: int, kf: int, dtype_failure: bool
) -> None:
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran import BloqError, QFxp, QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.rotations.phase_gradient import (
        AddIntoPhaseGrad,
        AddScaledValIntoPhaseReg,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    reference = AddScaledValIntoPhaseReg(QFxp(m, f), n, gamma, QFxp(k, kf))
    if dtype_failure:
        with pytest.raises(BloqError, match="register dtypes are not consistent"):
            reference.decompose_bloq()
        composite = None
    else:
        composite = reference.decompose_bloq()

    qubits = cirq.LineQubit.range(m + n)
    context = cirq.DecompositionContext(cirq.SimpleQubitManager())
    operations = tuple(
        cirq.flatten_op_tree(
            reference.decompose_from_registers(
                context=context,
                x=np.array(qubits[:m]),
                phase_grad=np.array(qubits[m:]),
            )
        )
    )
    terms = scaled_phase_add_terms(m, f, n, gamma, k, kf)
    assert len(operations) == len(terms)
    if composite is not None:
        assert [
            instance.bloq
            for instance, _, _ in composite.iter_bloqsoqs()
            if isinstance(instance.bloq, AddIntoPhaseGrad)
        ] == [op.gate for op in operations]
    source = tuple(range(m))
    gradient = tuple(range(m, m + n))
    sign = -1 if gamma < 0 else 1
    for (part, shift), op in zip(terms, operations, strict=True):
        assert isinstance(op.gate, AddIntoPhaseGrad)
        assert op.gate.sign == sign
        assert op.gate.right_shift == shift
        assert op.gate.x_bitsize == len(source[part])
        assert op.qubits == (*qubits[:m][part], *qubits[m:])

    required = max(
        (
            phase_add_workspace_size(len(source[part]), n, right_shift=shift)
            for part, shift in terms
        ),
        default=0,
    )
    work = tuple(range(m + n, m + n + required))
    bits = tuple(range(40, 40 + required))
    params = dict(
        input_fraction_bits=f,
        coefficient=gamma,
        coefficient_width=k,
        coefficient_fraction_bits=kf,
    )
    gates = scaled_phase_add_ir(source, gradient, work, bits, **params)
    inverse = scaled_phase_add_ir(source, gradient, work, bits, adjoint=True, **params)

    for reverse, stream in ((False, gates), (True, inverse)):
        expected = []
        for op in reversed(operations) if reverse else operations:
            leaf = op.gate
            assert isinstance(leaf, AddIntoPhaseGrad)
            active = min(int(leaf.x_bitsize), n - leaf.right_shift)
            prefix = active + leaf.right_shift
            ports = {
                "a": tuple(q.x for q in op.qubits[:active]),
                "b": gradient[:prefix],
            }
            component = Add(QUInt(active), QUInt(prefix))
            if (leaf.sign < 0) ^ reverse:
                component = component.adjoint()
            carry = work[: prefix - 1]
            expected.extend(
                reference_arithmetic_gates(
                    component,
                    ports,
                    carry,
                    dict(zip(carry, bits[: len(carry)], strict=True)),
                )
            )
        assert_source(stream, tuple(expected))
    counts = Counter(g.name for g in gates)
    events = counts["measure"]
    assert counts["t"] + counts["tdg"] == 4 * events
    assert counts["reset"] == events
    assert Counter(g.name for g in inverse) == counts
    model = get_cost_value(reference, QECGatesCost())
    assert model.toffoli == (n - 2) * min(len(terms), (k + 2) // 2)
    assert model.measurement == 0

    width = m + n + required
    before, after = [], []
    destinations = range(1 << n) if n <= 5 else (0, 1, (1 << (n - 1)) - 1, (1 << n) - 1)
    for x, y in product(range(1 << m), destinations):
        inputs = dict(x=x, phase_grad=y)
        target = reference.on_classical_vals(**inputs)
        if composite is not None:
            assert composite.on_classical_vals(**inputs) == target
        out = int(target["phase_grad"])
        native = y
        for op in operations:
            leaf = op.gate
            assert isinstance(leaf, AddIntoPhaseGrad)
            arg = sum(
                (x >> (m - q.x - 1) & 1) << i
                for i, q in enumerate(reversed(op.qubits[:-n]))
            )
            native = int(leaf.on_classical_vals(x=arg, phase_grad=native)["phase_grad"])
        assert native == out and target["x"] == x
        before.append(
            tuple((((x << n) | y) << required) >> i & 1 for i in reversed(range(width)))
        )
        after.append(
            tuple(
                (((x << n) | out) << required) >> i & 1 for i in reversed(range(width))
            )
        )
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(inverse, tuple(after)) == (tuple(before), events)

    def addresses(rows: list[tuple[int, ...]]) -> list[int]:
        return [
            2 * sum(bit << (width - i - 1) for i, bit in enumerate(row)) + s
            for row in rows
            for s in (0, 1)
        ]

    weights = torch.arange(1, 2 * len(before) + 1, dtype=torch.float64)
    amps = torch.complex(weights, weights * (weights % 7 - 3))
    amps /= amps.norm()
    start, end = addresses(before), addresses(after)
    if events <= 4 and width <= 12:
        a = torch.zeros(1 << (width + 1), dtype=torch.complex128)
        b = torch.zeros_like(a)
        a[start] = amps
        b[end] = amps
        for outcomes in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(a, gates, outcomes),
                b * 2 ** (-events / 2),
                atol=2e-12,
                rtol=0,
            )
            torch.testing.assert_close(
                forced_branch(b, inverse, outcomes),
                a * 2 ** (-events / 2),
                atol=2e-12,
                rtol=0,
            )
    else:
        a_sparse, b_sparse = dict(zip(start, amps.tolist(), strict=True)), dict(
            zip(end, amps.tolist(), strict=True)
        )
        for stream, initial, target in (
            (gates, a_sparse, b_sparse),
            (inverse, b_sparse, a_sparse),
        ):
            for seed in (1, 41):
                actual = execute_sparse_register(
                    CircuitIR(width + 1, stream), initial, seed=seed
                )
                assert all(
                    abs(actual.get(i, 0) - target.get(i, 0)) < 2e-12
                    for i in actual.keys() | target.keys()
                )


def test_scaled_phase_validation() -> None:
    with pytest.raises(ValueError, match="format"):
        scaled_phase_add_terms(2, 3, 2, 0.5, 2, 2)
    with pytest.raises(ValueError, match="gradient width"):
        scaled_phase_add_terms(2, 2, 0, 0.5, 2, 2)
    with pytest.raises(ValueError, match="finite"):
        scaled_phase_add_terms(2, 2, 2, float("inf"), 2, 2)
    params = dict(
        input_fraction_bits=2,
        coefficient=0.5,
        coefficient_width=2,
        coefficient_fraction_bits=2,
    )
    with pytest.raises(ValueError, match="adjoint"):
        scaled_phase_add_ir((0, 1), (2, 3), (4,), (0,), **params, adjoint=1)
    with pytest.raises(ValueError, match="distinct"):
        scaled_phase_add_ir((0, 1), (2, 3), (0,), (0,), **params)
    with pytest.raises(ValueError, match="work wires"):
        scaled_phase_add_ir((0, 1), (2, 3), (), (), **params)
    with pytest.raises(ValueError, match="classical slots"):
        scaled_phase_add_ir((0, 1), (2, 3), (4,), (), **params)


def test_append_scaled_phase_kickback() -> None:
    from examples.algorithms.register_scaled_phase_addition import main

    main()
