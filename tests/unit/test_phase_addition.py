"""Source fixed-point action and literal measured-adder component evidence."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.phase_addition import (
    phase_add_ir,
    phase_add_workspace_size,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "m,n,shift", [(1, 1, 0), (2, 5, 0), (5, 2, 0), (3, 4, 2), (2, 3, 3), (8, 8, 2)]
)
@pytest.mark.parametrize("cv", [None, 0, 1])
@pytest.mark.parametrize("sign", [-1, 1])
def test_fixed_point_action_components_and_complex_channel(
    m: int, n: int, shift: int, cv: int | None, sign: int
) -> None:
    pytest.importorskip("qualtran")
    import cirq
    from qualtran import DecomposeNotImplementedError, QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    controlled = cv is not None
    control = 0 if controlled else None
    a = tuple(range(int(controlled), int(controlled) + m))
    b = tuple(range(int(controlled) + m, int(controlled) + m + n))
    q = phase_add_workspace_size(m, n, right_shift=shift, controlled=controlled)
    work = tuple(range(int(controlled) + m + n, int(controlled) + m + n + q))
    bits = tuple(range(40, 40 + q))
    kwargs = dict(
        right_shift=shift,
        sign=sign,
        control=control,
        control_value=1 if cv is None else cv,
    )
    gates = phase_add_ir(a, b, work, bits, **kwargs)
    inverse = phase_add_ir(a, b, work, bits, **{**kwargs, "sign": -sign})
    reference = AddIntoPhaseGrad(m, n, right_shift=shift, sign=sign, controlled_by=cv)
    with pytest.raises(DecomposeNotImplementedError, match="does not declare"):
        reference.decompose_bloq()
    estimate = get_cost_value(reference, QECGatesCost())
    assert estimate.toffoli == (n - 2) * (1 + controlled)
    assert estimate.measurement == 0
    assert get_cost_value(reference, QubitCount()) == max(3, m + n + controlled)
    counts = Counter(g.name for g in gates)
    assert counts["t"] + counts["tdg"] == 4 * q
    assert counts["measure"] == counts["reset"] == q

    if shift >= n:
        assert gates == inverse == ()
    else:
        active = min(m, n - shift)
        prefix = active + shift
        ports = {"a": a[:active], "b": b[:prefix]}
        left, right = QUInt(active), QUInt(prefix)
        if cv is None:
            component = Add(left, right)
        else:
            assert control is not None
            component = CAdd(left, right, cv=cv)
            ports["ctrl"] = (control,)
        for stream, direction in ((gates, sign), (inverse, -sign)):
            source = component if direction == 1 else component.adjoint()
            expected = reference_arithmetic_gates(
                source, ports, work, dict(zip(work, bits, strict=True))
            )
            assert_source(stream, expected)
            native_model = get_cost_value(source, QECGatesCost())
            assert native_model.total_t_count() == 4 * q
            assert native_model.measurement == q

    width = int(controlled) + m + n + q
    destinations = (
        range(1 << n)
        if n <= 5
        else (0, 1, (1 << (n - 1)) - 1, 1 << (n - 1), (1 << n) - 1)
    )
    initial, expected_rows = [], []
    tensor = cirq.unitary(reference) if m + n + controlled <= 5 else None
    for enabled, x, destination in product(
        range(1 << controlled), range(1 << m), destinations
    ):
        inputs = {"x": x, "phase_grad": destination}
        if controlled:
            inputs["ctrl"] = enabled
        outputs = reference.on_classical_vals(**inputs)
        updated = int(outputs["phase_grad"])
        assert outputs["x"] == x
        if controlled:
            assert outputs["ctrl"] == enabled
        delta = (x << max(0, n - m - shift)) >> max(0, m + shift - n)
        assert updated == (
            destination + (sign * delta if cv is None or enabled == cv else 0)
        ) % (1 << n)
        index = (enabled << (m + n)) | (x << n) | destination
        mapped = (enabled << (m + n)) | (x << n) | updated
        if tensor is not None:
            assert tensor[mapped, index] == pytest.approx(1)
        initial.append(tuple((index << q) >> i & 1 for i in reversed(range(width))))
        expected_rows.append(
            tuple((mapped << q) >> i & 1 for i in reversed(range(width)))
        )
    assert certified_and_channel_map(gates, tuple(initial)) == (tuple(expected_rows), q)
    assert certified_and_channel_map(inverse, tuple(expected_rows)) == (
        tuple(initial),
        q,
    )

    def addresses(rows: list[tuple[int, ...]]) -> list[int]:
        return [
            2 * sum(bit << (width - 1 - i) for i, bit in enumerate(row)) + spectator
            for row in rows
            for spectator in (0, 1)
        ]

    w = torch.arange(1, 2 * len(initial) + 1, dtype=torch.float64)
    amps = torch.complex(w, w * (w % 5 - 2))
    amps /= amps.norm()
    start, end = addresses(initial), addresses(expected_rows)
    if width <= 10 and q <= 4:
        state = torch.zeros(1 << (width + 1), dtype=torch.complex128)
        expected_state = torch.zeros_like(state)
        state[start] = amps
        expected_state[end] = amps
        for outcomes in product((0, 1), repeat=q):
            torch.testing.assert_close(
                forced_branch(state, gates, outcomes),
                expected_state * 2 ** (-q / 2),
                atol=2e-12,
                rtol=0,
            )
            torch.testing.assert_close(
                forced_branch(expected_state, inverse, outcomes),
                state * 2 ** (-q / 2),
                atol=2e-12,
                rtol=0,
            )
    else:
        a_sparse = dict(zip(start, amps.tolist(), strict=True))
        b_sparse = dict(zip(end, amps.tolist(), strict=True))
        for stream, input_state, output_state in (
            (gates, a_sparse, b_sparse),
            (inverse, b_sparse, a_sparse),
        ):
            for seed in (1, 41):
                actual = execute_sparse_register(
                    CircuitIR(width + 1, stream), input_state, seed=seed
                )
                assert all(
                    abs(actual.get(k, 0) - output_state.get(k, 0)) < 2e-12
                    for k in actual.keys() | output_state.keys()
                )


def test_phase_add_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        phase_add_workspace_size(0, 3)
    with pytest.raises(ValueError, match="nonnegative"):
        phase_add_workspace_size(1, 3, right_shift=-1)
    with pytest.raises(ValueError, match="boolean"):
        phase_add_workspace_size(1, 3, controlled=1)
    with pytest.raises(ValueError, match="sign"):
        phase_add_ir((0,), (1,), (), (), sign=0)
    with pytest.raises(ValueError, match="control value"):
        phase_add_ir((0,), (1,), (), (), control_value=2)
    with pytest.raises(ValueError, match="distinct"):
        phase_add_ir((0,), (0,), (), ())
    with pytest.raises(ValueError, match="work wires"):
        phase_add_ir((0,), (1,), (2,), ())
    with pytest.raises(ValueError, match="classical slots"):
        phase_add_ir((0,), (1, 2), (3,), (), right_shift=1)


def test_append_shifted_phase_kickback() -> None:
    from examples.algorithms.register_phase_addition import main

    main()
