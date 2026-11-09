"""Pinned QVR sizing, source phase action and native measurement histories."""

from cmath import exp
from collections import Counter
from itertools import product
from math import sqrt, tau

import pytest
import torch

from flagquantum.compiler.registers.phase.phase_addition import phase_add_workspace_size
from flagquantum.compiler.registers.phase.qvr_phase_gradient import (
    QvrPhaseFormat,
    qvr_phase_format,
    qvr_phase_gradient_ir,
)
from flagquantum.compiler.registers.phase.scaled_phase_addition import (
    scaled_phase_add_ir,
    scaled_phase_add_terms,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "m,f,gamma,epsilon",
    [
        (3, 3, 0.75, 0.25),
        (4, 1, -0.375, 0.1),
        (4, 0, 0.125, 0.25),
        (8, 8, 0.625, 0.03),
        (39, 36, -2.2818271640892593, 3.187316510049381e-6),
        (53, 50, 1.0089192073911049, 1.0145781526456148e-6),
        (54, 54, 0.625, 0.25),
    ],
)
def test_qvr_source_precision_choice(
    m: int, f: int, gamma: float, epsilon: float
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QFxp, Register
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrPhaseGradient

    source = QvrPhaseGradient(Register("cost", QFxp(m, f)), gamma, epsilon)
    assert qvr_phase_format(m, f, gamma, epsilon) == QvrPhaseFormat(
        source.b_grad, source.gamma_dtype.bitsize, source.gamma_dtype.num_frac
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "m,f,gamma,epsilon",
    [(2, 2, 0.5, 0.5), (3, 3, 0.75, 0.25), (4, 1, -0.375, 0.1), (8, 8, 0.625, 0.03)],
)
def test_qvr_source_binding_complex_phase_and_cleanup(
    m: int, f: int, gamma: float, epsilon: float
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QFxp, Register
    from qualtran.bloqs.rotations.phase_gradient import AddScaledValIntoPhaseReg
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrPhaseGradient
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = QvrPhaseGradient(Register("cost", QFxp(m, f)), gamma, epsilon)
    graph = source.decompose_bloq()
    ((instance, args, outputs),) = tuple(graph.iter_bloqsoqs())
    leaf = instance.bloq
    assert isinstance(leaf, AddScaledValIntoPhaseReg)
    assert leaf.x_dtype == QFxp(m, f) and leaf.gamma == gamma
    assert leaf.gamma_dtype == source.gamma_dtype
    assert args["x"].reg.name == "cost" and args["phase_grad"].reg.name == "phase_grad"
    assert graph.final_soqs() == dict(zip(("cost", "phase_grad"), outputs, strict=True))
    n = int(leaf.phase_bitsize)
    terms = scaled_phase_add_terms(
        m, f, n, gamma, leaf.gamma_dtype.bitsize, leaf.gamma_dtype.num_frac
    )
    cost, gradient = tuple(range(m)), tuple(range(m, m + n))
    requirements = tuple(
        phase_add_workspace_size(len(cost[part]), n, right_shift=shift)
        for part, shift in terms
    )
    q = max(requirements, default=0)
    events = sum(requirements)
    width = m + n + q
    work, bits = tuple(range(m + n, width)), tuple(range(40, 40 + q))
    model = get_cost_value(source, QECGatesCost())
    assert model == get_cost_value(leaf, QECGatesCost())
    assert model.measurement == 0
    if (m, f) == (4, 1):
        with pytest.raises(RuntimeError, match="register dtypes are not consistent"):
            get_cost_value(source, QubitCount())
    else:
        assert get_cost_value(source, QubitCount()) == m + n
    increments = tuple(
        int(graph.on_classical_vals(cost=x, phase_grad=0)["phase_grad"])
        for x in range(1 << m)
    )
    assert (
        max(
            abs(exp(1j * tau * gamma * x / 2**f) - exp(1j * tau * delta / 2**n))
            for x, delta in enumerate(increments)
        )
        <= epsilon
    )
    for adjoint in (False, True):
        gates = qvr_phase_gradient_ir(
            cost,
            gradient,
            work,
            bits,
            cost_fraction_bits=f,
            coefficient=gamma,
            epsilon=epsilon,
            adjoint=adjoint,
        )
        expected = scaled_phase_add_ir(
            cost,
            gradient,
            work,
            bits,
            input_fraction_bits=f,
            coefficient=gamma,
            coefficient_width=leaf.gamma_dtype.bitsize,
            coefficient_fraction_bits=leaf.gamma_dtype.num_frac,
            adjoint=adjoint,
        )
        assert gates == expected
        counts = Counter(g.name for g in gates)
        assert counts["t"] + counts["tdg"] == 4 * events
        assert counts["measure"] == counts["reset"] == events
        basis, mapped = [], []
        before, after = {}, {}
        selected = range(1 << m) if n <= 4 else (0, 1, (1 << m) - 1)
        norm = sqrt(len(selected) * 2**n)
        sign = -1 if adjoint else 1
        for x, y in product(selected, range(1 << n)):
            before_index = ((x << n) | y) << q
            updated = (y + sign * increments[x]) % (1 << n)
            after_index = ((x << n) | updated) << q
            basis.append(tuple(before_index >> i & 1 for i in reversed(range(width))))
            mapped.append(tuple(after_index >> i & 1 for i in reversed(range(width))))
            before[before_index] = exp(1j * (x - tau * y / 2**n)) / norm
            after[before_index] = before[before_index] * exp(
                1j * tau * sign * increments[x] / 2**n
            )
        assert certified_and_channel_map(gates, tuple(basis)) == (tuple(mapped), events)
        if events <= 4:
            a = torch.zeros(1 << width, dtype=torch.complex128)
            b = torch.zeros_like(a)
            a[list(before)] = torch.tensor(
                list(before.values()), dtype=torch.complex128
            )
            b[list(after)] = torch.tensor(list(after.values()), dtype=torch.complex128)
            for outcomes in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(a, gates, outcomes),
                    b * 2 ** (-events / 2),
                    atol=2e-12,
                    rtol=0,
                )
        else:
            for seed in (3, 41):
                result = execute_sparse_register(
                    CircuitIR(width, gates), before, seed=seed
                )
                assert all(
                    abs(result.get(i, 0) - after.get(i, 0)) < 2e-12
                    for i in result.keys() | after.keys()
                )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("gamma", [0.0, 1e-9])
def test_qvr_source_quantized_zero_boundary(gamma: float) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrPhaseGradient

    source = QvrPhaseGradient.from_bitsize(4, gamma, 0.25)
    with pytest.raises(ValueError, match="math domain"):
        _ = source.b_grad
    with pytest.raises(ValueError, match="zero quantized"):
        qvr_phase_format(4, 4, gamma, 0.25)


def test_qvr_format_validation() -> None:
    with pytest.raises(ValueError, match="unsigned cost"):
        qvr_phase_format(4, 5, 0.5, 0.25)
    with pytest.raises(ValueError, match="epsilon"):
        qvr_phase_format(4, 4, 0.5, 0)
    with pytest.raises(ValueError, match="too small"):
        qvr_phase_format(4, 4, 0.5, 5e-324)
    with pytest.raises(ValueError, match="finite coefficient"):
        qvr_phase_format(4, 4, float("inf"), 0.25)
    with pytest.raises(ValueError, match="phase gradient"):
        qvr_phase_gradient_ir(
            (0, 1), (), (), (), cost_fraction_bits=2, coefficient=0.5, epsilon=0.5
        )


def test_append_qvr_example() -> None:
    from examples.algorithms.register_qvr_phase_gradient import main

    main()


def test_qvr_search_can_reject_an_unresolvable_width() -> None:
    # The locked reference overflows its fixed-point accumulator for this
    # integer-heavy cost. The native search has no accepted width either.
    with pytest.raises(ValueError, match="found no gradient width"):
        qvr_phase_format(47, 4, 2.393029727807961, 0.005779784227676086)
