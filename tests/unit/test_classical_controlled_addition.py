"""Original CAdd inactive paths, NumPy values, and coherent branch consistency."""

from itertools import product

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.controlled_addition import controlled_add_ir
from flagquantum.simulation.classical_controlled_addition import (
    controlled_add_classical_values,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_controlled_addition import layout

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "name,width,cv",
    tuple(product(("QUInt", "QInt", "QMontgomeryUInt"), (1, 2, 3), (0, 1))),
)
def test_both_activations_all_small_source_domains(name, width, cv):
    import qualtran
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    dt = getattr(qualtran, name)(width)
    source = CAdd(dt, cv=cv)
    signed = name == "QInt"
    domain = (
        range(-(1 << (width - 1)), 1 << (width - 1)) if signed else range(1 << width)
    )
    for ctrl, a, b in product((0, 1), domain, domain):
        expected = source.on_classical_vals(ctrl=ctrl, a=a, b=b)
        assert controlled_add_classical_values(
            ctrl, a, b, width, control_value=cv, signed=signed
        ) == (expected["ctrl"], expected["a"], expected["b"])


@pytest.mark.parametrize("cv", (0, 1))
def test_inactive_symbolic_target_returns_original_array_objects(cv):
    import sympy
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    a = np.array([1, 2], dtype=np.int64)
    b = np.array([3, 4], dtype=np.int64)
    for width in (sympy.Integer(3), sympy.Symbol("n", positive=True, integer=True)):
        source = CAdd(QUInt(width), cv=cv)
        expected = source.on_classical_vals(ctrl=1 - cv, a=a, b=b)
        result = controlled_add_classical_values(1 - cv, a, b, width, control_value=cv)
        assert result[1] is expected["a"] is a and result[2] is expected["b"] is b
        with pytest.raises(ValueError):
            source.on_classical_vals(ctrl=cv, a=a, b=b)
        with pytest.raises(ValueError):
            controlled_add_classical_values(cv, a, b, width, control_value=cv)


@pytest.mark.parametrize(
    "kind,signed", tuple(product(("int8", "uint8", "int64", "array"), (False, True)))
)
def test_active_numpy_type_promotion_matches_original_without_add_casts(kind, signed):
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    source = CAdd(QInt(4) if signed else QUInt(4))
    if kind == "array":
        a = np.array([0, 1, 7], dtype=np.int64)
        b = np.array([1, 4, 5], dtype=np.int64)
    else:
        a, b = getattr(np, kind)(7), getattr(np, kind)(3)
    expected = source.on_classical_vals(ctrl=1, a=a, b=b)["b"]
    result = controlled_add_classical_values(1, a, b, 4, signed=signed)[2]
    assert type(result) is type(expected)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "m,n,signed,cv",
    tuple(
        (m, n, s, c)
        for m, n, s in ((1, 1, False), (1, 2, False), (2, 2, False), (2, 2, True))
        for c in (0, 1)
    ),
)
def test_every_coherent_control_input_and_work_erasure_history(m, n, signed, cv):
    a, b, loaded, carry, lbits, cbits = layout(m, n)
    gates = controlled_add_ir(0, a, b, loaded, carry, lbits, cbits, control_value=cv)
    work = m + n - 1
    total = 2 * m + 2 * n + 1
    data = torch.arange(1 << (m + n + 2), dtype=torch.int64)
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[((data >> 1) << (work + 1)) | (data & 1)] = amps
    expected = torch.zeros_like(state)
    for label in data.tolist():
        ctrl = label >> (m + n + 1)
        left = (label >> (n + 1)) & ((1 << m) - 1)
        right = (label >> 1) & ((1 << n) - 1)
        if signed:
            left -= (1 << m) if left >= 1 << (m - 1) else 0
            right -= (1 << n) if right >= 1 << (n - 1) else 0
        ctrl, left, right = controlled_add_classical_values(
            ctrl, left, right, n, control_value=cv, signed=signed
        )
        logical = (ctrl << (m + n)) | ((left % (1 << m)) << n) | (right % (1 << n))
        expected[(logical << (work + 1)) | (label & 1)] = amps[label]
    for history in product((0, 1), repeat=work):
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            expected / 2 ** (work / 2),
            rtol=0,
            atol=1e-12,
        )


def test_original_unequal_signed_classical_decomposition_disagreement_remains_explicit():
    from qualtran import QInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    source = CAdd(QInt(1), QInt(2))
    assert controlled_add_classical_values(1, -1, 0, 2, signed=True) == (1, -1, -1)
    assert source.on_classical_vals(ctrl=1, a=-1, b=0)["b"] == -1
    gates = controlled_add_ir(0, *layout(1, 2))
    state = torch.zeros(64, dtype=torch.complex128)
    state[48] = 1  # ctrl=1,a=1,b=00,work=00
    expected = torch.zeros_like(state)
    expected[52] = 1  # zero extension gives b=01
    for history in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(state, gates, history), expected / 2, rtol=0, atol=1e-12
        )
