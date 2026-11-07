"""Controlled modular source parity, coherent channels and all-branch certificates."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.and_compute import and_compute_ir, and_uncompute_ir
from flagquantum.compiler.registers.controlled_modular_addition import (
    _controlled_modular_workspace,
    controlled_modular_add_ir,
    controlled_modular_requirements,
)
from flagquantum.simulation.classical_modular_addition import (
    controlled_modular_add_classical_values,
)
from flagquantum.simulation.classical_modular_subtraction import (
    controlled_modular_subtract_classical_values,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(1, 1), (2, 1), (2, 2), (2, 3), (3, 3), (3, 5), (5, 13), (8, 251)]


def layout(n: int) -> tuple[tuple[int, ...], ...]:
    q, c = controlled_modular_requirements(n)
    return (
        tuple(range(1, n + 1)),
        tuple(range(n + 1, 2 * n + 1)),
        tuple(range(2 * n + 1, 2 * n + 1 + q)),
        tuple(range(40, 40 + c)),
    )


def reference_binding(
    n: int, p: int, cv: int
) -> tuple[tuple[int, ...], dict[int, int]]:
    a, b, w, bits = layout(n)
    sign, constant, carry, compare = _controlled_modular_workspace(
        0, a, b, p, w, bits, cv
    )
    allocations = (
        sign,
        *constant[:n],
        *carry,
        *constant,
        *carry,
        *constant[:n],
        *carry[: n - 1],
        *compare[:2],
        *compare[2:-1][::-1],
        compare[-1],
    )
    measured = dict(zip((*constant[:n], *carry), bits[: 2 * n], strict=True))
    compare_targets = (*compare[2:-1][::-1][1:], compare[-1])
    for wire, bit in zip(compare_targets, bits[: n + 2], strict=True):
        assert wire not in measured or measured[wire] == bit
        measured[wire] = bit
    return allocations, measured


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("montgomery", [False, True])
@pytest.mark.parametrize("cv", [0, 1])
def test_controlled_modadd_source_gates_feedback_and_resources(
    n: int, p: int, montgomery: bool, cv: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAdd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, w, bits = layout(n)
    source = CModAdd((QMontgomeryUInt if montgomery else QUInt)(n), p, cv)
    actual = controlled_modular_add_ir(0, a, b, p, w, bits, control_value=cv)
    allocations, measurements = reference_binding(n, p, cv)
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": a, "y": b},
        allocations,
        measurements,
        reuse_allocations=False,
        preserve_components=True,
    )
    assert_source(actual, expected)
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in actual)
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 20 * n + 4
    assert counts["measure"] == model.measurement == 5 * n + 1
    assert get_cost_value(source, QubitCount()) == 1 + 2 * n + len(w)
    assert counts["cx"] == 58 * n + 1 + 2 * p.bit_count() + 4 * (n == 1)
    assert counts["x"] == 2 * ((-p) % (1 << (n + 1))).bit_count() + 4 * n + 7 + (
        4 * n + 4
    ) * (cv == 0)


@pytest.mark.parametrize("xy", range(4))
def test_certificate_and_primitive_columns_and_both_kraus_outcomes(xy: int) -> None:
    # Independently establish every valid column of the literal primitive blocks.
    state = torch.zeros(8, dtype=torch.complex128)
    state[xy << 1] = 1
    expected = torch.zeros_like(state)
    expected[(xy << 1) | ((xy >> 1) & (xy & 1))] = 1
    compute = and_compute_ir(0, 1, 2)
    erase = and_uncompute_ir(0, 1, 2, 40)
    actual = (
        Circuit(3, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(3, compute))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    for result in (0, 1):
        torch.testing.assert_close(
            forced_branch(actual, erase, (result,)), state / 2**0.5, rtol=0, atol=1e-12
        )
    words = ((xy >> 1, xy & 1, 0),)
    mapped, events = certified_and_channel_map((*compute, *erase), words)
    assert mapped == words and events == 1


def certificate_inputs(
    n: int, p: int, cv: int, *, subtract: bool = False
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    q, _ = controlled_modular_requirements(n)
    width = 1 + 2 * n + q
    inputs = []
    outputs = []
    for control in (0, 1):
        for a in range(p + int(not subtract)):
            for b in range(p):
                evaluate = (
                    controlled_modular_subtract_classical_values
                    if subtract
                    else controlled_modular_add_classical_values
                )
                out = int(evaluate(control, a, b, p, control_value=cv)[2])
                assert out == (
                    (b + (-a if subtract else a)) % p if control == cv else b
                )
                before = ((control << (2 * n)) | (a << n) | b) << q
                after = ((control << (2 * n)) | (a << n) | out) << q
                inputs.append(
                    tuple((before >> (width - 1 - i)) & 1 for i in range(width))
                )
                outputs.append(
                    tuple((after >> (width - 1 - i)) & 1 for i in range(width))
                )
    return tuple(inputs), tuple(outputs)


@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("cv", [0, 1])
def test_all_valid_basis_inputs_certify_every_measurement_branch(
    n: int, p: int, cv: int
) -> None:
    pytest.importorskip("numpy")
    a, b, w, bits = layout(n)
    gates = controlled_modular_add_ir(0, a, b, p, w, bits, control_value=cv)
    inputs, expected = certificate_inputs(n, p, cv)
    actual, events = certified_and_channel_map(gates, inputs)
    assert actual == expected and events == 5 * n + 1
    # Each of all 2**events branches has the same map and amplitude factor
    # 2**(-events/2); arbitrary input phases follow by the certified linear maps.


def coherent_states(
    n: int, p: int, cv: int, *, subtract: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    q, _ = controlled_modular_requirements(n)
    width = 2 + 2 * n + q
    words = [
        (ctrl, a, b, s)
        for ctrl in (0, 1)
        for a in range(p + int(not subtract))
        for b in range(p)
        for s in (0, 1)
    ]
    coeff = torch.arange(len(words), dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for (ctrl, a, b, s), amp in zip(words, amps, strict=True):
        evaluate = (
            controlled_modular_subtract_classical_values
            if subtract
            else controlled_modular_add_classical_values
        )
        out = int(evaluate(ctrl, a, b, p, control_value=cv)[2])
        assert out == ((b + (-a if subtract else a)) % p if ctrl == cv else b)
        state[(((ctrl << (2 * n)) | (a << n) | b) << (q + 1)) | s] = amp
        expected[(((ctrl << (2 * n)) | (a << n) | out) << (q + 1)) | s] = amp
    return state, expected


@pytest.mark.parametrize("n,p", [(1, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("cv", [0, 1])
def test_raw_primitive_complex_channels_crosscheck_certificate(
    n: int, p: int, cv: int
) -> None:
    a, b, w, bits = layout(n)
    gates = controlled_modular_add_ir(0, a, b, p, w, bits, control_value=cv)
    state, expected = coherent_states(n, p, cv)
    events = 5 * n + 1
    outcomes = (
        product((0, 1), repeat=events)
        if n == 1
        else ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events)))
    )
    for outcome in outcomes:
        torch.testing.assert_close(
            forced_branch(state, gates, outcome),
            expected / 2 ** (events / 2),
            rtol=0,
            atol=1e-12,
        )


def test_controlled_modular_validation_and_certificate_fail_closed() -> None:
    for n in (0, -1, True):
        with pytest.raises(ValueError, match="positive"):
            controlled_modular_requirements(n)
    a, b, w, bits = layout(2)
    for cv in (True, 2):
        with pytest.raises(ValueError, match="zero or one"):
            controlled_modular_add_ir(0, a, b, 2, w, bits, control_value=cv)
    for ctrl in (a[0], -1, True):
        with pytest.raises(ValueError, match="distinct"):
            controlled_modular_add_ir(ctrl, a, b, 2, w, bits)
    with pytest.raises(ValueError, match="clean work"):
        controlled_modular_add_ir(0, a, b, 2, w[:-1], bits)
    with pytest.raises(ValueError, match="classical"):
        controlled_modular_add_ir(0, a, b, 2, w, bits[:-1])
    from flagquantum.core.ir import Instruction

    with pytest.raises(AssertionError):
        certified_and_channel_map((Instruction("t", (0,)),), ((0,),))
    with pytest.raises(AssertionError, match="nonempty"):
        certified_and_channel_map((), ())
