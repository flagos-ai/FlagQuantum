"""Original controlled XOR connections and all small coherent erasure histories."""

from dataclasses import fields
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.control_predicate import ControlRegister
from flagquantum.compiler.registers.controlled_register_xor import (
    controlled_xor_ir,
    grouped_controlled_xor_ir,
    source_grouped_xor_cost,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_control_predicate_leaf,
    walk_source_graph,
)
from tests.unit.test_grouped_controlled_constant_addition import EXTRA
from tests.unit.test_register_control_predicate import CASES, control_fixture
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "case,n,encoding",
    tuple(product(CASES + EXTRA, (1, 2, 5), ("QUInt", "QInt", "QMontgomeryUInt"))),
)
def test_source_controlled_xor_connections_gates_and_resources(case, n, encoding):
    import qualtran
    from qualtran.bloqs.arithmetic.bitwise import Xor
    from qualtran.bloqs.basic_gates import Toffoli, XGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    spec, regs, groups = control_fixture(case)
    m = sum(map(len, groups))
    left = tuple(range(m, m + n))
    right = tuple(range(m + n, m + 2 * n))
    work = tuple(range(m + 2 * n, 2 * m + 2 * n - 1))
    bits = tuple(range(30, 30 + max(0, m - 1)))
    dtype = getattr(qualtran, encoding)(n)
    source = Xor(dtype).controlled(ctrl_spec=spec)
    ports = dict(zip(source.ctrl_reg_names, groups, strict=True)) | {
        "x": left,
        "y": right,
    }

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, Toffoli):
            return (Instruction("ccx", (*inputs["ctrl"], *inputs["target"])),), inputs
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        return source_control_predicate_leaf(bloq, inputs, work, bits)

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    actual = grouped_controlled_xor_ir(groups, regs, left, right, work, bits)
    assert_source(actual, expected)
    assert sum(g.name == "ccx" for g in actual) == n
    for legacy in (False, True):
        model = estimate_component_resources(
            source_grouped_xor_cost(n, regs, legacy_shims=legacy)
        )
        reference = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        for field in fields(model.gates):
            assert getattr(model.gates, field.name) == getattr(reference, field.name)
        assert model.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize(
    "values,n", tuple(product(((0,), (1,), (0, 1), (1, 0, 1)), (1, 2)))
)
def test_every_input_complex_phase_branch_and_clean_work(values, n):
    m = len(values)
    logical = m + 2 * n
    groups = (tuple(range(m)),)
    regs = (ControlRegister(1, values),)
    left = tuple(range(m, m + n))
    right = tuple(range(m + n, logical))
    work = tuple(range(logical, logical + m - 1))
    bits = tuple(range(m - 1))
    gates = grouped_controlled_xor_ir(groups, regs, left, right, work, bits)
    data = torch.arange(1 << (logical + 1), dtype=torch.int64)
    coefficients = data.to(torch.float64)
    amplitudes = torch.complex(coefficients + 1, 3 * coefficients - 5)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << (logical + m), dtype=torch.complex128)
    indices = ((data >> 1) << m) | (data & 1)
    state[indices] = amplitudes
    activation = sum(bit << (m - i - 1) for i, bit in enumerate(values))
    active = (data >> (2 * n + 1)) == activation
    x = (data >> (n + 1)) & ((1 << n) - 1)
    mapped = data ^ (active.to(torch.int64) * (x << 1))
    expected = torch.zeros_like(state)
    expected[((mapped >> 1) << m) | (mapped & 1)] = amplitudes
    for history in product((0, 1), repeat=m - 1):
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            expected / 2 ** ((m - 1) / 2),
            rtol=0,
            atol=1e-12,
        )
        torch.testing.assert_close(
            forced_branch(expected, gates, history),
            state / 2 ** ((m - 1) / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "control,left,right",
    (
        (0, (), ()),
        (0, (1,), (2, 3)),
        (0, (0,), (2,)),
        (True, (1,), (2,)),
        (0, (1,), (1,)),
    ),
)
def test_positive_body_rejects_bad_ports(control, left, right):
    with pytest.raises(ValueError):
        controlled_xor_ir(control, left, right)


@pytest.mark.parametrize("width", (0, -1, True))
def test_model_rejects_invalid_width(width):
    with pytest.raises(ValueError):
        source_grouped_xor_cost(width, (ControlRegister(1, (1,)),))
