"""Pinned reflection gate order, complex phases, and measured cleanup."""

from collections import Counter
from itertools import product
from math import pi, sqrt

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
)
from flagquantum.compiler.registers.uniform_preparation import (
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_uniform_preparation import embedded

pytestmark = pytest.mark.unit


def layout(width, value):
    controlled = value is not None
    start = width + int(controlled)
    work, slots = prepare_reflection_requirements(width, controlled=controlled)
    return (
        tuple(range(int(controlled), start)),
        tuple(range(start, start + work)),
        tuple(range(slots)),
        0 if controlled else None,
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width,value", [(0, None), (1, 0), (2, 1), (3, None)])
@pytest.mark.parametrize("coefficient", [1, 1j])
def test_pinned_identity_leaf_order_and_gate_counts(width, value, coefficient):
    pytest.importorskip("qualtran")
    import cirq
    from qualtran.bloqs.mcmt import MultiControlZ
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )
    from qualtran.cirq_interop.testing import GateHelper
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    sizes = (1, width - 1) if width > 1 else (width,) if width else ()
    sel, work, slots, control = layout(width, value)
    anchor = work[0] if control is None else control
    prefix = work[1:] if control is None else work
    for adjoint in (False, True):
        source = ReflectionUsingPrepare.reflection_around_zero(
            sizes,
            control_val=value,
            global_phase=coefficient.conjugate() if adjoint else coefficient,
        )
        context = cirq.DecompositionContext(cirq.SimpleQubitManager())
        operation = GateHelper(source).operation
        ops = tuple(
            cirq.flatten_op_tree(cirq.decompose_once(operation, context=context))
        )
        phase_reflection = next(
            op
            for op in ops
            if isinstance(getattr(op.gate, "bloq", op.gate), MultiControlZ)
        )
        source_mark = getattr(phase_reflection.gate, "bloq", phase_reflection.gate)
        assert source_mark.concrete_cvs == (0,) * width
        if width:
            mark = reference_arithmetic_gates(
                source_mark,
                {"controls": sel, "target": (anchor,)},
                prefix,
                dict(zip(prefix, slots, strict=True)),
            )
        else:
            assert (
                tuple(cirq.decompose_once(phase_reflection, context=context))[0].gate
                == cirq.Z
            )
            mark = (Instruction("z", (anchor,)),)
        expected = []
        for op in ops[1:-1]:
            if op is phase_reflection:
                expected.extend(mark)
            elif op.gate == cirq.X:
                expected.append(Instruction("x", (anchor,)))
            else:
                angle = (-1 if adjoint else 1) * pi / 2
                if control is None:
                    assert op.gate.coefficient == (
                        coefficient.conjugate() if adjoint else coefficient
                    )
                else:
                    assert float(op.gate.exponent) * pi == pytest.approx(angle)
                expected.append(Instruction("phase", (anchor,), {"theta": angle}))
        gates = prepare_reflection_ir(
            sel,
            work,
            slots,
            control=control,
            control_value=value if value is not None else 1,
            global_phase=coefficient,
            adjoint=adjoint,
        )
        assert gates == tuple(expected)
        assert Counter(g.name for g in gates) == Counter(g.name for g in expected)
        model = get_cost_value(source_mark, QECGatesCost())
        counts = Counter(g.name for g in gates)
        assert counts["t"] + counts["tdg"] == model.total_t_count()
        assert counts["measure"] == model.measurement == len(prefix)


@pytest.mark.parametrize("count,value", [(None, None), (None, 0), (3, 1), (3, None)])
def test_append_complex_reflection_and_adjoint(count, value):
    width = 2
    sel, work, slots, control = layout(width, value)
    data_width = width + int(control is not None)
    prep = unprep = temp = ()
    amplitude = torch.zeros(4, dtype=torch.complex128)
    if count is None:
        amplitude[0] = 1
    else:
        _, q, c = uniform_preparation_requirements(count)
        temp = tuple(range(data_width + len(work), data_width + len(work) + q))
        prep = uniform_preparation_ir(count, sel, temp, tuple(range(c)))
        unprep = uniform_preparation_ir(count, sel, temp, tuple(range(c)), adjoint=True)
        amplitude[:count] = 1 / sqrt(count)
    work_width = len(work) + len(temp)
    logical = torch.arange(1, (1 << data_width) + 1, dtype=torch.float64)
    initial = torch.complex(logical, 3 - logical)
    initial /= initial.norm()
    expected = initial.reshape(-1, 4).clone()
    branch = expected[0 if value is None else value]
    branch[:] = 1j * (branch - 2 * amplitude * torch.vdot(amplitude, branch))
    qubits = data_width + work_width
    options = dict(
        prepare=prep,
        unprepare=unprep,
        control=control,
        control_value=value if value is not None else 1,
        global_phase=1j,
    )
    gates = prepare_reflection_ir(sel, work, slots, **options)
    if count is not None:
        counts = Counter(g.name for g in gates)
        assert (counts["t"] + counts["tdg"], counts["measure"], counts["rz"]) == (
            44,
            11,
            4,
        )
    state = embedded(initial, work_width)
    wanted = embedded(expected.flatten(), work_width)
    circuit = DynamicCircuit(qubits, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(qubits, gates))
    if count is None:
        events = sum(g.name == "measure" for g in gates)
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(state, tuple(circuit.to_ir().instructions), history),
                wanted / sqrt(2**events),
                rtol=0,
                atol=3e-12,
            )
    else:
        sampled = run_dynamic(circuit, shots=8, seed=21, strategy="trajectory")
        torch.testing.assert_close(
            sampled.final_states.reshape(8, -1),
            wanted.expand(8, -1),
            rtol=0,
            atol=3e-12,
        )
    inverse = prepare_reflection_ir(sel, work, slots, adjoint=True, **options)
    circuit.append(FieldOperation(qubits, inverse))
    sampled = run_dynamic(circuit, shots=8, seed=12, strategy="trajectory")
    torch.testing.assert_close(
        sampled.final_states.reshape(8, -1), state.expand(8, -1), rtol=0, atol=3e-12
    )


@pytest.mark.ftqc_reference
def test_locked_adjoint_keeps_nonreal_phase():
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    source = ReflectionUsingPrepare.reflection_around_zero((1,), global_phase=1j)
    assert source.adjoint() == source
    matrix = source.tensor_contract()
    np.testing.assert_allclose(matrix @ source.adjoint().tensor_contract(), -np.eye(2))


def test_validation():
    with pytest.raises(ValueError, match="nonnegative width"):
        prepare_reflection_requirements(-1, controlled=False)
    sel, work, slots, _ = layout(2, None)
    for options, error in (
        ({"control_value": 2}, "zero or one"),
        ({"adjoint": 1}, "boolean"),
        ({"global_phase": 2}, "unit coefficient"),
        ({"prepare": (Instruction("h", (0,)),)}, "preparation pair"),
    ):
        with pytest.raises(ValueError, match=error):
            prepare_reflection_ir(sel, work, slots, **options)
    with pytest.raises(ValueError, match="clean work"):
        prepare_reflection_ir(sel, (), slots)
