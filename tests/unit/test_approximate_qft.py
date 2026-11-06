"""Truncated phase-gradient QFT with source operation bindings and branches."""

from collections import Counter

import pytest
import torch

from flagquantum.algorithms.primitives.approximate_qft import (
    approximate_qft_ir,
    approximate_qft_precision,
    approximate_qft_workspace_size,
)
from flagquantum.compiler.registers.phase_addition import phase_add_ir
from flagquantum.core.ir import Instruction
from tests.unit.test_qft_phase_gradient import check_source_operator, layout

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "width,precision,with_reverse",
    [(1, 0, True), (2, 2, True), (3, 2, False), (3, 3, True), (5, 3, False)],
)
def test_locked_approximate_stream_and_model(width, precision, with_reverse):
    pytest.importorskip("qualtran")
    import cirq
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.qft.approximate_qft import ApproximateQFT
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

    source = ApproximateQFT(width, precision, with_reverse)
    size = approximate_qft_workspace_size(width, precision)
    bound = layout(width, precision, (size, size))
    data, gradient, work, slots = bound
    source_op = source.on(*cirq.LineQubit.range(width + precision))
    layers = cirq.decompose(
        source_op,
        keep=lambda op: op.gate in (cirq.H, cirq.SWAP)
        or isinstance(getattr(op.gate, "bloq", op.gate), AddIntoPhaseGrad),
    )
    for adjoint in (False, True):
        expected = []
        modeled_t = 0
        for op in reversed(layers) if adjoint else layers:
            wires = tuple(q.x for q in op.qubits)
            if op.gate in (cirq.H, cirq.SWAP):
                expected.append(
                    Instruction("h" if op.gate == cirq.H else "swap", wires)
                )
                continue
            child = getattr(op.gate, "bloq", op.gate)
            assert child.right_shift == child.controlled_by == 1
            with pytest.raises(DecomposeNotImplementedError):
                child.decompose_bloq()
            count = child.x_bitsize
            assert child.phase_bitsize == count + 1
            expected.extend(
                phase_add_ir(
                    wires[1 : count + 1],
                    wires[count + 1 :],
                    work[: 2 * count],
                    slots[: 2 * count],
                    right_shift=child.right_shift,
                    sign=-1 if adjoint else 1,
                    control=wires[0],
                )
            )
            modeled_t += 8 * (count - 1)
        gates = approximate_qft_ir(*bound, with_reverse=with_reverse, adjoint=adjoint)
        assert gates == tuple(expected)
        counts = Counter(g.name for g in gates)
        assert counts["swap"] == (width // 2 if with_reverse else 0)
        assert counts["measure"] == sum(
            2 * min(i, max(0, precision - 1)) for i in range(width)
        )
        assert counts["t"] + counts["tdg"] == 4 * counts["measure"]
        if width == 5:
            assert modeled_t == 24
            assert counts["t"] + counts["tdg"] == 56
    if width <= 3:
        matrix = torch.tensor(cirq.unitary(source_op), dtype=torch.complex128)
        check_source_operator(
            matrix,
            (*data, *gradient),
            work,
            lambda adjoint: approximate_qft_ir(
                *bound, with_reverse=with_reverse, adjoint=adjoint
            ),
        )


@pytest.mark.ftqc_reference
def test_concrete_precision_formula_and_binding():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.qft.approximate_qft import ApproximateQFT

    for width, epsilon in ((1, 1 / 2), (8, 1 / 16), (1000, 1 / 1024)):
        assert (
            approximate_qft_precision(width, epsilon)
            == ApproximateQFT.from_epsilon(width, epsilon).phase_bitsize
        )
    # The singleton and one-bit gradient have no retained controlled phases.
    assert approximate_qft_ir((0,), (), (), ()) == (Instruction("h", (0,)),)
    assert approximate_qft_ir((0, 1), (2,), (), ()) == (
        Instruction("h", (0,)),
        Instruction("h", (1,)),
        Instruction("swap", (0, 1)),
    )


def test_truncated_gradient_kickback_and_cleanup():
    from math import pi

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.phase_gradient import phase_gradient_state_ir
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from tests.unit.test_uniform_preparation import embedded

    width, precision = 3, 2
    work = approximate_qft_workspace_size(width, precision)
    bound = layout(width, precision, (work, work))
    _, gradient, _, _ = bound
    logical = torch.arange(1, (1 << width) + 1, dtype=torch.float64)
    logical = torch.complex(logical, 1 - logical)
    logical /= logical.norm()
    reference = Circuit(width, inputs=logical, dtype=torch.complex128)
    reference.h(0)
    reference.gate("cphase", (1, 0), theta=pi / 2).h(1)
    reference.gate("cphase", (2, 1), theta=pi / 2).h(2)
    reference.gate("swap", (0, 2))
    state = embedded(logical, precision + work)
    expected = embedded(reference.state().flatten(), precision + work)
    size = width + precision + work
    gates = (
        *phase_gradient_state_ir(gradient, exponent=-1),
        *approximate_qft_ir(*bound),
        *phase_gradient_state_ir(gradient, inverse=True, exponent=-1),
    )
    circuit = DynamicCircuit(size, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(size, gates))
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=3e-12, rtol=0
    )
