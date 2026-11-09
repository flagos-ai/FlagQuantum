"""Pinned recursive QFT layout, phase-gradient kickback and clean inverse."""

from collections import Counter
from itertools import product
from math import sqrt

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.qft import _qft_circuit
from flagquantum.algorithms.primitives.qft_phase_gradient import (
    qft_phase_gradient_ir,
    qft_phase_gradient_requirements,
)
from flagquantum.compiler.registers.arithmetic.integer_product import (
    integer_product_requirements,
)
from flagquantum.compiler.registers.arithmetic.product_accumulation import (
    product_accumulation_ir,
)
from flagquantum.compiler.registers.phase.phase_gradient import phase_gradient_state_ir
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_uniform_preparation import embedded

pytestmark = pytest.mark.unit


def check_source_operator(matrix, qubits, workspace, emit):
    values = torch.arange(1, (1 << len(qubits)) + 1, dtype=torch.float64)
    initial = torch.complex(values, 2 - values)
    initial /= initial.norm()
    state = embedded(initial, len(workspace))
    size = len(qubits) + len(workspace)
    for adjoint in (False, True):
        expected = embedded(
            (matrix.mH if adjoint else matrix) @ initial, len(workspace)
        )
        circuit = DynamicCircuit(size, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(size, emit(adjoint)))
        stream = tuple(circuit.to_ir().instructions)
        events = sum(g.name == "measure" for g in stream)
        if events <= 5:
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(state, stream, history),
                    expected / sqrt(2**events),
                    atol=3e-12,
                    rtol=0,
                )
        else:
            result = run_dynamic(circuit, shots=8, seed=31, strategy="trajectory")
            torch.testing.assert_close(
                result.final_states.reshape(8, -1),
                expected.expand(8, -1),
                atol=3e-12,
                rtol=0,
            )


def layout(width, precision, requirements):
    work, slots = requirements
    data = tuple(range(width))
    gradient = tuple(range(width, width + precision))
    workspace = tuple(range(width + precision, width + precision + work))
    return data, gradient, workspace, tuple(range(slots))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "width,with_reverse", [(1, False), (2, True), (3, False), (4, True), (5, False)]
)
def test_locked_recursive_stream_and_model(width, with_reverse):
    pytest.importorskip("qualtran")
    import cirq
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct
    from qualtran.bloqs.qft.qft_phase_gradient import QFTPhaseGradient
    from qualtran.resource_counting import SympySymbolAllocator

    source = QFTPhaseGradient(width, with_reverse)
    bound = layout(width, width, qft_phase_gradient_requirements(width))
    data, gradient, work, slots = bound
    qubits = tuple(cirq.LineQubit.range(2 * width))
    source_op = source.on(*qubits)
    layers = cirq.decompose(
        source_op,
        keep=lambda op: op.gate in (cirq.H, cirq.SWAP)
        or isinstance(getattr(op.gate, "bloq", op.gate), PlusEqualProduct),
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
            a, b, c = child.a_bitsize, child.b_bitsize, child.result_bitsize
            with pytest.raises(DecomposeNotImplementedError):
                child.decompose_bloq()
            _, product_work = integer_product_requirements(a, b)
            expected.extend(
                product_accumulation_ir(
                    wires[:a],
                    wires[a : a + b],
                    wires[a + b :],
                    work[:product_work],
                    work[product_work : product_work + c - 1],
                    slots[: c - 1],
                    adjoint=adjoint,
                )
            )
            modeled_t += 8 * max(a, b) ** 2
        gates = qft_phase_gradient_ir(
            *bound, with_reverse=with_reverse, adjoint=adjoint
        )
        assert gates == tuple(expected)
        counts = Counter(g.name for g in gates)
        assert counts["h"] - 3 * counts["measure"] == width
        assert counts["swap"] == (width // 2 if with_reverse else 0)
        if width == 4:
            model = source.build_call_graph(SympySymbolAllocator())
            assert (
                model[QFTPhaseGradient(2, True)] == 1
            )  # Source merges identical dictionary keys.
            assert modeled_t == 48
            assert counts["t"] + counts["tdg"] == 52
            assert counts["measure"] == 13
    if width <= 3:
        matrix = torch.tensor(cirq.unitary(source_op), dtype=torch.complex128)
        check_source_operator(
            matrix,
            (*data, *gradient),
            work,
            lambda adjoint: qft_phase_gradient_ir(
                *bound, with_reverse=with_reverse, adjoint=adjoint
            ),
        )


def test_gradient_preserved_and_cleaned_on_append():
    width = 3
    data, gradient, work, slots = layout(
        width, width, qft_phase_gradient_requirements(width)
    )
    values = torch.arange(1, (1 << width) + 1, dtype=torch.float64)
    logical = torch.complex(values, 4 - values)
    logical /= logical.norm()
    size = width * 2 + len(work)
    circuit = DynamicCircuit(
        size, inputs=embedded(logical, width + len(work)), dtype=torch.complex128
    )
    gates = (
        *phase_gradient_state_ir(gradient, exponent=-1),
        *qft_phase_gradient_ir(data, gradient, work, slots),
        *phase_gradient_state_ir(gradient, inverse=True, exponent=-1),
    )
    circuit.append(FieldOperation(size, gates))
    result = run_dynamic(circuit, shots=8, seed=11, strategy="trajectory")
    expected = _qft_circuit(width)
    from flagquantum.circuit import Circuit

    wanted = (
        Circuit(width, inputs=logical, dtype=torch.complex128)
        .append(FieldOperation(width, tuple(expected.to_ir().instructions)))
        .state()
        .flatten()
    )
    torch.testing.assert_close(
        result.final_states.reshape(8, -1),
        embedded(wanted, width + len(work)).expand(8, -1),
        atol=3e-12,
        rtol=0,
    )
