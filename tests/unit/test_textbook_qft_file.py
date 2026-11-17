"""Check locked textbook Fourier declarations and physical resource separation."""

from collections import Counter
from itertools import product

import pytest
import torch
from sympy import Symbol, simplify

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.textbook_qft_catalog import (
    textbook_qft_examples,
    textbook_qft_graph,
)
from flagquantum.algorithms.primitives.qft import _qft_circuit
from flagquantum.algorithms.primitives.qft_resource_model import (
    source_textbook_qft_calls,
    source_textbook_qft_cost,
    source_textbook_qft_signature,
    source_textbook_qft_symbolic_formula,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.core.ir.program.registers import RegisterSide

pytestmark = pytest.mark.unit
_GATE_FIELDS = (
    "t",
    "toffoli",
    "cswap",
    "and_bloq",
    "clifford",
    "rotation",
    "measurement",
)


@pytest.mark.ftqc_reference
def test_original_unitary_assertions_through_seven_wires() -> None:
    cirq = pytest.importorskip("cirq")
    qft_package = pytest.importorskip("qualtran.bloqs.qft")
    for width, reverse in product(range(2, 8), (False, True)):
        source = qft_package.QFTTextBook(width, with_reverse=reverse)
        cirq_gate = cirq.QuantumFourierTransformGate(width, without_reverse=not reverse)
        forward = _qft_circuit(width, with_reverse=reverse)
        basis = torch.eye(2**width, dtype=torch.complex128)
        for inverse in (False, True):
            expected = torch.tensor(
                cirq.unitary(cirq_gate**-1 if inverse else cirq_gate)
            )
            original = source.adjoint() if inverse else source
            torch.testing.assert_close(
                torch.tensor(original.tensor_contract()), expected, rtol=0, atol=1e-12
            )
            native = forward.adjoint() if inverse else forward
            circuit = Circuit(width, inputs=basis, dtype=torch.complex128)
            circuit.append(FieldOperation(width, tuple(native.to_ir().instructions)))
            torch.testing.assert_close(
                circuit.state().reshape(basis.shape).T, expected, rtol=0, atol=2e-12
            )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 5, 10, 40, 41, 123, Symbol("n")])
@pytest.mark.parametrize("with_reverse", [False, True])
def test_locked_textbook_signature_calls_and_model(
    width: int | Symbol, with_reverse: bool
) -> None:
    qft_package = pytest.importorskip("qualtran.bloqs.qft")
    cost_package = pytest.importorskip("qualtran.resource_counting")
    source = qft_package.QFTTextBook(width, with_reverse=with_reverse)
    (original,) = source.signature
    (native,) = source_textbook_qft_signature(width)
    assert original.name == native.name == "q"
    assert native.side == RegisterSide.THRU
    assert original.bitsize == native.bitsize == width

    source_calls = source.build_call_graph(cost_package.SympySymbolAllocator())
    normalized = {}
    for child, count in source_calls.items():
        name = type(child).__name__
        if name == "PhaseGradientUnitary":
            assert child.exponent == 0.5 and child.is_controlled
            bits = child.bitsize
        else:
            bits = 1 if name == "Hadamard" else 2
        normalized[name, bits] = count
    assert source_textbook_qft_calls(width, with_reverse=with_reverse) == normalized

    source_cost = cost_package.get_cost_value(source, cost_package.QECGatesCost())
    source_qubits = cost_package.get_cost_value(source, cost_package.QubitCount())
    if isinstance(width, int):
        native_cost = estimate_component_resources(
            source_textbook_qft_cost(width, with_reverse=with_reverse)
        )
        assert native_cost.peak_logical_qubits == source_qubits
        for key in _GATE_FIELDS:
            assert getattr(native_cost.gates, key) == getattr(source_cost, key)
    else:
        symbolic = source_textbook_qft_symbolic_formula(
            width, with_reverse=with_reverse
        )
        assert simplify(symbolic.logical_qubits - source_qubits) == 0
        for key in _GATE_FIELDS:
            assert simplify(getattr(symbolic, key) - getattr(source_cost, key)) == 0


def test_small_angle_source_cost_and_emitted_records_keep_distinct_counts() -> None:
    width = 123
    model = estimate_component_resources(source_textbook_qft_cost(width))
    assert model.gates.rotation == 3774
    assert model.gates.t == 121
    assert model.gates.and_bloq == 7503
    circuit = _qft_circuit(width)
    assert Counter(gate.name for gate in circuit.to_ir().instructions) == {
        "h": 123,
        "cphase": 7503,
        "swap": 61,
    }


def test_original_example_graphs_and_selectable_reversal() -> None:
    concrete, symbolic = textbook_qft_examples()
    assert (concrete, symbolic) == (3, Symbol("n"))
    assert textbook_qft_graph(concrete).leaf_counts == {
        "Hadamard": 3,
        "PhaseGradientUnitary(1)": 1,
        "PhaseGradientUnitary(2)": 1,
        "TwoBitSwap": 1,
    }
    assert textbook_qft_graph(symbolic, with_reverse=False).leaf_counts == {
        "Hadamard": symbolic,
        "PhaseGradientUnitary(n - 1)": symbolic // 2,
    }
