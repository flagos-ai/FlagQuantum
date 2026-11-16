"""Check explicit Fourier measurement against the frozen shim and the DFT."""

from math import pi, sqrt

import pytest
import torch

from flagquantum.algorithms.primitives.measure_qft import (
    MeasureQFT,
    source_measure_qft_cost,
    source_measure_qft_signature,
    source_measure_qft_symbolic_formula,
)
from flagquantum.algorithms.primitives.qft import qft
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [0, 1, 2, 3, 4, 8])
def test_locked_factoring_shim_signature_and_declared_resources(width):
    pytest.importorskip("qualtran")
    from qualtran import QBit, Side
    from qualtran._infra.composite_bloq import BloqError
    from qualtran.bloqs.cryptography._factoring_shims import MeasureQFT as Source
    from qualtran.resource_counting import (
        QECGatesCost,
        QubitCount,
        SympySymbolAllocator,
        get_cost_value,
    )

    source = Source(width)
    (reg,) = source_measure_qft_signature(width)
    (original,) = source.signature
    assert (reg.name, reg.shape, reg.bitsize, reg.side) == (
        "x",
        (width,),
        1,
        RegisterSide.LEFT,
    )
    assert original.side == Side.LEFT and original.dtype == QBit()
    assert (reg.name, reg.shape) == (original.name, original.shape)
    model = source_measure_qft_cost(width)
    calls = source.build_call_graph(SympySymbolAllocator())
    assert [(edge.component.name, edge.repetitions) for edge in model.calls] == [
        (type(child).__name__, count) for child, count in calls.items()
    ]
    gates = estimate_component_resources(model).gates
    expected = get_cost_value(source, QECGatesCost())
    for key in ("t", "and_bloq", "clifford", "measurement", "rotation"):
        assert getattr(gates, key) == getattr(expected, key)

    transform = model.calls[0].component
    source_qft = next(iter(calls))
    if width:
        assert estimate_component_resources(
            transform
        ).peak_logical_qubits == get_cost_value(source_qft, QubitCount())
    with pytest.raises(
        BloqError, match="were not used" if width else "does not accept Soquets"
    ):
        source.decompose_bloq()
    with pytest.raises(RuntimeError, match="unexpected error"):
        get_cost_value(source, QubitCount())


@pytest.mark.ftqc_reference
def test_locked_symbolic_width_uses_grouped_qft_formula():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography._factoring_shims import MeasureQFT as Source
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import Symbol, simplify

    width = Symbol("n", integer=True, positive=True)
    source = Source(width)
    (reg,) = source_measure_qft_signature(width)
    assert reg.shape_symbolic == (width,)
    assert next(iter(source.signature)).shape_symbolic == (width,)
    model = source_measure_qft_symbolic_formula(width)
    expected = get_cost_value(source, QECGatesCost())
    for key in ("t", "and_bloq", "clifford", "measurement", "rotation"):
        assert simplify(getattr(model, key) - getattr(expected, key)) == 0
    assert simplify(model.logical_qubits - get_cost_value(source, QubitCount())) == 0


@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_bound_measure_qft_all_projected_complex_branches(width):
    wires = tuple(reversed(range(1, width + 1)))
    bits = tuple(2 * i + 1 for i in reversed(range(width)))
    n_wires = width + 1
    op = MeasureQFT(width).make_on(x=wires, result_bits=bits)
    ir = DynamicCircuit(n_wires).append(op).to_ir()
    transform = Circuit(n_wires).compose(qft(width), qubits=wires).to_ir()
    assert ir.instructions[:-width] == transform.instructions
    assert [
        (g.wires[0], g.metadata["classical_bit"]) for g in ir.instructions[-width:]
    ] == list(zip(wires, bits, strict=True))
    report = estimate_ftqc_resources(op, emitted_ir=ir)
    assert report.emitted is not None
    assert report.emitted.unconditional.get("measure") == width
    assert not report.emitted.conditional

    size = 1 << width
    values = torch.arange(1, 2 * size + 1, dtype=torch.float64)
    state = torch.complex(values, values.remainder(7) - 4)
    state /= state.norm()
    permutation = (0, *wires)
    ordered = state.reshape((2,) * n_wires).permute(permutation).reshape(2, size)
    positions = torch.arange(size, dtype=torch.float64)
    fourier = torch.exp((2j * pi / size) * torch.outer(positions, positions)) / sqrt(
        size
    )
    output = ordered @ fourier.T
    for outcome in range(size):
        history = tuple((outcome >> shift) & 1 for shift in reversed(range(width)))
        branch = torch.zeros_like(output)
        branch[:, outcome] = output[:, outcome]
        expected = branch.reshape((2,) * n_wires).permute(permutation).flatten()
        torch.testing.assert_close(
            forced_branch(state, ir.instructions, history),
            expected,
            atol=4e-12,
            rtol=0,
        )


def test_measure_qft_empty_and_invalid_bindings():
    noop = MeasureQFT(0).make_on(x=(), result_bits=())
    assert not Circuit(1).append(noop).to_ir().instructions
    with pytest.raises(ValueError, match="width"):
        MeasureQFT(2).make_on(x=(1,), result_bits=(3, 7))
    with pytest.raises(ValueError, match="distinct"):
        MeasureQFT(2).make_on(x=(0, 1), result_bits=(3, 3))
    bound = MeasureQFT(2).make_on(x=(0, 1), result_bits=(4, 2))
    with pytest.raises(ValueError, match="DynamicCircuit"):
        Circuit(2).append(bound)
    circuit = DynamicCircuit(1)
    with pytest.raises(ValueError, match="outside circuit range"):
        circuit.append(bound)
    assert not circuit.to_ir().instructions
    existing = DynamicCircuit(3).measure(2, classical_bit=4).append(bound)
    assert [
        (gate.wires, gate.metadata["classical_bit"])
        for gate in existing.to_ir().instructions
        if gate.name == "measure"
    ] == [((2,), 4), ((0,), 4), ((1,), 2)]


def test_terminal_execution_reports_named_result_slots():
    from flagquantum.runtime.dynamic.execution import run_dynamic

    wires, slots = (3, 1, 2), (5, 3, 1)
    circuit = DynamicCircuit(4, dtype=torch.complex128)
    circuit.x(0).measure(0, classical_bit=0)
    circuit.x(wires[0]).x(wires[2]).compose(qft(3, inverse=True), qubits=wires)
    circuit.append(MeasureQFT(3).make_on(x=wires, result_bits=slots))
    result = run_dynamic(circuit, shots=8, seed=7, strategy="trajectory")
    assert torch.all(result.classical_bits[:, slots] == torch.tensor([1, 0, 1]))
    assert torch.all(result.classical_bits[:, 0] == 1)
