"""Check the locked Ising leaf and ordered product connections and phases."""

from math import pi

import pytest
import torch

from flagquantum.algorithms.primitives.ising_evolution import (
    IsingXUnitary,
    IsingZZUnitary,
)
from flagquantum.algorithms.primitives.trotter_product import TrotterProduct
from flagquantum.circuit import Circuit
from flagquantum.core.ir import Instruction
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.simulation.unitary import get_unitary
from tests.unit.register_source_graph import walk_source_graph

pytestmark = pytest.mark.unit


def _source_instructions(source, wires):
    from qualtran.bloqs.basic_gates import CNOT, Rx, Rz

    def leaf(bloq, inputs, desired):
        if isinstance(bloq, CNOT):
            gate = Instruction("cx", inputs["ctrl"] + inputs["target"])
        elif isinstance(bloq, (Rx, Rz)):
            gate = Instruction(
                "rx" if isinstance(bloq, Rx) else "rz",
                inputs["q"],
                {"theta": float(bloq.angle)},
            )
        else:
            return None
        return (gate,), inputs

    ports = {"system": wires}
    instructions, output = walk_source_graph(source, ports, ports, leaf)
    assert output == ports
    return instructions


def _assert_source_cost(source, op, circuit):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    result = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
    source_cost = get_cost_value(source, QECGatesCost())
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(result.logical.gates, category) == getattr(source_cost, category)
    assert result.logical.peak_logical_qubits == get_cost_value(source, QubitCount())
    assert result.emitted is not None
    assert result.emitted.unconditional.get("measure", 0) == 0
    assert not result.emitted.conditional


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("axis,sites", [("x", 1), ("x", 3), ("zz", 2), ("zz", 4)])
@pytest.mark.parametrize("angle", [0.371, pi / 4, -pi / 2])
def test_ising_leaf_order_phase_and_resources(axis, sites, angle):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.trotter.ising import IsingXUnitary as SourceX
    from qualtran.bloqs.chemistry.trotter.ising import IsingZZUnitary as SourceZZ

    source = (SourceX if axis == "x" else SourceZZ)(sites, angle)
    term = (IsingXUnitary if axis == "x" else IsingZZUnitary)(sites)
    mapped = tuple(reversed(range(1, sites + 1)))
    op = term.make_on(system=mapped, angle=angle)
    circuit = Circuit(sites + 1).append(op)
    assert circuit.to_ir().instructions == _source_instructions(source, mapped)
    _assert_source_cost(source, op, circuit)

    contiguous = Circuit(sites).append(
        term.make_on(system=tuple(range(sites)), angle=angle)
    )
    torch.testing.assert_close(
        get_unitary(contiguous, dtype=torch.complex128),
        torch.from_numpy(source.tensor_contract()),
        atol=1e-12,
        rtol=0,
    )


_S2 = ((0, 1, 0), (0.05, 2, 0.05))
_S4 = (
    (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
    (
        0.20724538589718786,
        0.4144907717943757,
        0.4144907717943757,
        0.4144907717943757,
        -0.12173615769156357,
        -0.6579630871775028,
        -0.12173615769156357,
        0.4144907717943757,
        0.4144907717943757,
        0.4144907717943757,
        0.20724538589718786,
    ),
)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "indices,coefficients", [_S2, _S4, ((-1, 0, 1, 99), (0.2, -0.7, 0.4))]
)
@pytest.mark.parametrize("sites", [2, 3, 4])
def test_product_literal_schedule_forward_and_adjoint(sites, indices, coefficients):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.trotter.ising import IsingXUnitary as SourceX
    from qualtran.bloqs.chemistry.trotter.ising import IsingZZUnitary as SourceZZ
    from qualtran.bloqs.chemistry.trotter.trotterized_unitary import TrotterizedUnitary

    dt = 0.125
    native = TrotterProduct(
        (IsingXUnitary(sites), IsingZZUnitary(sites)), indices, coefficients, dt
    )
    source = TrotterizedUnitary(
        (SourceX(sites, 5), SourceZZ(sites, 5)), indices, coefficients, dt
    )
    wires = tuple(range(sites))
    forward = native.make_on(system=wires)
    circuit = Circuit(sites).append(forward)
    inverse = Circuit(sites).append(native.make_on(system=wires, adjoint=True))
    assert circuit.to_ir().instructions == _source_instructions(source, wires)
    assert inverse.to_ir().instructions == _source_instructions(source.adjoint(), wires)
    _assert_source_cost(source, forward, circuit)
    expected = torch.from_numpy(source.tensor_contract())
    torch.testing.assert_close(
        get_unitary(circuit, dtype=torch.complex128), expected, atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        get_unitary(inverse, dtype=torch.complex128), expected.mH, atol=1e-12, rtol=0
    )


def test_ising_layout_and_source_self_edge_boundary():
    with pytest.raises(ValueError, match="at least one"):
        IsingXUnitary(0)
    with pytest.raises(ValueError, match="at least two"):
        IsingZZUnitary(1)
    with pytest.raises(ValueError, match="identical"):
        TrotterProduct((IsingXUnitary(1), IsingZZUnitary(2)), (0, 1), (1, 1), 1)
    with pytest.raises(ValueError, match="distinct"):
        IsingZZUnitary(2).make_on(system=(0, 0), angle=0.5)


@pytest.mark.ftqc_reference
def test_symbolic_product_matches_original_angles_and_resource_categories():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.trotter.ising import IsingXUnitary as SourceX
    from qualtran.bloqs.chemistry.trotter.ising import IsingZZUnitary as SourceZZ
    from qualtran.bloqs.chemistry.trotter.trotterized_unitary import TrotterizedUnitary
    from sympy import symbols

    from flagquantum.core.parameters import Parameter

    source_dt, source_weight = symbols("dt weight", real=True)
    dt, weight = Parameter("dt"), Parameter("weight")
    source_terms = (SourceX(2, 0), SourceZZ(2, 0))
    source = TrotterizedUnitary(source_terms, (1, 0), (source_weight, -0.5), source_dt)
    product = TrotterProduct(
        (IsingXUnitary(2), IsingZZUnitary(2)), (1, 0), (weight, -0.5), dt
    )
    wires = (1, 0)
    op = product.make_on(system=wires)
    circuit = Circuit(2).append(op)
    _assert_source_cost(source, op, circuit)
    angles = [
        instance.bloq.angle
        for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
    ]
    assert angles == [2 * source_weight * source_dt, -1.0 * source_dt]
    for timestep, coefficient in [(0.125, 0.25), (-0.25, 1.5)]:
        bound = circuit.bind_parameters({dt: timestep, weight: coefficient})
        numeric = TrotterizedUnitary(
            source_terms, (1, 0), (coefficient, -0.5), timestep
        )
        assert bound.to_ir().instructions == _source_instructions(numeric, wires)
