"""Compare the grid kinetic call graph, complex branches and physical costs."""

from cmath import exp
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.algorithms.primitives.grid_kinetic import (
    GridKineticEnergy,
    source_grid_kinetic_cost,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.arithmetic.integer_square import (
    sum_of_squares_requirements,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("electrons,points", [(1, 1), (2, 2), (3, 4), (12, 17)])
def test_locked_kinetic_graph(electrons, points):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.trotter.grid_ham.kinetic import KineticEnergy
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = KineticEnergy(electrons, points)
    native = source_grid_kinetic_cost(electrons, points)
    instances = tuple(
        instance.bloq for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
    )
    assert [type(op).__name__ for op in instances] == [
        call.component.name for call in native.calls
    ]
    result = estimate_ftqc_resources(native)
    expected = get_cost_value(source, QECGatesCost())
    for field in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(result.logical.gates, field) == getattr(expected, field)
    assert result.logical.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("adjoint", [False, True])
def test_two_electron_small_grid_every_branch(adjoint):
    system = (((0,), (1,), (2,)), ((3,), (4,), (5,)))
    coefficient = 0.317
    op = GridKineticEnergy.make_on(
        grid_points=1,
        system=system,
        result=(6, 7, 8, 9),
        workspace=(10,),
        radians_per_unit=coefficient,
        adjoint=adjoint,
    )
    circuit = DynamicCircuit(12).append(op)
    ir = circuit.to_ir()
    input_state = torch.zeros(4096, dtype=torch.complex128)
    output = torch.zeros_like(input_state)
    for i, spectator in product(range(64), (0, 1)):
        index = (i << 6) | spectator
        value = exp(1j * (i - spectator / 3)) / sqrt(128)
        input_state[index] = value
        output[index] = value * exp(
            ((-1 if adjoint else 1) * 1j * coefficient) * i.bit_count()
        )
    for history in product((0, 1), repeat=4):
        torch.testing.assert_close(
            forced_branch(input_state, ir.instructions, history),
            output / 4,
            atol=2e-12,
            rtol=0,
        )
    report = estimate_ftqc_resources(op, emitted_ir=ir)
    assert report.emitted is not None
    assert report.emitted.t_count == 16
    assert report.emitted.unconditional["measure"] == 4
    assert report.logical.peak_logical_qubits == 10
    assert report.logical.gates.toffoli == 2


def test_two_bit_unsigned_grid_with_carries_and_slot_ownership():
    result_width, work_width = sum_of_squares_requirements(2, 3)
    result = tuple(range(6, 6 + result_width))
    work = tuple(range(6 + result_width, 6 + result_width + work_width))
    width = 6 + result_width + work_width
    system = (((0, 1), (2, 3), (4, 5)),)
    coefficient = -0.203
    op = GridKineticEnergy.make_on(
        grid_points=2,
        system=system,
        result=result,
        workspace=work,
        radians_per_unit=coefficient,
    )
    circuit = DynamicCircuit(width + 1)
    circuit.measure(width, classical_bit=3)
    circuit.append(op)
    assert {
        gate.metadata["classical_bit"]
        for gate in circuit.to_ir().instructions
        if gate.name == "measure"
    } == {3, 4}
    with pytest.raises(ValueError, match="DynamicCircuit"):
        Circuit(width).append(op)

    ir = DynamicCircuit(width).append(op).to_ir()
    before, after = {}, {}
    for x, y, z in product(range(4), repeat=3):
        word = (x << 4) | (y << 2) | z
        index = word << (width - 6)
        before[index] = exp(1j * word) / 8
        after[index] = before[index] * exp(1j * coefficient * (x * x + y * y + z * z))
    for seed in (4, 23, 51):
        observed = execute_sparse_register(
            CircuitIR(width, ir.instructions), before, seed=seed
        )
        assert all(
            abs(observed.get(i, 0) - after.get(i, 0)) < 2e-12
            for i in observed.keys() | after.keys()
        )
