"""Compare momentum-transfer costs and live workspace with the locked source."""

from __future__ import annotations

from itertools import product
from math import sqrt
from typing import Any

import pytest
import torch

from flagquantum.algorithms.first_quantization_nu import (
    prepare_nu_leaf_signatures,
    prepare_nu_m_on,
    prepare_nu_signature,
)
from flagquantum.algorithms.first_quantization_nu_resource_model import (
    source_prepare_nu_cost,
)
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_product_cost,
    source_sum_of_squares_cost,
)
from flagquantum.compiler.registers.comparison.ordering import source_greater_cost
from flagquantum.compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
    uniform_preparation_requirements,
)
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from tests._reference_dependencies import unavailable_reference
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _assert_signature(native: Signature, source: Any) -> None:
    try:
        from qualtran._infra.data_types import QAny, QBit
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    assert [
        (
            reg.name,
            reg.shape,
            reg.side.value,
            QBit() if reg.dtype.encoding == "bit" else QAny(reg.bitsize),
        )
        for reg in native
    ] == [(reg.name, reg.shape, reg.side.value, reg.dtype) for reg in source]


@pytest.mark.parametrize("count", [2, 3, 5, 6, 8, 9, 14, 31, 32768])
def test_uniform_cost(count: int) -> None:
    try:
        from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
            PrepareUniformSuperposition,
        )
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    assert_source_model(
        source_uniform_preparation_cost(count), PrepareUniformSuperposition(count)
    )


@pytest.mark.parametrize(
    ("left", "right"), [(1, 1), (1, 2), (2, 1), (2, 4), (5, 3), (4, 8), (8, 4)]
)
def test_momentum_arithmetic_costs(left: int, right: int) -> None:
    try:
        from qualtran._infra.bloq import DecomposeNotImplementedError
        from qualtran.bloqs.arithmetic.comparison import GreaterThan
        from qualtran.bloqs.arithmetic.multiplication import Product, SumOfSquares
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    assert_source_model(source_greater_cost(left, right), GreaterThan(left, right))
    for model, source in (
        (source_product_cost(left, right), Product(left, right)),
        (source_sum_of_squares_cost(left, right), SumOfSquares(left, right)),
    ):
        assert_source_model(model, source)
        with pytest.raises(DecomposeNotImplementedError):
            source.decompose_bloq()


@pytest.mark.parametrize(
    ("p", "m"), [(1, 3), (2, 2), (2, 5), (2, 64), (3, 9), (6, 32768)]
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_prepare_nu_cost_and_bystanders(p: int, m: int, adjoint: bool) -> None:
    try:
        from qualtran._infra.adjoint import Adjoint
        from qualtran._infra.bloq import DecomposeNotImplementedError
        from qualtran._infra.quantum_graph import LeftDangle
        from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu import (
            PrepareNuState,
        )
        from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
            PrepareUniformSuperposition,
        )
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    source: Any = PrepareNuState(p, m)
    if adjoint:
        source = source.adjoint()
    _assert_signature(prepare_nu_signature(p, m), source.signature)
    signatures = prepare_nu_leaf_signatures(p, m, adjoint=adjoint)
    assert_source_model(source_prepare_nu_cost(p, m, adjoint=adjoint), source)
    composite = source.decompose_bloq()
    model = source_prepare_nu_cost(p, m, adjoint=adjoint, decomposed=True)
    assert_source_model(model, composite)

    live = {edge for edge in composite.connections if edge.left.binst == LeftDangle}
    stages = {}
    for node, incoming, outgoing in composite.iter_bloqnections():
        live.difference_update(incoming)
        bloq = node.bloq.subbloq if isinstance(node.bloq, Adjoint) else node.bloq
        if isinstance(bloq, PrepareUniformSuperposition):
            assert len(incoming) == len(outgoing) == 1
            assert incoming[0].left.reg.name == outgoing[0].right.reg.name == "m"
            name = f"PrepareUniformSuperposition({m})"
        else:
            with pytest.raises(DecomposeNotImplementedError):
                node.bloq.decompose_bloq()
            name = type(bloq).__name__
            _assert_signature(signatures[name], node.bloq.signature)
        stages[name] = sum(edge.shape for edge in live)
        live.update(outgoing)
    assert stages == {
        call.component.name: call.bystander_qubits for call in model.calls
    }
    flags = {"flag", "flag_mu_prep", "flag_minus_zero", "flag_nu_lt_mu", "flag_ineq"}
    edges = {
        (edge.left.reg.name, edge.right.reg.name)
        for edge in composite.connections
        if edge.left.reg.name in flags or edge.right.reg.name in flags
    }
    forward = {
        ("flag", "flag_mu_prep"),
        ("flag_minus_zero", "flag_minus_zero"),
        ("flag_nu_lt_mu", "flag_ineq"),
    }
    assert edges == ({(end, start) for start, end in forward} if adjoint else forward)
    assert {
        (edge.left.reg.name, edge.right.reg.name)
        for edge in composite.connections
        if {"flag_nu", "succ"} == {edge.left.reg.name, edge.right.reg.name}
    } == {("flag_nu", "succ"), ("succ", "flag_nu")}


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("projectile", [False, True])
def test_momentum_m_native_append_matches_source_state_and_feedback(
    adjoint: bool, projectile: bool
) -> None:
    try:
        import cirq
        import numpy as np
        from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu import (
            PrepareNuState,
        )
        from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_nu import (
            PrepareNuStateWithProj,
        )
        from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
            PrepareUniformSuperposition,
        )
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    count = 3
    parent = (
        PrepareNuStateWithProj(1, 2, count) if projectile else PrepareNuState(2, count)
    )
    composite = parent.decompose_bloq()
    (source,) = (
        node.bloq
        for node, _, _ in composite.iter_bloqsoqs()
        if isinstance(node.bloq, PrepareUniformSuperposition)
    )
    target = (2, 0)
    _, work, slots = uniform_preparation_requirements(count)
    workspace = tuple(range(3, 3 + work))
    wires = tuple(cirq.LineQubit.range(3))
    operation = source.on(*(wires[q] for q in target))
    if adjoint:
        operation = cirq.inverse(operation)
    reference = cirq.Circuit(
        cirq.decompose(
            operation,
            context=cirq.DecompositionContext(
                cirq.GreedyQubitManager("m", maximize_reuse=True)
            ),
            on_stuck_raise=None,
            keep=lambda op: op.gate == cirq.H
            or isinstance(op.gate, (cirq.ControlledGate, cirq.Rz, cirq.ResetChannel))
            or cirq.is_measurement(op)
            or isinstance(op, cirq.ClassicallyControlledOperation),
        )
    )
    scratch = tuple(sorted(reference.all_qubits() - set(wires)))
    values = torch.arange(1, 9, dtype=torch.float64)
    logical = torch.complex(values, values.flip(0))
    logical /= logical.norm()
    source_input = np.zeros(8 << len(scratch), dtype=np.complex128)
    source_input[:: 1 << len(scratch)] = logical.numpy()
    ref = (
        cirq.Simulator(dtype=np.complex128, seed=3)
        .simulate(reference, qubit_order=(*wires, *scratch), initial_state=source_input)
        .final_state_vector.reshape(8, -1)
    )
    np.testing.assert_allclose(ref[:, 1:], 0, atol=3e-12, rtol=0)
    initial = torch.zeros(8 << work, dtype=torch.complex128)
    initial[:: 1 << work] = logical
    expected = torch.zeros_like(initial)
    expected[:: 1 << work] = torch.from_numpy(ref[:, 0].copy())
    circuit = DynamicCircuit(3 + work).append(
        prepare_nu_m_on(
            count,
            target,
            workspace=workspace,
            classical_bits=tuple(range(slots)),
            adjoint=adjoint,
        )
    )
    instructions = tuple(circuit.to_ir().instructions)
    measurements = sum(gate.name == "measure" for gate in instructions)
    assert (
        sum(gate.name in ("t", "tdg") for gate in instructions),
        measurements,
        work,
    ) == (20, 5, 4)
    for history in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(initial, instructions, history),
            expected / sqrt(2**measurements),
            atol=3e-12,
            rtol=0,
        )


def test_prepare_nu_source_empty_m_is_explicit() -> None:
    try:
        from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu import (
            PrepareNuState,
        )
        from qualtran.resource_counting._costing import get_cost_value
        from qualtran.resource_counting._qubit_counts import QubitCount
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    _assert_signature(prepare_nu_signature(2, 1), PrepareNuState(2, 1).signature)
    with pytest.raises(RuntimeError, match="does not accept Soquets"):
        get_cost_value(PrepareNuState(2, 1), QubitCount())
    with pytest.raises(ValueError, match="m_param >= 2"):
        source_prepare_nu_cost(2, 1)
