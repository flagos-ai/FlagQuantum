"""Reconcile the original four-term signed block and full complex execution."""

from collections import Counter
from dataclasses import asdict
from itertools import product
from math import pi, sqrt
from typing import Any

import pytest
import torch

from examples.algorithms.linear_combination_catalog import (
    linear_combination_example,
    linear_combination_example_circuit,
    linear_combination_example_model,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_workspace,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_partition_leaf,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_original_four_term_signature_signal_and_cost() -> None:
    from qualtran.bloqs.block_encoding.linear_combination import (
        _linear_combination_block_encoding,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    source = _linear_combination_block_encoding()
    assert (
        type(source).of_terms(
            *zip(source._lambd, source._block_encodings, strict=True),
            lambd_bits=source.lambd_bits,
        )
        == source
    )
    encoding = linear_combination_example()
    widths = (
        encoding.system_bitsize,
        encoding.ancilla_bitsize,
        encoding.resource_bitsize,
    )
    assert widths == tuple(int(register.total_bits()) for register in source.signature)
    assert (encoding.alpha, encoding.epsilon) == (source.alpha, source.epsilon)
    assert encoding.coefficients == tuple(source._lambd)
    assert encoding.signal_state.selection_width == 2
    assert encoding.signal_state.emit((1, 2), (), (), ()) == ()
    model = linear_combination_example_model()
    source_calls = source.call_graph(max_depth=1, generalizer=ignore_split_join)[1]
    assert tuple(call.repetitions for call in model.calls) == tuple(
        source_calls.values()
    )
    resource = estimate_component_resources(model)
    assert {
        key: value for key, value in asdict(resource.gates).items() if value
    } == get_cost_value(source, QECGatesCost()).asdict()
    assert resource.peak_logical_qubits == get_cost_value(source, QubitCount())
    for adjoint in (False, True):
        ir = linear_combination_example_circuit(adjoint=adjoint).to_ir()
        emitted = count_emitted_instructions(ir)
        assert emitted.t_count == 4 * resource.gates.and_bloq + 7 * resource.gates.cswap
        assert emitted.unconditional["measure"] == resource.gates.measurement
        target = [gate.name for gate in ir.instructions if gate.wires[-1:] == (0,)]
        assert Counter(target) == {"cphase": 1, "ch": 1, "cx": 1, "cz": 1}


@pytest.mark.parametrize("adjoint", [False, True])
def test_original_four_term_source_gate_operands_and_feedback(adjoint: bool) -> None:
    from qualtran.bloqs.block_encoding.linear_combination import (
        _linear_combination_block_encoding,
    )

    encoding = linear_combination_example()
    source = _linear_combination_block_encoding()
    if adjoint:
        source = source.adjoint()
    circuit = linear_combination_example_circuit(adjoint=adjoint)
    _assert_source_flow(source, encoding, circuit)


def _assert_source_flow(source, encoding, circuit):
    from qualtran import Adjoint
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare

    system_end = encoding.system_bitsize
    ancilla_end = system_end + encoding.ancilla_bitsize
    through = ancilla_end + encoding.resource_bitsize
    ports = {
        "system": tuple(range(system_end)),
        "ancilla": tuple(range(system_end, ancilla_end)),
        "resource": tuple(range(ancilla_end, through)),
    }
    work = tuple(range(through, circuit.n_qubits))
    prep_work, prep_slots = encoding.coefficient_preparation.workspace_requirements
    select_work = indexed_component_workspace((encoding.select.selection_bitsize,), 0)

    def leaf(
        bloq: Any,
        inputs: SourcePorts,
        desired: SourcePorts,
    ) -> SourceResult | None:
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        base = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if isinstance(base, BlackBoxPrepare):
            declared = encoding.prepare.emit(
                inputs["selection"],
                inputs.get("junk", ()),
                work[:prep_work],
                tuple(range(prep_slots)),
                adjoint=isinstance(bloq, Adjoint),
            )
            slots = {
                gate.wires[0]: gate.metadata["classical_bit"]
                for gate in declared
                if gate.name == "measure"
            }
            used_work = work[:prep_work]
        elif isinstance(base, BlackBoxSelect):
            slots = dict(zip(work[:select_work], range(select_work), strict=True))
            used_work = work[:select_work]
        else:
            return None
        gates = reference_arithmetic_gates(
            bloq.decompose_bloq(),
            inputs,
            used_work,
            slots,
            cirq_style=True,
            preserve_components=True,
            keep_pauli_leaves=True,
            keep_exchange_leaves=True,
        )
        return gates, inputs

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert_source_dependencies(circuit.to_ir().instructions, expected)


@pytest.mark.parametrize("adjoint", [False, True])
def test_original_four_term_full_complex_amplitudes_and_work(adjoint: bool) -> None:
    from qualtran.bloqs.block_encoding.linear_combination import (
        _linear_combination_block_encoding,
    )

    source = _linear_combination_block_encoding()
    if adjoint:
        source = source.adjoint()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    circuit = linear_combination_example_circuit(adjoint=adjoint)
    gates = circuit.to_ir().instructions
    logical = matrix.shape[0].bit_length() - 1
    reserved = circuit.n_qubits - logical
    inputs = torch.arange(matrix.shape[0], dtype=torch.float64)
    data = torch.complex(inputs + 1, inputs % 13 - 5)
    data /= data.norm()
    zeros = torch.zeros(1 << reserved, dtype=torch.complex128)
    zeros[0] = 1
    initial = torch.kron(data, zeros)
    expected = torch.kron(matrix @ data, zeros)
    measurements = sum(gate.name == "measure" for gate in gates)
    assert measurements == 8
    for history in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected / sqrt(2**measurements),
            rtol=0,
            atol=4e-12,
        )


# Coefficient grids adapted from Qualtran v0.7.0 linear_combination_test.py.
# Copyright 2024 Google LLC, Apache-2.0.
EXACT2 = [[0.0, 1.0], [1 / 3, 1 / 3], [0.5, 0.5], [0.25, 0.25], [2.0, 6.0], [1.0, 0.0]]
EXACT3 = [
    [0.0, 0.0, 1.0],
    [1.0, -0.5, 0.5],
    [1 / 4, 1 / 4, -1 / 2],
    [1 / 2, 1 / 4, 1 / 4],
    [3 / 16, -1 / 4, 1 / 16],
    [-1.0, 0.0, 0.0],
]
EXACT5 = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [9 / 16, -1 / 16, 1 / 8, -1 / 8, 1 / 8],
    [1 / 4, 1 / 8, -1 / 16, 1 / 32, -1 / 32],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 0.5, 0.5, -1.0, 1.0],
]
APPROX2 = [
    [1 / 3, 2 / 3],
    [2 / 3, 1 / 3],
    [1 / 2, -1 / 4],
    [1 / 8, 1 / 2],
    [-1 / 3, 1 / 2],
    [1 / 3, 1 / 5],
    [1.0, -7.0],
    [3.7, 2.1],
    [1.0, 0.0],
]
APPROX5 = [
    [1.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, -1.0, 1.0],
    [1.5, 2.4, -3.3, 4.2, 5.1],
    [1 / 7, -2 / 7, 3 / 7, 4 / 7, -5 / 7],
]

_COEFFICIENT_CASES = (
    ("two", EXACT2, 1, 1e-7),
    ("two-wire", EXACT2, 1, 1e-7),
    ("three", EXACT3, 1, 1e-7),
    ("five", EXACT5, 1, 1e-7),
    ("hermitian", EXACT3, 1, 1e-7),
    ("nonhermitian", EXACT3, 1, 1e-7),
    ("approx-two", APPROX2, 9, 0.003),
    ("approx-five", APPROX5, 9, 0.002),
    ("normalization", ((2.0, 3.0),), 1, 1e-7),
)


@pytest.fixture(scope="module")
def coefficient_gates():
    from random import Random

    import numpy as np
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, Ry, Swap, TGate, XGate, ZGate
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate

    t, h, x, z, cx, swap = TGate(), Hadamard(), XGate(), ZGate(), CNOT(), Swap(1)
    ry = Ry(angle=np.pi / 4)
    gates = {
        "two": (t, h),
        "two-wire": (cx, h.controlled()),
        "three": (t, h, x),
        "five": (t, h, x, z, ry),
        "hermitian": (h.controlled(), cx, swap),
        "nonhermitian": (t.controlled(), cx, swap),
        "approx-two": (t, h),
        "approx-five": (t.controlled(), h.controlled(), cx, swap, ry.controlled()),
        "normalization": (t, h),
    }
    gate_rng, coefficient_rng = np.random.RandomState(1234), Random(1234)
    dense = []
    for _ in range(10):
        size, width = gate_rng.randint(3, 6), gate_rng.randint(1, 3)
        leaves = tuple(
            MatrixGate.random(width, random_state=gate_rng) for _ in range(size)
        )
        dense.append((leaves, tuple(coefficient_rng.uniform(-10, 10) for _ in leaves)))
    return gates, dense


def _source_coefficient_projection(source):
    """Contract the locked block with zero signal and junk port boundary states."""
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import IntEffect, IntState

    builder = BloqBuilder()
    system = builder.add_register("system", source.system_bitsize)
    ancilla = builder.add(IntState(0, source.ancilla_bitsize))
    resource = builder.add(IntState(0, source.resource_bitsize))
    system, ancilla, resource = builder.add_t(
        source, system=system, ancilla=ancilla, resource=resource
    )
    builder.add(IntEffect(0, source.ancilla_bitsize), val=ancilla)
    builder.add(IntEffect(0, source.resource_bitsize), val=resource)
    return builder.finalize(system=system).tensor_contract() * source.alpha


def _bind_coefficient_fixture(
    gates, coefficients, precision, *, inverse, named=False, scales=()
):
    """Bind independent matrix leaves through the existing positive-control interface."""
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
        LinearCombinationBlockEncoding,
    )
    from flagquantum.algorithms.primitives.unitary_block_encoding import (
        UnitaryBlockEncoding,
    )
    from flagquantum.core.ir import Instruction
    from flagquantum.dynamic import DynamicCircuit

    matrices = tuple(torch.as_tensor(gate.tensor_contract()) for gate in gates)
    dim = matrices[0].shape[0]
    width = dim.bit_length() - 1

    def child(gate, matrix):
        positive = torch.block_diag(torch.eye(dim, dtype=matrix.dtype), matrix)

        def make_on(predicate, ports, work, slots, adjoint):
            assert not work and not slots
            wires = (predicate, *ports["system"])
            if named:
                label = type(gate).__name__
                if label == "TGate":
                    instruction = Instruction(
                        "cphase",
                        wires,
                        params={"theta": -pi / 4 if adjoint else pi / 4},
                    )
                else:
                    instruction = Instruction(
                        {"Hadamard": "ch", "XGate": "cx"}[label], wires
                    )
            else:
                instruction = Instruction(
                    "any",
                    wires,
                    matrix=positive.mH.contiguous() if adjoint else positive,
                )
            return FieldOperation(max(wires) + 1, (instruction,))

        return make_on

    native = LinearCombinationBlockEncoding(
        tuple(
            UnitaryBlockEncoding((("q", width),), lambda p, w, b, a: (), alpha=alpha)
            for alpha in (scales or (1.0,) * len(gates))
        ),
        coefficients,
        precision,
    )
    ancilla_end = width + native.ancilla_bitsize
    through = ancilla_end + native.resource_bitsize
    prep, prep_bits = native.coefficient_preparation.workspace_requirements
    select = indexed_component_workspace((native.select.selection_bitsize,), 0)
    work = tuple(range(through, through + max(prep, select)))
    op = native.make_on(
        tuple(
            child(gate, matrix) for gate, matrix in zip(gates, matrices, strict=True)
        ),
        system=tuple(range(width)),
        ancilla=tuple(range(width, ancilla_end)),
        resource=tuple(range(ancilla_end, through)),
        prepare_workspace=work[:prep],
        prepare_classical_bits=tuple(range(prep_bits)),
        traversal_workspace=work[:select],
        traversal_classical_bits=tuple(range(select)),
        workspaces=((),) * len(gates),
        classical_bits=((),) * len(gates),
        adjoint=inverse,
    )
    circuit = DynamicCircuit(op.required_qubits, dtype=torch.complex128)
    circuit.append(op)
    return native, matrices, circuit, work


@pytest.mark.parametrize(
    "family,row,coefficients,precision,tolerance",
    [
        pytest.param(
            family, index, tuple(row), precision, tolerance, id=f"{family}-{index}"
        )
        for family, rows, precision, tolerance in _COEFFICIENT_CASES
        for index, row in enumerate(rows)
    ]
    + [pytest.param("dense", i, (), 9, 0.02, id=f"dense-{i}") for i in range(10)],
)
def test_original_coefficient_grid_projection_and_work(
    family, row, coefficients, precision, tolerance, coefficient_gates
):
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary

    gates, dense = coefficient_gates
    if family == "dense":
        gates, coefficients = dense[row]
    else:
        gates = gates[family]
    scales = (2.0, 4.0) if family == "normalization" else (1.0,) * len(gates)
    source = LinearCombination(
        tuple(
            Unitary(gate, alpha=alpha)
            for gate, alpha in zip(gates, scales, strict=True)
        ),
        coefficients,
        precision,
    )
    target = torch.as_tensor(
        _source_coefficient_projection(source), dtype=torch.complex128
    )
    full_directional = (
        tuple(
            torch.as_tensor(block.tensor_contract(), dtype=torch.complex128)
            for block in (source, source.adjoint())
        )
        if family in ("hermitian", "nonhermitian")
        else ()
    )
    if full_directional:
        equal = torch.allclose(*full_directional, rtol=1e-7, atol=1e-16)
        assert equal == (family == "hermitian")
    for inverse in (False, True) if full_directional else (False,):
        native, matrices, circuit, work = _bind_coefficient_fixture(
            gates, coefficients, precision, inverse=inverse, scales=scales
        )
        assert native.alpha == source.alpha
        assert native.ancilla_bitsize == source.ancilla_bitsize
        assert native.resource_bitsize == source.resource_bitsize
        ideal = sum(
            value * alpha * matrix
            for value, alpha, matrix in zip(coefficients, scales, matrices, strict=True)
        )
        torch.testing.assert_close(target, ideal, rtol=0, atol=tolerance)
        expected = target.mH if inverse else target
        ir = circuit.to_ir()
        instructions = tuple(ir.instructions)
        measurements = sum(gate.name == "measure" for gate in instructions)
        histories = (
            tuple(product((0, 1), repeat=measurements))
            if measurements <= 4
            else ((0,) * measurements, tuple(i % 2 for i in range(measurements)))
        )
        dim = target.shape[0]
        auxiliary_dim = (1 << circuit.n_qubits) // dim
        work_dim = 1 << len(work)
        if full_directional:
            matrix = full_directional[int(inverse)]
            index = torch.arange(matrix.shape[0], dtype=torch.float64)
            coherent = torch.complex(index % 7 - 3, index % 11 - 5)
            coherent /= coherent.norm()
            clean = torch.eye(1, work_dim, dtype=torch.complex128).flatten()
            for history in histories:
                torch.testing.assert_close(
                    forced_branch(torch.kron(coherent, clean), instructions, history),
                    torch.kron(matrix @ coherent, clean) / sqrt(2**measurements),
                    rtol=0,
                    atol=3e-12,
                )
        for column in range(dim):
            if precision == 9:
                for seed in (19, 73):
                    output = execute_sparse_register(
                        ir, {column * auxiliary_dim: 1 + 0j}, seed=seed
                    )
                    observed = torch.tensor(
                        [output.get(row * auxiliary_dim, 0j) for row in range(dim)],
                        dtype=torch.complex128,
                    )
                    torch.testing.assert_close(
                        observed * native.alpha,
                        expected[:, column],
                        rtol=0,
                        atol=2e-11,
                    )
                    assert all(
                        abs(amplitude) < 2e-12
                        for basis, amplitude in output.items()
                        if basis % work_dim
                    )
                continue
            state = torch.zeros((dim, auxiliary_dim), dtype=torch.complex128)
            state[column, 0] = 1
            for history in histories:
                realized = forced_branch(state.flatten(), instructions, history) * sqrt(
                    2**measurements
                )
                torch.testing.assert_close(
                    realized.reshape(dim, auxiliary_dim)[:, 0] * native.alpha,
                    expected[:, column],
                    rtol=0,
                    atol=2e-11,
                )
                assert torch.max(realized.reshape(-1, work_dim)[:, 1:].abs()) < 2e-12


@pytest.mark.parametrize(
    "family,coefficients,precision",
    [
        ("two", tuple(EXACT2[0]), 1),
        ("three", tuple(EXACT3[1]), 1),
        ("approx-two", tuple(APPROX2[4]), 9),
    ],
)
@pytest.mark.parametrize("inverse", (False, True))
def test_coefficient_cases_named_primitive_flow_and_resource_profiles(
    family, coefficients, precision, inverse, coefficient_gates
):
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    gates = coefficient_gates[0][family]
    encoding, _, circuit, _ = _bind_coefficient_fixture(
        gates, coefficients, precision, inverse=inverse, named=True
    )
    source = LinearCombination(
        tuple(Unitary(gate) for gate in gates), coefficients, precision
    )
    if inverse:
        source = source.adjoint()
    _assert_source_flow(source, encoding, circuit)
    children = (
        ComponentCost("Controlled T", 2, GateCost(rotation=1)),
        ComponentCost("Controlled H", 2, GateCost(clifford=4, rotation=2)),
        ComponentCost("CNOT", 2, GateCost(clifford=1)),
    )[: len(gates)]
    for legacy in (False, True):
        prepare = encoding.coefficient_preparation.resource_model(legacy_shims=legacy)
        select = encoding.select.resource_model(
            children, adjoint=inverse, legacy_shims=legacy
        )
        report = estimate_component_resources(
            encoding.resource_model(
                select, prepare, prepare, adjoint=inverse, legacy_shims=legacy
            )
        )
        expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        assert asdict(report.gates) == {
            name: getattr(expected, name) for name in asdict(report.gates)
        }
        assert report.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize(
    "family,row,coefficients,precision",
    [
        (family, index, tuple(row), precision)
        for family, rows, precision, _ in _COEFFICIENT_CASES
        for index, row in enumerate(rows)
    ]
    + [("dense", row, (), 9) for row in range(10)],
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_complete_catalogue_erasure_history_certificate(
    family: str,
    row: int,
    coefficients: tuple[float, ...],
    precision: int,
    adjoint: bool,
    coefficient_gates: Any,
) -> None:
    from tests.unit.measurement_certificate import certify_erasure_histories

    named, dense = coefficient_gates
    if family == "dense":
        gates, coefficients = dense[row]
    else:
        gates = named[family]
    scales = (2.0, 4.0) if family == "normalization" else ()
    _, _, circuit, work = _bind_coefficient_fixture(
        gates, coefficients, precision, inverse=adjoint, scales=scales
    )
    instructions = circuit.to_ir().instructions
    certified = certify_erasure_histories(
        instructions, n_wires=circuit.n_qubits, zero_wires=work
    )
    assert certified == sum(gate.name == "measure" for gate in instructions)
