"""Compare the frozen Chebyshev call order, phase and resource routes."""

from collections import Counter
from collections.abc import Mapping
from itertools import product

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.algorithms.primitives.chebyshev_block_encoding import (
    ChebyshevBlockEncoding,
)
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.circuit import _CircuitOperation
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
)
from flagquantum.core.ir import Instruction
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "order,ancilla_width", [(0, 1), (1, 1), (4, 0), (5, 1), (2, 3)]
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_source_chebyshev_stages_and_complex_histories(
    order: int, ancilla_width: int, adjoint: bool
) -> None:
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.block_encoding import ChebyshevPolynomial, Unitary
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    source_child = Unitary(TGate(), ancilla_bitsize=ancilla_width, resource_bitsize=1)
    source = ChebyshevPolynomial(source_child, order)
    source_composite = (source.adjoint() if adjoint else source).decompose_bloq()

    def emit(
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        inverse: bool,
    ) -> tuple[Instruction, ...]:
        return (Instruction("tdg" if inverse else "t", ports["q"]),)

    child = UnitaryBlockEncoding(
        (("q", 1),), emit, ancilla_bitsize=ancilla_width, resource_bitsize=1
    )
    root = ChebyshevBlockEncoding(child, order)
    assert (root.alpha, root.epsilon) == (source.alpha, source.epsilon)
    assert [(r.name, r.bitsize) for r in root.signature] == [
        (r.name, r.bitsize) for r in source.signature
    ]
    assert root.signature.thru_registers_only
    assert root.signal_state.selection_width == ancilla_width
    assert root.signal_state.emit(tuple(range(ancilla_width)), (), (), ()) == ()
    through = ancilla_width + 2
    ancilla = tuple(range(1, 1 + ancilla_width))
    num_work, num_slots = root.reflection_requirements()
    work = tuple(range(through, through + num_work))
    slots = tuple(reversed(range(4, 4 + num_slots)))

    def bind(
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        inverse: bool,
    ) -> _CircuitOperation:
        return child.make_on(
            **ports, workspace=workspace, classical_bits=classical_bits, adjoint=inverse
        )

    operation = root.make_on(
        bind,
        system=(0,),
        ancilla=ancilla,
        resource=(through - 1,),
        reflection_workspace=work,
        reflection_classical_bits=slots,
        adjoint=adjoint,
    )
    reference: list[Instruction] = []
    reflection_count = 0
    for instance, _, _ in source_composite.iter_bloqsoqs():
        bloq = instance.bloq
        if isinstance(bloq, ReflectionUsingPrepare):
            reflection_count += 1
            reference.extend(
                prepare_reflection_ir(ancilla, work, slots, global_phase=-1)
            )
        else:
            assert bloq in (source_child, source_child.adjoint())
            reference.append(Instruction("t" if bloq == source_child else "tdg", (0,)))
    assert tuple(reference) == operation.instructions
    _, declared = source.call_graph(max_depth=1)
    declared_reflections = declared.get(source.reflection_bloq, 0) if ancilla else 0
    child_cost = ComponentCost("TBlock", through, GateCost(t=1))
    direct_model = root.resource_model(child_cost, child_cost, adjoint=adjoint)
    expanded_model = root.resource_model(
        child_cost, child_cost, adjoint=adjoint, reflection_model="decomposition"
    )
    assert (
        sum(
            call.repetitions
            for call in direct_model.calls
            if call.component.name == "ReflectionUsingPrepare"
        )
        == declared_reflections
    )
    assert (
        sum(
            call.repetitions
            for call in expanded_model.calls
            if call.component.name == "ReflectionUsingPrepare"
        )
        == reflection_count
    )
    direct, expanded = map(estimate_component_resources, (direct_model, expanded_model))
    assert direct.gates.measurement == declared_reflections * num_slots
    assert expanded.gates.measurement == reflection_count * num_slots
    assert expanded.gates.t == order
    count = Counter(gate.name for gate in operation.instructions)
    assert count["t"] + count["tdg"] == order + 4 * count["measure"]
    assert count["measure"] == expanded.gates.measurement
    assert expanded.peak_logical_qubits == through + num_work

    matrix = torch.as_tensor(source_composite.tensor_contract(), dtype=torch.complex128)
    data = torch.arange(1 << through, dtype=torch.float64)
    logical = torch.complex(data + 1, data % 5 - 2)
    logical /= logical.norm()
    zeros = torch.zeros(1 << num_work, dtype=torch.complex128)
    zeros[0] = 1
    initial = torch.kron(logical, zeros)
    expected = torch.kron(matrix @ logical, zeros)
    circuit = Circuit(through + num_work, dtype=torch.complex128).append(operation)
    assert circuit.to_ir().instructions == operation.instructions
    for history in product((0, 1), repeat=count["measure"]):
        branch = forced_branch(initial, operation.instructions, history)
        torch.testing.assert_close(
            branch, expected * 2 ** (-len(history) / 2), atol=2e-12, rtol=0
        )


@pytest.mark.parametrize("adjoint", [False, True])
def test_positive_outer_control_preserves_source_action(adjoint: bool) -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.block_encoding import ChebyshevPolynomial, Unitary

    source = ChebyshevPolynomial(Unitary(XGate(), ancilla_bitsize=1), 3)
    controlled = source.controlled()
    if adjoint:
        controlled = controlled.adjoint()
    expected = torch.as_tensor(controlled.tensor_contract(), dtype=torch.complex128)

    native = ChebyshevBlockEncoding(
        UnitaryBlockEncoding(
            (("q", 1),),
            lambda ports, work, bits, inverse: (Instruction("x", ports["q"]),),
            ancilla_bitsize=1,
        ),
        3,
    )

    def bind(
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> FieldOperation:
        return FieldOperation(3, (Instruction("cx", (0, *ports["system"])),))

    assert native.reflection_requirements(controlled=True) == (0, 0)
    operation = native.make_on(
        bind, control=0, system=(1,), ancilla=(2,), adjoint=adjoint
    )
    circuit = Circuit(3, dtype=torch.complex128).append(operation)
    torch.testing.assert_close(get_unitary(circuit), expected, atol=2e-12, rtol=0)


_RANDOM_CASES = tuple(range(20))


@pytest.mark.parametrize("seed_index", [None, *_RANDOM_CASES])
def test_original_order_four_matrix_projection(
    request: pytest.FixtureRequest, seed_index: int | None
) -> None:
    request.config.addinivalue_line("markers", "notebook: upstream notebook execution")

    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding import ChebyshevPolynomial
    from qualtran.bloqs.block_encoding.chebyshev_polynomial_test import (
        TestBlockEncoding,
        gate_test,
    )

    if seed_index is None:
        operator = (XGate().tensor_contract() + 3 * Hadamard().tensor_contract()) / 4
    else:
        from qualtran.linalg.matrix import random_hermitian_matrix

        rng = np.random.RandomState(1234)
        for _ in range(seed_index + 1):
            operator = random_hermitian_matrix(2, rng)
    source_child = TestBlockEncoding.from_matrix(operator)
    source = ChebyshevPolynomial(source_child, 4)
    expected = torch.as_tensor(gate_test(source), dtype=torch.complex128)
    expected_full = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    child_matrix = torch.as_tensor(
        source_child.tensor_contract(), dtype=torch.complex128
    )
    child = UnitaryBlockEncoding(
        (("q", 1),),
        lambda ports, work, bits, inverse: (),
        ancilla_bitsize=1,
        resource_bitsize=1,
    )

    def bind(
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> FieldOperation:
        wires = (*ports["system"], *ports["ancilla"], *ports["resource"])
        matrix = child_matrix.mH if inverse else child_matrix
        return FieldOperation(
            4, (Instruction("encoded_operator", wires, matrix=matrix),)
        )

    root = ChebyshevBlockEncoding(child, 4)
    operation = root.make_on(
        bind, system=(0,), ancilla=(1,), resource=(2,), reflection_workspace=(3,)
    )
    matrix = get_unitary(Circuit(4, dtype=torch.complex128).append(operation))
    selected = torch.tensor([0, 8])
    torch.testing.assert_close(
        matrix[selected][:, selected], expected, atol=3e-8, rtol=0
    )
    torch.testing.assert_close(matrix[::2, ::2], expected_full, atol=3e-8, rtol=0)
    assert torch.count_nonzero(matrix[1::2, ::2]) == 0
