"""Verify the locked coefficient transform and composed signed SELECT execution."""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict
from itertools import product
from math import sqrt

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.chebyshev_block_encoding import (
    ChebyshevBlockEncoding,
)
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
)
from flagquantum.algorithms.primitives.scaled_chebyshev_block_encoding import (
    ScaledChebyshevBlockEncoding,
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
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_workspace,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _unused_child(
    ports: Mapping[str, tuple[int, ...]],
    work: tuple[int, ...],
    slots: tuple[int, ...],
    inverse: bool,
) -> tuple[Instruction, ...]:
    raise AssertionError("The supplied controlled factory owns this child")


@pytest.mark.parametrize(
    "order,alpha,child_ancilla",
    [(0, 2.0, 0), (1, 2.0, 0), (2, 0.5, 1), (4, 2.0, 1), (5, 3.14, 0)],
)
def test_locked_scaled_coefficients_packed_widths_and_signal(
    order: int,
    alpha: float,
    child_ancilla: int,
    request: pytest.FixtureRequest,
) -> None:
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.block_encoding import ScaledChebyshevPolynomial, Unitary

    request.config.addinivalue_line("markers", "notebook: upstream notebook execution")
    from qualtran.bloqs.block_encoding.chebyshev_polynomial_test import (
        _scaled_chebyshev_poly_even,
        _scaled_chebyshev_poly_odd,
    )

    if order == 4:
        source = _scaled_chebyshev_poly_even()
    elif order == 5:
        source = _scaled_chebyshev_poly_odd()
    else:
        source = ScaledChebyshevPolynomial(
            Unitary(Hadamard(), alpha=alpha, ancilla_bitsize=child_ancilla), order
        )
    native = ScaledChebyshevBlockEncoding(
        UnitaryBlockEncoding(
            (("q", 1),),
            _unused_child,
            alpha=alpha,
            ancilla_bitsize=child_ancilla,
            resource_bitsize=source.block_encoding.resource_bitsize,
        ),
        order,
    )
    assert (native.alpha, native.epsilon) == (source.alpha, source.epsilon)
    assert (native.system_bitsize, native.ancilla_bitsize, native.resource_bitsize) == (
        source.system_bitsize,
        source.ancilla_bitsize,
        source.resource_bitsize,
    )
    assert native.signal_state.selection_width == native.ancilla_bitsize
    assert [(r.name, r.bitsize) for r in native.signature] == [
        (r.name, r.bitsize) for r in source.signature
    ]
    for actual, expected in (
        (native.target_registers, source.target_registers),
        (native.selection_registers, source.selection_registers),
        (native.junk_registers, source.junk_registers),
    ):
        assert [(r.name, r.bitsize) for r in actual] == [
            (r.name, r.bitsize) for r in expected
        ]
    term = native.linear_combination
    if order <= 1:
        assert isinstance(term, ChebyshevBlockEncoding)
        assert term.order == order and term.alpha == 1
    else:
        assert isinstance(term, LinearCombinationBlockEncoding)
        source_lcu = source.linear_combination
        assert term.coefficients == tuple(source_lcu._lambd)
        assert len(term.block_encodings) == order + 1
        for index, block in enumerate(term.block_encodings):
            assert isinstance(block, ChebyshevBlockEncoding) and block.order == index


@pytest.mark.parametrize("adjoint", [False, True])
def test_scaled_signed_select_complete_complex_branches(adjoint: bool) -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.block_encoding import ScaledChebyshevPolynomial, Unitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = ScaledChebyshevPolynomial(
        Unitary(XGate(), alpha=0.5, ancilla_bitsize=1), 2, lambd_bits=1
    )
    scaled = ScaledChebyshevBlockEncoding(
        UnitaryBlockEncoding((("q", 1),), _unused_child, alpha=0.5, ancilla_bitsize=1),
        2,
        1,
    )
    lcu = scaled.linear_combination
    assert isinstance(lcu, LinearCombinationBlockEncoding)
    ancilla = tuple(range(1, 1 + scaled.ancilla_bitsize))
    resource = tuple(
        range(1 + len(ancilla), 1 + len(ancilla) + scaled.resource_bitsize)
    )
    logical = 1 + len(ancilla) + len(resource)
    prepare_count, prepare_slots = lcu.coefficient_preparation.workspace_requirements
    traversal_count = indexed_component_workspace((lcu.select.selection_bitsize,), 0)
    work = tuple(range(logical, logical + max(prepare_count, traversal_count)))
    slots = tuple(range(10, 10 + len(work)))
    reserved = logical + len(work)

    def term_factory(
        term: ChebyshevBlockEncoding,
    ) -> Callable[
        [int, Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
        _CircuitOperation,
    ]:
        def bind(
            predicate: int,
            ports: Mapping[str, tuple[int, ...]],
            child_work: tuple[int, ...],
            child_slots: tuple[int, ...],
            inverse: bool,
        ) -> _CircuitOperation:
            def controlled_child(
                child: Mapping[str, tuple[int, ...]],
                private: tuple[int, ...],
                result_slots: tuple[int, ...],
                inverse_child: bool,
            ) -> _CircuitOperation:
                return FieldOperation(
                    max(predicate, *child["system"]) + 1,
                    (Instruction("cx", (predicate, *child["system"])),),
                )

            return term.make_on(
                controlled_child,
                system=ports["system"],
                ancilla=ports.get("ancilla", ()),
                resource=ports.get("resource", ()),
                control=predicate,
                workspace=child_work,
                classical_bits=child_slots,
                adjoint=inverse,
            )

        return bind

    terms = tuple(
        term for term in lcu.block_encodings if isinstance(term, ChebyshevBlockEncoding)
    )
    assert len(terms) == 3
    operation = lcu.make_on(
        tuple(term_factory(term) for term in terms),
        system=(0,),
        ancilla=ancilla,
        resource=resource,
        prepare_workspace=work[:prepare_count],
        prepare_classical_bits=slots[:prepare_slots],
        traversal_workspace=work[:traversal_count],
        traversal_classical_bits=slots[:traversal_count],
        workspaces=((),) * len(terms),
        classical_bits=((),) * len(terms),
        adjoint=adjoint,
    )
    instructions = DynamicCircuit(reserved).append(operation).to_ir().instructions
    matrix = torch.as_tensor(
        (source.adjoint() if adjoint else source).tensor_contract(),
        dtype=torch.complex128,
    )
    values = torch.arange(1 << logical, dtype=torch.float64)
    psi = torch.complex(values + 1, values % 11 - 4)
    psi /= psi.norm()
    zero_work = torch.zeros(1 << len(work), dtype=torch.complex128)
    zero_work[0] = 1
    initial = torch.kron(psi, zero_work)
    expected = torch.kron(matrix @ psi, zero_work)
    counts = Counter(gate.name for gate in instructions)
    child_model = ComponentCost("Controlled X", 3, GateCost(clifford=1))
    term_costs = tuple(
        term.resource_model(
            child_model,
            child_model,
            controlled=True,
            adjoint=adjoint,
            reflection_model="call_graph",
        )
        for term in terms
    )
    select_cost = lcu.select.resource_model(term_costs, adjoint=adjoint)
    prepare_cost = lcu.coefficient_preparation.resource_model()
    model = estimate_component_resources(
        lcu.resource_model(select_cost, prepare_cost, prepare_cost, adjoint=adjoint)
    )
    locked = source.adjoint() if adjoint else source
    source_cost = get_cost_value(locked, QECGatesCost())
    assert asdict(model.gates) == {
        field: getattr(source_cost, field) for field in asdict(model.gates)
    }
    assert model.peak_logical_qubits == get_cost_value(locked, QubitCount())
    expanded_terms = tuple(
        term.resource_model(
            child_model,
            child_model,
            controlled=True,
            adjoint=adjoint,
            reflection_model="decomposition",
        )
        for term in terms
    )
    expanded = estimate_component_resources(
        lcu.resource_model(
            lcu.select.resource_model(expanded_terms, adjoint=adjoint),
            prepare_cost,
            prepare_cost,
            adjoint=adjoint,
        )
    )
    assert expanded.gates.clifford == model.gates.clifford + 4
    assert (
        counts["t"] + counts["tdg"] == 4 * model.gates.and_bloq + 7 * model.gates.cswap
    )
    assert counts["measure"] == model.gates.measurement == 5
    for history in product((0, 1), repeat=counts["measure"]):
        actual = forced_branch(initial, instructions, history)
        torch.testing.assert_close(
            actual, expected / sqrt(2 ** len(history)), atol=4e-12, rtol=0
        )


def _lcu_scratch(block: LinearCombinationBlockEncoding, controlled: bool) -> int:
    prepare, _ = block.coefficient_preparation.workspace_requirements
    select = indexed_component_workspace(
        (block.select.selection_bitsize,), int(controlled)
    )
    return max(prepare, select)


def _bind_lcu(
    block: LinearCombinationBlockEncoding,
    factories: tuple[
        Callable[
            [
                int,
                Mapping[str, tuple[int, ...]],
                tuple[int, ...],
                tuple[int, ...],
                bool,
            ],
            _CircuitOperation,
        ],
        ...,
    ],
    ports: Mapping[str, tuple[int, ...]],
    work: tuple[int, ...],
    slots: tuple[int, ...],
    *,
    control: tuple[int, ...] = (),
    child_scratch: int = 0,
    adjoint: bool = False,
) -> _CircuitOperation:
    prepare, nc = block.coefficient_preparation.workspace_requirements
    traverse = indexed_component_workspace(
        (block.select.selection_bitsize,), len(control)
    )
    stage = max(prepare, traverse)
    slot_stage = max(nc, traverse)
    return block.make_on(
        factories,
        system=ports["system"],
        ancilla=ports.get("ancilla", ()),
        resource=ports.get("resource", ()),
        control=control,
        prepare_workspace=work[:prepare],
        prepare_classical_bits=slots[:nc],
        traversal_workspace=work[:traverse],
        traversal_classical_bits=slots[:traverse],
        workspaces=(work[stage : stage + child_scratch],) * len(factories),
        classical_bits=(slots[slot_stage : slot_stage + child_scratch],)
        * len(factories),
        adjoint=adjoint,
    )


def _controlled_leaf(
    name: str,
) -> Callable[
    [int, Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
    _CircuitOperation,
]:
    def bind(
        predicate: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> _CircuitOperation:
        wires = (predicate, *ports["system"])
        return FieldOperation(max(wires) + 1, (Instruction(name, wires),))

    return bind


def original_scaled_operation(
    order: int, *, adjoint: bool = False
) -> tuple[ScaledChebyshevBlockEncoding, _CircuitOperation]:
    """Bind the complete original scaled fixture through native child owners."""
    hadamard = UnitaryBlockEncoding(
        (("q", 1),), _unused_child, alpha=3.14 if order == 5 else 1
    )
    inner = LinearCombinationBlockEncoding(
        (UnitaryBlockEncoding((("q", 1),), _unused_child), hadamard), (1.0, 1.0), 1
    )
    leaf_factories = (_controlled_leaf("cx"), _controlled_leaf("ch"))
    scaled = ScaledChebyshevBlockEncoding(inner if order == 4 else hadamard, order)
    outer = scaled.linear_combination
    assert isinstance(outer, LinearCombinationBlockEncoding)
    logical = scaled.system_bitsize + scaled.ancilla_bitsize + scaled.resource_bitsize
    outer_scratch = _lcu_scratch(outer, False)
    child_scratch = _lcu_scratch(inner, True) if order == 4 else 0
    size = logical + outer_scratch + child_scratch

    def bind_term(
        term: ChebyshevBlockEncoding,
    ) -> Callable[
        [int, Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
        _CircuitOperation,
    ]:
        def bind(
            predicate: int,
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            slots: tuple[int, ...],
            adjoint: bool,
        ) -> _CircuitOperation:
            def bind_child(
                child_ports: Mapping[str, tuple[int, ...]],
                child_work: tuple[int, ...],
                child_slots: tuple[int, ...],
                inverse: bool,
            ) -> _CircuitOperation:
                if order == 4:
                    return _bind_lcu(
                        inner,
                        leaf_factories,
                        child_ports,
                        child_work,
                        child_slots,
                        control=(predicate,),
                        adjoint=inverse,
                    )
                return leaf_factories[1](
                    predicate, child_ports, child_work, child_slots, inverse
                )

            return term.make_on(
                bind_child,
                system=ports["system"],
                ancilla=ports.get("ancilla", ()),
                resource=ports.get("resource", ()),
                control=predicate,
                workspace=work,
                classical_bits=slots,
                adjoint=adjoint,
            )

        return bind

    terms = tuple(
        term
        for term in outer.block_encodings
        if isinstance(term, ChebyshevBlockEncoding)
    )
    assert len(terms) == order + 1
    ancilla_end = 1 + scaled.ancilla_bitsize
    operation = _bind_lcu(
        outer,
        tuple(bind_term(term) for term in terms),
        {
            "system": (0,),
            "ancilla": tuple(range(1, ancilla_end)),
            "resource": tuple(range(ancilla_end, logical)),
        },
        tuple(range(logical, size)),
        tuple(range(size - logical)),
        child_scratch=child_scratch,
        adjoint=adjoint,
    )
    return scaled, operation


@pytest.mark.parametrize("order", [4, 5])
@pytest.mark.parametrize("adjoint", (False, True))
def test_original_large_scaled_execution(
    order: int, request: pytest.FixtureRequest, adjoint: bool
) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import IntState
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        _scaled_chebyshev_poly_even,
        _scaled_chebyshev_poly_odd,
    )

    from tests.unit.test_sparse_chemistry_prepare import _branch

    request.config.addinivalue_line("markers", "notebook: upstream notebook execution")
    from qualtran.bloqs.block_encoding.chebyshev_polynomial_test import gate_test

    base_source = (
        _scaled_chebyshev_poly_even() if order == 4 else _scaled_chebyshev_poly_odd()
    )
    source = base_source.adjoint() if adjoint else base_source
    scaled, operation = original_scaled_operation(order, adjoint=adjoint)
    size = operation.required_qubits
    logical = scaled.system_bitsize + scaled.ancilla_bitsize + scaled.resource_bitsize
    gates = DynamicCircuit(size).append(operation).to_ir().instructions
    count = Counter(gate.name for gate in gates)
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    # Report the source graph independently of the emitted primitive stream.
    source_cost = get_cost_value(source, QECGatesCost())
    source_peak = get_cost_value(source, QubitCount())
    assert source_peak == (37 if order == 4 else 32)
    assert source_cost.measurement == (61 if order == 4 else 34)
    assert count["measure"] == source_cost.measurement
    assert count["t"] + count["tdg"] == 4 * source_cost.and_bloq + 7 * source_cost.cswap
    assert size == (39 if order == 4 else 32)
    history = tuple(i % 2 for i in range(count["measure"]))
    source_matrix = gate_test(base_source)
    if adjoint:
        source_matrix = source_matrix.conj().T
    observed = torch.zeros((2, 2), dtype=torch.complex128)
    target_mask = 1 << (size - 1)
    scratch_mask = (1 << (size - logical)) - 1
    for bit in (0, 1):
        initial = {bit * target_mask: 1 + 0j}
        all_zero = _branch(
            gates, size, (0,) * len(history), initial, normalize_measurements=True
        )
        alternating = _branch(
            gates, size, history, initial, normalize_measurements=True
        )
        assert all_zero == pytest.approx(alternating, abs=3e-10)
        assert sum(abs(value) ** 2 for value in all_zero.values()) == pytest.approx(1)
        assert all(index & scratch_mask == 0 for index in all_zero)
        builder = BloqBuilder()
        source_inputs = {
            reg.name: builder.add(
                IntState(bit if reg.name == "system" else 0, reg.bitsize)
            )
            for reg in source.signature
        }
        source_outputs = builder.add_d(source, **source_inputs)
        reference = builder.finalize(**source_outputs).tensor_contract().reshape(-1)
        indices = np.flatnonzero(np.abs(reference) > 1e-12)
        native_by_logical_index = {
            index >> (size - logical): amp for index, amp in all_zero.items()
        }
        assert native_by_logical_index == pytest.approx(
            dict(zip(indices.tolist(), reference[indices].tolist(), strict=True)),
            abs=3e-10,
        )
        observed[:, bit] = torch.tensor(
            [all_zero.get(0, 0j), all_zero.get(target_mask, 0j)], dtype=torch.complex128
        )
    torch.testing.assert_close(
        observed * float(scaled.alpha),
        torch.as_tensor(source_matrix, dtype=torch.complex128),
        rtol=0,
        atol=5e-10,
    )


@pytest.mark.parametrize("order", [4, 5])
def test_original_unscaled_child_layout_and_full_outputs(order: int) -> None:
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        _chebyshev_poly_even,
        _chebyshev_poly_odd,
    )

    from tests.unit.test_sparse_chemistry_prepare import _branch

    hadamard = UnitaryBlockEncoding((("q", 1),), _unused_child)
    inner = LinearCombinationBlockEncoding(
        (UnitaryBlockEncoding((("q", 1),), _unused_child), hadamard), (0.5, 0.5), 1
    )
    source = _chebyshev_poly_even() if order == 4 else _chebyshev_poly_odd()
    root = ChebyshevBlockEncoding(inner if order == 4 else hadamard, order)
    ancilla_end = 1 + root.ancilla_bitsize
    logical = ancilla_end + root.resource_bitsize
    scratch = _lcu_scratch(inner, False) if order == 4 else 0
    reflect, reflect_slots = root.reflection_requirements()
    size = logical + max(scratch, reflect)
    work = tuple(range(logical, size))
    slots = tuple(range(len(work)))

    def bind_child(
        ports: Mapping[str, tuple[int, ...]],
        child_work: tuple[int, ...],
        child_slots: tuple[int, ...],
        inverse: bool,
    ) -> _CircuitOperation:
        if order == 4:
            return _bind_lcu(
                inner,
                (_controlled_leaf("cx"), _controlled_leaf("ch")),
                ports,
                child_work,
                child_slots,
                adjoint=inverse,
            )
        return FieldOperation(1, (Instruction("h", ports["system"]),))

    operation = root.make_on(
        bind_child,
        system=(0,),
        ancilla=tuple(range(1, ancilla_end)),
        resource=tuple(range(ancilla_end, logical)),
        workspace=work[:scratch],
        classical_bits=slots[:scratch],
        reflection_workspace=work[:reflect],
        reflection_classical_bits=slots[:reflect_slots],
    )
    gates = DynamicCircuit(size).append(operation).to_ir().instructions
    count = Counter(gate.name for gate in gates)
    expected = source.tensor_contract()
    for bit in (0, 1):
        col = bit << (logical - 1)
        expected_sparse = {
            row << len(work): complex(value)
            for row, value in enumerate(expected[:, col])
            if abs(value) > 1e-12
        }
        for history in product((0, 1), repeat=count["measure"]):
            actual = _branch(
                gates,
                size,
                history,
                {col << len(work): 1 + 0j},
                normalize_measurements=True,
            )
            assert actual == pytest.approx(expected_sparse, abs=3e-12)


@pytest.mark.parametrize("order", (4, 5))
@pytest.mark.parametrize("adjoint", (False, True))
def test_original_scaled_every_measurement_history_and_private_work(
    order: int, adjoint: bool
) -> None:
    from flagquantum import Circuit
    from flagquantum.core.ir import CircuitIR
    from tests.unit.measurement_certificate import certify_erasure_histories

    scaled, operation = original_scaled_operation(order, adjoint=adjoint)
    logical = scaled.system_bitsize + scaled.ancilla_bitsize + scaled.resource_bitsize
    ir = Circuit(operation.required_qubits).append(operation).to_ir()
    work = tuple(range(logical, ir.n_wires))
    assert len(work) == (16 if order == 4 else 14)
    for owned in (ir, CircuitIR.from_json(ir.to_json())):
        count = certify_erasure_histories(
            owned.instructions, n_wires=owned.n_wires, zero_wires=work
        )
        assert count == (61 if order == 4 else 34)
        assert count == sum(g.name == "measure" for g in owned.instructions)
