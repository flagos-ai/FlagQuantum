"""Verify the locked coefficient transform and composed signed SELECT execution."""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict
from itertools import product
from math import sqrt

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

    def term_factory(term: ChebyshevBlockEncoding) -> Callable[
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
