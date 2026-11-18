"""Resolve source LCU declarations before binding physical append operations."""

from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
    source_linear_combination_signature,
)
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_workspace,
)
from flagquantum.ecosystem.cirq._block_encoding import import_block_encoding
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_foreign_block_encoding import original_models

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("port", ("system", "ancilla", "resource", "all"))
@pytest.mark.parametrize("controlled,adjoint", tuple(product((False, True), repeat=2)))
def test_resolved_symbolic_ports_use_original_lcu_and_snapshot_calls(
    port: str,
    controlled: bool,
    adjoint: bool,
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.basic_gates import Identity
    from qualtran.bloqs.block_encoding import LinearCombination, Unitary
    from sympy import Symbol, sympify

    s, a, r = (Symbol(name) for name in ("s", "a", "r"))
    width = s if port in ("system", "all") else 1
    ancilla = a if port in ("ancilla", "all") else 0
    resource = r if port in ("resource", "all") else 0
    declaration = LinearCombination(
        (
            Unitary(Identity(width), ancilla_bitsize=ancilla),
            Unitary(Identity(width), resource_bitsize=resource),
        ),
        (1.0, -2.0),
        1,
        is_controlled=controlled,
    )
    with pytest.raises(DecomposeTypeError, match="Cannot decompose symbolic"):
        declaration.decompose_bloq()
    substitutions = {s: 1, a: 1, r: 1}
    children = tuple(
        Unitary(
            Identity(int(sympify(child.system_bitsize).subs(substitutions))),
            ancilla_bitsize=int(sympify(child.ancilla_bitsize).subs(substitutions)),
            resource_bitsize=int(sympify(child.resource_bitsize).subs(substitutions)),
        )
        for child in declaration._block_encodings
    )
    original = LinearCombination(
        children, declaration._lambd, 1, is_controlled=controlled
    )
    source = original.adjoint() if adjoint else original
    blocks = tuple(
        import_block_encoding(child, models=original_models(child))
        for child in children
    )
    native = LinearCombinationBlockEncoding(blocks, (1.0, -2.0), 1)
    signature = source_linear_combination_signature(
        native.system_bitsize,
        tuple(block.ancilla_bitsize for block in blocks),
        tuple(block.resource_bitsize for block in blocks),
        prepare_selection=native.prepare.selection_width,
        prepare_junk=native.prepare.junk_width,
        controlled=controlled,
    )
    assert [(reg.name, reg.bitsize) for reg in signature] == [
        (reg.name, int(sympify(reg.bitsize).subs(substitutions)))
        for reg in declaration.signature
    ]
    prep = native.coefficient_preparation.resource_model()
    select = native.select.resource_model(
        tuple(
            block.resource_model(controlled=True, adjoint=adjoint) for block in blocks
        ),
        controlled=controlled,
        adjoint=adjoint,
    )
    assert_source_model(
        native.resource_model(
            select, prep, prep, controlled=controlled, adjoint=adjoint
        ),
        source,
    )
    cursor = int(controlled)
    system = tuple(range(cursor, cursor + native.system_bitsize))
    cursor += len(system)
    ancillary = tuple(range(cursor, cursor + native.ancilla_bitsize))
    cursor += len(ancillary)
    resources = tuple(range(cursor, cursor + native.resource_bitsize))
    cursor += len(resources)
    logical = cursor
    prep_work, prep_slots = native.coefficient_preparation.workspace_requirements
    traversal = indexed_component_workspace(
        (native.select.selection_bitsize,), int(controlled)
    )
    work = tuple(range(cursor, cursor + max(prep_work, traversal)))
    operation = native.make_on(
        tuple(block.bind_child for block in blocks),
        system=system,
        ancilla=ancillary,
        resource=resources,
        control=(0,) if controlled else (),
        prepare_workspace=work[:prep_work],
        prepare_classical_bits=tuple(range(prep_slots)),
        traversal_workspace=work[:traversal],
        traversal_classical_bits=tuple(range(traversal)),
        workspaces=((),) * len(blocks),
        classical_bits=((),) * len(blocks),
        adjoint=adjoint,
    )
    instructions = Circuit(logical + len(work)).append(operation).to_ir().instructions
    events = certify_erasure_histories(
        instructions, n_wires=logical + len(work), zero_wires=work
    )
    assert events == sum(g.name == "measure" for g in instructions)
    grid = torch.arange(1 << logical, dtype=torch.float64)
    coherent = torch.complex(grid + 1, grid % 5 - 2)
    coherent /= coherent.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    clean = torch.zeros(1 << len(work), dtype=torch.complex128)
    clean[0] = 1
    expected = torch.kron(matrix @ coherent, clean)
    for history in {(0,) * events, (1,) * events}:
        torch.testing.assert_close(
            forced_branch(torch.kron(coherent, clean), instructions, history),
            expected / sqrt(2**events),
            atol=4e-12,
            rtol=0,
        )


@pytest.mark.parametrize("kind", ("normalization", "precision"))
def test_resolved_scalar_declarations_build_the_original_alias_table(kind: str) -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.block_encoding import LinearCombination, Unitary
    from sympy import Symbol, sympify

    from flagquantum.algorithms.primitives.linear_combination_preparation import (
        LinearCombinationPreparation,
    )
    from tests.unit.test_alias_preparation import expected_state, layout

    alpha = Symbol("alpha", positive=True)
    mu = Symbol("mu", positive=True, integer=True)
    declaration = LinearCombination(
        (
            Unitary(XGate(), alpha=alpha if kind == "normalization" else 1),
            Unitary(XGate()),
        ),
        (1.0, -2.0),
        mu if kind == "precision" else 1,
    )
    substitutions = {alpha: 2, mu: 2}
    normalizations = tuple(
        float(sympify(child.alpha).subs(substitutions))
        for child in declaration._block_encodings
    )
    precision = int(sympify(declaration.lambd_bits).subs(substitutions))
    original = LinearCombination(
        tuple(Unitary(XGate(), alpha=value) for value in normalizations),
        declaration._lambd,
        precision,
    )
    native = LinearCombinationPreparation((1.0, -2.0), normalizations, precision)
    assert native.rescaled_coefficients == tuple(original.rescaled_lambd)
    table = original.prepare.prepare
    assert (native.table.alternates, native.table.keep, native.table.mu) == (
        tuple(table.alt),
        tuple(table.keep),
        table.mu,
    )
    bound, sparse = layout(native.table)
    selection, sigma, alternate, keep, flag, work, bits = bound
    junk = (*sigma, *alternate, *keep, flag)
    n = max((*selection, *junk, *work)) + 1
    operation = native.make_on(
        selection=selection, junk=junk, workspace=work, classical_bits=bits
    )
    instructions = Circuit(n).append(operation).to_ir().instructions
    events = certify_erasure_histories(instructions, n_wires=n, zero_wires=work)
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    initial[0] = 1
    expected = expected_state(native.table, bound, sparse).to(torch.complex128)
    for history in {(0,) * events, (1,) * events}:
        torch.testing.assert_close(
            forced_branch(initial, instructions, history),
            expected / sqrt(2**events),
            atol=3e-12,
            rtol=0,
        )


@pytest.mark.parametrize("case", range(9))
@pytest.mark.parametrize(
    "mode,adjoint", tuple(product(("default", "supplied"), (False, True)))
)
def test_structured_quantum_controls_have_complete_history_certificate(
    case: int,
    mode: str,
    adjoint: bool,
) -> None:
    from tests.unit.test_linear_combination_controls import CASES, fixture

    _, _, _, operation, _, predicate, _, work, _, _, _, _ = fixture(
        CASES[case], mode, adjoint
    )
    events = certify_erasure_histories(
        operation.instructions,
        n_wires=operation.required_qubits,
        zero_wires=(*predicate, *work),
    )
    assert events == sum(g.name == "measure" for g in operation.instructions)


@pytest.mark.parametrize("case,new_value", tuple(product(range(8), (0, 1))))
@pytest.mark.parametrize(
    "mode,adjoint", tuple(product(("default", "supplied"), (False, True)))
)
def test_repeated_quantum_control_roots_have_complete_history_certificate(
    case: int,
    new_value: int,
    mode: str,
    adjoint: bool,
) -> None:
    from tests.unit.test_linear_combination_control_stack import OLD_VALUES, fixture

    _, _, operation, _, predicate, _, work, _, _, _, _ = fixture(
        OLD_VALUES[case], new_value, mode, adjoint
    )
    events = certify_erasure_histories(
        operation.instructions,
        n_wires=operation.required_qubits,
        zero_wires=(*predicate, *work),
    )
    assert events == sum(g.name == "measure" for g in operation.instructions)


@pytest.mark.parametrize("value", (0, 1))
def test_classical_control_route_retains_declaration_and_source_missing_body(
    value: int,
) -> None:
    from qualtran import CBit, CtrlSpec, DecomposeNotImplementedError
    from qualtran.bloqs.block_encoding.linear_combination import (
        _linear_combination_block_encoding,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.linear_combination_catalog import (
        linear_combination_example,
        linear_combination_example_circuit,
    )
    from flagquantum.compiler.registers.component_feedback import (
        classical_control_signature,
    )
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.compiler.registers.primitives.classical_controlled import (
        classically_controlled_ir,
    )
    from flagquantum.core.ir.program.register_types import classical_bit_type

    original = _linear_combination_block_encoding()
    controlled, _ = original.get_ctrl_system(CtrlSpec(qdtypes=CBit(), cvs=value))
    native = linear_combination_example()
    signature = classical_control_signature(
        native.signature, ((classical_bit_type(), ()),)
    )
    assert [(reg.name, reg.shape) for reg in signature] == [
        (reg.name, reg.shape) for reg in controlled.signature
    ]
    assert signature[0].dtype == classical_bit_type()
    assert controlled.signature[0].dtype == CBit()
    assert signature.n_qubits() == get_cost_value(controlled, QubitCount())
    with pytest.raises(
        DecomposeNotImplementedError, match="does not declare a decomposition"
    ):
        controlled.decompose_bloq()
    with pytest.raises(DecomposeNotImplementedError):
        get_cost_value(controlled, QECGatesCost())
    body = linear_combination_example_circuit().to_ir().instructions
    with pytest.raises(ValueError, match="cannot guard measurement"):
        classically_controlled_ir(((128,),), (ControlRegister(1, (value,)),), body)
