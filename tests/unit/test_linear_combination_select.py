"""Source signed SELECT, padded ports, absolute phase and measured predicates."""

from dataclasses import asdict
from itertools import product
from math import sqrt

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.linear_combination_select import (
    LinearCombinationSelect,
)
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_workspace,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = pytest.mark.unit


def fixture(count, auxiliary, controlled, adjoint):
    from attrs import evolve
    from qualtran.bloqs.basic_gates import XGate, ZGate
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary

    coefficients = tuple(-1.0 if i % 2 else 1.0 for i in range(count))
    children = []
    originals = []
    factories = []
    costs = []
    for index in range(count):
        name = "x" if index % 2 == 0 else "z"
        anc, res = (index % 2, index % 3) if auxiliary else (0, 0)
        children.append(
            UnitaryBlockEncoding(
                (("q", 1),),
                lambda p, w, b, a: (),
                ancilla_bitsize=anc,
                resource_bitsize=res,
            )
        )
        originals.append(
            evolve(
                Unitary(XGate() if name == "x" else ZGate()),
                ancilla_bitsize=anc,
                resource_bitsize=res,
            )
        )

        def bind(predicate, ports, work, bits, inverse, label=name):
            assert not work and not bits
            return FieldOperation(
                max(predicate, *ports["system"]) + 1,
                (Instruction("c" + label, (predicate, *ports["system"])),),
            )

        factories.append(bind)
        costs.append(
            ComponentCost("ControlledBlock", 2 + anc + res, GateCost(clifford=1))
        )
    native = LinearCombinationSelect(tuple(children), coefficients)
    # The parent's untouched factory builds signed AutoPartition children.
    source = LinearCombination(tuple(originals), coefficients, 1).select
    if controlled:
        source = source.controlled()
    if adjoint:
        source = source.adjoint()
    ctrl = (0,) if controlled else ()
    start = int(controlled)
    selection = tuple(range(start, start + native.selection_bitsize))
    system = (start + len(selection),)
    ancilla = tuple(range(system[0] + 1, system[0] + 1 + native.ancilla_bitsize))
    resource = tuple(
        range(
            system[0] + 1 + len(ancilla),
            system[0] + 1 + len(ancilla) + native.resource_bitsize,
        )
    )
    through = (*ctrl, *selection, *system, *ancilla, *resource)
    nwork = indexed_component_workspace((len(selection),), len(ctrl))
    work = tuple(range(len(through), len(through) + nwork))
    bits = tuple(range(30, 30 + nwork))
    op = native.make_on(
        tuple(factories),
        selection=selection,
        system=system,
        ancilla=ancilla,
        resource=resource,
        control=ctrl,
        traversal_workspace=work,
        traversal_classical_bits=bits,
        workspaces=((),) * count,
        classical_bits=((),) * count,
        adjoint=adjoint,
    )
    ports = {"selection": selection, "system": (*system, *ancilla, *resource)}
    if controlled:
        ports["ctrl"] = ctrl
    return native, source, op, ports, work, bits, tuple(costs)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize(
    "auxiliary,controlled,adjoint", tuple(product((False, True), repeat=3))
)
def test_full_source_gates_and_cost_profiles(count, auxiliary, controlled, adjoint):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    native, source, op, ports, work, bits, costs = fixture(
        count, auxiliary, controlled, adjoint
    )
    original = reference_arithmetic_gates(
        source.decompose_bloq(),
        ports,
        work,
        dict(zip(work, bits, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
        keep_exchange_leaves=True,
        preserve_components=True,
    )
    assert_source_dependencies(op.instructions, original)
    for legacy in (False, True):
        model = estimate_ftqc_resources(
            native.resource_model(
                costs, controlled=controlled, adjoint=adjoint, legacy_shims=legacy
            )
        ).logical
        reference = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        assert asdict(model.gates) == {
            key: getattr(reference, key) for key in asdict(model.gates)
        }
        assert model.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "count,controlled,adjoint", tuple(product((2, 3), (False, True), (False, True)))
)
def test_entangled_all_ports_and_all_measurement_branches(count, controlled, adjoint):
    native, source, op, ports, work, bits, _ = fixture(count, True, controlled, adjoint)
    through = op.required_qubits - len(work)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.manual_seed(31)
    psi = torch.randn(1 << through, dtype=torch.complex128)
    psi /= torch.linalg.vector_norm(psi)
    expected = torch.zeros(1 << op.required_qubits, dtype=torch.complex128)
    initial = torch.zeros_like(expected)
    # Include nonzero, entangled bystander tails and all padded addresses.
    initial.reshape(-1, 1 << len(work))[:, 0] = psi
    expected.reshape(-1, 1 << len(work))[:, 0] = matrix @ psi
    circuit = DynamicCircuit(op.required_qubits)
    circuit.append(op)
    instructions = tuple(circuit.to_ir().instructions)
    events = sum(g.name == "measure" for g in instructions)
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, instructions, history),
            expected / sqrt(2**events),
            atol=3e-12,
            rtol=0,
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "count,controlled,adjoint", tuple(product((2, 3), (False, True), (False, True)))
)
def test_measured_children_reuse_private_pool_and_clean_every_branch(
    count, controlled, adjoint
):
    import numpy as np
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.arithmetic.controlled_addition import (
        controlled_add_ir,
    )

    block = UnitaryBlockEncoding((("a", 1), ("b", 1)), lambda p, w, b, a: ())
    coefficients = tuple((-1.0) ** i for i in range(count))
    native = LinearCombinationSelect((block,) * count, coefficients)
    source = LinearCombination(
        (Unitary(Add(QUInt(1))),) * count, coefficients, 1
    ).select
    if controlled:
        source = source.controlled()
    if adjoint:
        source = source.adjoint()
    control = (0,) if controlled else ()
    selection = tuple(
        range(int(controlled), int(controlled) + native.selection_bitsize)
    )
    system = (len(control) + len(selection), len(control) + len(selection) + 1)
    nthrough = len(control) + len(selection) + 2
    nparent = indexed_component_workspace((len(selection),), len(control))
    parent_work = tuple(range(nthrough, nthrough + nparent))
    child_work = (nthrough + nparent,)
    parent_bits = tuple(range(20, 20 + nparent))
    child_bits = (40,)
    width = nthrough + nparent + 1

    def bind(predicate, ports, work, bits, inverse):
        if inverse:
            from flagquantum.compiler.registers.primitives.controlled_not import (
                cnot_with_controls_ir,
            )

            # The original generic C[Add†] Cirq path controls its one CNOT;
            # it does not dispatch to the specialized CAdd† implementation.
            return FieldOperation(
                width,
                cnot_with_controls_ir(
                    ports["system"][0],
                    ports["system"][1],
                    controls=(predicate,),
                    values=(1,),
                ),
            )
        emitter = controlled_add_ir
        return FieldOperation(
            width,
            emitter(
                predicate, ports["system"][:1], ports["system"][1:], work, (), bits, ()
            ),
        )

    op = native.make_on(
        (bind,) * count,
        selection=selection,
        system=system,
        control=control,
        traversal_workspace=parent_work,
        traversal_classical_bits=parent_bits,
        workspaces=(child_work,) * count,
        classical_bits=(child_bits,) * count,
        adjoint=adjoint,
    )
    ports = {"selection": selection, "system": system}
    if control:
        ports["ctrl"] = control
    gates = reference_arithmetic_gates(
        source.decompose_bloq(),
        ports,
        (*parent_work, *child_work),
        dict(
            zip((*parent_work, *child_work), (*parent_bits, *child_bits), strict=True)
        ),
        cirq_style=True,
        preserve_components=True,
        keep_pauli_leaves=True,
        keep_exchange_leaves=True,
    )
    assert_source_dependencies(op.instructions, gates)
    if adjoint:
        from qualtran import DecomposeTypeError

        # Its declaration still recurses into the atomic controlled And-adjoint.
        # Do not replace that failed model with the executable Toffoli count.
        with pytest.raises(DecomposeTypeError, match="And"):
            get_cost_value(source, QECGatesCost())
    else:
        # Preserve the original one-bit Add model's negative Clifford anomaly.
        child_model = ComponentCost(
            "ControlledAdd1", 4, GateCost(and_bloq=1, clifford=-2, measurement=1)
        )
        modeled = estimate_ftqc_resources(
            native.resource_model((child_model,) * count, controlled=controlled)
        ).logical
        original_cost = get_cost_value(source, QECGatesCost())
        assert asdict(modeled.gates) == {
            key: getattr(original_cost, key) for key in asdict(modeled.gates)
        }
        assert modeled.peak_logical_qubits == get_cost_value(source, QubitCount())
    torch.manual_seed(33)
    psi = torch.randn(1 << nthrough, dtype=torch.complex128)
    psi /= psi.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    desired = torch.zeros_like(initial)
    initial.reshape(1 << nthrough, -1)[:, 0] = psi
    desired.reshape(1 << nthrough, -1)[:, 0] = (
        torch.as_tensor(np.asarray(source.tensor_contract())) @ psi
    )
    events = sum(g.name == "measure" for g in op.instructions)
    assert events <= 5
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, op.instructions, history),
            desired / sqrt(2**events),
            atol=4e-12,
            rtol=0,
        )


@pytest.mark.parametrize(
    "change",
    [
        "selection",
        "ancilla",
        "groups",
        "parent_alias",
        "child_alias",
        "escape",
        "controls",
        "direction",
    ],
)
def test_binding_rejects_invalid_contract_before_emission(change):
    block = UnitaryBlockEncoding((("q", 1),), lambda p, w, b, a: ())
    native = LinearCombinationSelect((block,) * 3, (1.0, -1.0, 1.0))

    def bind(predicate, ports, work, bits, inverse):
        return (
            FieldOperation(4, (Instruction("x", (1,)),))
            if change == "escape"
            else FieldOperation(4, ())
        )

    options = dict(
        selection=(0, 1),
        system=(2,),
        traversal_workspace=(3,),
        traversal_classical_bits=(0,),
        workspaces=((),) * 3,
        classical_bits=((),) * 3,
    )
    if change == "selection":
        options["selection"] = (0,)
    if change == "ancilla":
        options["ancilla"] = (4,)
    if change == "groups":
        options["workspaces"] = ((),)
    if change == "parent_alias":
        options["traversal_workspace"] = (2,)
    if change == "child_alias":
        options["workspaces"] = ((3,), (), ())
    if change == "controls":
        options["control"] = (4, 5)
    if change == "direction":
        options["adjoint"] = 1
    with pytest.raises(
        ValueError, match="another child" if change == "escape" else None
    ):
        native.make_on((bind,) * 3, **options)


@pytest.mark.parametrize("coefficients", [(0.0, 0.0), (float("nan"), 1.0), (1.0,)])
def test_invalid_term_list(coefficients):
    block = UnitaryBlockEncoding((("q", 1),), lambda p, w, b, a: ())
    with pytest.raises(ValueError):
        LinearCombinationSelect((block, block), coefficients)


def test_runnable_example(capsys):
    from examples.algorithms.linear_combination_select import main

    main()
    assert "inverse restored" in capsys.readouterr().out
