"""Original full LCU conjugation, metadata and scoped coefficient preparation."""

from dataclasses import asdict
from itertools import product
from math import sqrt

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
)
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.indexed_component import indexed_component_workspace
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = pytest.mark.unit


def fixture(count, precision, auxiliary, controlled, inverse):
    from attrs import evolve
    from qualtran.bloqs.basic_gates import XGate, ZGate
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary

    coefficients = tuple((-1.0) ** index * (index + 1) for index in range(count))
    children, sources, binders, costs = [], [], [], []
    for index in range(count):
        label = "x" if index % 2 == 0 else "z"
        anc, res = (index % 2, index % 2) if auxiliary else (0, 0)
        alpha = 1.0 + index / 2
        children.append(
            UnitaryBlockEncoding(
                (("q", 1),),
                lambda p, w, b, a: (),
                alpha=alpha,
                ancilla_bitsize=anc,
                resource_bitsize=res,
                epsilon=index / 100,
            )
        )
        sources.append(
            evolve(
                Unitary(XGate() if label == "x" else ZGate()),
                alpha=alpha,
                ancilla_bitsize=anc,
                resource_bitsize=res,
                epsilon=index / 100,
            )
        )

        def bind(predicate, ports, work, bits, adjoint, name=label):
            assert not work and not bits
            return FieldOperation(
                max(predicate, *ports["system"]) + 1,
                (Instruction("c" + name, (predicate, *ports["system"])),),
            )

        binders.append(bind)
        costs.append(
            ComponentCost("ControlledBlock", 2 + anc + res, GateCost(clifford=1))
        )
    native = LinearCombinationBlockEncoding(tuple(children), coefficients, precision)
    base = LinearCombination(tuple(sources), coefficients, precision)
    source = evolve(base, is_controlled=True) if controlled else base
    if inverse:
        source = source.adjoint()
    cursor = int(controlled)
    control = (0,) if controlled else ()
    system = (cursor,)
    cursor += 1
    ancilla = tuple(range(cursor, cursor + native.ancilla_bitsize))
    cursor += len(ancilla)
    resource = tuple(range(cursor, cursor + native.resource_bitsize))
    cursor += len(resource)
    prepare_work, prepare_bits = native.coefficient_preparation.workspace_requirements
    traversal = indexed_component_workspace(
        (native.select.selection_bitsize,), len(control)
    )
    work = tuple(range(cursor, cursor + max(prepare_work, traversal)))
    bits = tuple(range(30, 30 + len(work)))
    options = dict(
        system=system,
        ancilla=ancilla,
        resource=resource,
        control=control,
        prepare_workspace=work[:prepare_work],
        prepare_classical_bits=bits[:prepare_bits],
        traversal_workspace=work[:traversal],
        traversal_classical_bits=bits[:traversal],
        workspaces=((),) * count,
        classical_bits=((),) * count,
        adjoint=inverse,
    )
    op = native.make_on(tuple(binders), **options)
    ports = {"system": system, "ancilla": ancilla, "resource": resource}
    if control:
        ports["ctrl"] = control
    return (
        native,
        base,
        source,
        op,
        ports,
        work,
        bits,
        tuple(costs),
        tuple(binders),
        options,
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("count,precision", [(2, 1), (3, 1), (2, 8), (3, 8)])
@pytest.mark.parametrize(
    "auxiliary,controlled,inverse", tuple(product((False, True), repeat=3))
)
def test_complete_original_gate_flow_metadata_and_model(
    count, precision, auxiliary, controlled, inverse
):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    native, base, source, op, ports, work, bits, costs, _, _ = fixture(
        count, precision, auxiliary, controlled, inverse
    )
    assert native.alpha == base.alpha
    assert native.epsilon == base.epsilon
    assert native.ancilla_bitsize == base.ancilla_bitsize
    assert native.resource_bitsize == base.resource_bitsize
    from tests.unit.register_source_graph import (
        source_partition_leaf,
        walk_source_graph,
    )

    def leaf(bloq, inputs, desired):
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        base_bloq = bloq
        while hasattr(base_bloq, "subbloq"):
            base_bloq = base_bloq.subbloq
        kind = type(base_bloq).__name__
        if kind == "BlackBoxPrepare":
            inverse_prepare = type(bloq).__name__ == "Adjoint"
            nwork, nslots = native.coefficient_preparation.workspace_requirements
            # Classical storage belongs to this component scope, not to the
            # flattened parent's physical allocation index.
            declared = native.coefficient_preparation.prepare.emit(
                inputs["selection"],
                inputs["junk"],
                work[:nwork],
                bits[:nslots],
                adjoint=inverse_prepare,
            )
            slots = {
                g.wires[0]: g.metadata["classical_bit"]
                for g in declared
                if g.name == "measure"
            }
            gates = reference_arithmetic_gates(
                bloq.decompose_bloq(),
                inputs,
                work[:nwork],
                slots,
                cirq_style=True,
                preserve_components=True,
            )
            return gates, inputs
        if kind == "BlackBoxSelect":
            nwork = indexed_component_workspace(
                (native.select.selection_bitsize,), int(controlled)
            )
            gates = reference_arithmetic_gates(
                bloq.decompose_bloq(),
                inputs,
                work[:nwork],
                dict(zip(work[:nwork], bits[:nwork], strict=True)),
                cirq_style=True,
                preserve_components=True,
                keep_pauli_leaves=True,
                keep_exchange_leaves=True,
            )
            return gates, inputs
        return None

    original, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert_source_dependencies(op.instructions, original)
    prep = native.coefficient_preparation.resource_model()
    select = native.select.resource_model(costs, controlled=controlled, adjoint=inverse)
    model = estimate_ftqc_resources(
        native.resource_model(
            select, prep, prep, controlled=controlled, adjoint=inverse
        )
    ).logical
    original_cost = get_cost_value(source, QECGatesCost())
    assert asdict(model.gates) == {
        key: getattr(original_cost, key) for key in asdict(model.gates)
    }
    assert model.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "count,auxiliary,controlled,inverse",
    [(2, a, c, d) for a, c, d in product((False, True), repeat=3)]
    + [(3, False, c, d) for c, d in product((False, True), repeat=2)],
)
def test_full_matrix_all_complex_branches_and_clean_work(
    count, auxiliary, controlled, inverse
):
    native, base, source, op, ports, work, bits, costs, binders, options = fixture(
        count, 1, auxiliary, controlled, inverse
    )
    logical = op.required_qubits - len(work)
    torch.manual_seed(37)
    psi = torch.randn(1 << logical, dtype=torch.complex128)
    psi /= psi.norm()
    initial = torch.zeros(1 << op.required_qubits, dtype=torch.complex128)
    desired = torch.zeros_like(initial)
    initial.reshape(-1, 1 << len(work))[:, 0] = psi
    desired.reshape(-1, 1 << len(work))[:, 0] = (
        torch.as_tensor(source.tensor_contract()) @ psi
    )
    circuit = DynamicCircuit(op.required_qubits)
    circuit.append(op)
    instructions = tuple(circuit.to_ir().instructions)
    events = sum(g.name == "measure" for g in instructions)
    assert events <= 6
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, instructions, history),
            desired / sqrt(2**events),
            atol=4e-12,
            rtol=0,
        )


def test_invalid_packed_binding_and_overlapping_prepare_work():
    native, _, _, _, _, work, _, _, binders, options = fixture(
        2, 1, False, False, False
    )
    for replacement in (
        {"ancilla": ()},
        {"resource": ()},
        {"prepare_workspace": (0,)},
        {"prepare_classical_bits": (-1,)},
        {"system": ()},
        {"adjoint": 1},
    ):
        with pytest.raises(ValueError):
            native.make_on(binders, **(options | replacement))
    assert work


def test_model_requires_full_ports_and_no_circuit_emission(monkeypatch):
    native, _, _, _, _, _, _, costs, _, _ = fixture(2, 1, False, False, False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Model requested circuit emission")

    monkeypatch.setattr(LinearCombinationBlockEncoding, "make_on", forbidden)
    prep = native.coefficient_preparation.resource_model()
    selected = native.select.resource_model(costs)
    report = estimate_ftqc_resources(native.resource_model(selected, prep, prep))
    assert report.emitted is None
    with pytest.raises(ValueError, match="through ports"):
        native.resource_model(ComponentCost("missing", 0), prep, prep)


@pytest.mark.ftqc_reference
def test_source_epsilon_implementation_is_not_docstring_bound():
    native, source, _, _, _, _, _, _, _, _ = fixture(2, 1, False, False, False)
    assert native.epsilon == source.epsilon == 0.01
    assert sum(abs(c) for c in native.coefficients) * 0.01 == 0.03
    assert native.signal_state.normalization is None


def test_runnable_example(capsys):
    from examples.algorithms.linear_combination_block_encoding import main

    main()
    assert "full complex inverse verified" in capsys.readouterr().out


@pytest.mark.ftqc_reference
def test_symbolic_normalization_metadata_remains_separate_from_preparation():
    from dataclasses import replace

    import sympy
    from attrs import evolve
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary

    alpha = sympy.Symbol("alpha", positive=True)
    child = UnitaryBlockEncoding((("q", 1),), lambda p, w, b, a: ())
    native = LinearCombinationBlockEncoding(
        (replace(child, alpha=alpha), child), (1.0, -2.0), 1
    )
    original = LinearCombination(
        (evolve(Unitary(XGate()), alpha=alpha), Unitary(XGate())), (1.0, -2.0), 1
    )
    assert sympy.simplify(native.alpha - original.alpha) == 0
    with pytest.raises(ValueError, match="concrete block normalizations"):
        _ = native.coefficient_preparation
