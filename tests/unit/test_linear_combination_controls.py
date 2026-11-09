"""Pinned controlled LCU dispatch, predicate lifetimes and full source branches."""

from dataclasses import asdict
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.core.classical_control import instruction_condition_clauses
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    source_control_predicate_leaf,
    source_partition_leaf,
    walk_source_graph,
)
from tests.unit.test_linear_combination_block_encoding import fixture as default_fixture
from tests.unit.test_linear_combination_custom_oracles import (
    fixture as supplied_fixture,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_control_predicate import control_fixture
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = (
    (("QBit", 1, (), (0,), "unsigned"),),
    (("QBit", 1, (), (1,), "unsigned"),),
    (("QBit", 1, (2,), (0, 1), "unsigned"),),
    (("QBit", 1, (3,), (1, 0, 1), "unsigned"),),
    (("QUInt", 2, (), (2,), "unsigned"),),
    (("QInt", 2, (), (-1,), "twos_complement"), ("QBit", 1, (), (0,), "unsigned")),
    (("QIntOnesComp", 3, (), (-1,), "ones_complement"),),
    (("QMontgomeryUInt", 3, (), (3,), "unsigned"), ("BQUInt", 2, (), (1,), "unsigned")),
    (("QBit", 1, (2, 2), (1, 0, 0, 1), "unsigned"),),
)


def fixture(case, mode, inverse, precision=8, legacy=False):
    spec, registers, groups = control_fixture(case)
    count = sum(len(group) for group in groups)
    if mode == "default":
        native, base, _, _, _, _, _, children, binders, old = default_fixture(
            2, precision, True, True, False
        )
        prep = native.coefficient_preparation.resource_model(legacy_shims=legacy)
        select = native.select.resource_model(
            children, controlled=True, adjoint=inverse, legacy_shims=legacy
        )
        unprep = prep
    else:
        native, base, _, _, _, _, _, old_model, _, old, binders = supplied_fixture(
            "both_no_junk", True, True, False
        )
        prep = old_model.calls[0].component
        unprep = old_model.calls[2].component
        width = native.prepare.selection_width
        target = (
            native.system_bitsize
            + native.select.ancilla_bitsize
            + native.select.resource_bitsize
        )
        select = indexed_component_cost(
            (width,),
            (1 << width,),
            (ComponentCost("ControlledUnitary", target + 1, GateCost(clifford=1)),)
            * (1 << width),
            target_width=target,
            control_count=1,
            adjoint=inverse,
            legacy_shims=legacy,
        )
    original = (base.adjoint() if inverse else base).controlled(spec)
    cursor = count
    system = tuple(range(cursor, cursor + native.system_bitsize))
    cursor += len(system)
    ancilla = tuple(range(cursor, cursor + native.ancilla_bitsize))
    cursor += len(ancilla)
    resource = tuple(range(cursor, cursor + native.resource_bitsize))
    cursor += len(resource)
    predicate = tuple(range(cursor, cursor + count - 1))
    cursor += len(predicate)
    stage_width = max(len(old["prepare_workspace"]), len(old["traversal_workspace"]))
    work = tuple(range(cursor, cursor + stage_width))
    bits = tuple(
        range(
            20,
            20
            + max(
                len(old["prepare_classical_bits"]), len(old["traversal_classical_bits"])
            ),
        )
    )
    predicate_bits = tuple(range(100, 100 + len(predicate)))
    options = dict(
        system=system,
        ancilla=ancilla,
        resource=resource,
        prepare_workspace=work[: len(old["prepare_workspace"])],
        prepare_classical_bits=bits[: len(old["prepare_classical_bits"])],
        traversal_workspace=work[: len(old["traversal_workspace"])],
        traversal_classical_bits=bits[: len(old["traversal_classical_bits"])],
        workspaces=old["workspaces"],
        classical_bits=old["classical_bits"],
        adjoint=inverse,
        controls=groups,
        control_registers=registers,
        control_workspace=predicate,
        control_classical_bits=predicate_bits,
    )
    op = native.make_on(binders, **options)
    ports = {"system": system, "ancilla": ancilla}
    if resource:
        ports["resource"] = resource
    names = [reg.name for reg in original.signature if reg.name not in ports]
    ports.update(zip(names, groups, strict=True))
    model = native.resource_model(
        select,
        prep,
        unprep,
        control_registers=registers,
        adjoint=inverse,
        legacy_shims=legacy,
    )
    return (
        native,
        base,
        original,
        op,
        ports,
        predicate,
        predicate_bits,
        work,
        bits,
        options,
        binders,
        model,
    )


def original_scoped_gates(
    native, source, ports, predicate, predicate_bits, work, bits, options
):
    def leaf(bloq, inputs, desired):
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        control = source_control_predicate_leaf(bloq, inputs, predicate, predicate_bits)
        if control is not None:
            return control
        base = bloq
        while hasattr(base, "subbloq"):
            base = base.subbloq
        kind = type(base).__name__
        if kind == "XGate":
            return (Instruction("x", inputs["q"]),), inputs
        if kind == "BlackBoxPrepare":
            inverse = type(bloq).__name__ == "Adjoint"
            pw = options["prepare_workspace"]
            pb = options["prepare_classical_bits"]
            declared = native.prepare.emit(
                inputs["selection"], inputs.get("junk", ()), pw, pb, adjoint=inverse
            )
            slots = {
                g.wires[0]: g.metadata["classical_bit"]
                for g in declared
                if g.name == "measure"
            }
            return (
                reference_arithmetic_gates(
                    bloq.decompose_bloq(),
                    inputs,
                    pw,
                    slots,
                    cirq_style=True,
                    preserve_components=True,
                    keep_exchange_leaves=True,
                ),
                inputs,
            )
        if kind == "BlackBoxSelect":
            tw = options["traversal_workspace"]
            tb = options["traversal_classical_bits"]
            return (
                reference_arithmetic_gates(
                    bloq.decompose_bloq(),
                    inputs,
                    tw,
                    dict(zip(tw, tb, strict=True)),
                    cirq_style=True,
                    preserve_components=True,
                    keep_pauli_leaves=True,
                    keep_exchange_leaves=True,
                ),
                inputs,
            )
        return None

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    return gates


def assert_classical_dependencies(actual, expected):
    def slots(instruction):
        used = {
            bit
            for clause in instruction_condition_clauses(instruction)
            for bit, _ in clause
        }
        if instruction.name == "measure":
            used.add(instruction.metadata["classical_bit"])
        return used

    bits = {bit for gate in (*actual, *expected) for bit in slots(gate)}
    for bit in bits:
        assert [gate for gate in actual if bit in slots(gate)] == [
            gate for gate in expected if bit in slots(gate)
        ]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "mode,inverse", tuple(product(("default", "supplied"), (False, True)))
)
def test_original_dispatch_scoped_gate_operands_and_resource_profiles(
    case, mode, inverse
):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    for legacy in (False, True):
        native, _, source, op, ports, p, pb, w, b, options, _, model = fixture(
            case, mode, inverse, legacy=legacy
        )
        gates = original_scoped_gates(native, source, ports, p, pb, w, b, options)
        assert_source_dependencies(op.instructions, gates)
        assert_classical_dependencies(op.instructions, gates)
        expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        actual = estimate_ftqc_resources(model).logical
        assert asdict(actual.gates) == {
            name: getattr(expected, name) for name in asdict(actual.gates)
        }
        assert actual.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize(
    "case,mode",
    [(CASES[i], "supplied") for i in range(7)]
    + [(CASES[i], "default") for i in (0, 1, 2)],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_full_original_complex_tensor_every_history_and_clean_owned_work(
    case, mode, inverse
):
    _, _, source, op, ports, p, _, w, _, _, _, _ = fixture(
        case, mode, inverse, precision=1
    )
    scratch = len(p) + len(w)
    logical = op.required_qubits - scratch
    torch.manual_seed(47)
    psi = torch.randn(1 << logical, dtype=torch.complex128)
    psi /= psi.norm()
    initial = torch.zeros(1 << op.required_qubits, dtype=torch.complex128)
    desired = torch.zeros_like(initial)
    initial.reshape(-1, 1 << scratch)[:, 0] = psi
    desired.reshape(-1, 1 << scratch)[:, 0] = (
        torch.as_tensor(source.tensor_contract()) @ psi
    )
    circuit = DynamicCircuit(op.required_qubits)
    circuit.append(op)
    events = sum(g.name == "measure" for g in op.instructions)
    assert events <= 6
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, op.instructions, history),
            desired / sqrt(2**events),
            atol=5e-12,
            rtol=0,
        )


@pytest.mark.parametrize("case", [CASES[i] for i in (0, 2, 5)])
@pytest.mark.parametrize("inverse", [False, True])
def test_original_control_adder_interface_matches_returned_root(case, inverse):
    from qualtran import BloqBuilder

    _, base, original, _, _, _, _, _, _, _, _, _ = fixture(case, "supplied", inverse)
    spec, _, _ = control_fixture(case)
    child = base.adjoint() if inverse else base
    root, adder = child.get_ctrl_system(spec)
    assert root == original
    bb, soqs = BloqBuilder.from_signature(root.signature)
    names = [
        reg.name
        for reg in root.signature
        if reg.name not in {"system", "ancilla", "resource"}
    ]
    controls = [soqs.pop(name) for name in names]
    returned_controls, returned_payload = adder(bb, controls, soqs)
    outputs = dict(
        zip(
            (reg.name for reg in child.signature.rights()),
            returned_payload,
            strict=True,
        )
    )
    outputs.update(zip(names, returned_controls, strict=True))
    built = bb.finalize(**outputs)
    torch.testing.assert_close(
        torch.as_tensor(built.tensor_contract()),
        torch.as_tensor(root.tensor_contract()),
        atol=3e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "mixed_direct",
        "predicate_alias",
        "child_alias",
        "outer_slot",
        "predicate_size",
        "group_width",
        "work_without_control",
    ],
)
def test_control_binding_contracts_reject_ambiguous_or_overlapping_storage(change):
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister

    native, _, _, _, _, p, _, w, _, options, binders, _ = fixture(
        CASES[2], "default", False, precision=1
    )
    if change == "empty":
        options |= dict(
            controls=((),),
            control_registers=(ControlRegister(1, ()),),
            control_workspace=(),
            control_classical_bits=(),
        )
    if change == "mixed_direct":
        options["control"] = (0,)
    if change == "predicate_alias":
        options["control_workspace"] = options["system"]
    if change == "child_alias":
        options["prepare_workspace"] = (p[0], w[1])
    if change == "outer_slot":
        options["control_classical_bits"] = (-1,)
    if change == "predicate_size":
        options["control_workspace"] = ()
    if change == "group_width":
        options["controls"] = ((0,),)
    if change == "work_without_control":
        options |= dict(controls=(), control_registers=())
    with pytest.raises(ValueError):
        native.make_on(binders, **options)


def test_control_model_rejects_ambiguous_mode_and_does_not_emit(monkeypatch):
    from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
        LinearCombinationBlockEncoding,
    )
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister

    native, _, _, _, _, _, _, _, _, options, _, model = fixture(
        CASES[2], "default", False, precision=1
    )
    positive = model.calls[1].component
    prep, select, unprep = (call.component for call in positive.calls)

    def forbidden(*args, **kwargs):
        raise AssertionError("Model requested quantum emission")

    monkeypatch.setattr(LinearCombinationBlockEncoding, "make_on", forbidden)
    result = estimate_ftqc_resources(
        native.resource_model(
            select, prep, unprep, control_registers=options["control_registers"]
        )
    )
    assert result.emitted is None
    assert result.logical == estimate_ftqc_resources(model).logical
    for overrides in (
        dict(controlled=True, control_registers=options["control_registers"]),
        dict(control_registers=(ControlRegister(1, ()),)),
        dict(legacy_shims=1),
    ):
        with pytest.raises(ValueError):
            native.resource_model(select, prep, unprep, **overrides)


def test_runnable_structured_control_example(capsys):
    from examples.algorithms.linear_combination_controls import main

    main()
    assert "full complex inverse verified" in capsys.readouterr().out
