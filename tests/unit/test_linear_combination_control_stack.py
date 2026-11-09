"""Repeated scalar LCU roots versus unchanged source graph, states and models."""

from collections import Counter
from dataclasses import asdict
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import source_partition_leaf, walk_source_graph
from tests.unit.test_linear_combination_control_extension import (
    fixture as extension_fixture,
)
from tests.unit.test_linear_combination_controls import (
    CASES,
    assert_classical_dependencies,
    original_scoped_gates,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
OLD_VALUES = (
    (0, 1),
    (1, 1),
    *(tuple(bits) + (1,) for bits in product((0, 1), repeat=2)),
    (0, 1, 0, 1),
    (1, 0, 1, 1),
)


def fixture(values, new_value, mode, inverse, precision=8, legacy=False):
    from qualtran import CtrlSpec

    native, _, old_root, _, _, _, _, oldw, bits, options, binders, old_model = (
        extension_fixture(
            CASES[new_value], mode, inverse, precision=precision, legacy=legacy
        )
    )
    source = old_root.ctrl_bloq
    for activation in reversed((new_value, *values[:-1])):
        source = source.controlled(CtrlSpec(cvs=activation))
    assert source.cvs == (new_value, *values)
    if inverse:
        source = source.adjoint()
    k = len(values)
    payload = {
        key: tuple(q + k - 1 for q in options[key])
        for key in ("system", "ancilla", "resource")
    }
    start = 1 + k + sum(map(len, payload.values()))
    p = tuple(range(start, start + k))
    pb = tuple(range(100, 100 + k))
    w = tuple(q + 2 * (k - 1) for q in oldw)
    options |= payload | dict(
        existing_control=tuple(range(1, k + 1)),
        existing_control_values=values,
        control_workspace=p,
        control_classical_bits=pb,
        prepare_workspace=w[: len(options["prepare_workspace"])],
        traversal_workspace=w[: len(options["traversal_workspace"])],
    )
    op = native.make_on(binders, **options)
    ports = {name: wires for name, wires in payload.items() if wires}
    ports["ctrl"] = tuple(range(k + 1))
    positive = old_model.calls[1].component
    prep, select, unprep = [call.component for call in positive.calls]
    model = native.resource_model(
        select,
        prep,
        unprep,
        existing_control=True,
        existing_control_values=values,
        control_registers=options["control_registers"],
        adjoint=inverse,
        legacy_shims=legacy,
    )
    return native, source, op, ports, p, pb, w, bits, options, binders, model


def original_gates(native, source, ports, p, pb, w, bits, options):
    def leaf(bloq, inputs, desired):
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        base = bloq
        while hasattr(base, "subbloq"):
            base = base.subbloq
        if type(base).__name__ in {"And", "MultiAnd"}:
            erase = type(bloq).__name__ == "Adjoint" or getattr(
                bloq, "uncompute", False
            )
            binding = dict(inputs)
            if not erase:
                binding["target"] = p[-1:]
                if p[:-1]:
                    binding["junk"] = p[:-1]
            gates = reference_arithmetic_gates(
                (
                    bloq.decompose_bloq()
                    if erase and type(base).__name__ == "MultiAnd"
                    else bloq
                ),
                binding,
                (),
                dict(zip(p, pb, strict=True)),
                cirq_style=True,
                preserve_components=True,
                reuse_allocations=False,
                expected_outputs=dict(ctrl=inputs["ctrl"]) if erase else binding,
            )
            return gates, dict(ctrl=inputs["ctrl"]) if erase else binding
        if type(base).__name__ in {"BlackBoxPrepare", "BlackBoxSelect", "XGate"}:
            return (
                original_scoped_gates(native, bloq, inputs, p, pb, w, bits, options),
                inputs,
            )
        return None

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    return gates


@pytest.mark.parametrize("values,new_value", tuple(product(OLD_VALUES, (0, 1))))
@pytest.mark.parametrize(
    "mode,inverse", tuple(product(("default", "supplied"), (False, True)))
)
def test_original_repeated_root_primitive_counts_operands_and_feedback(
    values, new_value, mode, inverse
):
    native, source, op, ports, p, pb, w, bits, options, _, _ = fixture(
        values, new_value, mode, inverse
    )
    expected = original_gates(native, source, ports, p, pb, w, bits, options)
    assert Counter(g.name for g in op.instructions) == Counter(g.name for g in expected)
    assert_source_dependencies(op.instructions, expected)
    assert_classical_dependencies(op.instructions, expected)


@pytest.mark.parametrize("values,new_value", tuple(product(OLD_VALUES, (0, 1))))
@pytest.mark.parametrize(
    "mode,inverse,legacy",
    tuple(product(("default", "supplied"), (False, True), (False, True))),
)
def test_original_repeated_root_directional_resource_models(
    values, new_value, mode, inverse, legacy
):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    _, source, _, _, _, _, _, _, _, _, model = fixture(
        values, new_value, mode, inverse, legacy=legacy
    )
    actual = estimate_ftqc_resources(model).logical
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    assert asdict(actual.gates) == {
        key: getattr(expected, key) for key in asdict(actual.gates)
    }
    assert actual.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize(
    "values,new_value,mode",
    [(values, value, "supplied") for values in OLD_VALUES[:6] for value in (0, 1)]
    + [(OLD_VALUES[0], 0, "default")],
)
@pytest.mark.parametrize("inverse", (False, True))
def test_original_repeated_root_full_complex_channels_every_history(
    values, new_value, mode, inverse
):
    _, source, op, _, p, _, w, _, _, _, _ = fixture(
        values, new_value, mode, inverse, precision=1
    )
    scratch = len(p) + len(w)
    logical = op.required_qubits - scratch
    torch.manual_seed(83)
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
            atol=6e-12,
            rtol=0,
        )


@pytest.mark.parametrize("values,new_value", tuple(product(OLD_VALUES, (0, 1))))
def test_original_later_returned_adder_failure_is_not_silently_repaired(
    values, new_value
):
    from attrs import evolve
    from qualtran import BloqBuilder, CtrlSpec

    _, root, _, _, _, _, _, _, _, _, _ = fixture(
        values, new_value, "supplied", False, precision=1
    )
    previous = evolve(root, cvs=values)
    returned, adder = previous.get_ctrl_system(CtrlSpec(cvs=new_value))
    assert returned == root
    bb, soqs = BloqBuilder.from_signature(
        previous.signature, add_registers_allowed=True
    )
    extra = bb.add_register("extra", 1)
    with pytest.raises(TypeError, match="integer scalar arrays"):
        adder(bb, [extra], soqs)


@pytest.mark.parametrize("values", OLD_VALUES[:2])
def test_original_grouped_extension_failure_is_a_separate_execution_boundary(values):
    from attrs import evolve
    from qualtran import CtrlSpec
    from qualtran.resource_counting import QubitCount, get_cost_value

    native, root, _, _, _, _, _, _, options, binders, _ = fixture(
        values, 0, "supplied", False, precision=1
    )
    previous = evolve(root, cvs=values)
    grouped = previous.controlled(CtrlSpec(cvs=(0, 1)))
    with pytest.raises(TypeError, match="integer scalar arrays"):
        grouped.decompose_bloq()
    with pytest.raises(RuntimeError):
        get_cost_value(grouped, QubitCount())
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister

    options |= dict(
        controls=((0, options["system"][0]),),
        control_registers=(ControlRegister(1, (0, 1)),),
    )
    with pytest.raises(ValueError, match="grouped extension builder fails"):
        native.make_on(binders, **options)


@pytest.mark.parametrize(
    "change",
    [
        "missing_values",
        "wrong_count",
        "boolean",
        "negative_origin",
        "wire_alias",
        "work_alias",
        "short_work",
        "short_bits",
    ],
)
def test_repeated_root_storage_and_original_positive_origin_contracts(change):
    native, _, _, _, p, _, w, _, options, binders, _ = fixture(
        (0, 1), 1, "supplied", False, precision=1
    )
    if change == "missing_values":
        options["existing_control_values"] = ()
    elif change == "wrong_count":
        options["existing_control_values"] = (1,)
    elif change == "boolean":
        options["existing_control_values"] = (False, 1)
    elif change == "negative_origin":
        options["existing_control_values"] = (1, 0)
    elif change == "wire_alias":
        options["existing_control"] = (1, 1)
    elif change == "work_alias":
        options["existing_control"] = (1, w[0])
    elif change == "short_work":
        options["control_workspace"] = p[:-1]
    elif change == "short_bits":
        options["control_classical_bits"] = ()
    with pytest.raises(ValueError):
        native.make_on(binders, **options)


def test_repeated_root_model_contracts_and_circuit_free_query(monkeypatch):
    from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
        LinearCombinationBlockEncoding,
    )
    from flagquantum.compiler._component_trace.resource_model import ComponentCost
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister

    native, _, _, _, _, _, _, _, _, _, _ = fixture(
        (0, 1), 0, "supplied", False, precision=1
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Resource query emitted instructions")

    monkeypatch.setattr(LinearCombinationBlockEncoding, "make_on", forbidden)
    prep = ComponentCost(
        "Prepare", native.prepare.selection_width + native.prepare.junk_width
    )
    select = ComponentCost(
        "Select",
        native.system_bitsize + native.ancilla_bitsize + native.resource_bitsize + 1,
    )
    params = dict(
        existing_control=True,
        existing_control_values=(0, 1),
        control_registers=(ControlRegister(1, (0,)),),
    )
    report = estimate_ftqc_resources(
        native.resource_model(select, prep, prep, **params)
    )
    assert report.emitted is None
    for overrides in (
        dict(existing_control=False),
        dict(existing_control_values=(0,)),
        dict(existing_control_values=(0, True)),
        dict(control_registers=(ControlRegister(1, (0, 1)),)),
    ):
        with pytest.raises(ValueError):
            native.resource_model(select, prep, prep, **(params | overrides))


def test_runnable_repeated_control_example(capsys):
    from examples.algorithms.linear_combination_controls import main

    main(repeated=True)
    assert "full complex inverse verified" in capsys.readouterr().out


@pytest.mark.parametrize("new_value,inverse", tuple(product((0, 1), (False, True))))
def test_nonhermitian_phase_children_preserve_direction_and_absolute_branch_phase(
    new_value, inverse
):
    from attrs import evolve
    from qualtran.bloqs.basic_gates import SGate, TGate

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.phase.phase_family import phase_family_ir

    native, source, _, ports, p, pb, w, bits, options, _, _ = fixture(
        (0, 1), new_value, "default", inverse, precision=1
    )
    root = source.subbloq if inverse else source
    positive = root.ctrl_bloq
    positive = evolve(
        positive,
        block_encodings=tuple(
            evolve(child, U=gate)
            for child, gate in zip(
                positive._block_encodings, (SGate(), TGate()), strict=True
            )
        ),
    )
    root = evolve(root, ctrl_bloq=positive)
    source = root.adjoint() if inverse else root

    def factory(label):
        def bind(predicate, child_ports, work, slots, adjoint):
            assert not work and not slots
            gates = phase_family_ir(
                label,
                child_ports["system"][0],
                controls=(predicate,),
                values=(1,),
                adjoint=adjoint,
            )
            return FieldOperation(max(predicate, *child_ports["system"]) + 1, gates)

        return bind

    op = native.make_on((factory("s"), factory("t")), **options)
    expected_gates = original_gates(native, source, ports, p, pb, w, bits, options)
    assert Counter(g.name for g in op.instructions) == Counter(
        g.name for g in expected_gates
    )
    assert_source_dependencies(op.instructions, expected_gates)
    assert_classical_dependencies(op.instructions, expected_gates)
    scratch = len(p) + len(w)
    logical = op.required_qubits - scratch
    torch.manual_seed(89)
    psi = torch.randn(1 << logical, dtype=torch.complex128)
    psi /= psi.norm()
    initial = torch.zeros(1 << op.required_qubits, dtype=torch.complex128)
    desired = torch.zeros_like(initial)
    initial.reshape(-1, 1 << scratch)[:, 0] = psi
    matrix = torch.as_tensor(source.tensor_contract())
    desired.reshape(-1, 1 << scratch)[:, 0] = matrix @ psi
    assert not torch.allclose(matrix, matrix.mH)
    events = sum(g.name == "measure" for g in op.instructions)
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, op.instructions, history),
            desired / sqrt(2**events),
            atol=6e-12,
            rtol=0,
        )
