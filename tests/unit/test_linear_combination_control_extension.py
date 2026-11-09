"""Precontrolled LCU source roots, live predicates and explicit adjoint placement."""

from collections import Counter
from dataclasses import asdict
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    source_control_predicate_leaf,
    source_partition_leaf,
    walk_source_graph,
)
from tests.unit.test_linear_combination_controls import (
    CASES,
    assert_classical_dependencies,
    original_scoped_gates,
)
from tests.unit.test_linear_combination_controls import (
    fixture as controlled_fixture,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_control_predicate import control_fixture
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def fixture(case, mode, inverse, precision=8, legacy=False):
    from attrs import evolve

    native, base, _, _, _, _, _, oldw, bits, options, binders, _ = controlled_fixture(
        case, mode, False, precision=precision, legacy=legacy
    )
    spec, registers, groups = control_fixture(case)
    count = sum(map(len, groups))
    current = (count,)
    payload = {
        key: tuple(wire + 1 for wire in options[key])
        for key in ("system", "ancilla", "resource")
    }
    start = count + 1 + sum(map(len, payload.values()))
    p = tuple(range(start, start + count))
    w = tuple(wire + 2 for wire in oldw)
    pb = tuple(range(100, 100 + count))
    options |= payload | dict(
        existing_control=current,
        control_workspace=p,
        control_classical_bits=pb,
        prepare_workspace=w[: len(options["prepare_workspace"])],
        traversal_workspace=w[: len(options["traversal_workspace"])],
        adjoint=inverse,
    )
    op = native.make_on(binders, **options)
    root = evolve(base, is_controlled=True).controlled(spec)
    source = root.adjoint() if inverse else root
    ports = {key: value for key, value in payload.items() if value}
    if count == 1:
        ports["ctrl"] = (*groups[0], *current)
    else:
        names = [r.name for r in source.signature if r.name not in {*payload, "ctrl"}]
        ports.update(zip(names, groups, strict=True))
        ports["ctrl"] = current
    # Supply child direction/profile exactly as for the established LCU model.
    _, _, _, _, _, _, _, _, _, _, _, old_model = controlled_fixture(
        case, mode, inverse, precision=precision, legacy=legacy
    )
    positive = old_model
    if len(tuple(bit for reg in registers for bit in reg.bits())) > 1:
        positive = old_model.calls[1].component
    elif registers[0].bits() == (0,):
        positive = old_model.calls[0].component
    prep, select, unprep = [call.component for call in positive.calls]
    model = native.resource_model(
        select,
        prep,
        unprep,
        control_registers=registers,
        existing_control=True,
        adjoint=inverse,
        legacy_shims=legacy,
    )
    return native, source, root, op, ports, p, pb, w, bits, options, binders, model


def original_gates(native, source, ports, p, pb, w, bits, options):
    def leaf(bloq, inputs, desired):
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        outer = source_control_predicate_leaf(bloq, inputs, p[:-1], pb[:-1])
        if outer is not None:
            return outer
        if type(bloq).__name__ == "And":
            binding = dict(inputs)
            if not bloq.uncompute:
                binding["target"] = p[-1:]
            gates = reference_arithmetic_gates(
                bloq,
                binding,
                (),
                {p[-1]: pb[-1]},
                cirq_style=True,
                preserve_components=True,
            )
            return gates, dict(ctrl=inputs["ctrl"]) if bloq.uncompute else binding
        base = bloq
        while hasattr(base, "subbloq"):
            base = base.subbloq
        if type(base).__name__ in {"BlackBoxPrepare", "BlackBoxSelect", "XGate"}:
            return (
                original_scoped_gates(native, bloq, inputs, p, pb, w, bits, options),
                inputs,
            )
        return None

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    return gates


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "mode,inverse", tuple(product(("default", "supplied"), (False, True)))
)
def test_original_precontrolled_root_gate_dependencies(case, mode, inverse):
    native, source, _, op, ports, p, pb, w, bits, options, _, _ = fixture(
        case, mode, inverse
    )
    expected = original_gates(native, source, ports, p, pb, w, bits, options)
    assert Counter(g.name for g in op.instructions) == Counter(g.name for g in expected)
    assert_source_dependencies(op.instructions, expected)
    assert_classical_dependencies(op.instructions, expected)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "mode,legacy", tuple(product(("default", "supplied"), (False, True)))
)
@pytest.mark.parametrize("inverse", (False, True))
def test_original_precontrolled_root_resource_categories_and_live_width(
    case, mode, legacy, inverse
):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    _, source, _, _, _, _, _, _, _, _, _, model = fixture(
        case, mode, inverse, legacy=legacy
    )
    actual = estimate_ftqc_resources(model).logical
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    assert asdict(actual.gates) == {
        k: getattr(expected, k) for k in asdict(actual.gates)
    }
    assert actual.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize(
    "case,mode",
    [(CASES[i], "supplied") for i in (0, 1, 2, 3, 5)]
    + [(CASES[i], "default") for i in (0, 2)],
)
@pytest.mark.parametrize("inverse", (False, True))
def test_original_full_complex_state_every_history_and_clean_storage(
    case, mode, inverse
):
    _, source, _, op, _, p, _, w, _, _, _, _ = fixture(case, mode, inverse, precision=1)
    scratch = len(p) + len(w)
    logical = op.required_qubits - scratch
    torch.manual_seed(79)
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
    assert events <= 7
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, op.instructions, history),
            desired / sqrt(2**events),
            atol=5e-12,
            rtol=0,
        )


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "many",
        "mixed",
        "logical_alias",
        "work_alias",
        "short_work",
        "short_bits",
    ],
)
def test_extension_rejects_invalid_existing_control_or_storage(change):

    native, _, _, _, _, p, _, w, _, options, binders, _ = fixture(
        CASES[2], "supplied", False, precision=1
    )
    if change == "empty":
        options |= dict(
            controls=(),
            control_registers=(),
            control_workspace=(),
            control_classical_bits=(),
        )
    elif change == "many":
        options["existing_control"] = (2, 3)
    elif change == "mixed":
        options["control"] = (2,)
    elif change == "logical_alias":
        options["existing_control"] = options["system"]
    elif change == "work_alias":
        options["existing_control"] = w[:1]
    elif change == "short_work":
        options["control_workspace"] = p[:-1]
    elif change == "short_bits":
        options["control_classical_bits"] = ()
    with pytest.raises(ValueError):
        native.make_on(binders, **options)


def test_extension_model_never_requests_emission_and_rejects_ambiguous_modes(
    monkeypatch,
):
    from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
        LinearCombinationBlockEncoding,
    )
    from flagquantum.compiler._component_trace.resource_model import ComponentCost
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister

    native, _, _, _, _, _, _, _, _, _, _, _ = fixture(
        CASES[0], "supplied", False, precision=1
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Resource query requested emission")

    monkeypatch.setattr(LinearCombinationBlockEncoding, "make_on", forbidden)
    child = ComponentCost(
        "SELECT",
        native.system_bitsize
        + native.ancilla_bitsize
        + native.resource_bitsize
        + 1
        - native.prepare.junk_width,
    )
    prep = ComponentCost(
        "PREPARE", native.prepare.selection_width + native.prepare.junk_width
    )
    report = estimate_ftqc_resources(
        native.resource_model(
            child,
            prep,
            prep,
            existing_control=True,
            control_registers=(ControlRegister(1, (0,)),),
        )
    )
    assert report.emitted is None
    for overrides in (
        dict(existing_control=True),
        dict(existing_control=1),
        dict(
            existing_control=True,
            controlled=True,
            control_registers=(ControlRegister(1, (0,)),),
        ),
    ):
        with pytest.raises(ValueError):
            native.resource_model(child, prep, prep, **overrides)


def test_runnable_stacked_control_example(capsys):
    from examples.algorithms.linear_combination_controls import main

    main(stacked=True)
    assert "full complex inverse verified" in capsys.readouterr().out
