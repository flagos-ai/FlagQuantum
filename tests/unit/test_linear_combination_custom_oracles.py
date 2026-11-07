"""Original supplied-oracle validation, execution and packed LCU resources."""

from dataclasses import asdict
from itertools import product
from math import sqrt

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
)
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.indexed_component import (
    indexed_component_ir,
    indexed_component_workspace,
)
from flagquantum.compiler.registers.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_prepare(width, junk):
    import numpy as np
    from attrs import frozen
    from qualtran import QAny, QBit, Register
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, SGate, TGate, XGate
    from qualtran.bloqs.state_preparation.black_box_prepare import (
        BlackBoxPrepare as Packed,
    )
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    @frozen
    class Prepare(PrepareOracle):
        @property
        def selection_registers(self):
            return (Register("address", QAny(width)),)

        @property
        def junk_registers(self):
            return (
                (Register("tag", QBit()), Register("history", QBit())) if junk else ()
            )

        @property
        def l1_norm_of_coeffs(self):
            return 7.0

        def build_composite_bloq(self, bb, address, **soqs):
            wires = bb.split(address)
            wires[0] = bb.add(Hadamard(), q=wires[0])
            wires[0] = bb.add(SGate(), q=wires[0])
            if junk:
                wires[0], soqs["tag"] = bb.add(
                    CNOT(), ctrl=wires[0], target=soqs["tag"]
                )
                soqs["history"] = bb.add(XGate(), q=soqs["history"])
                soqs["history"] = bb.add(TGate(), q=soqs["history"])
            return dict(address=bb.join(np.asarray(wires), dtype=QAny(width)), **soqs)

    return Packed(Prepare())


def fixture(mode, auxiliary, controlled, inverse):
    import numpy as np
    from attrs import evolve
    from qualtran.bloqs.basic_gates import XGate, ZGate
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect

    prepare_custom = mode != "select"
    select_custom = mode != "prepare"
    width = 2 if mode.startswith("both") else 1
    junk = 0 if mode == "both_no_junk" else 2
    calls = []

    def prepare(ports, work, bits, adjoint):
        assert not work and not bits
        calls.append(("prepare", adjoint))
        gates = (
            Instruction("h", ports["address"][:1]),
            Instruction("s", ports["address"][:1]),
        )
        if junk:
            gates += (
                Instruction("cx", (*ports["address"][:1], *ports["tag"])),
                Instruction("x", ports["history"]),
                Instruction("t", ports["history"]),
            )
        if not adjoint:
            return gates
        return tuple(
            Instruction({"s": "sdg", "t": "tdg"}.get(g.name, g.name), g.wires)
            for g in reversed(gates)
        )

    native_prepare = BlackBoxPrepare(
        (("address", width),),
        (("tag", 1), ("history", 1)) if junk else (),
        prepare,
        7.0,
    )
    anc = res = int(auxiliary)
    blocks = tuple(
        UnitaryBlockEncoding(
            (("q", 1),),
            lambda p, w, b, a: (),
            ancilla_bitsize=anc if i else 0,
            resource_bitsize=res if i else 0,
        )
        for i in range(2)
    )
    sources = tuple(
        evolve(
            Unitary(XGate() if i == 0 else ZGate()),
            ancilla_bitsize=anc if i else 0,
            resource_bitsize=res if i else 0,
        )
        for i in range(2)
    )
    target_width = 1 + anc + res
    labels = tuple("z" if i % 2 == 0 else "x" for i in range(1 << width))

    def emit_select(ports, work, bits, adjoint):
        calls.append(("select", adjoint))
        return indexed_component_ir(
            (ports["choice"],),
            (len(labels),),
            lambda i, p: (Instruction("c" + labels[i], (p, ports["system"][0])),),
            controls=ports.get("control", ()),
            workspace=work,
            classical_bits=bits,
            adjoint=adjoint,
        )

    def controlled_select(value, ports, work, bits, adjoint):
        assert value == 1
        return emit_select(ports, work, bits, adjoint)

    native_select = SelectBlockEncoding(
        (("choice", width),), (("system", target_width),), emit_select, native_prepare
    )
    selected = BlackBoxSelect(
        ApplyLthBloq(
            np.asarray(
                [
                    evolve(
                        Unitary(ZGate() if name == "z" else XGate()),
                        ancilla_bitsize=anc,
                        resource_bitsize=res,
                    )
                    for name in labels
                ],
                dtype=object,
            )
        )
    )
    native = LinearCombinationBlockEncoding(
        blocks,
        (1.0, -3.0),
        1,
        prepare_override=native_prepare if prepare_custom else None,
        select_override=native_select if select_custom else None,
        controlled_select=controlled_select if select_custom else None,
    )
    base = LinearCombination(
        sources,
        (1.0, -3.0),
        1,
        prepare=source_prepare(width, junk) if prepare_custom else None,
        select=selected if select_custom else None,
    )
    original = evolve(base, is_controlled=True) if controlled else base
    if inverse:
        original = original.adjoint()
    control = (0,) if controlled else ()
    system = (int(controlled),)
    start = int(controlled) + 1
    ancilla = tuple(range(start, start + native.ancilla_bitsize))
    start += len(ancilla)
    resource = tuple(range(start, start + native.resource_bitsize))
    start += len(resource)
    pw, ps = (
        (0, 0)
        if prepare_custom
        else native.coefficient_preparation.workspace_requirements
    )
    tw = indexed_component_workspace((width,), int(controlled))
    work = tuple(range(start, start + max(pw, tw)))
    bits = tuple(range(20, 20 + len(work)))

    def binder(name):
        def bind(predicate, ports, work, bits, adjoint):
            calls.append(("child", adjoint))
            return FieldOperation(
                max(predicate, *ports["system"]) + 1,
                (Instruction("c" + name, (predicate, *ports["system"])),),
            )

        return bind

    binders = () if select_custom else (binder("x"), binder("z"))
    options = dict(
        system=system,
        ancilla=ancilla,
        resource=resource,
        control=control,
        prepare_workspace=work[:pw],
        prepare_classical_bits=bits[:ps],
        traversal_workspace=work[:tw],
        traversal_classical_bits=bits[:tw],
        workspaces=() if select_custom else ((), ()),
        classical_bits=() if select_custom else ((), ()),
        adjoint=inverse,
    )
    op = native.make_on(binders, **options)
    ports = dict(system=system, ancilla=ancilla)
    if resource:
        ports["resource"] = resource
    if control:
        ports["ctrl"] = control
    prep_cost = (
        ComponentCost(
            "SuppliedPrepare",
            width + junk,
            GateCost(t=int(bool(junk)), clifford=2 + 2 * int(bool(junk))),
        )
        if prepare_custom
        else native.coefficient_preparation.resource_model()
    )
    if select_custom:
        selected_cost = indexed_component_cost(
            (width,),
            (len(labels),),
            (
                ComponentCost(
                    "ControlledUnitary", target_width + 1, GateCost(clifford=1)
                ),
            )
            * len(labels),
            target_width=target_width,
            control_count=int(controlled),
            adjoint=inverse,
        )
    else:
        selected_cost = native.select.resource_model(
            tuple(
                ComponentCost(
                    "CX/CZ",
                    2 + b.ancilla_bitsize + b.resource_bitsize,
                    GateCost(clifford=1),
                )
                for b in blocks
            ),
            controlled=controlled,
            adjoint=inverse,
        )
    model = native.resource_model(
        selected_cost, prep_cost, prep_cost, controlled=controlled, adjoint=inverse
    )
    return native, base, original, op, ports, work, bits, model, calls, options, binders


@pytest.mark.parametrize("mode", ["prepare", "select", "both", "both_no_junk"])
@pytest.mark.parametrize(
    "auxiliary,controlled,inverse", tuple(product((False, True), repeat=3))
)
def test_source_override_gates_metadata_and_resources(
    mode, auxiliary, controlled, inverse
):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    native, base, original, op, ports, work, bits, model, calls, _, _ = fixture(
        mode, auxiliary, controlled, inverse
    )
    for attribute in ("alpha", "epsilon", "ancilla_bitsize", "resource_bitsize"):
        assert getattr(native, attribute) == getattr(base, attribute)
    if mode != "select":
        assert calls[0] == ("prepare", False) and calls[-1] == ("prepare", True)
    if mode != "prepare":
        assert ("select", inverse) in calls and not any(
            name == "child" for name, _ in calls
        )
    stream = reference_arithmetic_gates(
        original.decompose_bloq(),
        ports,
        work,
        dict(zip(work, bits, strict=True)),
        cirq_style=True,
        preserve_components=True,
        keep_pauli_leaves=True,
        keep_exchange_leaves=True,
    )
    assert_source_dependencies(op.instructions, stream)
    cost = estimate_ftqc_resources(model).logical
    expected = get_cost_value(original, QECGatesCost())
    assert asdict(cost.gates) == {
        name: getattr(expected, name) for name in asdict(cost.gates)
    }
    assert cost.peak_logical_qubits == get_cost_value(original, QubitCount())


@pytest.mark.parametrize("mode", ["prepare", "select", "both", "both_no_junk"])
@pytest.mark.parametrize("controlled,inverse", tuple(product((False, True), repeat=2)))
def test_all_complex_logical_inputs_measurement_branches_and_work(
    mode, controlled, inverse
):
    _, _, original, op, _, work, _, _, _, _, _ = fixture(
        mode, True, controlled, inverse
    )
    logical = op.required_qubits - len(work)
    torch.manual_seed(41)
    psi = torch.randn(1 << logical, dtype=torch.complex128)
    psi /= psi.norm()
    initial = torch.zeros(1 << op.required_qubits, dtype=torch.complex128)
    desired = torch.zeros_like(initial)
    initial.reshape(-1, 1 << len(work))[:, 0] = psi
    desired.reshape(-1, 1 << len(work))[:, 0] = (
        torch.as_tensor(original.tensor_contract()) @ psi
    )
    circuit = DynamicCircuit(op.required_qubits)
    circuit.append(op)
    events = sum(g.name == "measure" for g in op.instructions)
    assert events <= 4
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, op.instructions, history),
            desired / sqrt(2**events),
            atol=4e-12,
            rtol=0,
        )


@pytest.mark.parametrize(
    "case",
    ["prepare_width", "select_width", "both_width", "target_width", "target_name"],
)
def test_original_constructor_rejects_incompatible_oracles(case):
    import numpy as np
    from attrs import evolve
    from qualtran import QAny, Register
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect

    width = 2 if case in {"select_width", "both_width"} else 1
    source_select = BlackBoxSelect(
        ApplyLthBloq(np.asarray([Unitary(XGate())] * (1 << width), dtype=object))
    )
    native_prepare = BlackBoxPrepare((("address", 2),), (), lambda p, w, b, a: ())
    native_select = SelectBlockEncoding(
        (("choice", width),), (("system", 1),), lambda p, w, b, a: (), native_prepare
    )
    prepare_width = 3 if case == "both_width" else 2
    supplied_prepare = (
        source_prepare(prepare_width, 0)
        if case in {"prepare_width", "both_width"}
        else None
    )
    if case == "target_width":
        source_select = BlackBoxSelect(
            ApplyLthBloq(
                np.asarray(
                    [evolve(Unitary(XGate()), ancilla_bitsize=1)] * 2, dtype=object
                )
            )
        )
        native_select = SelectBlockEncoding(
            (("choice", 1),), (("system", 2),), lambda p, w, b, a: (), native_prepare
        )
    if case == "target_name":
        source_select = ApplyLthBloq(
            np.asarray([XGate(), XGate()], dtype=object),
            selection_regs=(Register("selection", QAny(1)),),
        )
        native_select = SelectBlockEncoding(
            (("choice", 1),), (("q", 1),), lambda p, w, b, a: (), native_prepare
        )
    block = UnitaryBlockEncoding((("q", 1),), lambda p, w, b, a: ())
    if case == "both_width":
        native_prepare = BlackBoxPrepare((("address", 3),), (), lambda p, w, b, a: ())
    with pytest.raises(ValueError):
        LinearCombination(
            (Unitary(XGate()),) * 2,
            (1.0, -3.0),
            1,
            prepare=supplied_prepare,
            select=None if case == "prepare_width" else source_select,
        )
    with pytest.raises(ValueError):
        LinearCombinationBlockEncoding(
            (block,) * 2,
            (1.0, -3.0),
            1,
            prepare_override=native_prepare if supplied_prepare is not None else None,
            select_override=None if case == "prepare_width" else native_select,
        )


@pytest.mark.parametrize("controlled,inverse", tuple(product((False, True), repeat=2)))
@pytest.mark.parametrize("mode", ["prepare", "both"])
def test_symbolic_precision_and_supplied_oracles_never_build_default(
    monkeypatch, controlled, inverse, mode
):
    from dataclasses import replace

    import sympy
    from attrs import evolve
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    native, base, _, op, ports, work, bits, model, _, options, binders = fixture(
        "both", True, controlled, inverse
    )
    precision = sympy.Symbol("mu", integer=True, positive=True)
    native = replace(native, coefficient_bits=precision)
    source = evolve(base, lambd_bits=precision, is_controlled=controlled)
    if inverse:
        source = source.adjoint()

    def forbidden(self):
        raise AssertionError(
            "Supplied oracle path requested the default coefficient factory"
        )

    monkeypatch.setattr(
        LinearCombinationBlockEncoding, "coefficient_preparation", property(forbidden)
    )
    current = native.make_on(binders, **options)
    assert current.instructions == op.instructions
    assert native.epsilon == base.epsilon
    assert native.prepare.normalization == base.prepare.l1_norm_of_coeffs == 7.0
    stream = reference_arithmetic_gates(
        source.decompose_bloq(),
        ports,
        work,
        dict(zip(work, bits, strict=True)),
        cirq_style=True,
        preserve_components=True,
        keep_pauli_leaves=True,
        keep_exchange_leaves=True,
    )
    assert_source_dependencies(current.instructions, stream)
    estimated = estimate_ftqc_resources(model).logical
    expected = get_cost_value(source, QECGatesCost())
    assert asdict(estimated.gates) == {
        name: getattr(expected, name) for name in asdict(estimated.gates)
    }


def test_original_default_factory_still_rejects_symbolic_precision():
    from dataclasses import replace

    import sympy
    from attrs import evolve
    from qualtran import DecomposeTypeError

    native, base, _, _, _, _, _, _, _, _, _ = fixture("both", False, False, False)
    precision = sympy.Symbol("mu", integer=True, positive=True)
    original = evolve(base, lambd_bits=precision, prepare=None, select=None)
    current = replace(
        native,
        coefficient_bits=precision,
        prepare_override=None,
        select_override=None,
        controlled_select=None,
    )
    with pytest.raises(DecomposeTypeError, match="symbolic"):
        _ = original.prepare
    with pytest.raises(ValueError, match="concrete coefficient precision"):
        _ = current.prepare


def test_custom_control_and_ownership_errors_are_explicit():
    from dataclasses import replace

    native, _, _, _, _, _, _, _, _, options, binders = fixture(
        "both", False, True, False
    )
    with pytest.raises(ValueError, match="positive-controlled emitter"):
        replace(native, controlled_select=None).make_on(**options)
    with pytest.raises(ValueError, match="child factories/work groups"):
        native.make_on((lambda *args: None,), **options)


def test_runnable_custom_oracle_example(capsys):
    from examples.algorithms.linear_combination_custom_oracles import main

    main()
    assert "full complex inverse verified" in capsys.readouterr().out


def test_original_zero_ancilla_decomposition_boundary():
    from attrs import frozen
    from qualtran import QAny, Register
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
    from qualtran.bloqs.multiplexers.select_base import SelectOracle
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    @frozen
    class IdentitySelect(SelectOracle):
        @property
        def selection_registers(self):
            return ()

        @property
        def control_registers(self):
            return ()

        @property
        def target_registers(self):
            return (Register("system", QAny(1)),)

        def build_composite_bloq(self, bb, **soqs):
            return soqs

    original = LinearCombination(
        (Unitary(XGate()),) * 2,
        (1.0, 1.0),
        1,
        prepare=source_prepare(0, 0),
        select=BlackBoxSelect(IdentitySelect()),
    )
    assert original.ancilla_bitsize == 0
    for query in (
        lambda: original.decompose_bloq(),
        lambda: get_cost_value(original, QECGatesCost()),
    ):
        with pytest.raises(TypeError, match="ancilla"):
            query()

    preparation = BlackBoxPrepare((("address", 0),), (), lambda p, w, b, a: ())
    selection = SelectBlockEncoding(
        (("choice", 0),), (("system", 1),), lambda p, w, b, a: (), preparation
    )
    block = UnitaryBlockEncoding((("q", 1),), lambda p, w, b, a: ())
    native = LinearCombinationBlockEncoding(
        (block,) * 2,
        (1.0, 1.0),
        1,
        prepare_override=preparation,
        select_override=selection,
    )
    assert native.ancilla_bitsize == 0
    with pytest.raises(ValueError, match="nonempty ancilla"):
        native.make_on(
            system=(0,),
            ancilla=(),
            resource=(),
            prepare_workspace=(),
            prepare_classical_bits=(),
            traversal_workspace=(),
            traversal_classical_bits=(),
        )
    with pytest.raises(ValueError, match="nonempty ancilla"):
        native.resource_model(*(ComponentCost("opaque", 1),) * 3)


@pytest.mark.parametrize("mode", ["prepare", "both"])
@pytest.mark.parametrize("precision", [0, -3])
def test_unused_precision_is_not_validated_by_supplied_prepare_path(mode, precision):
    from dataclasses import replace

    from attrs import evolve

    native, base, _, op, ports, work, bits, _, _, options, binders = fixture(
        mode, False, False, False
    )
    current = replace(native, coefficient_bits=precision)
    original = evolve(base, lambd_bits=precision)
    actual = current.make_on(binders, **options)
    assert actual.instructions == op.instructions
    stream = reference_arithmetic_gates(
        original.decompose_bloq(),
        ports,
        work,
        dict(zip(work, bits, strict=True)),
        cirq_style=True,
        preserve_components=True,
        keep_pauli_leaves=True,
        keep_exchange_leaves=True,
    )
    assert_source_dependencies(actual.instructions, stream)
    assert current.epsilon == original.epsilon
