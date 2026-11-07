"""Shared concrete control lifetimes versus pinned declarations and channels."""

from dataclasses import asdict
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.control_predicate import ControlRegister
from flagquantum.compiler.registers.controlled_component import (
    extend_controlled_component_ir,
)
from flagquantum.compiler.registers.controlled_component_resource_model import (
    controlled_component_cost,
    extended_controlled_component_cost,
)
from flagquantum.compiler.registers.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_control_predicate import CASES, control_fixture

pytestmark = pytest.mark.unit
CONTROL_CASES = (
    (("QBit", 1, (), (0,), "unsigned"),),
    (("QBit", 1, (), (1,), "unsigned"),),
    *CASES,
)


def assert_source(model, source, legacy):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    actual = estimate_component_resources(model)
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    assert asdict(actual.gates) == {
        key: getattr(expected, key) for key in asdict(actual.gates)
    }
    assert actual.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("case", CONTROL_CASES)
@pytest.mark.parametrize("legacy", [False, True])
def test_generic_control_model_matches_original(case, legacy):
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    spec, registers, _ = control_fixture(case)
    original = ControlledViaAnd(XGate(), spec)
    positive = ComponentCost("CNOT", 2, GateCost(clifford=1))
    native = controlled_component_cost(
        positive, registers, target_width=1, legacy_shims=legacy
    )
    assert_source(native, original, legacy)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("case", CONTROL_CASES)
@pytest.mark.parametrize("current,legacy", tuple(product((0, 1), (False, True))))
def test_extended_control_model_keeps_private_peak_out_of_signature(
    case, current, legacy
):
    import numpy as np
    from qualtran.bloqs.basic_gates import XGate, ZGate
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq

    spec, registers, _ = control_fixture(case)
    original = ApplyLthBloq(
        np.asarray([XGate(), ZGate()], dtype=object), control_val=current
    ).controlled(spec)
    # Positive qubits=4 includes its traversal scratch, but THRU payload is 2.
    positive = indexed_component_cost(
        (1,),
        (2,),
        (ComponentCost("CX/CZ", 2, GateCost(clifford=1)),) * 2,
        target_width=1,
        control_count=1,
        legacy_shims=legacy,
    )
    assert positive.qubits == 4
    native = extended_controlled_component_cost(
        positive,
        registers,
        current_control_value=current,
        target_width=2,
        legacy_shims=legacy,
    )
    assert_source(native, original, legacy)


@pytest.mark.parametrize(
    "values,current",
    [
        (values, current)
        for values in ((0,), (1,), (0, 1), (1, 0, 1))
        for current in (0, 1)
    ],
)
def test_extension_complete_complex_branches_clear_every_predicate(values, current):
    n = len(values)
    old = n
    target = n + 1
    work = tuple(range(n + 2, 2 * n + 2))
    bits = tuple(range(40, 40 + n))
    gates = extend_controlled_component_ir(
        (tuple(range(n)),),
        (ControlRegister(1, values),),
        (old, current),
        lambda predicate: (Instruction("cx", (predicate, target)),),
        work,
        bits,
    )
    initial = torch.zeros(1 << (2 * n + 2), dtype=torch.complex128)
    desired = torch.zeros_like(initial)
    torch.manual_seed(53)
    psi = torch.randn(1 << (n + 2), dtype=torch.complex128)
    psi /= psi.norm()
    active = sum(value << (n - 1 - i) for i, value in enumerate(values))
    for word, amplitude in enumerate(psi):
        initial[word << n] = amplitude
        enabled = word >> 2 == active and ((word >> 1) & 1) == current
        desired[(word ^ int(enabled)) << n] = amplitude
    assert sum(g.name == "measure" for g in gates) == n
    for history in product((0, 1), repeat=n):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            desired / sqrt(2**n),
            atol=3e-12,
            rtol=0,
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(current_control=(0, 1)),
        dict(current_control=(2, True)),
        dict(workspace=(4,)),
        dict(classical_bits=(0, 0)),
        dict(classical_bits=(-1, 1)),
        dict(controls=((0,),)),
    ],
)
def test_extension_invalid_storage_never_calls_body(change):
    def fail(predicate):
        pytest.fail("Invalid storage reached the controlled body")

    options = dict(
        controls=((0, 1),),
        registers=(ControlRegister(1, (0, 1)),),
        current_control=(2, 1),
        emit_positive=fail,
        workspace=(4, 5),
        classical_bits=(20, 21),
    )
    with pytest.raises(ValueError):
        extend_controlled_component_ir(**(options | change))


@pytest.mark.parametrize(
    "change",
    [
        dict(target_width=-1),
        dict(target_width=True),
        dict(legacy_shims=1),
        dict(registers=()),
        dict(positive=ComponentCost("missing", 0)),
    ],
)
def test_invalid_cost_contract_is_refused(change):
    options = dict(
        positive=ComponentCost("CNOT", 2, GateCost(clifford=1)),
        registers=(ControlRegister(1, (1,)),),
        target_width=1,
    )
    with pytest.raises(ValueError):
        controlled_component_cost(**(options | change))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("value", [0, 1])
def test_precontrolled_lcu_returned_root_runs_but_later_adder_retains_source_failure(
    value,
):
    from qualtran import BloqBuilder, CtrlSpec
    from qualtran.bloqs.basic_gates import XGate, ZGate
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    precontrolled = LinearCombination(
        (Unitary(XGate()), Unitary(ZGate())), (1.0, -3.0), 1, is_controlled=True
    )
    first = precontrolled.controlled(CtrlSpec(cvs=value))
    root, adder = first.get_ctrl_system(CtrlSpec(cvs=value))
    assert root.cvs == (value, value, 1)
    assert len(root.decompose_bloq().bloq_instances) == 3
    assert get_cost_value(root, QECGatesCost()).measurement == 5
    assert get_cost_value(root, QubitCount()) == 13
    bb, soqs = BloqBuilder.from_signature(first.signature, add_registers_allowed=True)
    extra = bb.add_register("extra", 1)
    with pytest.raises(TypeError, match="integer scalar arrays"):
        adder(bb, [extra], soqs)


@pytest.mark.ftqc_reference
def test_precontrolled_lcu_negative_extension_and_adjoint_order_are_distinct_source_paths():
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate, ZGate
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary

    from tests.unit.test_register_constant_addition import reference_arithmetic_gates

    precontrolled = LinearCombination(
        (Unitary(XGate()), Unitary(ZGate())), (1.0, -3.0), 1, is_controlled=True
    )
    after = precontrolled.controlled(CtrlSpec(cvs=0)).adjoint()
    before = precontrolled.adjoint().controlled(CtrlSpec(cvs=0))
    assert type(after).__name__ == "Adjoint"
    assert type(before).__name__ == "ControlledViaAnd"
    from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
        LinearCombinationBlockEncoding,
    )
    from flagquantum.algorithms.primitives.unitary_block_encoding import (
        UnitaryBlockEncoding,
    )
    from tests.unit.register_source_graph import (
        source_partition_leaf,
        walk_source_graph,
    )

    native = LinearCombinationBlockEncoding(
        (UnitaryBlockEncoding((("q", 1),), lambda p, w, b, a: ()),) * 2, (1.0, -3.0), 1
    )
    counts = []
    for source in (after, before):
        cursor = 0
        ports = {}
        for reg in source.signature:
            width = reg.total_bits()
            ports[reg.name] = tuple(range(cursor, cursor + width))
            cursor += width

        def leaf(bloq, inputs, desired, cursor=cursor):
            partition = source_partition_leaf(bloq, inputs, desired)
            if partition is not None:
                return partition
            base = bloq
            while hasattr(base, "subbloq"):
                base = base.subbloq
            kind = type(base).__name__
            if kind == "XGate":
                return (Instruction("x", inputs["q"]),), inputs
            if kind == "And":
                uncompute = bloq.uncompute
                binding = dict(inputs)
                if not uncompute:
                    binding["target"] = (cursor,)
                gates = reference_arithmetic_gates(
                    bloq,
                    binding,
                    (),
                    {cursor: 40},
                    cirq_style=True,
                    preserve_components=True,
                )
                return gates, dict(ctrl=inputs["ctrl"]) if uncompute else binding
            if kind == "BlackBoxPrepare":
                work = (cursor + 1, cursor + 2)
                declared = native.prepare.emit(
                    inputs["selection"],
                    inputs["junk"],
                    work,
                    (41,),
                    adjoint=type(bloq).__name__ == "Adjoint",
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
                        work,
                        slots,
                        cirq_style=True,
                        preserve_components=True,
                        keep_exchange_leaves=True,
                    ),
                    inputs,
                )
            if kind == "BlackBoxSelect":
                return (
                    reference_arithmetic_gates(
                        bloq.decompose_bloq(),
                        inputs,
                        (cursor + 1,),
                        {cursor + 1: 41},
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
        counts.append(sum(g.name == "x" and g.wires == (0,) for g in gates))
    assert counts == [4, 2]
    torch.testing.assert_close(
        torch.as_tensor(after.tensor_contract()),
        torch.as_tensor(before.tensor_contract()),
        atol=3e-12,
        rtol=0,
    )
