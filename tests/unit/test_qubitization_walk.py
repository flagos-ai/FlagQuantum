"""Check the locked walk's stage order, scalar and measured branches."""

from math import pi

import pytest

from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.lcu_block_encoding import LCUBlockEncoding
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.algorithms.primitives.qubitization_walk import QubitizationWalk
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control_predicate import ControlRegister
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_control_predicate_leaf,
    walk_source_graph,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import (
    assert_controlled_exchange_branches,
    assert_source_dependencies,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("shared", "lcu"), [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("pattern", [(), (1,), (0,), (0, 1), (1, 0, 1)])
@pytest.mark.ftqc_reference
def test_locked_walk_stages_scalar_and_all_branches(
    shared: bool, lcu: bool, adjoint: bool, pattern: tuple[int, ...]
) -> None:
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from attrs import frozen
    from qualtran import Adjoint, CtrlSpec, QAny, Register
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        LCUBlockEncoding as SourceLCU,
    )
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        SelectBlockEncoding as SourceBlock,
    )
    from qualtran.bloqs.mcmt import MultiControlZ
    from qualtran.bloqs.multiplexers.select_pauli_lcu import (
        SelectPauliLCU as SourceSelect,
    )
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    @frozen
    class Prepare(PrepareOracle):
        register: Register

        @property
        def selection_registers(self):
            return (self.register,)

        @property
        def junk_registers(self):
            return (Register("junk", QAny(1)),)

        @property
        def l1_norm_of_coeffs(self):
            return 3.0

        def build_composite_bloq(self, bb, **soqs):
            wires = bb.split(soqs[self.register.name])
            wires[0] = bb.add(Hadamard(), q=wires[0])
            return {
                self.register.name: bb.join(wires, dtype=self.register.dtype),
                "junk": bb.add(XGate(), q=soqs["junk"]),
            }

    words = ("X", "Z", "X")
    source_select = SourceSelect(2, 1, tuple(cirq.DensePauliString(w) for w in words))
    register = (
        source_select.selection_registers[0] if shared else Register("signal", QAny(1))
    )
    source_prepare = Prepare(register)
    source_block = SourceLCU if lcu else SourceBlock
    original = QubitizationWalkOperator(source_block(source_select, source_prepare))
    if lcu:
        assert isinstance(original.reflect.prepare_gate, PrepareIdentity)
    source = original.adjoint() if adjoint else original
    if pattern:
        source = source.controlled(
            CtrlSpec(cvs=pattern if len(pattern) > 1 else pattern[0])
        )
    signal_name = register.name
    ports = {"selection": (0, 1), "target": (2,)}
    if not shared:
        ports["signal"] = (3,)
    if lcu:
        ports["junk"] = (3,)
    data = 3 if shared and not lcu else 4
    signal_wires = ports[signal_name]
    outer = tuple(range(data, data + len(pattern)))
    if pattern:
        (control_port,) = {reg.name for reg in source.signature} - ports.keys()
        ports[control_port] = outer
    data += len(pattern)
    control_work = tuple(range(data, data + max(0, len(pattern) - 1)))
    control_bits = tuple(range(20, 20 + len(control_work)))
    phase_target = control_work[-1] if control_work else outer[0] if outer else data
    work_start = data + len(control_work)
    select_work = tuple(range(work_start, work_start + 1 + int(bool(pattern))))
    select_bits = tuple(range(5, 5 + len(select_work)))
    junk = ports["junk"] if lcu else (select_work[-1] + 1,)
    prefix = select_work[-1] + (1 if lcu else 2)
    reflection_work = (
        *(() if pattern else (phase_target,)),
        *((prefix,) if shared else ()),
    )
    work = tuple(range(data, prefix + int(shared)))
    select = SelectPauliLCU(2, 1, words)
    controlled_select = SelectPauliLCU(2, 1, words, 1)

    def emit_select(bound, workspace, bits, _inverse):
        return select.make_bounded_on(
            selection=bound["selection"],
            target=bound["target"],
            workspace=workspace,
            classical_bits=bits,
        ).emit()

    def emit_controlled(value, bound, workspace, bits, _inverse):
        assert value == 1
        return controlled_select.make_bounded_on(
            selection=bound["selection"],
            target=bound["target"],
            control=bound["control"],
            workspace=workspace,
            classical_bits=bits,
        ).emit()

    def emit_prepare(bound, _workspace, _bits, inverse):
        gates = (
            Instruction("h", bound[signal_name][:1]),
            Instruction("x", bound["junk"]),
        )
        return gates[::-1] if inverse else gates

    signal = BlackBoxPrepare(
        ((signal_name, len(signal_wires)),), (("junk", 1),), emit_prepare, 3.0
    )
    block = SelectBlockEncoding(
        (("selection", 2),), (("target", 1),), emit_select, signal
    )
    walk = QubitizationWalk(LCUBlockEncoding(block) if lcu else block)
    args = dict(
        selection={
            name: wires
            for name, wires in ports.items()
            if name in ("selection", "signal")
        },
        target=ports["target"],
        block_junk=junk if lcu else (),
        signal_junk=() if lcu else junk,
        select_workspace=select_work,
        select_classical_bits=select_bits,
        controls=(outer,) if outer else (),
        control_registers=(ControlRegister(1, pattern),) if outer else (),
        controlled_select=emit_controlled if pattern else None,
        control_workspace=control_work,
        control_classical_bits=control_bits,
        reflection_workspace=reflection_work,
        reflection_classical_bits=(9,) if shared else (),
    )
    circuit = DynamicCircuit(data + len(work)).append(
        walk.make_on(**args, adjoint=adjoint)
    )
    actual = circuit.to_ir().instructions
    inverse = walk.make_on(**args, adjoint=not adjoint).instructions
    assert walk.alpha == original.sum_of_lcu_coefficients
    assert set(ports) == {reg.name for reg in source.signature}
    stages = []

    def leaf(
        bloq: object, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if lcu and isinstance(bloq, (Prepare, Adjoint)):
            child = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
            if isinstance(child, Prepare):
                stages.append("unprepare" if isinstance(bloq, Adjoint) else "prepare")
                return reference_arithmetic_gates(bloq, inputs, (), {}), inputs
        predicate = source_control_predicate_leaf(
            bloq, inputs, control_work, control_bits
        )
        if predicate is not None:
            return predicate
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        if isinstance(bloq, SourceSelect):
            stages.append("select")
            return (
                reference_arithmetic_gates(
                    bloq,
                    inputs,
                    select_work,
                    dict(zip(select_work, select_bits, strict=True)),
                    cirq_style=True,
                    keep_pauli_leaves=True,
                ),
                inputs,
            )
        if not isinstance(bloq, ReflectionUsingPrepare):
            return None
        stages.append("reflection")
        operation = bloq.on_registers(
            **{
                name: np.array([cirq.LineQubit(w) for w in wires])
                for name, wires in inputs.items()
            }
        )
        ops = tuple(
            cirq.flatten_op_tree(
                cirq.decompose_once(
                    operation,
                    context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
                )
            )
        )
        if pattern:
            assert len(ops) == 4 and ops[2] == cirq.Z(cirq.LineQubit(phase_target))
            assert inputs["control"] == (phase_target,)
        else:
            assert len(ops) == 6 and ops[3].gate.coefficient == -1
            assert ops[1].gate == ops[4].gate == cirq.X
        mark_op = ops[1] if pattern else ops[2]
        mark = getattr(mark_op.gate, "bloq", mark_op.gate)
        assert isinstance(mark, MultiControlZ) and mark.concrete_cvs == (0,) * len(
            signal_wires
        )
        mark_work = reflection_work if pattern else reflection_work[1:]
        gates = reference_arithmetic_gates(
            mark,
            {"controls": signal_wires, "target": (phase_target,)},
            mark_work,
            dict(zip(mark_work, (9,) if shared else (), strict=True)),
        )
        prep_inputs = {signal_name: signal_wires, "junk": junk}
        before = (
            ()
            if lcu
            else reference_arithmetic_gates(
                source_prepare.adjoint(), prep_inputs, (), {}
            )
        )
        after = (
            ()
            if lcu
            else reference_arithmetic_gates(source_prepare, prep_inputs, (), {})
        )
        if pattern:
            return (*before, *gates, Instruction("z", (phase_target,)), *after), inputs
        flip = Instruction("x", (phase_target,))
        return (
            *before,
            flip,
            *gates,
            Instruction("phase", (phase_target,), {"theta": pi}),
            flip,
            *after,
        ), inputs

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    block_stages = ["prepare", "select", "unprepare"] if lcu else ["select"]
    assert stages == (
        ["reflection", *block_stages] if adjoint else [*block_stages, "reflection"]
    )
    assert outputs == ports
    assert_source_dependencies(actual, expected)
    assert_controlled_exchange_branches(actual, expected, inverse, data, work)
    selector = controlled_select if pattern else select
    select_cost = selector.make_bounded_on(
        selection=ports["selection"],
        target=ports["target"],
        control=(phase_target,) if pattern else (),
        workspace=select_work,
        classical_bits=select_bits,
    ).resource_model()
    prepare_cost = ComponentCost("Prepare", len(signal_wires) + 1, GateCost(clifford=2))
    model = walk.resource_model(
        select_cost,
        prepare_cost,
        prepare_cost,
        adjoint=adjoint,
        control_registers=args["control_registers"],
    )
    assert_source_model(model, source)
    report = estimate_ftqc_resources(model, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    assert report.emitted.t_count == 4 * (1 + int(shared) + len(pattern))


@pytest.mark.parametrize("width", range(6))
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.ftqc_reference
def test_reflection_source_model(width: int, controlled: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    from flagquantum.compiler.registers.prepare_reflection import (
        source_prepare_reflection_cost,
    )

    source = ReflectionUsingPrepare.reflection_around_zero(
        (width,) if width else (),
        global_phase=-1,
        control_val=1 if controlled else None,
    )
    identity = ComponentCost("Identity", width)
    model = source_prepare_reflection_cost(
        width, 0, identity, identity, controlled=controlled
    )
    assert_source_model(model, source)


def test_walk_with_no_logical_ports_preserves_clean_phase_wire() -> None:
    import torch

    from tests.unit.measurement_checks import forced_branch

    signal = BlackBoxPrepare((), (), lambda *_: (), normalization=1)
    walk = QubitizationWalk(SelectBlockEncoding((), (), lambda *_: (), signal))
    circuit = DynamicCircuit(1).append(
        walk.make_on(selection={}, target=(), reflection_workspace=(0,))
    )
    state = torch.tensor((1, 0), dtype=torch.complex128)
    torch.testing.assert_close(
        forced_branch(state, circuit.to_ir().instructions, ()),
        state,
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.ftqc_reference
def test_locked_walk_with_controlled_lcu_lacks_control_port(value: int) -> None:
    pytest.importorskip("qualtran")
    from attrs import evolve
    from qualtran.bloqs.block_encoding.lcu_block_encoding import _lcu_block
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )

    source = QubitizationWalkOperator(evolve(_lcu_block(), control_val=value))
    assert "ctrl" not in {reg.name for reg in source.signature}
    with pytest.raises(KeyError, match="ctrl"):
        source.decompose_bloq()
    signal = BlackBoxPrepare((), (), lambda *_: (), normalization=1)
    block = SelectBlockEncoding((), (), lambda *_: (), signal)
    controlled = LCUBlockEncoding(block, value, lambda *_: ())
    with pytest.raises(ValueError, match="omits the controlled LCU ctrl port"):
        QubitizationWalk(controlled)


def test_walk_rejects_control_aliases_and_missing_specialization() -> None:
    signal = BlackBoxPrepare((), (), lambda *_: (), normalization=1)
    walk = QubitizationWalk(
        SelectBlockEncoding((), (("target", 1),), lambda *_: (), signal)
    )
    with pytest.raises(ValueError, match="specialized SELECT"):
        walk.make_on(
            selection={},
            target=(0,),
            reflection_workspace=(),
            controls=((1,),),
            control_registers=(ControlRegister(1, (1,)),),
        )
    with pytest.raises(ValueError, match="distinct nonnegative"):
        walk.make_on(
            selection={},
            target=(0,),
            reflection_workspace=(),
            controls=((1, 2),),
            control_registers=(ControlRegister(1, (0, 1)),),
            controlled_select=lambda *_: (),
            control_workspace=(0,),
            control_classical_bits=(0,),
        )
