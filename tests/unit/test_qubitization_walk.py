"""Check the locked walk's stage order, scalar and measured branches."""

from math import pi

import pytest

from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.algorithms.primitives.qubitization_walk import QubitizationWalk
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import (
    assert_controlled_exchange_branches,
    assert_source_dependencies,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.ftqc_reference
def test_locked_walk_stages_scalar_and_all_branches(
    shared: bool, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from attrs import frozen
    from qualtran import QAny, Register
    from qualtran.bloqs.basic_gates import Hadamard, XGate
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
    original = QubitizationWalkOperator(SourceBlock(source_select, source_prepare))
    source = original.adjoint() if adjoint else original
    signal_name = register.name
    ports = {"selection": (0, 1), "target": (2,)}
    if not shared:
        ports["signal"] = (3,)
    data = 3 if shared else 4
    signal_wires = ports[signal_name]
    phase_target, junk = data, (data + 1,)
    reflection_work = (phase_target, data + 2) if shared else (phase_target,)
    # The block's measured predicate and reflection's phase target reuse a wire.
    select_work = (phase_target,)
    work = tuple(range(data, data + 2 + int(shared)))
    select = SelectPauliLCU(2, 1, words)

    def emit_select(bound, workspace, bits, _inverse):
        return select.make_bounded_on(
            selection=bound["selection"],
            target=bound["target"],
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
    walk = QubitizationWalk(block)
    args = dict(
        selection={name: wires for name, wires in ports.items() if name != "target"},
        target=ports["target"],
        signal_junk=junk,
        select_workspace=select_work,
        select_classical_bits=(5,),
        reflection_workspace=reflection_work,
        reflection_classical_bits=(9,) if shared else (),
    )
    circuit = DynamicCircuit(data + len(work)).append(
        walk.make_on(**args, adjoint=adjoint)
    )
    actual = circuit.to_ir().instructions
    inverse = walk.make_on(**args, adjoint=not adjoint).instructions
    assert walk.alpha == original.sum_of_lcu_coefficients
    assert set(ports) == {reg.name for reg in original.signature}
    stages = []

    def leaf(
        bloq: object, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, SourceSelect):
            stages.append("select")
            return (
                reference_arithmetic_gates(
                    bloq,
                    inputs,
                    select_work,
                    {phase_target: 5},
                    cirq_style=True,
                    keep_pauli_leaves=True,
                ),
                inputs,
            )
        if not isinstance(bloq, ReflectionUsingPrepare):
            return None
        stages.append("reflection")
        operation = bloq.on_registers(
            **{signal_name: np.array([cirq.LineQubit(w) for w in signal_wires])}
        )
        ops = tuple(
            cirq.flatten_op_tree(
                cirq.decompose_once(
                    operation,
                    context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
                )
            )
        )
        assert len(ops) == 6 and ops[3].gate.coefficient == -1
        assert ops[1].gate == ops[4].gate == cirq.X
        mark = getattr(ops[2].gate, "bloq", ops[2].gate)
        assert isinstance(mark, MultiControlZ) and mark.concrete_cvs == (0,) * len(
            signal_wires
        )
        mark_work = reflection_work[1:]
        gates = reference_arithmetic_gates(
            mark,
            {"controls": signal_wires, "target": (phase_target,)},
            mark_work,
            dict(zip(mark_work, (9,) if shared else (), strict=True)),
        )
        prep_inputs = {signal_name: signal_wires, "junk": junk}
        before = reference_arithmetic_gates(
            source_prepare.adjoint(), prep_inputs, (), {}
        )
        after = reference_arithmetic_gates(source_prepare, prep_inputs, (), {})
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
    assert stages == (["reflection", "select"] if adjoint else ["select", "reflection"])
    assert outputs == ports
    assert_source_dependencies(actual, expected)
    assert_controlled_exchange_branches(actual, expected, inverse, data, work)
    select_cost = select.make_bounded_on(
        selection=ports["selection"],
        target=ports["target"],
        workspace=select_work,
        classical_bits=(5,),
    ).resource_model()
    prepare_cost = ComponentCost("Prepare", len(signal_wires) + 1, GateCost(clifford=2))
    model = walk.resource_model(
        select_cost, prepare_cost, prepare_cost, adjoint=adjoint
    )
    assert_source_model(model, source)
    report = estimate_ftqc_resources(model, emitted_ir=circuit.to_ir())
    assert report.emitted is not None and report.emitted.t_count == (8 if shared else 4)


@pytest.mark.parametrize("width", range(6))
@pytest.mark.ftqc_reference
def test_uncontrolled_reflection_source_model(width: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    from flagquantum.compiler.registers.prepare_reflection import (
        source_uncontrolled_prepare_reflection_cost,
    )

    source = ReflectionUsingPrepare.reflection_around_zero(
        (width,) if width else (), global_phase=-1
    )
    identity = ComponentCost("Identity", width)
    model = source_uncontrolled_prepare_reflection_cost(width, 0, identity, identity)
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
