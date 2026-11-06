"""Verify source LCU block composition, specialized controls and clean work."""

import pytest

from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.lcu_block_encoding import LCUBlockEncoding
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
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


@pytest.mark.parametrize("control_value", [None, 0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.ftqc_reference
def test_lcu_block_source_gates_control_and_all_branches(control_value, adjoint):
    pytest.importorskip("qualtran")
    import cirq
    from attrs import frozen
    from qualtran import Adjoint, QBit, Register
    from qualtran.bloqs.basic_gates import Hadamard, SGate, XGate
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        LCUBlockEncoding as SourceLCU,
    )
    from qualtran.bloqs.multiplexers.select_pauli_lcu import (
        SelectPauliLCU as SourceSelect,
    )
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    @frozen
    class Prepare(PrepareOracle):
        register: Register

        @property
        def selection_registers(self):
            return (self.register,)

        @property
        def junk_registers(self):
            return (Register("junk", QBit()),)

        @property
        def l1_norm_of_coeffs(self):
            return 3.0

        def build_composite_bloq(self, bb, selection, junk):
            wires = bb.split(selection)
            wires[0] = bb.add(Hadamard(), q=wires[0])
            wires[0] = bb.add(SGate(), q=wires[0])
            return {
                "selection": bb.join(wires, dtype=self.register.dtype),
                "junk": bb.add(XGate(), q=junk),
            }

    words = ("X", "-Z", "Y")
    source_select = SourceSelect(
        2,
        1,
        tuple(
            cirq.DensePauliString(
                word.removeprefix("-"), coefficient=-1 if word.startswith("-") else 1
            )
            for word in words
        ),
    )
    source_prepare = Prepare(source_select.selection_registers[0])
    forward = SourceLCU(source_select, source_prepare, control_val=control_value)
    source = forward.adjoint() if adjoint else forward
    controlled = int(control_value is not None)
    control, address, junk, target = (
        (0,) if controlled else (),
        (controlled, controlled + 1),
        (controlled + 2,),
        (controlled + 3,),
    )
    data = controlled + 4
    workspace = tuple(range(data, data + 1 + controlled))
    bits = tuple(range(12, 12 + len(workspace)))
    direction = []

    def emit_prepare(ports, work, slots, inverse):
        assert ports == {"selection": address, "junk": junk} and not work and not slots
        direction.append(("prepare", inverse))
        first = address[:1]
        if inverse:
            return (
                Instruction("x", junk),
                Instruction("sdg", first),
                Instruction("h", first),
            )
        return Instruction("h", first), Instruction("s", first), Instruction("x", junk)

    def emit_select(value, ports, work, slots, inverse):
        assert value == control_value
        assert ports == (
            {"selection": address, "target": target, "control": control}
            if controlled
            else {"selection": address, "target": target}
        )
        direction.append(("select", inverse))
        return (
            SelectPauliLCU(2, 1, words, value)
            .make_bounded_on(
                selection=ports["selection"],
                target=ports["target"],
                control=ports.get("control", ()),
                workspace=work,
                classical_bits=slots,
            )
            .emit()
        )

    prepare = BlackBoxPrepare((("selection", 2),), (("junk", 1),), emit_prepare, 3.0)
    select = SelectBlockEncoding(
        (("selection", 2),),
        (("target", 1),),
        lambda ports, work, slots, inverse: emit_select(
            None, ports, work, slots, inverse
        ),
        prepare,
    )
    encoding = LCUBlockEncoding(
        select, control_value, emit_select if controlled else None
    )
    args = dict(
        selection=address,
        junk=junk,
        target=target,
        control=control,
        select_workspace=workspace,
        select_classical_bits=bits,
    )
    circuit = DynamicCircuit(data + len(workspace)).append(
        encoding.make_on(**args, adjoint=adjoint)
    )
    actual = circuit.to_ir().instructions
    assert direction == [("prepare", False), ("select", adjoint), ("prepare", True)]
    inverse = encoding.make_on(**args, adjoint=not adjoint).instructions
    assert (
        encoding.ancilla_bitsize,
        encoding.resource_bitsize,
        encoding.system_bitsize,
    ) == (forward.ancilla_bitsize, forward.resource_bitsize, forward.system_bitsize)
    assert encoding.alpha == forward.alpha and encoding.epsilon == forward.epsilon
    assert isinstance(forward.signal_state, PrepareIdentity)
    assert encoding.signal_state.selection_layout == (("selection", 2),)
    assert encoding.signal_state.emit(address, (), (), ()) == ()
    stages = []

    def leaf(
        bloq: object, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        child = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if isinstance(child, Prepare):
            stages.append(("prepare", isinstance(bloq, Adjoint)))
            return (
                reference_arithmetic_gates(
                    bloq, inputs, (), {}, keep_exchange_leaves=True
                ),
                inputs,
            )
        if isinstance(child, SourceSelect):
            stages.append(("select", child.control_val))
            return (
                reference_arithmetic_gates(
                    bloq,
                    inputs,
                    workspace,
                    dict(zip(workspace, bits, strict=True)),
                    cirq_style=True,
                    keep_pauli_leaves=True,
                ),
                inputs,
            )
        return None

    ports = {"selection": address, "target": target, "junk": junk}
    if control:
        ports["ctrl"] = control
    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert stages == [("prepare", False), ("select", control_value), ("prepare", True)]
    assert_source_dependencies(actual, expected)
    assert_controlled_exchange_branches(actual, expected, inverse, data, workspace)

    select_cost = (
        SelectPauliLCU(2, 1, words, control_value)
        .make_bounded_on(
            selection=address,
            target=target,
            control=control,
            workspace=workspace,
            classical_bits=bits,
        )
        .resource_model()
    )
    prepare_cost = ComponentCost("Prepare", 3, GateCost(clifford=3))
    model = encoding.resource_model(
        select_cost, prepare_cost, prepare_cost, adjoint=adjoint
    )
    assert_source_model(model, source)
    report = estimate_ftqc_resources(model, emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    assert report.emitted.t_count == 4 * (1 + controlled)
