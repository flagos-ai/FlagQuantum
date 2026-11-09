"""Compare CNOT and Toffoli control adders to their pinned source graphs."""

from collections import Counter
from dataclasses import asdict
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler._component_trace.compound_control_resource import (
    compound_control_key,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    FixedTCall,
)
from flagquantum.compiler.registers.primitives.controlled_not import (
    cnot_with_controls_ir,
    toffoli_with_controls_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_control_predicate_leaf,
    walk_source_graph,
)
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("original_count", [1, 2])
@pytest.mark.parametrize("values", [(), (1,), (0,), (1, 1), (0, 1)])
def test_added_controls_match_source_adder_and_every_branch(
    original_count: int, values: tuple[int, ...]
) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran import BloqBuilder, CtrlSpec, QBit, Register
    from qualtran.bloqs.basic_gates import CNOT, Toffoli, XGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    n = len(values)
    original = CNOT() if original_count == 1 else Toffoli()
    ports_width = n + original_count + 1
    control_wires = tuple(reversed(range(n)))
    original_wires = tuple(reversed(range(n, ports_width - 1)))
    target = ports_width - 1
    work_count = max(0, n - 1) if original_count == 1 else n + 1 if n else 0
    work = tuple(range(ports_width + 1, ports_width + 1 + work_count))
    bits = tuple(range(20, 20 + work_count))
    emitter = cnot_with_controls_ir if original_count == 1 else toffoli_with_controls_ir
    gates = emitter(
        *original_wires,
        target,
        controls=control_wires,
        values=values,
        workspace=work,
        classical_bits=bits,
    )
    if not n:
        source = original
        ports = {"ctrl": original_wires, "target": (target,)}
    else:
        cvs = np.asarray(values[0] if n == 1 else values)
        _, add = original.get_ctrl_system(CtrlSpec(cvs=cvs))
        builder = BloqBuilder()
        extra = builder.add_register(Register("extra", QBit(), shape=cvs.shape))
        old = builder.add_register(
            Register("old", QBit(), shape=() if original_count == 1 else (2,))
        )
        target_port = builder.add_register(Register("target", QBit()))
        (extra_out,), (old_out, target_out) = add(
            builder, (extra,), {"ctrl": old, "target": target_port}
        )
        source = builder.finalize(extra=extra_out, old=old_out, target=target_out)
        ports = {"extra": control_wires, "old": original_wires, "target": (target,)}

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        if isinstance(bloq, (CNOT, Toffoli)):
            opcode = "cx" if isinstance(bloq, CNOT) else "ccx"
            return (Instruction(opcode, (*inputs["ctrl"], *inputs["target"])),), inputs
        return source_control_predicate_leaf(bloq, inputs, work, bits)

    reference, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert_source(gates, reference)
    initial_key = (
        CliffordCall("cx", 2) if original_count == 1 else FixedTCall("toffoli")
    )
    cost_key = compound_control_key(initial_key, values) if n else initial_key
    assert cost_key is not None
    model = query_qec_gate_cost(cost_key)
    assert {name: count for name, count in asdict(model).items() if count} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    assert get_cost_value(source, QubitCount()) == ports_width + work_count
    counts = Counter(gate.name for gate in gates)
    assert counts["measure"] == model.measurement == work_count
    assert counts["t"] + counts["tdg"] == 4 * work_count
    assert counts["ccx"] == model.toffoli

    width = ports_width + 1 + work_count
    state = torch.zeros(1 << width, dtype=torch.complex128)
    data = torch.tensor(
        [complex(i + 1, i % 5 - 2) for i in range(1 << (ports_width + 1))],
        dtype=torch.complex128,
    )
    state[:: 1 << work_count] = data / data.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    order = (*control_wires, *original_wires, target, ports_width, *work)
    inverse_order = tuple(order.index(wire) for wire in range(width))
    expected = (
        (
            matrix
            @ state.reshape((2,) * width).permute(order).reshape(1 << ports_width, -1)
        )
        .reshape((2,) * width)
        .permute(inverse_order)
        .reshape(-1)
    )
    scale = 2 ** (-work_count / 2)
    histories = tuple(product((0, 1), repeat=work_count))
    for first, second in product(histories, repeat=2):
        branch = forced_branch(state, gates, first)
        torch.testing.assert_close(branch, expected * scale, rtol=0, atol=2e-12)
        torch.testing.assert_close(
            forced_branch(branch, gates, second), state * scale**2, rtol=0, atol=2e-12
        )


def test_added_controls_reject_operand_aliases_and_incomplete_workspace() -> None:
    with pytest.raises(ValueError, match="distinct"):
        cnot_with_controls_ir(0, 1, controls=(0,), values=(1,))
    with pytest.raises(ValueError, match="distinct"):
        toffoli_with_controls_ir(
            0, 1, 2, controls=(1,), values=(0,), workspace=(3, 4), classical_bits=(0, 1)
        )
    with pytest.raises(ValueError, match="clean work"):
        toffoli_with_controls_ir(0, 1, 2, controls=(3,), values=(1,))
