"""Check source fixed Pauli control dispatch and measured cleanup."""

from collections import Counter
from dataclasses import asdict
from itertools import product
from typing import Any, Literal

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
from flagquantum.compiler.registers.primitives.pauli_gate import pauli_gate_ir
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
@pytest.mark.parametrize("gate", ["x", "y", "z"])
@pytest.mark.parametrize("values", [(), (1,), (0,), (1, 1), (0, 1), (0, 1, 0)])
def test_pauli_dispatch_matches_source_and_all_histories(
    gate: Literal["x", "y", "z"], values: tuple[int, ...]
) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import (
        CNOT,
        CZ,
        CYGate,
        Toffoli,
        XGate,
        YGate,
        ZGate,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    n = len(values)
    wires = tuple(reversed(range(n)))
    work_count = 0 if gate == "x" and values == (1, 1) else max(0, n - 1)
    work = tuple(range(n + 2, n + 2 + work_count))
    bits = tuple(range(17, 17 + work_count))
    source_leaf = {"x": XGate(), "y": YGate(), "z": ZGate()}[gate]
    source = (
        source_leaf
        if not values
        else source_leaf.controlled(
            CtrlSpec(cvs=values[0] if n == 1 else np.asarray(values))
        )
    )
    emitted = pauli_gate_ir(
        gate, n, controls=wires, values=values, workspace=work, classical_bits=bits
    )

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, (XGate, YGate, ZGate)):
            opcode = {XGate: "x", YGate: "y", ZGate: "z"}[type(bloq)]
            return (Instruction(opcode, inputs["q"]),), inputs
        if isinstance(bloq, (CNOT, CYGate, Toffoli)):
            opcode = {CNOT: "cx", CYGate: "cy", Toffoli: "ccx"}[type(bloq)]
            return (Instruction(opcode, (*inputs["ctrl"], *inputs["target"])),), inputs
        if isinstance(bloq, CZ):
            return (Instruction("cz", (*inputs["q1"], *inputs["q2"])),), inputs
        return source_control_predicate_leaf(bloq, inputs, work, bits)

    if not values:
        ports = {"q": (n,)}
    elif isinstance(source, CZ):
        ports = {"q1": wires, "q2": (n,)}
    elif isinstance(source, (CNOT, CYGate, Toffoli)):
        ports = {"ctrl": wires, "target": (n,)}
    else:
        ports = {"ctrl": wires, "q": (n,)}
    reference, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert_source(emitted, reference)

    root = CliffordCall(gate, 1)
    key = compound_control_key(root, values) if values else root
    assert key is not None
    model = query_qec_gate_cost(key)
    assert {name: count for name, count in asdict(model).items() if count} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    assert get_cost_value(source, QubitCount()) == n + 1 + work_count
    counts = Counter(g.name for g in emitted)
    assert counts["measure"] == model.measurement == work_count
    assert counts["t"] + counts["tdg"] == 4 * work_count
    assert counts["ccx"] == model.toffoli

    width = n + 2 + work_count
    state = torch.zeros(1 << width, dtype=torch.complex128)
    data = torch.tensor(
        [complex(i + 1, i % 7 - 3) for i in range(1 << (n + 2))],
        dtype=torch.complex128,
    )
    state[:: 1 << work_count] = data / data.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    order = (*wires, n, n + 1, *work)
    inverse_order = tuple(order.index(wire) for wire in range(width))
    expected = (
        (matrix @ state.reshape((2,) * width).permute(order).reshape(1 << (n + 1), -1))
        .reshape((2,) * width)
        .permute(inverse_order)
        .reshape(-1)
    )
    scale = 2 ** (-work_count / 2)
    histories = tuple(product((0, 1), repeat=work_count))
    for first, second in product(histories, repeat=2):
        branch = forced_branch(state, emitted, first)
        torch.testing.assert_close(branch, expected * scale, rtol=0, atol=2e-12)
        torch.testing.assert_close(
            forced_branch(branch, emitted, second), state * scale**2, rtol=0, atol=2e-12
        )


def test_double_positive_x_specialization_takes_no_workspace() -> None:
    assert pauli_gate_ir("x", 0, controls=(3, 1), values=(1, 1)) == (
        Instruction("ccx", (3, 1, 0)),
    )
    assert compound_control_key(CliffordCall("x", 1), (1, 1)) == FixedTCall("toffoli")
    with pytest.raises(ValueError, match="no clean work"):
        pauli_gate_ir("x", 0, controls=(3, 1), values=(1, 1), workspace=(2,))
    with pytest.raises(ValueError, match="distinct"):
        pauli_gate_ir("x", 0, controls=(0, 1), values=(1, 1))
