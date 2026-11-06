"""Verify pinned fixed phases and every small measured control history."""

from collections import Counter
from dataclasses import asdict
from itertools import product
from math import pi
from typing import Any, Literal

import pytest
import torch

from flagquantum.compiler._component_trace.compound_control_resource import (
    compound_control_key,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    PhaseCall,
    ResourceCall,
)
from flagquantum.compiler.registers.phase_family import phase_family_ir
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
@pytest.mark.parametrize("gate", ["s", "t"])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("values", [(), (1,), (0,), (0, 1), (0, 1, 0)])
def test_fixed_phase_source_gates_amplitudes_and_resources(
    gate: Literal["s", "t"], adjoint: bool, values: tuple[int, ...]
) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    cirq = pytest.importorskip("cirq")
    from qualtran import CtrlSpec, DecomposeNotImplementedError
    from qualtran._infra.controlled import Controlled
    from qualtran.bloqs.basic_gates import SGate, TGate, XGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    n = len(values)
    target = n
    controls = tuple(reversed(range(n)))
    work = tuple(range(n + 2, n + 2 + max(n - 1, 0)))
    bits = tuple(range(30, 30 + len(work)))
    bound = dict(controls=controls, values=values, workspace=work, classical_bits=bits)
    source_leaf = (SGate if gate == "s" else TGate)(is_adjoint=adjoint)
    opcode = gate + "dg" if adjoint else gate
    theta = (-1 if adjoint else 1) * pi / (2 if gate == "s" else 4)
    gates = phase_family_ir(gate, target, adjoint=adjoint, **bound)
    inverse = phase_family_ir(gate, target, adjoint=not adjoint, **bound)
    key: ResourceCall = PhaseCall(gate, adjoint)
    if not n:
        source = source_leaf
        assert gates == (Instruction(opcode, (target,)),)
        wire = cirq.LineQubit(target)
        op, _ = source.as_cirq_op(qubit_manager=None, q=(wire,))
        assert op == ((cirq.S if gate == "s" else cirq.T) ** (-1 if adjoint else 1)).on(
            wire
        )
    else:
        spec = CtrlSpec(cvs=values[0] if n == 1 else np.asarray(values))
        source = source_leaf.controlled(spec)
        cost_key = compound_control_key(key, values)
        assert cost_key is not None
        key = cost_key

        def leaf(
            bloq: Any, inputs: SourcePorts, desired: SourcePorts
        ) -> SourceResult | None:
            if isinstance(bloq, XGate):
                return (Instruction("x", inputs["q"]),), inputs
            if isinstance(bloq, Controlled):
                assert bloq.subbloq == source_leaf and bloq.ctrl_spec == CtrlSpec()
                with pytest.raises(DecomposeNotImplementedError):
                    bloq.decompose_bloq()
                atomic = np.asarray(bloq.tensor_contract())
                np.testing.assert_allclose(
                    atomic, np.diag([1, 1, 1, np.exp(1j * theta)]), rtol=0, atol=2e-12
                )
                wires = (*inputs["ctrl"], *inputs["q"])
                return (Instruction("cphase", wires, {"theta": theta}),), inputs
            return source_control_predicate_leaf(bloq, inputs, work, bits)

        ports = {"ctrl": controls, "q": (target,)}
        reference, outputs = walk_source_graph(source, ports, ports, leaf)
        assert outputs == ports
        assert_source(gates, reference)
        assert [g for g in gates if g.name == "cphase"] == [
            g for g in reference if g.name == "cphase"
        ]
    model = query_qec_gate_cost(key)
    assert {name: count for name, count in asdict(model).items() if count} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    assert get_cost_value(source, QubitCount()) == n + 1 + len(work)
    counts = Counter(g.name for g in gates)
    assert counts["measure"] == len(work)
    assert counts["t"] + counts["tdg"] == 4 * len(work) + int(gate == "t" and not n)

    width = n + 2 + len(work)
    state = torch.zeros(1 << width, dtype=torch.complex128)
    data = torch.tensor([complex(i + 1, i % 3 - 2) for i in range(1 << (n + 2))])
    state[:: 1 << len(work)] = data / data.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    order = (*controls, target, n + 1, *work)
    inverse_order = tuple(order.index(wire) for wire in range(width))
    expected = (
        (matrix @ state.reshape((2,) * width).permute(order).reshape(1 << (n + 1), -1))
        .reshape((2,) * width)
        .permute(inverse_order)
        .reshape(-1)
    )
    scale = 2 ** (-len(work) / 2)
    histories = tuple(product((0, 1), repeat=len(work)))
    for first, second in product(histories, repeat=2):
        branch = forced_branch(state, gates, first)
        torch.testing.assert_close(branch, expected * scale, rtol=0, atol=2e-12)
        torch.testing.assert_close(
            forced_branch(branch, inverse, second), state * scale**2, rtol=0, atol=2e-12
        )
