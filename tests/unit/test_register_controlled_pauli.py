"""Verify CY/CZ second-level control order and both predicate lifetimes."""

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
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler.registers.primitives.pauli_gate import (
    controlled_pauli_with_controls_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_control_predicate_leaf,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("gate", ["y", "z"])
@pytest.mark.parametrize("values", [(), (0,), (1,), (0, 1), (0, 1, 0)])
def test_added_pauli_controls_match_locked_adder_and_every_branch(
    gate: Literal["y", "z"], values: tuple[int, ...]
) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran import BloqBuilder, CtrlSpec, QBit, Register
    from qualtran.bloqs.basic_gates import CZ, CYGate
    from qualtran.bloqs.mcmt import And
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    n = len(values)
    controls = tuple(reversed(range(n)))
    control, target = n + 1, n
    work = tuple(range(n + 3, 2 * n + 3))
    bits = tuple(range(40, 40 + n))
    opcode = f"c{gate}"
    original = CYGate() if gate == "y" else CZ()
    control_name, target_name = ("ctrl", "target") if gate == "y" else ("q1", "q2")
    emitted = controlled_pauli_with_controls_ir(
        gate,
        control,
        target,
        controls=controls,
        values=values,
        workspace=work,
        classical_bits=bits,
    )

    if not n:
        source = original
        ports = {control_name: (control,), target_name: (target,)}
    else:
        cvs = np.asarray(values[0] if n == 1 else values)
        _, add = original.get_ctrl_system(CtrlSpec(cvs=cvs))
        builder = BloqBuilder()
        extra = builder.add_register(Register("extra", QBit(), shape=cvs.shape))
        old = builder.add_register(Register("old", QBit()))
        data = builder.add_register(Register("data", QBit()))
        (extra_out,), (old_out, target_out) = add(
            builder, (extra,), {control_name: old, target_name: data}
        )
        source = builder.finalize(extra=extra_out, old=old_out, data=target_out)
        ports = {"extra": controls, "old": (control,), "data": (target,)}

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, (CYGate, CZ)):
            return (
                Instruction(opcode, (*inputs[control_name], *inputs[target_name])),
            ), inputs
        if isinstance(bloq, And):
            outputs = {"ctrl": inputs["ctrl"]}
            if not bloq.uncompute:
                outputs["target"] = work[-1:]
            gates = reference_arithmetic_gates(
                bloq,
                {**inputs, "target": work[-1:]},
                (),
                {work[-1]: bits[-1]},
                cirq_style=True,
            )
            return gates, outputs
        return source_control_predicate_leaf(bloq, inputs, work[:-1], bits[:-1])

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert_source(emitted, expected)
    # The inner predicate must be erased before the outer predicate.
    assert [
        g.metadata["classical_bit"] for g in emitted if g.name == "measure"
    ] == list(reversed(bits))
    root = CliffordCall("cy" if gate == "y" else "cz", 2)
    key = compound_control_key(root, values) if n else root
    assert key is not None
    model = query_qec_gate_cost(key)
    assert {name: count for name, count in asdict(model).items() if count} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    assert get_cost_value(source, QubitCount()) == 2 * n + 2
    counts = Counter(g.name for g in emitted)
    assert counts[opcode] == 1 + (n if gate == "z" else 0)
    assert sum(g.name == opcode and not g.metadata for g in emitted) == 1
    assert counts["measure"] == model.measurement == n
    assert counts["t"] + counts["tdg"] == 4 * model.and_bloq == 4 * n

    width = 2 * n + 3
    state = torch.zeros(1 << width, dtype=torch.complex128)
    data = torch.tensor(
        [complex(i + 1, i % 5 - 2) for i in range(1 << (n + 3))], dtype=torch.complex128
    )
    state[:: 1 << n] = data / data.norm()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    order = (*controls, control, target, n + 2, *work)
    inverse_order = tuple(order.index(wire) for wire in range(width))
    expected_state = (
        (matrix @ state.reshape((2,) * width).permute(order).reshape(1 << (n + 2), -1))
        .reshape((2,) * width)
        .permute(inverse_order)
        .reshape(-1)
    )
    scale = 2 ** (-n / 2)
    histories = tuple(product((0, 1), repeat=n))
    for first, second in product(histories, repeat=2):
        branch = forced_branch(state, emitted, first)
        torch.testing.assert_close(branch, expected_state * scale, rtol=0, atol=2e-12)
        torch.testing.assert_close(
            forced_branch(branch, emitted, second), state * scale**2, rtol=0, atol=2e-12
        )


def test_nested_control_validation_checks_all_work_and_slots() -> None:
    with pytest.raises(ValueError, match="clean work"):
        controlled_pauli_with_controls_ir("y", 0, 1, controls=(2,), values=(1,))
    with pytest.raises(ValueError, match="distinct"):
        controlled_pauli_with_controls_ir(
            "z",
            0,
            1,
            controls=(2, 3),
            values=(0, 1),
            workspace=(4, 5),
            classical_bits=(0, 0),
        )
