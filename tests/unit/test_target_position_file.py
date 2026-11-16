"""Whole-file checks against the locked target-position dispatch assets."""

from collections import Counter
from dataclasses import replace
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.algorithms.target_position_catalog import (
    Pauli,
    TargetPositionExample,
    apply_z_to_odd_call_graph,
    apply_z_to_odd_example,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source() -> Any:
    import kickmix
    from qualtran.bloqs.multiplexers import apply_gate_to_lth_target

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    directory = Path(apply_gate_to_lth_target.__file__).parent
    for suffix, digest in (
        (".ipynb", "7b733722e6587cd31d805804d46e66f18f83a53a"),
        (".py", "77f70157ced3fcd8195951f16a34d16a8cfd709a"),
        ("_test.py", "79e770fc3b13e4cfd0e8c4ffa21a14a8f100bee6"),
    ):
        data = (directory / f"apply_gate_to_lth_target{suffix}").read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    return apply_gate_to_lth_target


def _label(node: Any) -> str:
    name = type(node).__name__
    if name == "And":
        return f"And({node.cv1},{node.cv2})" + ("†" if node.uncompute else "")
    if name == "Identity":
        return f"I({node.bitsize})"
    assert name in ("CNOT", "CZ")
    return name


def test_original_fixture_declarations_and_all_measurement_outcomes(
    source: Any,
) -> None:
    import cirq
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    op = apply_z_to_odd_example()
    original = source._apply_z_to_odd.make()
    assert [(r.name, r.bitsize, r.side.value) for r in op.signature] == [
        (r.name, r.bitsize, r.side.value) for r in original.signature
    ]
    assert (
        dict(op.signature[1].dtype.parameters)["bound"]
        == original.selection_regs[0].dtype.iteration_length
    )
    assert op.wire_symbols == cirq.circuit_diagram_info(original).wire_symbols
    _, children = original.call_graph(max_depth=1, generalizer=ignore_split_join)
    assert apply_z_to_odd_call_graph().leaf_counts == {
        _label(node): n for node, n in children.items()
    }
    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    forward = circuit.to_ir()
    work = tuple(range(9, op.qubit_count))
    expected = reference_arithmetic_gates(
        original,
        {"control": (0, 1), "selection": (2, 3, 4), "target": (5, 6, 7, 8)},
        work,
        dict(zip(work, range(len(work)), strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(forward.instructions, expected)
    model = estimate_component_resources(op.resource_model())
    original_cost = get_cost_value(original, QECGatesCost())
    assert (model.gates.and_bloq, model.gates.clifford, model.gates.measurement) == (
        original_cost.and_bloq,
        original_cost.clifford,
        original_cost.measurement,
    )
    assert model.peak_logical_qubits == get_cost_value(original, QubitCount()) == 12
    counts = Counter(g.name for g in forward.instructions)
    events = counts["measure"]
    assert (counts["t"] + counts["tdg"], counts["cx"], events) == (16, 27, 4)

    state = torch.zeros(1 << op.qubit_count, dtype=torch.complex128)
    expected_state = torch.zeros_like(state)
    for controls, selection, word in product(range(4), range(4), range(16)):
        address = ((controls << 7) | (selection << 4) | word) << len(work)
        value = torch.tensor(
            complex(address + 1, address % 13 - 6), dtype=torch.complex128
        )
        neg = (
            controls == 3
            and original.nth_gate(selection) == cirq.Z
            and word & (8 >> selection)
        )
        state[address] = value
        expected_state[address] = -value if neg else value
    norm = state.norm()
    state /= norm
    expected_state /= norm
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(state, forward.instructions, history),
            expected_state / 2 ** (events / 2),
            atol=2e-12,
            rtol=0,
        )
    circuit.append(replace(op, adjoint=True))
    sparse = {int(i): complex(state[i]) for i in state.nonzero().flatten()}
    actual = execute_sparse_register(circuit.to_ir(), sparse, seed=41)
    assert all(
        abs(actual.get(k, 0) - sparse.get(k, 0)) < 2e-12
        for k in actual.keys() | sparse.keys()
    )


@pytest.mark.parametrize("width,count", ((3, 5), (3, 7), (4, 5)))
@pytest.mark.parametrize("label", ("x", "z"))
def test_original_make_on_wire_order_and_grid(
    width: int, count: int, label: Pauli, source: Any
) -> None:
    import cirq
    from qualtran import BQUInt, QAny, Register

    def child(index: int) -> cirq.Gate:
        if label == "x":
            return cirq.X
        return cirq.Z if index % 2 else cirq.I

    controls = 1 if label == "x" else 2
    paulis: tuple[Pauli, ...] = tuple(
        label if label == "x" or i % 2 else "i" for i in range(count)
    )
    op = TargetPositionExample(width, controls, paulis)
    source_op = source.ApplyGateToLthQubit(
        Register("selection", BQUInt(width, count)),
        child,
        control_regs=Register("control", QAny(controls)),
    )
    data = controls + width + count
    wires = cirq.LineQubit.range(data)
    names = {
        "control": tuple(range(controls)),
        "selection": tuple(range(controls, controls + width)),
        "target": tuple(range(controls + width, data)),
    }
    named = {name: tuple(wires[i] for i in indices) for name, indices in names.items()}
    inferred = source.ApplyGateToLthQubit.make_on(nth_gate=child, **named)
    assert inferred.qubits == source_op.on_registers(**named).qubits == tuple(wires)
    assert op.wire_symbols == cirq.circuit_diagram_info(inferred.gate).wire_symbols
    assert inferred.gate.signature == source_op.signature

    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    ir = circuit.to_ir()
    work = tuple(range(data, op.qubit_count))
    expected = reference_arithmetic_gates(
        inferred.gate,
        names,
        work,
        dict(zip(work, range(len(work)), strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(ir.instructions, expected)
    in_state, out_state = {}, {}
    for i in range(count):
        target = 0 if label == "x" else (1 << count) - 1
        prefix = ((1 << controls) - 1) << (width + count) | i << count
        value = complex(i + 1, 1 - i)
        output = target ^ (1 << (count - 1 - i)) if label == "x" else target
        in_state[(prefix | target) << len(work)] = value
        out_state[(prefix | output) << len(work)] = (
            -value if label == "z" and i % 2 else value
        )
    norm = sum(abs(v) ** 2 for v in in_state.values()) ** 0.5
    inp, out = (
        {k: v / norm for k, v in state.items()} for state in (in_state, out_state)
    )
    observed = execute_sparse_register(ir, inp, seed=19)
    assert all(
        abs(observed.get(k, 0) - out.get(k, 0)) < 2e-12
        for k in observed.keys() | out.keys()
    )


def test_controlled_non_pauli_child_on_its_original_target(source: Any) -> None:
    import cirq
    from qualtran import BQUInt, Register

    from flagquantum.compiler.registers.control.indexed_component import (
        indexed_target_component_ir,
    )
    from flagquantum.core.ir import Instruction

    original = source.ApplyGateToLthQubit(
        Register("selection", BQUInt(1, 2)),
        lambda i: cirq.H if i == 0 else cirq.X,
    )
    gates = indexed_target_component_ir(
        ((1,),),
        (2,),
        (3, 2),  # Deliberately use a nonmonotonic physical target bank.
        lambda i, predicate, target: (
            Instruction("ch" if i == 0 else "cx", (predicate, target)),
        ),
        controls=(0,),
        workspace=(4,),
        classical_bits=(0,),
    )
    expected = reference_arithmetic_gates(
        original,
        {"control": (0,), "selection": (1,), "target": (3, 2)},
        (4,),
        {4: 0},
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(gates, expected)

    state = torch.zeros(32, dtype=torch.complex128)
    coeff = torch.arange(1, 17, dtype=torch.float64)
    state[::2] = torch.complex(coeff, coeff - 5)
    state /= state.norm()
    ideal = torch.zeros_like(state)
    for word in range(16):
        amplitude = state[word << 1]
        if not word & 8:
            ideal[word << 1] += amplitude
        elif word & 4:
            ideal[(word ^ 2) << 1] += amplitude
        else:
            ideal[(word & ~1) << 1] += amplitude / 2**0.5
            ideal[(word | 1) << 1] += (-1 if word & 1 else 1) * amplitude / 2**0.5
    for outcome in (0, 1):
        torch.testing.assert_close(
            forced_branch(state, gates, (outcome,)),
            ideal / 2**0.5,
            rtol=0,
            atol=2e-13,
        )
