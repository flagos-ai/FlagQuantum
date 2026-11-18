"""Actual packed wrappers bind source swaps and every measured Add history."""

from collections import Counter
from hashlib import sha1
from itertools import product
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    ControlledSwapCall,
)
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.primitives.controlled_swap import controlled_swap_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.bookkeeping import auto_partition

    for name, blob in (
        ("auto_partition.py", "93cddd9cbb1ab309f713316f066f16d93e5fd8b0"),
        ("auto_partition_test.py", "adb86ad3fbfabf4f2aab180d3eecc6ffb9daebf5"),
    ):
        raw = (Path(auto_partition.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("unused", (False, True))
def test_actual_factories_capture_full_complex_isometries_and_untouched_slices(
    unused: bool,
) -> None:
    from qualtran.bloqs.bookkeeping.auto_partition import (
        _AUTO_PARTITION_DOC,
        _auto_partition,
        _auto_partition_unused,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    factory = _auto_partition_unused if unused else _auto_partition
    source = factory.make()
    assert factory in _AUTO_PARTITION_DOC.examples
    assert str(source) == str(source.bloq)
    emitter = import_register_emitter(source)
    assert emitter.requirements() == (0, 0) and emitter.binding_swaps == (0, 0)
    logical = 6 if unused else 3
    wires = tuple(reversed(range(logical)))
    ports = {"x": wires[: 3 if unused else 2], "y": wires[3:4] if unused else wires[2:]}
    if unused:
        ports["z"] = wires[4:]
    target = ports["x"][1]
    control, right = ports["x"][0], ports["y"][0]
    expected_gates = (
        Instruction("ccx", (control, target, right)),
        Instruction("ccx", (control, right, target)),
        Instruction("ccx", (control, target, right)),
    )
    values = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    initial = torch.complex(values, values.remainder(7) - 3)
    initial /= initial.norm()
    order = (*wires, logical)
    reverse = tuple(order.index(q) for q in range(logical + 1))
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    expected = (
        (
            matrix
            @ initial.reshape((2,) * (logical + 1))
            .permute(order)
            .reshape(1 << logical, 2)
        )
        .reshape((2,) * (logical + 1))
        .permute(reverse)
        .flatten()
    )
    for inverse in (False, True):
        gates = emitter(ports, (), (), inverse)
        reference = (
            controlled_swap_ir(control, (target,), (right,))
            if inverse
            else expected_gates
        )
        assert gates == reference
        assert len(gates) == (17 if inverse else 3)
        snapshot = CircuitIR.from_json(emitter.snapshots[int(inverse)].to_json())
        assert len(snapshot.instructions) == (17 if inverse else 3)
        circuit = Circuit(logical + 1, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(logical + 1, gates))
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=3e-12
        )
        circuit.append(FieldOperation(logical + 1, emitter(ports, (), (), not inverse)))
        torch.testing.assert_close(
            circuit.state().flatten(), initial, rtol=0, atol=3e-12
        )
    for root in (source, source.adjoint()):
        assert get_cost_value(root, QubitCount()) == logical
        model = query_qec_gate_cost(ControlledSwapCall(1)).asdict()
        assert {
            name: count for name, count in model.items() if count != 0
        } == get_cost_value(root, QECGatesCost()).asdict()


@pytest.mark.parametrize("inverse", (False, True))
def test_measured_child_exact_stream_all_branches_and_local_lifetime(
    inverse: bool,
) -> None:
    from qualtran import QAny, QUInt, Register
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.bookkeeping import AutoPartition as SourceAutoPartition
    from qualtran.bloqs.bookkeeping.auto_partition import Unused

    source = SourceAutoPartition(
        Add(QUInt(2), QUInt(3)),
        (
            (Register("word", QAny(5)), ("b", "a")),
            (Register("idle", QAny(1)), (Unused(1),)),
        ),
    )
    layout = AutoPartition((("word", (("b", 3), ("a", 2))), ("idle", ((None, 1),))))
    ports = {"word": (4, 3, 2, 1, 0), "idle": (5,)}
    child_ports, unused = layout.bind(ports)
    assert child_ports == {"b": (4, 3, 2), "a": (1, 0)}
    assert layout.restore(child_ports, unused) == ports
    emitter = import_register_emitter(source)
    assert emitter.requirements() == (2, 2) and emitter.binding_swaps == (0, 0)
    work, bits = (7, 8), (9, 10)
    gates = emitter(ports, work, bits, inverse)
    other = emitter(ports, work, bits, not inverse)
    root = source.adjoint() if inverse else source
    slots = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    reference = reference_arithmetic_gates(
        root, ports, work, slots, cirq_style=not inverse
    )
    assert_source(gates, reference)
    ir = Circuit(9).append(FieldOperation(9, gates)).to_ir()
    assert CircuitIR.from_json(ir.to_json()).instructions == gates
    values = torch.arange(1, 129, dtype=torch.float64)
    data = torch.complex(values, values.remainder(11) - 5)
    data /= data.norm()
    initial = torch.zeros(512, dtype=torch.complex128)
    initial[::4] = data
    matrix = torch.as_tensor(root.tensor_contract(), dtype=torch.complex128)
    order = (4, 3, 2, 1, 0, 5, 6, 7, 8)
    reverse = tuple(order.index(q) for q in range(9))
    expected = (
        (matrix @ initial.reshape((2,) * 9).permute(order).reshape(64, 8))
        .reshape((2,) * 9)
        .permute(reverse)
        .flatten()
    )
    histories = tuple(product((0, 1), repeat=2))
    for first in histories:
        branch = forced_branch(initial, gates, first)
        torch.testing.assert_close(branch, expected / 2, rtol=0, atol=3e-12)
        assert torch.count_nonzero(branch.reshape(-1, 4)[:, 1:]) == 0
        for second in histories:
            restored = forced_branch(branch, other, second)
            torch.testing.assert_close(restored, initial / 4, rtol=0, atol=3e-12)
    assert Counter(g.name for g in gates) == {
        "cx": 20,
        "h": 6,
        "t": 4,
        "tdg": 4,
        "s": 2,
        "measure": 2,
        "cz": 2,
        "reset": 2,
    }
