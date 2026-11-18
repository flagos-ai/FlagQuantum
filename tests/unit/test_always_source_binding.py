"""Capture complete Always wrappers, including sided outputs and measured erase."""

from collections import Counter
from dataclasses import replace
from hashlib import sha1
from itertools import product
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.primitives.always import Always
from flagquantum.compiler.registers.primitives.and_compute import (
    and_compute_ir,
    and_uncompute_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.bookkeeping import always

    for name, blob in (
        ("always.py", "905f79a2fef8cf14ffd3f09098d4d93ea356a250"),
        ("always_test.py", "2b712e48f42fdf79519c7a5493423815fe52872f"),
    ):
        raw = (Path(always.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_actual_factory_sided_capture_full_isometry_and_every_erase_branch() -> None:
    from qualtran.bloqs.bookkeeping.always import _ALWAYS_DOC, _always_and

    from tests.unit.test_register_tensor import native_signature

    source = _always_and.make()
    assert _ALWAYS_DOC.examples == (_always_and,)
    assert str(source) == str(source.subbloq)
    emitter = import_register_emitter(source, local_ports=("target",))
    emitter = replace(
        emitter,
        snapshots=(
            CircuitIR.from_json(emitter.snapshots[0].to_json()),
            CircuitIR.from_json(emitter.snapshots[1].to_json()),
        ),
    )
    assert emitter.requirements() == (0, 1) and emitter.binding_swaps == (0, 0)
    ports = {"ctrl": (2, 0), "target": (3,)}
    forward = emitter(ports, (), (7,), False)
    backward = emitter(ports, (), (7,), True)
    assert_source(forward, and_compute_ir(2, 0, 3))
    assert_source(backward, and_uncompute_ir(2, 0, 3, 7))
    wrapper = Always(
        lambda: forward, lambda: backward, native_signature(source.signature)
    )
    controls = ((1,),)
    passed, controlled = wrapper.controlled(controls)
    assert passed is controls and controlled is wrapper
    assert wrapper.adjoint().adjoint() == wrapper
    data = torch.tensor(
        [complex(i + 1, i - 3) for i in range(8)], dtype=torch.complex128
    )
    data /= data.norm()
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[::2] = data
    isometry = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    expected = isometry @ data.reshape(2, 2, 2).permute(2, 0, 1).reshape(4, 2)
    expected = expected.reshape((2,) * 4).permute(1, 3, 0, 2).flatten()
    circuit = Circuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(4, controlled.emit()))
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=3e-12, rtol=0)
    for outcome in (0, 1):
        erased = forced_branch(expected, wrapper.adjoint().emit(), (outcome,))
        torch.testing.assert_close(erased, initial / 2**0.5, atol=3e-12, rtol=0)
        assert torch.count_nonzero(erased.reshape(-1, 2)[:, 1]) == 0
    assert Counter(g.name for g in forward) == {
        "cx": 6,
        "h": 2,
        "t": 2,
        "tdg": 2,
        "s": 1,
    }
    assert Counter(g.name for g in backward) == {
        "h": 1,
        "measure": 1,
        "cz": 1,
        "reset": 1,
    }


@pytest.mark.parametrize("inverse", (False, True))
def test_actual_add_wrapper_source_stream_and_all_round_trip_histories(
    inverse: bool,
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.bookkeeping import Always as SourceAlways

    source = SourceAlways(Add(QUInt(2), QUInt(3)))
    emitter = import_register_emitter(source)
    assert emitter.requirements() == (2, 2) and emitter.binding_swaps == (0, 0)
    ports = {"a": (1, 0), "b": (4, 3, 2)}
    work = (6, 7)
    bits = (9, 10)
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
    ir = Circuit(8).append(FieldOperation(8, gates)).to_ir()
    assert ir.instructions == gates
    assert CircuitIR.from_json(ir.to_json()).instructions == gates
    values = torch.arange(1, 65, dtype=torch.float64)
    data = torch.complex(values, values % 7 - 3)
    data /= data.norm()
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[::4] = data
    matrix = torch.as_tensor(root.tensor_contract(), dtype=torch.complex128)
    order = (1, 0, 4, 3, 2, 5, 6, 7)
    reverse_order = tuple(order.index(q) for q in range(8))
    expected = matrix @ initial.reshape((2,) * 8).permute(order).reshape(32, 8)
    expected = expected.reshape((2,) * 8).permute(reverse_order).flatten()
    histories = tuple(product((0, 1), repeat=2))
    for first in histories:
        branch = forced_branch(initial, gates, first)
        torch.testing.assert_close(branch, expected / 2, atol=3e-12, rtol=0)
        assert torch.count_nonzero(branch.reshape(-1, 4)[:, 1:]) == 0
        for second in histories:
            restored = forced_branch(branch, other, second)
            torch.testing.assert_close(restored, initial / 4, atol=3e-12, rtol=0)
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
