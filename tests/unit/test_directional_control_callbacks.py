"""Directional packed inputs preserve controlled-adjoint channels and ownership."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import (
    expand_component_calls,
    expand_component_calls_once,
)
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.component_auto_partition import (
    _partition_program,
    auto_partition_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import SourcePorts
from tests.unit.test_component_auto_partition import source_partition
from tests.unit.test_component_control_selection import assert_gate_parameters, fixtures
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace
from tests.unit.test_packed_adjoint_control import (
    existing_adjoint_fixture,
    inverse_source_gates,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def directional_fixture(
    layers: int, leading: bool, inverse: bool, activation: int
) -> tuple[ProgramRecord, Any, Any]:
    """Supply real inverse Partition leaves to the existing graph-adjoint owner."""
    assert layers == 1 and leading and not inverse
    _, _, native, originals = fixtures("spectator", True)
    module, _ = native[activation]
    child = originals[activation]
    layout = AutoPartition(
        (("ctrl", (("ctrl", 1),)), ("data", (("q", 1), ("spectator", 1)))),
        left_only=True,
    )
    source = source_partition(child, layout)
    module = auto_partition_component_program(module, layout, name="packed0")
    registers = (
        (Register("ctrl", quantum_bit_type()),),
        (Register("q", quantum_bit_type()), Register("spectator", quantum_bit_type())),
    )
    inverses = tuple(
        f
        for i, parts in enumerate(registers)
        for f in _partition_program(parts, f"__fq_partition_0_join{i}", True).functions
    )
    module = replace(module, functions=(*module.functions, *inverses))
    return module, source, child


@pytest.mark.parametrize("old,new", tuple(product((0, 1), repeat=2)))
@pytest.mark.parametrize("expanded", (False, True))
def test_directional_whole_existing_adjoint(old: int, new: int, expanded: bool) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    module, source, callback, child, adder, entry, forward = existing_adjoint_fixture(
        1, True, old, new, directional_fixture, "ctrl"
    )
    assert verify_program_record(module) == ()
    observer = _source(Signature([]))
    assert tuple(
        observer.signature(Signature(module.functions[0].registers or ()))
    ) == tuple(source.signature)
    assert normalize_trace(_native_trace(module, entry)) == normalize_trace(
        _source_trace(observer, source.decompose_bloq())
    )
    bit = quantum_bit_type()
    signature = Signature(
        (
            Register("added", bit),
            Register("ctrl", bit),
            Register("q", bit, side=RegisterSide.LEFT),
            Register("spectator", bit, side=RegisterSide.LEFT),
            Register("data", RegisterType("opaque", 2), side=RegisterSide.RIGHT),
        )
    )
    builder = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer = _source(signature)
    values = {
        r.name: builder.inputs[r.name] for r in signature.lefts() if r.name != "added"
    }
    source_values = {
        r.name: observer.inputs[r.name] for r in signature.lefts() if r.name != "added"
    }
    added, data = adder(builder, (builder.inputs["added"],), values)
    src_added, src_data = callback(
        observer.builder, (observer.inputs["added"],), source_values
    )
    caller = builder.finalize({"added": added[0], "ctrl": data[0], "data": data[1]})
    composite = observer.builder.finalize(
        **dict(
            zip(
                (r.name for r in signature.rights()),
                (*src_added, *src_data),
                strict=True,
            )
        )
    )
    caller, _ = expand_component_calls_once(
        caller,
        "caller",
        predicate=lambda _, op: op.callee is not None
        and op.callee.startswith("__fq_adjoint_control_"),
    )
    assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
        _source_trace(observer, composite)
    )
    assert tuple(values) == tuple(source_values)
    model = get_cost_value(composite, QECGatesCost())
    if expanded:
        caller, _ = expand_component_calls(caller, "caller")
    bound = bind_register_program(
        caller, ((0,), (1,), (2,), (3,)), clean=(4, 5), entry="caller"
    )
    assert bound.outputs == ((0,), (1,), (2, 3))
    assert not any(
        event.kind == "free" and set(event.wires) & set(range(4))
        for event in bound.events
    )
    bits = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    ports: SourcePorts = {"added": (0,), "ctrl": (1,), "q": (2,), "spectator": (3,)}
    outputs: SourcePorts = {"added": (0,), "ctrl": (1,), "data": (2, 3)}
    assert_gate_parameters(
        bound.circuit.instructions,
        inverse_source_gates(composite, child, ports, (4, 5), bits, outputs),
    )
    direct = bind_register_program(
        module, ((0,), (1,), (2,), (3,)), clean=(4, 5), entry=entry
    )
    assert direct.circuit == bound.circuit
    emitted = Counter(g.name for g in bound.circuit.instructions)
    assert emitted["t"] + emitted["tdg"] == 8
    assert emitted["measure"] == emitted["reset"] == model.measurement == 2
    assert model.total_t_count() == 31
    assert get_cost_value(composite, QubitCount()) == bound.circuit.n_wires == 6
    forward_bound = bind_register_program(
        forward, ((0,), (1,), (2, 3)), clean=(4, 5), entry=entry
    )
    assert forward_bound.outputs == ((0,), (1,), (2,), (3,))
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[::4] = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(16)], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for ctrl0, ctrl, q, spare in product((0, 1), repeat=4):
        index = (ctrl0 * 8 + ctrl * 4 + q * 2 + spare) * 4
        if (ctrl0, ctrl) != (new, old):
            expected[index] += initial[index]
        else:
            for out in (0, 1):
                target = (ctrl0 * 8 + ctrl * 4 + out * 2 + spare) * 4
                expected[target] += (
                    initial[index]
                    * np.exp(-1j * np.pi * q / 4)
                    * (-1 if q and out else 1)
                    / 2**0.5
                )
    for second in product((0, 1), repeat=2):
        state = forced_branch(initial, bound.circuit.instructions, second)
        torch.testing.assert_close(state, expected / 2, atol=3e-12, rtol=0)
        assert torch.count_nonzero(state.reshape(-1, 4)[:, 1:]) == 0
        for first in product((0, 1), repeat=2):
            computed = forced_branch(initial, forward_bound.circuit.instructions, first)
            torch.testing.assert_close(
                forced_branch(computed, bound.circuit.instructions, second),
                initial / 4,
                atol=3e-12,
                rtol=0,
            )


@pytest.mark.parametrize("old,new", tuple(product((0, 1), repeat=2)))
def test_directional_packed_inverse_keeps_append(old: int, new: int) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    module, _, _, _, _, entry, _ = existing_adjoint_fixture(
        1, True, old, new, directional_fixture, "ctrl"
    )
    bound = bind_register_program(
        module, ((0,), (1,), (2,), (3,)), clean=(4, 5), entry=entry
    )
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[::4] = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(16)], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = forced_branch(initial, bound.circuit.instructions, (0, 0)) * 2
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(6, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=29, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=3e-12, rtol=0
    )


def test_original_left_only_wrapper_cannot_be_nested_as_a_child() -> None:
    module, source, _ = directional_fixture(1, True, False, 1)
    layout = AutoPartition(
        (("word", (("ctrl", 1), ("data", 2))),),
        left_only=True,
    )
    with pytest.raises(ValueError, match="Length of regs and signature do not match"):
        source_partition(source, layout)
    with pytest.raises(ValueError, match="every child declaration on the left"):
        auto_partition_component_program(module, layout, entry="packed0", name="nested")


def test_original_renamed_control_has_no_matching_directional_output() -> None:
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        _MultiControlledFromSinglyControlled,
    )

    from flagquantum.compiler.registers.control.specialized_component_control import (
        multi_controlled_component_program,
    )

    _, _, native, originals = fixtures("spectator", True)
    layout = AutoPartition(
        (("control", (("ctrl", 1),)), ("data", (("q", 1), ("spectator", 1)))),
        left_only=True,
    )
    source = source_partition(originals[1], layout)
    with pytest.raises(KeyError, match="control"):
        _MultiControlledFromSinglyControlled((1, 0), source, "control").decompose_bloq()
    module = auto_partition_component_program(native[1][0], layout, name="packed0")
    with pytest.raises(ValueError, match="scalar input/output qubit"):
        multi_controlled_component_program(module, "control", (1, 0), "packed0")
