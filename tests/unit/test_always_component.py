"""Check executable Always calls separately from their delegated cost models."""

from dataclasses import asdict
from hashlib import sha1
from itertools import product
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_add_cost,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.control.component_bookkeeping import (
    call_with_unchanged_controls,
)
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
)
from flagquantum.compiler.registers.primitives.always import always_component_cost
from flagquantum.compiler.registers.primitives.always_component import (
    always_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_add_calls import executable_add
from tests.unit.test_component_call_boundaries import original
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source():
    from qualtran.bloqs.bookkeeping import always

    raw = Path(always.__file__).read_bytes()
    assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == (
        "905f79a2fef8cf14ffd3f09098d4d93ea356a250"
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_always_typed_add_with_inherited_control(reverse):
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.bookkeeping import Always
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    name = "add_adjoint" if reverse else "add"
    record = always_component_program(ProgramRecord(executable_add(reverse)), name)
    wrapper, *children = record.functions
    child = next(f for f in children if f.name == name)
    assert wrapper.registers == child.registers
    graph = program_component_call_graph(record, "always", max_depth=1)
    assert graph.direct_calls == {("always", name): 1}
    assert child.registers is not None
    signature = Signature((*child.registers, Register("ctrl", quantum_bit_type())))
    builder = ComponentProgramBuilder(signature, record.functions)
    controls = (builder.inputs["ctrl"],)
    ports = {key: builder.inputs[key] for key in ("a", "b")}
    passed, (a, b) = call_with_unchanged_controls(builder, "always", controls, ports)
    assert passed is controls
    program = builder.finalize({"a": a, "b": b, "ctrl": passed[0]})

    source_add = Add(QUInt(2), QUInt(3))
    source_child = source_add.adjoint() if reverse else source_add
    expected = Always(source_child)
    assert [instance.bloq for instance in expected.decompose_bloq().bloq_instances] == [
        source_child
    ]
    source = original(signature)
    controls = (source.inputs["ctrl"],)
    _, control_adder = expected.get_ctrl_system(CtrlSpec())
    passed, (a, b) = control_adder(
        source.builder, controls, {key: source.inputs[key] for key in ("a", "b")}
    )
    assert passed is controls
    source.builder.finalize(a=a, b=b, ctrl=passed[0])

    bound = bind_register_program(program, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    assert bound.outputs == ((0, 1), (2, 3, 4), (5,))
    gates = bound.circuit.instructions
    slots = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in gates
        if gate.name == "measure"
    }
    assert len(slots) == 2
    assert_source(
        gates,
        reference_arithmetic_gates(
            source_child,
            {"a": (0, 1), "b": (2, 3, 4)},
            (6, 7) if reverse else (7, 6),
            slots,
            cirq_style=not reverse,
        ),
    )
    assert Circuit(8).append(FieldOperation(8, gates)).to_ir().instructions == gates

    values = torch.arange(1, 65, dtype=torch.float64)
    phases = torch.complex(values, (values % 7) - 3)
    phases /= phases.norm()
    state = torch.zeros(256, dtype=phases.dtype)
    state[::4] = phases
    result = torch.zeros_like(state)
    sign = -1 if reverse else 1
    for i in range(64):
        left, right = i >> 4, (i >> 1) & 7
        target = (left << 4) | (((right + sign * left) & 7) << 1) | (i & 1)
        result[target << 2] = state[i << 2]
    for history in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(state, gates, history), result / 2, rtol=0, atol=1e-12
        )
    estimate = estimate_component_resources(
        always_component_cost(lambda: source_add_cost(2, 3))
    )
    assert {k: v for k, v in asdict(estimate.gates).items() if v} == get_cost_value(
        expected, QECGatesCost()
    ).asdict()
    assert estimate.peak_logical_qubits == get_cost_value(expected, QubitCount())


def test_always_wraps_atomic_allocate_and_propagates_cost_error():
    from qualtran import DecomposeTypeError, QAny
    from qualtran.bloqs.bookkeeping import Allocate, Always
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    dtype = opaque_register_type(3)
    word = allocate_word_function("lease", dtype, False)
    record = always_component_program(ProgramRecord((word,)), "lease")
    assert record.functions[0].registers == word.registers
    bound = bind_register_program(
        record, (), clean=(2, 0, 1), entry="always", allow_workspace_outputs=True
    )
    assert bound.outputs == ((2, 0, 1),)
    assert bound.circuit.instructions == ()
    assert [(e.kind, e.wires) for e in bound.events] == [("allocate", (2, 0, 1))]
    with pytest.raises(DecomposeTypeError, match="Allocate is atomic"):
        get_cost_value(Always(Allocate(QAny(3))), QECGatesCost())
    diagnostic = ValueError("child has no direct decomposition")

    def unavailable():
        raise diagnostic

    with pytest.raises(ValueError) as error:
        always_component_cost(unavailable)
    assert error.value is diagnostic
