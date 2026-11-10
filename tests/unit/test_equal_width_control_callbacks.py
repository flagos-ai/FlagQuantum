"""Equal-width output ports expose the original specialized-adjoint permutation."""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_adjoint import adjoint_component_hierarchy
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_control import (
    control_component_hierarchy,
    control_component_via_predicate,
)
from flagquantum.compiler.registers.component_expansion import (
    expand_component_calls_once,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.component_auto_partition import (
    auto_partition_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import SourcePorts
from tests.unit.test_component_auto_partition import source_partition
from tests.unit.test_component_control import add_h, add_phase
from tests.unit.test_component_control_selection import assert_gate_parameters, fixtures
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_hierarchy_adjoint import source_adjoint
from tests.unit.test_component_structured_control import normalize_trace
from tests.unit.test_packed_adjoint_control import (
    existing_adjoint_fixture,
    inverse_source_gates,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def compact_fixture(
    layers: int, leading: bool, inverse: bool, activation: int
) -> tuple[ProgramRecord, Any, Any]:
    """Expose one data bit around real pinned H/phase/control decompositions."""
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import Hadamard, ZPowGate
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    original, _, _, _ = fixtures("spectator", True)
    signature = Signature((Register("q", quantum_bit_type()),))
    builder = ComponentProgramBuilder(signature, original.functions[1:])
    observer = _source(signature)
    q = builder.call_dict("h", builder.inputs)["q"]
    q = builder.call_dict("p", {"q": q})["q"]
    raw = builder.finalize({"q": q})
    source_q = observer.builder.add(Hadamard(), q=observer.inputs["q"])
    source_q = observer.builder.add(ZPowGate(0.25), q=source_q)
    source_data = observer.builder.finalize(q=source_q)
    if activation:
        module = control_component_hierarchy(raw, {"h": add_h, "p": add_phase})
        child = Controlled(source_data, CtrlSpec())
    else:
        module = control_component_via_predicate(
            raw,
            {"h": add_h, "p": add_phase},
            (Register("ctrl", quantum_bit_type()),),
            (ControlRegister(1, (0,)),),
        )
        child = ControlledViaAnd(source_data, CtrlSpec(cvs=0))
    if inverse:
        module = adjoint_component_hierarchy(
            module,
            {
                "ch": "ch",
                "cp": "cp_inverse",
                "__fq_predicate_0_x": "__fq_predicate_0_x",
            },
        )
        child = source_adjoint(child.decompose_bloq())
    entry = "main"
    child_wrapper = child
    for i in range(layers):
        groups = (
            ("control", (("ctrl" if i == 0 else "control", 1),)),
            ("data", (("q" if i == 0 else "data", 1),)),
        )
        layout = AutoPartition(groups if leading else groups[::-1])
        module = auto_partition_component_program(
            module, layout, entry=entry, name=f"packed{i}"
        )
        child_wrapper = source_partition(child_wrapper, layout)
        entry = f"packed{i}"
    return module, child_wrapper, child


def inverse_expected(
    initial: torch.Tensor, old: int, new: int, swap_input: bool
) -> torch.Tensor:
    """Apply the ideal inverse, optionally after the diagnosed input SWAP."""
    expected = torch.zeros_like(initial)
    for added, control, data in product((0, 1), repeat=3):
        amplitude = initial[(added * 4 + control * 2 + data) * 4]
        if swap_input:
            control, data = data, control
        if (added, control) != (new, old):
            expected[(added * 4 + control * 2 + data) * 4] += amplitude
        else:
            for out in (0, 1):
                expected[(added * 4 + control * 2 + out) * 4] += (
                    amplitude
                    * np.exp(-1j * np.pi * data / 4)
                    * (-1 if data and out else 1)
                    / 2**0.5
                )
    return expected


@pytest.mark.parametrize("layers", (1, 2))
@pytest.mark.parametrize("leading", (False, True))
@pytest.mark.parametrize("old,new", tuple(product((0, 1), repeat=2)))
def test_actual_equal_width_callback_permutation(
    layers: int, leading: bool, old: int, new: int
) -> None:
    module, source, callback, child, adder, entry, _ = existing_adjoint_fixture(
        layers, leading, old, new, compact_fixture
    )
    signature = Signature(
        (
            Register("added", quantum_bit_type()),
            Register("control", RegisterType("opaque", 1)),
            Register("data", RegisterType("opaque", 1)),
        )
    )
    builder = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer = _source(signature)
    values = {r.name: builder.inputs[r.name] for r in signature if r.name != "added"}
    source_values = {
        r.name: observer.inputs[r.name] for r in signature if r.name != "added"
    }
    added, data = adder(builder, (builder.inputs["added"],), values)
    src_added, src_data = callback(
        observer.builder, (observer.inputs["added"],), source_values
    )
    names = ("control", "data") if leading else ("data", "control")
    caller = builder.finalize(
        {"added": added[0], **dict(zip(names, data, strict=True))}
    )
    composite = observer.builder.finalize(
        added=tuple(src_added)[0], **dict(zip(names, src_data, strict=True))
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
    bound = bind_register_program(
        caller, ((0,), (1,), (2,)), clean=(3, 4), entry="caller"
    )
    outputs = ((0,), (1,), (2,)) if leading else ((0,), (2,), (1,))
    assert bound.outputs == outputs
    bits = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    ports: SourcePorts = {"added": (0,), "control": (1,), "data": (2,)}
    assert_gate_parameters(
        bound.circuit.instructions,
        inverse_source_gates(
            composite,
            child,
            ports,
            (3, 4),
            bits,
            dict(zip(ports, outputs, strict=True)),
        ),
    )
    direct = bind_register_program(
        module, ((0,), (1,), (2,)), clean=(3, 4), entry=entry
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    assert Counter(g.name for g in direct.circuit.instructions) == Counter(
        g.name for g in bound.circuit.instructions
    )
    assert get_cost_value(source, QECGatesCost()) == get_cost_value(
        composite, QECGatesCost()
    )
    assert get_cost_value(composite, QECGatesCost()).total_t_count() == 31
    assert get_cost_value(composite, QubitCount()) == bound.circuit.n_wires == 5
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[::4] = torch.tensor(
        [complex(i + 1, 3 - 2 * i) for i in range(8)], dtype=torch.complex128
    )
    initial /= initial.norm()
    intended = inverse_expected(initial, old, new, False)
    actual_expected = inverse_expected(initial, old, new, not leading)
    if not leading:
        assert not torch.allclose(intended, actual_expected, atol=1e-10, rtol=0)
        assert direct.circuit != bound.circuit
    for history in product((0, 1), repeat=2):
        physical = forced_branch(initial, bound.circuit.instructions, history)
        axes = (*tuple(wire for port in bound.outputs for wire in port), 3, 4)
        logical = physical.reshape((2,) * 5).permute(axes).reshape(-1)
        torch.testing.assert_close(logical, actual_expected / 2, atol=3e-12, rtol=0)
        torch.testing.assert_close(
            forced_branch(initial, direct.circuit.instructions, history),
            intended / 2,
            atol=3e-12,
            rtol=0,
        )
        assert torch.count_nonzero(logical.reshape(-1, 4)[:, 1:]) == 0
    original = torch.as_tensor(composite.tensor_contract(), dtype=torch.complex128)
    # Source tensors order ports by their declared names, as does the caller.
    torch.testing.assert_close(
        original @ initial[::4], actual_expected[::4], atol=3e-12, rtol=0
    )
