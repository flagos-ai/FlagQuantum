"""Verify the pinned ControlledViaAnd fixtures and teaching graph."""

from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import torch

from examples.algorithms.controlled_via_and import ControlledHadamardExample
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.component_control import quantum_control_signature
from flagquantum.compiler.registers.control.controlled_component_resource_model import (
    controlled_component_cost,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.simulation.tensor_network.controlled_tensor import (
    controlled_register_tensor,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_control_predicate_leaf,
    walk_source_graph,
)
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def assert_reference_blobs() -> None:
    import kickmix
    from qualtran.bloqs.mcmt import controlled_via_and

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    path = Path(controlled_via_and.__file__).parent
    for name, digest in (
        ("controlled_via_and.py", "3947ddf86be37f4ff3ba58fcc316a8a64922f84f"),
        ("controlled_via_and_test.py", "9e60e3eccb50cfdbcac9054f9471f107062c6d7a"),
        ("controlled_via_and.ipynb", "825c66f74d650124a0db19f21b433d680b8d4072"),
    ):
        raw = (path / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


def _key(node: Any) -> str:
    name = type(node).__name__
    if name == "Adjoint":
        return _key(node.subbloq) + "†"
    if name == "And" and node.uncompute:
        return "And†"
    return name


@pytest.mark.parametrize("kind", ["bits", "integers", "nested"])
def test_complete_notebook_fixtures(
    kind: Literal["bits", "integers", "nested"],
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import CHadamard
    from qualtran.bloqs.mcmt.controlled_via_and import (
        _controlled_via_and_ints,
        _controlled_via_and_qbits,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = (
        _controlled_via_and_ints.make()
        if kind == "integers"
        else _controlled_via_and_qbits.make()
    )
    if kind == "nested":
        source = source.controlled(CtrlSpec(cvs=[1, 1]))
    example = ControlledHadamardExample(kind)
    native_signature = example.signature
    assert [(r.name, r.bitsize, r.shape, r.side.value) for r in native_signature] == [
        (r.name, r.bitsize, r.shape, r.side.value) for r in source.signature
    ]
    for depth in (1, None):
        graph = example.call_graph(depth=depth)
        original, leaves = source.call_graph(max_depth=depth)
        assert graph.direct_calls == {
            (_key(a), _key(b)): data["n"] for a, b, data in original.edges(data=True)
        }
        assert graph.leaf_counts == {
            _key(node): weight for node, weight in leaves.items()
        }

    n = sum(len(group.bits()) for group in example.groups)
    circuit = DynamicCircuit(2 * n).append(example)
    gates = circuit.to_ir().instructions
    work = tuple(range(n + 1, 2 * n))
    slots = tuple(range(n - 1))
    offset = 0
    ports: SourcePorts = {}
    for name, group in zip(source.ctrl_reg_names, example.groups, strict=True):
        size = len(group.bits())
        ports[name] = tuple(range(offset, offset + size))
        offset += size
    ports["q"] = (n,)

    def leaf(
        node: Any, inputs: SourcePorts, _desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(node, CHadamard):
            return (Instruction("ch", (*inputs["ctrl"], *inputs["target"])),), inputs
        return source_control_predicate_leaf(node, inputs, work, slots)

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert_source(gates, expected)
    model = get_cost_value(source, QECGatesCost())
    native_cost = estimate_component_resources(
        controlled_component_cost(
            ComponentCost("CHadamard", 2, GateCost(clifford=4, rotation=2)),
            example.groups,
            target_width=1,
        )
    )
    assert native_cost.gates.and_bloq == model.and_bloq == n - 1
    assert native_cost.gates.clifford == model.clifford
    assert (
        native_cost.peak_logical_qubits == get_cost_value(source, QubitCount()) == 2 * n
    )
    assert sum(g.name == "measure" for g in gates) == model.measurement == n - 1
    assert sum(g.name in ("t", "tdg") for g in gates) == 4 * (n - 1)

    if kind != "integers":
        weights = torch.arange(1, (1 << (n + 1)) + 1, dtype=torch.float64)
        ket = torch.complex(weights, weights.flip(0))
        ket /= ket.norm()
        initial = torch.zeros(1 << (2 * n), dtype=ket.dtype)
        initial[:: 1 << (n - 1)] = ket
        source_state = torch.as_tensor(
            source.tensor_contract().reshape(ket.numel(), -1)
        )
        expected_state = torch.zeros_like(initial)
        expected_state[:: 1 << (n - 1)] = source_state @ ket
        for history in product((0, 1), repeat=n - 1):
            torch.testing.assert_close(
                forced_branch(initial, gates, history),
                expected_state * 2 ** (-(n - 1) / 2),
                atol=2e-12,
                rtol=0,
            )


def test_original_dense_integer_tensor_matrix_cases() -> None:
    from qualtran import Controlled, CtrlSpec, QInt, QUInt
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    child = MatrixGate.random(2, random_state=np.random.RandomState(42))
    payload = Signature((Register("q", unsigned_register_type(2)),))
    cases = (
        ((QInt(4),), (signed_register_type(4),), ([1, -2],)),
        ((QUInt(4),), (unsigned_register_type(4),), ([2, 5],)),
        (
            (QInt(2), QUInt(2)),
            (signed_register_type(2), unsigned_register_type(2)),
            ([1, -2], [2, 3]),
        ),
    )
    for upstream, native, raw in cases:
        values = tuple(np.asarray(group) for group in raw)
        spec = CtrlSpec(upstream, cvs=values)
        original = ControlledViaAnd(child, spec)
        signature = quantum_control_signature(
            payload,
            tuple(
                (dtype, group.shape)
                for dtype, group in zip(native, values, strict=True)
            ),
        )
        controls = Signature(tuple(signature)[:-1])
        data = controlled_register_tensor(
            controls, payload, values, lambda: np.asarray(child.matrix)
        )
        expected = Controlled(child, spec).tensor_contract()
        np.testing.assert_allclose(
            data.reshape(expected.shape), expected, atol=1e-14, rtol=0
        )
        np.testing.assert_allclose(
            expected, original.tensor_contract(), atol=1e-14, rtol=0
        )

    with pytest.raises(ValueError, match="quantum data type"):
        quantum_control_signature(
            Signature((Register("q", quantum_bit_type()),)),
            ((classical_bit_type(), ()),),
        )


def test_original_nested_integer_controls_prepend_signed_group() -> None:
    from qualtran import CtrlSpec, QInt, QUInt
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.bloqs.mcmt import ControlledViaAnd

    child = MatrixGate.random(2, random_state=np.random.RandomState(42))
    source = ControlledViaAnd(child, CtrlSpec(QUInt(4), [2, 3])).controlled(
        CtrlSpec(QInt(4), [1, 2])
    )
    payload = Signature((Register("q", unsigned_register_type(2)),))
    signed, unsigned = signed_register_type(4), unsigned_register_type(4)
    native = quantum_control_signature(payload, ((signed, (2,)), (unsigned, (2,))))
    controls = tuple(native)[:2]
    assert tuple(port.dtype for port in controls) == (signed, unsigned)
    assert [(p.name, p.bitsize, p.shape) for p in controls] == [
        (p.name, p.bitsize, p.shape) for p in tuple(source.signature)[:2]
    ]
    assert tuple(group.tolist() for group in source.ctrl_spec.cvs) == ([1, 2], [2, 3])


def test_original_nested_x_uses_one_predicate_and_preserves_all_branches() -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt import ControlledViaAnd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler.registers.component_builder import (
        BindingCollection,
        ComponentProgramBuilder,
    )
    from flagquantum.compiler.registers.component_gate_function import bit_gate_function
    from flagquantum.compiler.registers.control.component_extension import (
        extend_component_control_system,
    )
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir import CircuitIR

    bit = quantum_bit_type()
    q = Register("q", bit)
    forward = bit_gate_function("x_leaf", (q,), CircuitIR(1, (Instruction("x", (0,)),)))
    positive = bit_gate_function(
        "positive", (Register("c", bit), q), CircuitIR(2, (Instruction("cx", (0, 1)),))
    )

    def append_positive(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        inputs: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        outputs = builder.call_dict("positive", {"c": control, "q": inputs["q"]})
        return outputs["c"], (outputs["q"],)

    builder = ComponentProgramBuilder(Signature((q,)), (forward, positive))
    outputs = builder.call_dict("x_leaf", {"q": builder.inputs["q"]})
    record = builder.finalize(outputs)
    ports = (Register("ctrl", bit, (2,)),)
    patterns = (ControlRegister(1, (1, 1)),)
    native, _ = extend_component_control_system(
        record,
        {"x_leaf": append_positive},
        ports,
        patterns,
        (Register("outer", bit, (2,)),),
        patterns,
    )
    original = ControlledViaAnd(XGate(), CtrlSpec(cvs=[1, 1])).controlled(
        CtrlSpec(cvs=[1, 1])
    )
    source_cost = get_cost_value(original, QECGatesCost())
    model = estimate_component_resources(
        controlled_component_cost(
            ComponentCost("CNOT", 2, GateCost(clifford=1)),
            (*patterns, *patterns),
            target_width=1,
        )
    )
    assert (
        (model.gates.and_bloq, model.gates.clifford, model.gates.measurement)
        == (source_cost.and_bloq, source_cost.clifford, source_cost.measurement)
        == (3, 4, 3)
    )
    bound = bind_register_program(
        native,
        tuple((wire,) for wire in range(5)),
        clean=(5, 6, 7),
        measured_workspace="reset",
    )
    gates = bound.circuit.instructions
    assert bound.outputs == tuple((wire,) for wire in range(5))
    assert sum(gate.name == "measure" for gate in gates) == 3
    incoming = torch.arange(1, 33, dtype=torch.float64)
    state = torch.complex(incoming, incoming.flip(0))
    state /= state.norm()
    initial, expected = (torch.zeros(256, dtype=state.dtype) for _ in range(2))
    initial[::8] = state
    matrix = torch.as_tensor(original.tensor_contract().reshape(32, 32))
    expected[::8] = matrix @ state
    for history in product((0, 1), repeat=3):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected * 2**-1.5,
            atol=2e-12,
            rtol=0,
        )
