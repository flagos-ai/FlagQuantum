"""Complete atomic interface examples use the existing native component owners."""

from hashlib import sha1
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler import estimate_resources
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler.registers.component_expansion import (
    NoComponentExpansionError,
    expand_component_calls_once,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_component_root_for_tensor,
    expand_components_for_tensor,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord
from tests.unit.atomic_register_fixtures import AtomicRegisterFixture, AtomKind

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("qualtran")
pytest.importorskip("quimb")


@pytest.fixture(scope="module", autouse=True)
def frozen_interface_source() -> None:
    from qualtran._infra import bloq

    data = Path(bloq.__file__).with_name("bloq_test.py").read_bytes()
    assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == (
        "fcc5ddd2a2ae3ab03138b39ec9b78c40d8a0cf89"
    )


@pytest.mark.parametrize("kind", ["x", "cx"])
def test_atomic_signature_wrapping_and_append_match_source(kind: AtomKind) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.for_testing import TestAtom, TestTwoBitOp
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = TestAtom() if kind == "x" else TestTwoBitOp()
    native = AtomicRegisterFixture(kind)
    assert [(r.name, r.bitsize, r.side.name) for r in native.signature] == [
        (r.name, r.bitsize, r.side.name) for r in source.signature
    ]
    assert native.label == str(source).removeprefix("Test")
    with pytest.raises(DecomposeTypeError, match="atomic"):
        source.decompose_bloq()
    leaf = bit_gate_function(
        "atom",
        tuple(native.signature),
        CircuitIR(len(native.signature), native.instructions),
    )
    record = ProgramRecord((leaf,))
    wrapped, labels = expand_component_root_for_tensor(
        record, lambda _: True, entry="atom", full_flatten=False
    )
    assert labels == (0,)
    assert wrapped.functions[0].blocks[0].operations[0].callee == "atom"
    # An existing composite is already the representation. Repeated selection
    # retains its single instance instead of wrapping another root around it.
    selected, labels = expand_components_for_tensor(
        wrapped, lambda _: True, entry="tensor_root", full_flatten=False
    )
    assert selected == wrapped and labels == (0,)
    with pytest.raises(NoComponentExpansionError):
        expand_component_calls_once(selected, entry="tensor_root")
    composite = source.as_composite_bloq()
    assert composite.as_composite_bloq() is composite
    (instance,) = composite.bloq_instances
    assert instance.bloq == source
    width = len(native.signature)
    bound = bind_register_program(
        selected, tuple((i,) for i in range(width)), entry="tensor_root"
    )
    assert bound.circuit.instructions == native.instructions and bound.events == ()
    assert estimate_resources(bound.circuit).operation_counts == {kind: 1}
    size = 2 ** (width + 1)
    state = torch.complex(
        torch.arange(1, size + 1, dtype=torch.float64),
        torch.arange(size, dtype=torch.float64).flip(0),
    )
    state /= state.norm()
    circuit = Circuit(width + 1, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(width + 1, bound.circuit.instructions))
    matrix = torch.as_tensor(
        source.tensor_contract().copy(), dtype=torch.complex128
    ).reshape(2**width, 2**width)
    expected = (matrix @ state.reshape(2**width, 2)).flatten()
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-14)
    if kind == "x":
        assert native.declared_cost is not None
        assert (
            query_qec_gate_cost(native.declared_cost).t
            == get_cost_value(source, QECGatesCost()).t
            == 100
        )
        assert estimate_resources(bound.circuit).operation_counts.get("t", 0) == 0


@pytest.mark.pennylane
def test_two_bit_source_pennylane_wrapper_matches_native_export() -> None:
    qml = pytest.importorskip("pennylane")
    from qualtran.bloqs.for_testing import TestTwoBitOp

    from flagquantum.ecosystem.pennylane import export_pennylane, import_pennylane

    native = AtomicRegisterFixture("cx")
    ir = CircuitIR(2, native.instructions)
    exported = export_pennylane(ir)
    source = TestTwoBitOp()
    assert source.as_pl_op(wires=[0, 1]) == qml.FromBloq(source, wires=[0, 1])
    expected = torch.as_tensor(
        qml.FromBloq(source, wires=[0, 1]).matrix(), dtype=torch.complex128
    )
    actual = torch.as_tensor(
        qml.matrix(exported.quantum_script, wire_order=[0, 1]), dtype=torch.complex128
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored = import_pennylane(exported.quantum_script)
    assert restored.ir.instructions == ir.instructions
