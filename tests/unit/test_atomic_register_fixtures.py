"""Check source atomic signatures, tensors, declarations and append action."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from tests.unit.atomic_register_fixtures import AtomicRegisterFixture, AtomKind

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ["x", "cx", "identity"])
def test_atomic_fixture_matches_frozen_reference(kind: AtomKind) -> None:
    pytest.importorskip("qualtran")
    pytest.importorskip("quimb")
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.for_testing.atom import TestAtom, TestGWRAtom, TestTwoBitOp
    from qualtran.resource_counting import QECGatesCost

    source = {"x": TestAtom(), "cx": TestTwoBitOp(), "identity": TestGWRAtom()}[kind]
    native = AtomicRegisterFixture(kind)
    assert [(r.name, r.bitsize, r.side.name) for r in native.signature] == [
        (r.name, r.bitsize, r.side.name) for r in source.signature
    ]
    with pytest.raises(DecomposeTypeError, match="is atomic"):
        source.decompose_bloq()
    expected_label = str(source)
    if kind != "identity":
        expected_label = expected_label.removeprefix("Test")
    assert native.label == expected_label

    width = len(native.signature)
    size = 1 << width
    source_tensor = torch.as_tensor(
        source.tensor_contract().copy(), dtype=torch.complex128
    )
    expected = source_tensor.reshape(size, size)
    torch.testing.assert_close(
        native.tensor().reshape(size, size), expected, atol=0, rtol=0
    )
    incoming = {r.name: f"input:{r.name}" for r in native.signature}
    outgoing = {r.name: f"output:{r.name}" for r in native.signature}
    (raw,) = source.my_tensors(incoming, outgoing)
    order = tuple((port, 0) for port in (*outgoing.values(), *incoming.values()))
    axes = tuple(raw.inds.index(port) for port in order)
    torch.testing.assert_close(
        native.tensor(),
        torch.as_tensor(raw.data.copy(), dtype=torch.complex128).permute(axes),
        rtol=0,
        atol=0,
    )
    initial = torch.arange(1, size + 1, dtype=torch.float64)
    ket = torch.complex(initial, 2 - initial)
    ket /= ket.norm()
    circuit = Circuit(width, inputs=ket, dtype=ket.dtype).append(
        FieldOperation(width, native.instructions)
    )
    torch.testing.assert_close(
        circuit.state().flatten(), expected @ ket, rtol=0, atol=1e-14
    )
    cost = native.declared_cost
    if cost is None:
        assert source.my_static_costs(QECGatesCost()) is NotImplemented
    else:
        assert (
            query_qec_gate_cost(cost).t
            == source.my_static_costs(QECGatesCost()).t
            == 100
        )
    assert all(gate.name not in {"t", "tdg", "measure"} for gate in native.instructions)


def test_identity_fixture_adjoint_and_tag() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.atom import TestAtom, TestGWRAtom

    for tag in (None, "", "label"):
        atom = AtomicRegisterFixture("x", tag)
        assert atom.label == str(TestAtom(tag)).removeprefix("Test")
        gate = AtomicRegisterFixture("identity", tag)
        source = TestGWRAtom(tag)
        assert gate.label == str(source)
        assert gate.adjoint().is_adjoint == source.adjoint().is_adjoint
        assert gate.adjoint().adjoint() == gate
        torch.testing.assert_close(
            gate.adjoint().tensor(), gate.tensor(), atol=0, rtol=0
        )
