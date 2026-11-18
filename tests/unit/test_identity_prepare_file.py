"""Close the complete identity PREPARE declaration and original test scenarios."""

import ast
from collections import Counter
from hashlib import sha1
from pathlib import Path

import pytest
import torch
from sympy import Symbol

from flagquantum import Circuit
from flagquantum.algorithms.primitives.identity_prepare import IdentityPrepare
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    identity_cost_key,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    opaque_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_frozen_whole_files_factory_and_empty_package() -> None:
    from qualtran.bloqs.reflections import prepare_identity

    for name, blob in (
        ("__init__.py", "15780de77b1637deab59d60722c84f30b3d8e75d"),
        ("prepare_identity.py", "cbd8eb35d88d5c02b2428e988e44de6615b7db24"),
        ("prepare_identity_test.py", "ee6cac40ed0eafc51f69594d94c93172ef70e8f8"),
    ):
        raw = (Path(prepare_identity.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
        if name == "__init__.py":
            assert ast.parse(raw).body == []
    source = prepare_identity._prepare_identity.make()
    native = IdentityPrepare.from_bitsizes((10, 4, 1))
    assert [(r.name, r.total_bits()) for r in native.signature] == [
        (r.name, r.total_bits()) for r in source.signature
    ]
    doc = prepare_identity._PREPARE_IDENTITY_DOC
    assert doc.bloq_cls is type(source) and doc.examples == (
        prepare_identity._prepare_identity,
    )
    assert source.adjoint() is source and native.adjoint() is native
    assert source.junk_registers == native.junk_registers == ()
    for declaration in (source, native):
        with pytest.raises(NotImplementedError):
            _ = declaration.l1_norm_of_coeffs


@pytest.mark.parametrize("widths", ((), (1,), (2, 1), (4, 4, 5)))
def test_complete_identity_body_count_and_append(widths: tuple[int, ...]) -> None:
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = PrepareIdentity.from_bitsizes(widths)
    native = IdentityPrepare.from_bitsizes(widths)
    bound = native.bind()
    assert bound.normalization is None and bound.junk_width == 0
    assert tuple(native.signature) == native.selection_registers
    _, terminal = source.call_graph()
    leaves = source.build_call_graph(None)
    if not widths:
        assert terminal == {source: 1}
    declared = Counter(identity_cost_key(reg.bitsize) for reg in native.signature)
    assert declared == Counter(
        {identity_cost_key(g.bitsize): n for g, n in leaves.items()}
    )
    assert [
        instance.bloq.bitsize
        for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
    ] == list(widths)
    assert all(query_qec_gate_cost(key).asdict() == {} for key in declared)
    assert get_cost_value(source, QECGatesCost()).asdict() == {}
    assert get_cost_value(source, QubitCount()) == sum(widths)
    if sum(widths) > 5:
        # The original 4/4/5 resource fixture is checked without a dense 8192-square operator.
        assert bound.emit(tuple(range(sum(widths))), (), (), ()) == tuple(
            Instruction("i", (q,)) for q in range(sum(widths))
        )
        return
    n = sum(widths) + 1
    selection = tuple(reversed(range(1, n)))
    state = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(1 << n)], dtype=torch.complex128
    )
    state /= state.norm()
    for inverse in (False, True):
        operation = bound.make_on(selection=selection, junk=(), adjoint=inverse)
        circuit = Circuit(n, inputs=state, dtype=torch.complex128).append(operation)
        assert circuit.to_ir().instructions == tuple(
            Instruction("i", (q,)) for q in selection
        )
        torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=1e-12)
        torch.testing.assert_close(
            get_unitary(circuit), torch.eye(1 << n, dtype=torch.complex128)
        )
        assert (
            CircuitIR.from_json(circuit.to_ir().to_json()).instructions
            == circuit.to_ir().instructions
        )
    if sum(widths):
        torch.testing.assert_close(
            torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
            torch.eye(1 << sum(widths), dtype=torch.complex128),
        )
        capture = import_register_emitter(source)
        assert capture.requirements() == (0, 0) and capture.binding_swaps == (0, 0)
        for inverse in (False, True):
            assert capture.snapshots[int(inverse)].instructions == tuple(
                Instruction("i", (sum(widths[:i]),))
                for i, width in enumerate(widths)
                if width == 1
            )


def test_register_converter_symbolic_declaration_and_invalid_bodies() -> None:
    from qualtran import BloqError, CBit, QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity

    reg = Register("data", opaque_register_type(2))
    original = SourceRegister("data", QAny(2))
    assert (
        IdentityPrepare(reg) == IdentityPrepare([reg]) == IdentityPrepare(iter((reg,)))
    )
    assert PrepareIdentity(original) == PrepareIdentity([original])
    width = Symbol("n", positive=True, integer=True)
    symbolic = IdentityPrepare.from_bitsizes((width,))
    source = PrepareIdentity.from_bitsizes((width,))
    assert (
        symbolic.selection_registers[0].bitsize
        == source.selection_registers[0].bitsize
        == width
    )
    with pytest.raises(ValueError, match="concrete width"):
        symbolic.bind()
    for side in (RegisterSide.LEFT, RegisterSide.RIGHT):
        native = IdentityPrepare(Register("q", opaque_register_type(2), side=side))
        source = PrepareIdentity(SourceRegister("q", QAny(2), side=Side(side.value)))
        with pytest.raises(BloqError):
            source.decompose_bloq()
        with pytest.raises(ValueError, match="scalar through"):
            native.bind()
    shaped = IdentityPrepare(Register("q", opaque_register_type(2), shape=(1,)))
    with pytest.raises(ValueError, match="scalar through"):
        shaped.bind()

    bad_shape = PrepareIdentity(SourceRegister("q", QAny(2), shape=(1,)))
    with pytest.raises(AttributeError, match="reg"):
        bad_shape.decompose_bloq()
    with pytest.raises(BloqError, match="does not accept"):
        PrepareIdentity.from_bitsizes((0,)).decompose_bloq()
    with pytest.raises(ValueError, match="zero-width"):
        IdentityPrepare.from_bitsizes((0,)).bind()

    with pytest.raises(BloqError, match="dtypes are not consistent"):
        PrepareIdentity(SourceRegister("q", CBit())).decompose_bloq()
    with pytest.raises(ValueError, match="quantum selection"):
        IdentityPrepare(Register("q", classical_bit_type())).bind()
