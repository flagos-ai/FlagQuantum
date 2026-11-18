"""Exercise an external BlockEncoding subclass through native owned boundaries."""

from hashlib import sha1
from itertools import product
from pathlib import Path
from typing import Any, TypedDict

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_metadata import _BlockDeclaration
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._block_encoding import import_block_encoding
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_foreign_block_encoding import original_models

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


class BlockBinding(TypedDict):
    system: tuple[int, ...]
    ancilla: tuple[int, ...]
    resource: tuple[int, ...]
    control: tuple[int, ...]
    adjoint: bool


@pytest.fixture(scope="module")
def external_block_type() -> Any:
    from attrs import field, make_class
    from qualtran import Signature
    from qualtran.bloqs.basic_gates import OnEach, SGate
    from qualtran.bloqs.block_encoding import block_encoding_base
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare

    raw = Path(block_encoding_base.__file__).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "8e6acb38828863e959c0a8543d79bebc0bda1d2f"
    )
    base = block_encoding_base.BlockEncoding
    assert base.__abstractmethods__ == {
        "signature",
        "alpha",
        "epsilon",
        "system_bitsize",
        "ancilla_bitsize",
        "resource_bitsize",
        "signal_state",
    }
    assert block_encoding_base._BLOCK_ENCODING_DOC.examples == ()

    def signature(source: Any) -> Any:
        return Signature.build(
            system=source.system_bitsize,
            ancilla=source.ancilla_bitsize,
            resource=source.resource_bitsize,
        )

    def signal_state(source: Any) -> Any:
        return BlackBoxPrepare(PrepareIdentity.from_bitsizes([source.ancilla_bitsize]))

    def build(source: Any, bb: Any, **ports: Any) -> dict[str, Any]:
        ports["system"] = bb.add(
            OnEach(source.system_bitsize, SGate()), q=ports["system"]
        )
        return ports

    def calls(source: Any, ssa: Any) -> dict[Any, int]:
        return {OnEach(source.system_bitsize, SGate()): 1}

    # attrs supplies the same immutable property fields used by source subclasses.
    return make_class(
        "ExternalBlock",
        {
            name: field()
            for name in (
                "system_bitsize",
                "ancilla_bitsize",
                "resource_bitsize",
                "alpha",
                "epsilon",
            )
        },
        bases=(base,),
        class_body={
            "signature": property(signature),
            "signal_state": property(signal_state),
            "build_composite_bloq": build,
            "build_call_graph": calls,
        },
        frozen=True,
        slots=True,
    )


@pytest.mark.parametrize("width,auxiliary", ((1, False), (1, True), (2, True)))
def test_arbitrary_source_subclass_four_way_capture_and_full_complex_replacement(
    external_block_type: Any,
    width: int,
    auxiliary: bool,
) -> None:
    from qualtran.bloqs.basic_gates import OnEach, SGate
    from qualtran.bloqs.block_encoding import Unitary

    source = external_block_type(width, int(auxiliary), int(auxiliary), 2.0, 0.01)
    models = original_models(source)
    block = import_block_encoding(source, models=models)
    canonical = import_block_encoding(
        Unitary(
            OnEach(width, SGate()),
            alpha=2.0,
            ancilla_bitsize=int(auxiliary),
            resource_bitsize=int(auxiliary),
            epsilon=0.01,
        )
    )
    declaration: _BlockDeclaration[int, object] = block
    assert (declaration.alpha, declaration.epsilon) == (source.alpha, source.epsilon)
    assert (
        declaration.system_bitsize,
        declaration.ancilla_bitsize,
        declaration.resource_bitsize,
    ) == (width, int(auxiliary), int(auxiliary))
    assert [(r.name, r.bitsize) for r in block.signature] == [
        (r.name, r.bitsize) for r in source.signature
    ]
    assert block.signal_state.emit(tuple(range(int(auxiliary))), (), (), ()) == ()
    assert block.binding_swaps == (0, 0, 0, 0)
    for controlled, adjoint in product((False, True), repeat=2):
        root = source.controlled() if controlled else source
        if adjoint:
            root = root.adjoint()
        assert_source_model(
            block.resource_model(controlled=controlled, adjoint=adjoint), root
        )
        assert block.requirements(controlled=controlled) == (0, 0)
        offset = int(controlled)
        logical = width + 2 * int(auxiliary) + offset
        wires = tuple(reversed(range(logical)))
        system = wires[offset : offset + width]
        ancilla = wires[offset + width : offset + width + int(auxiliary)]
        resource = wires[offset + width + int(auxiliary) :]
        options = BlockBinding(
            system=system,
            ancilla=ancilla,
            resource=resource,
            control=wires[:offset],
            adjoint=adjoint,
        )
        native = Circuit(logical + 1).append(block.make_on(**options)).to_ir()
        alternate = Circuit(logical + 1).append(canonical.make_on(**options)).to_ir()
        assert native.instructions == alternate.instructions
        assert len(native.instructions) == width
        assert all(
            g.name == ("cphase" if controlled else "sdg" if adjoint else "s")
            for g in native.instructions
        )
        restored = CircuitIR.from_json(native.to_json())
        source_matrix = torch.as_tensor(root.tensor_contract(), dtype=torch.complex128)
        order = (*wires, logical)
        reverse_order = tuple(order.index(q) for q in range(logical + 1))
        state = torch.tensor(
            [complex(i + 1, 2 - i) for i in range(1 << (logical + 1))],
            dtype=torch.complex128,
        )
        state /= state.norm()
        expected = (
            source_matrix
            @ state.reshape((2,) * (logical + 1))
            .permute(order)
            .reshape(1 << logical, 2)
        ).reshape((2,) * (logical + 1))
        expected = expected.permute(reverse_order).flatten()
        circuit = Circuit(logical + 1, inputs=state, dtype=torch.complex128)
        circuit.append(block.make_on(**options))
        torch.testing.assert_close(
            circuit.state().flatten(), expected, atol=3e-12, rtol=0
        )
        options["adjoint"] = not adjoint
        circuit.append(block.make_on(**options))
        torch.testing.assert_close(circuit.state().flatten(), state, atol=3e-12, rtol=0)
        torch.testing.assert_close(
            get_unitary(restored, dtype=torch.complex128),
            get_unitary(alternate, dtype=torch.complex128),
            atol=3e-12,
            rtol=0,
        )


def test_external_interface_symbolic_and_nonidentity_signal_admission(
    external_block_type: Any,
) -> None:
    from attrs import evolve, make_class
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
    from sympy import Symbol

    symbolic = external_block_type(Symbol("s", integer=True, positive=True), 1, 0, 2, 0)
    assert str(symbolic.system_bitsize) == "s"
    with pytest.raises(ValueError, match="concrete nonnegative widths"):
        import_block_encoding(symbolic)
    resolved = evolve(symbolic, system_bitsize=1)
    assert import_block_encoding(resolved).system_bitsize == 1
    nonidentity_type = make_class(
        "NonidentitySignalBlock",
        {},
        bases=(external_block_type,),
        class_body={
            "signal_state": property(lambda source: BlackBoxPrepare(Hadamard()))
        },
        frozen=True,
        slots=True,
    )
    nonidentity = nonidentity_type(1, 1, 0, 2, 0)
    with pytest.raises(ValueError, match="identity signal state"):
        import_block_encoding(nonidentity)
    unpriced = import_block_encoding(resolved)
    with pytest.raises(ValueError, match="No independent resource model"):
        unpriced.resource_model()
