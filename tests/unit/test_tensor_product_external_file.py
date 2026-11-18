"""Whole TensorProduct source objects, shaped partitions and owned execution."""

from collections import Counter
from collections.abc import Callable, Mapping
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.circuit_block_encoding import (
    CircuitBlockEncoding,
)
from flagquantum.algorithms.primitives.tensor_product_block_encoding import (
    TensorProductBlockEncoding,
)
from flagquantum.circuit import _CircuitOperation
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._block_encoding import import_block_encoding
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_foreign_block_encoding import original_models

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_tensor_sources() -> None:
    import kickmix
    from qualtran.bloqs.block_encoding import tensor_product

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(tensor_product.__file__).parent
    for name, blob in (
        ("tensor_product.py", "d1a01a8eccee2fff1f319668b6bd34f4a3890d13"),
        ("tensor_product_test.py", "1f997603c83beb1ca274a42e5c7ce102335c41e2"),
        ("tensor_product.ipynb", "f3f3652c6dd29f21f8151392c164267d689ae7b5"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == blob


def _binder(
    block: CircuitBlockEncoding,
) -> Callable[
    [Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
    _CircuitOperation,
]:
    def bind(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        inverse: bool,
    ) -> _CircuitOperation:
        return block.make_on(
            system=ports["system"],
            ancilla=ports["ancilla"],
            resource=ports["resource"],
            workspace=work,
            classical_bits=bits,
            adjoint=inverse,
        )

    return bind


@pytest.mark.parametrize("overrides", (False, True))
def test_actual_factories_four_way_owned_snapshots(overrides: bool) -> None:
    from qualtran.bloqs.block_encoding.tensor_product import (
        _tensor_product_block_encoding,
        _tensor_product_block_encoding_properties,
    )

    source = (
        _tensor_product_block_encoding_properties
        if overrides
        else _tensor_product_block_encoding
    )()
    block = import_block_encoding(source, models=original_models(source))
    for controlled, adjoint in product((False, True), repeat=2):
        root = source.controlled() if controlled else source
        root = root.adjoint() if adjoint else root
        offset = int(controlled)
        system = tuple(range(offset, offset + block.system_bitsize))
        start = offset + block.system_bitsize
        ancilla = tuple(range(start, start + block.ancilla_bitsize))
        start += block.ancilla_bitsize
        resource = tuple(range(start, start + block.resource_bitsize))
        n = start + block.resource_bitsize
        operation = block.make_on(
            system=system,
            ancilla=ancilla,
            resource=resource,
            control=(0,) if controlled else (),
            adjoint=adjoint,
        )
        ir = Circuit(n).append(operation).to_ir()
        expected = torch.as_tensor(root.tensor_contract(), dtype=torch.complex128)
        assert len(ir.instructions) == 2
        assert_source_model(
            block.resource_model(controlled=controlled, adjoint=adjoint), root
        )
        for actual in (ir, CircuitIR.from_json(ir.to_json())):
            torch.testing.assert_close(
                get_unitary(actual, dtype=torch.complex128),
                expected,
                atol=4e-12,
                rtol=0,
            )


@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("ancillas", ((0, 2), (2, 1)))
def test_resolved_symbolic_children_replace_native_consumer(
    adjoint: bool, ancillas: tuple[int, int]
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.block_encoding import TensorProduct, Unitary
    from qualtran.bloqs.block_encoding.tensor_product import (
        _tensor_product_block_encoding_symb,
    )
    from sympy import sympify

    declaration = _tensor_product_block_encoding_symb()
    with pytest.raises(DecomposeTypeError, match="Cannot decompose symbolic"):
        declaration.decompose_bloq()
    values = dict(
        a1=ancillas[0], a2=ancillas[1], alpha1=0.5, alpha2=0.5, eps1=0.01, eps2=0.1
    )
    substitutions = {
        s: values[str(s)]
        for s in sympify(
            declaration.alpha + declaration.epsilon + declaration.ancilla_bitsize
        ).free_symbols
    }
    children = tuple(
        Unitary(
            child.U,
            ancilla_bitsize=int(sympify(child.ancilla_bitsize).subs(substitutions)),
            alpha=float(sympify(child.alpha).subs(substitutions)),
            epsilon=float(sympify(child.epsilon).subs(substitutions)),
        )
        for child in declaration.block_encodings
    )
    blocks = tuple(
        import_block_encoding(child, models=original_models(child))
        for child in children
    )
    native = TensorProductBlockEncoding.of(*blocks)
    assert native == TensorProductBlockEncoding(blocks)
    source = TensorProduct.of(*children)
    assert (native.alpha, native.epsilon) == (source.alpha, source.epsilon)
    assert [(r.name, r.bitsize) for r in native.signature] == [
        (r.name, int(sympify(r.bitsize).subs(substitutions)))
        for r in declaration.signature
    ]

    n = native.system_bitsize + native.ancilla_bitsize
    operation = native.make_on(
        tuple(_binder(block) for block in blocks),
        system=(0, 1),
        ancilla=tuple(range(2, n)),
        workspaces=((), ()),
        classical_bits=((), ()),
        adjoint=adjoint,
    )
    ir = Circuit(n).append(operation).to_ir()
    locked = source.adjoint() if adjoint else source
    assert_source_model(
        native.resource_model(
            tuple(b.resource_model(adjoint=adjoint) for b in blocks), adjoint=adjoint
        ),
        locked,
    )
    for actual in (ir, CircuitIR.from_json(ir.to_json())):
        torch.testing.assert_close(
            get_unitary(actual, dtype=torch.complex128),
            torch.as_tensor(locked.tensor_contract(), dtype=torch.complex128),
            atol=3e-12,
            rtol=0,
        )


def test_external_measured_tensor_all_histories_and_work() -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.block_encoding import TensorProduct, Unitary
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = TensorProduct.of(Unitary(Add(QUInt(2))), Unitary(TGate()))
    block = import_block_encoding(source, include_controlled=False)
    work, slots = block.requirements()
    assert work == slots == 1
    grid = torch.arange(32, dtype=torch.float64)
    state = torch.complex(grid + 1, grid % 7 - 3)
    state /= state.norm()
    clean = torch.tensor([1, 0], dtype=torch.complex128)
    for adjoint in (False, True):
        operation = block.make_on(
            system=tuple(range(5)),
            workspace=(5,),
            classical_bits=(17,),
            adjoint=adjoint,
        )
        ir = Circuit(6).append(operation).to_ir()
        count = sum(g.name == "measure" for g in ir.instructions)
        assert count == 1
        locked = source.adjoint() if adjoint else source
        model = get_cost_value(locked, QECGatesCost())
        counts = Counter(g.name for g in ir.instructions)
        assert counts["measure"] == model.measurement
        assert counts["t"] + counts["tdg"] == 4 * model.and_bloq + model.t
        matrix = torch.as_tensor(locked.tensor_contract(), dtype=torch.complex128)
        for actual in (ir, CircuitIR.from_json(ir.to_json())):
            # Source moment scheduling interleaves disjoint gates inside the
            # And macros. Check the full branch operator, without reordering.
            for vector in (state, *torch.eye(32, dtype=torch.complex128)):
                expected = torch.kron(matrix @ vector, clean)
                for history in ((0,), (1,)):
                    torch.testing.assert_close(
                        forced_branch(
                            torch.kron(vector, clean), actual.instructions, history
                        ),
                        expected / 2**0.5,
                        atol=4e-12,
                        rtol=0,
                    )


def test_shaped_original_child_preserves_row_major_ports_and_tensor_partition() -> None:
    from typing import Any

    from attrs import frozen
    from qualtran import QBit, Register, Signature
    from qualtran.bloqs.basic_gates import CNOT, TGate, XGate
    from qualtran.bloqs.block_encoding import BlockEncoding, TensorProduct, Unitary
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare

    @frozen
    class ShapedBlock(BlockEncoding):
        alpha: float = 1
        epsilon: float = 0
        system_bitsize: int = 2
        ancilla_bitsize: int = 2
        resource_bitsize: int = 1

        @property
        def signature(self) -> Signature:
            return Signature(
                (
                    Register("system", QBit(), shape=(1, 2)),
                    Register("ancilla", QBit(), shape=(2,)),
                    Register("resource", QBit(), shape=(1, 1)),
                )
            )

        @property
        def signal_state(self) -> BlackBoxPrepare:
            return BlackBoxPrepare(PrepareIdentity.from_bitsizes([2]))

        def build_composite_bloq(
            self, bb: Any, system: Any, ancilla: Any, resource: Any
        ) -> dict[str, Any]:
            system[0, 0] = bb.add(XGate(), q=system[0, 0])
            ancilla[0], system[0, 1] = bb.add_t(
                CNOT(), ctrl=ancilla[0], target=system[0, 1]
            )
            return dict(system=system, ancilla=ancilla, resource=resource)

    shaped = ShapedBlock()
    source = TensorProduct.of(shaped, Unitary(TGate()))
    direct = import_block_encoding(shaped)
    root = import_block_encoding(source)
    assert direct.system_bitsize == 2
    assert direct.ancilla_bitsize == 2 and direct.resource_bitsize == 1
    assert root.binding_swaps == (0, 0, 0, 0)
    t = import_block_encoding(Unitary(TGate()))
    composed = TensorProductBlockEncoding.of(direct, t)
    for adjoint in (False, True):
        operation = composed.make_on(
            (_binder(direct), _binder(t)),
            system=(0, 1, 2),
            ancilla=(3, 4),
            resource=(5,),
            workspaces=((), ()),
            classical_bits=((), ()),
            adjoint=adjoint,
        )
        ir = Circuit(6).append(operation).to_ir()
        reference = source.adjoint() if adjoint else source
        torch.testing.assert_close(
            get_unitary(ir, dtype=torch.complex128),
            torch.as_tensor(reference.tensor_contract(), dtype=torch.complex128),
            atol=4e-12,
            rtol=0,
        )
    for block, locked in ((direct, shaped), (root, source)):
        for controlled, adjoint in product((False, True), repeat=2):
            offset = int(controlled)
            system = tuple(range(offset, offset + block.system_bitsize))
            start = offset + len(system)
            ancilla = tuple(range(start, start + block.ancilla_bitsize))
            resource = (start + len(ancilla),)
            operation = block.make_on(
                system=system,
                ancilla=ancilla,
                resource=resource,
                control=(0,) if controlled else (),
                adjoint=adjoint,
            )
            ir = Circuit(resource[0] + 1).append(operation).to_ir()
            reference = locked.controlled() if controlled else locked
            reference = reference.adjoint() if adjoint else reference
            expected = torch.as_tensor(
                reference.tensor_contract(), dtype=torch.complex128
            )
            for actual in (ir, CircuitIR.from_json(ir.to_json())):
                torch.testing.assert_close(
                    get_unitary(actual, dtype=torch.complex128),
                    expected,
                    atol=4e-12,
                    rtol=0,
                )


def test_original_serialization_and_zero_system_boundaries() -> None:
    from qualtran.bloqs.basic_gates import Identity, TGate
    from qualtran.bloqs.block_encoding import TensorProduct, Unitary
    from qualtran.bloqs.block_encoding.tensor_product import (
        _tensor_product_block_encoding,
        _tensor_product_block_encoding_properties,
        _tensor_product_block_encoding_symb,
    )
    from qualtran.testing import BloqCheckResult, check_bloq_example_serializes

    for example in (
        _tensor_product_block_encoding,
        _tensor_product_block_encoding_properties,
        _tensor_product_block_encoding_symb,
    ):
        result, diagnostic = check_bloq_example_serializes(example)
        assert result is BloqCheckResult.FAIL
        assert "Object arrays cannot be saved when allow_pickle=False" in diagnostic
    empty = Unitary(Identity(0))
    with pytest.raises(TypeError, match="required positional argument: 'system'"):
        TensorProduct.of(empty).decompose_bloq()
    with pytest.raises(KeyError, match="system"):
        TensorProduct.of(empty, Unitary(TGate())).decompose_bloq()
    # Scalar admission now reaches the same unchanged source failure.
    with pytest.raises(TypeError, match="required positional argument: 'system'"):
        import_block_encoding(TensorProduct.of(empty), include_controlled=False)


def test_shaped_boundary_refuses_mismatched_sides_and_symbolic_dimensions() -> None:
    from types import SimpleNamespace

    from qualtran import QAny, QBit, Register, Side, Signature
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
    from sympy import Symbol

    for signature, message in (
        (
            Signature(
                (
                    Register("system", QBit(), shape=(2,), side=Side.LEFT),
                    Register("system", QAny(2), side=Side.RIGHT),
                )
            ),
            "matching declared",
        ),
        (
            Signature((Register("system", QBit(), shape=(Symbol("m"),)),)),
            "symbolic",
        ),
    ):
        source = SimpleNamespace(
            system_bitsize=2,
            ancilla_bitsize=0,
            resource_bitsize=0,
            signal_state=BlackBoxPrepare(PrepareIdentity.from_bitsizes([0])),
            signature=signature,
        )
        with pytest.raises(ValueError, match=message):
            import_block_encoding(source, include_controlled=False)
