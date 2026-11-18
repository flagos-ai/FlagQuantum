"""Whole-source Product external binding, concrete resolution and source failures."""

from collections import Counter
from hashlib import sha1
from itertools import product
from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.product_block_encoding import (
    ProductBlockEncoding,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._block_encoding import import_block_encoding
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_foreign_block_encoding import original_models
from tests.unit.test_product_block_encoding import _bind

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_product_sources() -> None:
    from importlib.metadata import version

    import kickmix
    from qualtran.bloqs.block_encoding import product as source_module

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(source_module.__file__).parent
    for filename, blob in (
        ("product.py", "0c806922e9d734f7a2d823c960369c1a443ebb4a"),
        ("product_test.py", "389ffb997e324e16e3bb64e87e9bbd274188bdbe"),
        ("product.ipynb", "727fb156b779618d838b430868681e886a902ca6"),
    ):
        raw = (folder / filename).read_bytes()
        assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == blob


@pytest.mark.parametrize("ancillas", ((0, 0), (2, 1), (1, 3), (2, 2)))
@pytest.mark.parametrize("adjoint", (False, True))
def test_actual_foreign_constituents_and_resolved_symbolic_product(
    ancillas: tuple[int, int],
    adjoint: bool,
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.block_encoding import Product, Unitary
    from qualtran.bloqs.block_encoding.product import _product_block_encoding_symb
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from sympy import sympify

    declaration = _product_block_encoding_symb()
    with pytest.raises(DecomposeTypeError, match="Cannot decompose symbolic"):
        declaration.decompose_bloq()
    values = {
        symbol: {
            "a1": ancillas[0],
            "a2": ancillas[1],
            "alpha1": 0.5,
            "alpha2": 0.5,
            "eps1": 0.01,
            "eps2": 0.1,
        }[str(symbol)]
        for symbol in sympify(
            declaration.alpha + declaration.epsilon + declaration.ancilla_bitsize
        ).free_symbols
    }
    children = tuple(
        Unitary(
            child.U,
            alpha=float(sympify(child.alpha).subs(values)),
            ancilla_bitsize=int(sympify(child.ancilla_bitsize).subs(values)),
            epsilon=float(sympify(child.epsilon).subs(values)),
        )
        for child in declaration.block_encodings
    )
    source = Product.of(*children)
    assert source == Product(children)
    locked = source.adjoint() if adjoint else source
    blocks = tuple(
        import_block_encoding(child, models=original_models(child))
        for child in children
    )
    native, operation, options, _ = _bind(blocks, adjoint)
    assert native == ProductBlockEncoding.of(*blocks)
    assert (native.alpha, native.epsilon) == (source.alpha, source.epsilon)
    assert [(reg.name, reg.bitsize) for reg in native.signature] == [
        (reg.name, int(sympify(reg.bitsize).subs(values)))
        for reg in declaration.signature
    ]
    costs = tuple(block.resource_model(adjoint=adjoint) for block in blocks)
    assert_source_model(native.resource_model(costs, adjoint=adjoint), locked)
    ir = Circuit(operation.required_qubits).append(operation).to_ir()
    restored = CircuitIR.from_json(ir.to_json())
    work = options["flag_workspace"]
    events = certify_erasure_histories(
        ir.instructions, n_wires=ir.n_wires, zero_wires=work
    )
    assert events == sum(g.name == "measure" for g in ir.instructions)
    concrete_cost = get_cost_value(locked.decompose_bloq(), QECGatesCost())
    counts = Counter(g.name for g in ir.instructions)
    assert counts["measure"] == concrete_cost.measurement
    assert counts["t"] + counts["tdg"] == 4 * concrete_cost.and_bloq + 1
    logical = ir.n_wires - len(work)
    grid = torch.arange(1 << logical, dtype=torch.float64)
    state = torch.complex(grid + 1, grid % 5 - 2)
    state /= state.norm()
    clean = torch.zeros(1 << len(work), dtype=torch.complex128)
    clean[0] = 1
    matrix = torch.as_tensor(locked.tensor_contract(), dtype=torch.complex128)
    expected = torch.kron(matrix @ state, clean)
    for history in product((0, 1), repeat=events):
        for actual in (ir, restored):
            torch.testing.assert_close(
                forced_branch(torch.kron(state, clean), actual.instructions, history),
                expected / (2 ** (events / 2)),
                atol=4e-12,
                rtol=0,
            )


@pytest.mark.parametrize("overrides", (False, True))
def test_two_original_factory_objects_bind_as_owned_whole_blocks(
    overrides: bool,
) -> None:
    from qualtran.bloqs.block_encoding.product import (
        _product_block_encoding,
        _product_block_encoding_properties,
    )

    source = (
        _product_block_encoding_properties if overrides else _product_block_encoding
    )()
    block = import_block_encoding(source, include_controlled=False)
    for adjoint in (False, True):
        system = tuple(range(block.system_bitsize))
        ancilla = tuple(range(len(system), len(system) + block.ancilla_bitsize))
        resource = tuple(
            range(
                len(system) + len(ancilla),
                len(system) + len(ancilla) + block.resource_bitsize,
            )
        )
        logical = len(system) + len(ancilla) + len(resource)
        work, slots = block.requirements()
        operation = block.make_on(
            system=system,
            ancilla=ancilla,
            resource=resource,
            workspace=tuple(range(logical, logical + work)),
            classical_bits=tuple(range(slots)),
            adjoint=adjoint,
        )
        ir = Circuit(logical + work).append(operation).to_ir()
        expected = torch.as_tensor(
            (source.adjoint() if adjoint else source).tensor_contract(),
            dtype=torch.complex128,
        )
        assert work == slots == 0
        from flagquantum.simulation.unitary import get_unitary

        torch.testing.assert_close(
            get_unitary(ir, dtype=torch.complex128), expected, atol=3e-12, rtol=0
        )


def test_all_original_dense_families_import_as_complete_owned_products() -> None:
    import numpy as np
    from qualtran.bloqs.block_encoding import Product, Unitary
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate

    from flagquantum.simulation.unitary import get_unitary

    random_state = np.random.RandomState(1234)
    for _ in range(10):
        count, width = int(random_state.randint(3, 6)), int(random_state.randint(1, 3))
        source = Product.of(
            *(
                Unitary(MatrixGate.random(width, random_state=random_state))
                for _ in range(count)
            )
        )
        block = import_block_encoding(source)
        for controlled, adjoint in product((False, True), repeat=2):
            root = source.controlled() if controlled else source
            if adjoint:
                root = root.adjoint()
            offset = int(controlled)
            system = tuple(range(offset, offset + width))
            ancilla = tuple(
                range(offset + width, offset + width + block.ancilla_bitsize)
            )
            n = offset + width + len(ancilla)
            operation = block.make_on(
                system=system,
                ancilla=ancilla,
                control=(0,) if controlled else (),
                adjoint=adjoint,
            )
            ir = Circuit(n).append(operation).to_ir()
            assert len(ir.instructions) == count
            assert all(
                g.name == "any" and g.matrix is not None for g in ir.instructions
            )
            restored = CircuitIR.from_json(ir.to_json())
            expected = torch.as_tensor(root.tensor_contract(), dtype=torch.complex128)
            for actual in (ir, restored):
                torch.testing.assert_close(
                    get_unitary(actual, dtype=torch.complex128),
                    expected,
                    atol=3e-12,
                    rtol=0,
                )
            with pytest.raises(ValueError, match="No independent resource model"):
                block.resource_model(controlled=controlled, adjoint=adjoint)


def test_nested_original_product_keeps_child_order() -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.block_encoding import Product, Unitary

    from flagquantum.simulation.unitary import get_unitary

    x = Unitary(XGate())
    source = Product.of(Product.of(x, x), x)
    block = import_block_encoding(source, include_controlled=False)
    operation = block.make_on(system=(0,), ancilla=(1, 2))
    ir = Circuit(3).append(operation).to_ir()
    assert [g.name for g in ir.instructions] == ["x"] * 3
    assert all(g.wires == (0,) for g in ir.instructions)
    torch.testing.assert_close(
        get_unitary(ir, dtype=torch.complex128),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.parametrize("resources", ((1, 2), (2, 1)))
def test_external_import_preserves_unequal_resource_partition_failure(
    resources: tuple[int, int],
) -> None:
    from qualtran import BloqError
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding import Product, Unitary

    source = Product.of(
        Unitary(TGate(), resource_bitsize=resources[0]),
        Unitary(Hadamard(), resource_bitsize=resources[1]),
    )
    with pytest.raises(BloqError):
        source.decompose_bloq()
    with pytest.raises((BloqError, ValueError)):
        import_block_encoding(source, include_controlled=False)


def test_original_factory_serialization_failure_stays_explicit() -> None:
    from qualtran.bloqs.block_encoding.product import (
        _product_block_encoding,
        _product_block_encoding_properties,
        _product_block_encoding_symb,
    )
    from qualtran.testing import BloqCheckResult, check_bloq_example_serializes

    for example in (
        _product_block_encoding,
        _product_block_encoding_properties,
        _product_block_encoding_symb,
    ):
        result, detail = check_bloq_example_serializes(example)
        assert result is BloqCheckResult.FAIL
        assert "Object arrays cannot be saved when allow_pickle=False" in detail
