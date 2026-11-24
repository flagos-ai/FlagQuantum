"""Hermitian sparse typed roots preserve original calls and complex operators."""

import pytest
import torch

from flagquantum.compiler.registers.lookup.matrix_entry_component import (
    uniform_entry_program,
)
from flagquantum.compiler.registers.preparation.hermitian_sparse_matrix_component import (
    hermitian_sparse_matrix_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_sparse_matrix_component import _pair

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from hashlib import sha1
    from importlib.metadata import version
    from pathlib import Path

    import kickmix
    from qualtran.bloqs.block_encoding import sparse_matrix_hermitian

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, expected in (
        ("sparse_matrix_hermitian.py", "95b077d8eb6ac7deb814c910276b4a1d9d603bee"),
        ("sparse_matrix_hermitian_test.py", "367de6a3ae0ac011a16a5cc13b63dbb77cd7ba53"),
    ):
        raw = Path(sparse_matrix_hermitian.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected


@pytest.mark.parametrize("banded", (False, True))
@pytest.mark.parametrize("inverse", ("forward", "source_self", "generic"))
@pytest.mark.parametrize("controlled", (False, True))
def test_original_root_calls_and_complete_complex_operator(
    banded: bool, inverse: str, controlled: bool
) -> None:
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        SymmetricBandedRowColumnOracle,
        TopLeftRowColumnOracle,
    )
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        SparseMatrixHermitian,
        UniformSqrtEntryOracle,
    )

    column = (
        SymmetricBandedRowColumnOracle(2, 1) if banded else TopLeftRowColumnOracle(2)
    )
    entry = UniformSqrtEntryOracle(2, 0.3)
    source = SparseMatrixHermitian(column, entry, eps=0)
    assert source.adjoint() is source
    if inverse == "source_self":
        source = source.adjoint()
    if controlled:
        source = source.controlled()
    if inverse == "generic":
        from qualtran._infra.adjoint import Adjoint

        source = Adjoint(source)
    angle = float(2 * np.arccos(np.sqrt(entry.entry)))
    record = hermitian_sparse_matrix_component_program(
        2,
        column.num_nonzero,
        _pair(column),
        (
            uniform_entry_program(2, angle),
            uniform_entry_program(2, angle, adjoint=True),
        ),
        controlled=controlled,
        adjoint=inverse == "generic",
    )
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record))
    n = int(source.signature.n_qubits())
    offset = int(controlled)
    inputs: tuple[tuple[int, ...], ...] = (
        tuple(range(offset, offset + 2)),
        tuple(range(offset + 2, n)),
    )
    if controlled:
        inputs = ((0,), *inputs)
    bound = bind_register_program(record, inputs, clean=tuple(range(n, n + 8)))
    torch.testing.assert_close(
        logical_operator(bound.circuit, n),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=4e-11,
        rtol=0,
    )


@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_constructor_examples_and_source_codec_boundary(
    symbolic: bool, inverse: bool
) -> None:
    from qualtran.bloqs.block_encoding import sparse_matrix_hermitian
    from qualtran.serialization.bloq import bloqs_to_proto

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._sparse_matrix_component_library import (
        sparse_matrix_library_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    name = (
        "_sparse_matrix_symb_hermitian_block_encoding"
        if symbolic
        else "_sparse_matrix_hermitian_block_encoding"
    )
    source = getattr(sparse_matrix_hermitian, name).make()
    if inverse:
        source = source.adjoint()
    library = _library(source, 0)
    record = resolve_component_program(library, 0, sparse_matrix_library_factories())
    assert not verify_program_record(record)
    assert record.functions[0].registers == library.entries[0].declaration.registers
    if symbolic:
        assert not record.functions[0].blocks
    else:
        compare_layers(record, source, atomic_leaves(record))
        bound = bind_register_program(record, ((0, 1), (2, 3, 4, 5)))
        torch.testing.assert_close(
            logical_operator(bound.circuit, 6),
            torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
            atol=4e-11,
            rtol=0,
        )
    with pytest.raises(ValueError, match="inhomogeneous|sequence"):
        bloqs_to_proto(source, max_depth=1)


@pytest.mark.parametrize("pattern", ((1,), (0,), (1, 0)))
def test_controlled_constructor_calls_all_raw_histories_and_work_zero(
    pattern: tuple[int, ...],
) -> None:
    from itertools import product
    from math import sqrt

    from qualtran import CtrlSpec
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        _sparse_matrix_hermitian_block_encoding,
    )

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._sparse_matrix_component_library import (
        sparse_matrix_library_factories,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_legacy_arithmetic_factories import _library

    spec = CtrlSpec(cvs=pattern[0] if len(pattern) == 1 else pattern)
    source = _sparse_matrix_hermitian_block_encoding.make().controlled(spec)
    record = resolve_component_program(
        _library(source, 0), 0, sparse_matrix_library_factories()
    )
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record))
    count = len(pattern)
    logical = count + 6
    bound = bind_register_program(
        record,
        (
            *((index,) for index in range(count)),
            tuple(range(count, count + 2)),
            tuple(range(count + 2, logical)),
        ),
        clean=tuple(range(logical, logical + count - 1)),
    )
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), matrix, atol=5e-11, rtol=0
    )
    index = torch.arange(1 << logical, dtype=torch.float64)
    state = torch.complex(index % 7 - 3, index % 11 - 5)
    state /= state.norm()
    clean = torch.zeros(1 << (count - 1), dtype=torch.complex128)
    clean[0] = 1
    measurements = sum(gate.name == "measure" for gate in bound.circuit.instructions)
    assert measurements == count - 1
    for history in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(
                torch.kron(state, clean), bound.circuit.instructions, history
            ),
            torch.kron(matrix @ state, clean) / sqrt(2**measurements),
            atol=5e-11,
            rtol=0,
        )


def test_hermitian_source_free_constructor_resolution_and_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        _sparse_matrix_hermitian_block_encoding,
    )

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._sparse_matrix_component_library import (
        sparse_matrix_library_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = Adjoint(_sparse_matrix_hermitian_block_encoding.make())
    library = _library(source, 0)
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        record = resolve_component_program(
            library, 0, sparse_matrix_library_factories()
        )
        bound = bind_register_program(record, ((0, 1), (2, 3, 4, 5)))
        circuit = Circuit(6)
        circuit.append(_BoundBlockOperation(bound.circuit.instructions, 6))
        assert circuit.to_ir().instructions == bound.circuit.instructions
        torch.testing.assert_close(
            logical_operator(circuit.to_ir(), 6), expected, atol=4e-11, rtol=0
        )


@pytest.mark.parametrize("width", (0, 1))
@pytest.mark.parametrize("value", (0.0, 77 / 256, 1.0))
def test_sqrt_entry_constructor_values_zero_indices_and_epsilon(
    width: int, value: float
) -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        UniformSqrtEntryOracle,
    )

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._sparse_matrix_component_library import (
        sparse_matrix_library_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = UniformSqrtEntryOracle(width, value, eps=0.125)
    assert source.epsilon == 0.125
    record = resolve_component_program(
        _library(source, 0), 0, sparse_matrix_library_factories()
    )
    compare_layers(record, source, atomic_leaves(record))
    bound = bind_register_program(record, tuple((i,) for i in range(2 * width + 1)))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 2 * width + 1),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=4e-11,
        rtol=0,
    )
