"""Sparse typed graphs retain oracle types, rotations and source cleanup."""

from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.sparse_index_component import (
    banded_index_program,
    top_left_index_program,
)
from flagquantum.compiler.registers.lookup.matrix_entry_component import (
    explicit_entry_program,
    uniform_entry_program,
)
from flagquantum.compiler.registers.preparation.sparse_matrix_component import (
    sparse_matrix_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_sparse_matrix_source() -> None:
    from hashlib import sha1
    from importlib.metadata import version
    from pathlib import Path

    from qualtran.bloqs.block_encoding import sparse_matrix

    assert version("qualtran") == "0.7.0"
    for name, expected in (
        ("sparse_matrix.py", "cacbdab733f0be4f0f28bf2a749e1bad79987bfd"),
        ("sparse_matrix_test.py", "afcb0771b405ed26bfce7827d70d81c3d7f7a0d8"),
    ):
        raw = Path(sparse_matrix.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected


def _pair(source: Any) -> tuple[ProgramRecord, ProgramRecord]:
    np = pytest.importorskip("numpy")
    from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
        ExplicitEntryOracle,
    )

    kind = type(source).__name__
    n = source.system_bitsize
    if kind == "TopLeftRowColumnOracle":
        return (
            top_left_index_program(n, source.num_nonzero),
            top_left_index_program(n, source.num_nonzero, adjoint=True),
        )
    if kind == "SymmetricBandedRowColumnOracle":
        return (
            banded_index_program(n, source.bandsize),
            banded_index_program(n, source.bandsize, adjoint=True),
        )
    if kind == "UniformEntryOracle":
        angle = float(2 * np.arccos(source.entry))
        return (
            uniform_entry_program(n, angle),
            uniform_entry_program(n, angle, adjoint=True),
        )
    assert kind == "ExplicitEntryOracle"
    table = ExplicitEntryOracle(
        n, tuple(map(tuple, source.data.tolist())), source.entry_bitsize
    ).table
    return (
        explicit_entry_program(n, table),
        explicit_entry_program(n, table, adjoint=True),
    )


@pytest.mark.parametrize("kind", ("top", "band", "uniform", "explicit"))
@pytest.mark.parametrize("inverse", (False, True))
def test_direct_oracle_calls_and_full_complex_operators(
    kind: str, inverse: bool
) -> None:
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        ExplicitEntryOracle,
        SymmetricBandedRowColumnOracle,
        TopLeftRowColumnOracle,
        UniformEntryOracle,
    )

    source = {
        "top": lambda: TopLeftRowColumnOracle(2, 3),
        "band": lambda: SymmetricBandedRowColumnOracle(2, 1),
        "uniform": lambda: UniformEntryOracle(1, 0.3),
        "explicit": lambda: ExplicitEntryOracle(
            1, np.array([[0.0, 0.25], [1 / 3, 0.467]]), 3
        ),
    }[kind]()
    record = _pair(source)[int(inverse)]
    if inverse:
        source = source.adjoint()
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record))
    regs = record.functions[0].registers or ()
    cursor = 0
    inputs = []
    from flagquantum.core.ir.program.signatures import Signature

    for reg in Signature(regs).lefts():
        width = int(reg.total_bits())
        inputs.append(tuple(range(cursor, cursor + width)))
        cursor += width
    bound = bind_register_program(
        record, tuple(inputs), clean=tuple(range(cursor, cursor + 6))
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, cursor),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=3e-11,
        rtol=0,
    )


@pytest.mark.parametrize(
    "name", ("_sparse_matrix_block_encoding", "_symmetric_banded_matrix_block_encoding")
)
@pytest.mark.parametrize("inverse", (False, True))
def test_original_uniform_sparse_roots_full_call_chain(
    name: str, inverse: bool
) -> None:
    from qualtran.bloqs.block_encoding import sparse_matrix

    source = getattr(sparse_matrix, name).make()
    record = sparse_matrix_component_program(
        source.system_bitsize,
        source.alpha,
        _pair(source.row_oracle),
        _pair(source.col_oracle),
        _pair(source.entry_oracle),
        adjoint=inverse,
    )
    if inverse:
        source = source.adjoint()
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record))
    n = int(source.signature.n_qubits())
    width = int(source.signature.get_left("system").bitsize)
    bound = bind_register_program(
        record,
        (tuple(range(width)), tuple(range(width, n))),
        clean=tuple(range(n, n + 8)),
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, n),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=4e-11,
        rtol=0,
    )


@pytest.mark.parametrize(
    "name",
    (
        "_sparse_matrix_block_encoding",
        "_sparse_matrix_symb_block_encoding",
        "_explicit_matrix_block_encoding",
        "_symmetric_banded_matrix_block_encoding",
    ),
)
@pytest.mark.parametrize("inverse", (False, True))
def test_all_original_sparse_constructor_examples_and_source_codec_boundary(
    name: str, inverse: bool
) -> None:
    from qualtran.bloqs.block_encoding import sparse_matrix
    from qualtran.serialization.bloq import bloqs_to_proto

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._sparse_matrix_component_library import (
        sparse_matrix_library_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = getattr(sparse_matrix, name).make()
    if inverse:
        source = source.adjoint()
    library = _library(source, 0)
    record = resolve_component_program(library, 0, sparse_matrix_library_factories())
    assert not verify_program_record(record)
    assert record.functions[0].registers == library.entries[0].declaration.registers
    if "symb" in name:
        assert not record.functions[0].blocks
        with pytest.raises(ValueError):
            bind_register_program(record, ((0,), (1, 2)))
    else:
        compare_layers(record, source, atomic_leaves(record))
        width = int(source.signature.get_left("system").bitsize)
        logical = 2 * width + 1
        precision = 10 if "explicit" in name else 0
        bound = bind_register_program(
            record,
            (tuple(range(width)), tuple(range(width, logical))),
            clean=tuple(range(logical, logical + max(8, precision + 3))),
        )
        torch.testing.assert_close(
            logical_operator(bound.circuit, logical),
            torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
            atol=6e-11,
            rtol=0,
        )
    with pytest.raises(ValueError, match="inhomogeneous|sequence"):
        bloqs_to_proto(source, max_depth=1)


@pytest.mark.parametrize("inverse", (False, True))
def test_common_eight_bit_explicit_table_full_sparse_operator(inverse: bool) -> None:
    import attrs
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        _explicit_matrix_block_encoding,
    )

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._sparse_matrix_component_library import (
        sparse_matrix_library_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    original = _explicit_matrix_block_encoding.make()
    source = attrs.evolve(
        original, entry_oracle=attrs.evolve(original.entry_oracle, entry_bitsize=8)
    )
    if inverse:
        source = source.adjoint()
    record = resolve_component_program(
        _library(source, 0), 0, sparse_matrix_library_factories()
    )
    compare_layers(record, source, atomic_leaves(record))
    bound = bind_register_program(record, ((0,), (1, 2)), clean=tuple(range(3, 13)))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 3),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=5e-11,
        rtol=0,
    )


def test_small_sparse_combined_measurement_histories_and_private_zero() -> None:
    from itertools import product
    from math import sqrt

    import attrs
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        _explicit_matrix_block_encoding,
    )

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._sparse_matrix_component_library import (
        sparse_matrix_library_factories,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_legacy_arithmetic_factories import _library

    original = _explicit_matrix_block_encoding.make()
    source = attrs.evolve(
        original, entry_oracle=attrs.evolve(original.entry_oracle, entry_bitsize=2)
    )
    bounds = []
    for node in (source, source.adjoint()):
        record = resolve_component_program(
            _library(node, 0), 0, sparse_matrix_library_factories()
        )
        bound = bind_register_program(record, ((0,), (1, 2)), clean=(3, 4, 5))
        torch.testing.assert_close(
            logical_operator(bound.circuit, 3),
            torch.as_tensor(node.tensor_contract(), dtype=torch.complex128),
            atol=5e-11,
            rtol=0,
        )
        bounds.append(bound)
    instructions = (*bounds[0].circuit.instructions, *bounds[1].circuit.instructions)
    count = sum(g.name == "measure" for g in instructions)
    assert count > 0
    seed = torch.arange(1, 9, dtype=torch.float64)
    amplitudes = torch.complex(seed, seed.remainder(3) - 1)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(64, dtype=torch.complex128)
    state[::8] = amplitudes
    for history in product((0, 1), repeat=count):
        torch.testing.assert_close(
            forced_branch(state, instructions, history) * sqrt(2**count),
            state,
            atol=6e-11,
            rtol=0,
        )


def test_sparse_source_free_resolution_binding_and_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran.bloqs.block_encoding.sparse_matrix import (
        _explicit_matrix_block_encoding,
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

    library = _library(_explicit_matrix_block_encoding.make().adjoint(), 0)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        record = resolve_component_program(
            library, 0, sparse_matrix_library_factories()
        )
        bound = bind_register_program(record, ((0,), (1, 2)), clean=tuple(range(3, 16)))
        circuit = Circuit(bound.circuit.n_wires)
        circuit.append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        assert (
            circuit.to_ir().to_dict()["instructions"]
            == bound.circuit.to_dict()["instructions"]
        )


@pytest.mark.parametrize("inverse", (False, True))
def test_atomic_controlled_ry_leaves_match_full_source_operators(inverse: bool) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.basic_gates import Ry
    from qualtran.bloqs.block_encoding.sparse_matrix import ExplicitEntryOracle

    from flagquantum.compiler.registers.component_definition_match import (
        _named_definitions,
    )
    from flagquantum.core.ir.program.signatures import Signature

    np = pytest.importorskip("numpy")
    source = ExplicitEntryOracle(1, np.array([[0.0, 0.25], [1 / 3, 0.467]]), 8)
    record = _pair(source)[int(inverse)]
    leaves = [
        f
        for f in _named_definitions(record, record.functions[0].name)
        if f.registers == tuple(Signature.build(ctrl=1, q=1))
        and any(op.opcode == "cry" for b in f.blocks for op in b.operations)
    ]
    assert len(leaves) == 8
    angles = []
    for leaf in leaves:
        bound = bind_register_program(
            ProgramRecord((leaf,)), ((0,), (1,)), entry=leaf.name
        )
        gate = next(g for g in bound.circuit.instructions if g.name == "cry")
        angle = gate.params["theta"]
        angles.append(angle)
        controlled = Ry(angle).controlled()
        with pytest.raises(DecomposeTypeError, match="atomic"):
            controlled.decompose_bloq()
        torch.testing.assert_close(
            logical_operator(bound.circuit, 2),
            torch.as_tensor(controlled.tensor_contract(), dtype=torch.complex128),
            atol=2e-12,
            rtol=0,
        )
    sign = -1 if inverse else 1
    assert sorted(angles) == sorted(sign * 2 * np.pi * 2 ** -(k + 1) for k in range(8))


def test_explicit_float32_payload_preserves_source_quantized_words() -> None:
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.block_encoding.sparse_matrix import ExplicitEntryOracle

    from flagquantum.ecosystem._sparse_matrix_component_library import _table

    data = np.array([[0.0, 0.25], [1 / 3, 0.467]], dtype=np.float32)
    source = ExplicitEntryOracle(1, data, 28)
    lookup = next(
        instance.bloq
        for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
        if type(instance.bloq).__name__ == "QROM"
    )
    table = _table(1, data, 28)
    assert table.datasets == tuple(
        tuple(int(value) for value in array.flat) for array in lookup.data
    )
    # The original float32 arccos must run before the lookup's integer encoding.
    promoted = _table(1, data.astype(np.float64), 28)
    assert table.datasets != promoted.datasets


@pytest.mark.parametrize("inverse", (False, True))
def test_nondefault_top_left_bound_recovers_omitted_source_constructor_field(
    inverse: bool,
) -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        SparseMatrix,
        TopLeftRowColumnOracle,
        UniformEntryOracle,
    )
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._sparse_matrix_component_library import (
        sparse_matrix_library_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    oracle = TopLeftRowColumnOracle(2, 3)
    library = _library(oracle, 0)
    assert {a.name for a in library.entries[0].declaration.arguments} == {
        "system_bitsize"
    }
    restored = bloqs_from_proto(bloqs_to_proto(oracle, max_depth=0))[0]
    assert restored.num_nonzero == 4 and oracle.num_nonzero == 3
    source = SparseMatrix(oracle, oracle, UniformEntryOracle(2, 0.3), eps=0)
    if inverse:
        source = source.adjoint()
    record = resolve_component_program(
        _library(source, 0), 0, sparse_matrix_library_factories()
    )
    compare_layers(record, source, atomic_leaves(record))
    bound = bind_register_program(
        record, ((0, 1), (2, 3, 4)), clean=tuple(range(5, 13))
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, 5),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=6e-11,
        rtol=0,
    )


@pytest.mark.parametrize("inverse", (False, True))
def test_explicit_symbolic_precision_stays_declared_and_refuses_physical_binding(
    inverse: bool,
) -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.block_encoding.sparse_matrix import ExplicitEntryOracle
    from sympy import Symbol

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._sparse_matrix_component_library import (
        sparse_matrix_library_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = ExplicitEntryOracle(
        1, ((0.0, 0.25), (1 / 3, 0.467)), Symbol("k", positive=True, integer=True)
    )
    with pytest.raises(DecomposeTypeError, match="symbolic"):
        source.decompose_bloq()
    if inverse:
        source = source.adjoint()
    library = _library(source, 0)
    record = resolve_component_program(library, 0, sparse_matrix_library_factories())
    assert not verify_program_record(record) and not record.functions[0].blocks
    assert record.functions[0].registers == library.entries[0].declaration.registers
    with pytest.raises(ValueError):
        bind_register_program(record, ((0,), (1,), (2,)))


def test_fixed_width_symbolic_ordinal_bound_retains_physical_storage_gap() -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import TopLeftRowColumnOracle
    from sympy import Symbol

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._sparse_matrix_component_library import (
        sparse_matrix_library_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = TopLeftRowColumnOracle(2, Symbol("s", positive=True, integer=True))
    torch.testing.assert_close(
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        torch.eye(16, dtype=torch.complex128),
        atol=0,
        rtol=0,
    )
    library = _library(source, 0)
    record = resolve_component_program(library, 0, sparse_matrix_library_factories())
    assert not verify_program_record(record)
    assert record.functions[0].registers == library.entries[0].declaration.registers
    with pytest.raises(ValueError, match="unsupported executable register descriptor"):
        bind_register_program(record, ((0, 1), (2, 3)))


@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("explicit", (False, True))
def test_zero_index_entry_signature_and_original_execution_boundary(
    explicit: bool, inverse: bool
) -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        ExplicitEntryOracle,
        UniformEntryOracle,
    )

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.ecosystem._sparse_matrix_component_library import (
        sparse_matrix_library_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = (
        ExplicitEntryOracle(0, ((0.3,),), 2) if explicit else UniformEntryOracle(0, 0.3)
    )
    if explicit:
        with pytest.raises(TypeError, match="required positional arguments"):
            source.decompose_bloq()
    if inverse:
        source = source.adjoint()
    library = _library(source, 0)
    record = resolve_component_program(library, 0, sparse_matrix_library_factories())
    assert record.functions[0].registers == library.entries[0].declaration.registers
    assert len(record.functions[0].registers or ()) == 1
    if explicit:
        assert not record.functions[0].blocks
        with pytest.raises(ValueError):
            bind_register_program(record, ((0,),))
    else:
        bound = bind_register_program(record, ((0,),))
        torch.testing.assert_close(
            logical_operator(bound.circuit, 1),
            torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
            atol=2e-12,
            rtol=0,
        )
