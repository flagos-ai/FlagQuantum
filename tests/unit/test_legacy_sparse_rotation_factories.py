"""Sparse rotation constructors preserve source partitions, cycles and adjoints."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_sparse_rotation_preparation_factories import (
    sparse_rotation_preparation_factories,
)
from flagquantum.compiler.registers.comparison.equality_component import (
    equality_constant_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.permutation_component import (
    permutation_cycle_program,
    permutation_program,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_legacy_rotation_preparation_factories import _terminals
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def pinned_sparse_rotation() -> None:
    import kickmix
    from qualtran.bloqs.state_preparation import sparse_state_preparation_via_rotations

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, digest in (
        (
            "sparse_state_preparation_via_rotations.py",
            "e5d631263c199129e23c246bfab0fd36fd5f774a",
        ),
        (
            "sparse_state_preparation_via_rotations_test.py",
            "ea6a6748f0837615f45e7e6082188b5cc7cfc1d9",
        ),
    ):
        raw = (
            Path(sparse_state_preparation_via_rotations.__file__).parent / name
        ).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.mark.parametrize("width", (1, 2, 3))
def test_shared_equality_and_cycle_exact_inverses_follow_original_calls(
    width: int,
) -> None:
    from qualtran.bloqs.arithmetic.comparison import EqualsAConstant
    from qualtran.bloqs.arithmetic.permutation import Permutation, PermutationCycle

    eq = equality_constant_program(width, 1, adjoint=True)
    compare_layers(eq, EqualsAConstant(width, 1).adjoint(), atomic_leaves(eq))
    cycle = (0, (1 << width) - 1)
    for composed in (False, True):
        record = (
            permutation_program(1 << width, (cycle,), width, adjoint=True)
            if composed
            else permutation_cycle_program(1 << width, cycle, width, adjoint=True)
        )
        source = (
            Permutation(1 << width, (cycle,))
            if composed
            else PermutationCycle(1 << width, cycle)
        ).adjoint()
        assert verify_program_record(record) == ()
        compare_layers(record, source, atomic_leaves(record))
        bound = bind_register_program(
            record, (tuple(range(width)),), clean=tuple(range(width, width + 8))
        )
        torch.testing.assert_close(
            logical_operator(bound.circuit, width),
            torch.as_tensor(source.tensor_contract()),
            atol=2e-11,
            rtol=0,
        )


@pytest.mark.parametrize(
    "example",
    (
        "_sparse_state_prep_via_rotations",
        "_sparse_state_prep_via_rotations_with_large_target_bitsize",
    ),
)
@pytest.mark.parametrize("inverse", (False, True))
def test_complete_original_examples_source_order_and_original_ragged_codec(
    example: str, inverse: bool
) -> None:
    from qualtran.bloqs.state_preparation import sparse_state_preparation_via_rotations

    forward = getattr(sparse_state_preparation_via_rotations, example).make()
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(
        _library(source, 0), 0, sparse_rotation_preparation_factories()
    )
    assert verify_program_record(record) == ()
    compare_layers(record, source, _terminals(record, 2))
    with pytest.raises(ValueError, match="inhomogeneous"):
        _library(source, 1)


@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("width", (1, 2, 3))
def test_small_full_complex_operator_all_histories_and_append(
    width: int, inverse: bool
) -> None:
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
    )

    forward = SparseStatePreparationViaRotations(
        (0, 1) if width == 1 else (1, 3), (0.6, 0.8j), 2 if width == 1 else 4, 2, width
    )
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(
        _library(source, 0), 0, sparse_rotation_preparation_factories()
    )
    compare_layers(record, source, _terminals(record, 2))
    inputs = (tuple(range(width)), tuple(range(width, width + 2)))
    bound = bind_register_program(
        record, inputs, clean=tuple(range(width + 2, width + 14))
    )
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, width + 2), expected, atol=3e-11, rtol=0
    )
    assert bound.outputs == inputs
    appended = (
        Circuit(bound.circuit.n_wires)
        .append(_BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires))
        .to_ir()
    )
    assert appended.instructions == bound.circuit.instructions
    assert bound.circuit.from_json(bound.circuit.to_json()) == bound.circuit


def test_source_import_free_concrete_and_original_symbolic_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    import sympy
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
        _sparse_state_prep_via_rotations,
    )

    library = _library(_sparse_state_prep_via_rotations.make(), 0)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        record = resolve_component_program(
            library, 0, sparse_rotation_preparation_factories()
        )
    assert verify_program_record(record) == ()
    symbolic = SparseStatePreparationViaRotations.from_n_coeffs(
        sympy.Symbol("N"), sympy.Symbol("d"), sympy.Symbol("b")
    )
    with pytest.raises(ValueError, match="Cannot serialize HasLength"):
        _library(symbolic, 0)


@pytest.mark.parametrize("dtype", ("float32", "complex64"))
@pytest.mark.parametrize("inverse", (False, True))
def test_input_precision_retains_source_eight_bit_tables(
    dtype: str, inverse: bool
) -> None:
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
    )

    coefficients = np.asarray((0.31, 0.47, 0.83), dtype=dtype)
    if dtype == "complex64":
        coefficients *= np.asarray((1, 1j, -1j), dtype=dtype)
    forward = SparseStatePreparationViaRotations(
        (1, 3, 5), tuple(coefficients), 8, 8, 3
    )
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(
        _library(source, 0), 0, sparse_rotation_preparation_factories()
    )
    compare_layers(record, source, _terminals(record, 8))


def test_partial_map_defect_and_singleton_source_boundary_remain_explicit() -> None:
    from qualtran import BloqError
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
    )

    source = SparseStatePreparationViaRotations((3, 0), (1, 1j), 4, 2)
    assert source._basis_permutation_bloq.cycles == ((0, 3),)
    # The source loses the requested 1 -> 0 transition. Preserve that source
    # operator in the interchange constructor, rather than replacing its map.
    record = resolve_component_program(
        _library(source, 0), 0, sparse_rotation_preparation_factories()
    )
    compare_layers(record, source, _terminals(record, 2))
    bound = bind_register_program(record, ((0, 1), (2, 3)), clean=tuple(range(4, 12)))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 4),
        torch.as_tensor(source.tensor_contract()),
        atol=3e-11,
        rtol=0,
    )
    singleton = SparseStatePreparationViaRotations((2,), (1,), 3, 2)
    with pytest.raises(BloqError, match="does not accept Soquets"):
        singleton.decompose_bloq()
    declared = resolve_component_program(
        _library(singleton, 0), 0, sparse_rotation_preparation_factories()
    )
    assert not declared.functions[0].blocks


@pytest.mark.parametrize("inverse", (False, True))
def test_symbolic_phase_keeps_source_direct_graph_with_unexecutable_dense_child(
    inverse: bool,
) -> None:
    import sympy
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
    )

    forward = SparseStatePreparationViaRotations(
        (1, 3), (0.6, 0.8j), 4, sympy.Symbol("b"), 2
    )
    source = forward.adjoint() if inverse else forward
    record = resolve_component_program(
        _library(source, 0), 0, sparse_rotation_preparation_factories()
    )
    assert verify_program_record(record) == ()
    assert record.functions[0].blocks
    compare_layers(record, source, atomic_leaves(record))
