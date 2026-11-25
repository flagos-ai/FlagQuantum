"""Compare the complete frozen Kikuchi Hamiltonian and both sparse wrappers."""

from collections import Counter
from dataclasses import replace
from hashlib import sha1
from pathlib import Path

import pytest
import sympy
from qualtran import Bloq
from qualtran.bloqs.basic_gates import Swap
from qualtran.bloqs.bookkeeping import Always
from qualtran.bloqs.optimization.k_xor_sat import kikuchi_block_encoding as source
from qualtran.resource_counting import (
    QECGatesCost,
    SympySymbolAllocator,
    get_cost_value,
)

from flagquantum.algorithms.kikuchi_block_encoding import (
    KikuchiEntryOracle,
    KikuchiHamiltonian,
    KikuchiRowColumnOracle,
    example_kikuchi_hamiltonian,
    symbolic_kikuchi_hamiltonian,
)
from flagquantum.algorithms.kxor_instance import Constraint, KXorInstance
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_kikuchi_adjacency_file import identity

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("symbolic", (False, True))
def test_literal_sparse_inventory_and_only_two_controlled_swaps(symbolic: bool) -> None:
    native = (
        symbolic_kikuchi_hamiltonian() if symbolic else example_kikuchi_hamiltonian()
    )
    original = source._kikuchi_matrix_symb() if symbolic else source._kikuchi_matrix()
    sparse = original._sparse_matrix_encoding
    for controlled in (False, True):
        mapped: Counter[Bloq] = Counter()
        for call, count in native.source_sparse_calls(controlled=controlled).items():
            component = call.component
            if isinstance(component, KikuchiEntryOracle):
                assert component == native.entry_oracle
                node = sparse.entry_oracle
            elif isinstance(component, KikuchiRowColumnOracle):
                assert component == native.column_oracle
                node = sparse.col_oracle
            elif isinstance(component, tuple):
                assert component == (
                    sparse.system_bitsize,
                    sparse.col_oracle.num_nonzero,
                )
                node = sparse.diffusion
            else:
                node = Swap(component)
                if controlled:
                    node = node.controlled()
                mapped[node] += count
                continue
            assert not call.controlled
            mapped[Always(node.adjoint() if call.adjoint else node)] += count
        expected = sparse.build_call_graph(SympySymbolAllocator())
        if controlled:
            expected = {
                node.controlled() if isinstance(node, Swap) else node: count
                for node, count in expected.items()
            }
        assert mapped == expected
    plain = native.source_sparse_calls()
    controlled_calls = native.source_sparse_calls(controlled=True)
    removed = set(plain.items()) - set(controlled_calls.items())
    added = set(controlled_calls.items()) - set(plain.items())
    assert {(call.component, count) for call, count in removed} == {
        (native.system_bitsize, 1),
        (1, 1),
    }
    assert all(not call.controlled and not call.adjoint for call, _ in removed)
    assert {
        (replace(call, controlled=False), count) for call, count in added
    } == removed


def test_sparse_inventory_combines_identical_one_bit_swaps() -> None:
    native = KikuchiHamiltonian(
        KXorInstance(2, 1, (Constraint((0,), 1),)), 1, entry_bitsize=3
    )
    for controlled in (False, True):
        swaps = {
            call: count
            for call, count in native.source_sparse_calls(controlled=controlled).items()
            if call.component == 1
        }
        ((call, count),) = swaps.items()
        assert count == 2 and call.controlled is controlled and not call.adjoint


@pytest.mark.parametrize("controlled", (0, 1, None, "yes"))
def test_sparse_inventory_rejects_non_boolean_control(controlled: bool) -> None:
    with pytest.raises(ValueError, match="boolean"):
        example_kikuchi_hamiltonian().source_sparse_calls(controlled=controlled)


def test_complete_source_and_original_suite_are_frozen() -> None:
    assert source.__file__ is not None
    for filename, blob in (
        ("kikuchi_block_encoding.py", "1929db7ae25185281fd2a93cb0f60a3c7cf3b166"),
        ("kikuchi_block_encoding_test.py", "c3a11d5b6f1b49eb87b4cdd03857723b157cb7a6"),
    ):
        raw = Path(source.__file__).with_name(filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("controlled", (False, True))
def test_all_factory_properties_signatures_wrappers_and_exact_resource_categories(
    symbolic: bool, controlled: bool
) -> None:
    native = (
        symbolic_kikuchi_hamiltonian() if symbolic else example_kikuchi_hamiltonian()
    )
    original = source._kikuchi_matrix_symb() if symbolic else source._kikuchi_matrix()
    for field in (
        "ell",
        "entry_bitsize",
        "s",
        "alpha",
        "system_bitsize",
        "ancilla_bitsize",
        "resource_bitsize",
        "epsilon",
    ):
        assert getattr(native, field) == getattr(original, field)
    assert str(native) == str(original) == "B[K_l]"
    wrapped = original._sparse_matrix_encoding
    for declaration, expected in (
        (native, original),
        (native.entry_oracle, wrapped.entry_oracle),
        (native.column_oracle, wrapped.col_oracle),
        (native.signal_state, original.signal_state),
    ):
        assert [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name)
            for r in declaration.signature
        ] == [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name)
            for r in expected.signature
        ]
    assert (
        native.entry_oracle.phase_gradient_bitsize
        == wrapped.entry_oracle._phasegrad_bitsize
    )
    assert native.entry_oracle.epsilon == wrapped.entry_oracle.epsilon == 0
    assert (
        dict(native.column_oracle.signature[0].dtype.parameters)["bound"]
        == wrapped.col_oracle.num_nonzero
    )
    assert native.source_graph(root="root", sparse_encoding="sparse").leaf_counts == {
        "sparse": 1
    }
    assert native.entry_oracle.source_graph(
        root="entry", entry="matrix"
    ).leaf_counts == {"matrix": 1}
    assert native.column_oracle.source_graph(
        root="column", neighbor="neighbor"
    ).leaf_counts == {"neighbor": 1}
    for actual, expected in (
        (
            native.resource_gates(controlled=controlled),
            original.controlled() if controlled else original,
        ),
        (native.entry_oracle.resource_gates(), wrapped.entry_oracle),
        (
            native.entry_oracle.resource_gates(adjoint=True),
            wrapped.entry_oracle.adjoint(),
        ),
        (native.column_oracle.resource_gates(), wrapped.col_oracle),
    ):
        cost = get_cost_value(expected, QECGatesCost())
        for field in (
            "t",
            "toffoli",
            "cswap",
            "and_bloq",
            "clifford",
            "rotation",
            "measurement",
        ):
            assert sympy.simplify(getattr(actual, field) - getattr(cost, field)) == 0
    if not symbolic:
        cost = native.resource_gates(controlled=controlled)
        assert cost.toffoli == 112 and cost.and_bloq == cost.measurement == 28268
        assert cost.cswap == (13345 if controlled else 13312)
        assert cost.clifford == (310446 if controlled else 310479)
        assert cost.rotation == 6


def test_entry_and_root_typed_programs_preserve_complete_original_port_wiring() -> None:
    native = example_kikuchi_hamiltonian()
    original = source._kikuchi_matrix()
    entry = native.entry_oracle
    program = entry.program(identity("matrix", tuple(entry.entry.signature)))
    assert verify_program_record(program) == ()
    root = next(
        function for function in program.functions if function.name == "main"
    ).blocks[0]
    call = root.operations[0]
    assert call.operands == (root.arguments[1], root.arguments[2], root.arguments[0])
    assert root.return_values == (call.results[2], call.results[0], call.results[1])
    ((instance, inputs, outputs),) = (
        original._sparse_matrix_encoding.entry_oracle.decompose_bloq().iter_bloqsoqs()
    )
    assert type(instance.bloq).__name__ == "KikuchiMatrixEntry"
    assert [inputs[name].reg.name for name in ("S", "T", "q")] == ["i", "j", "q"]
    assert [word.reg.name for word in outputs] == ["S", "T", "q"]
    root_program = native.program(identity("sparse", tuple(native.signature)))
    assert verify_program_record(root_program) == ()
    body = next(
        function for function in root_program.functions if function.name == "main"
    ).blocks[0]
    assert body.operations[0].operands == body.arguments
    assert body.return_values == body.operations[0].results
    assert len(tuple(original.decompose_bloq().iter_bloqsoqs())) == 1
    with pytest.raises(ValueError, match="signature"):
        entry.program(identity("bad", ()))
    with pytest.raises(ValueError, match="signature"):
        native.program(identity("bad", ()))


def test_literal_packed_to_array_neighbor_call_retains_source_failure() -> None:
    native = example_kikuchi_hamiltonian()
    original = source._kikuchi_matrix()._sparse_matrix_encoding.col_oracle
    with pytest.raises(IndexError, match="0-dimensional"):
        original.decompose_bloq()
    with pytest.raises(IndexError, match="0-dimensional"):
        native.column_oracle.program(
            identity("neighbor", tuple(native.neighbor_query.signature))
        )


def test_default_precision_explicit_overrides_and_sparsity_metadata() -> None:
    inst = KXorInstance(4, 2, (Constraint((0, 1), 1),), max_rhs=1)
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import (
        Constraint as SourceConstraint,
    )

    source_inst = source.KXorInstance(4, 2, (SourceConstraint((0, 1), 1),), max_rhs=1)
    for factory in (
        lambda: KikuchiHamiltonian(inst, 2),
        lambda: source.KikuchiHamiltonian(source_inst, 2),
    ):
        with pytest.raises(ValueError, match="explicit entry_bitsize"):
            factory()
    native = KikuchiHamiltonian(inst, 2, entry_bitsize=5, s=2)
    assert native.entry_bitsize == 5 and native.alpha == 2
    assert replace(native, s=3).alpha == 3
