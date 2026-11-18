"""Whole-source neighbor declarations, costs and literal typed root wiring."""

from dataclasses import replace
from hashlib import sha1
from pathlib import Path

import numpy as np
import pytest
import sympy
from qualtran import DecomposeNotImplementedError, DecomposeTypeError
from qualtran.bloqs.optimization.k_xor_sat import kikuchi_adjacency_list as source
from qualtran.resource_counting import (
    QECGatesCost,
    SympySymbolAllocator,
    get_cost_value,
)

from flagquantum.algorithms.kikuchi_adjacency import (
    IndexOfNonZeroColumn,
    KikuchiNonZeroIndex,
    example_column_query,
    example_kikuchi_nonzero_index,
    symbolic_column_query,
    symbolic_kikuchi_nonzero_index,
)
from flagquantum.algorithms.kxor_instance import Constraint, KXorInstance
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register

_BLOBS = {
    "kikuchi_adjacency_list.py": "9668cd5f28391e8f3a789024451ef686b64073cf",
    "kikuchi_adjacency_list_test.py": "88e97c19d8c15a7e473c067c7cacfd568687da9f",
}

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_source_and_original_tests_are_frozen() -> None:
    assert source.__file__ is not None
    for name, blob in _BLOBS.items():
        raw = Path(source.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_all_query_class_fields_calls_operands_and_qec(
    symbolic: bool, reverse: bool
) -> None:
    native = symbolic_column_query() if symbolic else example_column_query()
    original = source._col_kth_nz_symb() if symbolic else source._col_kth_nz()
    if reverse:
        native = IndexOfNonZeroColumn(native.inst, native.ell)
        original = source.IndexOfNonZeroColumn(original.inst, original.ell)
    assert native.adjoint() is native and original.adjoint() is original
    assert native.ell == original.ell
    assert native.index_dtype.num_qubits == original.index_dtype.num_qubits
    assert [
        (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in native.signature
    ] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in original.signature
    ]
    calls = original.build_call_graph(SympySymbolAllocator())
    normalized = {}
    for child, count in calls.items():
        name = type(child).__name__
        if name == "Controlled":
            body = child.subbloq
            if type(body).__name__ == "Xor":
                role = "cxor"
                assert body.dtype.num_qubits == native.index_dtype.num_qubits
            else:
                inverse = type(body).__name__ == "Adjoint"
                body = body.subbloq if inverse else body
                assert type(body).__name__ == "AddK" and body.k == 1
                assert body.dtype.num_qubits == native.index_dtype.num_qubits
                role = "increment_inverse" if inverse else "increment"
        elif name == "And":
            role = "erase" if child.uncompute else "and"
            assert child.cv1 == child.cv2 == 1
        elif name == "Xor":
            role = "xor_bit"
            assert child.dtype.num_qubits == 1
        elif name == "CNOT":
            role = "cnot"
        else:
            inverse = name == "Adjoint"
            body = child.subbloq if inverse else child
            if type(body).__name__ == "SymmetricDifference":
                assert (
                    body.n_lhs == body.n_diff == native.ell
                    and body.n_rhs == native.inst.k
                )
                assert body.dtype.num_qubits == native.index_dtype.num_qubits
                role = "diff_inverse" if inverse else "diff"
            else:
                assert type(body).__name__ == "Equals"
                assert body.dtype.num_qubits == native.index_dtype.num_qubits
                role = "equals_inverse" if inverse else "equals"
        normalized[role] = count
    graph = native.source_graph(
        root="query",
        controlled_xor="cxor",
        bit_xor="xor_bit",
        difference="diff",
        inverse_difference="diff_inverse",
        increment="increment",
        inverse_increment="increment_inverse",
        equality="equals",
        inverse_equality="equals_inverse",
        conjunction="and",
        erase_conjunction="erase",
        cnot="cnot",
    )
    assert graph.leaf_counts == normalized
    cost, expected = native.resource_gates(), get_cost_value(original, QECGatesCost())
    for field in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert sympy.simplify(getattr(cost, field) - getattr(expected, field)) == 0
    if not symbolic:
        assert cost.toffoli == 28 and cost.cswap == 3072
        assert cost.and_bloq == cost.measurement == 6414 and cost.clifford == 71335
    with pytest.raises(
        DecomposeTypeError if symbolic else DecomposeNotImplementedError
    ):
        original.decompose_bloq()


@pytest.mark.parametrize("symbolic", (False, True))
def test_both_root_factories_signatures_dependency_models_and_unused_sparsity(
    symbolic: bool,
) -> None:
    native = (
        symbolic_kikuchi_nonzero_index()
        if symbolic
        else example_kikuchi_nonzero_index()
    )
    original = (
        source._kikuchi_nonzero_index_symb()
        if symbolic
        else source._kikuchi_nonzero_index()
    )
    assert native.s == original.s and native.ell == original.ell
    assert [
        (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in native.signature
    ] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.name) for r in original.signature
    ]
    calls = original.build_call_graph(SympySymbolAllocator())
    assert {type(c).__name__: v for c, v in calls.items()} == {
        "ColumnOfKthNonZeroEntry": 1,
        "IndexOfNonZeroColumn": 1,
    }
    assert native.source_graph(
        root="root", forward="forward", reverse="reverse"
    ).leaf_counts == {"forward": 1, "reverse": 1}
    cost = native.resource_gates()
    expected = get_cost_value(original, QECGatesCost())
    for field in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert sympy.simplify(getattr(cost, field) - getattr(expected, field)) == 0


def identity(name: str, signature: tuple[Register, ...]) -> FunctionRecord:
    values = tuple(
        ValueRef("entry", index, "register")
        for index, _ in enumerate(
            (register, idx) for register in signature for idx in register.all_idxs()
        )
    )
    return FunctionRecord(
        name, (BlockRecord("entry", values, return_values=values),), registers=signature
    )


def test_root_program_preserves_literal_source_positional_return_and_lifetimes() -> (
    None
):
    inst = KXorInstance(4, 2, (Constraint((0, 1), 1),))
    native = KikuchiNonZeroIndex(inst, 2, 1)
    forward = identity("forward", tuple(native.forward.signature))
    reverse = identity("reverse", tuple(native.reverse.signature))
    program = native.program(forward, reverse)
    assert verify_program_record(program) == ()
    root = next(function for function in program.functions if function.name == "main")
    block = root.blocks[0]
    calls = [op for op in block.operations if op.opcode == "call"]
    first = next(op for op in calls if op.callee == "forward")
    second = next(op for op in calls if op.callee == "reverse")
    assert second.operands == first.results
    # Source assigns (S,T,k,flag) from an (S,k,T,flag)-ordered tuple.
    # Thus returned root k is reverse k, and reverse T is released.
    assert block.return_values == second.results[:4]
    released = [op.operands[0] for op in calls[calls.index(second) + 1 :]]
    assert released == list(second.results[4:])
    assert np.asarray(native.forward.signature[0].shape).tolist() == [2]
    with pytest.raises(ValueError, match="signatures"):
        native.program(identity("bad", ()), reverse)
    symbolic = symbolic_kikuchi_nonzero_index()
    with pytest.raises(ValueError, match="concrete"):
        symbolic.program(
            replace(forward, registers=tuple(symbolic.forward.signature)),
            replace(reverse, registers=tuple(symbolic.reverse.signature)),
        )


def test_untouched_root_composite_confirms_positional_return_and_release() -> None:
    composite = source._kikuchi_nonzero_index().decompose_bloq()
    stages = tuple(composite.iter_bloqsoqs())
    forward = next(
        row for row in stages if type(row[0].bloq).__name__ == "ColumnOfKthNonZeroEntry"
    )
    reverse = next(
        row for row in stages if type(row[0].bloq).__name__ == "IndexOfNonZeroColumn"
    )
    for name, outputs in zip(("S", "k", "T", "flag"), forward[2], strict=True):
        np.testing.assert_array_equal(reverse[1][name], outputs)
    final = composite.final_soqs()
    np.testing.assert_array_equal(final["S"], reverse[2][0])
    np.testing.assert_array_equal(final["k"], reverse[2][1])
    released = [row[1]["reg"] for row in stages if type(row[0].bloq).__name__ == "Free"]
    assert set(released) == set(reverse[2][2])
    effects = [
        row[1]["q"] for row in stages if type(row[0].bloq).__name__ == "ZeroEffect"
    ]
    assert effects == [reverse[2][3]]
