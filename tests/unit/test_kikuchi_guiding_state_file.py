"""Frozen complete guiding-state declarations and coherent zero-flag execution."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import sympy
import torch
from qualtran._infra.bloq import DecomposeNotImplementedError
from qualtran.bloqs.optimization.k_xor_sat import kikuchi_guiding_state as source
from qualtran.resource_counting._bloq_counts import QECGatesCost
from qualtran.resource_counting._costing import get_cost_value

from flagquantum.algorithms import kikuchi_guiding_state as native
from flagquantum.circuit import Circuit
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import assert_source_wire_streams
from tests.unit.test_register_constant_addition import reference_arithmetic_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

FACTORIES = (
    (native.example_scope_guiding_state, source._simple_guiding_state),
    (native.symbolic_scope_guiding_state, source._simple_guiding_state_symb),
    (native.example_guiding_state, source._guiding_state),
    (native.symbolic_guiding_state, source._guiding_state_symb),
    (native.symbolic_count_guiding_state, source._guiding_state_symb_c),
)


@pytest.mark.parametrize("index", range(5))
def test_all_factories_signatures_properties_and_seven_resource_categories(
    index: int,
) -> None:
    build, reference = FACTORIES[index]
    actual, expected = build(), reference()
    for field in ("target_bitsize", "eps"):
        assert getattr(actual, field) == getattr(expected, field)
    for name in ("selection_registers", "junk_registers", "signature"):
        assert [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name)
            for r in getattr(actual, name)
        ] == [
            (r.name, r.bitsize, r.shape_symbolic, r.side.name)
            for r in getattr(expected, name)
        ]
    if isinstance(actual, native.GuidingState):
        assert isinstance(expected, source.GuidingState)
        for field in (
            "c",
            "sort_ancilla_bitsize",
            "ancilla_bitsize",
            "coeff_good",
            "amplitude_good_part",
        ):
            assert getattr(actual, field) == getattr(expected, field)
        assert actual.scope_guiding_state.eps == expected.simple_guiding_state.eps
        assert actual.source_graph(
            root="root",
            preparation="prep",
            sorting="sort",
            duplicates="duplicates",
            uncompute="erase",
            flag_predicate="mcx",
            flag_x="x",
        ).leaf_counts == {
            "prep": actual.c,
            "sort": 1,
            "duplicates": 1,
            "erase": 1,
            "mcx": 1,
            "x": 1,
        }
    else:
        assert isinstance(expected, source.SimpleGuidingState)
        assert actual.phasegrad_bitsize == expected.phasegrad_bitsize
        assert actual.source_graph(root="root", preparation="prep").leaf_counts == {
            "prep": 1
        }
    left, right = actual.resource_gates(), get_cost_value(expected, QECGatesCost())
    for field in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert sympy.simplify(getattr(left, field) - getattr(right, field)) == 0


@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_probabilistic_uncompute_literal_source_stream_and_separate_costs(
    width: int,
) -> None:
    q = tuple(range(1, width + 1))
    flag = width + 1
    work = tuple(range(width + 2, 2 * width + 1))
    bits = tuple(range(40, 40 + len(work)))
    op = native.ProbabilisticUncompute(width).bind(q, flag, work, bits)
    actual = DynamicCircuit(2 * width + 2).append(op).to_ir().instructions
    expected = reference_arithmetic_gates(
        source.ProbabilisticUncompute(width),
        {"q": q, "flag": (flag,)},
        work,
        dict(zip(work, bits, strict=True)),
    )
    assert_source_wire_streams(actual, expected)
    counts = Counter(g.name for g in actual)
    source_model = get_cost_value(source.ProbabilisticUncompute(width), QECGatesCost())
    model = native.ProbabilisticUncompute(width).resource_gates()
    for field in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(model, field) == getattr(source_model, field)
    assert counts["t"] + counts["tdg"] == 4 * (width - 1)
    assert counts["measure"] == counts["reset"] == width - 1


@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_all_measurement_branches_complex_amplitudes_clean_work_and_spectator(
    width: int,
) -> None:
    q = tuple(range(1, width + 1))
    flag = width + 1
    work = tuple(range(width + 2, 2 * width + 1))
    bits = tuple(range(len(work)))
    qubits = 2 * width + 1
    logical = torch.arange(1 << (width + 2))
    amplitudes = torch.complex(logical.double() + 1, 3 * logical.double() - 7)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << qubits, dtype=torch.complex128)
    state[logical << len(work)] = amplitudes
    ideal_h = tuple(Instruction("h", (wire,)) for wire in q)
    transformed = Circuit(qubits, inputs=state, dtype=torch.complex128)
    for wire in q:
        transformed.h(wire)
    expected = transformed.state().flatten()
    addresses = torch.arange(expected.numel())
    mask = ((addresses >> len(work)) & ((1 << (width + 1)) - 1)) < 2
    permutation = addresses ^ (mask.long() << len(work))
    expected = expected[permutation]
    op = native.ProbabilisticUncompute(width).bind(q, flag, work, bits)
    actual = DynamicCircuit(qubits).append(op).to_ir().instructions
    assert actual[:width] == ideal_h
    for outcomes in product((0, 1), repeat=len(work)):
        branch = forced_branch(state, actual, outcomes)
        torch.testing.assert_close(
            branch, expected / 2 ** (len(work) / 2), atol=2e-13, rtol=0
        )
        if work:
            assert torch.count_nonzero(branch.reshape(-1, 1 << len(work))[:, 1:]) == 0


def test_symbolic_uncompute_and_explicit_source_physical_boundaries() -> None:
    n = sympy.Symbol("n", positive=True, integer=True)
    original = get_cost_value(source.ProbabilisticUncompute(n), QECGatesCost())
    actual = native.ProbabilisticUncompute(n).resource_gates()
    for field in ("and_bloq", "clifford", "measurement"):
        assert sympy.simplify(getattr(actual, field) - getattr(original, field)) == 0
    with pytest.raises(DecomposeNotImplementedError):
        source._simple_guiding_state().decompose_bloq()
    with pytest.raises(ValueError, match="concrete"):
        native.symbolic_scope_guiding_state().state_preparation
    prep = native.example_scope_guiding_state().state_preparation
    source_prep = source._simple_guiding_state()._state_prep_bloq
    assert isinstance(source_prep.sparse_indices, tuple)
    assert isinstance(source_prep.nonzero_coeffs, tuple)
    assert prep.target_width == source_prep.target_bitsize
    assert prep.phase_width == source_prep.phase_bitsize
    assert dict(zip(prep.indices, prep.coefficients, strict=True)) == dict(
        zip(source_prep.sparse_indices, source_prep.nonzero_coeffs, strict=True)
    )


@pytest.mark.parametrize(
    "width,word,flag,work,bits",
    (
        (0, (), 1, (), ()),
        (2, (0,), 1, (), ()),
        (2, (0, 1), 1, (2,), (0,)),
        (2, (0, 1), 2, (), ()),
    ),
)
def test_invalid_bindings_fail_before_append(
    width: int,
    word: tuple[int, ...],
    flag: int,
    work: tuple[int, ...],
    bits: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        native.ProbabilisticUncompute(width).bind(word, flag, work, bits)


def test_complete_source_and_original_tests_keep_frozen_blob_identity() -> None:
    from hashlib import sha1
    from pathlib import Path

    assert source.__file__ is not None
    for name, blob in (
        ("kikuchi_guiding_state.py", "0a96fbb1e60e21ece4d6c33f504a7187fae986b1"),
        ("kikuchi_guiding_state_test.py", "c215aad2516a4c61cc745c8909ef90f55fe74e5b"),
    ):
        raw = Path(source.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_full_root_original_child_operands_and_native_typed_admission_boundary() -> (
    None
):
    from qualtran.bloqs.arithmetic.lists.sort_in_place import SortInPlace
    from qualtran.resource_counting._call_graph import SympySymbolAllocator

    from flagquantum.compiler.registers.sorting.in_place_sort_declaration import (
        in_place_sort_signature,
    )
    from flagquantum.core.ir.program.register_types import (
        quantum_bit_type,
        unsigned_register_type,
    )
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.core.ir.program.signatures import Signature
    from tests.unit.test_kikuchi_adjacency_file import identity

    original = source._guiding_state()
    allocator: Any = SympySymbolAllocator
    calls = original.build_call_graph(allocator())
    assert len(calls) == 6
    for child, multiplicity in calls.items():
        name = type(child).__name__
        assert multiplicity == (original.c if name == "SimpleGuidingState" else 1)
        from qualtran.bloqs.arithmetic.lists.has_duplicates import HasDuplicates
        from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlX

        if isinstance(child, (SortInPlace, HasDuplicates)):
            assert (
                child.l == original.ell
                and child.dtype.num_qubits == original.inst.index_bitsize
            )
        elif isinstance(child, source.SimpleGuidingState):
            assert child.eps == original.eps / original.c
        elif isinstance(child, source.ProbabilisticUncompute):
            assert child.bitsize == original.sort_ancilla_bitsize
        elif isinstance(child, MultiControlX):
            assert child.cvs == (0, 0)
        else:
            assert name == "XGate"
    # The source root itself decomposes, but its sorting child cannot.
    assert original.decompose_bloq().signature == original.signature
    with pytest.raises(DecomposeNotImplementedError):
        SortInPlace(original.ell, original._index_dtype).decompose_bloq()
    actual = native.example_guiding_state()
    signatures = (
        actual.scope_guiding_state.signature,
        in_place_sort_signature(4, unsigned_register_type(2)),
        Signature(
            (
                Register("xs", unsigned_register_type(2), shape=(4,)),
                Register("flag", quantum_bit_type()),
            )
        ),
        native.ProbabilisticUncompute(8).signature,
        Signature(
            (
                Register("controls", quantum_bit_type(), shape=(2,)),
                Register("target", quantum_bit_type()),
            )
        ),
        Signature.build(q=1),
    )
    bodies = tuple(
        identity(f"child_{index}", tuple(signature))
        for index, signature in enumerate(signatures)
    )
    # Identity bodies establish typed wiring with the admitted bounded pi array.
    from flagquantum.core.ir.program import verify_program_record

    program = actual.program(*bodies)
    assert verify_program_record(program) == ()
    root = next(function for function in program.functions if function.name == "main")
    assert root.registers == tuple(actual.signature)
    from dataclasses import replace

    # The literal source root invokes packed adapters. Their enclosed child
    # bodies retain exact identity modulo deterministic namespace isolation.
    aliases = {
        function.name: body.name
        for body in bodies
        for function in program.functions
        if replace(function, name=body.name) == body
    }
    child_calls = Counter(
        aliases[operation.callee]
        for function in program.functions
        for block in function.blocks
        for operation in block.operations
        if operation.callee in aliases
    )
    assert child_calls == {
        "child_0": 2,
        **{f"child_{index}": 1 for index in range(1, 6)},
    }
    with pytest.raises(ValueError, match="concrete"):
        native.symbolic_count_guiding_state().program(*bodies)
