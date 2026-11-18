"""Compare the complete classical k-XOR source and its unmodified test file."""

import itertools
import runpy
from hashlib import sha1
from pathlib import Path

import numpy as np
import pytest
import sympy
from qualtran.bloqs.optimization.k_xor_sat import kxor_instance as source
from qualtran.symbolics import HasLength

from flagquantum.algorithms import kxor_instance as native
from flagquantum.simulation.value_dimensions import SymbolicSequenceLength

_BLOBS = {
    "kxor_instance.py": "70f08821a68cb1c708f6544ba7cf5170dc223e0e",
    "kxor_instance_test.py": "9b80e6b70b5af53a111792989981fa4667e7cb11",
}

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_source_and_test_are_frozen() -> None:
    assert source.__file__ is not None
    directory = Path(source.__file__).parent
    for name, expected in _BLOBS.items():
        raw = (directory / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected


def test_all_concrete_methods_multisets_subsets_packing_and_sparsity() -> None:
    scopes = tuple(itertools.combinations(range(4), 2))
    for first, second, rhs in itertools.product(scopes, scopes, (-1, 1)):
        constraints = ((tuple(reversed(first)), rhs), (second, -1), (first, 1))
        original = source.KXorInstance(
            4, 2, tuple(source.Constraint(scope, b) for scope, b in constraints)
        )
        migrated = native.KXorInstance(
            4, 2, tuple(native.Constraint(scope, b) for scope, b in constraints)
        )
        assert migrated.batched_scopes == original.batched_scopes
        assert migrated.max_rhs == original.max_rhs
        assert migrated.m == original.m == 3
        assert migrated.num_unique_constraints == original.num_unique_constraints
        assert migrated.index_bitsize == original.index_bitsize == 2
        assert migrated.rhs_sum_bitsize == original.rhs_sum_bitsize
        assert not migrated.is_symbolic()
        for scope in scopes:
            assert migrated.scope_as_int(scope) == original.scope_as_int(scope)
        for ell in range(5):
            assert migrated.brute_force_sparsity(ell) == original.brute_force_sparsity(
                ell
            )
        for assignment in itertools.product((-1, 1), repeat=4):
            values = np.array(assignment, dtype=np.int64)
            assert isinstance(migrated.constraints, tuple)
            for a, b in zip(migrated.constraints, original.constraints, strict=True):
                assert a.S == b.S and a.b == b.b
                assert a.evaluate(values) == b.evaluate(values)
        for indices in ((), (0,), (2, 1), (1, 1)):
            native_subset, source_subset = migrated.subset(indices), original.subset(
                indices
            )
            assert native_subset.batched_scopes == source_subset.batched_scopes
            # attrs.evolve preserves the parent's supplied/default max_rhs.
            assert native_subset.max_rhs == source_subset.max_rhs == original.max_rhs


def test_symbolic_constraints_lengths_expressions_and_subset() -> None:
    n, m, k, count = sympy.symbols("n m k count", positive=True, integer=True)
    a, b = native.Constraint.symbolic(k, 7), source.Constraint.symbolic(k, 7)
    assert isinstance(a.S, SymbolicSequenceLength) and isinstance(b.S, HasLength)
    assert a.S.length == b.S.n == k
    assert a.b == b.b and a.is_symbolic() and b.is_symbolic()
    x, y = native.KXorInstance.symbolic(n, m, k), source.KXorInstance.symbolic(n, m, k)
    for field in (
        "n",
        "k",
        "m",
        "max_rhs",
        "index_bitsize",
        "rhs_sum_bitsize",
        "num_unique_constraints",
    ):
        assert getattr(x, field) == getattr(y, field)
    assert x.is_symbolic() and y.is_symbolic()
    assert isinstance(x.batched_scopes, SymbolicSequenceLength)
    assert x.batched_scopes.length == y.batched_scopes.n
    assert x.subset(SymbolicSequenceLength(count)).m == y.subset(HasLength(count)).m
    assert x.subset((0, 1)).m == y.subset((0, 1)).m == 2
    concrete_a = native.KXorInstance(4, 2, (a,))
    concrete_b = source.KXorInstance(4, 2, (b,))
    assert concrete_a.is_symbolic() == concrete_b.is_symbolic()
    assert concrete_a.max_rhs == concrete_b.max_rhs == 2


@pytest.mark.parametrize("rho", (0.0, 0.8, 1.0))
def test_random_factories_preserve_rng_stream_and_original_planted_sign(
    rho: float,
) -> None:
    a_rng, b_rng = np.random.default_rng(19), np.random.default_rng(19)
    a = native.Constraint.random(12, 4, rng=a_rng)
    b = source.Constraint.random(12, 4, rng=b_rng)
    assert a.S == b.S and a.b == b.b
    assert isinstance(a.S, tuple)
    assert type(a.S[0]) is type(b.S[0])
    z = np.ones(12, dtype=np.int64)
    a = native.Constraint.random_planted(12, 4, rho=rho, z=z, rng=a_rng)
    b = source.Constraint.random_planted(12, 4, rho=rho, z=z, rng=b_rng)
    assert a.S == b.S and a.b == b.b
    if rho == 1:
        assert a.b == -1  # The frozen source has expectation -rho.
    x = native.KXorInstance.random_instance(12, 30, 4, planted_advantage=rho, rng=a_rng)
    y = source.KXorInstance.random_instance(12, 30, 4, planted_advantage=rho, rng=b_rng)
    assert isinstance(x.constraints, tuple)
    assert [(c.S, c.b) for c in x.constraints] == [(c.S, c.b) for c in y.constraints]
    assert x.batched_scopes == y.batched_scopes and x.max_rhs == y.max_rhs
    assert a_rng.bit_generator.state == b_rng.bit_generator.state


def test_example_and_original_empty_cancelled_scope_boundaries() -> None:
    x, y = native.example_kxor_instance(), source.example_kxor_instance()
    assert x.batched_scopes == y.batched_scopes
    assert x.max_rhs == y.max_rhs == 2
    for owner in (native, source):
        with pytest.raises(ValueError, match="max"):
            owner.KXorInstance(4, 2, ())
        cancelled = owner.KXorInstance(
            4, 2, (owner.Constraint((0, 1), 1), owner.Constraint((0, 1), -1))
        )
        assert cancelled.batched_scopes == (((0, 1), 0),)
        assert cancelled.max_rhs == 0 and cancelled.rhs_sum_bitsize == 0
        assert cancelled.num_unique_constraints == 1
        assert cancelled.brute_force_sparsity(2) == 1
        with pytest.raises(IndexError):
            owner.Constraint((), 1).evaluate(np.ones(4, dtype=np.int64))


@pytest.mark.parametrize("rho", (0.0, 0.8, 0.9))
def test_complete_original_statistical_test_on_both_owners(rho: float) -> None:
    assert source.__file__ is not None
    path = Path(source.__file__).with_name("kxor_instance_test.py")
    original_test = runpy.run_path(str(path))["test_max_rhs"]
    original_test(rho)
    original_test.__globals__["KXorInstance"] = native.KXorInstance
    original_test(rho)
