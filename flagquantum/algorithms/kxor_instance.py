# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 optimization/k_xor_sat/kxor_instance.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Classical k-XOR instances shared by scope-loading and Kikuchi algorithms.

Scopes contain zero-based variable indices. Packed scopes concatenate fixed-width
indices in big-endian order. This module owns classical preprocessing only.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Sequence
from functools import cached_property
from typing import TypeAlias

import numpy as np
import sympy
from attrs import evolve, field, frozen
from numpy.typing import NDArray

from ..core.value_math import SymbolicInt, bit_length, ceil, log2
from ..core.value_parameters import is_symbolic_value as is_symbolic
from ..simulation.value_dimensions import SymbolicSequenceLength as HasLength
from ..simulation.value_dimensions import value_length as slen

Scope: TypeAlias = tuple[int, ...] | HasLength
"""A subset of variables"""


def _sort_scope(scope: Scope) -> Scope:
    if isinstance(scope, HasLength):
        return scope
    return tuple(sorted(scope))


@frozen
class Constraint:
    """A single kXOR constraint.

    Definition 2.1.

    Note: n, k are not stored here, but only in the instance.

    Attributes:
        scope: the scope - subset of `[n]` of size k.
        b: +1 or -1.
    """

    S: Scope = field(converter=_sort_scope)
    b: SymbolicInt = field()

    @classmethod
    def random(cls, n: int, k: int, *, rng: np.random.Generator) -> Constraint:
        """Single random constraint, Notation 2.3."""
        scope: tuple[int, ...] = tuple(rng.choice(n, k, replace=False))
        b: SymbolicInt = rng.choice([-1, +1])
        return cls(scope, b)

    @classmethod
    def random_planted(
        cls,
        n: int,
        k: int,
        *,
        rho: float,
        z: NDArray[np.int64],
        rng: np.random.Generator,
    ) -> Constraint:
        """Single planted constraint, Notation 2.4."""
        scope: tuple[int, ...] = tuple(rng.choice(n, k, replace=False))
        eta = (-1) ** (
            rng.random() < (1 + rho) / 2
        )  # Source sign convention has expectation -rho.
        unplanted = cls(scope, 1)  # supporting constraint to evaluate z^scope
        b = eta * unplanted.evaluate(z)
        return cls(scope, b)

    @classmethod
    def symbolic(cls, n: SymbolicInt, ix: int) -> Constraint:
        return cls(HasLength(n), sympy.Symbol(f"b_{ix}"))

    def is_symbolic(self) -> bool:
        return is_symbolic(self.S, self.b)

    def evaluate(self, x: NDArray[np.int64]) -> int:
        assert isinstance(self.S, tuple)
        result: int = np.prod(x[np.array(self.S)])
        return result


@frozen
class KXorInstance:
    r"""A kXOR instance $\mathcal{I}$.

    Definition 2.1: A kXOR instance $\mathcal{I}$ over variables indexed by $[n]$
    consists of a multiset of constraints $\mathcal{C} = (scope, b)$, where each
    $scope \subseteq [n]$ has cardinality $k$ and is called a scope, and each right-hand side
    satisfies $b \in \{\pm 1\}$.

    Examples:
        >>> instance = example_kxor_instance()
        >>> instance.max_rhs
        2

    Attributes:
        n: number of variables.
        k: number of variables per clause.
        constraints: a tuple of `m` Constraints.
        max_rhs: maximum value of the RHS polynomial $B_\mathcal{I}(scope)$.
            see default constructor for default value. In case the instance is symbolic,
            the user can specify an expression for this, to avoid the default value.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Definition 2.1.
    """

    n: SymbolicInt
    k: SymbolicInt
    constraints: tuple[Constraint, ...] | HasLength
    max_rhs: SymbolicInt = field()

    @max_rhs.default
    def _default_max_rhs(self) -> SymbolicInt:
        """With very high probability, the max entry will be quite small.

        This is a classical preprocesing step. Time $m$.
        """
        if is_symbolic(self.constraints):
            return 2
        assert isinstance(self.constraints, tuple)
        if is_symbolic(*self.constraints):
            # user did not provide a value, assume some small constant
            return 2

        # instance is not symbolic, so we can compute the exact value.
        assert isinstance(self.batched_scopes, tuple)
        return max(abs(b) for _, b in self.batched_scopes)

    @cached_property
    def m(self) -> SymbolicInt:
        return slen(self.constraints)

    @classmethod
    def random_instance(
        cls,
        n: int,
        m: int,
        k: int,
        *,
        planted_advantage: float = 0,
        rng: np.random.Generator,
    ) -> KXorInstance:
        r"""Generate a random kXOR instance with the given planted advantage.

        `planted_advantage=0` generates random instances, and `1` generates a
        linear system with a solution.

        Args:
            n: number of variables
            m: number of clauses
            k: number of terms per clause
            planted_advantage: $\rho$
            rng: random generator

        References:
            [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
            Notation 2.4.
        """
        # planted vector
        z = rng.choice([-1, +1], size=n)

        # constraints
        constraints = tuple(
            Constraint.random_planted(n=n, k=k, rho=planted_advantage, z=z, rng=rng)
            for _ in range(m)
        )

        return cls(n=n, k=k, constraints=constraints)

    @classmethod
    def symbolic(
        cls, n: SymbolicInt, m: SymbolicInt, k: SymbolicInt, *, max_rhs: SymbolicInt = 2
    ) -> KXorInstance:
        """Create a symbolic instance with n variables, m constraints."""
        constraints = HasLength(m)
        return cls(n=n, k=k, constraints=constraints, max_rhs=max_rhs)

    def is_symbolic(self) -> bool:
        if is_symbolic(self.n, self.m, self.k, self.constraints):
            return True
        assert isinstance(self.constraints, tuple)
        return is_symbolic(*self.constraints)

    def subset(self, indices: Sequence[int] | HasLength) -> "KXorInstance":
        """Pick a subset of clauses defined by the set of indices provided."""
        if self.is_symbolic() or is_symbolic(indices):
            return evolve(self, constraints=HasLength(slen(indices)))
        assert isinstance(self.constraints, tuple)

        assert not isinstance(indices, HasLength)
        constraints = tuple(self.constraints[i] for i in indices)
        return evolve(self, constraints=constraints)

    @cached_property
    def index_bitsize(self) -> SymbolicInt:
        """number of bits required to represent the index of a variable, i.e. `[n]`

        We assume zero-indexing.
        """
        return ceil(log2(self.n))

    @cached_property
    def num_unique_constraints(self) -> SymbolicInt:
        return slen(self.batched_scopes)

    @cached_property
    def batched_scopes(self) -> tuple[tuple[Scope, SymbolicInt], ...] | HasLength:
        r"""Group all the constraints by Scope, and add up the $b$ values.

        A scope is a subset of variables of size $k$.
        Given an instance $\mathcal{I} = \{ (S_i, b_i) \}_i$, this function groups
        all equal sets $S_i$ by summing up the rhs-values $b_i$ corresponding to them.
        The resulting sequence therefore has all unique scopes.

        This is a classical preprocessing step. Time $k m \log m$.

        See the reference for the definition of the distinct scopes.

        References:
            [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
            Theorem 4.17, proof paragraph 2.
        """
        if self.is_symbolic():
            return HasLength(self.m)

        assert isinstance(self.constraints, tuple)

        batches: dict[Scope, SymbolicInt] = defaultdict(lambda: 0)
        for con in self.constraints:
            assert isinstance(con.S, tuple)
            batches[con.S] += con.b

        batches_sorted = sorted(batches.items(), key=lambda c: c[1])
        return tuple(batches_sorted)

    @cached_property
    def rhs_sum_bitsize(self) -> SymbolicInt:
        r"""number of bits to represent the RHS polynomial $B_{\mathcal{I}}(scope)$."""
        return bit_length(2 * self.max_rhs)

    def scope_as_int(self, scope: Scope) -> int:
        r"""Convert a scope into a single integer.

        Given a scope `scope = (x_1, x_2, ..., x_k)`, and a bitsize `r` for each index,
        the integer representation is given by concatenating `r`-bit unsigned repr
        of each `x_i`. That is, $\sum_i r^{k - i} x_i$.

        This uses Big-endian representation, for all native scope-loading consumers.

        The bitsize `r` is picked as `ceil(log(n))` for an n-variable instance.
        """
        assert not is_symbolic(scope)
        assert isinstance(scope, tuple)

        bitsize = self.index_bitsize
        assert isinstance(bitsize, int)

        result = 0
        for x in scope:
            result = (result << bitsize) + x
        return result

    def brute_force_sparsity(self, ell: int) -> int:
        r"""Compute the sparsity of the Kikuchi matrix with parameter $\ell$ by brute force.

        Takes time `O(C(n, l) * m * l)`. Extremely slow, use with caution.
        """
        assert isinstance(self.n, int)
        s = 0
        scopes = self.batched_scopes
        assert isinstance(scopes, tuple)
        for scope in itertools.combinations(range(self.n), ell):
            nz = 0
            for constraint_scope, _ in scopes:
                assert isinstance(constraint_scope, tuple)
                neighbor = set(scope).symmetric_difference(constraint_scope)
                if len(neighbor) == ell:
                    nz += 1
            s = max(s, nz)
        return s


def example_kxor_instance() -> KXorInstance:
    n, k = 10, 4
    cs = (
        Constraint((0, 1, 2, 3), -1),
        Constraint((0, 2, 4, 5), 1),
        Constraint((0, 3, 4, 5), 1),
        Constraint((0, 3, 4, 5), 1),
        Constraint((1, 2, 3, 4), -1),
        Constraint((1, 3, 4, 5), -1),
        Constraint((1, 3, 4, 5), -1),
        Constraint((2, 3, 4, 5), 1),
    )
    inst = KXorInstance(n, k, cs)
    return inst
