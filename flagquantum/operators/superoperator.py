"""A finite sum of left and right multiplication actions on a matrix.

A superoperator is a linear map on the space of matrices, written as a sum of
terms

    coefficient * left @ state @ right

where a missing ``left`` means the term acts only from the right and a missing
``right`` means it acts only from the left. Both are present in the Lindblad
dissipator's ``L rho L^dag`` term; only one is present in ``-i H rho`` and in the
``-1/2 {L^dag L, rho}`` anticommutator terms. The algebra exists so that a
generator can be assembled from those actions, inspected, and applied without
writing a density-form expression by hand.

The factors are *matrix operators*, not necessarily dense matrices. A factor
must provide ``dtype``, ``dimension`` (for a ``torch.Tensor`` these are ``dtype``
and ``shape[0]``), ``__matmul__`` and ``__rmatmul__`` with a square state, and a
``dense()`` matrix. ``torch.Tensor`` is one implementation;
:class:`flagquantum.simulation.matrix_free_hamiltonian.PauliSum` is the other,
and a matrix-free ``H`` therefore joins a term without ever being materialized.
Only :meth:`SuperOperator.adjoint` additionally requires ``adjoint()`` from a
non-tensor factor, because a factor that can be applied cannot necessarily be
reversed; a factor that is never adjointed does not need it.

The adjoint is a statement about the term list, not about a dense matrix:
:meth:`SuperOperator.adjoint` returns the sum whose terms carry the conjugated
coefficient and the adjoint of each factor on the side that factor already
occupied, because the Hilbert--Schmidt adjoint of ``rho -> A rho B`` is
``sigma -> A^dag sigma B^dag``. The sides do not swap. That map is what carries
a cost gradient backwards through a generator, so a generator no longer has to
be written twice to be differentiated.

The dense form uses the row-major vectorization the rest of the repository uses,
``vec(A rho B) = (A (x) B^T) vec(rho)``, so the right factor is transposed and
never conjugate-transposed. :meth:`SuperOperator.dense` is the map's matrix in the
matrix units; :meth:`SuperOperator.matrix_in_basis` asks the same question of any
orthonormal basis of matrices, which is how a channel's spectrum and its inverse
are read in the Pauli basis. A Kraus list enters the algebra through
:meth:`SuperOperator.from_kraus`, so a channel is expressed as a term list and
transformed without a representation of its own.

:meth:`SuperOperator.choi` reads the same map in a third index order: the Choi
matrix is the object that carries the map's *positivity*, not merely its action,
because ``sum_ij E_ij (x) E(E_ij)`` is positive semidefinite exactly when the map
is completely positive. The two index orders are related by a reshuffle of the
same ``d**4`` entries and by nothing else, so the Choi matrix of a channel is the
channel and cannot drift from it. :meth:`SuperOperator.is_completely_positive`
asks that question and refuses to answer it about a map whose Choi matrix is not
Hermitian, since a map that does not preserve Hermiticity is not a candidate for
positivity in the first place. Tracing one register of that same matrix is the
map's normalization read off rather than re-derived, and
:meth:`SuperOperator.partial_trace` returns it: the input trace is ``E(I)``, the
identity exactly for a unital map, and the output trace is the identity exactly
for a trace-preserving one.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol

import torch

__all__ = ("DEFAULT_DENSE_MATRIX_BYTES", "SuperOperator")

DEFAULT_DENSE_MATRIX_BYTES = 16 * 1024 * 1024
# Byte ceiling above which :meth:`SuperOperator.dense` refuses to allocate.
#
# The dense form is ``(2**n)**4`` entries, so it is a reference for small systems and
# the exact generator of a dense matrix exponential, never a production path. The
# default ceiling admits a four-wire map and refuses the next size up.

_BASIS_TOLERANCE_IN_EPSILON = 8.0
# How far a basis may sit from orthonormal under ``trace(B_i^dag B_j)``, counted in
# units of the factor dtype's own epsilon, before the transform refuses it.
#
# The deviation is what ``P^dag @ dense() @ P`` is wrong by when ``P^dag`` is not
# ``P^-1``, so the threshold is an accuracy statement rather than a formatting one,
# and the currency it is stated in is the dtype's rounding rather than an absolute
# number -- a single threshold in absolute units would either refuse a correct
# complex64 basis or admit a broken complex128 one. Measured margins: the normalized
# Pauli basis deviates by 2.22e-16 (one epsilon) in complex128 and 5.96e-08 (half an
# epsilon) in complex64, while the smallest non-orthonormality a caller can introduce
# by accident -- handing over the unnormalized Pauli words -- is exactly ``d - 1``,
# 1.0 at one wire and 3.0 at two. Eight epsilons sit roughly sixteen times above the
# worst accepted basis and six to fifteen orders below the worst refused one, so no
# basis is admitted or refused here by a rounding accident.

_CHOI_TOLERANCE_IN_EPSILON = 8.0
# How far the smallest Choi eigenvalue of a completely positive map may sit below
# zero -- and how far a Choi matrix may sit from Hermitian -- before the certificate
# refuses, counted in units of the map dtype's own epsilon times the Choi matrix's
# own spectral scale.
#
# The scale is required, not a convenience. Measured over random Kraus lists (hence
# completely positive maps by construction) at dimensions 2, 3, 4, 5, 8, 11, and 16
# with 1, 2, 3, 5, 9, and 17 operators, forty draws each, the negative deviation of
# the smallest eigenvalue grows without bound in *absolute* units -- 1.98 epsilons at
# two dimensions to 27.44 at sixteen in complex64, and 23.24 at sixteen in
# complex128 -- so an absolute tolerance would either refuse a correct channel at one
# size or admit a wrong one at another. Divided by the largest absolute eigenvalue of
# the same matrix the ratio stays bounded: the worst of the whole sweep is 1.751 in
# complex64 and 1.930 in complex128, and no single size carries it. Eight epsilons
# therefore clear every legitimate channel by a factor of 4.1, and the nearest genuine
# violation is not close: the transpose map's Choi eigenvalue is exactly ``-1`` against
# a scale of exactly ``1``, so the margin there is fifteen orders. The anti-Hermitian
# part of the same draws stays under 1.37 of the largest entry, which the same
# tolerance clears because the largest entry never exceeds the spectral scale.


class OperatorFactor(Protocol):
    """The matrix-operator contract a superoperator term is built from.

    A factor acts on a square matrix from one side. ``torch.Tensor`` provides
    every member directly apart from ``dimension``, which is read from the shape.

    ``adjoint`` is required only by :meth:`SuperOperator.adjoint`, so a factor
    that is never adjointed does not have to supply it and construction does not
    demand it.
    """

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def dimension(self) -> int: ...

    def __matmul__(self, state: torch.Tensor) -> torch.Tensor: ...

    def __rmatmul__(self, state: torch.Tensor) -> torch.Tensor: ...

    def dense(self) -> torch.Tensor: ...

    def adjoint(self) -> OperatorFactor: ...


Factor = Any
# A superoperator factor: a ``torch.Tensor`` or an :class:`OperatorFactor`.

_Term = tuple[complex, Factor | None, Factor | None]


def _factor_dtype(operator: object) -> torch.dtype:
    dtype = getattr(operator, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"an operator factor must expose a torch dtype, got {operator!r}"
        )
    if not dtype.is_complex:
        raise ValueError(f"an operator factor must have a complex dtype, got {dtype}")
    return dtype


def _factor_dimension(operator: object) -> int:
    if isinstance(operator, torch.Tensor):
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise ValueError(
                "a tensor operator factor must be a square matrix, got shape "
                f"{tuple(operator.shape)}"
            )
        return int(operator.shape[0])
    for name in ("__matmul__", "__rmatmul__", "dense"):
        if not callable(getattr(operator, name, None)):
            raise ValueError(
                f"an operator factor must support {name} with a square state"
            )
    dimension = getattr(operator, "dimension", None)
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError(
            "an operator factor must declare a positive integer dimension, got "
            f"{dimension!r}"
        )
    return dimension


def _dense_matrix(operator: Factor) -> torch.Tensor:
    if isinstance(operator, torch.Tensor):
        return operator
    matrix = operator.dense()
    if not isinstance(matrix, torch.Tensor):
        raise ValueError(
            f"an operator factor must return a tensor from dense(), got {matrix!r}"
        )
    return matrix


def _factor_adjoint(operator: Factor) -> Factor:
    if isinstance(operator, torch.Tensor):
        return torch.conj(operator).T
    if not callable(getattr(operator, "adjoint", None)):
        raise ValueError(
            "a non-tensor operator factor must support adjoint() to be adjointed, "
            f"got {operator!r}"
        )
    return operator.adjoint()


def _coefficient(value: object) -> complex:
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise ValueError(f"a superoperator coefficient must be a scalar, got {value!r}")
    return complex(value)


def _basis_entry(entry: object, dimension: int, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(entry, torch.Tensor):
        raise ValueError(f"a basis element must be a tensor, got {entry!r}")
    shape = tuple(int(size) for size in entry.shape)
    if shape != (dimension, dimension):
        raise ValueError(
            f"every basis element must be a {dimension} x {dimension} matrix, got "
            f"shape {shape}"
        )
    if entry.dtype != dtype:
        raise ValueError(
            f"every basis element must share the superoperator dtype {dtype}, got "
            f"{entry.dtype}"
        )
    return entry


def _choi_eigenvalues(
    matrix: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return a Choi matrix's eigenvalues, its eigenvectors, and the tolerance.

    Both callers of this read the same spectrum -- one asks whether the smallest
    eigenvalue is negative, the other keeps the eigenvectors of the non-zero ones
    -- so the eigenproblem and, more importantly, the tolerance rule are stated
    once. A second copy of the rule would be a second answer to what a zero Choi
    eigenvalue is, and the two answers would disagree exactly on the rank-deficient
    channels the rule exists for.

    The tolerance is relative to the matrix's own spectral scale rather than
    absolute, for the reason the certificate documents: the deviation of a
    legitimate rank-deficient channel's smallest eigenvalue from zero grows with
    the channel's scale, so no absolute number admits every correct channel and
    refuses the transpose map.

    A map whose Choi matrix is not Hermitian is refused here rather than answered,
    because every question this spectrum is asked about such a matrix -- positivity,
    a Kraus representation -- presumes the map takes Hermitian states to Hermitian
    states, and a generator does not.
    """

    adjoint = matrix.conj().transpose(0, 1)
    # ``eigh`` reads one triangle, so the matrix is symmetrized rather than handed
    # over as it stands: the eigenvalues must describe the same matrix the
    # Hermiticity check below measures.
    eigenvalues, eigenvectors = torch.linalg.eigh((matrix + adjoint) / 2)
    scale = float(eigenvalues.abs().max())
    tolerance = _CHOI_TOLERANCE_IN_EPSILON * float(torch.finfo(dtype).eps) * scale
    deviation = float(torch.max(torch.abs(matrix - adjoint)))
    if deviation > tolerance:
        raise ValueError(
            "this map does not preserve Hermiticity: its Choi matrix is "
            f"{deviation:.3e} away from Hermitian, above this certificate's "
            f"{tolerance:.3e} tolerance at a Choi scale of {scale:.3e}, so it "
            "has no positivity to report. Complete positivity is a property of "
            "maps that take Hermitian states to Hermitian states, and a "
            "generator is not one of them."
        )
    return eigenvalues, eigenvectors, tolerance


def _same_factor(left: Factor | None, right: Factor | None) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    return bool(left == right)


class SuperOperator:
    """A sum of left and right multiplication actions on a matrix.

    The instance is constructed empty and grown with ``+=``, which appends
    terms, exactly as the CUDA-Q superoperator accumulates terms. ``+`` returns
    a new instance with the concatenated terms, scalar multiplication scales
    every coefficient, and equality compares the term sequence structurally.
    Every term names at least one side, because each of the three constructors
    supplies one; a term that multiplies from neither side is unrepresentable.

    Examples:
        >>> import torch
        >>> from flagquantum.operators import SuperOperator
        >>> lowering = torch.tensor(
        ...     [[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128
        ... )
        >>> number = lowering.conj().T @ lowering
        >>> decay = SuperOperator.left_right_multiply(lowering, lowering.conj().T)
        >>> decay += SuperOperator.left_multiply((-0.5 + 0.0j) * number)
        >>> decay += SuperOperator.right_multiply((-0.5 + 0.0j) * number)
        >>> excited = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
        >>> decay.apply(excited).tolist()
        [[(1+0j), 0j], [0j, (-1+0j)]]
    """

    __slots__ = ("_dimension", "_dtype", "_terms")

    def __init__(self) -> None:
        self._terms: list[_Term] = []
        self._dimension: int | None = None
        self._dtype: torch.dtype | None = None

    @classmethod
    def _from_terms(cls, terms: list[_Term]) -> SuperOperator:
        instance = cls()
        instance._terms = list(terms)
        instance._resolve_summary()
        return instance

    @classmethod
    def left_multiply(cls, operator: Factor) -> SuperOperator:
        """Return the map ``rho -> operator @ rho``."""

        return cls._from_terms([(1.0 + 0.0j, operator, None)])

    @classmethod
    def right_multiply(cls, operator: Factor) -> SuperOperator:
        """Return the map ``rho -> rho @ operator``."""

        return cls._from_terms([(1.0 + 0.0j, None, operator)])

    @classmethod
    def left_right_multiply(cls, left: Factor, right: Factor) -> SuperOperator:
        """Return the map ``rho -> left @ rho @ right``."""

        return cls._from_terms([(1.0 + 0.0j, left, right)])

    @classmethod
    def from_kraus(cls, operators: Sequence[Factor]) -> SuperOperator:
        """Return the map ``rho -> sum_k operators[k] @ rho @ operators[k]^dag``.

        This is a Kraus list as a member of this algebra: one two-sided term per
        operator, so the result is a term list like any other and is applied,
        adjointed, added to, and transformed without being converted to a matrix
        first. It is also how a channel reaches
        :meth:`matrix_in_basis`, because a Kraus list is the representation a
        channel arrives in.

        Trace preservation is deliberately not checked. ``sum_k k^dag k == I`` is
        a claim about one family of maps rather than about the algebra, and a list
        that fails it still denotes a linear map on matrices this algebra carries.
        :class:`flagquantum.noise.KrausChannel` is the type that owns that check.

        Args:
            operators: The Kraus operators, in order, all of one shared square
                dimension and complex dtype. A non-tensor factor must also supply
                ``adjoint()``, because each operator is used beside its own
                adjoint.

        Raises:
            ValueError: If the sequence is empty, or an entry is not a usable
                factor, or the entries disagree on dimension or dtype.

        Examples:
            The completely dephasing channel, whose Kraus list is the two
            projectors ``|0><0|`` and ``|1><1|``, erases the coherence of an equal
            superposition while preserving its populations:

            >>> import torch
            >>> from flagquantum.operators import SuperOperator
            >>> zero = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128)
            >>> one = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
            >>> dephasing = SuperOperator.from_kraus((zero, one))
            >>> dephasing
            SuperOperator(terms=2, dimension=2, dtype=torch.complex128)
            >>> superposition = torch.tensor(
            ...     [[0.5, 0.5], [0.5, 0.5]], dtype=torch.complex128
            ... )
            >>> dephasing.apply(superposition).tolist()
            [[(0.5+0j), 0j], [0j, (0.5+0j)]]
        """

        entries = tuple(operators)
        if not entries:
            raise ValueError(
                "a Kraus list needs at least one operator; an empty sum is the "
                "zero map and has no dimension to act on"
            )
        return cls._from_terms(
            [(1.0 + 0.0j, operator, _factor_adjoint(operator)) for operator in entries]
        )

    @property
    def terms(self) -> tuple[_Term, ...]:
        """Return the ``(coefficient, left, right)`` terms in insertion order."""

        return tuple(self._terms)

    def _resolve_summary(self) -> None:
        dimensions: set[int] = set()
        dtypes: set[torch.dtype] = set()
        for _, left, right in self._terms:
            for factor in (left, right):
                if factor is None:
                    continue
                dimensions.add(_factor_dimension(factor))
                dtypes.add(_factor_dtype(factor))
        if len(dimensions) > 1:
            raise ValueError(
                "every superoperator factor must share one dimension, got "
                f"{sorted(dimensions)}"
            )
        if len(dtypes) > 1:
            raise ValueError(
                "every superoperator factor must share one dtype, got "
                f"{sorted(str(dtype) for dtype in dtypes)}"
            )
        self._dimension = next(iter(dimensions), None)
        self._dtype = next(iter(dtypes), None)

    def _require_summary(self) -> tuple[int, torch.dtype]:
        if self._dimension is None or self._dtype is None:
            raise ValueError(
                "an empty superoperator has no dimension or dtype; accumulate at "
                "least one term before inspecting or applying it"
            )
        return self._dimension, self._dtype

    @property
    def dimension(self) -> int:
        """Return the side length of the matrices the map acts on."""

        return self._require_summary()[0]

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype every term and every view uses."""

        return self._require_summary()[1]

    def __len__(self) -> int:
        return len(self._terms)

    def __iter__(self) -> Iterator[_Term]:
        """Iterate over the ``(coefficient, left, right)`` terms."""

        return iter(self._terms)

    def __iadd__(self, other: SuperOperator) -> SuperOperator:
        if not isinstance(other, SuperOperator):
            return NotImplemented
        appended = len(other._terms)
        self._terms = self._terms + other._terms
        try:
            self._resolve_summary()
        except ValueError:
            self._terms = self._terms[: len(self._terms) - appended]
            raise
        return self

    def __add__(self, other: SuperOperator) -> SuperOperator:
        if not isinstance(other, SuperOperator):
            return NotImplemented
        return self._from_terms(self._terms + other._terms)

    def __mul__(self, coefficient: complex) -> SuperOperator:
        scalar = _coefficient(coefficient)
        return self._from_terms(
            [(scalar * value, left, right) for value, left, right in self._terms]
        )

    __rmul__ = __mul__

    def adjoint(self) -> SuperOperator:
        """Return the Hilbert--Schmidt adjoint of the map.

        The adjoint of ``rho -> A rho B`` is ``sigma -> A^dag sigma B^dag``, so
        every term conjugates its coefficient and adjoints each factor on the
        side that factor already occupied. The sides do not swap and a one-sided
        term stays one-sided on the same side. This is the map that pairs with
        the original under ``<sigma, rho> = trace(sigma^dag rho)``, which is the
        inner product a scalar cost over matrices differentiates through.

        Raises:
            ValueError: If a non-tensor factor does not support ``adjoint()``.
        """

        return self._from_terms(
            [
                (
                    value.conjugate(),
                    None if left is None else _factor_adjoint(left),
                    None if right is None else _factor_adjoint(right),
                )
                for value, left, right in self._terms
            ]
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SuperOperator):
            return NotImplemented
        if self._dimension != other._dimension or self._dtype != other._dtype:
            return False
        if len(self._terms) != len(other._terms):
            return False
        return all(
            mine[0] == theirs[0]
            and _same_factor(mine[1], theirs[1])
            and _same_factor(mine[2], theirs[2])
            for mine, theirs in zip(self._terms, other._terms, strict=True)
        )

    __hash__ = None  # type: ignore[assignment]

    def _validate_state(self, state: torch.Tensor) -> None:
        dimension, dtype = self._require_summary()
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"a superoperator state must be a tensor, got {state!r}")
        if state.dtype != dtype:
            raise ValueError(
                "the state dtype must match the superoperator dtype: "
                f"{state.dtype} != {dtype}"
            )
        shape = tuple(int(size) for size in state.shape)
        expected: tuple[int, ...]
        if state.ndim == 2:
            expected = (dimension, dimension)
        elif state.ndim == 3:
            expected = (shape[0], dimension, dimension)
        else:
            raise ValueError(
                f"the state must have shape ({dimension}, {dimension}) or "
                f"(batch, {dimension}, {dimension}), got {shape}"
            )
        if shape != expected:
            raise ValueError(
                f"the state must have shape ({dimension}, {dimension}) or "
                f"(batch, {dimension}, {dimension}), got {shape}"
            )

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Return the map applied to a ``(d, d)`` or ``(batch, d, d)`` state.

        Raises:
            ValueError: If the state shape or dtype does not match the map.
        """

        self._validate_state(state)
        total = torch.zeros_like(state)
        for coefficient, left, right in self._terms:
            term = state
            if left is not None:
                term = (
                    left @ term
                    if isinstance(left, torch.Tensor)
                    else left.__matmul__(term)
                )
            if right is not None:
                term = (
                    term @ right
                    if isinstance(right, torch.Tensor)
                    else right.__rmatmul__(term)
                )
            total = total + coefficient * term
        return total

    def dense(self, *, max_bytes: int = DEFAULT_DENSE_MATRIX_BYTES) -> torch.Tensor:
        """Return the row-major ``d**2 x d**2`` matrix of the map.

        The form is refused above ``max_bytes`` rather than allocated, because
        the entry count is ``(2**n)**4``, so this is a small-system reference and
        a test oracle rather than a production path.

        Raises:
            ValueError: If the materialized matrix would exceed ``max_bytes``.
        """

        dimension, dtype = self._require_summary()
        side = dimension * dimension
        entries = side * side
        element_bytes = dtype.itemsize
        if entries * element_bytes > int(max_bytes):
            raise ValueError(
                f"a {dimension}-dimensional superoperator is {side} x {side} with "
                f"{entries} entries ({entries * element_bytes} bytes), above the "
                f"{int(max_bytes)}-byte ceiling; apply the map instead of "
                "materializing it"
            )
        identity = torch.eye(dimension, dtype=dtype)
        matrix = torch.zeros((side, side), dtype=dtype)
        for coefficient, left, right in self._terms:
            # Both products are materialized in the layout they will be read in,
            # because a factor is free to hand back a transposed or lazily
            # conjugated view -- an adjoint factor naturally does -- and a
            # Kronecker product of non-contiguous operands is not defined.
            if left is None:
                block = torch.kron(
                    identity, _dense_matrix(right).transpose(0, 1).contiguous()
                )
            elif right is None:
                block = torch.kron(_dense_matrix(left).contiguous(), identity)
            else:
                block = torch.kron(
                    _dense_matrix(left).contiguous(),
                    _dense_matrix(right).transpose(0, 1).contiguous(),
                )
            matrix = matrix + coefficient * block
        return matrix

    def choi(self, *, max_bytes: int = DEFAULT_DENSE_MATRIX_BYTES) -> torch.Tensor:
        """Return the Choi matrix ``sum_ij E_ij (x) E(E_ij)`` of the map.

        The Choi matrix is the map read in an index order that carries its
        positivity rather than only its action: it is positive semidefinite
        exactly when the map is completely positive. It holds the same ``d**4``
        entries as :meth:`dense` and is related to it by a reshuffle, so the two
        cannot describe different maps -- there is no second formula here, only a
        second way of grouping the four indices. Since ``dense()`` reads a row by
        the pair ``(row of rho, column of rho)`` and a Choi entry
        ``[(a, i), (b, j)]`` is ``Phi(E_ab)[i, j]``, the regroup is
        ``permute(2, 0, 3, 1)`` on the ``(p, q, r, s)`` shape. The two other
        transposes of the same four indices produce a different matrix that still
        has the map's trace along its diagonal, which is why the convention is
        pinned by a test against the map's own action rather than by a trace check.

        The trace of the result is a property of the map and not of the convention:
        ``trace(choi())`` is ``trace(E(I))``, so a trace-preserving map has
        ``trace(choi()) == d``. Measured exactly on the identity map, on a
        depolarizing channel, on amplitude damping at ``gamma = 0.4``, and on the
        transpose map, in both dtypes. The two normalization conditions are read
        off the same matrix as its partial traces -- ``Tr_first[J]`` is ``E(I)``, so
        a unital map has the identity there, and ``Tr_second[J]`` is the identity
        exactly for a trace-preserving map -- and :meth:`partial_trace` is the
        accessor that exposes them rather than re-deriving them.

        Args:
            max_bytes: The byte ceiling, inherited from :meth:`dense` because this
                matrix is that one's entries regrouped and has the same size.

        Raises:
            ValueError: If the materialized matrix would exceed ``max_bytes``.

        Examples:
            A bit-flip channel is trace preserving, so its Choi matrix has trace
            two at one qubit, and the reshape is the whole transform:

            >>> import torch
            >>> from flagquantum.operators import SuperOperator
            >>> flip = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
            >>> bit_flip = SuperOperator.from_kraus((flip,))
            >>> bit_flip.choi().shape
            torch.Size([4, 4])
            >>> bit_flip.choi().trace()
            tensor(2.+0.j, dtype=torch.complex128)
        """

        dimension, _ = self._require_summary()
        side = dimension * dimension
        entries = (
            self.dense(max_bytes=max_bytes).reshape(
                dimension, dimension, dimension, dimension
            )
            # ``dense()`` reads the pair ``(row of rho, column of rho)``; the Choi
            # order pairs the matrix unit with its image, which regroups the four
            # indices from ``(i, j, a, b)`` to ``(a, i, b, j)``.
            .permute(2, 0, 3, 1)
        )
        return entries.reshape(side, side)

    def partial_trace(
        self, register: str, *, max_bytes: int = DEFAULT_DENSE_MATRIX_BYTES
    ) -> torch.Tensor:
        """Return the Choi matrix traced over one of its two registers.

        The Choi matrix of this map is an operator on the pair ``(input, output)``
        -- entry ``[(a, i), (b, j)]`` is ``E(E_ab)[i, j]`` -- so tracing one
        register of it asks a question about the map rather than about a state.
        Both answers are ``(d, d)`` matrices and both are statements the map
        already had; the accessor exists so that reading them no longer means
        reshaping ``choi()`` by hand and writing an ``einsum``.

        ``register="input"`` sums the input index and returns ``E(I)``, an operator
        on the output register: it is the identity exactly when the map is unital.
        It is also what ``apply`` returns for the identity, and the owning suite
        asserts that agreement rather than assuming it, so the two routes to the
        quantity are one quantity.

        ``register="output"`` sums the output index and returns the matrix
        ``trace(E(E_ab))`` indexed by the input labels. It is the identity exactly
        when the map is trace preserving, and it is *not* the same statement as
        the input trace: amplitude damping is trace preserving and not unital, so
        its two traces differ, which is what separates them.

        Both traces are differentiable in whatever the factors are, because they
        are read off :meth:`choi` by a sum rather than by a branch on values.

        Args:
            register: ``"input"`` or ``"output"``, naming the register summed.
            max_bytes: The byte ceiling, inherited from :meth:`choi` because the
                trace is read off that matrix.

        Returns:
            The ``(d, d)`` complex matrix of the other register, in the map's
            dtype on the CPU.

        Raises:
            ValueError: If ``register`` is neither value, or if the materialized
                Choi matrix would exceed ``max_bytes``.

        Examples:
            A unitary channel is both unital and trace preserving, so both traces
            are the identity, while the transpose map is trace preserving and not
            unital:

            >>> import torch
            >>> from flagquantum.operators import SuperOperator
            >>> flip = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
            >>> bit_flip = SuperOperator.from_kraus((flip,))
            >>> bit_flip.partial_trace("input").tolist()
            [[(1+0j), 0j], [0j, (1+0j)]]
            >>> bit_flip.partial_trace("output").tolist()
            [[(1+0j), 0j], [0j, (1+0j)]]
        """

        dimension, _ = self._require_summary()
        if register not in {"input", "output"}:
            raise ValueError(
                f"a partial trace of a superoperator sums one of its two "
                f"registers, 'input' or 'output', and {register!r} is neither; "
                "the map's action on a subsystem is not this operation"
            )
        # ``choi()`` reads ``(a, i, b, j)``: the input matrix unit ``(a, b)`` and
        # the output matrix ``(i, j)``. Summing one pair leaves the other.
        blocks = self.choi(max_bytes=max_bytes).reshape(
            dimension, dimension, dimension, dimension
        )
        if register == "input":
            return torch.einsum("aiaj->ij", blocks)
        return torch.einsum("aibi->ab", blocks)

    def is_completely_positive(
        self, *, max_bytes: int = DEFAULT_DENSE_MATRIX_BYTES
    ) -> bool:
        """Return whether the map is completely positive.

        The map is completely positive exactly when its Choi matrix is positive
        semidefinite, so this reads the eigenvalues of the Hermitian part of
        :meth:`choi` and compares the smallest against a tolerance. The tolerance is
        not a convenience in either of its two factors. A legitimate channel may have
        a rank-deficient Choi matrix, whose smallest eigenvalue is exactly zero and
        therefore comes back an epsilon or two below zero once the entries are
        rounded, so an exact comparison would refuse a channel that is on the
        boundary rather than one that is outside it. And that deviation is not a
        constant of the channel: a rotated rank-deficient dephasing channel, whose
        smallest Choi eigenvalue is exactly zero in exact arithmetic, is measured at
        0.42 epsilons at two dimensions and 1.46 at sixteen in complex64, and at
        0.05 and 4.17 in complex128, and the same channel rescaled by ``c`` reaches
        1.34e+06 epsilons at ``c = 1024``. The threshold is therefore stated relative
        to the matrix's own largest eigenvalue rather than in absolute units;
        nothing in the arithmetic of this class supplies a scale of its own.

        A map whose Choi matrix is not Hermitian does not preserve Hermiticity,
        and positivity is not a question about it -- the Lindblad generator is
        such a map, being a sum of one-sided actions. That case raises rather than
        returning ``False``, because ``False`` would read as "this map is not
        completely positive" about a map that is not a candidate for the question.

        Args:
            max_bytes: The byte ceiling, inherited from :meth:`choi`.

        Returns:
            ``True`` if the smallest Choi eigenvalue is at or above the negative
            of the tolerance, ``False`` if it is below it.

        Raises:
            ValueError: If the materialized matrix would exceed ``max_bytes``, or
                if the Choi matrix is not Hermitian to the tolerance.

        Examples:
            A Kraus list denotes a completely positive map by construction, and a
            dephasing channel built from its two projectors is one:

            >>> import torch
            >>> from flagquantum.operators import SuperOperator
            >>> zero = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128)
            >>> one = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
            >>> SuperOperator.from_kraus((zero, one)).is_completely_positive()
            True
        """

        _, dtype = self._require_summary()
        eigenvalues, _, tolerance = _choi_eigenvalues(
            self.choi(max_bytes=max_bytes), dtype
        )
        return float(eigenvalues.min()) >= -tolerance

    def kraus(
        self, *, max_bytes: int = DEFAULT_DENSE_MATRIX_BYTES
    ) -> tuple[torch.Tensor, ...]:
        """Return Kraus operators that reproduce this map.

        A completely positive map *is* a Kraus list, and the Choi matrix is where
        that list is read off: ``J = sum_k |v_k><v_k|`` scaled by the eigenvalues,
        so the eigenvectors of :meth:`choi` unstacked back into ``d`` x ``d``
        matrices are the operators, each weighted by the square root of its
        eigenvalue. Nothing is fitted and nothing is iterated to convergence; the
        decomposition is a read-off of the spectrum :meth:`is_completely_positive`
        reads, sharing that method's tolerance rather than choosing a second one.
        This is the inverse of :meth:`from_kraus` on the level of maps, so a channel
        that arrived as a Kraus list, a matrix, a term list or a Lindblad route can
        leave as one.

        The index convention is measured rather than assumed, because a transposed
        reading is a different map that is easy to mistake for this one. The
        eigenvector is unstacked and then transposed, and the unstacked form without
        that transpose moves the population the wrong way on amplitude damping: its
        ``[0, 0]`` entry is exactly ``-gamma * rho[1, 1]`` away from the map's own
        action, measured at 2.625e-01 on the owning suite's fixture at
        ``gamma = 0.4``, while this form agrees with :meth:`apply` to 5.594e-17.

        The list is a Kraus representation and not a canonical one. The Choi
        eigenvectors are defined up to a phase, and up to any unitary mixing inside a
        degenerate eigenvalue, so two calls agree as operators but an individual
        operator's phase is arbitrary. What is canonical is the count -- the Choi
        rank, at most ``d**2`` and often far below it, so a unitary channel returns
        one operator rather than a padded list -- and the map the list denotes. The
        terms are ordered by descending eigenvalue, so the operator carrying the most
        weight comes first, and the order inside one degenerate eigenvalue is
        whatever the eigenproblem returned.

        Args:
            max_bytes: The byte ceiling, inherited from :meth:`choi` because the
                decomposition is read off that matrix.

        Returns:
            The Kraus operators as ``(d, d)`` complex matrices in this map's dtype,
            on the CPU, at most ``d**2`` of them and at least one.

        Raises:
            ValueError: If the materialized Choi matrix would exceed ``max_bytes``,
                if it is not Hermitian, or if the map is not completely positive --
                the last naming the negative eigenvalue, because then no Kraus list
                denotes this map and returning the closest one would silently
                substitute a different map.

        Examples:
            The completely dephasing channel is not unitary, so it takes two
            operators: its two projectors, each with weight one.

            >>> import torch
            >>> from flagquantum.operators import SuperOperator
            >>> zero = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128)
            >>> one = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
            >>> dephasing = SuperOperator.from_kraus((zero, one))
            >>> operators = dephasing.kraus()
            >>> len(operators)
            2
            >>> sum(operator.conj().T @ operator for operator in operators).tolist()
            [[(1+0j), 0j], [0j, (1+0j)]]
        """

        dimension, dtype = self._require_summary()
        eigenvalues, eigenvectors, tolerance = _choi_eigenvalues(
            self.choi(max_bytes=max_bytes), dtype
        )
        smallest = float(eigenvalues.min())
        if smallest < -tolerance:
            raise ValueError(
                "this map is not completely positive: its Choi matrix has the "
                f"eigenvalue {smallest:.3e} against a {tolerance:.3e} tolerance, so "
                "no Kraus list denotes it and the closest one would be a different "
                "map. The transpose map is the standard example."
            )
        # Descending weight, and nothing is padded: an eigenvalue at or below the
        # tolerance is zero for this purpose, which is what makes a unitary channel
        # come back as one operator. The sort is stable, so operators sharing an
        # eigenvalue keep the order the eigenproblem returned rather than acquiring a
        # second arbitrary one here.
        operators = []
        for index in sorted(
            range(len(eigenvalues)), key=lambda i: -float(eigenvalues[i])
        ):
            weight = float(eigenvalues[index])
            if weight <= tolerance:
                continue
            # ``vec(A rho B) = (A (x) B^T) vec(rho)`` fixes the unstacking: the
            # eigenvector read as ``(a, i)`` unstacked to ``(d, d)`` gives the
            # transpose of the operator, and dropping that transpose is a different
            # map rather than a rounding difference.
            operator = (
                eigenvectors[:, index].reshape(dimension, dimension).transpose(0, 1)
            )
            operators.append(operator * weight**0.5)
        return tuple(operators)

    def matrix_in_basis(self, basis: Sequence[torch.Tensor]) -> torch.Tensor:
        """Return the map's matrix in an orthonormal basis of matrices.

        Entry ``[i, j]`` is ``trace(B_i^dag E(B_j))``, where ``E`` is this map and
        ``B`` is ``basis``. Stacking the basis into the ``[d**2, k]`` matrix ``P``
        whose column ``i`` is ``vec(B_i)``, that entry count is exactly
        ``P^dag @ dense() @ P``, because the row-major vectorization this package
        uses satisfies ``trace(A^dag B) = vec(A)^dag vec(B)``. The transform is the
        map's spectrum in whatever basis the caller names, so a channel's Pauli
        transfer matrix and the Kraus list it came from are one object read two
        ways.

        Two properties are required of the basis and both are checked, because
        dropping either makes the returned matrix a different quantity under the
        same name. ``k == d**2`` entries are required so the matrix represents the
        whole map rather than its projection onto a subspace; without it a caller
        who passed four of sixteen two-wire Pauli words would receive a 4x4
        matrix that looks like a small map and is not one.
        ``trace(B_i^dag B_j) == delta_ij`` is required so ``P^dag`` is ``P^-1``;
        with a merely spanning basis the product is the Gram-corrected
        ``P^dag D P`` rather than the similarity transform ``P^-1 D P``, which
        agrees with the right matrix only when the basis was already orthonormal.

        The map's dense form is built inside, so this transform inherits both of
        :meth:`dense`'s limits: its byte ceiling, and the fact that it is a CPU
        reference. A basis on another device is refused by name rather than
        reaching a Kronecker product whose operands disagree.

        Args:
            basis: ``d**2`` matrices, each ``(d, d)``, sharing the map's complex
                dtype and living on the CPU.

        Raises:
            ValueError: If the basis is empty, an entry is not a ``(d, d)`` tensor
                of the map's dtype, the entries disagree on device, the basis is
                not on the CPU, the count is not ``d**2``, or the basis is not
                orthonormal to ``_BASIS_TOLERANCE_IN_EPSILON`` times the dtype's
                epsilon.
        """

        dimension, dtype = self._require_summary()
        entries = tuple(basis)
        if not entries:
            raise ValueError(
                "a basis of matrices needs at least one element; an empty basis "
                "has no entry to read the map in"
            )
        expected = dimension * dimension
        if len(entries) != expected:
            raise ValueError(
                f"a {dimension}-dimensional superoperator is represented in a "
                f"basis of {expected} matrices, and {len(entries)} were given; a "
                "shorter basis would report the map's projection onto it rather "
                "than the map"
            )
        validated = [_basis_entry(entry, dimension, dtype) for entry in entries]
        # The device is settled before the stack, not after: ``torch.stack``
        # refuses tensors that disagree on device with a bare ``RuntimeError``, so
        # a check placed after it would name the wrong device never. Each entry is
        # validated first so this reads ``.device`` only from a tensor.
        devices = {entry.device for entry in validated}
        if len(devices) > 1:
            raise ValueError(
                "every basis element must share one device, got "
                f"{sorted(str(entry) for entry in devices)}"
            )
        device = next(iter(devices))
        if device.type != "cpu":
            raise ValueError(
                "a basis must live on the CPU device dense() builds its reference "
                f"on, and this one is on {device}; the dense form is a small-system "
                "reference rather than a device-resident one"
            )
        stacked = torch.stack(validated)
        # ``column i`` is ``vec(B_i)``, which is the layout ``P^dag D P`` reads:
        # the stack arrives with one basis element per row instead.
        columns = stacked.reshape(expected, expected).transpose(0, 1)
        gram = columns.mH @ columns
        deviation = float(torch.max(torch.abs(gram - torch.eye(expected, dtype=dtype))))
        tolerance = _BASIS_TOLERANCE_IN_EPSILON * float(torch.finfo(dtype).eps)
        if deviation > tolerance:
            raise ValueError(
                f"the basis is not orthonormal under trace(B_i^dag B_j): it is "
                f"{deviation:.3e} away from the identity, above this transform's "
                f"{tolerance:.3e} tolerance. A transform by a "
                "non-orthonormal basis computes P^dag D P rather than P^-1 D P, "
                "which is a different matrix, so the basis is refused rather than "
                "inverted."
            )
        return columns.mH @ self.dense() @ columns

    def __repr__(self) -> str:
        return (
            f"SuperOperator(terms={len(self._terms)}, dimension={self._dimension}, "
            f"dtype={self._dtype})"
        )
