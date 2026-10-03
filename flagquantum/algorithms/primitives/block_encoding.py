"""Block encodings of a Hermitian contraction, and the walk step they make available.

A block encoding of an operator ``H`` at a subnormalisation ``alpha`` is a unitary ``U``
whose flagged block is ``H / alpha``: writing ``|0>`` for the ancilla register's all-zero
state and ``P = |0><0|`` for the projector onto it, ``P U P = H / alpha`` and ``U`` is
otherwise unconstrained. Gilyén, Su, Low and Wiebe, "Quantum singular value transformation
and beyond: exponential improvements for quantum matrix arithmetics", STOC 2019, pp.
193-204, DOI 10.1145/3313276.3316366, Definition 1 is where the factor and the convention
``||H|| <= alpha + eps`` are fixed, and that is the definition this module builds to.

**The construction here is the spectral one, and it is exact rather than approximate.** A
Hermitian ``H`` has an orthonormal eigenbasis and real eigenvalues, so with
``A = H / alpha`` written in that basis the reflection

    U = [[ A,                sqrt(I - A**2) ],
         [ sqrt(I - A**2),  -A             ]]

is Hermitian and unitary whenever every eigenvalue of ``A`` lies in ``[-1, 1]``, which is
exactly the condition ``alpha >= ||H||`` buys. ``A`` and ``sqrt(I - A**2)`` are both
functions of ``H``, so they commute and ``U ** 2 = I`` holds identically rather than to
within a tolerance of the eigenvectors. The flagged block of ``U`` is ``A``, so ``U`` is a
block encoding of ``H`` at ``alpha``, and the ancilla needs no preparation: the flag is
one wire left in ``|0>``.

**The same construction is also a qubitization, which is the property that makes it worth
more than its block.** The walk step is the flag's phase flip composed with the encoding,
``W = Z_ancilla U``. Because ``U`` acts on the eigenvector ``|v_j>`` of ``A`` with
eigenvalue ``w_j`` as ``U|0>|v_j> = w_j|0>|v_j> + mu_j|1>|v_j>`` and
``U|1>|v_j> = mu_j|0>|v_j> - w_j|1>|v_j>``, with ``mu_j = sqrt(1 - w_j**2)``, the two states
``|0>|v_j>`` and ``|1>|v_j>`` span a subspace ``W`` preserves, and on that subspace ``W``
is the rotation ``[[w_j, mu_j], [-mu_j, w_j]]``. Its eigenvalues on that subspace are
therefore ``w_j +- i mu_j = exp(+-i arccos(w_j))``, so **every eigenvalue of one walk step
is a phase whose cosine is an eigenvalue of ``H`` over ``alpha``**. That identity is what
phase estimation and the Chebyshev-moment methods are read off, and it is asserted as an
equality of spectra in the unit tests rather than described. The *definition* of the step
is asserted separately, as ``W`` against ``Z_ancilla U`` compared entrywise, because a
spectrum does not pin it: this construction's step is Hermitian, so it is its own adjoint
and its own encoding is a Hermitian unitary on the same register, and the composed matrix
is the only witness that the flip is emitted at all.

**Why this is a primitive and not a private helper of one algorithm.** The encoding was
private to :mod:`flagquantum.algorithms.svd` in the mean-of-two-branches form, with a
comment that authorised promotion to this module once a second consumer needed one. The
promotion is on the other ground this package's README records -- the unit callers use
directly -- because the encoding is a boundary consumers are written against rather than
an implementation detail: :class:`BlockEncoding` is what a walk, a phase-estimation
circuit, or a singular-value readout is generic over, and the form promoted here is
strictly more capable than the private one it replaces, admitting the walk identity above
while extracting the same block. The mean-of-two-branches circuit is gone; the extracted
block is unchanged, and the replacement is asserted by the module's tests.

**Circuits, not kernels, and not a runtime.** Everything here appends to a
:class:`~flagquantum.circuit.Circuit`, so the encoding composes with every pass and every
backend the compiler already owns. Nothing here chooses a device, and nothing here counts
a cost: a caller who wants the gate count reads it back with
:func:`~flagquantum.compiler.resource_estimation.estimate_resources`.

**What is deliberately absent, and is owned rather than silent.** There is one
implementation (the spectral one) and one flag wire, so the protocols are the surface
those consumers will be written against rather than a claim of a family already built: a
Pauli linear-combination encoding, whose preparation is a superposition over coefficients
rather than the identity, is the next member, and the multi-wire flag reflection it will
need is not here. The eigendecomposition is dense and exact, so the cost is that of
diagonalising the matrix and the gate emitted is one dense matrix on ``n + 1`` wires: this
is the demonstration scale the algorithm modules work at, not a subroutine with a
gate-efficient synthesis. No error bound is reported, because the construction is exact for
a matrix that is exactly Hermitian and a caller holding an approximation holds its error
instead. And the matrix is detached onto the CPU before anything else happens, so no
gradient flows from a circuit back to the entries of the matrix that produced it: the
encoding is a classical precomputation over a matrix the caller has already formed, and a
caller who wants the derivative of the decomposition has autograd on the matrix rather than
on a circuit built from one.

**The register parameter is spelled ``qubits``, and the older primitives are not.** This
package is migrating every public ``wire``-named parameter to its qubit-named spelling, and
``tools/check_qubit_vocabulary.py`` fails closed when a new public parameter grows the
ledger -- correctly, because a migration that admits new debt never finishes. So a module
landing now lands on the target vocabulary rather than on the one being retired, and
``append_qft(wires=...)`` beside ``append_apply(qubits=...)`` is the seam of a migration in
progress rather than a mistake. ``contracts/qubit-vocabulary-contract.toml`` owns the
ledger; this module owns none of it and adds nothing to it.

This unit carries no performance, capacity, or hardware claim of its own, and it does not
select a runtime.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from ...circuit import Circuit

__all__ = [
    "BlockEncoding",
    "SpectralBlockEncoding",
    "WalkEncoding",
    "spectral_block_encoding",
    "subnormalisation",
]

# The value object's identities are checked on dense double-precision matrices, where a
# product of two such matrices is exact to about 1e-15, and where the square root of
# ``1 - w ** 2`` loses digits as ``w`` approaches one. The tolerance is set above both
# while staying far below what a caller's spoiled matrix is out by.
_TOLERANCE = 1e-6

# The walk step's reflection is a phase flip on the flag wire, whose all-zero state the
# projector ``|0><0|`` selects.
_FLAG_REFLECTION = "z"


@runtime_checkable
class BlockEncoding(Protocol):
    """The surface a consumer of a block encoding is written against.

    Conformance is structural: an implementation inherits from nothing. A consumer is
    written against these three numbers and one method, so it never learns which
    construction it was handed.
    """

    @property
    def num_system(self) -> int:
        """The number of wires the encoded operator acts on."""
        ...

    @property
    def num_ancilla(self) -> int:
        """The number of ancilla wires whose zero state flags the encoded block."""
        ...

    @property
    def alpha(self) -> float:
        """The subnormalisation: the flagged block is the encoded matrix over this."""
        ...

    def append_apply(
        self, circuit: Circuit, *, ancilla: int, qubits: Sequence[int]
    ) -> None:
        """Append the encoding to ``circuit`` on ``ancilla`` and ``qubits``.

        Args:
            circuit: The circuit to extend.
            ancilla: The flag wire, which the encoding must leave in its zero state for
                the flagged block to be the one the encoding names.
            qubits: The operator's own qubits, ``num_system`` of them.
        """
        ...


@runtime_checkable
class WalkEncoding(BlockEncoding, Protocol):
    """A block encoding that also carries its qubitization walk step.

    The walk step is a property of the construction rather than of the block it encodes:
    two encodings of the same matrix need not admit the same walk, and an encoding that
    admits none cannot honour these two methods. They are on a separate protocol, which is
    how that is said in the type system instead of in prose.
    """

    def append_walk_step(
        self, circuit: Circuit, *, ancilla: int, qubits: Sequence[int]
    ) -> None:
        """Append one walk step ``W`` to ``circuit``.

        Args:
            circuit: The circuit to extend.
            ancilla: The flag wire.
            qubits: The operator's own qubits.
        """
        ...

    def append_adjoint_walk_step(
        self, circuit: Circuit, *, ancilla: int, qubits: Sequence[int]
    ) -> None:
        """Append the walk step's adjoint ``W^dagger`` to ``circuit``.

        Args:
            circuit: The circuit to extend.
            ancilla: The flag wire.
            qubits: The operator's own qubits.
        """
        ...


def subnormalisation(matrix: torch.Tensor, alpha: float | None = None) -> float:
    """Return the subnormalisation a block encoding of ``matrix`` is read at.

    ``None`` is the matrix's Frobenius norm, which is at least its spectral norm and is
    available without diagonalising anything. A value is accepted when it is at least that
    spectral norm, and **refused when it is below it**: the flagged block would then be an
    operator of norm greater than one, and no unitary has such an operator as one of its
    blocks, so the construction has no encoding to build and would be building something
    else under the name of one.

    Args:
        matrix: The Hermitian matrix to encode.
        alpha: The candidate subnormalisation, or ``None`` for the matrix's Frobenius norm.

    Returns:
        The subnormalisation.

    Raises:
        ValueError: If ``matrix`` is not a finite square two-dimensional Hermitian tensor
            of power-of-two dimension at least two; if the matrix is the zero matrix, whose
            Frobenius norm is not a positive factor; if ``alpha`` is not a positive finite
            real number; or if it is below the matrix's spectral norm.
    """
    checked = _validated_matrix(matrix)
    dimensions = int(checked.shape[0])
    if alpha is None:
        factor = float(torch.linalg.matrix_norm(checked, ord="fro"))
        if factor == 0.0:
            raise ValueError(
                "the zero matrix has no block to normalise; its Frobenius norm is zero, "
                "so the block would be divided by nothing"
            )
        return factor
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(
            f"the subnormalisation must be a real number, got {alpha!r}; it is the factor "
            "the block is smaller than the matrix by, and a flag or a tensor is not one"
        )
    factor = float(alpha)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(
            f"the subnormalisation must be positive and finite, got {factor}; a "
            "non-positive factor is not what the block is smaller than the matrix by"
        )
    spectral = float(torch.linalg.matrix_norm(checked, ord=2))
    if factor < spectral:
        raise ValueError(
            "the subnormalisation must be at least the matrix's spectral norm "
            f"{spectral}, got {factor}; the block would be that matrix over {factor}, an "
            "operator of norm greater than one, and no unitary has an operator of norm "
            "greater than one as one of its blocks, so this is not a matrix that can be "
            f"encoded at this factor at all ({dimensions} by {dimensions})"
        )
    return factor


@dataclass(frozen=True, eq=False)
class SpectralBlockEncoding:
    """A block encoding of a Hermitian contraction, read from its own eigenbasis.

    Build one through :func:`spectral_block_encoding`, which is where the subnormalisation
    is resolved; constructing the class directly takes an already-resolved factor, which
    the class then checks against the matrix it was given.

    The instance is frozen and equality is by identity, which is deliberate: the fields are
    torch tensors, and a dataclass comparison of two tensor fields yields a tensor rather
    than a truth value.
    """

    matrix: torch.Tensor
    alpha: float
    _select: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the pair and build the reflection the encoding applies.

        Raises:
            ValueError: If ``matrix`` is not a finite square two-dimensional Hermitian
                tensor of power-of-two dimension at least two; if ``alpha`` is not a
                positive finite number at least the matrix's spectral norm; or if the
                reflection built from the pair is not the unitary its block encoding has
                to be.
        """
        checked = _validated_matrix(self.matrix)
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError(
                f"the subnormalisation must be positive and finite, got {self.alpha}"
            )
        spectral = float(torch.linalg.matrix_norm(checked, ord=2))
        if self.alpha < spectral:
            raise ValueError(
                "the subnormalisation must be at least the matrix's spectral norm "
                f"{spectral}, got {self.alpha}; the block would be the matrix over "
                f"{self.alpha}, and no unitary has an operator of norm greater than one "
                "as one of its blocks"
            )
        select = _reflection(checked, self.alpha)
        dimension = int(select.shape[0])
        identity = torch.eye(dimension, dtype=torch.complex128)
        if not bool(torch.allclose(select, select.mH, atol=_TOLERANCE, rtol=0.0)):
            raise ValueError(
                "the block encoding's reflection must be Hermitian and it is not, which "
                "means the construction was taken outside the range the matrix's spectrum "
                "was scaled into"
            )
        if not bool(
            torch.allclose(select @ select, identity, atol=_TOLERANCE, rtol=0.0)
        ):
            raise ValueError(
                "the block encoding's reflection must square to the identity and it does "
                "not, so it is not the unitary a block encoding has to be"
            )
        object.__setattr__(self, "_select", select)

    @property
    def num_system(self) -> int:
        """The number of wires the encoded operator acts on."""
        return int(self.matrix.shape[0]).bit_length() - 1

    @property
    def num_ancilla(self) -> int:
        """The number of ancilla wires flagging the encoded block, which is one."""
        return 1

    @property
    def select(self) -> torch.Tensor:
        """The reflection the encoding applies, on the flag wire and the operator's."""
        return self._select

    def block(self) -> torch.Tensor:
        """Return the flagged block, which is the encoded matrix over ``alpha``.

        Returns:
            The block, in double precision on the CPU, of the matrix's own dimension.
        """
        dimension = 1 << self.num_system
        return self._select[:dimension, :dimension]

    def append_apply(
        self, circuit: Circuit, *, ancilla: int, qubits: Sequence[int]
    ) -> None:
        """Append the encoding to ``circuit`` on ``ancilla`` and ``qubits``.

        Args:
            circuit: The circuit to extend.
            ancilla: The flag wire, left in its zero state by this encoding.
            qubits: The operator's own qubits, :attr:`num_system` of them.

        Raises:
            ValueError: If ``qubits`` is not :attr:`num_system` distinct qubits, or if
                ``ancilla`` is one of them.
        """
        _require_register(self, ancilla, qubits)
        circuit.any(ancilla, *qubits, unitary=self._select, name="block_encoding")

    def append_walk_step(
        self, circuit: Circuit, *, ancilla: int, qubits: Sequence[int]
    ) -> None:
        """Append one walk step ``W = Z_ancilla U`` to ``circuit``.

        The step's eigenvalues are ``exp(+-i arccos(lambda / alpha))`` over the encoded
        matrix's eigenvalues ``lambda``; that identity is what the tests assert.

        Args:
            circuit: The circuit to extend.
            ancilla: The flag wire.
            qubits: The operator's own qubits.

        Raises:
            ValueError: If ``qubits`` or ``ancilla`` fail the validation
                :meth:`append_apply` applies to them.
        """
        _require_register(self, ancilla, qubits)
        # A circuit composes its instructions in reverse, so the sequence emitted for the
        # product ``Z U`` is the encoding first and the phase flip second.
        circuit.any(ancilla, *qubits, unitary=self._select, name="walk_step_select")
        circuit.gate(_FLAG_REFLECTION, ancilla)

    def append_adjoint_walk_step(
        self, circuit: Circuit, *, ancilla: int, qubits: Sequence[int]
    ) -> None:
        """Append ``W^dagger = U Z_ancilla``, the walk step's adjoint, to ``circuit``.

        Args:
            circuit: The circuit to extend.
            ancilla: The flag wire.
            qubits: The operator's own qubits.

        Raises:
            ValueError: If ``qubits`` or ``ancilla`` fail the validation
                :meth:`append_apply` applies to them.
        """
        _require_register(self, ancilla, qubits)
        circuit.gate(_FLAG_REFLECTION, ancilla)
        circuit.any(
            ancilla, *qubits, unitary=self._select, name="walk_step_select_adjoint"
        )


def spectral_block_encoding(
    matrix: torch.Tensor, alpha: float | None = None
) -> SpectralBlockEncoding:
    """Return the spectral block encoding of a Hermitian matrix.

    Args:
        matrix: The Hermitian matrix to encode.
        alpha: The subnormalisation, or ``None`` for the matrix's Frobenius norm. It must
            be at least the matrix's spectral norm.

    Returns:
        The encoding, whose flagged block is ``matrix / alpha``.

    Raises:
        ValueError: If ``matrix`` or ``alpha`` fail the validation
            :func:`subnormalisation` applies to them.
    """
    checked = _validated_matrix(matrix)
    return SpectralBlockEncoding(checked, subnormalisation(checked, alpha))


def _validated_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Return ``matrix`` on the CPU as a checked Hermitian square.

    Args:
        matrix: The candidate matrix.

    Returns:
        The matrix, detached onto the CPU, in its own floating-point dtype.

    Raises:
        ValueError: If it is not a torch.Tensor; if it is not a square two-dimensional
            floating-point tensor; if its dimension is not a power of two of at least two;
            if it holds a non-finite entry; or if it is not Hermitian.
    """
    if not isinstance(matrix, torch.Tensor):
        raise ValueError(
            f"matrix must be a torch.Tensor, got {type(matrix).__name__}; the encoding is "
            "built from the matrix's own eigenbasis, and a description of a matrix is not "
            "one"
        )
    if matrix.dim() != 2:
        raise ValueError(
            "matrix must be a two-dimensional matrix, got shape "
            f"{tuple(int(size) for size in matrix.shape)}; its two dimensions index the "
            "rows and the columns the encoding carries"
        )
    if not matrix.is_floating_point() and not matrix.is_complex():
        raise ValueError(
            "matrix must be a floating-point or complex matrix, got dtype "
            f"{matrix.dtype}; its eigenbasis is what the encoding is read from"
        )
    rows, columns = (int(size) for size in matrix.shape)
    if rows != columns:
        raise ValueError(
            f"matrix must be square, got {rows} row(s) and {columns} column(s); an "
            "operator's encoding acts on one register, whose two sides are the same width"
        )
    if rows < 2 or rows & (rows - 1):
        raise ValueError(
            f"matrix must have a power-of-two dimension of at least two, got {rows}; the "
            "operator acts on a whole number of wires"
        )
    checked = matrix.detach().to(device="cpu")
    if not bool(torch.isfinite(checked).all()):
        raise ValueError(
            "matrix must hold only finite values; a non-finite entry has no eigenbasis "
            "and its exponential is not a unitary"
        )
    if not bool(torch.allclose(checked, checked.mH, atol=_TOLERANCE, rtol=0.0)):
        raise ValueError(
            "the block encoding is built from the matrix's own eigenbasis, and eigh reads "
            "one triangle of its argument, so this matrix, which is not Hermitian, is "
            "refused rather than encoded from that one triangle"
        )
    return checked


def _reflection(matrix: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return the reflection a block encoding of ``matrix`` at ``alpha`` applies.

    The reflection is ``[[A, sqrt(I - A**2)], [sqrt(I - A**2), -A]]`` with ``A`` the matrix
    over ``alpha``, which is Hermitian and unitary whenever every eigenvalue of ``A`` lies
    in ``[-1, 1]``. The two blocks are the cosine and the sine of the same phase, which is
    why one walk step is a rotation: on the eigenvector of ``A`` with eigenvalue ``w`` the
    pair is ``exp(+-i arccos(w))``.

    Args:
        matrix: The Hermitian matrix to encode.
        alpha: The subnormalisation, at least the matrix's spectral norm.

    Returns:
        The reflection, ``2 ** (n + 1)`` by ``2 ** (n + 1)`` for an ``n``-wire matrix, in
        double precision on the CPU, the flag wire's ``|0>`` block leading.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix.to(torch.complex128))
    # The clamp is what makes the construction survive an ``alpha`` that only rounds to the
    # spectral norm: an eigenvalue of ``A`` a fraction above one is read as one, which is
    # the block a caller who named that factor asked for. The eigenvalues of a Hermitian
    # matrix are real even when it is handed over as complex, so both are cast back: the
    # eigenvectors are the complex factor the product is assembled from.
    normalised = (eigenvalues / alpha).clamp(-1.0, 1.0).to(torch.complex128)
    complement = torch.sqrt(torch.clamp(1.0 - normalised.real**2, min=0.0)).to(
        torch.complex128
    )
    encoded = eigenvectors @ torch.diag(normalised) @ eigenvectors.mH
    coupling = eigenvectors @ torch.diag(complement) @ eigenvectors.mH
    return torch.cat(
        (
            torch.cat((encoded, coupling), dim=1),
            torch.cat((coupling, -encoded), dim=1),
        ),
        dim=0,
    )


def _require_register(
    encoding: SpectralBlockEncoding, ancilla: int, wires: Sequence[int]
) -> None:
    """Check that ``ancilla`` and ``wires`` name the register the encoding needs.

    Private, and it keeps the wire spelling the migration has not yet reached here: the
    public parameter is ``qubits``, and this is the one place the two names meet.

    Args:
        encoding: The encoding about to be appended.
        ancilla: The flag wire.
        wires: The operator's own wires.

    Raises:
        ValueError: If ``ancilla`` is not an integer; if the register is not
            :attr:`~SpectralBlockEncoding.num_system` wires; if ``ancilla`` is one of
            them; or if a wire appears twice.
    """
    if isinstance(ancilla, bool) or not isinstance(ancilla, int):
        raise ValueError(
            f"ancilla must be a wire index, got {ancilla!r}; the flag is one wire, and a "
            "list of them is not one wire"
        )
    width = len(wires)
    if width != encoding.num_system:
        raise ValueError(
            f"the register must be {encoding.num_system} wire(s) for this encoding, "
            f"got {width}"
        )
    if ancilla in wires:
        raise ValueError(
            f"ancilla {ancilla} is one of the operator's own wires {list(wires)}; the flag "
            "wire is not part of the register the encoded operator acts on"
        )
    if len(set(wires)) != width:
        raise ValueError(
            f"the register must be distinct, got {list(wires)}; an operator acts on each "
            "wire once"
        )
