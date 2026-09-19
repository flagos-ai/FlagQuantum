"""Quantum principal component analysis over a classically formed density matrix.

The eigenvalues of the density matrix ``rho = A A^T / tr(A A^T)`` of a data matrix
``A`` are estimated with the phase-estimation readout of Seth Lloyd, Masoud Mohseni
and Patrick Rebentrost, "Quantum principal component analysis", *Nature Physics*
**10**, 631-633 (2014), DOI 10.1038/nphys3029, arXiv:1307.0401. The paper estimates
the spectrum of ``rho`` by phase-estimating a unitary built from it, and this unit
uses the same subroutine: a purification of ``rho`` is phase-estimated against
``exp(-2 pi i rho)``, and the counter register is read out as the eigenvalue
distribution.

**The premise is qPCA's input model, and this unit does not meet it.** The paper's
contribution is that ``rho`` is never formed: its subroutine consumes copies of the
density matrix, one use per run, and the ``O(1/eps**3)`` copies it needs is the cost
it counts. This unit forms ``rho`` classically from the full matrix ``A``, builds
``exp(-2 pi i rho)`` with :func:`torch.matrix_exp`, and hands the result to the
circuit as a dense gate matrix, so every property the paper's input model is what
buys is absent here. The purification step inherits its own premise as well: it is
built with
:func:`~flagquantum.algorithms.primitives.state_preparation.append_arbitrary_state`,
which solves for its rotation angles with a classical pass over all ``2**n``
amplitudes and a ``2**n`` by ``2**n`` linear solve, so the caller must already hold
the entire amplitude vector. Nothing here is faster, or smaller, than diagonalising
``rho`` with :func:`torch.linalg.eigvalsh` directly.

**Reading a counter value as an eigenvalue.** At ``t = 2*pi`` the unitary
``exp(-2 pi i rho)`` has eigenphase ``exp(-2 pi i * lambda)`` on an eigenvector of
``rho`` with eigenvalue ``lambda``, which is the phase ``phi = (-lambda) mod 1``;
for ``lambda`` in ``(0, 1)`` that is ``phi = 1 - lambda``, so the counter value
``k`` of ``m`` counting wires reads the eigenvalue ``lambda = 1 - k / 2**m``. The
inversion is part of the readout and not a cosmetic detail: dropping it reports
``phi``, which is a wrong eigenvalue rather than an error.

**The estimate is a resolution, not a distribution.** The counter register resolves
the phase to ``1 / 2**m``, so a resolved eigenvalue is accurate to about half that
step and :class:`PcaResult` carries the step as its ``resolution``. What the readout
does *not* report is a probability per eigenvalue: the sample's share at the counter
value nearest an eigenvalue is not that eigenvalue. Measured at six counting wires,
the small eigenvalue ``0.0038`` comes back on its own counter value with probability
``0.0331``, about nine times its weight, because phase estimation spreads the
dominant peak's tail into the neighbouring value. :meth:`PcaResult.within` is the
contract for the resolution, and no confidence interval is computed or reported
anywhere in this module.

**Which eigenvalue the mode reports is a weight comparison, not a rule about the
peak's shape.** The mode is the counter value carrying the largest share of the
sample, and nothing else; the readout is that counter value's eigenvalue,
``1 - k / 2**m``. So :meth:`PcaResult.within` accepts the largest eigenvalue exactly
when the mode's counter value lies within half a step of that eigenvalue's phase,
and rejects it when some other counter value's share is the larger one.

What decides that is where each eigenvalue's phase falls on the counter grid, because
it sets how that eigenvalue's weight is spread across counter values, and therefore
how large a share each of its counter values can carry. An eigenvalue whose phase
lands on a counter value keeps its whole share there. One whose phase lands near a
counter midpoint is split across the two neighbouring values, and each half is
smaller than the whole share would have been. **A split is a risk and not a cause:**
the split eigenvalue keeps the mode whenever each half beats every other counter
value's share, and it then passes ``within`` when the half that kept it is the one
within half a step of the phase, which the nearer of the two halves is. It loses the
mode, and ``within`` with it, as soon as a concentrated competitor's share is larger
than both halves -- the contract working as written, since the readout is not
claiming to have resolved the largest eigenvalue, only reporting the counter value
that was sampled most. Neither the counter width nor the shape of the peak settles
the question; only the comparison of shares does.

Both outcomes are measured, at six counting wires, with the largest eigenvalue held
fixed at phase ``31.35/64`` -- split across counters 31 and 32, whose shares are
``0.3385`` and ``0.0973`` -- and only a competitor's weight moving. With a
competitor of weight ``0.296875`` on counter 45 (share ``0.2981``) the split peak
*keeps* the mode: both halves beat the competitor, the readout is ``0.515625``, and
it is within half a step of the largest eigenvalue. With a competitor of weight
``0.390625`` on counter 39 (share ``0.3889``) the split peak *loses* it: the
competitor's share is above both halves, the readout is ``0.390625``, and the
largest eigenvalue is correctly rejected. A caller that needs the largest eigenvalue
specifically has to check the readout, and cannot infer it from the peak having been
split or from the counter width.

**Wire layout.** The counting register is wires ``0 .. m - 1``, the data register
``a`` follows it, and the purification register ``b`` follows that. The counting
register's bits lead every sample key, as
:meth:`flagquantum.circuit.Circuit.counts` returns them, one character per wire with
wire 0 the most significant.

This unit is demonstration scale. It makes no performance, capacity, convergence, or
hardware claim, and it does not select a runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..circuit import Circuit
from .primitives.phase_estimation import append_phase_estimation
from .primitives.state_preparation import append_arbitrary_state

__all__ = ["PcaResult", "principal_components"]

# The default sample size, matching the other sampling units in this package.
_DEFAULT_SHOTS = 4096


@dataclass(frozen=True, kw_only=True)
class PcaResult:
    """The outcome of one sampled :func:`principal_components` run.

    ``kw_only`` is not optional here: every field below is one of two numeric types,
    and two of them are integers, so a positional spelling would let a caller
    transpose ``n_counting_wires`` and ``shots``-derived counts without an error.

    Attributes:
        dominant_eigenvalue: The eigenvalue read out at the mode ``k`` of
            ``distribution``, which is ``1 - k / 2**n_counting_wires``. It is the
            dominant eigenvalue of ``rho``, to within the counter's resolution, when
            the mode's counter value lies within half a step of that eigenvalue's
            phase. Whether it does is a comparison between counter values' shares and
            not a property of the peak's shape: a split peak can keep the mode or
            lose it. See the module docstring for both outcomes, measured.
        dominant_probability: The share of the sample that landed on that mode.
            It is not the eigenvalue's weight in ``rho``: see the module docstring.
        distribution: The counter register's marginal distribution, keyed by its
            big-endian bit string, one character per counting wire. Shares of the
            sample, so the values sum to one. The evaluation registers' bits are
            folded away rather than ignored: a key is truncated to its leading
            ``n_counting_wires`` characters, because reading the full-register keys
            instead would follow the registers the counting register is correlated
            with.
        n_counting_wires: The width of the counting register the estimate came from.
        resolution: The phase step the counter resolves, ``1 / 2**n_counting_wires``,
            expressed in eigenvalue units because ``lambda = 1 - phi`` maps a phase
            step to an eigenvalue step of the same size. A resolved eigenvalue is
            accurate to about half of it, which is the accuracy
            :meth:`within` checks and the whole of the accuracy contract.
    """

    dominant_eigenvalue: float
    dominant_probability: float
    distribution: Mapping[str, float]
    n_counting_wires: int
    resolution: float

    def __post_init__(self) -> None:
        """Reject a result that cannot be an eigenvalue readout of a density matrix.

        Raises:
            ValueError: If the dominant eigenvalue is outside ``(0, 1]``, if the
                dominant probability is outside ``[0, 1]``, if the distribution is
                empty, or if the register width and the resolution are not positive.
        """
        if not 0.0 < self.dominant_eigenvalue <= 1.0:
            raise ValueError(
                "a density matrix eigenvalue readout must lie in (0, 1], got "
                f"dominant_eigenvalue={self.dominant_eigenvalue}; the counter value "
                "0 reads 1 and the largest counter value reads 2**-n_counting_wires"
            )
        if not 0.0 <= self.dominant_probability <= 1.0:
            raise ValueError(
                "a sample share must lie in [0, 1], got "
                f"dominant_probability={self.dominant_probability}"
            )
        if not self.distribution:
            raise ValueError(
                "the counter distribution must not be empty; with no counter value "
                "there is no eigenvalue to read"
            )
        if self.n_counting_wires < 1:
            raise ValueError(
                "the counting register needs at least one wire, got "
                f"n_counting_wires={self.n_counting_wires}"
            )
        if self.resolution <= 0.0:
            raise ValueError(
                f"the counter step must be positive, got resolution={self.resolution}"
            )

    def within(self, eigenvalue: float) -> bool:
        """Return whether ``eigenvalue`` is within half a counter step of the readout.

        This is the accuracy contract the module makes, expressed once here rather
        than re-derived by every caller. Half a step is the whole of it: ``within``
        accepts ``eigenvalue`` when the mode's counter value lies within half a step
        of that eigenvalue's phase, and rejects it otherwise. A full step would
        accept a value the neighbouring counter value could equally claim, which is
        why the bound is half and not whole.

        **It does not claim that the readout resolves the largest eigenvalue.** The
        mode is the counter value with the largest share, so whether it lands within
        half a step of the largest eigenvalue's phase is decided by a comparison
        between counter values' shares. The largest eigenvalue's own phase sets how
        its weight is spread over counter values -- concentrated on one of them, or
        split across two -- and both outcomes are reachable: a split peak keeps the
        mode when each half beats every other counter value's share, and loses it to
        a concentrated competitor whose share is larger than both halves. ``within``
        rejects the largest eigenvalue in that second case, correctly, because the
        readout did not report it. The module docstring states this and gives a
        measured spectrum for each outcome.

        It is a resolution statement and not a distribution statement: a value this
        rejects is not excluded at any stated confidence level, and an eigenvalue
        whose weight in ``rho`` is small is not disqualified by the readout at all.

        Args:
            eigenvalue: The eigenvalue to compare against, normally a known one.

        Returns:
            ``True`` if ``abs(self.dominant_eigenvalue - eigenvalue) <= resolution / 2``.
        """
        return abs(self.dominant_eigenvalue - eigenvalue) <= self.resolution / 2


def principal_components(
    A: torch.Tensor,  # noqa: N803
    *,
    n_counting_wires: int,
    shots: int = _DEFAULT_SHOTS,
    seed: int | None = None,
) -> PcaResult:
    """Estimate the eigenvalues of ``A``'s density matrix by phase estimation.

    ``A`` is spelled as the construction and the paper read it, which is the one
    place this package keeps an upper-case parameter name and the reason the naming
    rule is suppressed above: every formula in the module docstring is stated in
    terms of ``A``, ``rho = A A^T / tr(A A^T)`` and ``vec(A)``.

    The density matrix is ``rho = A A^T / tr(A A^T)``, formed classically: see the
    module docstring for what that costs and what it gives up. ``rho``'s
    purification ``vec(A) / ||A||_F`` is prepared on the data register and the
    purification register, ``exp(-2 pi i rho)`` is phase-estimated against it, and
    the counter register is read out as described there. The normalisation is what
    makes the readout depend on ``A``'s direction and not on its scale: ``A`` and
    ``c * A`` give the same result for any positive ``c``.

    Args:
        A: The data matrix, a real floating-point tensor with at least two rows and
            two columns, each a power of two so that the two registers carry whole
            wires. Its Gram matrix must have a positive trace, so the zero matrix is
            refused rather than normalised into a NaN.
        n_counting_wires: The width of the counting register, at least one. The
            resolution is ``2**-n_counting_wires`` in eigenvalue units.
        shots: The number of samples to draw, at least one. Every share in the
            returned distribution is a count over exactly this many samples.
        seed: The sampler's seed, or ``None`` to draw from the ambient generator.
            The same seed and the same arguments replay the same result exactly.

    Returns:
        The eigenvalue read out at the counter register's mode with that value's
        measured share, the counter register's distribution, and the resolution they
        were read at. The readout reports the dominant eigenvalue when the mode's
        counter value lies within half a step of that eigenvalue's phase; a split peak
        can keep the mode or lose it to a concentrated competitor, and the module
        docstring measures both.

    Raises:
        ValueError: If ``A`` is not a two-dimensional real floating-point tensor
            whose dimensions are powers of two of at least two; if ``A`` holds a
            non-finite entry; if ``A``'s Gram matrix has a non-positive trace; if
            ``n_counting_wires`` is less than one; or if ``shots`` is less than one.
    """
    _validate_counting_width(n_counting_wires)
    if shots < 1:
        raise ValueError(
            f"a quantum PCA run needs at least one shot, got shots={shots}; with no "
            "samples there is no counter value to read an eigenvalue from"
        )
    rho, n_a_wires, n_b_wires = _density_matrix(A)
    circuit = _pca_circuit(
        rho,
        amplitudes=A.reshape(-1),
        n_counting_wires=n_counting_wires,
        n_a_wires=n_a_wires,
        n_b_wires=n_b_wires,
    )
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    observed = circuit.counts(shots, generator=generator)[0]
    # ``counts`` is typed ``dict[str | int, int]`` because its ``format`` can be
    # "int"; this call takes the default "bin", so every key is already the bit
    # string that this mapping is typed as, and the coercion below is a no-op at
    # run time.
    counts = {str(key): count for key, count in observed.items()}
    distribution = _counter_distribution(counts, n_counting_wires)
    mode = max(distribution, key=distribution.__getitem__)
    # The base is a float because the stubs type ``int ** int`` as ``Any``, since a
    # negative exponent yields a float, and an ``Any`` expression fails the type
    # gate.
    resolution = 1.0 / 2.0**n_counting_wires
    return PcaResult(
        dominant_eigenvalue=1.0 - int(mode, 2) / 2.0**n_counting_wires,
        dominant_probability=distribution[mode],
        distribution=distribution,
        n_counting_wires=n_counting_wires,
        resolution=resolution,
    )


class _PhaseFromDensityMatrix:
    """``exp(-2 pi i rho)`` in the controlled forms phase estimation needs.

    The unitary is one dense matrix, so every form is that matrix or a power of it
    with the identity prepended on the control branch: a controlled ``U`` is
    ``diag(I, U**power)`` in the control wire's block order, which is the block
    diagonal of the two. The unconditional ``apply`` embeds the matrix on the data
    register alone.

    Phase estimation consumes the power form only; the other two are what the
    protocol asks of an operator that can also be applied plainly or under one
    control, and they are implemented rather than refused because a dense matrix is
    the case where both are well defined.
    """

    def __init__(self, rho: torch.Tensor) -> None:
        self._matrix = torch.matrix_exp(-2j * math.pi * rho.to(torch.complex128)).to(
            torch.complex64
        )
        self._n_wires = int(self._matrix.shape[0]).bit_length() - 1

    @property
    def n_wires(self) -> int:
        """The number of wires the data register carries."""
        return self._n_wires

    def apply(self, circuit: Circuit, wires: Sequence[int]) -> None:
        """Append ``U`` to ``circuit`` on the data register.

        Args:
            circuit: The circuit to extend.
            wires: The data register's wires.
        """
        circuit.any(*wires, unitary=self._matrix, name="pca_phase")

    def apply_controlled(
        self, circuit: Circuit, control: int, wires: Sequence[int]
    ) -> None:
        """Append ``U`` controlled on ``control``, its power form at exponent one.

        Args:
            circuit: The circuit to extend.
            control: The wire ``U`` is controlled on.
            wires: The data register's wires.
        """
        self.apply_power_controlled(circuit, control, wires, 1)

    def apply_power_controlled(
        self, circuit: Circuit, control: int, wires: Sequence[int], power: int
    ) -> None:
        """Append ``U`` raised to ``power``, controlled on ``control``.

        The exponent is the counting wire's significance, so it reaches
        ``2**(n_counting_wires - 1)``; the power is taken of the matrix itself
        rather than of its exponential, which keeps the emitted gate one dense
        block encoding of the same unitary.

        Args:
            circuit: The circuit to extend.
            control: The wire ``U`` is controlled on.
            wires: The data register's wires.
            power: The exponent, at least zero.
        """
        block = torch.linalg.matrix_power(self._matrix, power)
        identity = torch.eye(block.shape[0], dtype=block.dtype, device=block.device)
        zeros = torch.zeros_like(block)
        controlled = torch.cat(
            (
                torch.cat((identity, zeros), dim=1),
                torch.cat((zeros, block), dim=1),
            ),
            dim=0,
        )
        circuit.any(control, *wires, unitary=controlled, name="pca_phase_power")


def _density_matrix(A: torch.Tensor) -> tuple[torch.Tensor, int, int]:  # noqa: N803
    """Validate the caller's ``A`` and return its density matrix and register widths.

    The Gram matrix and the normalisation are computed in double precision on the
    CPU whatever device the caller holds ``A`` on: they are a classical
    precomputation for the gate matrix, not part of the state the circuit evolves,
    and the trace normalisation is exact enough to matter.

    Args:
        A: The candidate data matrix.

    Returns:
        ``(rho, n_a_wires, n_b_wires)``, where ``rho`` is ``A A^T / tr(A A^T)`` in
        double precision and the widths are the logarithms of ``A``'s dimensions.

    Raises:
        ValueError: If ``A`` is not a two-dimensional real floating-point tensor
            whose dimensions are powers of two of at least two; if it holds a
            non-finite entry; or if its Gram matrix has a non-positive trace.
    """
    if not isinstance(A, torch.Tensor):
        raise ValueError(
            f"A must be a torch.Tensor, got {type(A).__name__}; the density matrix "
            "is built from a data matrix, not from a description of one"
        )
    if A.dim() != 2:
        raise ValueError(
            "A must be a two-dimensional data matrix, got shape "
            f"{tuple(int(size) for size in A.shape)}; its two dimensions are the "
            "registers the purification is prepared on"
        )
    if not A.is_floating_point():
        raise ValueError(
            f"A must be a real floating-point tensor, got dtype {A.dtype}; a density "
            "matrix is built from real data here, and an integer matrix would be "
            "silently promoted"
        )
    rows, columns = (int(size) for size in A.shape)
    for name, size in (("rows", rows), ("columns", columns)):
        if size < 2 or size & (size - 1):
            raise ValueError(
                f"A's {name} must be a power of two of at least two, got {size}; a "
                "register of whole wires carries a power-of-two number of amplitudes"
            )
    if not bool(torch.isfinite(A).all()):
        raise ValueError(
            "A must be finite; the density matrix's trace is a sum of squares of "
            "A's entries, so a single non-finite entry makes every eigenvalue "
            "undefined rather than merely inaccurate"
        )
    data = A.detach().to(device="cpu", dtype=torch.float64)
    gram = data @ data.T
    trace = float(torch.trace(gram))
    if trace <= 0.0:
        raise ValueError(
            f"A's Gram matrix has trace {trace}; the zero matrix has no density "
            "matrix to normalise, and dividing by its trace would return NaN "
            "eigenvalues instead of a refusal"
        )
    return gram / trace, rows.bit_length() - 1, columns.bit_length() - 1


def _pca_circuit(
    rho: torch.Tensor,
    *,
    amplitudes: torch.Tensor,
    n_counting_wires: int,
    n_a_wires: int,
    n_b_wires: int,
) -> Circuit:
    """Build the quantum PCA circuit for ``rho`` and its purification amplitudes.

    The circuit prepares the purification on the data and purification registers,
    then hands the counting register and the data register to
    :func:`~flagquantum.algorithms.primitives.phase_estimation.append_phase_estimation`,
    which emits the counting register's Hadamards, the controlled powers of
    ``exp(-2 pi i rho)``, and the inverse Fourier transform itself.

    Args:
        rho: The density matrix, ``A A^T / tr(A A^T)``.
        amplitudes: The purification ``vec(A)``, one entry per basis state of the
            data and purification registers together.
        n_counting_wires: The width of the counting register, at least one.
        n_a_wires: The width of the data register.
        n_b_wires: The width of the purification register.

    Returns:
        A circuit over ``n_counting_wires + n_a_wires + n_b_wires`` wires.
    """
    data_wires = list(range(n_counting_wires, n_counting_wires + n_a_wires))
    purification_wires = list(
        range(n_counting_wires + n_a_wires, n_counting_wires + n_a_wires + n_b_wires)
    )
    circuit = Circuit(n_counting_wires + n_a_wires + n_b_wires)
    append_arbitrary_state(circuit, amplitudes, data_wires + purification_wires)
    append_phase_estimation(
        circuit,
        unitary=_PhaseFromDensityMatrix(rho),
        counting_wires=list(range(n_counting_wires)),
        evaluation_wires=data_wires,
    )
    return circuit


def _counter_distribution(
    counts: Mapping[str, int], n_counting_wires: int
) -> dict[str, float]:
    """Return the counter register's marginal distribution as shares of the sample.

    Each count key carries every wire, so the data and purification registers' bits
    are folded away before the distribution is formed: a key is truncated to its
    leading ``n_counting_wires`` characters and its count accumulated onto that
    counter value. Reading the full-register keys instead would follow the registers
    the counting register is correlated with and return a distribution that is not
    the eigenvalue distribution at all, without raising.

    Args:
        counts: The sample counts, keyed by the big-endian bit string of the whole
            register.
        n_counting_wires: The width of the counting register, at least one.

    Returns:
        One share per observed counter value, summing to one.
    """
    folded: dict[str, int] = {}
    for key, count in counts.items():
        counter = key[:n_counting_wires]
        folded[counter] = folded.get(counter, 0) + count
    total = sum(folded.values())
    return {key: count / total for key, count in folded.items()}


def _validate_counting_width(n_counting_wires: int) -> None:
    """Check the width of a quantum PCA counting register.

    Args:
        n_counting_wires: The register width as given.

    Raises:
        ValueError: If ``n_counting_wires`` is less than one.
    """
    if n_counting_wires < 1:
        raise ValueError(
            f"quantum PCA needs at least one counting wire, got "
            f"n_counting_wires={n_counting_wires}; a register with no wires resolves "
            "no phase and reads no eigenvalue"
        )
