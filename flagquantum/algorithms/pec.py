"""Probabilistic error cancellation built from a channel's Pauli transfer matrix.

A channel is inverted rather than the circuit being folded. For a Pauli channel
the inverse is an exact finite combination of Pauli words applied after the
channel's own location -- ``E^-1 = sum_i c_i P_i . P_i`` with signed ``c_i``
summing to one -- so inserting that combination immediately after each declared
channel makes the program's composite map the identity, and the observable read
off the corrected program is the noiseless one up to the arithmetic's own floor.
``gamma``, the absolute sum of the weights at a location, is what the correction
costs: an implementation of the same combination that sampled its terms would pay
``gamma**2`` in shots, because a quasi-probability carries the sign of every
weight into its variance.

**The method's one assumption, stated as an obligation.** The noise the program
experiences is the channel the model declares, at the location the model declares
it. Probabilistic error cancellation inverts that channel and nothing else. An
error the model does not carry -- a miscalibrated gate, a leakage process, a drift
between the declaration and the run -- is not inverted and survives the
combination, so a mitigated value that is closer to the ideal is evidence about
the model before it is evidence about the device.

**Why the channel must be a Pauli channel, and what that excludes.** The inverse
is a finite signed Pauli combination exactly when the channel's Pauli transfer
matrix is diagonal, and that is measured here rather than assumed: every
off-diagonal entry is formed and compared with a tolerance. ``bit_flip``,
``phase_flip``, ``depolarizing``, ``two_qubit_depolarizing`` and ``phase_damping``
are diagonal to the arithmetic's floor; ``amplitude_damping`` is not -- at
``gamma = 0.1`` it carries an off-diagonal entry of magnitude ``1.000e-01`` --
and neither are ``coherent_overrotation`` (``1.987e-01`` at angle ``0.2``),
``reset_error`` (``5.000e-02`` at probability ``0.05``) and ``thermal_relaxation``
(``1.000e+00``). Those four are refused by name rather than approximated, because
inverting a non-Pauli channel would silently return the inverse of a different
channel.

**A vanishing eigenvalue is a refusal too.** ``bit_flip`` at probability ``0.5``
sends two Pauli words to exactly zero and drives their weights to infinity, so the
channel has no inverse and the combination is not merely expensive but undefined.
A spectrum that comes within the same tolerance of zero is refused for the same
reason: the weights would exceed ``1e6`` and the exact combination's own rounding
would dominate the answer it reported.

**What is not here.** Clifford data regression and readout-error mitigation are
absent. The estimate is a point value with no confidence interval, because this
slice combines exact state expectations rather than samples; ``sampling_overhead``
is the cost a sampled implementation would pay rather than a measured one.

Dense Pauli-product mathematics lives in :mod:`flagquantum.simulation.pauli` and
is reused here rather than restated. What this module owns is the quasi-probability
arithmetic and the composition of one inverted channel per noise location, which
is the same division of labour ``error_mitigation`` uses for its own fits.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from ..core.ir import CircuitIR, Instruction
from ..noise import KrausChannel, NoiseModel
from ..simulation.density_matrix import density_matrix_from_ir
from ..simulation.pauli import pauli_product_operator
from .core import Hamiltonian

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for typing only
    from ..circuit import Circuit

__all__ = (
    "PEC_ASSUMPTIONS",
    "PEC_LIMITATIONS",
    "PauliTwirlDecomposition",
    "PecLocation",
    "PecResult",
    "pauli_twirl_decomposition",
    "run_pec",
)

#: The margin a channel's Pauli transfer matrix must be diagonal within, and the
#: distance from zero at which a transfer eigenvalue counts as vanished. The two
#: uses share one number because they are the same question -- how far the channel
#: is from the Pauli family this unit can invert -- and one threshold is easier to
#: state than two. It sits between the two margins the channel families actually
#: show: the largest off-diagonal entry of a legitimate family is 3.5e-08 (single
#: precision ``phase_damping``, whose exact value is zero and whose residual is
#: complex64 rounding), while the smallest of a refused family is 5.000e-02
#: (``reset_error`` at probability ``0.05``). Six orders separate them, so no
#: channel is admitted or refused here by a rounding accident.
_DECOMPOSITION_TOLERANCE = 1e-6

#: How many exact programs one combination may evaluate. Each term is one full
#: density simulation, so the term count -- four per single-qubit noise location
#: and sixteen per two-qubit one -- is the unit's real cost. 1024 admits five
#: single-qubit locations, or two single-qubit and three two-qubit ones, and
#: refuses beyond that instead of running a request whose cost the caller cannot
#: have intended.
_MAXIMUM_TERMS = 1024

_PAULI_CHARACTERS = ("I", "X", "Y", "Z")

#: What must hold for the mitigated value to be closer to the noiseless value than
#: the unmitigated one. These are the method's obligations, not the unit's
#: properties, and every result carries a copy so a reader sees them where the
#: number is.
PEC_ASSUMPTIONS: tuple[str, ...] = (
    "The noise the program experiences is exactly the channel the model declares "
    "at the location the model declares it. The inverse cancels the declared "
    "channel, so a device error that differs from the declaration is cancelled "
    "only where it agrees with it.",
    "Each declared channel is a Pauli channel, so its inverse is a finite signed "
    "combination of Pauli words. A channel whose Pauli transfer matrix is not "
    "diagonal, and one whose transfer spectrum reaches zero, are refused rather "
    "than approximated.",
    "The observable is read exactly from the state, Tr(O rho). Readout confusion "
    "is a classical misassignment applied after measurement and is not part of "
    "rho, so a model that declares a readout rule is refused rather than measured "
    "without it.",
)

#: The limitations every result carries.
PEC_LIMITATIONS: tuple[str, ...] = (
    "The combination is not a physical state. Its weights are signed, their "
    "absolute sum exceeds one, and the scaled terms are not density matrices, so "
    "an implementation that sampled the same combination would return values "
    "outside the observable's spectrum, with a variance that grows with "
    "sampling_overhead. This path sums the terms exactly and reports no variance, "
    "and the exactness it has is the exactness of the density simulation beneath "
    "it rather than evidence about a device.",
    "sampling_overhead is gamma squared, the shot cost an implementation that "
    "sampled the same combination would pay. This path consumes no shots: it "
    "evaluates every term exactly, so the figure is arithmetic rather than "
    "measured, and no confidence interval is reported beside the estimate.",
    "The inverse cancels the noise the model declares. An error the model does not "
    "carry -- a miscalibrated gate, a leakage process, a drift between the "
    "declaration and the run -- survives the combination, so a mitigated value "
    "closer to the ideal is evidence about the model before it is evidence about "
    "the device.",
    "The combination needs one exact program per quasi-probability term, so the "
    "cost is four terms per single-qubit noise location and sixteen per two-qubit "
    "one, and the term count is what the run spends rather than a sampled "
    "estimate's shot count. A request above the unit's term cap is refused by name "
    "rather than run.",
    "The program is re-executed once per term from its IR, and an IR begins at the "
    "all-zero state, so a circuit that declares its own input state is refused "
    "rather than measured from a different state than the one it declares.",
    "Clifford data regression and readout-error mitigation are absent, and no "
    "gate-folding scale factor is offered: the inverse is built from the channel's "
    "Pauli transfer matrix rather than by folding gates, so the cost is reported "
    "as a term count rather than as a fold count.",
)


def _pauli_words(n_qubits: int) -> tuple[str, ...]:
    """Every Pauli word on ``n_qubits``, identity first, in one fixed order.

    The order is load-bearing: a transfer spectrum indexed by this tuple has the
    identity at position zero, which is what makes ``sum(coefficients) == 1`` the
    trace-preservation check it is.
    """

    return tuple(
        "".join(characters)
        for characters in itertools.product(_PAULI_CHARACTERS, repeat=n_qubits)
    )


def _word_operators(word: str, n_qubits: int) -> tuple[tuple[int, str], ...]:
    """One word's non-identity factors, as the channel-local wire/name pairs."""

    if len(word) != n_qubits:
        raise ValueError(
            f"a Pauli word on {n_qubits} qubit(s) needs {n_qubits} character(s), and "
            f"{word!r} has {len(word)}"
        )
    return tuple(
        (wire, character.lower())
        for wire, character in enumerate(word)
        if character != "I"
    )


def _commutation_parity(left: str, right: str) -> int:
    """``0`` when two Pauli words commute and ``1`` when they anticommute.

    Two words built from ``I``, ``X``, ``Y`` and ``Z`` anticommute exactly at the
    positions where they differ and neither is the identity, so the parity of that
    count is the sign of ``P_left P_right P_left`` against ``P_right``.
    """

    return (
        sum(
            1
            for a, b in zip(left, right, strict=True)
            if a != b and a != "I" and b != "I"
        )
        % 2
    )


def _pauli_transfer_matrix(channel: KrausChannel) -> torch.Tensor:
    """The channel's action on the Pauli basis, indexed by :func:`_pauli_words`.

    Entry ``[i, j]`` is ``Tr(P_i E(P_j)) / d`` in double precision whatever
    precision the channel's operators carry, so the decomposition is not limited
    by the width of the channel it reads.
    """

    n_qubits = channel.n_wires
    dimension = 1 << n_qubits
    operators = [
        pauli_product_operator(
            _word_operators(word, n_qubits),
            n_qubits,
            dtype=torch.complex128,
            device="cpu",
        )
        for word in _pauli_words(n_qubits)
    ]
    kraus = [
        op.detach().to(device="cpu", dtype=torch.complex128) for op in channel.kraus
    ]
    images = torch.stack(
        [
            sum(
                (op @ operator @ op.mH for op in kraus),
                start=torch.zeros_like(operator),
            )
            for operator in operators
        ]
    )
    return torch.einsum("iab,jba->ij", torch.stack(operators), images) / dimension


def _quasi_probabilities(
    words: tuple[str, ...], spectrum: torch.Tensor
) -> tuple[float, ...]:
    """The inverse channel's signed Pauli weights.

    Writing the inverse as ``E^-1 = sum_i c_i P_i . P_i`` and using ``E(P_j) =
    lambda_j P_j``, applying that to ``P_j`` gives ``P_j sum_i c_i
    (-1)**parity(i, j)``, so the weights must satisfy ``sum_i c_i
    (-1)**parity(i, j) == 1 / lambda_j``. The sign matrix ``(-1)**parity(i, j)``
    is its own inverse up to ``1/N``, which is the Walsh transform over the Pauli
    group, so ``c_i = (1/N) sum_j (-1)**parity(i, j) / lambda_j``. The reciprocal
    sits inside the sum because what is being expanded is the inverse of ``E``.
    """

    size = len(words)
    weights: list[float] = []
    for left in words:
        total = 0.0
        for right, eigenvalue in zip(words, spectrum.tolist(), strict=True):
            sign = -1.0 if _commutation_parity(left, right) else 1.0
            total += sign / eigenvalue
        weights.append(total / size)
    return tuple(weights)


@dataclass(frozen=True, slots=True)
class PauliTwirlDecomposition:
    """One Pauli channel's inverse as a signed distribution over Pauli words.

    Attributes:
        channel_name: The channel this inverts, by the name it declares.
        n_wires: How many wires the channel acts on.
        pauli_words: The basis the weights are indexed by, identity first, in the
            fixed order :func:`_pauli_words` produces.
        coefficients: The signed weight of each word in the inverse channel.
        transfer_spectrum: The eigenvalue each word's operator is sent to, in the
            same order, so a reader can see what the inversion divided by.
        largest_offdiagonal: The largest off-diagonal entry the channel's Pauli
            transfer matrix carried. It is at or below the unit's tolerance by
            construction, and it is recorded because it is the margin by which the
            channel is a Pauli channel rather than an assertion that it is one.
    """

    channel_name: str
    n_wires: int
    pauli_words: tuple[str, ...]
    coefficients: tuple[float, ...]
    transfer_spectrum: tuple[float, ...]
    largest_offdiagonal: float

    def __post_init__(self) -> None:
        if not self.channel_name:
            raise ValueError("a decomposition must name the channel it inverts")
        if self.n_wires < 1:
            raise ValueError(
                f"a channel acts on at least one wire, and {self.n_wires} was given"
            )
        expected = _pauli_words(self.n_wires)
        if self.pauli_words != expected:
            raise ValueError(
                "the Pauli words must be every word on this many wires, identity "
                "first, in the unit's own order, so that a weight and the word it "
                "belongs to cannot be read against different bases"
            )
        if not (
            len(self.coefficients)
            == len(self.transfer_spectrum)
            == len(self.pauli_words)
        ):
            raise ValueError(
                "every Pauli word needs one coefficient and one transfer "
                "eigenvalue, and the three sequences differ in length"
            )
        for name, values in (
            ("coefficients", self.coefficients),
            ("transfer_spectrum", self.transfer_spectrum),
        ):
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must be finite, and one entry is not")
        if not math.isfinite(self.largest_offdiagonal):
            raise ValueError("largest_offdiagonal must be finite, and it is not")
        # The inverse of a trace-preserving channel is a signed distribution that
        # still sums to one: the identity's weight is what survives the transform.
        # Checking it here makes the record refutable rather than self-declared.
        total = math.fsum(self.coefficients)
        if abs(total - 1.0) > _DECOMPOSITION_TOLERANCE:
            raise ValueError(
                "the coefficients must sum to one, because they are the inverse of "
                f"a trace-preserving channel, and they sum to {total!r}"
            )

    @property
    def gamma(self) -> float:
        """The absolute sum of the weights, which is what the estimate costs."""

        return math.fsum(abs(value) for value in self.coefficients)

    @property
    def sampling_overhead(self) -> float:
        """The shot cost a sampled implementation of this channel would pay."""

        return self.gamma**2

    @property
    def dominant_word(self) -> tuple[str, float]:
        """The largest-magnitude term, which dominates both the value and the cost."""

        index = max(
            range(len(self.coefficients)),
            key=lambda position: abs(self.coefficients[position]),
        )
        return self.pauli_words[index], self.coefficients[index]


def pauli_twirl_decomposition(channel: KrausChannel) -> PauliTwirlDecomposition:
    """Decompose one Pauli channel's inverse into signed Pauli words.

    Args:
        channel: The channel to invert. It must be a Pauli channel -- its Pauli
            transfer matrix diagonal to ``1e-6`` -- and its transfer spectrum must
            stay ``1e-6`` away from zero.

    Returns:
        The weights, the spectrum they were read from, and the off-diagonal margin
        the channel was admitted at.

    Raises:
        TypeError: if ``channel`` is not a :class:`~flagquantum.noise.KrausChannel`.
        ValueError: if the transfer matrix is not diagonal, if the spectrum is not
            real, or if it reaches zero. All three mean the inverse is not the
            finite signed Pauli combination this unit can apply, so the channel is
            refused by name instead of approximated.
    """

    if not isinstance(channel, KrausChannel):
        raise TypeError(
            "a Pauli twirl decomposition needs a Kraus channel, and "
            f"{type(channel).__name__} was given"
        )
    words = _pauli_words(channel.n_wires)
    transfer = _pauli_transfer_matrix(channel)
    diagonal = torch.diag(transfer.diagonal())
    off_diagonal = transfer - diagonal
    largest_offdiagonal = float(torch.abs(off_diagonal).max())
    if largest_offdiagonal > _DECOMPOSITION_TOLERANCE:
        flat = int(torch.argmax(torch.abs(off_diagonal)))
        row, column = divmod(flat, len(words))
        raise ValueError(
            f"channel {channel.name!r} is not a Pauli channel: its Pauli transfer "
            f"matrix sends {words[column]} partly to {words[row]}, an off-diagonal "
            f"entry of magnitude {largest_offdiagonal:.3e}, above this unit's "
            f"{_DECOMPOSITION_TOLERANCE:.0e} tolerance. The inverse of such a "
            "channel is not a finite signed combination of Pauli words, so it is "
            "refused rather than approximated."
        )
    imaginary = float(torch.abs(transfer.diagonal().imag).max())
    if imaginary > _DECOMPOSITION_TOLERANCE:
        raise ValueError(
            f"channel {channel.name!r} has a Pauli transfer spectrum with an "
            f"imaginary part of magnitude {imaginary:.3e}, above this unit's "
            f"{_DECOMPOSITION_TOLERANCE:.0e} tolerance. Its inverse is then not a "
            "real signed combination of Pauli words, so the channel is refused."
        )
    spectrum = transfer.diagonal().real
    vanished = [
        index
        for index, eigenvalue in enumerate(spectrum.tolist())
        if abs(eigenvalue) <= _DECOMPOSITION_TOLERANCE
    ]
    if vanished:
        index = vanished[0]
        raise ValueError(
            f"channel {channel.name!r} has a vanishing Pauli transfer eigenvalue at "
            f"{words[index]} (value {float(spectrum[index]):.3e}), within this "
            f"unit's {_DECOMPOSITION_TOLERANCE:.0e} tolerance of zero. The channel "
            "has no inverse there, so its quasi-probability weights are unbounded "
            "and the combination is undefined rather than merely expensive."
        )
    return PauliTwirlDecomposition(
        channel_name=channel.name,
        n_wires=channel.n_wires,
        pauli_words=words,
        coefficients=_quasi_probabilities(words, spectrum),
        transfer_spectrum=tuple(float(value) for value in spectrum.tolist()),
        largest_offdiagonal=largest_offdiagonal,
    )


@dataclass(frozen=True, slots=True)
class PecLocation:
    """One noise location the combination inverts.

    Attributes:
        program_index: Where the channel sits in the lowered program, which is what
            distinguishes two occurrences of the same channel on the same wires.
        channel_name: The channel's own name.
        wires: The wires the channel acts on, in the order its words are indexed by.
        decomposition: The inverse the combination applies here.
    """

    program_index: int
    channel_name: str
    wires: tuple[int, ...]
    decomposition: PauliTwirlDecomposition

    def __post_init__(self) -> None:
        if self.program_index < 0:
            raise ValueError(
                f"a location's program index cannot be negative, and "
                f"{self.program_index} was given"
            )
        if not isinstance(self.decomposition, PauliTwirlDecomposition):
            raise TypeError("a location needs the decomposition it applies")
        if self.channel_name != self.decomposition.channel_name:
            raise ValueError(
                "a location and its decomposition must name the same channel, and "
                f"they name {self.channel_name!r} and "
                f"{self.decomposition.channel_name!r}"
            )
        if len(self.wires) != self.decomposition.n_wires:
            raise ValueError(
                f"channel {self.channel_name!r} acts on "
                f"{self.decomposition.n_wires} wire(s) and the location names "
                f"{len(self.wires)}"
            )

    @property
    def gamma(self) -> float:
        """The cost this location's inverse adds to the combination."""

        return self.decomposition.gamma


@dataclass(frozen=True, slots=True)
class PecResult:
    """A mitigated estimate together with the combination it came from.

    Attributes:
        estimate: The quasi-probability average of the combined terms.
        unmitigated: The same observable read from the program under the model,
            without any inverse applied. It is what the combination corrected, and
            it is not a noiseless value.
        locations: The noise locations the combination inverted, in program order.
        term_count: How many exact programs the combination summed. Every term is
            one full density simulation, so this is the unit's cost.
        assumptions: What had to hold for the estimate to mean anything.
        limitations: What the estimate does not carry.
    """

    estimate: float
    unmitigated: float
    locations: tuple[PecLocation, ...]
    term_count: int
    assumptions: tuple[str, ...] = PEC_ASSUMPTIONS
    limitations: tuple[str, ...] = PEC_LIMITATIONS

    def __post_init__(self) -> None:
        for name in ("estimate", "unmitigated"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, and {value!r} is not")
        if not self.locations:
            raise ValueError(
                "a result needs at least one inverted location, because a "
                "combination with no channel in it has mitigated nothing"
            )
        for location in self.locations:
            if not isinstance(location, PecLocation):
                raise TypeError("a result needs the locations it inverted")
        # The cost is a property of the locations, so stating it separately is
        # only honest if the two agree: a result that under-reports the work it
        # did cannot be read as evidence about what the work bought.
        expected = math.prod(
            len(location.decomposition.pauli_words) for location in self.locations
        )
        if self.term_count != expected:
            raise ValueError(
                f"the term count must be the product of the locations' word counts, "
                f"which is {expected}, and {self.term_count} was recorded"
            )
        for name in ("assumptions", "limitations"):
            declared = getattr(self, name)
            if not declared or any(not item.strip() for item in declared):
                raise ValueError(
                    f"{name} must state at least one non-empty entry, because an "
                    "estimate whose caveats are empty reads as a claim the method "
                    "has not made"
                )

    @property
    def gamma(self) -> float:
        """The absolute sum of the combined weights over every location."""

        return math.prod(location.gamma for location in self.locations)

    @property
    def sampling_overhead(self) -> float:
        """The shot cost a sampled implementation of this combination would pay."""

        return self.gamma**2

    @property
    def executions(self) -> int:
        """How many exact programs were simulated, the unmitigated read included."""

        return self.term_count + 1

    @property
    def applied_correction(self) -> float:
        """How far the estimate moved from the unmitigated read."""

        return self.estimate - self.unmitigated


def _corrected(
    lowered: CircuitIR,
    corrections: dict[int, str],
) -> CircuitIR:
    """The lowered program with one Pauli conjugation inserted per noise location.

    Each correction goes immediately after the channel it inverts, which is what
    makes the insertion a statement about that channel alone: ``E^-1`` composed
    with the ``E`` it follows is the identity channel, so the whole program's
    composite map telescopes back to the noiseless one whatever sits between the
    locations. Nothing already in the program moves, so the corrected program is
    the same program with the same gates in the same order, and every term in the
    combination is a statement about that one program.
    """

    instructions: list[Instruction] = []
    for index, instruction in enumerate(lowered):
        instructions.append(instruction)
        word = corrections.get(index)
        if word is None:
            continue
        n_qubits = len(instruction.wires)
        operator = pauli_product_operator(
            _word_operators(word, n_qubits),
            n_qubits,
            dtype=torch.complex128,
            device="cpu",
        )
        instructions.append(
            Instruction(
                name="pec_pauli",
                wires=instruction.wires,
                params={"pauli_word": word.lower()},
                matrix=(operator,),
                metadata={"is_channel": True},
            )
        )
    return replace(lowered, instructions=tuple(instructions))


def _declared_channel(instruction: Instruction) -> KrausChannel:
    """Rebuild the channel a lowered instruction carries.

    The lowering names the parameter the channel was built with beside the Kraus
    operators those were turned into, so reconstructing the channel here reads the
    same declaration the lowering wrote rather than inverting a probability back
    out of the operators.
    """

    declared = instruction.matrix
    if not isinstance(declared, (tuple, list)) or not declared:
        raise ValueError(
            f"the lowered channel instruction {instruction.name!r} carries no Kraus "
            "operators, so there is no channel to invert; rebuild the noise model "
            "from a channel factory rather than from an operator-less declaration."
        )
    return KrausChannel(
        instruction.name,
        tuple(declared),
        tuple((str(key), value) for key, value in dict(instruction.params).items()),
    )


def _read(
    ir: CircuitIR,
    hamiltonian: Hamiltonian,
    *,
    bsz: int,
    device: torch.device | str,
    dtype: torch.dtype | None,
) -> float:
    """Simulate one exact program and read one number off it."""

    density = density_matrix_from_ir(ir, bsz=bsz, device=device, dtype=dtype)
    values = torch.as_tensor(hamiltonian.expectation(density)).reshape(-1)
    if values.numel() != 1:
        raise ValueError(
            "probabilistic error cancellation needs one expectation value per "
            f"term, and the Hamiltonian returned {values.numel()}; run it at batch "
            "size one"
        )
    value = float(values[0])
    if not math.isfinite(value):
        raise ValueError(
            f"a term of the combination read a non-finite expectation ({value!r}), "
            "so the weighted sum would report it as the estimate"
        )
    return value


def run_pec(
    circuit: Circuit,
    hamiltonian: Hamiltonian,
    *,
    noise_model: NoiseModel,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> PecResult:
    """Invert a declared noise model and read the observable off the result.

    Args:
        circuit: The program whose observable is mitigated. It is re-executed once
            per quasi-probability term, unchanged except at the noise locations, so
            every term is a statement about the same program.
        hamiltonian: The observable, evaluated exactly on each term's density
            matrix. Its expectation is a single number, so the circuit's batch size
            must be one.
        noise_model: The model to invert. Every channel it puts in the program is
            inverted at the location it was put in. It must not declare a readout
            rule: the observable is read as ``Tr(O rho)``, which is before
            measurement, so a classical readout confusion would be silently unused.
        device: The device the density simulations run on.
        dtype: The density simulations' complex dtype. The default is the runtime
            configuration's, which is single precision; the decomposition's own
            arithmetic is double precision regardless.

    Returns:
        The estimate, the unmitigated read, the inverted locations, the term count
        they cost, and the assumptions and limitations that go with them.

    Raises:
        TypeError: if ``noise_model`` is not a :class:`~flagquantum.noise.NoiseModel`.
        ValueError: if the model puts no channel in the program, if it declares a
            readout rule the observable path cannot apply, if the circuit's batch
            size is not one, if the circuit declares its own input state, if the
            combination needs more terms than the unit's cap, or if a declared
            channel is not one this unit can invert. All of it is checked before
            the first simulation runs, so a mis-specified request costs no
            simulation time.
    """

    from ..compiler import lower_noise_model

    if not isinstance(noise_model, NoiseModel):
        raise TypeError(
            "probabilistic error cancellation needs the noise model to invert, and "
            f"{type(noise_model).__name__} was given"
        )
    if noise_model.readout_rules:
        raise ValueError(
            "the noise model declares a readout rule, and this unit evaluates the "
            "observable as Tr(O rho) before measurement, where classical readout "
            "confusion is not represented; measuring anyway would report a "
            "state-preparation estimate while the rule stayed silently unused, so "
            "the model is refused instead. Drop the readout rules to mitigate the "
            "state-preparation estimate, or mitigate readout separately."
        )
    if int(circuit.bsz) != 1:
        raise ValueError(
            "probabilistic error cancellation reads one expectation per term, so "
            f"the circuit's batch size must be one, and it is {circuit.bsz}"
        )
    if getattr(circuit, "_inputs", None) is not None:
        raise ValueError(
            "the circuit declares its own input state, and this unit re-executes "
            "the program once per quasi-probability term from its IR, where every "
            "program begins at the all-zero state; build the same program from "
            "gates instead of an input state."
        )
    lowered = lower_noise_model(circuit, noise_model)
    locations = tuple(
        PecLocation(
            program_index=index,
            channel_name=instruction.name,
            wires=tuple(instruction.wires),
            decomposition=pauli_twirl_decomposition(_declared_channel(instruction)),
        )
        for index, instruction in enumerate(lowered)
        if instruction.metadata.get("is_channel")
    )
    if not locations:
        raise ValueError(
            "the noise model puts no channel in this program, so there is nothing "
            "to invert and no mitigated estimate to report; an identity model's "
            "mitigation is the unmitigated value under another name"
        )
    term_count = math.prod(
        len(location.decomposition.pauli_words) for location in locations
    )
    if term_count > _MAXIMUM_TERMS:
        raise ValueError(
            f"this model's combination needs {term_count} exact programs, above "
            f"this unit's cap of {_MAXIMUM_TERMS}: four terms per single-qubit "
            "noise location and sixteen per two-qubit one. Reduce the number of "
            "noisy locations, or mitigate the largest channel alone and read the "
            "result as a statement about that channel."
        )
    bsz = int(circuit.bsz)
    estimate = 0.0
    for selection in itertools.product(
        *(range(len(location.decomposition.pauli_words)) for location in locations)
    ):
        weight = 1.0
        corrections: dict[int, str] = {}
        for location, word_index in zip(locations, selection, strict=True):
            decomposition = location.decomposition
            # The weight is the quasi-probability itself. gamma is not a factor
            # of the combination -- it is the cost, read off the weights' absolute
            # sum -- so folding it in here would return gamma times the answer.
            weight *= decomposition.coefficients[word_index]
            corrections[location.program_index] = decomposition.pauli_words[word_index]
        estimate += weight * _read(
            _corrected(lowered, corrections),
            hamiltonian,
            bsz=bsz,
            device=device,
            dtype=dtype,
        )
    return PecResult(
        estimate=estimate,
        unmitigated=_read(
            lowered,
            hamiltonian,
            bsz=bsz,
            device=device,
            dtype=dtype,
        ),
        locations=locations,
        term_count=term_count,
    )
