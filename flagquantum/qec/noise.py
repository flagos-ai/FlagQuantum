"""Circuit-location noise profiles and finite-shot QEC sweeps."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral, Real

from ..noise import NoiseModel, ReadoutError, bit_flip_channel
from .repetition import run_repetition_memory_experiment
from .types import NoiseSweepPoint


def _probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real probability")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return normalized


def _rate_vector(values: object, *, name: str) -> tuple[float, ...]:
    """Return ``values`` as a tuple of probabilities, naming the bad element.

    The element's index is part of the message because a vector is written as a
    list: ``data_flip_per_qubit[3]`` is what a caller can act on, where the field
    name alone leaves them counting entries.
    """

    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{name} must be an iterable of probabilities")
    return tuple(
        _probability(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )


def _resolved_rates(
    vector: tuple[float, ...],
    scalar: float,
    count: int,
    *,
    vector_name: str,
    element: str,
) -> tuple[float, ...]:
    """Return one effective rate per element, resolving the override rule.

    The rule is stated on :class:`PhenomenologicalNoise`: a non-empty vector
    replaces the scalar for every element. The length is checked here rather than
    at construction because the record is built without the code it is read
    beside, and a vector that names fewer elements than the code has would
    silently read one element's rate as another's.
    """

    if isinstance(count, bool) or not isinstance(count, Integral):
        raise TypeError(f"the number of {element}s must be an integer")
    total = int(count)
    if total < 0:
        raise ValueError(f"the number of {element}s cannot be negative")
    if not vector:
        return (scalar,) * total
    if len(vector) != total:
        raise ValueError(
            f"{vector_name} states {len(vector)} rate(s) but the code declares "
            f"{total} {element}(s), so one {element}'s rate would be read as "
            f"another's"
        )
    return vector


def _symmetric_readout_error(probability: float) -> ReadoutError:
    return ReadoutError(
        ((1.0 - probability, probability), (probability, 1.0 - probability))
    )


@dataclass(frozen=True)
class RepetitionNoiseProfile:
    """Bounded circuit-location noise configuration for the reference code."""

    data_bit_flip_probability: float = 0.0
    syndrome_readout_error_probability: float = 0.0
    final_readout_error_probability: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "data_bit_flip_probability",
            "syndrome_readout_error_probability",
            "final_readout_error_probability",
        ):
            object.__setattr__(
                self,
                name,
                _probability(getattr(self, name), name=name),
            )

    def to_noise_model(self) -> NoiseModel:
        """Map QEC locations onto the backend-neutral NoiseModel authority."""

        model = NoiseModel()
        if self.data_bit_flip_probability:
            channel = bit_flip_channel(self.data_bit_flip_probability)
            for wire in range(3):
                model.add("cx", channel, qubits=wire)
        if self.syndrome_readout_error_probability:
            error = _symmetric_readout_error(self.syndrome_readout_error_probability)
            model.add_readout((3, 4), error)
        if self.final_readout_error_probability:
            error = _symmetric_readout_error(self.final_readout_error_probability)
            model.add_readout((0, 1, 2), error)
        return model


@dataclass(frozen=True)
class PhenomenologicalNoise:
    """Independent per-location data and measurement faults.

    The three data fields name the three single-qubit Pauli faults:
    ``data_flip`` is an X fault, ``phase_flip`` a Z fault, and ``both_flip`` a Y
    fault, which is the two of them at once. Each applies to every data qubit at
    the start of every syndrome round, before that round's parity-check CNOTs.
    ``measurement_flip`` applies to every syndrome measurement of every check in
    every round. All four are independent per location per round.

    The data fault families are separate because a code-capacity model detects
    them differently: an X fault flips the Z-type checks, a Z fault flips the
    X-type checks, and a Y fault flips both. Setting all three to the same rate is
    therefore a depolarizing channel, and setting only ``data_flip`` is the
    bit-flip channel this record started as.

    The rates are uniform *unless* a per-element vector is stated beside them.
    ``data_flip_per_qubit``, ``phase_flip_per_qubit``, ``both_flip_per_qubit`` and
    ``measurement_flip_per_check`` name upstream's ``px_per_qubit``,
    ``pz_per_qubit``, ``py_per_qubit`` and ``pm_per_check`` element for element.
    The mapping is by fault rather than by position, because the two records list
    their faults in a different order: this one states the X fault, the Z fault,
    the Y fault and the measurement fault, while upstream's ``CssNoise`` states
    ``px``, ``py``, ``pz`` and ``pm`` -- the Z and Y names change places, so the
    third field here is the second there.

    A stated vector **overrides** its scalar and it overrides it **wholesale**:
    there is no element-by-element mixture of the two, which is why a vector has
    to name every element. That is upstream's rule ("per-element override vectors
    take priority when non-empty"), and it makes an empty vector the unstated
    state rather than a vector of no-ops.

    A vector is read only beside a code, so the record states one and the length
    is refused where the code is known, by the ``*_rates`` accessors below. The
    index of an element is stated there too: the per-qubit vectors are indexed by
    the code's own ``data_qubits`` order, which is the column order
    ``css_code_matrices`` gives every matrix rather than a qubit number, and the
    per-check vector by the matrix row order upstream uses -- Z-type checks
    first, in the order the code declares them, and then the X-type checks. A
    code that declares only Z-type checks, the repetition code among them, sees
    those two orders coincide.

    This is a description of noise locations, not a ``NoiseModel``. The
    round-boundary data location has no equivalent in the executor's
    instruction-ordered noise grammar, so no ``to_noise_model`` is offered and
    the DEM's detector-rate cross-check drives this record through forced
    injection instead.
    """

    data_flip: float = 0.0
    phase_flip: float = 0.0
    both_flip: float = 0.0
    measurement_flip: float = 0.0
    data_flip_per_qubit: tuple[float, ...] = ()
    phase_flip_per_qubit: tuple[float, ...] = ()
    both_flip_per_qubit: tuple[float, ...] = ()
    measurement_flip_per_check: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("data_flip", "phase_flip", "both_flip", "measurement_flip"):
            object.__setattr__(self, name, _probability(getattr(self, name), name=name))
        for name in (
            "data_flip_per_qubit",
            "phase_flip_per_qubit",
            "both_flip_per_qubit",
            "measurement_flip_per_check",
        ):
            object.__setattr__(self, name, _rate_vector(getattr(self, name), name=name))

    def data_flip_rates(self, *, num_qubits: int) -> tuple[float, ...]:
        """Return the effective X-fault rate of every data qubit, in qubit order.

        An element whose rate is zero is a location that cannot fire, so the
        construction routes enumerate no mechanism for it.
        """

        return _resolved_rates(
            self.data_flip_per_qubit,
            self.data_flip,
            num_qubits,
            vector_name="data_flip_per_qubit",
            element="data qubit",
        )

    def phase_flip_rates(self, *, num_qubits: int) -> tuple[float, ...]:
        """Return the effective Z-fault rate of every data qubit, in qubit order."""

        return _resolved_rates(
            self.phase_flip_per_qubit,
            self.phase_flip,
            num_qubits,
            vector_name="phase_flip_per_qubit",
            element="data qubit",
        )

    def both_flip_rates(self, *, num_qubits: int) -> tuple[float, ...]:
        """Return the effective Y-fault rate of every data qubit, in qubit order."""

        return _resolved_rates(
            self.both_flip_per_qubit,
            self.both_flip,
            num_qubits,
            vector_name="both_flip_per_qubit",
            element="data qubit",
        )

    def measurement_flip_rates(self, *, num_checks: int) -> tuple[float, ...]:
        """Return the effective measurement-fault rate of every check.

        The order is the matrix row order: Z-type checks first, then X-type
        checks, which is upstream's ``pm_per_check`` order and not the order the
        code declares its checks in.
        """

        return _resolved_rates(
            self.measurement_flip_per_check,
            self.measurement_flip,
            num_checks,
            vector_name="measurement_flip_per_check",
            element="check",
        )


def run_repetition_memory_noise_sweep(
    probabilities: Iterable[float],
    *,
    rounds: int = 3,
    shots: int = 1024,
    seed: int | None = 0,
    strategy: str = "auto",
    feedback_mode: str = "compiled_lookup",
    syndrome_readout_error_probability: float = 0.0,
    final_readout_error_probability: float = 0.0,
) -> tuple[NoiseSweepPoint, ...]:
    """Measure finite-shot logical outcomes without a suppression claim."""

    values = tuple(
        _probability(value, name="data_bit_flip_probability") for value in probabilities
    )
    if not values:
        raise ValueError("noise sweep requires at least one probability")
    points = []
    for index, probability in enumerate(values):
        profile = RepetitionNoiseProfile(
            data_bit_flip_probability=probability,
            syndrome_readout_error_probability=syndrome_readout_error_probability,
            final_readout_error_probability=final_readout_error_probability,
        )
        result = run_repetition_memory_experiment(
            rounds=rounds,
            shots=shots,
            seed=None if seed is None else seed + index,
            strategy=strategy,
            feedback_mode=feedback_mode,
            noise_model=profile.to_noise_model(),
        )
        points.append(
            NoiseSweepPoint(
                probability=probability,
                shots=result.shot_count,
                logical_failures=result.logical_failures,
                logical_error_rate=result.logical_error_rate,
                bit_flip_events=result.bit_flip_events,
                readout_errors=result.readout_errors,
            )
        )
    return tuple(points)


__all__ = (
    "PhenomenologicalNoise",
    "RepetitionNoiseProfile",
    "run_repetition_memory_noise_sweep",
)
