"""Output requests and their lowering to shared measurement IR."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module as _import_public_namespace

import torch

from flagquantum.core._qubit_aliases import OMITTED, Omitted, warn_qubit_alias
from flagquantum.core.ir import MeasurementNode
from flagquantum.core.operators.observable import (
    Observable,
    _PauliTerm,
    _qubits,
)
from flagquantum.errors import CapabilityError

_DEFAULT_MAX_PAULI_SAMPLE_WIRES = 8


def log_base_divisor(log_base: float | None) -> float:
    """Return the number to divide a natural-log entropy by.

    ``None`` means nats and divides by one. PennyLane 0.45.1 divides by
    ``log(log_base)`` and validates nothing, so ``log_base=1`` returns ``inf``,
    ``log_base=-1`` returns ``nan`` and ``log_base=0`` returns the natural-log
    answer by accident of a truthiness test. Zero, negative and one are refused
    here instead: each of those is an answer the caller cannot have meant, and a
    silent ``nan`` travelling out through a result field is exactly the failure
    mode the package refuses elsewhere. The divergence from PennyLane is recorded
    in ``docs/api-changes/FQ-VN-ENTROPY-OUTPUT-20261101.md``.
    """

    if log_base is None:
        return 1.0
    if isinstance(log_base, bool) or not isinstance(log_base, (int, float)):
        raise TypeError(f"log_base must be a number, got {log_base!r}")
    value = float(log_base)
    if not math.isfinite(value):
        raise ValueError("log_base must be finite")
    if value <= 0.0:
        raise ValueError("log_base must be positive")
    if value == 1.0:
        raise ValueError("log_base must not be 1, which has no logarithm")
    return math.log(value)


log_base_divisor.__module__ = "flagquantum.simulation.entropy"


@dataclass(frozen=True, slots=True, eq=False)
class OutputRequest:
    """A backend-neutral description of one requested execution output."""

    kind: str
    qubits: tuple[int, ...] = ()
    observable: Observable | torch.Tensor | None = None
    name: str | None = None
    log_base: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.observable, torch.Tensor) and self.kind != "expectation":
            raise TypeError("matrix observables require expectation output")
        if self.kind not in {
            "counts",
            "density_matrix",
            "expectation",
            "probabilities",
            "samples",
            "vn_entropy",
        }:
            raise ValueError(f"unsupported output kind {self.kind!r}")
        object.__setattr__(
            self, "qubits", _qubits(self.qubits, owner=f"{self.kind} output")
        )
        if self.name is not None and (
            not isinstance(self.name, str) or not self.name.strip()
        ):
            raise ValueError("output name must be a non-empty string")
        if self.kind == "expectation":
            if isinstance(self.observable, torch.Tensor):
                operator = self.observable
                if (
                    operator.layout != torch.strided
                    or operator.is_nested
                    or operator.is_quantized
                    or operator.device.type == "meta"
                    or operator.dtype == torch.bool
                ):
                    raise TypeError(
                        "expectation requires an ordinary numeric dense tensor"
                    )
                if not operator.is_floating_point() and not operator.is_complex():
                    try:
                        torch.iinfo(operator.dtype)
                    except TypeError as exc:
                        raise TypeError(
                            "expectation requires a numeric tensor"
                        ) from exc
                if (
                    operator.ndim != 2
                    or not operator.shape[0]
                    or operator.shape[0] != operator.shape[1]
                ):
                    raise ValueError("expectation matrix must be nonempty and square")
                if self.qubits:
                    raise ValueError(
                        "full-space expectations do not accept qubit selectors"
                    )
            elif not isinstance(self.observable, Observable):
                raise TypeError(
                    "expectation output requires an Observable or dense torch.Tensor"
                )
        if self.kind in {"density_matrix", "probabilities"} and (
            self.observable is not None
        ):
            raise TypeError(f"{self.kind} output does not accept an observable")
        if self.kind == "vn_entropy":
            if self.observable is not None:
                raise TypeError("vn_entropy output does not accept an observable")
            if not self.qubits:
                raise ValueError(
                    "vn_entropy output requires at least one qubit: the entropy "
                    "of no subsystem is not a number"
                )
            # The base is validated here rather than at execution, so a base that has no
            # logarithm is refused where the request is written instead of in the middle
            # of a run. The rule itself lives with the entropy arithmetic so that one
            # logarithm convention serves the request and the measurement.
            log_base_divisor(self.log_base)
        elif self.log_base is not None:
            raise TypeError(f"{self.kind} output does not accept a log_base")
        if self.kind in {"samples", "counts"} and self.observable is not None:
            assert isinstance(self.observable, Observable)
            _sampled_pauli_term(self.observable)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutputRequest) or type(self) is not type(other):
            return NotImplemented
        if (self.kind, self.qubits, self.name, self.log_base) != (
            other.kind,
            other.qubits,
            other.name,
            other.log_base,
        ):
            return False
        if isinstance(self.observable, torch.Tensor) or isinstance(
            other.observable, torch.Tensor
        ):
            return self.observable is other.observable
        return self.observable == other.observable

    def __hash__(self) -> int:
        operator = (
            id(self.observable)
            if isinstance(self.observable, torch.Tensor)
            else self.observable
        )
        return hash((self.kind, self.qubits, operator, self.name, self.log_base))

    @property
    def wires(self) -> tuple[int, ...]:
        """Deprecated alias for :attr:`qubits`."""

        warn_qubit_alias("wires", "qubits")
        return self.qubits


def expectation(
    observable: Observable | torch.Tensor, *, name: str | None = None
) -> OutputRequest:
    """Request a Pauli or full-space dense matrix expectation.

    Dense matrices use the existing CPU ``flagquantum.lindblad`` path. They
    follow the declared state-space basis and do not accept local selectors.
    Circuit and remote outputs refuse them before execution or submission.

    Examples:
        >>> import flagquantum as fq
        >>> output = fq.expectation(fq.X(0) @ fq.X(1), name="correlation")
        >>> output.name
        'correlation'
    """

    return OutputRequest("expectation", observable=observable, name=name)


def _selection_alias(
    qubits: Iterable[int] | int | Observable | None | Omitted,
    wires: Iterable[int] | int | Observable | None | Omitted,
) -> Iterable[int] | int | Observable | None:
    if not isinstance(wires, Omitted):
        if not isinstance(qubits, Omitted):
            raise TypeError("pass qubits or deprecated wires, not both")
        warn_qubit_alias("wires", "qubits", stacklevel=4)
        return wires
    return None if isinstance(qubits, Omitted) else qubits


def probabilities(
    qubits: Iterable[int] | int | None | Omitted = OMITTED,
    *,
    name: str | None = None,
    wires: Iterable[int] | int | None | Omitted = OMITTED,
) -> OutputRequest:
    """Request exact computational-basis probabilities on selected qubits."""
    selected = _selection_alias(qubits, wires)
    if isinstance(selected, Observable):
        raise TypeError("probabilities requires qubit indices, not an Observable")
    return OutputRequest(
        "probabilities", _optional_qubits(selected, kind="probabilities"), name=name
    )


def density_matrix(
    qubits: Iterable[int] | int | None | Omitted = OMITTED,
    *,
    name: str | None = None,
) -> OutputRequest:
    """Request the density matrix of selected qubits, tracing the rest out.

    Args:
        qubits: The qubits to keep, in the basis order the result should use.
            Omitting them keeps every qubit.
        name: An optional label for selecting this output from the result.

    The result is whatever the execution state is, mixed or pure: a noisy run
    returns the state its noise left behind, and readout confusion is refused
    rather than folded in, because readout acts on outcomes after the state.

    Examples:
        >>> import flagquantum as fq
        >>> circuit = fq.Circuit(2).h(0).cx(0, 1)
        >>> density = fq.run(circuit, outputs=fq.density_matrix(qubits=0))
        >>> density.density_matrix.shape
        torch.Size([1, 2, 2])
    """

    selected = _selection_alias(qubits, OMITTED)
    if isinstance(selected, Observable):
        raise TypeError("density_matrix requires qubit indices, not an Observable")
    return OutputRequest(
        "density_matrix", _optional_qubits(selected, kind="density_matrix"), name=name
    )


def vn_entropy(
    qubits: Iterable[int] | int,
    *,
    log_base: float | None = None,
    name: str | None = None,
) -> OutputRequest:
    """Request the von Neumann entropy of a subsystem, in nats by default.

    Args:
        qubits: The qubits whose entropy is requested, named in the basis order
            the reduced state would use. At least one is required, and the
            selection may be any subset rather than a contiguous run.
        log_base: The logarithm's base. ``None`` is the natural logarithm, so the
            answer is in nats; ``2`` gives bits.
        name: An optional label for selecting this output from the result.

    The entropy is a property of the state the run left behind, so every
    execution mode that holds a state answers with the same number: dense modes
    reduce and diagonalise, while matrix-product and tensor-network modes contract
    the named sites out of the state they already hold.

    Examples:
        >>> import flagquantum as fq
        >>> circuit = fq.Circuit(2).h(0).cx(0, 1)
        >>> result = fq.run(circuit, outputs=fq.vn_entropy(qubits=0))
        >>> round(float(result.vn_entropy), 6)
        0.693147
    """

    selected = _qubits(qubits, owner="vn_entropy output")
    return OutputRequest("vn_entropy", selected, name=name, log_base=log_base)


def samples(
    qubits: Iterable[int] | int | Observable | None | Omitted = OMITTED,
    *,
    name: str | None = None,
    wires: Iterable[int] | int | Observable | None | Omitted = OMITTED,
) -> OutputRequest:
    """Request computational- or Pauli-basis samples."""
    selected = _selection_alias(qubits, wires)
    if isinstance(selected, Observable):
        return OutputRequest("samples", observable=selected, name=name)
    return OutputRequest(
        "samples", _optional_qubits(selected, kind="samples"), name=name
    )


def counts(
    qubits: Iterable[int] | int | Observable | None | Omitted = OMITTED,
    *,
    name: str | None = None,
    wires: Iterable[int] | int | Observable | None | Omitted = OMITTED,
) -> OutputRequest:
    """Request computational- or Pauli-basis outcome counts."""
    selected = _selection_alias(qubits, wires)
    if isinstance(selected, Observable):
        return OutputRequest("counts", observable=selected, name=name)
    return OutputRequest("counts", _optional_qubits(selected, kind="counts"), name=name)


def _optional_qubits(
    qubits: Iterable[int] | int | None, *, kind: str
) -> tuple[int, ...]:
    if qubits is None:
        return ()
    return _qubits(qubits, owner=f"{kind} output")


def _sampled_pauli_term(observable: Observable) -> _PauliTerm:
    if len(observable.terms) != 1:
        raise ValueError("Pauli-basis sampling requires exactly one Pauli product")
    term = observable.terms[0]
    if term.coefficient != 1.0 or not term.factors:
        raise ValueError(
            "Pauli-basis sampling requires an unweighted, non-identity product"
        )
    return term


def lower_outputs(
    outputs: OutputRequest | Sequence[OutputRequest] | None,
    *,
    n_qubits: int,
    shots: int | None,
    seed: int | None = None,
) -> tuple[MeasurementNode, ...] | None:
    """Lower public output semantics to the existing Core IR representation."""

    if outputs is None:
        return None
    requests = (outputs,) if isinstance(outputs, OutputRequest) else tuple(outputs)
    if not requests or any(not isinstance(item, OutputRequest) for item in requests):
        raise TypeError("outputs must be an OutputRequest or a non-empty sequence")
    if any(isinstance(request.observable, torch.Tensor) for request in requests):
        raise CapabilityError(
            "Dense matrix expectations currently require the Lindblad path (flagquantum.lindblad); "
            "CircuitIR 1.0 cannot represent full-space matrix outputs"
        )
    names = tuple(request.name for request in requests if request.name is not None)
    if len(set(names)) != len(names):
        raise ValueError("output names must be unique")
    if shots is not None and not any(
        request.kind in {"samples", "counts"} for request in requests
    ):
        raise ValueError("shots requires fq.samples(...) or fq.counts(...) output")
    lowered: list[MeasurementNode] = []
    for output_index, request in enumerate(requests):
        metadata = {
            "fq_output_index": output_index,
            "fq_output_kind": request.kind,
            "fq_output_name": request.name,
        }
        if request.name is not None:
            metadata["name"] = request.name
        if request.kind == "expectation":
            assert isinstance(request.observable, Observable)
            term_count = len(request.observable.terms)
            for term_index, term in enumerate(request.observable.terms):
                axes = {
                    axis: tuple(qubit for qubit, value in term.factors if value == axis)
                    for axis in ("x", "y", "z")
                }
                term_qubits = tuple(qubit for qubit, _ in term.factors)
                lowered.append(
                    MeasurementNode(
                        "expectation_identity" if not term_qubits else "expectation_ps",
                        term_qubits or (0,),
                        metadata={
                            **metadata,
                            **axes,
                            "fq_output_term": term_index,
                            "fq_output_terms": term_count,
                            "fq_coefficient": term.coefficient,
                        },
                    )
                )
            continue
        if request.observable is not None:
            assert isinstance(request.observable, Observable)
            term = _sampled_pauli_term(request.observable)
            axes = {
                axis: tuple(qubit for qubit, value in term.factors if value == axis)
                for axis in ("x", "y", "z")
            }
            metadata.update(axes)
            metadata["max_marginal_wires"] = _DEFAULT_MAX_PAULI_SAMPLE_WIRES
            if seed is not None:
                # Pauli-basis sampling draws from the same sampler as the plain path, so
                # it carries the seed the same way; without it the runtime falls back to
                # an unseeded generator and a seeded call is not reproducible.
                metadata["seed"] = seed
            lowered.append(
                MeasurementNode(
                    "sample_ps" if request.kind == "samples" else "counts_ps",
                    tuple(qubit for qubit, _ in term.factors),
                    shots=shots,
                    metadata=metadata,
                )
            )
            continue
        qubits = request.qubits or tuple(range(n_qubits))
        request_shots = shots if request.kind in {"samples", "counts"} else None
        if request.kind in {"samples", "counts"} and request_shots is None:
            raise ValueError(f"{request.kind} output requires shots")
        if request.kind == "vn_entropy":
            # The base travels in metadata because the measurement node is the Core IR
            # type and adding a base field to it would put an entropy convention into
            # the representation every other measurement shares. For every other kind
            # the field is left absent rather than written as ``None``, so a node that
            # happens to carry metadata from elsewhere cannot be read as requesting a
            # base it never named.
            metadata["fq_log_base"] = request.log_base
        if seed is not None:
            metadata["seed"] = seed
        lowered.append(
            MeasurementNode(
                "sample" if request.kind == "samples" else request.kind,
                qubits,
                shots=request_shots,
                metadata=metadata,
            )
        )
    return tuple(lowered)


__all__ = (
    "OutputRequest",
    "expectation",
    "probabilities",
    "density_matrix",
    "vn_entropy",
    "samples",
    "counts",
    "lower_outputs",
)


# Preserve historical import identities for pickles and public introspection.
for _identity_export in (
    OutputRequest,
    expectation,
    _selection_alias,
    probabilities,
    density_matrix,
    vn_entropy,
    samples,
    counts,
    _optional_qubits,
    _sampled_pauli_term,
    lower_outputs,
):
    _identity_export.__module__ = "flagquantum.observables"
del _identity_export

# Bind the historical public namespace for class annotation introspection.

_public_namespace = _import_public_namespace("flagquantum.observables")
for _public_name in (
    "OutputRequest",
    "_optional_qubits",
    "_sampled_pauli_term",
    "_selection_alias",
    "counts",
    "density_matrix",
    "expectation",
    "lower_outputs",
    "probabilities",
    "samples",
    "vn_entropy",
):
    setattr(_public_namespace, _public_name, globals()[_public_name])
del _public_namespace, _public_name
