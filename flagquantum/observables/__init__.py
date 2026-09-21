"""Mathematical observables and requested execution outputs."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

from ..core._qubit_aliases import OMITTED, Omitted, warn_qubit_alias
from ..core.ir import MeasurementNode

_DEFAULT_MAX_PAULI_SAMPLE_WIRES = 8


@dataclass(frozen=True, slots=True)
class _PauliTerm:
    coefficient: float
    factors: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class Observable:
    """An immutable real linear combination of Pauli products.

    Examples:
        >>> import flagquantum as fq
        >>> observable = 0.5 * (fq.X(0) @ fq.X(1)) - fq.Z(0)
        >>> len(observable.terms)
        2
    """

    terms: tuple[_PauliTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("an observable requires at least one Pauli term")

    def __matmul__(self, other: object) -> "Observable":
        if not isinstance(other, Observable):
            return NotImplemented
        products: list[_PauliTerm] = []
        for left in self.terms:
            for right in other.terms:
                left_wires = {wire for wire, _ in left.factors}
                right_wires = {wire for wire, _ in right.factors}
                if left_wires & right_wires:
                    raise ValueError("Pauli tensor products require disjoint wires")
                factors = tuple(sorted((*left.factors, *right.factors)))
                products.append(
                    _PauliTerm(left.coefficient * right.coefficient, factors)
                )
        return Observable(tuple(products))

    def __add__(self, other: object) -> "Observable":
        if not isinstance(other, Observable):
            return NotImplemented
        return Observable((*self.terms, *other.terms))

    def __sub__(self, other: object) -> "Observable":
        if not isinstance(other, Observable):
            return NotImplemented
        return self + (-other)

    def __neg__(self) -> "Observable":
        return Observable(
            tuple(_PauliTerm(-term.coefficient, term.factors) for term in self.terms)
        )

    def __mul__(self, scalar: object) -> "Observable":
        value = _real_scalar(scalar)
        if value is None:
            return NotImplemented
        return Observable(
            tuple(
                _PauliTerm(value * term.coefficient, term.factors)
                for term in self.terms
            )
        )

    def __rmul__(self, scalar: object) -> "Observable":
        return self * scalar


@dataclass(frozen=True, slots=True)
class OutputRequest:
    """A backend-neutral description of one requested execution output."""

    kind: str
    wires: tuple[int, ...] = ()
    observable: Observable | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"counts", "expectation", "probabilities", "samples"}:
            raise ValueError(f"unsupported output kind {self.kind!r}")
        object.__setattr__(
            self, "wires", _wires(self.wires, owner=f"{self.kind} output")
        )
        if self.name is not None and (
            not isinstance(self.name, str) or not self.name.strip()
        ):
            raise ValueError("output name must be a non-empty string")
        if self.kind == "expectation" and not isinstance(self.observable, Observable):
            raise TypeError("expectation output requires an Observable")
        if self.kind in {"probabilities"} and self.observable is not None:
            raise TypeError(f"{self.kind} output does not accept an observable")
        if self.kind in {"samples", "counts"} and self.observable is not None:
            _sampled_pauli_term(self.observable)


def _real_scalar(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("observable coefficients must be finite")
    return result


def _wire(owner: str, value: Any) -> int:
    """Read one wire label without quietly rewriting or iterating the caller's value.

    A label is an integer, read through the interpreter's own ``__index__`` protocol,
    which admits NumPy integers and zero-dimensional torch integer tensors while refusing
    floats and strings.  ``bool`` is refused although it satisfies ``operator.index``, for
    the reason the package's counts give: it is a flag rather than a label.  ``TypeError``
    reports a wrong Python type, which is what the errors-module boundary reserves for it;
    a negative label is a magnitude and keeps its own error below.
    """

    if isinstance(value, bool):
        raise TypeError(f"{owner} wire must be an integer, got {value!r}")
    try:
        label = operator.index(value)
    except TypeError:
        raise TypeError(f"{owner} wire must be an integer, got {value!r}") from None
    if label < 0:
        raise ValueError(f"{owner} wire must be a non-negative integer")
    return label


def _wires(wires: Iterable[int] | int, *, owner: str) -> tuple[int, ...]:
    """Read a selection of wire labels, one label at a time.

    A single label and a sequence of labels are both accepted, and the order matters: a
    label is tried first, so a bare zero-dimensional tensor is read as the label it
    denotes, and a value that cannot be iterated at all -- a float is the case that
    occurs -- is reported as the bad label it is instead of as ``'float' object is not
    iterable``.  ``str`` is excluded from the sequence form, because ``"01"`` is a
    mistyped label rather than two labels.
    """

    normalized: tuple[int, ...]
    if isinstance(wires, (str, bytes)):
        normalized = (_wire(owner, wires),)
    else:
        try:
            normalized = (_wire(owner, wires),)
        except TypeError as scalar_error:
            if not hasattr(wires, "__iter__"):
                raise
            try:
                items = tuple(cast("Iterable[int]", wires))
            except TypeError:
                # An iterable that refuses to be iterated as a sequence -- a
                # zero-dimensional tensor is the case that occurs -- is a bad label, not
                # a bad sequence, so the scalar refusal is the true one.
                raise scalar_error from None
            normalized = tuple(_wire(owner, wire) for wire in items)
    if len(set(normalized)) != len(normalized):
        raise ValueError("output wires must be unique")
    return normalized


def _pauli(axis: str, wire: int) -> Observable:
    return Observable((_PauliTerm(1.0, ((_wire("observable", wire), axis),)),))


def I(wire: int | None = None) -> Observable:
    """Return the identity observable.

    ``wire`` may be supplied for readable Hamiltonian notation but does not
    affect the mathematical value.
    """

    if wire is not None:
        _wire("observable", wire)
    return Observable((_PauliTerm(1.0, ()),))


def X(wire: int) -> Observable:
    """Return the Pauli-X observable on ``wire``."""

    return _pauli("x", wire)


def Y(wire: int) -> Observable:
    """Return the Pauli-Y observable on ``wire``."""

    return _pauli("y", wire)


def Z(wire: int) -> Observable:
    """Return the Pauli-Z observable on ``wire``."""

    return _pauli("z", wire)


def expectation(observable: Observable, *, name: str | None = None) -> OutputRequest:
    """Request the expectation value of an observable.

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
        "probabilities", _optional_wires(selected, kind="probabilities"), name=name
    )


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
        "samples", _optional_wires(selected, kind="samples"), name=name
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
    return OutputRequest("counts", _optional_wires(selected, kind="counts"), name=name)


def _optional_wires(wires: Iterable[int] | int | None, *, kind: str) -> tuple[int, ...]:
    if wires is None:
        return ()
    return _wires(wires, owner=f"{kind} output")


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
    n_wires: int,
    shots: int | None,
    seed: int | None = None,
) -> tuple[MeasurementNode, ...] | None:
    """Lower public output semantics to the existing Core IR representation."""

    if outputs is None:
        return None
    requests = (outputs,) if isinstance(outputs, OutputRequest) else tuple(outputs)
    if not requests or any(not isinstance(item, OutputRequest) for item in requests):
        raise TypeError("outputs must be an OutputRequest or a non-empty sequence")
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
            assert request.observable is not None
            term_count = len(request.observable.terms)
            for term_index, term in enumerate(request.observable.terms):
                axes = {
                    axis: tuple(wire for wire, value in term.factors if value == axis)
                    for axis in ("x", "y", "z")
                }
                term_wires = tuple(wire for wire, _ in term.factors)
                lowered.append(
                    MeasurementNode(
                        "expectation_identity" if not term_wires else "expectation_ps",
                        term_wires or (0,),
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
            term = _sampled_pauli_term(request.observable)
            axes = {
                axis: tuple(wire for wire, value in term.factors if value == axis)
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
                    tuple(wire for wire, _ in term.factors),
                    shots=shots,
                    metadata=metadata,
                )
            )
            continue
        wires = request.wires or tuple(range(n_wires))
        request_shots = shots if request.kind in {"samples", "counts"} else None
        if request.kind in {"samples", "counts"} and request_shots is None:
            raise ValueError(f"{request.kind} output requires shots")
        if seed is not None:
            metadata["seed"] = seed
        lowered.append(
            MeasurementNode(
                "sample" if request.kind == "samples" else request.kind,
                wires,
                shots=request_shots,
                metadata=metadata,
            )
        )
    return tuple(lowered)


__all__ = (
    "I",
    "Observable",
    "OutputRequest",
    "X",
    "Y",
    "Z",
    "counts",
    "expectation",
    "probabilities",
    "samples",
)
