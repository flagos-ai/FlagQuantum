"""Immutable probe designs for identifiable local Twin parameter families."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from ..core.ir import CircuitIR, ensure_circuit_ir

_PROBE_BANK_SCHEMA = "flagquantum.twin_probe_bank.v2"
_LOCAL_FAMILY = "local_pauli_readout"

TwinProbeBankStatus = Literal["identifiable", "underdetermined"]


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: str, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _target(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("target must be a string in provider:backend form")
    provider, separator, backend = value.partition(":")
    provider = provider.strip().lower()
    backend = backend.strip()
    if separator != ":" or not provider or not backend or ":" in backend:
        raise ValueError("target must use provider:backend form")
    return f"{provider}:{backend}"


def _qubits(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("physical_qubits must be a sequence of integers")
    normalized: list[int] = []
    for value in tuple(values):
        if isinstance(value, bool):
            raise TypeError("physical_qubits must contain integers, not bools")
        try:
            qubit = operator.index(value)
        except TypeError:
            raise TypeError("physical_qubits must contain integers") from None
        normalized.append(int(qubit))
    result = tuple(normalized)
    if (
        not result
        or len(result) != len(set(result))
        or any(item < 0 for item in result)
    ):
        raise ValueError("physical_qubits must be non-empty, unique, and non-negative")
    return result


def _matrix_rank(matrix: Sequence[Sequence[int]]) -> int:
    if not matrix:
        return 0
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("design_matrix rows must have one value per parameter")
    reduced = [[Fraction(value) for value in row] for row in matrix]
    rank = 0
    for column in range(width):
        pivot = next(
            (row for row in range(rank, len(reduced)) if reduced[row][column]),
            None,
        )
        if pivot is None:
            continue
        reduced[rank], reduced[pivot] = reduced[pivot], reduced[rank]
        scale = reduced[rank][column]
        reduced[rank] = [value / scale for value in reduced[rank]]
        for row in range(len(reduced)):
            if row == rank or not reduced[row][column]:
                continue
            factor = reduced[row][column]
            reduced[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(reduced[row], reduced[rank], strict=True)
            ]
        rank += 1
        if rank == len(reduced):
            break
    return rank


def _identifiable_parameters(
    matrix: tuple[tuple[int, ...], ...], parameter_names: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    rank = _matrix_rank(matrix)
    identifiable: list[str] = []
    unidentified: list[str] = []
    for column, name in enumerate(parameter_names):
        without_column = tuple(
            tuple(value for index, value in enumerate(row) if index != column)
            for row in matrix
        )
        destination = (
            identifiable if _matrix_rank(without_column) < rank else unidentified
        )
        destination.append(name)
    return tuple(identifiable), tuple(unidentified)


@dataclass(frozen=True)
class TwinProbeBank:
    """A frozen local-error design whose identifiability is rank-qualified.

    The integer design matrix describes a declared linear local-error burden
    model over per-qubit marginals and per-coupler correlations. It is not a
    fitted noise model, a hardware result, or execution authority.
    """

    target: str
    physical_qubits: tuple[int, ...]
    directed_couplers: tuple[tuple[int, int], ...]
    probe_circuit_identities: tuple[str, ...]
    parameter_family: str
    parameter_names: tuple[str, ...]
    probe_parameter_names: tuple[tuple[str, tuple[str, ...]], ...]
    design_row_labels: tuple[str, ...]
    ideal_row_probabilities: tuple[tuple[float, ...], ...]
    design_matrix: tuple[tuple[int, ...], ...]
    matrix_rank: int
    identifiable_parameters: tuple[str, ...]
    unidentified_parameters: tuple[str, ...]
    identifiability_status: TwinProbeBankStatus
    schema: str = _PROBE_BANK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _PROBE_BANK_SCHEMA:
            raise ValueError("unsupported Twin probe-bank schema")
        target = _target(self.target)
        mapping = _qubits(self.physical_qubits)
        raw_couplers = tuple(tuple(edge) for edge in self.directed_couplers)
        physical = set(mapping)
        if len(raw_couplers) != len(set(raw_couplers)) or any(
            len(edge) != 2
            or type(edge[0]) is not int
            or type(edge[1]) is not int
            or edge[0] == edge[1]
            or edge[0] not in physical
            or edge[1] not in physical
            for edge in raw_couplers
        ):
            raise ValueError(
                "directed_couplers must be unique directed pairs on physical_qubits"
            )
        couplers: tuple[tuple[int, int], ...] = tuple(
            (edge[0], edge[1]) for edge in raw_couplers
        )
        if couplers != tuple(sorted(couplers)):
            raise ValueError("directed_couplers must use canonical sorted order")
        probes = tuple(self.probe_circuit_identities)
        if not probes or len(probes) != len(set(probes)):
            raise ValueError("probe circuit identities must be non-empty and unique")
        for identity in probes:
            _digest(identity, "probe_circuit_identity")
        if probes != tuple(sorted(probes)):
            raise ValueError("probe circuit identities must use canonical sorted order")
        if self.parameter_family != _LOCAL_FAMILY:
            raise ValueError("unsupported Twin probe-bank parameter family")
        parameters = tuple(self.parameter_names)
        if not parameters or len(parameters) != len(set(parameters)):
            raise ValueError("parameter_names must be non-empty and unique")
        if parameters != _parameter_names(mapping, couplers):
            raise ValueError(
                "parameter_names must match the declared mapping and couplers"
            )
        supports = tuple(
            (identity, tuple(names)) for identity, names in self.probe_parameter_names
        )
        if tuple(identity for identity, _ in supports) != probes:
            raise ValueError("probe_parameter_names must align with probe identities")
        if any(
            len(names) != len(set(names))
            or any(name not in parameters for name in names)
            for _, names in supports
        ):
            raise ValueError("probe parameter names must be unique bank parameters")
        labels = tuple(self.design_row_labels)
        ideal_rows = tuple(
            tuple(float(value) for value in row) for row in self.ideal_row_probabilities
        )
        matrix = tuple(tuple(row) for row in self.design_matrix)
        if not labels or len(labels) != len(matrix) or len(labels) != len(set(labels)):
            raise ValueError("design row labels must uniquely align with matrix rows")
        if len(ideal_rows) != len(labels):
            raise ValueError("ideal_row_probabilities must align with every design row")
        for label, probabilities in zip(labels, ideal_rows, strict=True):
            expected_width = 4 if ":zz[" in label else 2
            if (
                len(probabilities) != expected_width
                or any(
                    not math.isfinite(value) or value < 0.0 for value in probabilities
                )
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9)
            ):
                raise ValueError(
                    "ideal row probabilities must be normalized Z or ZZ distributions"
                )
        if any(type(value) is not int or value < 0 for row in matrix for value in row):
            raise ValueError("design_matrix must contain non-negative integers")
        rows_per_probe = len(mapping) + len(couplers)
        if len(matrix) != len(probes) * rows_per_probe:
            raise ValueError(
                "design_matrix must contain every declared marginal and correlation row"
            )
        for probe_index, (identity, names) in enumerate(supports):
            start = probe_index * rows_per_probe
            stop = start + rows_per_probe
            if any(
                not label.startswith(f"{identity}:") for label in labels[start:stop]
            ):
                raise ValueError("design row labels must remain bound to each probe")
            expected_names = tuple(
                name
                for column, name in enumerate(parameters)
                if any(row[column] for row in matrix[start:stop])
            )
            if names != expected_names:
                raise ValueError(
                    "probe_parameter_names do not match the frozen design matrix"
                )
        rank = _matrix_rank(matrix)
        if self.matrix_rank != rank:
            raise ValueError("matrix_rank does not match the frozen design matrix")
        identifiable, unidentified = _identifiable_parameters(matrix, parameters)
        if self.identifiable_parameters != identifiable:
            raise ValueError(
                "identifiable_parameters do not match the frozen design matrix"
            )
        if self.unidentified_parameters != unidentified:
            raise ValueError(
                "unidentified_parameters do not match the frozen design matrix"
            )
        expected_status = (
            "identifiable" if rank == len(parameters) else "underdetermined"
        )
        if self.identifiability_status != expected_status:
            raise ValueError("identifiability_status does not match the matrix rank")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "physical_qubits", mapping)
        object.__setattr__(self, "directed_couplers", couplers)
        object.__setattr__(self, "probe_circuit_identities", probes)
        object.__setattr__(self, "parameter_names", parameters)
        object.__setattr__(self, "probe_parameter_names", supports)
        object.__setattr__(self, "design_row_labels", labels)
        object.__setattr__(self, "ideal_row_probabilities", ideal_rows)
        object.__setattr__(self, "design_matrix", matrix)

    @property
    def identity(self) -> str:
        """Return the deterministic identity of the complete frozen design."""

        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible representation of the design."""

        return {
            "schema": self.schema,
            "target": self.target,
            "physical_qubits": list(self.physical_qubits),
            "directed_couplers": [list(edge) for edge in self.directed_couplers],
            "probe_circuit_identities": list(self.probe_circuit_identities),
            "parameter_family": self.parameter_family,
            "parameter_names": list(self.parameter_names),
            "probe_parameter_names": [
                {"circuit_identity": identity, "parameter_names": list(names)}
                for identity, names in self.probe_parameter_names
            ],
            "design_row_labels": list(self.design_row_labels),
            "ideal_row_probabilities": [
                list(row) for row in self.ideal_row_probabilities
            ],
            "design_matrix": [list(row) for row in self.design_matrix],
            "matrix_rank": self.matrix_rank,
            "identifiable_parameters": list(self.identifiable_parameters),
            "unidentified_parameters": list(self.unidentified_parameters),
            "identifiability_status": self.identifiability_status,
        }


@dataclass(frozen=True)
class _ProbeDesign:
    ir: CircuitIR
    ideal_probabilities: tuple[float, ...]
    one_qubit_counts: Mapping[int, int]
    two_qubit_counts: Mapping[tuple[int, int], int]


def _probe_design(ir: CircuitIR, mapping: tuple[int, ...]) -> _ProbeDesign:
    if ir.n_wires != len(mapping):
        raise ValueError("every probe must map every physical qubit in order")
    if ir.observables or ir.measurements:
        raise ValueError(
            "probe circuits must use the implicit computational-basis readout"
        )
    one_qubit_counts = dict.fromkeys(mapping, 0)
    two_qubit_counts: dict[tuple[int, int], int] = {}
    for instruction in ir.instructions:
        if instruction.metadata.get("is_channel") or instruction.metadata.get(
            "is_dynamic"
        ):
            raise ValueError("probe circuits must contain static unitary operations")
        if len(instruction.wires) == 1:
            physical = mapping[instruction.wires[0]]
            one_qubit_counts[physical] += 1
        elif len(instruction.wires) == 2:
            edge = (
                mapping[instruction.wires[0]],
                mapping[instruction.wires[1]],
            )
            two_qubit_counts[edge] = two_qubit_counts.get(edge, 0) + 1
        else:
            raise ValueError(
                "local_pauli_readout probes support only one- and two-qubit operations"
            )
    from ..circuit import Circuit

    ideal_tensor = Circuit.from_ir(ir).probabilities().detach().cpu().reshape(-1)
    ideal = tuple(float(value) for value in ideal_tensor)
    total = sum(ideal)
    if (
        len(ideal) != 2**ir.n_wires
        or any(not math.isfinite(value) or value < 0.0 for value in ideal)
        or not math.isclose(total, 1.0, abs_tol=1e-6)
    ):
        raise ValueError("probe ideal probabilities must form one normalized batch")
    ideal = tuple(value / total for value in ideal)
    return _ProbeDesign(
        ir=ir,
        ideal_probabilities=ideal,
        one_qubit_counts=one_qubit_counts,
        two_qubit_counts=two_qubit_counts,
    )


def _parameter_names(
    mapping: tuple[int, ...], couplers: tuple[tuple[int, int], ...]
) -> tuple[str, ...]:
    return (
        *(f"readout_bit_flip[Q{qubit}]" for qubit in mapping),
        *(f"single_qubit_pauli[Q{qubit}]" for qubit in mapping),
        *(f"two_qubit_pauli[Q{source}->Q{target}]" for source, target in couplers),
    )


def _design_rows(
    probe: _ProbeDesign,
    mapping: tuple[int, ...],
    couplers: tuple[tuple[int, int], ...],
    parameter_names: tuple[str, ...],
) -> tuple[tuple[str, tuple[int, ...], tuple[float, ...]], ...]:
    indices = {name: index for index, name in enumerate(parameter_names)}
    rows: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
    identity = probe.ir.content_hash
    logical_indices = {physical: logical for logical, physical in enumerate(mapping)}

    def marginal(logical_qubits: tuple[int, ...]) -> tuple[float, ...]:
        values = [0.0] * (2 ** len(logical_qubits))
        for basis_index, probability in enumerate(probe.ideal_probabilities):
            bitstring = format(basis_index, f"0{probe.ir.n_wires}b")
            outcome = "".join(bitstring[index] for index in logical_qubits)
            values[int(outcome, 2)] += probability
        return tuple(values)

    for qubit in mapping:
        row = [0] * len(parameter_names)
        row[indices[f"readout_bit_flip[Q{qubit}]"]] = 1
        row[indices[f"single_qubit_pauli[Q{qubit}]"]] = probe.one_qubit_counts[qubit]
        for edge, count in probe.two_qubit_counts.items():
            if qubit in edge:
                row[indices[f"two_qubit_pauli[Q{edge[0]}->Q{edge[1]}]"]] = count
        rows.append(
            (
                f"{identity}:z[Q{qubit}]",
                tuple(row),
                marginal((logical_indices[qubit],)),
            )
        )
    for source, target in couplers:
        row = [0] * len(parameter_names)
        row[indices[f"readout_bit_flip[Q{source}]"]] = 1
        row[indices[f"readout_bit_flip[Q{target}]"]] = 1
        row[indices[f"single_qubit_pauli[Q{source}]"]] = probe.one_qubit_counts[source]
        row[indices[f"single_qubit_pauli[Q{target}]"]] = probe.one_qubit_counts[target]
        row[indices[f"two_qubit_pauli[Q{source}->Q{target}]"]] = (
            probe.two_qubit_counts.get((source, target), 0)
        )
        rows.append(
            (
                f"{identity}:zz[Q{source}->Q{target}]",
                tuple(row),
                marginal((logical_indices[source], logical_indices[target])),
            )
        )
    return tuple(rows)


def prepare_probe_bank(
    *,
    target: str,
    physical_qubits: Sequence[int],
    probes: Sequence[Any],
    parameter_family: str,
) -> TwinProbeBank:
    """Freeze an exact-rank design for a restricted local Twin model.

    Probe ordering does not affect the resulting identity. The function never
    submits hardware work, fits parameters, or mutates a Twin.

    Examples:
        bank = fq.twin.prepare_probe_bank(
            target="quafu:Shenglian",
            physical_qubits=(20, 27),
            probes=(
                fq.Circuit(2).cx(0, 1),
                fq.Circuit(2).h(0).cx(0, 1),
                fq.Circuit(2).h(1).cx(0, 1),
            ),
            parameter_family="local_pauli_readout",
        )
        print(bank.identifiability_status)
        print(bank.unidentified_parameters)
    """

    canonical_target = _target(target)
    mapping = _qubits(physical_qubits)
    if parameter_family != _LOCAL_FAMILY:
        raise ValueError(
            "parameter_family must be 'local_pauli_readout' in this release"
        )
    if isinstance(probes, (str, bytes)):
        raise TypeError("probes must be a sequence of circuits")
    designs = tuple(
        sorted(
            (
                _probe_design(ensure_circuit_ir(probe), mapping)
                for probe in tuple(probes)
            ),
            key=lambda item: item.ir.content_hash,
        )
    )
    identities = tuple(item.ir.content_hash for item in designs)
    if not identities:
        raise ValueError("at least one probe circuit is required")
    if len(identities) != len(set(identities)):
        raise ValueError("probe circuits must have unique identities")
    couplers = tuple(
        sorted({edge for design in designs for edge in design.two_qubit_counts})
    )
    parameters = _parameter_names(mapping, couplers)
    labelled_rows = tuple(
        row
        for design in designs
        for row in _design_rows(design, mapping, couplers, parameters)
    )
    matrix = tuple(row for _, row, _ in labelled_rows)
    rank = _matrix_rank(matrix)
    identifiable, unidentified = _identifiable_parameters(matrix, parameters)
    supports: list[tuple[str, tuple[str, ...]]] = []
    rows_per_probe = len(mapping) + len(couplers)
    for probe_index, identity in enumerate(identities):
        start = probe_index * rows_per_probe
        probe_rows = matrix[start : start + rows_per_probe]
        names = tuple(
            name
            for column, name in enumerate(parameters)
            if any(row[column] for row in probe_rows)
        )
        supports.append((identity, names))
    return TwinProbeBank(
        target=canonical_target,
        physical_qubits=mapping,
        directed_couplers=couplers,
        probe_circuit_identities=identities,
        parameter_family=parameter_family,
        parameter_names=parameters,
        probe_parameter_names=tuple(supports),
        design_row_labels=tuple(label for label, _, _ in labelled_rows),
        ideal_row_probabilities=tuple(ideal for _, _, ideal in labelled_rows),
        design_matrix=matrix,
        matrix_rank=rank,
        identifiable_parameters=identifiable,
        unidentified_parameters=unidentified,
        identifiability_status=(
            "identifiable" if rank == len(parameters) else "underdetermined"
        ),
    )


__all__ = (
    "prepare_probe_bank",
    "TwinProbeBank",
    "TwinProbeBankStatus",
)
