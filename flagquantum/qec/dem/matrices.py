"""The code-capacity description: a CSS code's matrices and the faults they imply.

:mod:`flagquantum.qec.dem_construction` derives the mechanisms a detector error
model states from the circuit an experiment runs. This module derives them from a
**matrix** description instead, and owns the mechanisms those matrices imply.
The code-owned record, validation and extraction live in
:mod:`flagquantum.qec.codes.matrices` and remain re-exported here for existing
callers and checkpoints. The two descriptions reach the same model shape by
different routes, and
neither is the authority for the other, because the experiment each describes is
not the same one.

A matrix description is read. :func:`_code_matrix_entries` takes the
parity-check matrix a code is defined by and the matrix of its logical operators,
and derives the signatures combinatorially; nothing is lowered or executed. Its
detector geometry is consequently the code-capacity one:
``num_rounds * num_checks`` detectors, each round compared against the round
before it, with no terminal data readout. A memory circuit measures the data
qubits and so ends in terminal detectors, which is why the circuit route's
geometry differs; the difference belongs to the description, not to the model,
and a test pins each geometry to its own route.

Both routes carry the same family of faults. On the matrix route the three data
faults are the three Pauli errors: an X fault reaches the Z-type detectors and
the Z-type logical operators, a Z fault reaches the X-type detectors and the
X-type logical operators, and a Y fault reaches both. Each data fault in round
``r`` shows up in the detector band of round ``r`` and in the band after it, and
a measurement fault shows up in its own check's detector and in the next round's,
which is the code-capacity experiment stated as detector differences.

The two other grammars the circuit route reads -- a fault forced into the
experiment's own source, and a fault spliced after a named gate -- have no
counterpart here, because a matrix states no location inside a program.
"""

from __future__ import annotations

from numbers import Integral

import torch

from flagquantum.qec.codes import StabilizerCode as StabilizerCode
from flagquantum.qec.codes._compat import _retain_module
from flagquantum.qec.codes.matrices import CssCodeMatrices as CssCodeMatrices
from flagquantum.qec.codes.matrices import _binary_matrix as _binary_matrix
from flagquantum.qec.codes.matrices import css_code_matrices as css_code_matrices
from flagquantum.qec.noise import PhenomenologicalNoise

__all__ = ("CssCodeMatrices", "css_code_matrices")

# One mechanism: its rate, the detectors it flips, and the observables it flips.
Entry = tuple[float, tuple[int, ...], tuple[int, ...]]


def _column_supports(
    matrix: torch.Tensor, num_qubits: int
) -> tuple[tuple[int, ...], ...]:
    """Return the rows each column of ``matrix`` selects, in ascending order."""

    selected = matrix.to(torch.bool)
    return tuple(
        tuple(int(row) for row in selected[:, qubit].nonzero().flatten())
        for qubit in range(num_qubits)
    )


def _code_matrix_entries(
    matrices: CssCodeMatrices,
    noise: PhenomenologicalNoise,
    num_rounds: int = 1,
) -> tuple[int, int, tuple[Entry, ...]]:
    """Return the model shape and every mechanism a code's matrices imply.

    The matrices are read, not simulated. Each has one column per data qubit, and
    a fault is read as the columns it selects: ``hz[k, q]`` is one when an X fault
    on qubit ``q`` flips Z-type check ``k``, ``hx[k, q]`` is one when a Z fault
    flips X-type check ``k``, and the two logical matrices state the same relation
    for the observable rows.

    A round's detector band holds one row per check, Z-type checks first, and the
    band is compared against the band before it, so a data fault lands in its own
    round's band and in the band after it and the final round has no band after
    it. A measurement fault lands in its own check's row and in the next round's,
    and flips no observable, because the logical measurement is taken once at the
    end of the experiment.

    The model describes the code-capacity experiment: the data begins in the
    code's ``+1`` eigenspace of every check, so every round-zero check is
    deterministic and there is no preparation-fault family. A code whose
    preparation is not a code state is the circuit route's business, not this
    one's, and it is there that a check whose round-zero outcome is a coin toss
    has to be handled.

    Every data qubit the noise reaches contributes one mechanism per round per
    fault family, and every check the noise reaches contributes one measurement
    mechanism per round, where "reaches" is the element's own effective rate: a
    per-qubit or per-check vector states one rate per location, and a location
    whose rate is zero is not enumerated at all. An entry that flips nothing at
    all -- a qubit outside every check and outside every logical operator -- is
    not a mechanism a model can store, and an entry whose signature another entry
    already has is one fault to a decoder. Both are resolved where every
    construction route resolves them, in ``DetectorErrorModel._merge_mechanisms``,
    by the independent-parity rule.

    Raises:
        TypeError: If ``noise`` is not a
            :class:`~flagquantum.qec.PhenomenologicalNoise`.
        ValueError: If ``num_rounds`` is below one, if the matrices declare no
            check -- a model needs at least one detector -- or if a per-element
            rate vector the noise states does not name every data qubit or every
            check the matrices declare.
    """

    if isinstance(num_rounds, bool) or not isinstance(num_rounds, Integral):
        raise TypeError("num_rounds must be an integer")
    if num_rounds < 1:
        raise ValueError("num_rounds must be at least one")
    if not isinstance(noise, PhenomenologicalNoise):
        raise TypeError("noise must be a PhenomenologicalNoise")
    if not isinstance(matrices, CssCodeMatrices):
        raise TypeError("matrices must be a CssCodeMatrices")
    rounds = int(num_rounds)

    # ``CssCodeMatrices`` has already refused a disagreement about the number of
    # columns, so one column count describes all four matrices here.
    num_qubits = matrices.num_qubits
    num_z_checks = matrices.num_z_checks
    num_x_checks = matrices.num_x_checks
    num_checks = num_z_checks + num_x_checks
    if num_checks == 0:
        raise ValueError(
            f"the matrices declare {num_qubits} data qubits but no check, so the "
            "model would have no detector to describe"
        )
    num_z_logicals = matrices.num_z_logicals

    # A vector's length is stated against the code, and the matrices are the
    # code here: one column per data qubit and one row per check, Z-type checks
    # first, which is the order `measurement_flip_per_check` is indexed in.
    data_flip_rates = noise.data_flip_rates(num_qubits=num_qubits)
    phase_flip_rates = noise.phase_flip_rates(num_qubits=num_qubits)
    both_flip_rates = noise.both_flip_rates(num_qubits=num_qubits)
    measurement_flip_rates = noise.measurement_flip_rates(num_checks=num_checks)

    hz_columns = _column_supports(matrices.hz, num_qubits)
    hx_columns = _column_supports(matrices.hx, num_qubits)
    lz_columns = _column_supports(matrices.lz, num_qubits)
    lx_columns = _column_supports(matrices.lx, num_qubits)
    entries: list[Entry] = []
    for round_index in range(rounds):
        band = round_index * num_checks
        following = band + num_checks
        has_next = round_index + 1 < rounds
        for qubit in range(num_qubits):
            data_rate = data_flip_rates[qubit]
            phase_rate = phase_flip_rates[qubit]
            both_rate = both_flip_rates[qubit]
            if not (data_rate or phase_rate or both_rate):
                # A location that cannot fire is not a mechanism, so it is not
                # enumerated and does not become a zero-rate column.
                continue
            # An X fault reaches the Z-type checks, an X-type check being blind to
            # it, and the same for the Z fault and the X-type checks. A Y fault is
            # the two of them at once and therefore reaches both blocks.
            z_block = [band + row for row in hz_columns[qubit]]
            x_block = [band + num_z_checks + row for row in hx_columns[qubit]]
            if has_next:
                z_block.extend(following + row for row in hz_columns[qubit])
                x_block.extend(
                    following + num_z_checks + row for row in hx_columns[qubit]
                )
            z_logicals = lz_columns[qubit]
            x_logicals = tuple(num_z_logicals + row for row in lx_columns[qubit])
            if data_rate:
                entries.append((data_rate, tuple(z_block), tuple(z_logicals)))
            if phase_rate:
                entries.append((phase_rate, tuple(x_block), tuple(x_logicals)))
            if both_rate:
                entries.append(
                    (
                        both_rate,
                        tuple(z_block) + tuple(x_block),
                        tuple(z_logicals) + tuple(x_logicals),
                    )
                )
        for check in range(num_checks):
            measurement_rate = measurement_flip_rates[check]
            if not measurement_rate:
                continue
            detectors = [band + check]
            if has_next:
                detectors.append(following + check)
            entries.append((measurement_rate, tuple(detectors), ()))
    return rounds * num_checks, matrices.num_observables, tuple(entries)


_retain_module("flagquantum.qec.dem_matrices", _column_supports, _code_matrix_entries)
