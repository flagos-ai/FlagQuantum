"""The code-capacity description: a CSS code's matrices and the faults they imply.

:mod:`flagquantum.qec.dem_construction` derives the mechanisms a detector error
model states from the circuit an experiment runs. This module derives them from a
**matrix** description instead, and it owns both halves of that description: the
record a code's four matrices are held in, and the mechanisms those matrices
imply. The two descriptions reach the same model shape by different routes, and
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

from dataclasses import dataclass
from numbers import Integral

import torch

from .codes import StabilizerCode
from .noise import PhenomenologicalNoise

__all__ = ("CssCodeMatrices", "css_code_matrices")

# One mechanism: its rate, the detectors it flips, and the observables it flips.
Entry = tuple[float, tuple[int, ...], tuple[int, ...]]


def _binary_matrix(matrix: object, *, name: str) -> torch.Tensor:
    """Return ``matrix`` as a two-dimensional integer tensor of zeros and ones.

    A floating-point matrix is refused rather than compared against a tolerance:
    a caller who reached for one is describing rates, not a support, and rounding
    it here would decide which checks a fault triggers.
    """

    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if matrix.dim() != 2:
        raise ValueError(
            f"{name} must be two-dimensional -- one row per check and one column "
            f"per data qubit -- but has {matrix.dim()} dimension(s)"
        )
    if matrix.is_floating_point() or matrix.is_complex():
        raise TypeError(
            f"{name} must be a binary matrix, so it cannot carry a "
            f"{matrix.dtype} element"
        )
    values = matrix.to(torch.int64)
    if values.numel() and bool(((values != 0) & (values != 1)).any()):
        raise ValueError(f"{name} must contain only zeros and ones")
    return values


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


@dataclass(frozen=True)
class CssCodeMatrices:
    """The four sparse binary matrices a Calderbank-Shor-Steane code is read by.

    Column ``q`` of every matrix is the ``q``-th data qubit, so the four are
    indexed alike and the record refuses a set that disagrees about how many data
    qubits the code has. ``hz`` holds the Z-type checks and ``lz`` the Z-type
    logical operators, which are the two matrices an X fault reaches -- an X fault
    on qubit ``q`` flips exactly the Z-type checks whose support holds ``q`` and
    exactly the Z-type logical operators whose support holds it. ``hx`` and ``lx``
    are the same pair for a Z fault, which the X-type checks are sensitive to and
    the Z-type checks are blind to. All four are carried because a code's two
    fault families are two different experiments over one set of data qubits, and
    a record that carried only one pair would be silently modelling half a CSS
    code.

    ``hx``, ``lz`` and ``lx`` may be omitted, and a block with no row is how a code
    states that it has no check or no logical operator of that type -- an X-type
    check measured on a code that has none is an empty block rather than a
    missing record. A block that has rows but no column is malformed rather than
    empty, because it states a check over no qubit.

    Raises:
        TypeError: If a block is not an integer tensor.
        ValueError: If a block is not two-dimensional, carries an element that is
            neither zero nor one, has rows but no column, or disagrees with
            another block about how many data qubits the code has.

    Examples:
        >>> import torch
        >>> from flagquantum.qec import CssCodeMatrices
        >>> hz = torch.tensor([[1, 1, 0], [0, 1, 1]])
        >>> matrices = CssCodeMatrices(hz=hz, lz=torch.tensor([[1, 1, 1]]))
        >>> matrices.num_checks, matrices.num_observables
        (2, 1)
    """

    hz: torch.Tensor
    hx: torch.Tensor | None = None
    lz: torch.Tensor | None = None
    lx: torch.Tensor | None = None

    def __post_init__(self) -> None:
        blocks: dict[str, torch.Tensor | None] = {}
        for name in ("hz", "hx", "lz", "lx"):
            block = getattr(self, name)
            blocks[name] = None if block is None else _binary_matrix(block, name=name)
        for name, block in blocks.items():
            if block is not None and int(block.shape[0]) and not int(block.shape[1]):
                raise ValueError(
                    f"{name} has {int(block.shape[0])} row(s) but no column, so it "
                    "states a check or a logical operator over no data qubit"
                )

        widths = {
            name: int(block.shape[1])
            for name, block in blocks.items()
            if block is not None and int(block.shape[1])
        }
        if len(set(widths.values())) > 1:
            stated = ", ".join(f"{name} has {width}" for name, width in widths.items())
            raise ValueError(
                f"the matrices must be indexed by the same data qubits but "
                f"{stated}; every non-empty block has one column per data qubit"
            )
        columns = next(iter(widths.values()), 0)

        for name, block in blocks.items():
            if block is None or not int(block.shape[0]):
                blocks[name] = torch.zeros((0, columns), dtype=torch.int64)
            else:
                blocks[name] = block
        for name, block in blocks.items():
            object.__setattr__(self, name, block)

    @property
    def num_qubits(self) -> int:
        """The number of data qubits every matrix is indexed by."""

        return int(self.hz.shape[1])

    @property
    def num_z_checks(self) -> int:
        """The number of Z-type checks, which are the Z-type detector rows."""

        return int(self.hz.shape[0])

    @property
    def num_x_checks(self) -> int:
        """The number of X-type checks, which are the X-type detector rows."""

        return int(self.hx.shape[0])  # type: ignore[union-attr]

    @property
    def num_checks(self) -> int:
        """The number of detectors in one round, every check counted once."""

        return self.num_z_checks + self.num_x_checks

    @property
    def num_z_logicals(self) -> int:
        """The number of Z-type logical operators, read by an X fault."""

        return int(self.lz.shape[0])  # type: ignore[union-attr]

    @property
    def num_x_logicals(self) -> int:
        """The number of X-type logical operators, read by a Z fault."""

        return int(self.lx.shape[0])  # type: ignore[union-attr]

    @property
    def num_observables(self) -> int:
        """The number of observable rows, the Z-type logicals first."""

        return self.num_z_logicals + self.num_x_logicals


def css_code_matrices(code: StabilizerCode) -> CssCodeMatrices:
    """Return a code record's CSS generator matrices.

    Column ``q`` of every matrix is the ``q``-th qubit :attr:`data_qubits` names, so
    a code is free to declare its data qubits on any qubits, and the rows follow the
    code's own declaration order. Each of the code's checks contributes one row to
    the matrix of its own type and none to the other, and each logical observable
    does the same, so a code that declares an X-type logical operator is modelled
    with one and a code that declares none states an empty block.

    A logical observable that is neither pure X-type nor pure Z-type is refused
    rather than read as one of the two. It anticommutes with some check of each
    type, so it is not a CSS logical operator, and reading only its X support or
    only its Z support would report a different observable than the code declared.

    A **check** that is neither pure X-type nor pure Z-type is refused for the
    same reason, and the refusal is about the matrices rather than about the
    check: this route states a code as two families, and a mixed check would need
    a row in each. The two rows would then leave ``hz hx^T`` non-symmetric, while
    a code whose family rows commute needs it to vanish, so the pair of matrices
    this function returned would describe a code whose checks do not commute.
    Such a code is a stabilizer code and is modelled by
    :meth:`~flagquantum.qec.DetectorErrorModel.from_memory_circuit`, which carries
    each check as the one mixed operator it is.

    Raises:
        TypeError: If ``code`` is not a :class:`~flagquantum.qec.StabilizerCode`.
        ValueError: If a logical observable is of mixed type, or if a check or
            observable names a qubit the code does not declare as a data qubit.
    """

    if not isinstance(code, StabilizerCode):
        raise TypeError("code must be a StabilizerCode")
    data_wires = tuple(code.data_qubits)
    columns = {wire: index for index, wire in enumerate(data_wires)}

    def _rows(entries: tuple[tuple[str, tuple[int, ...]], ...]) -> torch.Tensor:
        matrix = torch.zeros((len(entries), len(data_wires)), dtype=torch.int64)
        for row, (label, support) in enumerate(entries):
            for wire in support:
                if wire not in columns:
                    raise ValueError(
                        f"{label} acts on qubit {wire}, which "
                        f"{type(code).__name__} does not declare as a data qubit"
                    )
                matrix[row, columns[wire]] = 1
        return matrix

    checks = tuple(code.checks)
    for check in checks:
        if check.stabilizer.x_qubits and check.stabilizer.z_qubits:
            raise ValueError(
                f"check {check.index} measures a mixed X-and-Z stabilizer, so it "
                "is not a Calderbank-Shor-Steane check: writing it as a row of "
                "each family would state a pair of matrices whose checks do not "
                "commute, and the code it describes is not this one"
            )
    hz = _rows(
        tuple(
            (f"check {check.index}", check.stabilizer.z_qubits)
            for check in checks
            if check.stabilizer.z_qubits
        )
    )
    hx = _rows(
        tuple(
            (f"check {check.index}", check.stabilizer.x_qubits)
            for check in checks
            if check.stabilizer.x_qubits
        )
    )
    observables = tuple(code.logical_observables)
    for index, observable in enumerate(observables):
        if observable.x_qubits and observable.z_qubits:
            raise ValueError(
                f"logical observable {index} is neither an X-type nor a Z-type "
                "operator, so it is not a CSS logical operator: reading it as one "
                "of the two would report an observable the code does not declare"
            )
    lz = _rows(
        tuple(
            (f"logical observable {index}", observable.z_qubits)
            for index, observable in enumerate(observables)
            if observable.z_qubits
        )
    )
    lx = _rows(
        tuple(
            (f"logical observable {index}", observable.x_qubits)
            for index, observable in enumerate(observables)
            if observable.x_qubits
        )
    )
    return CssCodeMatrices(hz=hz, hx=hx, lz=lz, lx=lx)
