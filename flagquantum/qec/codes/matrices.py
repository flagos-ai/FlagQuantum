"""CSS matrix records, validation and extraction, independent of DEM execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import StabilizerCode

__all__ = ("CssCodeMatrices", "css_code_matrices")


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
