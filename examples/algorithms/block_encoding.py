"""Read a matrix out of a unitary's flagged block, and then out of its walk step.

`flagquantum.algorithms.primitives.block_encoding` is reachable through the
primitives surface -- `from flagquantum.algorithms.primitives import
spectral_block_encoding` -- because the algorithms package adds no root-level
`fq.` name.

A block encoding of a Hermitian `H` at a subnormalisation `alpha` is a unitary
`U` whose flagged block is `H / alpha`: writing `|0>` for the flag register's
all-zero state and `P = |0><0|` for the projector onto it, `P U P = H / alpha`
and `U` is otherwise unconstrained. Two claims are what this script exists to put
a number on, and neither is checkable by reading the code that makes them.

The first is that the block really is `H / alpha`. The block is read back out of
the *circuit*, one basis state at a time -- each column is the state the
composition leaves on the operator's wires when the flag enters in `|0>` -- so a
construction that returned the matrix it was handed, or returned a constant,
fails here rather than in a comment. The second claim is the one that makes the
construction worth more than its block: one **walk step** `W = Z_ancilla U` has
an eigenphase for every encoded eigenvalue, `exp(+-i arccos(lambda / alpha))`, so
its cosines are the encoded spectrum twice over. Both sides of that identity are
printed, and they come from different routines on different objects -- the
observed side from `torch.linalg.eigvals` of the circuit's unitary, the expected
side from `torch.linalg.eigvalsh` of the matrix -- because two spellings of the
same computation would agree whether or not either was right.

A spectrum cannot tell the walk step from the encoding it was built out of, since
`W = Z_ancilla U` and `U` are two Hermitian unitaries on the same register. So the
step's definition is printed as well, `W` against `Z_ancilla U` as matrices, and
that is the line the second implementation below is held to: it has no matrix for
anyone to compare against, and the composed unitary is still the one it must
produce. The adjoint is printed as `W^dagger W` against the identity rather than as
its cosines, which are `W`'s own.

The premise the unit rests on, and does not check: the eigendecomposition is
**dense and exact**, so the gate emitted is one dense matrix on `n + 1` qubits
and the cost is that of diagonalising `H`. This is the demonstration scale the
algorithm modules work at, and the error of an approximate `H` belongs to the
caller rather than being estimated here, because nothing in this unit bounds it.
The construction is exact in the other sense, and that is also printed: `A` and
`sqrt(I - A**2)` are both functions of the same Hermitian matrix, so they commute
and `U ** 2 = I` holds identically rather than to within the quality of an
eigenbasis.

The last two blocks are why the surface is a protocol and not a class. A consumer
written against `BlockEncoding` and `WalkEncoding` names three numbers and one
method, so a second implementation that holds no dense matrix at all -- here the
encoding of a single Pauli `Z`, whose complement block is identically zero -- is
read by the same code, and the two are printed side by side. That, and not the
interface's shape, is what a replacement buys: `flagquantum.algorithms.svd` used
to carry a private mean-of-two-branches construction, this unit is the direct
spectral reflection that replaced it, and the walk identity above is the
capability the private form did not have.

Sizes, and why: two qubits, because a one-qubit matrix is the smallest admitted
and carries a single eigenvalue, so a spectrum comparison would have nothing to
order; and a sparse symmetric matrix rather than random noise, so the printed
numbers are readable. `docs/guides/ALGORITHMS.md` carries the per-unit boundary.

Run it with:

    python -m examples.algorithms.block_encoding
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from flagquantum.algorithms.primitives import (
    BlockEncoding,
    WalkEncoding,
    spectral_block_encoding,
    subnormalisation,
)
from flagquantum.circuit import Circuit
from flagquantum.simulation.unitary import get_unitary

DTYPE = torch.complex128
LABEL_WIDTH = 40
MATRIX = torch.tensor(
    [
        [1.0, 0.5, 0.0, 0.0],
        [0.5, -2.0, 0.25, 0.0],
        [0.0, 0.25, 0.75, 0.0],
        [0.0, 0.0, 0.0, 1.5],
    ],
    dtype=torch.float64,
)
SMALL = torch.tensor([[1.0, 0.5], [0.5, -2.0]], dtype=torch.float64)


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def block_of(encoding: BlockEncoding) -> torch.Tensor:
    """Return the encoding's flagged block, read out of the circuit it appends.

    The block is `<0|_flag U |0>_flag`, so its column `j` is the state the
    circuit leaves when it is entered in the basis state `|j>` with the flag in
    `|0>`. Every column therefore comes from a circuit of its own, and no unitary
    is written down as a matrix: what is read is what the composition leaves,
    which is the only thing this repository exposes.
    """

    width = encoding.num_system
    dimension = 1 << width
    qubits = list(range(encoding.num_ancilla, encoding.num_ancilla + width))
    columns = []
    for basis in range(dimension):
        circuit = Circuit(encoding.num_ancilla + width)
        for index in range(width):
            if (basis >> (width - 1 - index)) & 1:
                circuit.gate("x", qubits[index])
        encoding.append_apply(circuit, ancilla=0, qubits=qubits)
        columns.append(circuit.state().reshape(-1)[:dimension])
    return torch.stack(columns, dim=1)


def step_unitary(encoding: WalkEncoding, step: str = "walk") -> torch.Tensor:
    """Return the unitary one walk step, or its adjoint, applies to the register."""

    circuit = Circuit(encoding.num_ancilla + encoding.num_system)
    qubits = list(
        range(encoding.num_ancilla, encoding.num_ancilla + encoding.num_system)
    )
    if step == "walk":
        encoding.append_walk_step(circuit, ancilla=0, qubits=qubits)
    else:
        encoding.append_adjoint_walk_step(circuit, ancilla=0, qubits=qubits)
    return get_unitary(circuit).to(DTYPE)


def step_cosines(encoding: WalkEncoding, step: str = "walk") -> torch.Tensor:
    """Return the cosines of the eigenphases of one walk step or its adjoint."""

    eigenvalues = torch.linalg.eigvals(step_unitary(encoding, step))
    return torch.sort(torch.cos(torch.angle(eigenvalues))).values


def encoding_unitary(encoding: BlockEncoding) -> torch.Tensor:
    """Return the unitary the encoding alone applies to the register."""

    circuit = Circuit(encoding.num_ancilla + encoding.num_system)
    qubits = list(
        range(encoding.num_ancilla, encoding.num_ancilla + encoding.num_system)
    )
    encoding.append_apply(circuit, ancilla=0, qubits=qubits)
    return get_unitary(circuit).to(DTYPE)


def flag_phase_flip(encoding: BlockEncoding) -> torch.Tensor:
    """Return the flag wire's phase flip, in the register's own basis order.

    One flag wire, and it leads the register the encoding is appended on, so the flip is
    `diag(1, -1)` on that wire and the identity on the operator's. Nothing here
    generalises to several flag wires, because the surface this example reads holds one.
    """

    return torch.kron(
        torch.diag(torch.tensor([1.0, -1.0], dtype=DTYPE)),
        torch.eye(1 << encoding.num_system, dtype=DTYPE),
    )


def encoded_spectrum(encoding: BlockEncoding) -> torch.Tensor:
    """Return the encoded matrix's eigenvalues over alpha, each appearing twice."""

    normalised = (torch.linalg.eigvalsh(encoding.matrix) / encoding.alpha).real
    return torch.sort(torch.cat((normalised, normalised))).values


def rounded(values: torch.Tensor) -> list[float]:
    """Return the tensor's entries as rounded Python floats, for printing."""

    return [round(float(value), 6) for value in values]


class PauliZEncoding:
    """A second implementation of both protocols that holds no dense matrix.

    It encodes the Pauli `Z` on one qubit at `alpha` exactly one, which is that
    operator's spectral norm, so the complement block is identically zero and the
    reflection is `Z_flag (x) Z`. The walk step is then `Z` on the operator's
    qubit alone. Every gate is emitted by name and nothing here reads a matrix,
    which is the point: a consumer written against the protocols must not be able
    to tell the two implementations apart.
    """

    @property
    def num_system(self) -> int:
        """The number of qubits the encoded operator acts on."""

        return 1

    @property
    def num_ancilla(self) -> int:
        """The number of flag qubits."""

        return 1

    @property
    def alpha(self) -> float:
        """The subnormalisation, which is `Z`'s spectral norm exactly."""

        return 1.0

    def append_apply(
        self, circuit: Circuit, *, ancilla: int, qubits: Sequence[int]
    ) -> None:
        """Append the reflection `Z_flag (x) Z` to `circuit`."""

        circuit.gate("z", qubits[0])
        circuit.gate("z", ancilla)

    def append_walk_step(
        self, circuit: Circuit, *, ancilla: int, qubits: Sequence[int]
    ) -> None:
        """Append one walk step, which is `Z` on the operator's qubit alone."""

        del ancilla
        circuit.gate("z", qubits[0])

    def append_adjoint_walk_step(
        self, circuit: Circuit, *, ancilla: int, qubits: Sequence[int]
    ) -> None:
        """Append the walk step's adjoint, which is the step itself here."""

        del ancilla
        circuit.gate("z", qubits[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Block encoding and walk step demo")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="the subnormalisation; the default is the matrix's Frobenius norm",
    )
    args = parser.parse_args()

    encoding = spectral_block_encoding(MATRIX, args.alpha)
    qubits = list(range(1, 1 + encoding.num_system))

    print("=" * 72)
    print("Block encoding -- flagquantum.algorithms.primitives.block_encoding")
    print("=" * 72)
    report("task", "a Hermitian matrix into the flagged block of a unitary")
    report("method", "the spectral reflection, built exactly")
    report("premise", "the eigendecomposition is dense and exact, so the gate is one")
    print(f"  {'':<{LABEL_WIDTH}}  dense matrix on n + 1 qubits and the cost is")
    print(f"  {'':<{LABEL_WIDTH}}  diagonalising H. The error of an approximate H")
    print(f"  {'':<{LABEL_WIDTH}}  belongs to the caller: nothing here bounds it.")
    print()

    print("the encoding")
    report("matrix", "4x4 symmetric, 2 qubits")
    report("num_system", encoding.num_system)
    report("num_ancilla", encoding.num_ancilla)
    report("alpha", f"{encoding.alpha:.12f}")
    report("Frobenius norm", f"{subnormalisation(MATRIX):.12f}")
    report("spectral norm", f"{float(torch.linalg.norm(MATRIX, 2)):.12f}")
    report(
        "alpha clears the spectral norm",
        bool(encoding.alpha >= float(torch.linalg.norm(MATRIX, 2))),
    )
    print(f"  {'':<{LABEL_WIDTH}}  the default factor is the Frobenius norm, which is")
    print(f"  {'':<{LABEL_WIDTH}}  at least the spectral norm and needs no")
    print(f"  {'':<{LABEL_WIDTH}}  eigendecomposition; a factor below it is refused")
    print(f"  {'':<{LABEL_WIDTH}}  because no unitary has such a block")
    print()

    print("the flagged block is H / alpha, read back out of the circuit")
    block = block_of(encoding)
    expected = MATRIX.to(DTYPE) / encoding.alpha
    report("circuit block vs H / alpha", f"{float((block - expected).abs().max()):.3e}")
    report(
        "dense block vs circuit block",
        f"{float((encoding.block() - block).abs().max()):.3e}",
    )
    report("circuit block vs H", f"{float((block - MATRIX.to(DTYPE)).abs().max()):.6f}")
    print(f"  {'':<{LABEL_WIDTH}}  the third line is the one a construction that")
    print(f"  {'':<{LABEL_WIDTH}}  returned its argument prints as zero: the block is")
    print(f"  {'':<{LABEL_WIDTH}}  H over alpha rather than H")
    print()

    print("the reflection is Hermitian, unitary, and its own inverse")
    select = encoding.select
    identity = torch.eye(int(select.shape[0]), dtype=DTYPE)
    report("U is Hermitian", f"{float((select - select.mH).abs().max()):.3e}")
    report(
        "U squared is the identity",
        f"{float((select @ select - identity).abs().max()):.3e}",
    )
    report("U is unitary", f"{float((select @ select.mH - identity).abs().max()):.3e}")
    print(f"  {'':<{LABEL_WIDTH}}  A and sqrt(I - A**2) are functions of the same")
    print(f"  {'':<{LABEL_WIDTH}}  Hermitian matrix, so they commute and the square is")
    print(f"  {'':<{LABEL_WIDTH}}  the identity identically rather than to within the")
    print(f"  {'':<{LABEL_WIDTH}}  quality of an eigenbasis")
    print()

    print("one walk step W = Z_ancilla U has a phase per encoded eigenvalue")
    observed = step_cosines(encoding)
    spectrum = encoded_spectrum(encoding)
    report("matrix eigenvalues", rounded(torch.linalg.eigvalsh(MATRIX)))
    report("expected cosines", rounded(spectrum))
    report("observed cosines", rounded(observed))
    report("agreement", f"{float((observed - spectrum).abs().max()):.3e}")
    report(
        "cosines off the endpoints",
        bool((torch.abs(torch.abs(observed) - 1.0) > 1e-6).any()),
    )
    print(f"  {'':<{LABEL_WIDTH}}  arccos of each encoded eigenvalue, twice over. The")
    print(
        f"  {'':<{LABEL_WIDTH}}  observed side is eigvals of the circuit's unitary and"
    )
    print(
        f"  {'':<{LABEL_WIDTH}}  the expected side is eigvalsh of the matrix, so these"
    )
    print(f"  {'':<{LABEL_WIDTH}}  are not one computation written twice")
    print()

    print("the phase flip is a construction: without it the identity fails")
    plain = Circuit(1 + encoding.num_system)
    encoding.append_apply(plain, ancilla=0, qubits=qubits)
    plain_cosines = torch.sort(
        torch.cos(torch.angle(torch.linalg.eigvals(get_unitary(plain).to(DTYPE))))
    ).values
    report("plain U cosines", rounded(plain_cosines))
    report(
        "plain vs encoded spectrum",
        f"{float((plain_cosines - spectrum).abs().max()):.6f}",
    )
    # The step is defined as the flag flip composed with the encoding, so that is printed
    # too: a spectrum alone cannot tell the walk step from the encoding it was built from,
    # and this is the line that can. It is also the line a second implementation is held
    # to below, where there is no matrix to compare against.
    step = step_unitary(encoding)
    report(
        "walk step vs flag flip of the encoding",
        f"{float((step - flag_phase_flip(encoding) @ encoding_unitary(encoding)).abs().max()):.3e}",
    )
    report(
        "adjoint step inverts the step",
        f"{float((step_unitary(encoding, 'adjoint') @ step - identity).abs().max()):.3e}",
    )
    print(f"  {'':<{LABEL_WIDTH}}  U on its own is Hermitian -- the flag flip is what")
    print(
        f"  {'':<{LABEL_WIDTH}}  makes it a rotation -- so its eigenphases are 0 and pi"
    )
    print(
        f"  {'':<{LABEL_WIDTH}}  and its cosines sit on the endpoints. That degenerate"
    )
    print(
        f"  {'':<{LABEL_WIDTH}}  reading is what the walk step replaces, and the flag"
    )
    print(f"  {'':<{LABEL_WIDTH}}  flip composed with U is exactly what it is")
    print()

    print("the surface is a protocol, and a second implementation fits it")
    built = PauliZEncoding()
    report("spectral is a BlockEncoding", isinstance(encoding, BlockEncoding))
    report("spectral is a WalkEncoding", isinstance(encoding, WalkEncoding))
    report("built is a WalkEncoding", isinstance(built, WalkEncoding))
    report("built alpha", built.alpha)
    report("built source", "two named gates, no matrix anywhere")
    report("built walk cosines", rounded(step_cosines(built)))
    report(
        "built walk vs flag flip of the encoding",
        f"{float((step_unitary(built) - flag_phase_flip(built) @ encoding_unitary(built)).abs().max()):.3e}",
    )
    report("spectral walk cosines", rounded(observed))
    print(f"  {'':<{LABEL_WIDTH}}  the consumer above names three numbers and one")
    print(f"  {'':<{LABEL_WIDTH}}  method, so it reads both without learning which it")
    print(f"  {'':<{LABEL_WIDTH}}  was handed -- which is what a boundary buys and an")
    print(f"  {'':<{LABEL_WIDTH}}  interface alone does not")
    print()

    print("calls the unit refuses by name")
    cases = (
        ("a matrix that is not a tensor", "not a tensor", None),
        ("a matrix that is not square", torch.zeros(2, 3, dtype=torch.float64), None),
        (
            "a dimension that is not a power of two",
            torch.zeros(3, 3, dtype=torch.float64),
            None,
        ),
        ("an integer matrix", torch.zeros(2, 2, dtype=torch.int64), None),
        ("the zero matrix", torch.zeros(2, 2, dtype=torch.float64), None),
        (
            "a matrix with a non-finite entry",
            torch.tensor([[1.0, float("nan")], [0.0, 1.0]]),
            None,
        ),
        (
            "a matrix that is not Hermitian",
            torch.tensor([[1.0, 0.5], [0.25, 1.0]]),
            None,
        ),
        (
            "a factor below the spectral norm",
            SMALL,
            float(torch.linalg.norm(SMALL, 2)) * 0.99,
        ),
        ("a factor that is zero", SMALL, 0.0),
        ("a factor that is not a real number", SMALL, True),
    )
    for label, candidate, alpha in cases:
        try:
            spectral_block_encoding(candidate, alpha)
        except Exception as exc:
            report(label, f"refused -- {type(exc).__name__}: {exc}")
        else:
            report(label, "NOT REFUSED")
    print(f"  {'':<{LABEL_WIDTH}}  the non-Hermitian argument is the one that would")
    print(f"  {'':<{LABEL_WIDTH}}  otherwise be encoded from half of itself, because")
    print(f"  {'':<{LABEL_WIDTH}}  eigh reads one triangle and would not have said so")
    print()

    print("a wrong register is refused before the first gate is emitted")
    for label, ancilla, wrong in (
        ("the flag inside the register", 1, [1, 2]),
        ("one qubit for a two-qubit operator", 0, [1]),
        ("a repeated qubit", 0, [1, 1]),
    ):
        circuit = Circuit(3)
        try:
            encoding.append_apply(circuit, ancilla=ancilla, qubits=wrong)
        except Exception as exc:
            report(label, f"refused -- {type(exc).__name__}: {exc}")
        else:
            report(label, "NOT REFUSED")
        report(f"gates after {label}", len(circuit))
    print(f"  {'':<{LABEL_WIDTH}}  every one of these is checked before the gate is")
    print(
        f"  {'':<{LABEL_WIDTH}}  appended, so a rejected call leaves an empty circuit"
    )
    print(f"  {'':<{LABEL_WIDTH}}  rather than a half-built encoding")
    print()

    print("take away")
    print("  a block encoding is a unitary whose flagged block is a matrix over a")
    print("  factor the caller chooses, and its walk step turns that matrix's")
    print("  spectrum into phases, which is what makes it a subroutine rather than")
    print("  a repackaging. The construction here is exact and dense: the gate is")
    print("  one matrix on n + 1 qubits, the cost is diagonalising H, and no error")
    print("  bound is reported because a caller's approximation is the caller's.")
    print("  What is beside it: a Pauli linear-combination encoding, in")
    print("  examples/algorithms/linear_combination.py, whose preparation is a")
    print("  superposition over the coefficients rather than the identity, at the")
    print("  sum of the coefficient magnitudes as the factor. It is a second")
    print("  implementation of the same two protocols, so a consumer written")
    print("  against those protocols reads either one. What is")
    print("  not here: a gate-efficient synthesis, qubitization as a public walk")
    print("  object, QSVT, and double factorization.")


if __name__ == "__main__":
    main()
