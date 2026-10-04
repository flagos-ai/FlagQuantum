"""Block encode a Pauli sum from its coefficients, without forming the matrix.

`flagquantum.algorithms.primitives.LinearCombinationEncoding` is reachable through
the primitives surface -- `from flagquantum.algorithms.primitives import
LinearCombinationEncoding` -- because the algorithms package adds no root-level
`fq.` name. It is a second implementation of the same two protocols the
`block_encoding` example covers, and this script exists to put numbers on the
three things that are true of it and not of the spectral construction.

The first is that the factor is read off the data rather than off a matrix. A
Pauli sum's subnormalisation here is `sum_j |c_j|`, stated from the caller's
coefficients before any circuit exists, so `alpha` is printed beside both the
spectral norm and the Frobenius norm: it must clear the first and it deliberately
is not the second. The two are not even ordered -- the Frobenius norm of a Pauli
product on `n` wires is `2 ** (n / 2)`, so a one-wire sum can have a Frobenius
norm above the sum of its magnitudes -- which is why the factor is stated rather
than named after a norm.

The second is that the block really is the sum. It is read back out of the
*circuit*, one basis state at a time, with each column taken from the amplitudes
the composition leaves on the operator's wires when every flag wire enters in
`|0>`; a construction that returned the matrix it was handed, or that dropped a
sign, fails here. The sign is the part worth pinning, because the preparation
primitive carries no relative phase and the coefficients cannot therefore be
prepared by signing the amplitudes: the sign is a separate diagonal. The block is
checked against the *signed* sum and against the sum of the magnitudes, and the
two differ, so a block that came back unsigned is visible rather than merely
imprecise.

The third is that the walk step is the same walk step. One step `W = R U`, with
`R` the reflection on the index register, has an eigenphase for every encoded
eigenvalue, `exp(+-i arccos(lambda / alpha))`, so its cosines are the encoded
spectrum twice over. The observed side comes from `torch.linalg.eigvals` of the
circuit's unitary and the expected side from `torch.linalg.eigvalsh` of the
matrix, because two spellings of one computation would agree whether or not
either was right. Unlike the spectral encoding, `U` here is **not** Hermitian --
its terms share the ladder ancillas the controlled gates are built from, so the
product of them is not its own reverse -- and that is printed, because it is the
reason the adjoint reverses the select's terms instead of reusing them. The
adjoint is printed as `W^dagger W` against the identity over the *whole*
register, since a reversal skipped leaves a residual of order one off the flagged
subspace while looking correct on it.

The premise the unit rests on, and does not check: the preparation is a **dense
classical precomputation over `2 ** k` amplitudes**, exponential in the index
register's width, and the select is one multi-controlled Pauli per term. This is
the demonstration scale the algorithm modules work at: a gate-efficient
synthesis, amplitude amplification, and any gate-count claim are absent, and
nothing here bounds the cost of a term count this unit has not been run at. The
ancilla count is `k + max(k - 2, 0)` and not one wire per term, so a wider sum
costs a wider index register rather than a longer ladder.

The last block is why the surface is a protocol and not a class. A consumer
written against `BlockEncoding` and `WalkEncoding` names three numbers and one
method, so the spectral encoding of the same Hamiltonian -- which holds a dense
matrix this one never forms -- is read by the same code, and the two are printed
side by side. That, and not the interface's shape, is what a second
implementation buys.

Sizes, and why: two qubits and six terms, because six is the smallest count whose
index register is wide enough to need a ladder ancilla, which is where the select
stops being a single multi-controlled gate and the two orders stop agreeing. A
five-term sum would exercise neither.
"""

from __future__ import annotations

import torch

from flagquantum.algorithms.core import Hamiltonian, HamiltonianTerm
from flagquantum.algorithms.primitives import (
    BlockEncoding,
    LinearCombinationEncoding,
    WalkEncoding,
    spectral_block_encoding,
)
from flagquantum.circuit import Circuit
from flagquantum.simulation.unitary import get_unitary

DTYPE = torch.complex128
LABEL_WIDTH = 42

# A six-term two-qubit sum with both signs present and no identity term: every
# coefficient contributes to `alpha`, two of them contribute a sign diagonal, and
# the term count is what forces a ladder ancilla rather than a bare multi-controlled
# gate. The matrix is never formed in the construction; it is formed below only to
# check the block the construction emits.
TERMS: tuple[tuple[float, dict[int, str]], ...] = (
    (1.0, {0: "x"}),
    (0.5, {1: "z"}),
    (-1.5, {0: "z", 1: "x"}),
    (0.25, {0: "z", 1: "z"}),
    (0.75, {0: "x", 1: "x"}),
    (-0.5, {0: "x", 1: "z"}),
)


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def hamiltonian() -> Hamiltonian:
    """Return the six-term sum as the algorithms layer's own value object."""

    return _build(TERMS)


def hamiltonian_with_magnitudes() -> Hamiltonian:
    """Return the same six-term sum with every coefficient replaced by its magnitude."""

    return _build(tuple((abs(c), axes) for c, axes in TERMS))


def _build(terms: tuple[tuple[float, dict[int, str]], ...]) -> Hamiltonian:
    return Hamiltonian(
        [
            HamiltonianTerm(coefficient, dict(axes), sorted(axes))
            for coefficient, axes in terms
        ]
    )


def flag_zero_rows(encoding: BlockEncoding) -> torch.Tensor:
    """Return the basis indices whose every flag wire reads zero.

    The flag register is named by its first wire and is consecutive, but it is not
    assumed to lead the register: the mask is built from wire positions, so an
    encoding that put its flag wires anywhere would still be read correctly here.
    """

    n_wires = encoding.num_ancilla + encoding.num_system
    wanted = [
        basis
        for basis in range(1 << n_wires)
        if all(
            ((basis >> (n_wires - 1 - wire)) & 1) == 0
            for wire in range(encoding.num_ancilla)
        )
    ]
    return torch.tensor(wanted, dtype=torch.long)


def block_of(encoding: BlockEncoding) -> torch.Tensor:
    """Return the encoding's flagged block, read out of the circuit it appends.

    The block is `<0|_flag U |0>_flag`, so its column `j` is what the circuit
    leaves on the operator's wires when it is entered in the basis state `|j>`
    with every flag wire in `|0>`. Each column therefore comes from a circuit of
    its own and no unitary is written down as a matrix, which is the only thing
    this repository exposes.
    """

    width = encoding.num_system
    qubits = list(range(encoding.num_ancilla, encoding.num_ancilla + width))
    rows = flag_zero_rows(encoding)
    columns = []
    for basis in range(1 << width):
        circuit = Circuit(encoding.num_ancilla + width)
        for index in range(width):
            if (basis >> (width - 1 - index)) & 1:
                circuit.gate("x", qubits[index])
        encoding.append_apply(circuit, ancilla=0, qubits=qubits)
        columns.append(circuit.state().reshape(-1)[rows])
    return torch.stack(columns, dim=1)


def ladder_zero_rows(encoding: LinearCombinationEncoding) -> torch.Tensor:
    """Return the state indices whose ladder ancillas are all ``|0>``.

    The ladder ancillas are the flag register's last ``num_ancilla - index_width``
    wires. This is the subspace the walk identity is stated on: the encoding is
    exactly block diagonal in the ladder register, but it is not the identity
    there, so the arccosine spectrum is a claim about this subspace rather than
    about the whole register.
    """

    n_wires = encoding.num_ancilla + encoding.num_system
    mask = 0
    for wire in range(encoding.index_width, encoding.num_ancilla):
        mask |= 1 << (n_wires - 1 - wire)
    return torch.tensor(
        [state for state in range(1 << n_wires) if state & mask == 0], dtype=torch.long
    )


def register_unitary(encoding: BlockEncoding, step: str) -> torch.Tensor:
    """Return the unitary the encoding alone, or one walk step, applies."""

    circuit = Circuit(encoding.num_ancilla + encoding.num_system)
    qubits = list(
        range(encoding.num_ancilla, encoding.num_ancilla + encoding.num_system)
    )
    if step == "walk":
        encoding.append_walk_step(circuit, ancilla=0, qubits=qubits)
    elif step == "adjoint":
        encoding.append_adjoint_walk_step(circuit, ancilla=0, qubits=qubits)
    else:
        encoding.append_apply(circuit, ancilla=0, qubits=qubits)
    return get_unitary(circuit).to(DTYPE)


def step_cosines(encoding: LinearCombinationEncoding) -> torch.Tensor:
    """Return the cosines of one walk step's eigenphases on the ladder-zero subspace."""

    unitary = register_unitary(encoding, "walk")
    rows = ladder_zero_rows(encoding)
    restricted = unitary[rows][:, rows]
    return torch.sort(torch.cos(torch.angle(torch.linalg.eigvals(restricted)))).values


def attained(values: torch.Tensor, wanted: torch.Tensor, *, tolerance: float) -> int:
    """Return how many of ``wanted`` a value of ``values`` lands within ``tolerance`` of."""

    return sum(
        int(float((values - target).abs().min()) < tolerance)
        for target in wanted.tolist()
    )


def rounded(values: torch.Tensor) -> list[float]:
    """Return the tensor's entries as rounded Python floats, for printing."""

    return [round(float(value), 6) for value in values]


def main() -> None:
    print("Linear-combination block encoding --")
    print("  flagquantum.algorithms.primitives.LinearCombinationEncoding")
    print()

    print("The register, and the factor the construction states in advance")
    operator = hamiltonian()
    encoding = LinearCombinationEncoding(operator)
    dense = operator.matrix()
    alpha = encoding.alpha
    magnitudes = sum(abs(coefficient) for coefficient, _ in TERMS)
    report("num_system", encoding.num_system)
    report("terms", len(encoding.terms))
    report("index_width", encoding.index_width)
    report("num_ancilla", encoding.num_ancilla)
    report("sum of the coefficient magnitudes", round(magnitudes, 6))
    report("alpha", round(alpha, 6))
    report("alpha is the sum of the magnitudes", alpha == magnitudes)
    report("spectral norm", round(float(torch.linalg.norm(dense, ord=2)), 6))
    report(
        "alpha clears the spectral norm",
        alpha >= float(torch.linalg.norm(dense, ord=2)),
    )
    report("Frobenius norm", round(float(torch.linalg.norm(dense)), 6))
    report(
        "alpha is the Frobenius norm",
        abs(alpha - float(torch.linalg.norm(dense))) < 1e-12,
    )
    report("index_width is one wire per term", encoding.index_width == len(TERMS))
    print()

    print("The block, read out of the circuit one column at a time")
    block = block_of(encoding)
    expected_block = dense / alpha
    # The sum with every coefficient replaced by its magnitude. It is the sum a
    # construction that prepared the magnitudes and dropped the signs would encode,
    # and it differs from the requested operator because two coefficients are
    # negative, so a block that came back unsigned is visible here rather than merely
    # imprecise.
    unsigned = hamiltonian_with_magnitudes().matrix() / alpha
    report(
        "circuit block vs H / alpha",
        f"{float((block - expected_block).abs().max()):.2e}",
    )
    report("circuit block vs H", f"{float((block - dense).abs().max()):.3f}")
    report(
        "circuit block vs the sum of magnitudes",
        f"{float((block - unsigned).abs().max()):.3f}",
    )
    report(
        "the signed and unsigned sums differ",
        f"{float((unsigned - expected_block).abs().max()):.3f}",
    )
    print()

    print("The encoding is a unitary, and it is not its own adjoint")
    unit = register_unitary(encoding, "apply")
    identity = torch.eye(1 << (encoding.num_ancilla + encoding.num_system), dtype=DTYPE)
    report("U is unitary", f"{float((unit @ unit.mH - identity).abs().max()):.2e}")
    report("U is Hermitian", f"{float((unit - unit.mH).abs().max()):.3f}")
    report("ladder ancillas", encoding.num_ancilla - encoding.index_width)
    print()

    print("One walk step, and the spectrum it carries on the ladder-zero subspace")
    encoded = torch.sort(torch.linalg.eigvalsh(dense).real / alpha).values
    observed = step_cosines(encoding)
    report("encoded cosines", rounded(encoded))
    report("subspace cosines", rounded(observed))
    report(
        "encoded cosines attained",
        f"{attained(observed, encoded, tolerance=1e-6)} of {encoded.numel()}",
    )
    report("subspace eigenvalues", observed.numel())
    report(
        "the subspace holds more than the encoding",
        observed.numel() > 2 * encoded.numel(),
    )
    report("cosines off the endpoints", bool(float(encoded.abs().max()) < 1 - 1e-6))
    print()

    print("The adjoint, entrywise rather than on the flagged subspace")
    forward = register_unitary(encoding, "walk")
    backward = register_unitary(encoding, "adjoint")
    report(
        "W^dagger W vs the identity",
        f"{float((backward @ forward - identity).abs().max()):.2e}",
    )
    report(
        "W W^dagger vs the identity",
        f"{float((forward @ backward - identity).abs().max()):.2e}",
    )
    report(
        "U^dagger U vs the identity",
        f"{float((unit.mH @ unit - identity).abs().max()):.2e}",
    )
    print()

    print("A second implementation behind the same two protocols")
    report("this is a BlockEncoding", isinstance(encoding, BlockEncoding))
    report("this is a WalkEncoding", isinstance(encoding, WalkEncoding))
    report("H is real", bool(float(dense.imag.abs().max()) < 1e-12))
    spectral = spectral_block_encoding(dense.real.to(torch.float64), alpha)
    spectral_block = block_of(spectral)
    report("spectral num_system", spectral.num_system)
    report("spectral num_ancilla", spectral.num_ancilla)
    report("spectral alpha", round(spectral.alpha, 6))
    report("the two blocks agree", f"{float((block - spectral_block).abs().max()):.2e}")
    print()

    report("premise", "the preparation is a dense classical precomputation over")
    print(f"  {'':<{LABEL_WIDTH}}  2 ** k amplitudes, exponential in the index")
    print(f"  {'':<{LABEL_WIDTH}}  register's width, and the select is one")
    print(
        f"  {'':<{LABEL_WIDTH}}  multi-controlled Pauli per term. Nothing here bounds"
    )
    print(f"  {'':<{LABEL_WIDTH}}  a term count this unit has not been run at, the")
    print(f"  {'':<{LABEL_WIDTH}}  ancilla count is k + max(k - 2, 0) rather than one")
    print(
        f"  {'':<{LABEL_WIDTH}}  wire per term, and the cost of the preparation belongs"
    )
    print(
        f"  {'':<{LABEL_WIDTH}}  to the caller's 2 ** k amplitudes rather than to any"
    )
    print(f"  {'':<{LABEL_WIDTH}}  bound this unit states. A gate-efficient synthesis,")
    print(
        f"  {'':<{LABEL_WIDTH}}  amplitude amplification, and any gate-count claim are"
    )
    print(f"  {'':<{LABEL_WIDTH}}  absent.")


if __name__ == "__main__":
    main()
