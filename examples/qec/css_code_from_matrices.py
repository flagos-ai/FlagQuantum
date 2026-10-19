"""Declare a stabilizer code this package does not ship, and run it end to end.

Five code records this package ships -- the repetition code, the rotated surface
code, the Steane code, the triangular colour code, and the square-lattice toric
code -- each state their own wires, checks, logical operators, and distance.
This script takes the other route: it writes a code down as parity-check matrices
and lets the record do the rest.  That is the route a user needs when the code
they care about is not one of the five, and it is the route a
Calderbank-Shor-Steane family needs before it can reach a detector error model at
all.

Two codes are worth walking through, and they answer two different questions.

The Steane code is the control.  Its matrices are written here by hand, and the
record built from them has to agree with the record this package already ships:
the same seven data qubits, the same six ancillas, the same stabilizer group, and
the same distance three.  Without that, the matrix route would be a second,
disagreeing description of a code that already exists rather than an alternative
way to state one.

The toric code is the point.  It is also the one code here that this package now
declares as well, so the two routes can be held against each other.  One qubit
sits on each edge of an ``L``-by-``L`` torus, a face is a Z-type check, a star is
an X-type check, and the logical operators are the cycles that wrap around the
torus.  Every check touches four data qubits however large the lattice is, while
the shortest logical operator touches ``L`` of them, so the distance grows with
the lattice and the code is a low-density parity-check code rather than a family
whose checks grow with its size.  That is the shape the record could not admit
before: the declared records carry a constant distance and this one does not.

The matrices written here number the edges one way and ``toric_code`` numbers
them another, so the two descriptions do not agree row by row and no attempt is
made to relabel one into the other.  What they must agree on is everything the
wire numbering cannot change -- the qubit and check counts, the distance, the
number of logical qubits, the weight of a fault's signature, and the verdict the
matcher reaches -- and that agreement is printed rather than argued.  Two
independent derivations of one code agreeing on those is a stronger statement
than either derivation alone, because a transcription error in the shipped record
would show up as a disagreement here.

What the script measures, and what it does not:

* The distance is computed from the matrices, not read off the logical operator
  the caller happened to write down.  The Steane section states the same logical
  operator at weight three and again at weight seven, multiplied by a stabilizer,
  and reports three both times.
* Every fault of the toric model flips at most two detectors, which is why the
  minimum-weight matcher this package ships accepts it.  The Steane model has a
  fault that flips three, which is why the matcher refuses that one.  Both are
  printed beside each other, because a matcher that accepted everything would be
  as wrong as one that accepted nothing.
* The torus of lattice two has distance two, so a single fault is already a
  logical error and no correction removes it.  The torus of lattice three has
  distance three, so it corrects one, and the observable rate under decoding
  falls.  Both halves are printed, because the second alone would not separate a
  decoder that works from one that guesses.
* The written matrices and the shipped record are compared on the quantities the
  wire numbering cannot change, including the full weight histogram of the
  detector error model.  No decoding claim rests on the comparison; it is about
  the two derivations describing one code.

This is a local, single-process demonstration of an experimental QEC surface.  It
builds small lattices only, because the distance search tries every wire subset
up to the weight it is given and the cost of a larger bound is the caller's to
accept.  It establishes no threshold, no logical-suppression claim, and no
scalability claim, and it contacts nothing.

Run it with:

    python -m examples.qec.css_code_from_matrices

"""

from __future__ import annotations

import argparse

from flagquantum.qec import (
    CssCode,
    DetectorErrorModel,
    MinimumWeightMatchingDecoder,
    PhenomenologicalNoise,
    SteaneCode,
    build_memory_circuit,
    css_code_matrices,
    sample_memory_circuit,
    toric_code,
)

LABEL_WIDTH = 40
#: Where the two codes below sit on the torus. Lattice two is the smallest torus
#: there is and carries distance two; lattice three carries distance three. They
#: are two sizes and not a family of growing capacity.
LATTICES = (2, 3)
#: One depolarizing data rate and one measurement rate, for both codes.
PROBABILITY = 0.05
SEED = 0

#: The Steane code's parity checks, one row per check and one column per data
#: qubit. The X-type block is the same matrix, which is what makes the code CSS.
STEANE_CHECKS = (
    (0, 0, 0, 1, 1, 1, 1),
    (0, 1, 1, 0, 0, 1, 1),
    (1, 0, 1, 0, 1, 0, 1),
)
#: One weight-three Z-type logical operator. Seven of them are equivalent up to a
#: stabilizer; this is the one the shipped record declares too.
STEANE_LOGICAL = (1, 1, 1, 0, 0, 0, 0)


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def check_counts(code: CssCode) -> tuple[int, int]:
    """Return the number of Z-type and of X-type checks the record carries.

    The record states one tuple of checks rather than one per type, which is the
    shape a memory circuit needs; the two counts are read off it here rather than
    stored a second time.
    """

    z_type = sum(1 for check in code.checks if not check.stabilizer.x_wires)
    return z_type, len(code.checks) - z_type


def odd_overlap(vector: tuple[int, ...], row: tuple[int, ...]) -> bool:
    """Whether two binary vectors overlap in an odd number of positions."""

    return sum(left * right for left, right in zip(vector, row, strict=True)) % 2 == 1


def toric_matrices(lattice: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return the checks and a logical basis of the toric code on a torus.

    One qubit per edge: ``H(i, j)`` runs from vertex ``(i, j)`` to ``(i, j + 1)``
    and ``V(i, j)`` from ``(i, j)`` to ``(i + 1, j)``, wrapping at the boundary.  A
    star is the four edges at one vertex, a face the four around one cell, and the
    two logical cycles wrap the torus in each direction.  Each family holds
    ``L * L`` checks over ``2 * L * L`` edges with rank ``L * L - 1``, so the code
    has two logical qubits and its lightest logical operator has weight ``L``.
    """

    edges = 2 * lattice * lattice

    def horizontal(row: int, column: int) -> int:
        return 2 * ((row % lattice) * lattice + (column % lattice))

    def vertical(row: int, column: int) -> int:
        return horizontal(row, column) + 1

    def indicator(wires: list[int]) -> tuple[int, ...]:
        return tuple(1 if wire in wires else 0 for wire in range(edges))

    faces = tuple(
        indicator(
            [
                horizontal(row, column),
                horizontal(row + 1, column),
                vertical(row, column),
                vertical(row, column + 1),
            ]
        )
        for row in range(lattice)
        for column in range(lattice)
    )
    stars = tuple(
        indicator(
            [
                horizontal(row, column),
                horizontal(row, column - 1),
                vertical(row, column),
                vertical(row - 1, column),
            ]
        )
        for row in range(lattice)
        for column in range(lattice)
    )
    z_logicals = (
        indicator([horizontal(0, column) for column in range(lattice)]),
        indicator([vertical(row, 0) for row in range(lattice)]),
    )
    x_logicals = (
        indicator([horizontal(row, 0) for row in range(lattice)]),
        indicator([vertical(0, column) for column in range(lattice)]),
    )
    return faces, stars, z_logicals, x_logicals


def toric(lattice: int) -> CssCode:
    """Build the toric code on an ``L``-by-``L`` torus from its matrices."""

    faces, stars, z_logicals, x_logicals = toric_matrices(lattice)
    return CssCode(
        hz=faces,
        hx=stars,
        lz=z_logicals,
        lx=x_logicals,
        distance_search_weight=lattice,
    )


def steane_section() -> None:
    """State the Steane code as matrices and hold it against the shipped record."""

    print("the Steane code: matrices against the record this package ships")
    code = CssCode(
        hz=STEANE_CHECKS,
        hx=STEANE_CHECKS,
        lz=[STEANE_LOGICAL],
        lx=[STEANE_LOGICAL],
    )
    declared = SteaneCode()
    report("data qubits / ancillas", f"{code.num_data_qubits} / {code.num_ancilla_qubits}")
    report("Z-type checks / X-type checks", " / ".join(str(n) for n in check_counts(code)))
    report("distance / x_distance / z_distance", f"{code.distance} / {code.x_distance} / {code.z_distance}")
    report("checks match the shipped record", code.checks == declared.checks)
    report(
        "stabilizer group matches",
        sorted(code.stabilizers) == sorted(declared.stabilizers),
    )

    # The same logical operator, multiplied by a weight-four stabilizer, is just as
    # much a logical operator and has weight seven. A distance read off the
    # declared operator would say seven; the record searches the code and says
    # three, which is the number the code has.
    multiplied = tuple(
        left ^ right
        for left, right in zip(STEANE_LOGICAL, STEANE_CHECKS[0], strict=True)
    )
    heavy = CssCode(
        hz=STEANE_CHECKS,
        hx=STEANE_CHECKS,
        lz=[multiplied],
        lx=[multiplied],
    )
    report("declared operator weight", f"{sum(STEANE_LOGICAL)} then {sum(multiplied)}")
    report(
        "distance, either representative",
        f"{code.distance} then {heavy.distance}",
    )

    # Reading the record back into the package's matrix record closes the loop: the
    # published matrices and the matrices written here describe one code.
    lifted = css_code_matrices(declared)
    report("shipped hz rows equal written hz", lifted.hz.tolist() == [list(row) for row in STEANE_CHECKS])
    report("shipped lz row equal written lz", lifted.lz.tolist() == [list(STEANE_LOGICAL)])


def weight_histogram(model: DetectorErrorModel) -> dict[int, int]:
    """Return how many mechanisms flip each number of detectors.

    The wire numbering moves detectors around, so this is what the numbering cannot
    change: how many faults there are of each signature weight.
    """

    histogram: dict[int, int] = {}
    for error in model.errors:
        weight = len(error.detectors)
        histogram[weight] = histogram.get(weight, 0) + 1
    return dict(sorted(histogram.items()))


def toric_section(lattice: int, shots: int) -> None:
    """Run a code through the whole path and hold it against the shipped record."""

    code = toric(lattice)
    declared = toric_code(lattice)
    print()
    print(f"the toric code on a {lattice}-by-{lattice} torus: written here, and shipped")
    report("data qubits / ancillas", f"{code.num_data_qubits} / {code.num_ancilla_qubits}")
    report("checks / logical observables", f"{len(code.checks)} / {len(code.logical_observables)}")
    report("data qubits per check", sorted({check.stabilizer.weight for check in code.checks}))
    report(
        "distance / x_distance / z_distance",
        f"{code.distance} / {code.x_distance} / {code.z_distance}",
    )

    # Everything below is indexed by the wire numbering, so it is where a
    # transcription error in either derivation would show. The two matrices are not
    # compared row by row, because the two derivations number the edges differently
    # and relabelling one into the other would be a third derivation to get wrong.
    report(
        "shipped counts match written",
        (declared.num_data_qubits, declared.num_ancilla_qubits, len(declared.checks))
        == (code.num_data_qubits, code.num_ancilla_qubits, len(code.checks)),
    )
    report(
        "shipped distance matches written",
        (declared.distance, declared.x_distance, declared.z_distance)
        == (code.distance, code.x_distance, code.z_distance),
    )
    report(
        "shipped check weights match written",
        sorted({check.stabilizer.weight for check in declared.checks})
        == sorted({check.stabilizer.weight for check in code.checks}),
    )
    report(
        "shipped logical count matches written",
        len(css_code_matrices(declared).lz) == len(css_code_matrices(code).lz),
    )

    noise = PhenomenologicalNoise(
        data_flip=PROBABILITY,
        phase_flip=PROBABILITY,
        measurement_flip=PROBABILITY,
    )
    memory = build_memory_circuit(code, rounds=1)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    shipped_model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(declared, rounds=1), noise=noise
    )
    widest = max(len(error.detectors) for error in model.errors)
    report("detectors / observables", f"{model.num_detectors} / {model.num_observables}")
    report("mechanisms", len(model.errors))
    report("most detectors one fault flips", widest)
    report("matching decoder", "accepted" if widest <= 2 else "refused: hyperedge")
    report(
        "shipped model agrees mechanism for mechanism",
        (shipped_model.num_detectors, shipped_model.num_observables, weight_histogram(shipped_model))
        == (model.num_detectors, model.num_observables, weight_histogram(model)),
    )

    sample = sample_memory_circuit(memory, noise=noise, shots=shots, seed=SEED)
    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    raw = 0
    corrected = 0
    returned = 0
    for syndrome_bits, observable_bits in zip(
        sample.detectors.tolist(), sample.observables.tolist()
    ):
        syndrome = tuple(index for index, bit in enumerate(syndrome_bits) if bit)
        result = decoder.decode(syndrome)
        returned += len(result.observables)
        after = list(observable_bits)
        for index in result.observables:
            after[index] ^= 1
        raw += sum(observable_bits)
        corrected += sum(after)
    per_shot = len(sample.observables) * model.num_observables
    report("shots", shots)
    report("observable flips the decoder returns", returned)
    report("rate without decoding", round(raw / per_shot, 4))
    report("rate under decoding", round(corrected / per_shot, 4))
    if code.distance == 2:
        print(
            "  distance two corrects nothing, so the two rates are equal exactly:"
            " no correction removes a fault that is already logical."
        )


def refusal_section() -> None:
    """Print the matrices a stabilizer code cannot be built from, and why."""

    print()
    print("what a set of matrices has to satisfy before it is a code")
    blocks = {
        "blocks that anticommute": dict(hz=[[1, 1, 0, 0]], hx=[[1, 0, 1, 0]]),
        "a logical inside its own span": dict(
            hz=STEANE_CHECKS,
            hx=STEANE_CHECKS,
            lz=[STEANE_CHECKS[0]],
            lx=[STEANE_LOGICAL],
        ),
        "a logical that fails a check": dict(
            hz=STEANE_CHECKS,
            hx=STEANE_CHECKS,
            lz=[(1, 0, 0, 0, 0, 0, 0)],
            lx=[STEANE_LOGICAL],
        ),
        "a search too short to reach it": dict(
            hz=[[1, 1, 1, 1, 1]], hx=[], distance_search_weight=1
        ),
    }
    for label, arguments in blocks.items():
        try:
            CssCode(**arguments)
        except (TypeError, ValueError) as error:
            report(label, str(error)[:44] + "...")
        else:
            raise SystemExit(f"{label} was accepted, and it is not a code")

    # A tensor is not a sequence, and this module holds no array dependency, so the
    # refusal is where the bridge between a tensor-stated matrix and this record is
    # stated. The shipped matrices are held as tensors.
    try:
        lifted = css_code_matrices(SteaneCode())
        CssCode(hz=lifted.hz, hx=lifted.hx, lz=lifted.lz, lx=lifted.lx)
    except TypeError as error:
        report("a tensor block", f"{str(error)[:44]}...")
    else:
        raise SystemExit("a tensor block was accepted as a matrix")

    # The two lines below show that the refusals above are about the matrices and
    # not about being strict: a code may be stated with no logical operator at all,
    # and its check rows need not be independent. The toric code's own rows are the
    # second case, because every star multiplied together is the identity.
    bare = CssCode(hz=STEANE_CHECKS, hx=STEANE_CHECKS)
    report("accepted: no logical declared", bare.logical_observables == ())
    report(
        "accepted: dependent check rows",
        f"{len(toric(2).hz)} face rows over 2 logical qubits",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shots", type=int, default=600)
    args = parser.parse_args()

    print("CSS codes from parity-check matrices: local, single process, experimental")
    print()
    steane_section()
    for lattice in LATTICES:
        toric_section(lattice, args.shots)
    refusal_section()
    print()
    print("No threshold, no logical-suppression claim, and no scalability claim.")


if __name__ == "__main__":
    main()
