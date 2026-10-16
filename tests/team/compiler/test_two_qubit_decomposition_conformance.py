"""Conformance of the two-qubit decomposition boundary against one shared contract.

An operator no target vocabulary names reaches the compiler as an instruction
carrying a matrix, and there are four answers to it: `synthesize_two_qubit` (the
Weyl/KAK route), the product-first `_local_replacement`, the dispatcher
`_matrix_replacement` that chooses between them, and `split_two_qubit_blocks`,
which re-spells a two-wire matrix as its two single-qubit factors. Each is a
separate implementation of the same task -- express an operator the target does not
speak natively -- so adding or rewriting one is acceptable only while it satisfies
the same postconditions. This module states those postconditions once and drives
every route through all of them, rather than asserting a different subset per
route.

The contract has six parts, and it holds for every route that answers:

* the vocabulary is published: no route invents an opcode, and no route emits an
  entangler outside `SUPERCONTROLLED_ENTANGLERS`;
* the reach is stated rather than assumed: with an entangler published every case
  in the family is served, and without one only a product of two single-qubit
  unitaries is, because that route needs a z-rotation and a pulse and no entangler
  at all;
* the observable is preserved up to exactly one global phase, so the instrument is
  the overlap and not the entrywise residual -- measured, and controlled below;
* the entangler count is a property of the unitary rather than of the spelling of
  the basis it is spent in, and it is the same for every route that answers;
* the routes are deterministic, do not mutate their input, and never choose a
  longer answer than the route they delegate to; and
* the refusals are typed: `None` for a basis this compiler cannot synthesize over,
  `ValueError` for an input that is not a two-qubit unitary, and
  `NativeGateLegalizationError` for an operator the target has no contract for.

**Three of the assertions are floors with a stated source, not counts.** The
products are drawn from a generator, and both product criteria decide with an
exact-equality comparison against `_MATCH_EPS`, so *how many* perturbed inputs are
accepted is a platform quantity, while the property being asserted -- that the
tolerance is a width whose middle is accepted and whose ends are refused -- is not.
The same reasoning applies to the frame sweep and to the near-product band. Every
floor below says where its margin comes from.

The checks are themselves controlled, because a check that has never been seen to
fail is not evidence. Four controls measure the distance between a correct and an
incorrect answer rather than asserting one: the overlap assertion re-run against
the entrywise residual, which cannot divide a global phase out; the perturbation
that walks a product criterion from acceptance to refusal across its own constant;
the undeclared-opcode arm of the split criterion, which the same matrix makes a
candidate; and the entangler-free arm, on which the product route is the only one
that answers.

The case family is generated rather than enumerated. Each point is a Weyl-chamber
coordinate conjugated by deliberate random single-qubit frames, because the cost of
a two-qubit unitary is a Weyl invariant while its numerics are not: a candidate
family built from "round" coordinates pushed a degenerate point into the Weyl
diagonalizer's refusal band, which is frame-dependent, and
`test_the_frame_does_not_change_the_count_or_the_refusal` re-measures the shipped
family against exactly that. This is the two-qubit-decomposition analogue of
`test_routing_conformance.py` and `test_optimization_pass_conformance.py`: the same
boundary-with-several-implementations problem, and the same answer. What
`tests/unit/test_compilation_*.py` and
`tests/hybrid_compiler/test_native_gate_legalization.py` cannot give it is that
those files state each route's own rule deeply and none of them state what the four
share.
"""

from __future__ import annotations

import cmath
import math
import random
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pytest
import torch

from flagquantum.compiler.native_gate_legalization import (
    NativeGateLegalizationError,
    _local_replacement,
    _matrix_replacement,
    legalize_native_gates,
)
from flagquantum.compiler.one_qubit_synthesis import (
    HALF_PI_PULSE_OPCODES,
    Z_ROTATION_OPCODES,
)
from flagquantum.compiler.two_qubit_optimization import (
    _MATCH_EPS,
    _split_instruction,
    split_two_qubit_blocks,
)
from flagquantum.compiler.two_qubit_synthesis import (
    _UNITARY_ATOL,
    SUPERCONTROLLED_ENTANGLERS,
    _BasisDecomposer,
    synthesize_two_qubit,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.core.target_capabilities import (
    CapabilityFact,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

#: The seed the pinned numbers in the comments below were measured at on the
#: authoring host. Nothing here is asserted to that host's last bit; see the floor
#: comments for the margin each number is asserted with.
_SEED = 20261006

#: The fixed basis every route is driven with. `x` is published and unused by both
#: routes, so a route that emitted it would be caught by the vocabulary check.
_NATIVE_BASE = ("x", "rz", "sx")
_Z_ROTATION = "rz"
_PULSE = "sx"

#: The six entries of the roster, read from the mapping rather than restated.
_ENTANGLERS = tuple(SUPERCONTROLLED_ENTANGLERS)

#: The two spellings the product factor route emits, taken from the module that
#: owns the choice so this file cannot drift from it.
_ONE_QUBIT = frozenset({Z_ROTATION_OPCODES[0], HALF_PI_PULSE_OPCODES[0]})

#: Every leaf name any route may emit for an undeclared input. `opaque` is the
#: input's own opcode, echoed by the fold route when it declines to rewrite.
_PUBLISHED = frozenset(_ENTANGLERS) | _ONE_QUBIT | {"opaque"}

#: The declared arity-2 unitary opcodes, read from the schema table so this file
#: cannot hold a second copy of the named two-wire vocabulary.
_DECLARED_TWO_WIRE = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.arity == 2 and schema.semantic_kind == "unitary"
    )
)


_NOW = datetime(2026, 10, 5, 8, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("two-qubit-decomposition:0",))
_SOURCE = FactSource(kind="conformance_test", ref="two-qubit-decomposition-evidence")

#: The Weyl-chamber interior coordinates the family is built from. The four `a`
#: values, three `b` values and three `c` values are all distinct, and the closest
#: pair anywhere in the family differs by 0.01 -- `math.pi / 4 - 0.01` against
#: `math.pi / 4` -- so no two points collide and none sits on a chamber wall.
_A_VALUES = (math.pi / 8, 0.22, 0.35, math.pi / 4 - 0.01)
_B_VALUES = (0.13, 0.29, 0.44)
_C_VALUES = (0.0, 0.26, -0.26)

#: The five boundary points, kept separate because they are the ones whose
#: entangler count sits at an end of the range: the identity, the three canonical
#: points below the chamber's first two coordinates, and the product.
_BOUNDARY_POINTS = (
    (math.pi / 4, math.pi / 4, 0.0),
    (math.pi / 4, 0.0, 0.0),
    (math.pi / 8, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (math.pi / 4, math.pi / 8, 0.0),
)

#: The 41 chamber points, in the order the family is built.
_POINTS = (
    tuple((a, b, c) for a in _A_VALUES for b in _B_VALUES for c in _C_VALUES)
    + _BOUNDARY_POINTS
)


def _snapshot(native: tuple[object, ...]) -> TargetCapabilitySnapshot:
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="two-qubit-decomposition-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="two-qubit-decomposition-environment",
        ),
        scope=_SCOPE,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=(
            CapabilityFact(
                name="gates.native",
                value=tuple(native),
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.DECLARED,
                source=_SOURCE,
            ),
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="two-qubit-decomposition-evidence",
                sha256="b" * 64,
                level=EvidenceLevel.BASIC,
                scope=_SCOPE,
            ),
        ),
    )


def _instruction_matrix(instruction: Instruction) -> torch.Tensor:
    width = 2 ** len(instruction.wires)
    matrix = gate_matrix(
        instruction, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
    )
    # A parameterized instruction comes back with a leading batch axis, so the
    # reshape is required rather than cosmetic.
    return matrix.detach().reshape(-1, width, width)[0]


_PAULI = {
    name: _instruction_matrix(Instruction(name, (0,))) for name in ("x", "y", "z")
}


def _expm_antihermitian(generator: torch.Tensor) -> torch.Tensor:
    """``exp(i * generator)`` for a Hermitian generator, by scaling and squaring.

    Written here because the suite must not import `numpy`, which is neither a
    runtime nor a CI dependency, while `torch` is. Generating the Weyl points
    rather than reading them from a table is what keeps this file from depending on
    one library's `expm`.
    """

    size = generator.shape[0]
    scale = 1.0
    norm = float(torch.sum(torch.abs(generator), dim=1).max())
    while norm / scale > 0.25:
        scale *= 2.0
    step = 1j * generator / scale
    result = torch.eye(size, dtype=torch.complex128)
    term = torch.eye(size, dtype=torch.complex128)
    for order in range(1, 60):
        term = term @ step / order
        result = result + term
    for _ in range(int(round(math.log2(scale)))):
        result = result @ result
    return result


def _weyl_element(a: float, b: float, c: float) -> torch.Tensor:
    generator = (
        a * torch.kron(_PAULI["x"], _PAULI["x"])
        + b * torch.kron(_PAULI["y"], _PAULI["y"])
        + c * torch.kron(_PAULI["z"], _PAULI["z"])
    )
    return _expm_antihermitian(generator)


def _random_su2(rng: random.Random) -> torch.Tensor:
    axis = torch.tensor([rng.gauss(0.0, 1.0) for _ in range(3)], dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    generator = angle * (
        axis[0] * _PAULI["x"] + axis[1] * _PAULI["y"] + axis[2] * _PAULI["z"]
    )
    return cmath.exp(1j * rng.uniform(0.0, 2.0 * math.pi)) * _expm_antihermitian(
        generator
    )


def _frame(unitary: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Conjugate by two independent random single-qubit frames, one per side."""

    left = torch.kron(_random_su2(rng), _random_su2(rng))
    right = torch.kron(_random_su2(rng), _random_su2(rng))
    return left @ unitary @ right


def _family(seed: int = _SEED) -> list[tuple[tuple[float, float, float], torch.Tensor]]:
    rng = random.Random(seed)
    return [(point, _frame(_weyl_element(*point), rng)) for point in _POINTS]


_FAMILY = _family()

#: One product stream, so every check and every control below sees the same inputs.
_PRODUCT_SEED = 4711


def _products(count: int, seed: int = _PRODUCT_SEED) -> list[torch.Tensor]:
    rng = random.Random(seed)
    return [torch.kron(_random_su2(rng), _random_su2(rng)) for _ in range(count)]


def _opaque(matrix: torch.Tensor, wires: tuple[int, ...] = (0, 1)) -> Instruction:
    """A two-wire instruction no schema declares, carrying `matrix`.

    `opaque` is undeclared, so `Instruction` accepts an arbitrary matrix on it and
    every route in this file is willing to look at it. This is the shape the whole
    boundary exists for: an operator no target vocabulary names.
    """

    return Instruction(
        "opaque",
        wires,
        matrix=[[complex(entry) for entry in row] for row in matrix.tolist()],
    )


def _synthesize(matrix: object, entangler: str) -> tuple[Instruction, ...] | None:
    """Drive the KAK route from a `torch` matrix or from a plain nested sequence."""

    rows = matrix.tolist() if hasattr(matrix, "tolist") else matrix
    return synthesize_two_qubit(
        [[complex(entry) for entry in row] for row in rows],
        qubits=(0, 1),
        entangler=entangler,
        z_rotation=_Z_ROTATION,
        pulse_opcode=_PULSE,
    )


def _dispatch(
    instruction: Instruction, entangler: str | None
) -> tuple[Instruction, ...] | None:
    return _matrix_replacement(
        instruction, z_rotation=_Z_ROTATION, pulse_opcode=_PULSE, entangler=entangler
    )


def _local(instruction: Instruction) -> tuple[Instruction, ...] | None:
    return _local_replacement(instruction, z_rotation=_Z_ROTATION, pulse_opcode=_PULSE)


def _split(
    instruction: Instruction, entangler: str | None
) -> tuple[Instruction, ...] | None:
    program = split_two_qubit_blocks(CircuitIR(2, (instruction,), dtype="complex128"))
    return tuple(program.instructions)


def _consume(
    instruction: Instruction, entangler: str | None
) -> tuple[Instruction, ...] | None:
    native = _NATIVE_BASE if entangler is None else (*_NATIVE_BASE, entangler)
    try:
        result = legalize_native_gates(
            CircuitIR(2, (instruction,), dtype="complex128"),
            snapshot=_snapshot(native),
            evaluated_at=_NOW,
        )
    except NativeGateLegalizationError:
        return None
    return tuple(result.program.instructions)


#: The five routes into the boundary, each normalized to
#: ``(instruction, entangler | None) -> leaves | None``. Only the first three need
#: an entangler; `split_two_qubit_blocks` folds a product and ignores it, and
#: `legalize_native_gates` asks the dispatcher.
_ROUTES: tuple[tuple[str, Callable[..., tuple[Instruction, ...] | None]], ...] = (
    (
        "synthesize_two_qubit",
        lambda item, entangler: (
            None if entangler is None else _synthesize(item.matrix, entangler)
        ),
    ),
    ("_local_replacement", lambda item, entangler: _local(item)),
    ("_matrix_replacement", _dispatch),
    ("split_two_qubit_blocks", _split),
    ("legalize_native_gates", _consume),
)

#: The routes whose subject is a whole non-product unitary. Each needs an
#: entangler, which is what makes the entangler-free arm the reach check.
_SYNTHESIS_ROUTES = (
    "synthesize_two_qubit",
    "_matrix_replacement",
    "legalize_native_gates",
)

#: The shape of the answer, per route and per entangler arm, over the family.
#: `split_two_qubit_blocks` folds rather than synthesizes, so it answers every arm
#: with a program that is unchanged except for a product.
_REACH: dict[str, dict[str | None, int]] = {
    "synthesize_two_qubit": dict.fromkeys(_ENTANGLERS, 41) | {None: 0},
    "_local_replacement": dict.fromkeys(_ENTANGLERS, 1) | {None: 1},
    "_matrix_replacement": dict.fromkeys(_ENTANGLERS, 41) | {None: 1},
    "split_two_qubit_blocks": dict.fromkeys(_ENTANGLERS, 41) | {None: 41},
    "legalize_native_gates": dict.fromkeys(_ENTANGLERS, 41) | {None: 1},
}


def _lifted(instruction: Instruction, n_wires: int = 2) -> torch.Tensor:
    """One instruction's unitary over the whole register, wire 0 first."""

    dimension = 2**n_wires
    local = _instruction_matrix(instruction)
    wires = tuple(instruction.wires)
    full = torch.zeros((dimension, dimension), dtype=torch.complex128)
    for row in range(dimension):
        bits_row = [(row >> (n_wires - 1 - wire)) & 1 for wire in range(n_wires)]
        for column in range(dimension):
            bits_column = [
                (column >> (n_wires - 1 - wire)) & 1 for wire in range(n_wires)
            ]
            if any(
                bits_row[wire] != bits_column[wire]
                for wire in range(n_wires)
                if wire not in wires
            ):
                continue
            local_row = 0
            local_column = 0
            for wire in wires:
                local_row = (local_row << 1) | bits_row[wire]
                local_column = (local_column << 1) | bits_column[wire]
            full[row][column] = local[local_row][local_column]
    return full


def _unitary(leaves: tuple[Instruction, ...]) -> torch.Tensor:
    total = torch.eye(4, dtype=torch.complex128)
    for instruction in leaves:
        total = _lifted(instruction) @ total
    return total


def _overlap(source: torch.Tensor, produced: torch.Tensor) -> float:
    """``|Tr(source^dagger produced)| / 4``: the phase-blind agreement.

    This is the instrument the contract is stated over, because a synthesis route is
    licensed to drop exactly one global phase and this is the only statistic here
    that cannot see one. `test_the_overlap_instrument_can_see_a_dropped_global_phase`
    is its control.
    """

    return float(torch.abs(torch.trace(source.conj().T @ produced)) / 4.0)


def _raw_gap(source: torch.Tensor, produced: torch.Tensor) -> float:
    return float(torch.max(torch.abs(produced - source)))


def _phase(source: torch.Tensor, produced: torch.Tensor) -> float:
    return cmath.phase(complex(torch.trace(source.conj().T @ produced) / 4.0))


def _arc_width(angles: list[float]) -> float:
    """The width of the shortest arc covering angles on the circle.

    A global phase lives on the circle, so a linear ``max - min`` over raw
    `cmath.phase` values wraps at +-pi and can report about 2*pi where the truth is
    about zero. Every phase assertion in this file uses this statistic.
    """

    wrapped = sorted(angle % (2.0 * math.pi) for angle in angles)
    gaps = [wrapped[index + 1] - wrapped[index] for index in range(len(wrapped) - 1)]
    gaps.append(2.0 * math.pi - (wrapped[-1] - wrapped[0]))
    return 2.0 * math.pi - max(gaps)


def _keys(leaves: tuple[Instruction, ...] | None):
    if leaves is None:
        return None
    return tuple((item.name, tuple(item.wires)) for item in leaves)


def _entanglers_in(leaves: tuple[Instruction, ...]) -> int:
    return sum(1 for item in leaves if item.name in _ENTANGLERS)


def _bumped(matrix: torch.Tensor, eps: float, row: int = 0, column: int = 0):
    """`matrix` with ``eps`` added to one entry; a non-product once ``eps`` is real."""

    bumped = matrix.clone()
    bumped[row][column] = bumped[row][column] + eps
    return bumped


def test_the_declared_arity_two_group_is_the_eleven_names_the_schema_table_holds() -> (
    None
):
    """The named vocabulary the split criterion is stated against.

    `_split_instruction` refuses a candidate whose opcode a schema declares, on the
    grounds that the named route already owns it. That rule is only as meaningful as
    the group it names, so the group is read from `operator_schema` rather than from
    a list here, and the count the measured table holds is pinned.
    """

    assert _DECLARED_TWO_WIRE == (
        "cphase",
        "crx",
        "cry",
        "crz",
        "cx",
        "cy",
        "cz",
        "rxx",
        "ryy",
        "rzz",
        "swap",
    )
    assert len(_DECLARED_TWO_WIRE) == 11


def test_a_declared_opcode_is_never_a_split_candidate_even_with_its_own_matrix() -> (
    None
):
    """The control for the schema rule, measured from both arms.

    The rule is exact-equality over the schema table, so the control is the same
    matrix handed to two opcodes: an undeclared one, which the criterion accepts, and
    each of the eleven declared ones, which it must refuse. Without the first arm a
    criterion that refused everything would pass.
    """

    products = _products(3)

    accepted = sum(1 for matrix in products if _split_instruction(_opaque(matrix)))
    assert accepted == 3

    refused = 0
    for name in _DECLARED_TWO_WIRE:
        schema = OPERATOR_SCHEMAS[name]
        params = dict.fromkeys(schema.parameters, 0.7)
        for matrix in products:
            carried = Instruction(
                name,
                (0, 1),
                params=params,
                matrix=[[complex(e) for e in row] for row in matrix.tolist()],
            )
            assert _split_instruction(carried) is None
            refused += 1
    assert refused == 11 * 3


def test_the_entangler_roster_is_exactly_the_supercontrolled_set() -> None:
    """The roster is checked against the class it claims to be, not restated.

    `_BasisDecomposer.__init__` raises for anything it cannot decompose, and its
    docstring states the acceptance condition: Weyl angles with ``a == pi/4`` and
    ``c == 0``. Every entry of the mapping is driven through that constructor, and
    then every declared two-wire opcode *absent* from the mapping is driven through
    it too and must be refused -- so the mapping is neither missing an entry it could
    accept nor carrying one it would reject.
    """

    for entangler, angle in SUPERCONTROLLED_ENTANGLERS.items():
        decomposer = _BasisDecomposer(entangler)
        assert decomposer.entangler == entangler
        assert decomposer.angle == angle
        assert math.isclose(decomposer.basis.a, math.pi / 4, rel_tol=1.0e-9)
        assert math.isclose(decomposer.basis.c, 0.0, abs_tol=1.0e-12)

    absent = [
        name for name in _DECLARED_TWO_WIRE if name not in SUPERCONTROLLED_ENTANGLERS
    ]
    assert absent == ["cphase", "crx", "cry", "crz", "swap"]
    for name in absent:
        with pytest.raises(ValueError):
            _BasisDecomposer(name)

    assert _ENTANGLERS == ("cx", "cz", "cy", "rzz", "rxx", "ryy")


def test_the_family_is_what_this_file_claims_it_is() -> None:
    """The generator's own shape, so the assertions over it cannot be vacuous.

    Forty-one distinct points, no duplicate, and no two coordinate values anywhere
    in the family closer than 0.01 -- `math.pi / 4 - 0.01` against `math.pi / 4`.
    This is what keeps the frame sweep meaningful: a family of one point would pass
    every other test in the file.
    """

    assert len(_POINTS) == 41
    assert len(set(_POINTS)) == 41
    assert [point for point, _ in _FAMILY] == list(_POINTS)

    coordinates = sorted({value for point in _POINTS for value in point})
    gaps = [
        following - current
        for current, following in zip(coordinates, coordinates[1:], strict=False)
    ]
    assert len(coordinates) == 11
    assert min(gaps) == pytest.approx(0.01, abs=1.0e-12)

    for _, matrix in _FAMILY:
        assert matrix.shape == (4, 4)
        assert (
            float(
                torch.max(
                    torch.abs(
                        matrix.conj().T @ matrix - torch.eye(4, dtype=torch.complex128)
                    )
                )
            )
            < 1.0e-12
        )


def test_every_route_speaks_only_published_opcodes() -> None:
    """The vocabulary part, over every route, every entangler arm, every case."""

    emitted: set[str] = set()
    for name, route in _ROUTES:
        for entangler in (*_ENTANGLERS, None):
            for _, matrix in _FAMILY:
                leaves = route(_opaque(matrix), entangler)
                if leaves is None:
                    continue
                foreign = {item.name for item in leaves} - _PUBLISHED
                assert not foreign, f"{name} emitted {sorted(foreign)}"
                emitted |= {item.name for item in leaves}

    # `opaque` is in the union because the fold route answers a non-product by
    # handing it back untouched, which is an answer and not a rewrite.
    assert emitted == {"rz", "sx", "opaque", *SUPERCONTROLLED_ENTANGLERS}
    assert emitted == _PUBLISHED


def test_the_matrix_gives_every_route_real_work() -> None:
    """The family is not vacuous, and neither is any route's part in it.

    The generator is what keeps the rest of the file honest, so it is checked first:
    all four entangler counts 0 through 3 are reached, every entangler arm is
    answered, and each of the five routes produces a non-empty answer somewhere.
    """

    counts = {
        _entanglers_in(leaves)
        for _, matrix in _FAMILY
        if (leaves := _synthesize(matrix, "cx")) is not None
    }
    assert counts == {0, 1, 2, 3}

    for name, route in _ROUTES:
        served = 0
        for entangler in (*_ENTANGLERS, None):
            for _, matrix in _FAMILY:
                if route(_opaque(matrix), entangler):
                    served += 1
        assert served > 0, name


def test_the_reach_of_every_route_is_exactly_the_arm_it_is_licensed_for() -> None:
    """The reach part, pinned as a count per route per entangler arm.

    The measured shape: the three synthesis routes answer all 41 cases in each of
    the six entangler arms and none of them without one; the product route answers
    exactly the family's one product, with or without an entangler; and the fold
    route answers everything, because an unchanged program is an answer. The
    entangler-free arm is the control for the other two: without it a dispatcher
    that simply refused everything would pass the middle rows.
    """

    for name, route in _ROUTES:
        measured = {}
        for entangler in (*_ENTANGLERS, None):
            measured[entangler] = sum(
                1 for _, matrix in _FAMILY if route(_opaque(matrix), entangler)
            )
        assert measured == _REACH[name], name


def test_every_route_leaves_the_source_unitary_alone_up_to_one_phase() -> None:
    """The observable part, per route per case rather than as one worst case.

    A single worst case can be met by one route while another is wrong, so the check
    is per case: the overlap must be 1 to within the module's own `_UNITARY_ATOL`,
    which is also the constant its own unit tests compare matrices at. The worst
    value measured here is 1.78e-14, more than four orders of magnitude below the
    bound; the margin is deliberately left that wide because this file was measured
    on one libm while CI runs on another.
    """

    worst = 0.0
    compared = 0
    for name, route in _ROUTES:
        for entangler in (*_ENTANGLERS, None):
            for _, matrix in _FAMILY:
                leaves = route(_opaque(matrix), entangler)
                if leaves is None:
                    continue
                gap = abs(1.0 - _overlap(matrix, _unitary(leaves)))
                assert gap < _UNITARY_ATOL, f"{name} entangler={entangler} gap={gap!r}"
                worst = max(worst, gap)
                compared += 1

    assert compared == sum(sum(row.values()) for row in _REACH.values())
    assert worst < 1.0e-13


def test_the_overlap_instrument_can_see_a_dropped_global_phase() -> None:
    """The control for the observable check: what the instrument must NOT miss.

    The contract's observable is "up to one global phase", so the check is an
    overlap. The control is the same answer measured with the entrywise residual,
    which cannot divide a phase out: measured over the family at entangler `cx` the
    raw gap reaches 1.81 while the overlap is 1 to within `_UNITARY_ATOL` and every
    one of the 41 answers carries a phase above 1e-3. Without this the overlap
    assertion would be indistinguishable from asserting a matrix equality.
    """

    worst_overlap = 0.0
    smallest_raw = math.inf
    phased = 0
    for _, matrix in _FAMILY:
        leaves = _synthesize(matrix, "cx")
        assert leaves is not None
        produced = _unitary(leaves)
        worst_overlap = max(worst_overlap, abs(1.0 - _overlap(matrix, produced)))
        smallest_raw = min(smallest_raw, _raw_gap(matrix, produced))
        phased += abs(_phase(matrix, produced)) > 1.0e-3

    assert worst_overlap < _UNITARY_ATOL
    assert smallest_raw > 1.0e-3
    assert phased == len(_POINTS) == 41


def test_the_global_phase_is_real_and_entangler_dependent() -> None:
    """Why the phase assertion is an arc and not an equality.

    Two measured facts, both of which a naive assertion gets wrong. First, the phase
    of one input genuinely depends on which entangler the answer is spent in: for one
    case the six phases sit at deltas of
    ``[0, -0.75, -1.0, -0.25, -0.75, 0.25] * pi`` from the `cx` answer, so equality
    across entanglers is false. Second, over the family the phase spans an arc of
    5.712 rad at `cx` -- most of the circle -- while the same statistic for the fold
    route is exactly 0. Both are asserted, because a suite that only pinned the
    second would not know the first.

    The per-case floor is a quarter turn rather than a half turn because the spread
    is read off a grid: every difference is a whole number of quarter turns, while
    which multiple a case lands on is not portable. The first version of this test
    asserted a half turn and passed here while failing on all three CI interpreters,
    which is the same class of defect as asserting a count where the width of a
    transition is the property -- the number moved, the structure did not.
    """

    silent = []
    spreads = []
    for point, matrix in _FAMILY:
        angles = []
        for entangler in _ENTANGLERS:
            leaves = _synthesize(matrix, entangler)
            assert leaves is not None
            angles.append(_phase(matrix, _unitary(leaves)))
        base = angles[0]
        spread = max(abs(angle - base) for angle in angles)
        # A coarse guard, an order of magnitude above a quarter turn: the widest
        # per-case arc measured over the shipped family is 1.25*pi.
        assert _arc_width(angles) <= 1.5 * math.pi
        if spread < 1.0e-9:
            silent.append(point)
            continue
        # The spread is a whole number of quarter turns, and that -- not which
        # multiple -- is the portable statement. Over 21 independently framed
        # families (861 cases, 5166 entangler arms) the largest distance from a
        # whole number of quarter turns is 8.9e-16 of one, so the tolerance is a
        # thousand-fold margin and a thousandth of the grid. The multiple itself
        # is not portable: one point of this family spreads by two quarter turns
        # on arm64 and by one on x86-64, because the phase it is read from sits on
        # a grid boundary and libm rounds it either way. A floor of a quarter turn
        # therefore holds on both, and a floor of a half turn does not.
        quarters = spread / (math.pi / 4.0)
        assert abs(quarters - round(quarters)) < 1.0e-12
        assert round(quarters) >= 1
        spreads.append(round(quarters))
    # The identity is the one input with no phase to differ about, and it is the
    # only silent case: 40 of 41 spread by at least a quarter turn across the six
    # entanglers, so the entangler dependence is the rule and not one outlier.
    assert silent == [(0.0, 0.0, 0.0)]
    # And the spread is not one number standing in for the family: it takes at
    # least four values, measured six, so "the phase depends on the entangler" is
    # asserted of the family and not only of its extremes.
    assert len(set(spreads)) >= 4

    family_arc = _arc_width(
        [
            _phase(matrix, _unitary(leaves))
            for _, matrix in _FAMILY
            if (leaves := _synthesize(matrix, "cx")) is not None
        ]
    )
    assert family_arc > math.pi
    assert family_arc < 2.0 * math.pi

    folded = _arc_width(
        [
            _phase(matrix, _unitary(leaves))
            for _, matrix in _FAMILY
            if (leaves := _split(_opaque(matrix), "cx")) is not None
        ]
    )
    assert folded == 0.0


def test_the_fold_route_is_raw_exact_where_the_synthesis_routes_are_not() -> None:
    """The one route that records the phase, and the three that record none.

    `two_qubit_optimization._product_factors` reads the Kronecker scale into the
    emitted factors, which its own docstring states is why that route records no
    phase; `split_two_qubit_blocks` calls it. Measured over the family, the fold
    route's raw residual is 1.24e-16 and its phase is exactly 0, while every other
    route's raw residual is 1.44 and above. That is why the two kinds of route need
    two verification rules -- an overlap rule for the synthesis routes and an
    exact-equality rule for the fold -- and it is asserted here rather than left to
    the reader of the two docstrings.
    """

    folded_worst = 0.0
    folded_phases = []
    for _, matrix in _FAMILY:
        leaves = _split(_opaque(matrix), "cx")
        assert leaves is not None
        produced = _unitary(leaves)
        folded_worst = max(folded_worst, _raw_gap(matrix, produced))
        folded_phases.append(_phase(matrix, produced))
    assert folded_worst < 1.0e-14
    assert max(abs(angle) for angle in folded_phases) < 1.0e-15

    for name in _SYNTHESIS_ROUTES:
        route = dict(_ROUTES)[name]
        raw = 0.0
        for _, matrix in _FAMILY:
            leaves = route(_opaque(matrix), "cx")
            assert leaves is not None
            raw = max(raw, _raw_gap(matrix, _unitary(leaves)))
        assert raw > 1.0, name


def test_the_product_criterion_is_phase_exact_before_the_one_qubit_route_spends_it() -> (
    None
):
    """The two conventions, on the same input, measured apart.

    `_split_instruction` returns two single-qubit factors whose Kronecker product is
    the source matrix with no phase at all -- the factor on wire 0 first, measured at
    3.72e-16 entrywise. The named route those factors are then handed to is
    `synthesize_one_qubit_matrix`, which documents that it returns the form "up to
    one global phase, which FlagQuantum IR has no field to record"; measured at up to
    2.54 rad here, and on all six products in the stream. Both halves are asserted
    because a check that only saw the exact half would not know the other convention
    exists.
    """

    worst_gap = 0.0
    worst_phase = 0.0
    for matrix in _products(12):
        halves = _split_instruction(_opaque(matrix))
        assert halves is not None
        assert [tuple(half.wires) for half in halves] == [(0,), (1,)]
        rebuilt = torch.kron(
            _instruction_matrix(halves[0]), _instruction_matrix(halves[1])
        )
        worst_gap = max(worst_gap, _raw_gap(matrix, rebuilt))
        worst_phase = max(worst_phase, abs(_phase(matrix, rebuilt)))
    assert worst_gap < 1.0e-15
    assert worst_phase < 1.0e-15

    spent = 0
    worst_spent = 0.0
    for matrix in _products(6):
        leaves = _local(_opaque(matrix))
        assert leaves is not None
        produced = _unitary(leaves)
        assert _overlap(matrix, produced) > 1.0 - _UNITARY_ATOL
        dropped = abs(_phase(matrix, produced))
        assert dropped > 1.0e-6
        worst_spent = max(worst_spent, dropped)
        spent += 1
    assert spent == 6
    assert worst_spent < math.pi


def test_every_synthesis_route_agrees_leaf_for_leaf_with_the_others() -> None:
    """The strongest statement in the file: the routes are the same answer.

    The dispatcher asks the product route first and falls back to the synthesis
    route, and the consumer asks the dispatcher; measured over the family, all three
    produce an identical sequence of names and wires -- 41 of 41 in each of the six
    entangler arms. That is what makes the consumer a pass-through rather than a
    second implementation, and it is asserted rather than inferred from the call
    graph.
    """

    routes = [(name, dict(_ROUTES)[name]) for name in _SYNTHESIS_ROUTES]
    compared = 0
    for entangler in _ENTANGLERS:
        for _, matrix in _FAMILY:
            instruction = _opaque(matrix)
            expected = _keys(_synthesize(matrix, entangler))
            assert expected is not None
            for name, route in routes:
                assert _keys(route(instruction, entangler)) == expected, name
            compared += 1
    assert compared == len(_ENTANGLERS) * len(_POINTS)


def test_the_entangler_count_is_the_input_rather_than_the_spelling() -> None:
    """Cost invariance, and the distribution that makes it non-vacuous.

    The entangler count of a two-qubit unitary is a property of the unitary, so a
    target publishing `rzz` instead of `cx` must pay the same count. Measured across
    the family the count is identical for all six entanglers and every route that
    answers, and its distribution over the family is `{0: 1, 1: 1, 2: 15, 3: 24}` --
    24 of 41 cases at the maximum, so the assertion cannot be met by a family that
    only ever needs one entangler. The fold route's own count is 0 everywhere, which
    is the control: it never spends an entangler, so a count of 0 is a real answer
    rather than a sign the counter is broken.
    """

    for name, route in _ROUTES:
        for _, matrix in _FAMILY:
            counts = set()
            for entangler in _ENTANGLERS:
                leaves = route(_opaque(matrix), entangler)
                if leaves is None:
                    continue
                counts.add(_entanglers_in(leaves))
            assert len(counts) <= 1, f"{name} disagreed across entanglers"

    for name in (*_SYNTHESIS_ROUTES, "_matrix_replacement"):
        distribution: dict[int, int] = {}
        for _, matrix in _FAMILY:
            leaves = dict(_ROUTES)[name](_opaque(matrix), "cx")
            assert leaves is not None
            count = _entanglers_in(leaves)
            distribution[count] = distribution.get(count, 0) + 1
        assert distribution == {0: 1, 1: 1, 2: 15, 3: 24}, name

    folded = {
        _entanglers_in(leaves)
        for _, matrix in _FAMILY
        if (leaves := _split(_opaque(matrix), "cx")) is not None
    }
    assert folded == {0}


def test_the_routes_are_deterministic_and_leave_their_input_alone() -> None:
    """Two calls, one answer; and the input instruction survives both."""

    for name, route in _ROUTES:
        for entangler in ("cx", "rzz"):
            for _, matrix in _FAMILY[:12]:
                instruction = _opaque(matrix)
                before = [[complex(e) for e in row] for row in instruction.matrix]
                assert _keys(route(instruction, entangler)) == _keys(
                    route(instruction, entangler)
                ), name
                after = [[complex(e) for e in row] for row in instruction.matrix]
                assert after == before, f"{name} mutated its input"


def test_the_product_tolerance_is_a_width_rather_than_a_boundary() -> None:
    """The control for the product criterion, measured from both ends and the middle.

    `_product_factors` and `_split_instruction` decide by exact comparison against
    `_MATCH_EPS`, so the honest assertion is that the middle of the transition is
    accepted and both ends are refused -- not a count, which is a platform quantity.
    Measured on `kron(su2, su2)` with `eps` added to one entry: 40 of 40 accepted at
    `eps <= 1e-12`, 36 of 40 at `1.1e-12`, 24 of 40 at `1.5e-12`, and 0 of 40 at
    `eps >= 2e-12`. Both criteria are measured at every arm, so a criterion that had
    drifted from the other would fail here.
    """

    assert _MATCH_EPS == 1e-12
    products = _products(40)

    for eps in (0.0, 5.0e-13, 1.0e-12):
        accepted = sum(
            1
            for matrix in products
            if _dispatch(_opaque(_bumped(matrix, eps)), None) is not None
            and _split_instruction(_opaque(_bumped(matrix, eps))) is not None
        )
        assert accepted == 40, eps

    for eps in (2.0e-12, 1.0e-11):
        refused = sum(
            1
            for matrix in products
            if _dispatch(_opaque(_bumped(matrix, eps)), None) is None
            and _split_instruction(_opaque(_bumped(matrix, eps))) is None
        )
        assert refused == 40, eps

    # Between the two measured arms the transition is genuinely gradual, which is
    # what makes the arms above a width rather than a boundary. The count is
    # asserted as strictly-between and never as a number, because it is a property
    # of this libm's rounding of the comparison.
    split = sum(
        1
        for matrix in products
        if _dispatch(_opaque(_bumped(matrix, 1.25e-12)), None) is not None
    )
    assert 0 < split < 40


def test_the_product_route_reaches_what_the_synthesis_route_refuses_without_an_entangler() -> (
    None
):
    """The reach part, and the reason the product route exists at all.

    A product of two single-qubit unitaries needs no entangler, so on a target
    publishing none the dispatcher must still answer it -- measured at 20 of 20 --
    while the synthesis route has nothing to say about any of them. The control is
    the non-product arm under the same entangler-free snapshot, which the dispatcher
    must refuse: without it, a dispatcher that answered everything would pass.
    """

    products = _products(20, seed=11)
    others = [
        _frame(_weyl_element(0.31 + 0.01 * index, 0.27, 0.19), random.Random(index))
        for index in range(20)
    ]

    for matrix in products:
        leaves = _dispatch(_opaque(matrix), None)
        assert leaves is not None
        assert _entanglers_in(leaves) == 0
        assert abs(1.0 - _overlap(matrix, _unitary(leaves))) < _UNITARY_ATOL

    for matrix in others:
        assert _dispatch(_opaque(matrix), None) is None
        leaves = _dispatch(_opaque(matrix), "cx")
        assert leaves is not None
        assert _entanglers_in(leaves) > 0
        assert abs(1.0 - _overlap(matrix, _unitary(leaves))) < _UNITARY_ATOL


def test_the_near_product_band_is_refused_by_both_routes() -> None:
    """The band between the two criteria, and the shape of the refusal.

    Past the product criterion's tolerance a near-product is no longer a product,
    and the synthesis route's Weyl diagonalizer refuses it as well: measured with
    `eps` added to a diagonal entry, 20 of 20 come back with zero entanglers at
    `eps <= 1e-13` and 20 of 20 raise at `eps >= 1e-10`. Above 1e-9 the input is no
    longer unitary and the refusal moves to that check instead, so the band is
    narrow and is what it is.

    **The refusal is not wrapped.** The 3-wire arm below reaches the caller as
    `NativeGateLegalizationError`, but this arm reaches it as the synthesis layer's
    own `ValueError`, unlike every other failure in `legalize_native_gates`. That is
    a diagnosed limitation reported to the Compiler owner, not a behaviour this
    suite endorses: when the consumer starts wrapping it, this is the test that must
    change, and this docstring is why.
    """

    products = _products(20, seed=101)

    counts = set()
    for matrix in products:
        leaves = _synthesize(_bumped(matrix, 1.0e-13, 1, 1), "cx")
        assert leaves is not None
        counts.add(_entanglers_in(leaves))
    assert counts == {0}

    for eps in (1.0e-10, 1.0e-9):
        for matrix in products:
            with pytest.raises(ValueError):
                _synthesize(_bumped(matrix, eps, 1, 1), "cx")
            # Both product criteria have given up by here as well, so the band is
            # refused by the pair rather than only by the synthesis route.
            assert _split_instruction(_opaque(_bumped(matrix, eps, 1, 1))) is None
            assert _dispatch(_opaque(_bumped(matrix, eps, 1, 1)), None) is None

    # The dispatcher serves the whole near-product band up to the product
    # criterion's own tolerance, which is the reason it is asked before the
    # synthesis route.
    for eps in (0.0, 1.0e-13, 5.0e-13, 1.0e-12):
        served = sum(
            1
            for matrix in products
            if _dispatch(_opaque(_bumped(matrix, eps, 1, 1)), "cx") is not None
        )
        assert served == 20, eps

    # Past it, the synthesis layer's refusal escapes the dispatcher unwrapped.
    with pytest.raises(ValueError, match="diagonalize M2"):
        _dispatch(_opaque(_bumped(products[0], 1.0e-10, 1, 1)), "cx")


def test_the_synthesis_refusal_surface_is_exactly_the_documented_one() -> None:
    """Every arm of `synthesize_two_qubit`'s own docstring, driven by hand.

    Two arms: a basis this module cannot synthesize over returns `None`, and an
    input that is not a two-qubit unitary raises `ValueError`. The control is the
    first row of each -- the same call with a published entangler and a valid matrix
    answers -- so a route that refused everything would fail this test.
    """

    identity = torch.eye(4, dtype=torch.complex128).tolist()
    assert (
        synthesize_two_qubit(
            identity,
            qubits=(0, 1),
            entangler="cx",
            z_rotation=_Z_ROTATION,
            pulse_opcode=_PULSE,
        )
        == ()
    )

    for kwargs in (
        dict(entangler="swap", z_rotation=_Z_ROTATION, pulse_opcode=_PULSE),
        dict(entangler="cx", z_rotation="rx", pulse_opcode=_PULSE),
        dict(entangler="cx", z_rotation=_Z_ROTATION, pulse_opcode="h"),
        dict(entangler="cx", z_rotation="rz", pulse_opcode=None),
    ):
        assert synthesize_two_qubit(identity, qubits=(0, 1), **kwargs) is None, kwargs

    for wires, message in (
        ((0,), "qubits must name exactly two qubits"),
        ((0, 1, 2), "qubits must name exactly two qubits"),
        ((0, 0), "qubits must name two distinct qubits"),
        ((-1, 0), "qubits must be non-negative integers"),
    ):
        with pytest.raises(ValueError, match=message):
            synthesize_two_qubit(
                identity,
                qubits=wires,
                entangler="cx",
                z_rotation=_Z_ROTATION,
                pulse_opcode=_PULSE,
            )

    with pytest.raises(ValueError, match="matrix must have exactly 4 entries"):
        synthesize_two_qubit(
            torch.eye(3, dtype=torch.complex128).tolist(),
            qubits=(0, 1),
            entangler="cx",
            z_rotation=_Z_ROTATION,
        )
    with pytest.raises(ValueError, match="matrix must be unitary"):
        synthesize_two_qubit(
            torch.diag(
                torch.tensor([2.0, 1.0, 1.0, 1.0], dtype=torch.complex128)
            ).tolist(),
            qubits=(0, 1),
            entangler="cx",
            z_rotation=_Z_ROTATION,
        )


def test_the_product_route_needs_a_basis_where_the_synthesis_route_needs_an_entangler() -> (
    None
):
    """The two preconditions, each asserted on the arm where it is the blocker."""

    matrix = _products(1)[0]
    instruction = _opaque(matrix)
    assert (
        _matrix_replacement(
            instruction, z_rotation=None, pulse_opcode=_PULSE, entangler="cx"
        )
        is None
    )
    assert (
        _matrix_replacement(
            instruction, z_rotation=_Z_ROTATION, pulse_opcode=None, entangler="cx"
        )
        is None
    )

    non_product = _frame(_weyl_element(0.3, 0.35, 0.22), random.Random(1))
    assert _local(_opaque(non_product)) is None
    assert _synthesize(non_product, "cx") is not None
    assert _dispatch(_opaque(non_product), None) is None


def test_the_product_answer_is_never_longer_than_the_one_it_replaces() -> None:
    """The dispatcher's ordering claim, pinned to what is measured.

    `_matrix_replacement` asks the product route first because that route "needs a
    z-rotation and a pulse and no entangler at all". Its docstring also cites a
    benchmark for the stronger claim that the product answer is strictly shorter on
    some inputs; that stronger form is not reproducible from this file's inputs,
    where the two answers are equal on all 21 comparable products, so what is
    asserted is the ordering claim itself plus the extra reach without an entangler.
    """

    cases = [matrix for matrix in _products(20, seed=11)] + [
        matrix for _, matrix in _FAMILY
    ]
    for entangler in (*_ENTANGLERS, None):
        equal = 0
        product_only = 0
        for matrix in cases:
            instruction = _opaque(matrix)
            product = _local(instruction)
            whole = None if entangler is None else _synthesize(matrix, entangler)
            if product is not None and whole is not None:
                assert len(product) <= len(whole)
                assert _keys(product) == _keys(whole)
                equal += len(product) == len(whole)
            elif product is not None:
                product_only += 1
        if entangler is None:
            assert (equal, product_only) == (0, 21)
        else:
            assert (equal, product_only) == (21, 0)


def test_the_consumer_refuses_an_opaque_matrix_with_no_contract_and_honours_its_cap() -> (
    None
):
    """The consumer's failure arm and its one budget.

    An opaque matrix on three wires has no native contract and is refused as a
    `NativeGateLegalizationError` carrying the opcode, rather than as whatever the
    layer below happens to raise. The budget is checked at the exact number of added
    operations: the arm below it must refuse and the arm at it must succeed, because
    a cap that is never reached is not a measured cap.
    """

    three_wire = Instruction(
        "opaque",
        (0, 1, 2),
        matrix=[
            [complex(1.0 if row == column else 0.0) for column in range(8)]
            for row in range(8)
        ],
    )
    with pytest.raises(NativeGateLegalizationError, match="has no native contract"):
        legalize_native_gates(
            CircuitIR(3, (three_wire,), dtype="complex128"),
            snapshot=_snapshot(_NATIVE_BASE),
            evaluated_at=_NOW,
        )

    instruction = _opaque(_FAMILY[0][1])
    added = len(_dispatch(instruction, "cx") or ()) - 1
    assert added == 28

    with pytest.raises(NativeGateLegalizationError, match="max_added_operations"):
        legalize_native_gates(
            CircuitIR(2, (instruction,), dtype="complex128"),
            snapshot=_snapshot((*_NATIVE_BASE, "cx")),
            evaluated_at=_NOW,
            max_added_operations=added - 1,
        )
    result = legalize_native_gates(
        CircuitIR(2, (instruction,), dtype="complex128"),
        snapshot=_snapshot((*_NATIVE_BASE, "cx")),
        evaluated_at=_NOW,
        max_added_operations=added,
    )
    assert len(result.program.instructions) == added + 1


def test_the_named_route_reaches_the_declared_group_and_the_sparse_one_does_not() -> (
    None
):
    """Basis reach, measured through the consumer for every declared opcode.

    A basis publishing an entangler reaches every declared arity-2 unitary -- 11 of
    11 for both `cx` bases measured, one of them naming no `sx` at all -- while an
    `rzz` basis reaches only the three rotation entanglers, because `rzz` is not a
    universal entangler and the basis publishes no single-qubit set that spans the
    gap. The sparse arms are the control for the wide ones: without them, a matcher
    that accepted everything would pass.
    """

    reach = {}
    for label, native in {
        "x,sx,rz,cx": (*_NATIVE_BASE, "cx"),
        "h,s,rz,cx": ("h", "s", "rz", "cx"),
        "x,sx,rz,rzz": (*_NATIVE_BASE, "rzz"),
        "x,sx,rz": _NATIVE_BASE,
    }.items():
        reached = []
        for name in _DECLARED_TWO_WIRE:
            schema = OPERATOR_SCHEMAS[name]
            params = dict.fromkeys(schema.parameters, 0.7)
            program = CircuitIR(
                2, (Instruction(name, (0, 1), params=params),), dtype="complex128"
            )
            try:
                result = legalize_native_gates(
                    program, snapshot=_snapshot(native), evaluated_at=_NOW
                )
            except NativeGateLegalizationError:
                continue
            foreign = {item.name for item in result.program.instructions} - set(native)
            if not foreign:
                reached.append(name)
        reach[label] = reached

    assert reach["x,sx,rz,cx"] == list(_DECLARED_TWO_WIRE)
    assert reach["h,s,rz,cx"] == list(_DECLARED_TWO_WIRE)
    assert reach["x,sx,rz,rzz"] == ["rxx", "ryy", "rzz"]
    assert reach["x,sx,rz"] == []


def test_the_frame_does_not_change_the_count_or_the_refusal() -> None:
    """Why this family and not a grid of round coordinates.

    The cost of a two-qubit unitary is a Weyl invariant and its numerics are not, so
    each point is dressed in ten independent random frames and the count must not
    move. A candidate family built from "round" coordinates failed exactly here: the
    deliberate frames pushed a degenerate point into the Weyl diagonalizer's refusal
    band, whose tolerance sits below that class's intrinsic residual. The shipped
    family measures 0 of 41 unstable and 0 of 41 refused, and the frame index is the
    only thing that differs between the ten arms -- so this is a control on the
    family, not a re-run of the check above.
    """

    refused = []
    unstable = []
    for point in _POINTS:
        counts = set()
        for frame in range(10):
            matrix = _frame(_weyl_element(*point), random.Random(frame))
            try:
                leaves = _synthesize(matrix, "cx")
            except ValueError:
                refused.append((point, frame))
                continue
            assert leaves is not None
            counts.add(_entanglers_in(leaves))
        if len(counts) > 1:
            unstable.append((point, counts))
    assert not refused, refused
    assert not unstable, unstable


def test_the_identity_is_the_empty_answer_where_the_basis_allows_one() -> None:
    """The degenerate end of the family, where three routes agree on nothing.

    An empty tuple is `synthesize_two_qubit`'s correct answer for the two-wire
    identity and the dispatcher and the consumer inherit it, while the fold route
    answers with the two one-wire identity halves it always folds a product into.
    Both are answers; the point of asserting them together is that the empty tuple is
    not a refusal.
    """

    identity = torch.eye(4, dtype=torch.complex128)
    instruction = _opaque(identity)

    assert _synthesize(identity, "cx") == ()
    assert _dispatch(instruction, "cx") == ()
    assert _consume(instruction, "cx") == ()
    assert _dispatch(instruction, None) == ()
    assert _consume(instruction, None) == ()

    halves = _split(instruction, "cx")
    assert halves is not None
    assert len(halves) == 2
    assert {tuple(half.wires) for half in halves} == {(0,), (1,)}
    # Compared entrywise rather than through `pytest.approx`, which reaches for
    # numpy to compare a tensor and therefore answers differently in the CI lanes
    # that install no numpy than it does here. The bound is a guard rather than a
    # measurement: both halves are exact 2x2 identities, so the reconstruction is
    # exact and the gap measured here is exactly 0.
    assert _raw_gap(_unitary(halves), identity) < 1.0e-15


def test_the_one_qubit_vocabulary_is_read_from_the_module_that_owns_it() -> None:
    """The published leaf names, so the vocabulary check cannot be a tautology.

    `_PUBLISHED` is assembled from three modules rather than typed out, and this test
    pins what was read: the product route's own preferred spellings are the first
    entry of each tuple, and the family exercises exactly the two of them plus the
    entanglers.
    """

    assert Z_ROTATION_OPCODES[0] == "rz"
    assert HALF_PI_PULSE_OPCODES[0] == "sx"
    assert {"rz", "sx"} == _ONE_QUBIT
    assert {"rz", "sx", "opaque", *SUPERCONTROLLED_ENTANGLERS} == _PUBLISHED

    emitted = {
        item.name
        for _, matrix in _FAMILY
        for entangler in _ENTANGLERS
        if (leaves := _synthesize(matrix, entangler)) is not None
        for item in leaves
    }
    assert emitted == {"rz", "sx", *SUPERCONTROLLED_ENTANGLERS}
