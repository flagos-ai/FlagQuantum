"""Clifford stabilizer sampling backed by the optional Stim engine.

A stabilizer circuit conjugates a Pauli group instead of evolving an amplitude
vector, so its state is a tableau whose size grows quadratically with the wire
count rather than exponentially. That is the whole reason a stabilizer regime
exists: circuits made only of Clifford gates reach wire counts no amplitude
store can hold, and they answer exactly the question a fault-tolerance workflow
asks -- what do the measurements read out.

What this module owns
---------------------
One conversion and four entry points:

- translating validated FlagQuantum IR into the engine's circuit form, refusing
  every instruction this engine cannot represent,
- sampling measurement outcomes for a requested wire list,
- expanding a Pauli string through the circuit and reading it against the zero
  state, which is the stabilizer-basis readout a planned expectation request is
  served with,
- sampling every measurement a Clifford program records when that program
  carries positioned noise channels, and
- surveying what that sampling route makes of a program, without running it, so
  a caller can decide between the stabilizer regime and a dense one before
  spending a dense state on the question.

The positioned-noise entry point exists because a detector error model is defined
between syndrome rounds rather than at a named gate: quantum error correction
places its data error at a round boundary and its measurement error immediately
before one check's readout, and the program is the only place that says where
those positions are. It is deliberately not reachable from the planned stabilizer
mode, which refuses a channel outright. The readout is reachable from it, and is
the one entry point here that needs no external engine at all: expanding a Pauli
through a Clifford circuit is Pauli algebra, not tableau simulation.

It owns no device selection, no shot policy, no seed stream, no dispatch, and no
public result assembly. A caller that needs those goes through Runtime, which
owns them. The survey is not a claim about a program: it reports which
instructions the noise route would execute and which it would refuse, and it
decides nothing on the caller's behalf.

Why the engine is an external distribution
------------------------------------------
The dependency policy requires a new production dependency to record a
documented need, an ownership boundary, a licence and supply-chain review, a
replacement interface, and an exit plan. For this module those are:

- Need. Stim is the reference implementation of stabilizer simulation and of the
  detector error model that a fault-tolerance workflow needs. Re-deriving it
  would be research spent on an algorithm that is already solved and already
  published, and it would put this repository's own Clifford phase bookkeeping
  in the path of every sampling result.
- Ownership boundary. `flagquantum.simulation.stabilizer` owns the conversion
  and the sampling call. The dependency is never imported by Core, by the
  import-time path, or by any other domain.
- Licence and supply chain. Stim is Apache-2.0, published by Google Quantum AI,
  and imports no NVIDIA component, so it sits in the open-neutral dependency
  class rather than in the accelerated-compute class this project has to
  replace.
- Replacement interface and exit plan. The entry points are the seam. A native
  tableau implementation can replace the bodies of `sample_stabilizer` and
  `sample_noisy_measurements` without changing a caller, and the Clifford gate
  set below is the contract a replacement has to satisfy. That work is scheduled
  separately rather than scaffolded here, so this module holds exactly one
  implementation of each. The readout entry point is already native for the same
  reason it can be: the Clifford phase bookkeeping this module would otherwise
  owe the engine is what it performs there, and it performs it without importing
  one.

Fail-closed contract
--------------------
An instruction outside the Clifford gate set is refused with a named error
naming the accepted set. The positioned-noise entry accepts one class of channel
and one measurement shape: a channel whose Kraus operators are a mixture of Pauli
operators up to a global phase, on as many wires as the channel acts on, and one
wire per measurement.
That class is not a list of names - it is decided by the operators the channel
carries, through `KrausChannel.unitary_mixture` - and it is exactly the class a
tableau absorbs, because sampling one branch of it moves the Pauli frame and
leaves the tableau alone. An amplitude damping, phase damping, reset, or thermal
relaxation channel is refused because no Pauli frame represents it; so is a
unitary that is not a Pauli up to a global phase, such as a coherent
over-rotation by an angle that is not a multiple of pi, even though it is a
one-branch unitary mixture. A multi-wire measurement is refused because the
caller reads the record back one recorded bit at a time. Nothing is approximated,
decomposed, or silently dropped, because a stabilizer result that quietly
answers a different circuit is worse than a refusal: the caller cannot tell the
two apart from the samples.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from itertools import product
from types import ModuleType
from typing import Any

import torch

from ...core.ir import CircuitIR, MeasurementNode, ensure_circuit_ir
from ...core.operator_schema import canonical_opcode, get_operator_schema
from ...errors import CapabilityError, FlagQuantumError, ValidationError
from ...noise import KrausChannel

# FlagQuantum opcode -> engine gate name. Only Clifford gates appear here, and
# the mapping is total over the Clifford subset of the operator schema: an
# opcode this engine can represent exactly is never missing from it.
#
# `ccx` and `cswap` are absent deliberately rather than by omission. Both are
# permutations of the computational basis whose induced map is not GF(2)-linear
# (for `cswap`, wire b reads b + a(b + c)), so neither normalizes the Pauli
# group and neither is a Clifford gate. They belong to a higher level of the
# Clifford hierarchy, which is a different simulation regime and a different
# capacity curve.
_GATE_NAMES: dict[str, str] = {
    "i": "I",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "h": "H",
    "s": "S",
    "sdg": "S_DAG",
    "sx": "SQRT_X",
    "sxdg": "SQRT_X_DAG",
    "cx": "CX",
    "cy": "CY",
    "cz": "CZ",
    "swap": "SWAP",
}

# The opcodes this engine executes, as canonical FlagQuantum gate names.
CLIFFORD_GATE_NAMES = frozenset(_GATE_NAMES)

_CLIFFORD_SET_TEXT = ", ".join(sorted(CLIFFORD_GATE_NAMES))

# The Core measurement kinds this representation answers, and the subset it
# answers by drawing outcomes. A tableau holds a state and a frame, so it answers
# a sampling request from the frame and a Pauli expectation by expansion; it holds
# no amplitudes and no marginals, so a distribution or a Pauli-basis sampling
# request is refused rather than approximated. Runtime reads both sets from here,
# because a list kept in the planner or the executor would be a second answer to a
# question the representation already answers.
STABILIZER_SAMPLING_KINDS = frozenset({"sample", "counts"})
STABILIZER_MEASUREMENT_KINDS = (
    frozenset(
        {
            "expectation_identity",
            "expectation_ps",
            "expectation_z",
        }
    )
    | STABILIZER_SAMPLING_KINDS
)

# `stim` documents a seed as an integer in `range(2**64)`.
_SEED_LIMIT = 2**64

# The channel this engine translates. A channel is executed exactly when its
# Kraus operators are a mixture of Pauli unitaries, because a Pauli error is the
# one error a stabilizer tableau absorbs: the frame moves and the tableau stays.
# A general channel has no frame, and this engine refuses it rather than sampling
# the nearest Pauli approximation.
_MEASURE_OPCODE = "measure"
_RESET_OPCODE = "reset"
_MEASURE_ENGINE_NAME = "M"
_RESET_ENGINE_NAME = "R"

# The engine spells one frame distribution two ways. On one or two wires it is a
# single channel instruction whose arguments are the non-identity frames in
# `_frame_labels` order, with the identity taking the remainder. On any number of
# wires it is a chain of correlated-error instructions, one term per non-identity
# frame, because the engine names no channel instruction past two wires and a
# frame wider than that is still a frame. Both spellings read the same frame
# order, so a frame's argument cannot drift away from the frame its position
# names.
_PAULI_ENGINE_NAMES: dict[int, str] = {1: "PAULI_CHANNEL_1", 2: "PAULI_CHANNEL_2"}
_CORRELATED_ERROR_ENGINE_NAME = "E"
_ELSE_CORRELATED_ERROR_ENGINE_NAME = "ELSE_CORRELATED_ERROR"
_PAULI_LABELS = "IXYZ"

# Every engine operation that carries a Pauli frame rather than a gate, a
# reset, or a readout. A frame distribution is one of these however it was
# spelled, so a reader of the translated program can tell a frame from an
# operation by its name alone.
_FRAME_OPERATION_NAMES = frozenset(_PAULI_ENGINE_NAMES.values()) | {
    _CORRELATED_ERROR_ENGINE_NAME,
    _ELSE_CORRELATED_ERROR_ENGINE_NAME,
}
_PAULI_MATRICES: dict[str, torch.Tensor] = {
    "I": torch.tensor(((1, 0), (0, 1)), dtype=torch.complex128),
    "X": torch.tensor(((0, 1), (1, 0)), dtype=torch.complex128),
    "Y": torch.tensor(((0, -1j), (1j, 0)), dtype=torch.complex128),
    "Z": torch.tensor(((1, 0), (0, -1)), dtype=torch.complex128),
}

# A branch matrix comes from a channel the caller built, so it is compared against
# a Pauli at the tolerance the channel class itself uses.
_PAULI_TOLERANCE = 1e-6

# The Pauli algebra the readout entry point works in. A tracked Pauli is one
# letter per wire plus a single `i`-power, so `i**phase * letters` is the
# operator; the letters alone would lose every sign, because `X Z = -i Y`. The
# product table is written out rather than generated from a closed form so a
# reader can check each row against `X Z = -i Y` and `Z X = i Y`.
_PAULI_PRODUCTS: dict[tuple[str, str], tuple[str, int]] = {
    ("I", "I"): ("I", 0),
    ("I", "X"): ("X", 0),
    ("I", "Y"): ("Y", 0),
    ("I", "Z"): ("Z", 0),
    ("X", "I"): ("X", 0),
    ("X", "X"): ("I", 0),
    ("X", "Y"): ("Z", 1),
    ("X", "Z"): ("Y", 3),
    ("Y", "I"): ("Y", 0),
    ("Y", "X"): ("Z", 3),
    ("Y", "Y"): ("I", 0),
    ("Y", "Z"): ("X", 1),
    ("Z", "I"): ("Z", 0),
    ("Z", "X"): ("Y", 1),
    ("Z", "Y"): ("X", 3),
    ("Z", "Z"): ("I", 0),
}

# A Clifford gate conjugates each single-wire generator into a Pauli, so four
# images describe it completely: `X` and `Z` on each wire of its support, in
# support order, each as `(letters, i-power)`. A one-wire gate uses the first two
# images and its support is one wire long, so the last two are never read. A
# two-letter image is read onto the support in order. `CX` sends `X_a` to
# `X_a X_b` and `Z_b` to `Z_a Z_b`, which is why the control's `Z` is the one that
# is left alone.
_CLIFFORD_IMAGES: dict[str, tuple[tuple[str, int], ...]] = {
    "I": (("X", 0), ("Z", 0), ("X", 0), ("Z", 0)),
    "X": (("X", 0), ("Z", 2), ("X", 0), ("Z", 2)),
    "Y": (("X", 2), ("Z", 2), ("X", 2), ("Z", 2)),
    "Z": (("X", 2), ("Z", 0), ("X", 2), ("Z", 0)),
    "H": (("Z", 0), ("X", 0), ("Z", 0), ("X", 0)),
    "S": (("Y", 0), ("Z", 0), ("Y", 0), ("Z", 0)),
    "S_DAG": (("Y", 2), ("Z", 0), ("Y", 2), ("Z", 0)),
    "SQRT_X": (("X", 0), ("Y", 2), ("X", 0), ("Y", 2)),
    "SQRT_X_DAG": (("X", 0), ("Y", 0), ("X", 0), ("Y", 0)),
    "CX": (("XX", 0), ("ZI", 0), ("IX", 0), ("ZZ", 0)),
    "CZ": (("XZ", 0), ("ZI", 0), ("ZX", 0), ("IZ", 0)),
    "CY": (("XY", 0), ("ZI", 0), ("ZX", 0), ("ZZ", 0)),
    "SWAP": (("IX", 0), ("IZ", 0), ("XI", 0), ("ZI", 0)),
}

# The four gates whose inverse is a different gate. Every other gate in
# `_GATE_NAMES` is its own inverse, which is why this table is not a mirror of
# the image table.
_ADJOINT_GATES: dict[str, str] = {
    "S": "S_DAG",
    "S_DAG": "S",
    "SQRT_X": "SQRT_X_DAG",
    "SQRT_X_DAG": "SQRT_X",
}

# One engine instruction: its name, its target list, and its arguments. A gate, a
# reset, a measurement, and a channel instruction name wire indices as targets; a
# correlated error names Pauli targets, which the engine builds from a frame
# label, so the target list is the one part whose element type depends on the
# instruction being built.
_EngineOperation = tuple[str, tuple[Any, ...], float | tuple[float, ...] | None]

_EXTRA_NAME = "stim"


class StabilizerDependencyError(CapabilityError, ImportError):
    """Raised only when stabilizer sampling is requested without Stim."""


def _stim() -> ModuleType:
    """Import the engine, or name the extra that provides it."""

    try:
        return import_module("stim")
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise StabilizerDependencyError(
            "stabilizer sampling requires the optional dependency; install it "
            f"with `pip install 'flagquantum[{_EXTRA_NAME}]'`."
        ) from exc


def _validate_seed(seed: Any) -> int | None:
    """Return a seed the engine accepts, refusing anything that is not one.

    A seed that is not an integer denotes no seed, so it is refused rather than
    coerced: `int(2.5)` would sample a different stream than the caller wrote
    while the result reported the requested seed. `bool` is refused for the
    reason the IR count helpers give: it is a flag, not an index.
    """

    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError(
            f"stabilizer sampling seed must be an integer or None, got {seed!r}"
        )
    if not 0 <= seed < _SEED_LIMIT:
        raise ValidationError(
            f"stabilizer sampling seed must be in range(2**64), got {seed}"
        )
    return seed


def _sampling_request(
    program: Any,
    *,
    wires: Sequence[int] | None,
    shots: int | None,
) -> tuple[CircuitIR, MeasurementNode]:
    """Return the validated IR and the measurement the caller asked for.

    The wire list and the shot count are normalized by the Core measurement
    contract, so this module does not restate what a wire or a shot count is.
    A program that already carries lowered measurement nodes is refused: the
    caller's requested wires and the nodes' requested wires are two answers to
    the same question, and this entry point is not the one that reconciles them.
    """

    ir = ensure_circuit_ir(program)
    if ir.measurements:
        kinds = ", ".join(sorted({node.kind for node in ir.measurements}))
        raise CapabilityError(
            "sample_stabilizer does not read lowered measurement nodes "
            f"(found kind(s): {kinds}); pass the wires and shots to sample "
            "directly, and let Runtime lower output requests before an executor "
            "consumes them"
        )
    selected = tuple(range(ir.n_wires)) if wires is None else tuple(wires)
    request = MeasurementNode("sample", selected, shots=shots)
    if request.shots is None:
        raise ValidationError("stabilizer sampling requires a positive shot count")
    outside = tuple(wire for wire in request.wires if wire >= ir.n_wires)
    if outside:
        raise ValidationError(
            f"stabilizer sampling references wire(s) {outside} outside circuit "
            f"range [0, {ir.n_wires - 1}]"
        )
    return ir, request


def require_clifford_program(program: Any) -> CircuitIR:
    """Return the program's IR, refusing anything this representation cannot hold.

    The Clifford gate set is a property of the representation, so the module that
    owns the representation is the only place that decides membership. Runtime
    calls this while planning, before it selects a route, so a circuit this engine
    cannot represent is reported by the stage that knows it instead of after a
    plan has been published.

    Raises:
        CapabilityError: An instruction is a noise channel or is not a Clifford
            gate. The message names the offending instruction and gate.
        ValidationError: The program is not a valid circuit IR.
    """

    ir = ensure_circuit_ir(program)
    _clifford_instructions(ir)
    return ir


def _clifford_instructions(ir: CircuitIR) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return engine gate names and wires, or refuse the first gate outside the set."""

    translated: list[tuple[str, tuple[int, ...]]] = []
    for index, instruction in enumerate(ir.instructions):
        opcode = canonical_opcode(instruction.name)
        schema = get_operator_schema(opcode)
        if schema is not None and schema.channel:
            raise CapabilityError(
                f"instruction {index} {opcode!r} is a noise channel; stabilizer "
                "sampling executes noiseless Clifford circuits, and a channel "
                "here would return samples from a different circuit"
            )
        gate = _GATE_NAMES.get(opcode)
        if gate is None:
            raise CapabilityError(
                f"instruction {index} {opcode!r} is not a Clifford gate; "
                f"stabilizer sampling accepts {_CLIFFORD_SET_TEXT} and fails "
                "closed rather than approximating"
            )
        translated.append((gate, tuple(int(wire) for wire in instruction.wires)))
    return tuple(translated)


def _engine_circuit(
    instructions: tuple[tuple[str, tuple[int, ...]], ...],
    wires: tuple[int, ...],
) -> Any:
    """Build the engine circuit: the gates, then one measurement per wire."""

    stim = _stim()
    circuit = stim.Circuit()
    for gate, gate_wires in instructions:
        circuit.append(gate, list(gate_wires))
    # One explicit measurement per requested wire fixes the sample bit order to
    # the requested wire order, which is the order the returned tensor uses.
    circuit.append("M", list(wires))
    return circuit


def sample_stabilizer(
    program: Any,
    *,
    shots: int,
    wires: Sequence[int] | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample computational-basis measurement outcomes from a Clifford circuit.

    Args:
        program: A `~flagquantum.Circuit` or a validated
            `~flagquantum.CircuitIR` whose instructions are all Clifford.
        shots: Positive number of sampling repetitions.
        wires: Wires to measure, in output order. Defaults to every wire.
        seed: Seed for the sampling stream. The engine documents identical
            samples for an identical seed only on one engine version and one
            machine's instruction set, so a seed reproduces a run rather than
            pinning a bit pattern.

    Returns:
        An `int64` tensor of shape `(shots, len(wires))` holding `0`/`1`
        measurement outcomes. Column `j` is `wires[j]`, so the leftmost column is
        the first requested wire, matching the dense statevector sampler.

    Raises:
        StabilizerDependencyError: The optional engine is not installed.
        CapabilityError: The program carries a noiseless non-Clifford gate, a
            noise channel, or lowered measurement nodes.
        ValidationError: The wire list, shot count, or seed is not a valid one.
    """

    ir, request = _sampling_request(program, wires=wires, shots=shots)
    validated_seed = _validate_seed(seed)
    instructions = _clifford_instructions(ir)
    circuit = _engine_circuit(instructions, request.wires)
    sampler = circuit.compile_sampler(seed=validated_seed)
    samples = sampler.sample(shots=request.shots)
    return torch.as_tensor(samples, dtype=torch.int64)


@dataclass(frozen=True, slots=True)
class PauliReadout:
    """A Pauli string pushed through a Clifford circuit and read out.

    Expanding a measurement in a stabilizer basis is conjugation. Pushing the
    circuit's adjoint across the measured string turns `P` into `U^dagger P U`,
    which for a Clifford circuit is one Pauli again up to a power of `i`. That
    single Pauli is the observable a table of measurements would have flipped,
    and its value on the all-zero state is the expectation the caller asked for.

    ``letters`` and ``phase`` are that conjugation, `i**phase * letters`, and
    ``expectation`` is what the zero state reads from it. The value is exact and
    is one of `-1`, `0`, and `1`: a Pauli string whose expansion still carries an
    `X` or a `Y` is traceless against a computational-basis product state, and a
    diagonal one is `+-1` because `Z|0>` is `|0>`. No sampling, no tolerance, and
    no external engine take part in it, so two callers holding the same program
    read the same value.
    """

    letters: str
    phase: int
    expectation: int


def _multiply_paulis(
    letters: str,
    phase: int,
    image: str,
    image_phase: int,
) -> tuple[str, int]:
    """Return the product of two tensor-product Paulis as letters and `i`-power."""

    product: list[str] = []
    total = phase + image_phase
    for one, two in zip(letters, image, strict=True):
        letter, power = _PAULI_PRODUCTS[(one, two)]
        product.append(letter)
        total += power
    return "".join(product), total % 4


def _conjugate_step(
    gate: str,
    support: tuple[int, ...],
    letters: str,
    phase: int,
) -> tuple[str, int]:
    """Return the tracked Pauli after one Clifford gate is pushed across it.

    Only the gate's own support changes. `G P G^dagger` is the product of the
    images of the tracked letters on that support, because `G^dagger G` is the
    identity and a letter outside the support is conjugated to itself. `Y` is
    `i * X * Z`, so a tracked `Y` contributes a factor of `i` of its own on top
    of both of its images. Reading the tracked letters directly is what makes the
    walk cost one step per instruction rather than one per wire.
    """

    images = _CLIFFORD_IMAGES[gate]
    tracked = "".join(letters[wire] for wire in support)
    conjugated = "I" * len(support)
    conjugated_phase = 0
    for position, letter in enumerate(tracked):
        if letter in "XY":
            conjugated, conjugated_phase = _multiply_paulis(
                conjugated, conjugated_phase, *images[2 * position]
            )
        if letter in "ZY":
            conjugated, conjugated_phase = _multiply_paulis(
                conjugated, conjugated_phase, *images[2 * position + 1]
            )
        if letter == "Y":
            conjugated_phase += 1
    updated = list(letters)
    for offset, wire in enumerate(support):
        updated[wire] = conjugated[offset]
    return "".join(updated), (phase + conjugated_phase) % 4


def _pauli_letters(
    n_wires: int,
    x: Sequence[int],
    y: Sequence[int],
    z: Sequence[int],
) -> str:
    """Return the tracked Pauli, refusing a wire named twice or off the program.

    A wire the string does not name carries the identity, and the three axes are
    disjoint by construction, so a wire appearing twice is a malformed string
    rather than a string two spellings agree about. `bool` is refused for the
    reason the seed check gives: it is a flag, not a wire index.
    """

    letters = ["I"] * n_wires
    for axis, wires in (("x", x), ("y", y), ("z", z)):
        for wire in wires:
            if isinstance(wire, bool):
                raise ValidationError(
                    f"Pauli readout {axis} wires must be integers, got {wire!r}"
                )
            # `operator.index` is the protocol for "I am exactly an integer", so
            # `0.0` and `'1'` are refused while a NumPy or Torch scalar is not.
            # `int(...)` would also have accepted them, and `0.5` would have
            # truncated to wire 0 rather than being refused.
            try:
                index = operator.index(wire)
            except TypeError as error:
                raise ValidationError(
                    f"Pauli readout {axis} wires must be integers, got {wire!r}"
                ) from error
            if not 0 <= index < n_wires:
                raise ValidationError(
                    f"Pauli readout {axis} wire {index} is outside a "
                    f"{n_wires}-wire program"
                )
            if letters[index] != "I":
                raise ValidationError(
                    f"Pauli readout names wire {index} on more than one axis; a "
                    "Pauli string carries one letter per wire"
                )
            letters[index] = axis.upper()
    return "".join(letters)


def pauli_readout(
    program: Any,
    *,
    x: Sequence[int] = (),
    y: Sequence[int] = (),
    z: Sequence[int] = (),
) -> PauliReadout:
    """Expand a Pauli string through a Clifford program and read the zero state.

    This is the readout half of Pauli tracking. The sampling entry points answer
    what a circuit's measurements recorded; this one answers what a measured
    Pauli is worth, which is the other question a stabilizer run is asked and the
    one a table of circuit outcomes is answered against. The string is conjugated
    by the whole program, which names the observable that a run of that program
    would have flipped, and that observable is read against the all-zero state,
    the state a Clifford circuit starts from. The program's own measurement nodes
    are not read, because the question is about the circuit: reconciling a
    lowered request with a representation is Runtime's, and Runtime strips them
    before it reaches here.

    Args:
        program: A `~flagquantum.Circuit` or a validated
            `~flagquantum.CircuitIR` whose instructions are all Clifford.
        x: Wires the string carries an `X` on.
        y: Wires the string carries a `Y` on.
        z: Wires the string carries a `Z` on. A wire appears in at most one of
            the three, and every wire the string does not name is the identity,
            so an empty string is the identity operator and reads `1`.

    Returns:
        A `PauliReadout` holding the conjugated Pauli, its `i`-power, and the
        exact value the zero state gives it.

    Raises:
        CapabilityError: An instruction is a noise channel or a gate outside the
            Clifford set. The message names the offending instruction and gate.
        ValidationError: The program is not a valid circuit IR, or an axis names
            a wire twice or outside the program.

    Examples:
        A parity on a Bell pair is a single-wire observable after the circuit
        moves it, and being a stabilizer it is certain:

        >>> import flagquantum as fq
        >>> from flagquantum.simulation.stabilizer import pauli_readout
        >>> readout = pauli_readout(fq.Circuit(2).h(0).cx(0, 1), z=(0, 1))
        >>> readout.letters, readout.expectation
        ('IZ', 1)
    """

    ir = ensure_circuit_ir(program)
    letters = _pauli_letters(ir.n_wires, x, y, z)
    phase = 0
    # Walking the instructions backwards while pushing each gate's adjoint is
    # `U^dagger P U` for the whole program: `U^dagger = G_1^dagger ... G_n^dagger`
    # acts on `P` from the left as `G_n^dagger` down to `G_1^dagger`.
    for gate, support in reversed(_clifford_instructions(ir)):
        letters, phase = _conjugate_step(
            _ADJOINT_GATES.get(gate, gate), support, letters, phase
        )
    if any(letter in "XY" for letter in letters):
        return PauliReadout(letters=letters, phase=phase, expectation=0)
    # `U^dagger P U` is Hermitian because `P` is, so `i**phase` is real here and
    # the power is even. An odd power would mean the conjugation left the Pauli
    # group, which a Clifford gate cannot do.
    assert phase % 2 == 0, (letters, phase)
    return PauliReadout(
        letters=letters,
        phase=phase,
        expectation=1 if phase == 0 else -1,
    )


def _validate_shots(shots: Any) -> int:
    """Return the shot count, refusing anything that is not a positive integer."""

    if isinstance(shots, bool) or not isinstance(shots, int):
        raise ValidationError(f"shot count must be an integer, got {shots!r}")
    if shots <= 0:
        raise ValidationError(f"shot count must be positive, got {shots}")
    return shots


def _pauli_matrix(label: str) -> torch.Tensor:
    """Return the tensor-product Pauli operator named by ``label``."""

    matrix = _PAULI_MATRICES[label[0]]
    for character in label[1:]:
        matrix = torch.kron(matrix, _PAULI_MATRICES[character])
    return matrix


def _pauli_frames(n_wires: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """Return every Pauli frame on ``n_wires`` wires, identity first.

    The engine names a frame by its Pauli on each wire, and both the label and
    the argument order come from the lexicographic order over the labels. That
    order is built here rather than listed, so a frame's sampled arguments cannot
    drift away from the frame its label names.

    The identity is the first entry, which is what makes the engine's argument
    order the entries after it: the engine's channel instructions carry one
    argument per non-identity frame and take the identity's probability as the
    remainder, so the identity is the one frame that has no argument of its own.

    A Pauli frame is phase-free: `X` and `-X` are the same frame, because a global
    phase is unobservable and does not change which measurement outcomes a frame
    predicts. Nothing here returns a phase.
    """

    return tuple(
        ("".join(labels), _pauli_matrix("".join(labels)))
        for labels in product(_PAULI_LABELS, repeat=n_wires)
    )


def _frame_index(
    unitary: torch.Tensor, table: tuple[tuple[str, torch.Tensor], ...]
) -> int | None:
    """Return the position in ``table`` of the frame ``unitary`` is, or ``None``.

    A branch is a frame when it equals a Pauli times a single phase. The phase
    comes out of the Hilbert-Schmidt inner product rather than a comparison of
    matrix entries, so a branch written as `-i X` is recognised as `X` and a
    branch written as `exp(-i pi X / 4)` is not recognised at all - which is the
    distinction the caller needs, because only the first is an error a tableau
    absorbs.
    """

    for position, (_, target) in enumerate(table):
        phase = torch.trace(target.mH @ unitary) / target.shape[0]
        if bool(torch.allclose(unitary, phase * target, atol=_PAULI_TOLERANCE)):
            return position
    return None


def _frame_weights(
    instruction: Any, index: int, wires: tuple[int, ...]
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Return the non-identity frame labels and weights of a channel, or refuse it.

    The channel's operators decide, not its name: an instruction named for one
    channel while carrying another is translated as the channel it carries.
    Branches that land on the same frame are added together, because a frame is
    one argument however many branches reach it. A branch that is the identity
    contributes to no entry at all, so what comes back is the part of the channel
    that has to be expressed and the caller reads the identity as the remainder.

    This is the whole of the channel classification, and both the sampling route
    and the survey read it, so a channel cannot be a frame for one and not for
    the other.
    """

    matrix = instruction.matrix
    operators = (
        matrix.kraus if isinstance(matrix, KrausChannel) else tuple(matrix or ())
    )
    if not operators:
        raise CapabilityError(
            f"instruction {index} {instruction.name!r} carries no Kraus operators, "
            "so there is no channel to translate"
        )
    if isinstance(matrix, KrausChannel):
        channel = matrix
    else:
        # The operators are the caller's channel, so a set that is not a channel
        # at all is malformed input rather than a missing capability: the two are
        # refused differently because only one of them is the caller's to fix by
        # choosing another channel.
        try:
            channel = KrausChannel(instruction.name, operators)
        except ValueError as error:
            raise ValidationError(
                f"instruction {index} {instruction.name!r} is not a well-formed "
                f"channel: {error}"
            ) from error
    mixture = channel.unitary_mixture
    if mixture is None:
        raise CapabilityError(
            f"instruction {index} {instruction.name!r} is not a mixture of unitary "
            "operators; a Pauli frame is what this engine tracks, and a general "
            "channel leaves the tableau unchanged only if it is one"
        )
    frames = _pauli_frames(len(wires))
    weights = [0.0] * (len(frames) - 1)
    for branch, unitary in enumerate(mixture.unitaries):
        position = _frame_index(
            unitary.detach().to(device="cpu", dtype=torch.complex128), frames
        )
        if position is None:
            raise CapabilityError(
                f"instruction {index} {instruction.name!r} is a mixture of "
                f"unitaries, but branch {branch} is not a Pauli operator up to a "
                "global phase, so it is not a frame this engine can track"
            )
        if position:
            weights[position - 1] += float(mixture.probabilities[branch])
    if math.fsum(weights) > 1.0:
        raise CapabilityError(
            f"instruction {index} {instruction.name!r} is a mixture of unitaries "
            "whose non-identity frames carry "
            f"{math.fsum(weights):.12g} of probability; the engine's channels take "
            "the identity as the remainder, so they cannot express it"
        )
    return tuple(label for label, _ in frames[1:]), tuple(weights)


def _frame_operations(
    labels: tuple[str, ...],
    weights: tuple[float, ...],
    wires: tuple[int, ...],
    index: int,
    name: str,
) -> tuple[_EngineOperation, ...]:
    """Return the engine operations that carry one frame distribution.

    Up to two wires the engine has a channel instruction whose arguments are the
    non-identity frames in label order, so the distribution goes out as it is.
    Past two wires it has none, and a frame is still a frame: the distribution
    goes out as a chain of correlated errors instead, one term per non-identity
    frame, and the identity is the chain falling through without a term firing.

    A chain term is conditional on no earlier term having fired, so a caller's
    weight for frame ``k`` becomes ``w_k / (1 - sum of the weights before it)``.
    Emitting the caller's weights directly would make every term after the first
    far more likely than the caller asked for, because the engine multiplies a
    term by the probability that the frames before it did not fire. The
    arithmetic is done here rather than left to the engine so that the
    distribution the caller built is the distribution that is sampled.
    """

    engine_name = _PAULI_ENGINE_NAMES.get(len(wires))
    if engine_name is not None:
        return ((engine_name, wires, weights),)
    operations: list[_EngineOperation] = []
    previous = 0.0
    for label, weight in zip(labels, weights, strict=True):
        if weight <= 0.0:
            # A zero-weight frame is not an event, and a chain term for it would
            # consume the remainder that the frames after it have to chain
            # against.
            continue
        remaining = 1.0 - previous
        # A frame that takes the whole remainder is certain once the frames before
        # it have not fired, and dividing would only reach the same one through a
        # rounding error - and the engine refuses a probability it reads as above
        # one, so rounding must not be allowed to decide that.
        conditional = (
            1.0 if remaining - weight <= _PAULI_TOLERANCE else weight / remaining
        )
        # A frame is a Pauli on the wires it acts on and nothing on the rest, and
        # the engine's correlated error names only the wires it acts on. Carrying
        # the identity wires into the target list would ask the engine for an
        # identity Pauli target, which it does not have.
        targets = tuple(
            (wire, character)
            for wire, character in zip(wires, label, strict=True)
            if character != "I"
        )
        operations.append(
            (
                (
                    _CORRELATED_ERROR_ENGINE_NAME
                    if not operations
                    else _ELSE_CORRELATED_ERROR_ENGINE_NAME
                ),
                targets,
                conditional,
            )
        )
        previous += weight
    return tuple(operations)


def _declares_channel(instruction: Any, schema: Any) -> bool:
    """Whether an instruction is a channel: a declared opcode, or a lowered one.

    A channel reaches the IR two ways: as an opcode the operator schema declares
    as a channel, or as an instruction a noise model lowered. Only four opcodes
    declare the channel kind, so the lowered flag is what keeps the rest - a
    two-qubit depolarizing, a thermal relaxation - from being read as a gate and
    refused for the wrong reason.
    """

    return bool(instruction.metadata.get("is_channel")) or (
        schema is not None and schema.channel
    )


def _instruction_operations(
    instruction: Any, index: int
) -> tuple[_EngineOperation, ...]:
    """Return the engine operations one instruction translates to, or refuse it.

    A channel is one or more operations: one channel instruction up to two wires,
    or a correlated-error chain past that. Every other instruction is one
    operation, and an instruction that is none of the accepted forms is refused
    where it stands, so the refusal names the instruction rather than the
    program.
    """

    opcode = canonical_opcode(instruction.name)
    schema = get_operator_schema(opcode)
    wires = tuple(int(wire) for wire in instruction.wires)
    if _declares_channel(instruction, schema):
        labels, weights = _frame_weights(instruction, index, wires)
        return _frame_operations(labels, weights, wires, index, instruction.name)
    gate = _GATE_NAMES.get(opcode)
    if gate is not None:
        return ((gate, wires, None),)
    if opcode == _MEASURE_OPCODE or opcode == _RESET_OPCODE:
        if len(wires) != 1:
            raise CapabilityError(
                f"instruction {index} {opcode!r} names {len(wires)} wire(s); "
                "stabilizer sampling records one bit per measurement, so a "
                "measurement must name one wire"
            )
        return (
            (
                (
                    _MEASURE_ENGINE_NAME
                    if opcode == _MEASURE_OPCODE
                    else _RESET_ENGINE_NAME
                ),
                wires,
                None,
            ),
        )
    raise CapabilityError(
        f"instruction {index} {opcode!r} is not a Clifford gate, a reset, a "
        "measurement, or a noise channel whose Kraus operators are a mixture "
        f"of Pauli unitaries; stabilizer sampling accepts {_CLIFFORD_SET_TEXT}, "
        f"{_MEASURE_OPCODE!r}, {_RESET_OPCODE!r}, and such a channel, and fails "
        "closed rather than approximating"
    )


def _noisy_operations(ir: CircuitIR) -> tuple[_EngineOperation, ...]:
    """Return engine operations for a Clifford program with positioned noise.

    Gates, resets, measurements, and every channel whose Kraus operators are a
    mixture of Pauli unitaries all translate; anything else is refused. A channel
    stays where the caller put it, which is the whole point of this conversion:
    the position of an error relative to the gates around it is what makes it a
    data error at a round boundary or a measurement error at one readout.
    """

    operations: list[_EngineOperation] = []
    for index, instruction in enumerate(ir.instructions):
        operations.extend(_instruction_operations(instruction, index))
    return tuple(operations)


def _engine_targets(stim: ModuleType, name: str, targets: tuple[Any, ...]) -> list[Any]:
    """Return the engine's target list for one operation.

    A gate, a reset, a measurement, and a channel instruction name wire indices.
    A correlated error names Pauli targets instead, one per wire, because its
    whole content is which Pauli the frame applies where; a chain term built as
    plain wire indices would be refused by the engine rather than silently
    mistranslated, but the two are different target kinds and are built apart.
    """

    if name not in (
        _CORRELATED_ERROR_ENGINE_NAME,
        _ELSE_CORRELATED_ERROR_ENGINE_NAME,
    ):
        return list(targets)
    builder = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}
    return [builder[character](int(wire)) for wire, character in targets]


def _noisy_engine_circuit(
    operations: tuple[_EngineOperation, ...],
    terminal_wires: tuple[int, ...],
) -> Any:
    """Build the engine circuit for positioned noise and interleaved readouts."""

    stim = _stim()
    circuit = stim.Circuit()
    for name, targets, arguments in operations:
        resolved = _engine_targets(stim, name, targets)
        if arguments is None:
            circuit.append(name, resolved)
        else:
            circuit.append(name, resolved, arguments)
    if terminal_wires:
        circuit.append(_MEASURE_ENGINE_NAME, list(terminal_wires))
    return circuit


@dataclass(frozen=True, slots=True)
class StabilizerSurvey:
    """What the positioned-noise route makes of a program, without running it.

    A stabilizer regime is worth choosing exactly when the program is one the
    engine can execute, and the engine answers that question today by refusing to
    run anything else. A caller that learns the answer from an exception has
    already spent the attempt on it, and a caller holding a program whose channel
    is not a Pauli frame would rather know that before placing it in a workflow
    than after a round of sampling. This record is that same classification taken
    first, read through the same translation the sampling route uses, so the two
    cannot disagree about what a program is.

    It is a census and not a promise. An empty ``blockers`` says every
    instruction translated; it does not say the samples are right, that a dense
    route would agree, or that the program is worth running in this regime.
    """

    n_wires: int
    clifford_gates: int
    resets: int
    recorded_measurements: int
    noise_instructions: int
    all_pauli_frames: bool
    blockers: tuple[str, ...]

    @property
    def executable(self) -> bool:
        """Whether every instruction translated, rather than being refused."""

        return not self.blockers


def survey_stabilizer_program(program: Any) -> StabilizerSurvey:
    """Report what :func:`sample_noisy_measurements` would make of ``program``.

    Every instruction is put through the translation the sampling route uses and
    the outcome is counted, so a refusal is collected as a blocker and named with
    the instruction that caused it instead of stopping the survey. Nothing is
    executed, no channel is approximated, and no instruction is dropped: a
    program with a blocker is reported as one rather than repaired.

    ``all_pauli_frames`` is the narrower question within that census, and it is
    the one this engine exists to answer. It is true when every channel in the
    program is a mixture of Pauli frames - the class a stabilizer tableau absorbs
    - and false when a channel was refused for not being one. A program can have
    blockers and still report ``all_pauli_frames`` when what it carries instead
    is a non-Clifford gate, because a gate that leaves the Clifford group is a
    different regime question than an error that has no frame.

    Args:
        program: A `~flagquantum.Circuit` or a validated
            `~flagquantum.CircuitIR`.

    Returns:
        A `StabilizerSurvey` of the program's instruction census and its
        refusals, each refusal naming the instruction index and opcode.

    Raises:
        ValidationError: The program is not a well-formed circuit and cannot be
            read at all.
    """

    ir = ensure_circuit_ir(program)
    gates = 0
    resets = 0
    measurements = 0
    noise = 0
    all_pauli_frames = True
    blockers: list[str] = []
    for index, instruction in enumerate(ir.instructions):
        declares_channel = False
        try:
            schema = get_operator_schema(canonical_opcode(instruction.name))
            declares_channel = _declares_channel(instruction, schema)
            if declares_channel:
                noise += 1
            operations = _instruction_operations(instruction, index)
        except FlagQuantumError as error:
            # Each translation refusal names the instruction it refused, so the
            # census records it as it stands rather than wrapping it. A wrapper
            # would print the index and the opcode twice - once from here and
            # once from the engine - and the routed mode surfaces this text to
            # the caller verbatim.
            blockers.append(str(error))
            if declares_channel:
                all_pauli_frames = False
            continue
        for name, _targets, _arguments in operations:
            if name == _MEASURE_ENGINE_NAME:
                measurements += 1
            elif name == _RESET_ENGINE_NAME:
                resets += 1
            elif name not in _FRAME_OPERATION_NAMES:
                gates += 1
    return StabilizerSurvey(
        n_wires=ir.n_wires,
        clifford_gates=gates,
        resets=resets,
        recorded_measurements=measurements,
        noise_instructions=noise,
        all_pauli_frames=all_pauli_frames,
        blockers=tuple(blockers),
    )


def sample_noisy_measurements(
    program: Any,
    *,
    shots: int,
    terminal_wires: Sequence[int] | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample every measurement a Clifford program records.

    Unlike :func:`sample_stabilizer`, which measures a requested wire list once
    at the end of a noiseless circuit, this entry point executes the program's
    own measurements in order and accepts positioned Pauli noise channels
    between them. That is the shape a detector error model needs: a syndrome
    round is read where the program reads it, and an error lands exactly where
    the caller put it.

    Args:
        program: A `~flagquantum.Circuit` or a validated `~flagquantum.CircuitIR`
            whose instructions are Clifford gates, single-wire resets,
            single-wire measurements, and noise channels whose Kraus operators
            are a mixture of Pauli operators up to a global phase, on any number
            of wires. `survey_stabilizer_program` reports the same classification
            without sampling, which is the cheaper way to ask.
        shots: Positive number of sampling repetitions.
        terminal_wires: Wires to measure once after the program has run, in
            record order. Defaults to no terminal measurement.
        seed: Seed for the sampling stream. The engine documents identical
            samples for an identical seed only on one engine version and one
            machine's instruction set, so a seed reproduces a run rather than
            pinning a bit pattern.

    Returns:
        An `int64` tensor of shape `(shots, num_recorded)`. The first columns are
        the program's own measurements in instruction order, one column per
        single-wire measurement; the remaining columns are ``terminal_wires`` in
        the order given.

    Raises:
        StabilizerDependencyError: The optional engine is not installed.
        CapabilityError: The program carries a channel whose operators are not a
            mixture of Pauli frames, a non-Clifford gate, or a measurement naming
            more than one wire.
        ValidationError: The shot count, seed, terminal wire, or record shape is
            not a valid one.
    """

    ir = ensure_circuit_ir(program)
    if ir.measurements:
        kinds = ", ".join(sorted({node.kind for node in ir.measurements}))
        raise CapabilityError(
            "sample_noisy_measurements does not read lowered measurement nodes "
            f"(found kind(s): {kinds}); the program's own measurements are what "
            "it samples"
        )
    shots = _validate_shots(shots)
    validated_seed = _validate_seed(seed)
    selected = () if terminal_wires is None else tuple(terminal_wires)
    outside = tuple(wire for wire in selected if wire >= ir.n_wires)
    if outside:
        raise ValidationError(
            f"terminal wire(s) {outside} outside circuit range "
            f"[0, {ir.n_wires - 1}]"
        )
    operations = _noisy_operations(ir)
    recorded = sum(1 for name, _, _ in operations if name == _MEASURE_ENGINE_NAME)
    if recorded + len(selected) == 0:
        raise ValidationError(
            "sampling requires at least one recorded measurement; the program "
            "records none and no terminal wire was requested"
        )
    circuit = _noisy_engine_circuit(operations, selected)
    sampler = circuit.compile_sampler(seed=validated_seed)
    samples = sampler.sample(shots=shots)
    return torch.as_tensor(samples, dtype=torch.int64)
