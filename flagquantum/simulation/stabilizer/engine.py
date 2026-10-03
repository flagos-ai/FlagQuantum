"""Clifford stabilizer sampling backed by the optional Stim engine.

A stabilizer circuit conjugates a Pauli group instead of evolving an amplitude
vector, so its state is a tableau whose size grows quadratically with the wire
count rather than exponentially. That is the whole reason a stabilizer regime
exists: circuits made only of Clifford gates reach wire counts no amplitude
store can hold, and they answer exactly the question a fault-tolerance workflow
asks -- what do the measurements read out.

What this module owns
---------------------
One conversion and two numeric entry points:

- translating validated FlagQuantum IR into the engine's circuit form, refusing
  every instruction this engine cannot represent,
- sampling measurement outcomes for a requested wire list, and
- sampling every measurement a Clifford program records when that program
  carries positioned noise channels.

The second entry point exists because a detector error model is defined between
syndrome rounds rather than at a named gate: quantum error correction places its
data error at a round boundary and its measurement error immediately before one
check's readout, and the program is the only place that says where those
positions are. It is deliberately not reachable from the planned stabilizer
mode, which refuses a channel outright -- a routed `mode="stabilizer"` run still
executes noiseless Clifford circuits.

It owns no device selection, no shot policy, no seed stream, no dispatch, and no
public result assembly. A caller that needs those goes through Runtime, which
owns them.

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
  implementation of each.

Fail-closed contract
--------------------
An instruction outside the Clifford gate set is refused with a named error
naming the accepted set. The positioned-noise entry accepts one class of channel
and one measurement shape: a one- or two-wire channel whose Kraus operators are a
mixture of Pauli operators up to a global phase, and one wire per measurement.
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
from collections.abc import Sequence
from importlib import import_module
from itertools import product
from types import ModuleType
from typing import Any

import torch

from ...core.ir import CircuitIR, MeasurementNode, ensure_circuit_ir
from ...core.operator_schema import canonical_opcode, get_operator_schema
from ...errors import CapabilityError, ValidationError
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

# `stim` documents a seed as an integer in `range(2**64)`.
_SEED_LIMIT = 2**64

# The channel rule the positioned-noise entry translates by, and the engine's
# names for the frames it tracks. A channel is executed exactly when its Kraus
# operators are a mixture of Pauli unitaries, because a Pauli error is the one
# error a stabilizer tableau absorbs: the frame moves and the tableau stays. A
# general channel has no frame, and this engine refuses it rather than sampling
# the nearest Pauli approximation.
_MEASURE_OPCODE = "measure"
_RESET_OPCODE = "reset"
_MEASURE_ENGINE_NAME = "M"
_RESET_ENGINE_NAME = "R"

# The engine's Pauli-frame vocabulary, one instruction per wire count. The label
# order is the engine's own argument order, and `_pauli_weight_table` is what
# keeps the two in step: a wrong order here would mislabel every two-wire frame.
_PAULI_ENGINE_NAMES: dict[int, str] = {1: "PAULI_CHANNEL_1", 2: "PAULI_CHANNEL_2"}
_PAULI_LABELS = "IXYZ"
_PAULI_MATRICES: dict[str, torch.Tensor] = {
    "I": torch.tensor(((1, 0), (0, 1)), dtype=torch.complex128),
    "X": torch.tensor(((0, 1), (1, 0)), dtype=torch.complex128),
    "Y": torch.tensor(((0, -1j), (1j, 0)), dtype=torch.complex128),
    "Z": torch.tensor(((1, 0), (0, -1)), dtype=torch.complex128),
}

# A branch matrix comes from a channel the caller built, so it is compared against
# a Pauli at the tolerance the channel class itself uses.
_PAULI_TOLERANCE = 1e-6

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


def _pauli_channel_operation(
    instruction: Any, index: int, wires: tuple[int, ...]
) -> tuple[str, tuple[float, ...]]:
    """Return the engine instruction for a positioned channel, or refuse it.

    The channel's operators decide, not its name: an instruction named for one
    channel while carrying another executes as the channel it carries. Branches
    that land on the same frame are added together, because a frame is one
    argument however many branches reach it.

    A branch that is the identity contributes to no argument at all. The engine's
    channel instructions take one argument per non-identity frame and sample the
    identity with whatever probability is left over, so the identity's own weight
    is what those arguments are checked against rather than an argument of its own.
    """

    engine_name = _PAULI_ENGINE_NAMES.get(len(wires))
    if engine_name is None:
        raise CapabilityError(
            f"instruction {index} {instruction.name!r} names {len(wires)} wire(s); "
            "a Pauli frame is tracked on one or two wires"
        )
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
    return engine_name, tuple(weights)


def _noisy_operations(
    ir: CircuitIR,
) -> tuple[tuple[str, tuple[int, ...], float | tuple[float, ...] | None], ...]:
    """Return engine operations for a Clifford program with positioned noise.

    Gates, resets, measurements, and every channel whose Kraus operators are a
    mixture of Pauli unitaries all translate; anything else is refused. A channel
    stays where the caller put it, which is the whole point of this conversion:
    the position of an error relative to the gates around it is what makes it a
    data error at a round boundary or a measurement error at one readout.
    """

    operations: list[tuple[str, tuple[int, ...], float | tuple[float, ...] | None]] = []
    for index, instruction in enumerate(ir.instructions):
        opcode = canonical_opcode(instruction.name)
        schema = get_operator_schema(opcode)
        wires = tuple(int(wire) for wire in instruction.wires)
        # A channel reaches the IR two ways: as a declared channel opcode, or as
        # an instruction a noise model lowered. Only four opcodes declare the
        # channel kind, so the lowered flag is what keeps the rest - a
        # two-qubit depolarizing, a thermal relaxation - from being read as a
        # gate and refused for the wrong reason.
        if bool(instruction.metadata.get("is_channel")) or (
            schema is not None and schema.channel
        ):
            engine_name, weights = _pauli_channel_operation(instruction, index, wires)
            operations.append((engine_name, wires, weights))
            continue
        gate = _GATE_NAMES.get(opcode)
        if gate is not None:
            operations.append((gate, wires, None))
            continue
        if opcode == _MEASURE_OPCODE or opcode == _RESET_OPCODE:
            if len(wires) != 1:
                raise CapabilityError(
                    f"instruction {index} {opcode!r} names {len(wires)} wire(s); "
                    "stabilizer sampling records one bit per measurement, so a "
                    "measurement must name one wire"
                )
            operations.append(
                (
                    (
                        _MEASURE_ENGINE_NAME
                        if opcode == _MEASURE_OPCODE
                        else _RESET_ENGINE_NAME
                    ),
                    wires,
                    None,
                )
            )
            continue
        raise CapabilityError(
            f"instruction {index} {opcode!r} is not a Clifford gate, a reset, a "
            "measurement, or a noise channel whose Kraus operators are a mixture "
            f"of Pauli unitaries; stabilizer sampling accepts {_CLIFFORD_SET_TEXT}, "
            f"{_MEASURE_OPCODE!r}, {_RESET_OPCODE!r}, and such a channel, and fails "
            "closed rather than approximating"
        )
    return tuple(operations)


def _noisy_engine_circuit(
    operations: tuple[
        tuple[str, tuple[int, ...], float | tuple[float, ...] | None], ...
    ],
    terminal_wires: tuple[int, ...],
) -> Any:
    """Build the engine circuit for positioned noise and interleaved readouts."""

    stim = _stim()
    circuit = stim.Circuit()
    for name, wires, arguments in operations:
        if arguments is None:
            circuit.append(name, list(wires))
        else:
            circuit.append(name, list(wires), arguments)
    if terminal_wires:
        circuit.append(_MEASURE_ENGINE_NAME, list(terminal_wires))
    return circuit


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
            are a mixture of Pauli operators up to a global phase.
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
