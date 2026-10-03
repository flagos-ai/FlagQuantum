"""Clifford stabilizer sampling backed by the optional Stim engine.

A stabilizer circuit conjugates a Pauli group instead of evolving an amplitude
vector, so its state is a tableau whose size grows quadratically with the qubit
count rather than exponentially. That is the whole reason a stabilizer regime
exists: circuits made only of Clifford gates reach qubit counts no amplitude
store can hold, and they answer exactly the question a fault-tolerance workflow
asks -- what do the measurements read out.

What this module owns
---------------------
One conversion and two numeric entry points:

- translating validated FlagQuantum IR into the engine's circuit form, refusing
  every instruction this engine cannot represent,
- sampling measurement outcomes for a requested qubit list, and
- sampling every measurement a Clifford program records when that program
  carries positioned bit-flip channels.

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
naming the accepted set. The positioned-noise entry accepts one channel and one
measurement shape: a single-qubit `bit_flip` whose Kraus operators are the
bit-flip pair, and one qubit per measurement. Amplitude damping and phase
damping are refused because this engine cannot represent them as one Pauli error
at one position, and a multi-qubit measurement is refused because the caller
reads the record back one recorded bit at a time. Nothing is approximated,
decomposed, or silently dropped, because a stabilizer result that quietly
answers a different circuit is worse than a refusal: the caller cannot tell the
two apart from the samples.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib import import_module
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
# (for `cswap`, qubit b reads b + a(b + c)), so neither normalizes the Pauli
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

# The opcodes the positioned-noise entry translates, and the engine's names for
# them. `bit_flip` is the only channel it executes: the flip is a Pauli error at
# one position, which the engine represents exactly, while amplitude and phase
# damping are not Pauli errors at all.
_NOISE_OPCODE = "bit_flip"
_MEASURE_OPCODE = "measure"
_RESET_OPCODE = "reset"
_NOISE_ENGINE_NAME = "X_ERROR"
_MEASURE_ENGINE_NAME = "M"
_RESET_ENGINE_NAME = "R"

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
    qubits: Sequence[int] | None,
    shots: int | None,
) -> tuple[CircuitIR, MeasurementNode]:
    """Return the validated IR and the measurement the caller asked for.

    The qubit list and the shot count are normalized by the Core measurement
    contract, so this module does not restate what a qubit or a shot count is.
    A program that already carries lowered measurement nodes is refused: the
    caller's requested qubits and the nodes' requested qubits are two answers to
    the same question, and this entry point is not the one that reconciles them.
    """

    ir = ensure_circuit_ir(program)
    if ir.measurements:
        kinds = ", ".join(sorted({node.kind for node in ir.measurements}))
        raise CapabilityError(
            "sample_stabilizer does not read lowered measurement nodes "
            f"(found kind(s): {kinds}); pass the qubits and shots to sample "
            "directly, and let Runtime lower output requests before an executor "
            "consumes them"
        )
    selected = tuple(range(ir.n_wires)) if qubits is None else tuple(qubits)
    request = MeasurementNode("sample", selected, shots=shots)
    if request.shots is None:
        raise ValidationError("stabilizer sampling requires a positive shot count")
    outside = tuple(qubit for qubit in request.wires if qubit >= ir.n_wires)
    if outside:
        raise ValidationError(
            f"stabilizer sampling references qubit(s) {outside} outside circuit "
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
    """Return engine gate names and qubits, or refuse the first gate outside the set."""

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
        translated.append((gate, tuple(int(qubit) for qubit in instruction.wires)))
    return tuple(translated)


def _engine_circuit(
    instructions: tuple[tuple[str, tuple[int, ...]], ...],
    qubits: tuple[int, ...],
) -> Any:
    """Build the engine circuit: the gates, then one measurement per qubit."""

    stim = _stim()
    circuit = stim.Circuit()
    for gate, gate_qubits in instructions:
        circuit.append(gate, list(gate_qubits))
    # One explicit measurement per requested qubit fixes the sample bit order to
    # the requested qubit order, which is the order the returned tensor uses.
    circuit.append("M", list(qubits))
    return circuit


def sample_stabilizer(
    program: Any,
    *,
    shots: int,
    qubits: Sequence[int] | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample computational-basis measurement outcomes from a Clifford circuit.

    Args:
        program: A `~flagquantum.Circuit` or a validated
            `~flagquantum.CircuitIR` whose instructions are all Clifford.
        shots: Positive number of sampling repetitions.
        qubits: Wires to measure, in output order. Defaults to every qubit.
        seed: Seed for the sampling stream. The engine documents identical
            samples for an identical seed only on one engine version and one
            machine's instruction set, so a seed reproduces a run rather than
            pinning a bit pattern.

    Returns:
        An `int64` tensor of shape `(shots, len(qubits))` holding `0`/`1`
        measurement outcomes. Column `j` is `qubits[j]`, so the leftmost column is
        the first requested qubit, matching the dense statevector sampler.

    Raises:
        StabilizerDependencyError: The optional engine is not installed.
        CapabilityError: The program carries a noiseless non-Clifford gate, a
            noise channel, or lowered measurement nodes.
        ValidationError: The qubit list, shot count, or seed is not a valid one.
    """

    ir, request = _sampling_request(program, qubits=qubits, shots=shots)
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


def _bit_flip_probability(instruction: Any, index: int) -> float:
    """Return the flip probability a positioned bit-flip channel carries.

    The channel's Kraus operators are the only thing an instruction carries, so
    the probability is read back off them rather than trusted from a parameter
    map. The operators are also compared against the bit-flip pair: an
    instruction named ``bit_flip`` whose operators are a different channel would
    otherwise be sampled as a bit flip, and the caller could not tell the two
    apart from the samples.
    """

    matrix = instruction.matrix
    operators = (
        matrix.kraus if isinstance(matrix, KrausChannel) else tuple(matrix or ())
    )
    if len(operators) != 2:
        raise CapabilityError(
            f"instruction {index} {instruction.name!r} carries "
            f"{len(operators)} Kraus operator(s); stabilizer sampling executes a "
            "bit-flip channel as exactly two"
        )
    flipped = torch.as_tensor(operators[1])
    probability = float(torch.real(torch.trace(flipped.mH @ flipped) / 2).item())
    if not 0.0 <= probability <= 1.0:
        raise ValidationError(
            f"instruction {index} {instruction.name!r} flips with probability "
            f"{probability}, which is not a probability"
        )
    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=flipped.dtype)
    if not torch.allclose(flipped, math.sqrt(probability) * pauli_x, atol=1e-6):
        raise CapabilityError(
            f"instruction {index} {instruction.name!r} does not carry the "
            "bit-flip pair; stabilizer sampling executes a bit-flip channel as "
            "the identity and the Pauli X"
        )
    return probability


def _noisy_operations(
    ir: CircuitIR,
) -> tuple[tuple[str, tuple[int, ...], float | None], ...]:
    """Return engine operations for a Clifford program with positioned noise.

    Gates, resets, measurements, and single-qubit bit-flip channels all translate;
    anything else is refused. A channel stays where the caller put it, which is
    the whole point of this conversion: the position of a bit flip relative to
    the gates around it is what makes it a data error at a round boundary or a
    measurement error at one readout.
    """

    operations: list[tuple[str, tuple[int, ...], float | None]] = []
    for index, instruction in enumerate(ir.instructions):
        opcode = canonical_opcode(instruction.name)
        schema = get_operator_schema(opcode)
        qubits = tuple(int(qubit) for qubit in instruction.wires)
        if schema is not None and schema.channel:
            if opcode != _NOISE_OPCODE:
                raise CapabilityError(
                    f"instruction {index} {opcode!r} is not supported by "
                    "stabilizer sampling with positioned noise; the only channel "
                    f"it executes exactly is {_NOISE_OPCODE!r}"
                )
            operations.append(
                (_NOISE_ENGINE_NAME, qubits, _bit_flip_probability(instruction, index))
            )
            continue
        gate = _GATE_NAMES.get(opcode)
        if gate is not None:
            operations.append((gate, qubits, None))
            continue
        if opcode == _MEASURE_OPCODE or opcode == _RESET_OPCODE:
            if len(qubits) != 1:
                raise CapabilityError(
                    f"instruction {index} {opcode!r} names {len(qubits)} qubit(s); "
                    "stabilizer sampling records one bit per measurement, so a "
                    "measurement must name one qubit"
                )
            operations.append(
                (
                    (
                        _MEASURE_ENGINE_NAME
                        if opcode == _MEASURE_OPCODE
                        else _RESET_ENGINE_NAME
                    ),
                    qubits,
                    None,
                )
            )
            continue
        raise CapabilityError(
            f"instruction {index} {opcode!r} is not a Clifford gate, a reset, a "
            f"measurement, or a {_NOISE_OPCODE!r} channel; stabilizer sampling "
            f"accepts {_CLIFFORD_SET_TEXT}, {_MEASURE_OPCODE!r}, "
            f"{_RESET_OPCODE!r}, and {_NOISE_OPCODE!r} and fails closed rather "
            "than approximating"
        )
    return tuple(operations)


def _noisy_engine_circuit(
    operations: tuple[tuple[str, tuple[int, ...], float | None], ...],
    terminal_qubits: tuple[int, ...],
) -> Any:
    """Build the engine circuit for positioned noise and interleaved readouts."""

    stim = _stim()
    circuit = stim.Circuit()
    for name, qubits, probability in operations:
        if probability is None:
            circuit.append(name, list(qubits))
        else:
            circuit.append(name, list(qubits), probability)
    if terminal_qubits:
        circuit.append(_MEASURE_ENGINE_NAME, list(terminal_qubits))
    return circuit


def sample_noisy_measurements(
    program: Any,
    *,
    shots: int,
    terminal_qubits: Sequence[int] | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample every measurement a Clifford program records.

    Unlike :func:`sample_stabilizer`, which measures a requested qubit list once
    at the end of a noiseless circuit, this entry point executes the program's
    own measurements in order and accepts single-qubit bit-flip channels placed
    between them. That is the shape a detector error model needs: a syndrome
    round is read where the program reads it, and a bit flip lands exactly where
    the caller put it.

    Args:
        program: A `~flagquantum.Circuit` or a validated `~flagquantum.CircuitIR`
            whose instructions are Clifford gates, single-qubit resets,
            single-qubit measurements, and single-qubit bit-flip channels.
        shots: Positive number of sampling repetitions.
        terminal_qubits: Wires to measure once after the program has run, in
            record order. Defaults to no terminal measurement.
        seed: Seed for the sampling stream. The engine documents identical
            samples for an identical seed only on one engine version and one
            machine's instruction set, so a seed reproduces a run rather than
            pinning a bit pattern.

    Returns:
        An `int64` tensor of shape `(shots, num_recorded)`. The first columns are
        the program's own measurements in instruction order, one column per
        single-qubit measurement; the remaining columns are ``terminal_qubits`` in
        the order given.

    Raises:
        StabilizerDependencyError: The optional engine is not installed.
        CapabilityError: The program carries a channel other than a bit flip, a
            non-Clifford gate, or a measurement naming more than one qubit.
        ValidationError: The shot count, seed, terminal qubit, or record shape is
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
    selected = () if terminal_qubits is None else tuple(terminal_qubits)
    outside = tuple(qubit for qubit in selected if qubit >= ir.n_wires)
    if outside:
        raise ValidationError(
            f"terminal qubit(s) {outside} outside circuit range "
            f"[0, {ir.n_wires - 1}]"
        )
    operations = _noisy_operations(ir)
    recorded = sum(1 for name, _, _ in operations if name == _MEASURE_ENGINE_NAME)
    if recorded + len(selected) == 0:
        raise ValidationError(
            "sampling requires at least one recorded measurement; the program "
            "records none and no terminal qubit was requested"
        )
    circuit = _noisy_engine_circuit(operations, selected)
    sampler = circuit.compile_sampler(seed=validated_seed)
    samples = sampler.sample(shots=shots)
    return torch.as_tensor(samples, dtype=torch.int64)
