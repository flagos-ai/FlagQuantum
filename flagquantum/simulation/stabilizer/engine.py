"""Clifford stabilizer sampling backed by the optional Stim engine.

A stabilizer circuit conjugates a Pauli group instead of evolving an amplitude
vector, so its state is a tableau whose size grows quadratically with the wire
count rather than exponentially. That is the whole reason a stabilizer regime
exists: circuits made only of Clifford gates reach wire counts no amplitude
store can hold, and they answer exactly the question a fault-tolerance workflow
asks -- what do the measurements read out.

What this module owns
---------------------
One conversion and one numeric entry point:

- translating validated FlagQuantum IR into the engine's circuit form, refusing
  every instruction this engine cannot represent, and
- sampling measurement outcomes for a requested wire list.

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
- Replacement interface and exit plan. The entry point is the seam. A native
  tableau implementation can replace the body of `sample_stabilizer` without
  changing a caller, and the Clifford gate set below is the contract a
  replacement has to satisfy. That work is scheduled separately rather than
  scaffolded here, so this module holds exactly one implementation.

Fail-closed contract
--------------------
An instruction outside the Clifford gate set is refused with a named error
naming the accepted set. Nothing is approximated, decomposed, or silently
dropped, because a stabilizer result that quietly answers a different circuit is
worse than a refusal: the caller cannot tell the two apart from the samples.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from types import ModuleType
from typing import Any

import torch

from ...core.ir import CircuitIR, MeasurementNode, ensure_circuit_ir
from ...core.operator_schema import canonical_opcode, get_operator_schema
from ...errors import CapabilityError, ValidationError

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
