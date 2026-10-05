"""A detector error model read back as an executable circuit.

The other direction of this domain's construction lives in
:mod:`flagquantum.qec.dem_construction`: it reads a circuit and a noise record and
derives the mechanisms a shot can fire. This module is the reverse reading, and
what makes it possible is that a detector error model is already a circuit's worth
of arithmetic stated in another vocabulary. A mechanism is a set of detectors and
observables it flips together with a probability, and mechanisms are independent
unless an error id says otherwise, so a model is exactly a register of detector
wires and observable wires carrying one Pauli frame per fault. Reading a model back
is therefore not an inverse problem with a fitted answer: the frame a mechanism
names *is* the operator a realization applies, and the probability *is* the weight
of the branch that applies it.

The realization is executable. `flagquantum.simulation.stabilizer` samples a
program whose channels are mixtures of Pauli frames, which is what every
instruction here is, so a model can be handed to an engine that shares no
arithmetic with :meth:`DetectorErrorModel.dem_sampling`. That is what the
realization is for: the two routes are independent enough that holding their
sampled rates against each other is evidence about both.

What a realization is, and what it is not
-----------------------------------------
It is not the circuit the model came from, and no reading of a model can produce
that circuit, because the model does not contain it. A detector error model is a
quotient: it states which detector parities a fault flips, never which
measurements compose a detector, so the syndrome-extraction circuit, its gate
sequence, its depth and its ancilla layout are all outside what the model says.
Two different memory experiments of the same distance and the same noise record
can produce the same model, and a reader that returned a circuit for one of them
would be asserting a fact the model never carried. What this module returns
instead is the canonical detector-level circuit every model *does* determine: one
wire per detector and one per observable, one frame per fault, and one measurement
per wire at the end.

The detector and observable split is not recoverable from the circuit either, for
the same reason: a wire index says nothing about which side of the split it falls
on, so :func:`detector_error_model_from_circuit` takes the detector count from its
caller. A caller who states the wrong count gets a model whose mechanisms are
right and whose split is not, rather than an error, which is why the two functions
here are meant to be read as a pair.

Round trip
----------
:func:`circuit_from_detector_error_model` followed by
:func:`detector_error_model_from_circuit` returns the model it started from, for
every model whose mechanisms all fire with a non-zero probability. The order is
restored rather than preserved because
:class:`~flagquantum.qec.dem.DetectorErrorModel` sorts its mechanisms into a
canonical order on construction, so a group of alternatives that the sort
interleaved is regrouped whichever order the reader emits its members in.

A mechanism at probability zero is the one model this pair does not carry, and it
is refused rather than dropped. A realization is a channel whose branches are the
ways a fault fires, and a fault that never fires is not one of them: its frame
leaves no trace in the channel's arithmetic -- every branch mass is zero, so the
operator it would apply is zero -- and no reading of the channel could tell it
from the identity. Dropping it would silently make the realization stand for a
model with fewer mechanisms than the one it was read from, so the writer states the
refusal and names the mechanism.

The two halves are read back from different parts of the instruction, and the
division is deliberate. A frame is read from the operators, because the operators
are what the engine executes: the instruction's wire list fixes the register, each
branch's matrix fixes one frame on that register, and the identity branch is what
distinguishes a channel that always fires from one that fires with a probability.
A probability is read from the instruction's declared parameter, because a
probability cannot survive a round trip through an operator. A branch is written
``sqrt(p) * U`` for a unitary ``U``, so reading ``p`` back means squaring a square
root, and squaring a square root is not the identity on a double: it disagrees
with ``p`` for roughly half of all doubles, ``0.01`` and ``0.5`` among them. The
declared parameter is therefore the probability, the matrix is checked against it,
and a program whose two halves disagree is refused rather than averaged. This is
the same division ``flagquantum/qec/sampling.py`` writes its noise instructions
with: a channel states its rates and carries the operators those rates were turned
into.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import torch

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..core.operator_schema import canonical_opcode
from .dem_alternatives import fault_groups

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from .dem import DemError, DetectorErrorModel

__all__ = (
    "circuit_from_detector_error_model",
    "detector_error_model_from_circuit",
)

# The opcode of one fault's frame. It carries its operators rather than declaring
# them, because the operator schema's four channels each name a single-wire family
# with one rate, and a fault is a frame on however many detectors and observables
# it flips: `bit_flip` on two wires would state two independent faults where the
# model states one fault with a two-detector signature.
_FRAME_OPCODE = "dem_frame"
_MEASURE_OPCODE = "measure"
_PROBABILITIES_KEY = "probabilities"
_ERROR_ID_KEY = "error_id"

# A probability is stored as a double, so the operator test compares dense
# matrices and carries a numerical slack of its own rather than an exact-bit
# expectation.
_OPERATOR_TOLERANCE = 1e-9
# The declared mass is the model's own double, so the completeness check is held
# to the width the arithmetic was done at rather than to the operator slack.
_MASS_TOLERANCE = 1e-12

_IDENTITY = torch.eye(2, dtype=torch.complex128)
_BIT_FLIP = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)


def circuit_from_detector_error_model(model: DetectorErrorModel) -> CircuitIR:
    """Return an executable circuit whose measurements are the model's targets.

    The circuit has one wire per detector followed by one wire per observable, one
    Pauli-frame channel per fault, and one measurement per wire. Wires are ordered
    so that a sample's columns are the detectors in index order and then the
    observables in index order, which is the layout
    :meth:`DetectorErrorModel.dem_sampling` returns its two tensors in.

    A fault is a group of mechanisms: the ones sharing an error id, or a single
    mechanism with no id. A group becomes one channel rather than several, because
    the group is one fault -- its members' frames are the branches of that channel,
    each carrying the member's own probability, and the identity is the left-over
    mass of the fault firing none of them. That is the reading
    :mod:`flagquantum.qec.dem_alternatives` states, and it is why a realization of
    an id-carrying model draws at most one member of a group per shot instead of
    drawing them independently.

    Args:
        model: The model to realize.

    Returns:
        A :class:`~flagquantum.core.ir.CircuitIR` over ``num_detectors +
        num_observables`` wires.

    Raises:
        ValueError: A mechanism states probability zero, and so has no branch.
    """

    detectors = _count(model.num_detectors, name="num_detectors")
    observables = _count(model.num_observables, name="num_observables")
    width = detectors + observables
    _require_every_mechanism_fires(model.errors)

    instructions: list[Instruction] = []
    for group in fault_groups(model.errors):
        members = [model.errors[index] for index in group]
        supports = [_support(member, detectors) for member in members]
        layout = tuple(sorted({target for support in supports for target in support}))
        operators = [
            math.sqrt(member.probability) * _frame(layout, support)
            for member, support in zip(members, supports, strict=True)
        ]
        probabilities = [member.probability for member in members]
        # ``DetectorErrorModel`` already refused a group whose members sum above
        # one, so the left-over mass is the group firing none of its members and is
        # never negative. It is summed in the order those members are stored in,
        # which is the order the refusal accumulated them in.
        remainder = 1.0 - sum(probabilities)
        if remainder > 0.0:
            operators.append(math.sqrt(remainder) * _frame(layout, ()))
            probabilities.append(remainder)
        metadata: dict[str, Any] = {"is_channel": True}
        if members[0].error_id is not None:
            # The grouping is the instruction's own structure, but which group this
            # is among the model's groups is not, and it is what makes the round
            # trip return the model it started from rather than a renumbered copy
            # of it.
            metadata[_ERROR_ID_KEY] = int(members[0].error_id)
        instructions.append(
            Instruction(
                name=_FRAME_OPCODE,
                wires=layout,
                params={_PROBABILITIES_KEY: tuple(probabilities)},
                matrix=tuple(operators),
                metadata=metadata,
            )
        )
    instructions.extend(
        Instruction(
            name=_MEASURE_OPCODE, wires=(target,), metadata={"is_dynamic": True}
        )
        for target in range(width)
    )
    return CircuitIR(n_wires=width, instructions=tuple(instructions))


def detector_error_model_from_circuit(
    circuit: Any, *, num_detectors: int
) -> DetectorErrorModel:
    """Read a circuit back into the detector error model it realizes.

    The circuit must be one :func:`circuit_from_detector_error_model` built, or a
    program written the same way: every instruction is a fault frame or a
    measurement, the measurements are one per wire in wire order and come after
    every frame, and every wire a frame names is flipped by one of its branches.
    Anything else is refused, because this reader states what a realization is
    rather than guessing at what an arbitrary annotated circuit might mean.

    Each frame instruction becomes one mechanism per non-identity branch, with the
    branch's declared probability and the branch's support split into detector
    indices and observable indices at ``num_detectors``. The identity branch is the
    fault's left-over mass and states no mechanism, which is what makes an id-free
    channel read back as one independent fault and a wider one as a group of
    alternatives.

    Args:
        circuit: A realization, or any program that normalizes to an executable IR.
        num_detectors: How many of the circuit's wires are detectors. Everything
            above them is read as an observable.

    Returns:
        The model the realization stands for, with its mechanisms in the order its
        frames are executed rather than in the model's canonical order, which the
        returned record applies to them.

    Raises:
        TypeError: The circuit is not a program, or ``num_detectors`` is not an
            integer.
        ValueError: The circuit is not a realization, ``num_detectors`` does not
            split it, or a frame's operators and its declared probabilities do not
            agree.
    """

    ir = ensure_circuit_ir(circuit)
    if ir.measurements:
        kinds = ", ".join(sorted({node.kind for node in ir.measurements}))
        raise ValueError(
            "a detector error model is read from a realization's own measurements, "
            f"and this program also carries lowered measurement node(s): {kinds}"
        )
    split = _count(num_detectors, name="num_detectors")
    if not 1 <= split <= ir.n_wires:
        raise ValueError(
            f"num_detectors must split the circuit's {ir.n_wires} wire(s) into at "
            f"least one detector and any number of observables, got {split}"
        )

    errors: list[DemError] = []
    measured = 0
    for index, instruction in enumerate(ir.instructions):
        opcode = canonical_opcode(instruction.name)
        if opcode == _MEASURE_OPCODE:
            if tuple(instruction.wires) != (measured,):
                raise ValueError(
                    f"a realization measures wire {measured} at instruction "
                    f"{index}, in wire order and with no repetition; this "
                    f"instruction measures {tuple(instruction.wires)}"
                )
            measured += 1
            continue
        if opcode != _FRAME_OPCODE:
            raise ValueError(
                f"instruction {index} {opcode!r} is neither a fault frame nor a "
                "measurement, so this program is not a detector error model's "
                "realization"
            )
        if measured:
            raise ValueError(
                f"instruction {index} {opcode!r} follows a measurement; a "
                "realization applies every frame to the register it starts in and "
                "reads the whole register out afterwards"
            )
        errors.extend(_mechanisms(instruction, index, split=split))

    if measured != ir.n_wires:
        raise ValueError(
            f"a realization measures each of its {ir.n_wires} wire(s) once, and this "
            f"program measures {measured}"
        )

    from .dem import DetectorErrorModel

    return DetectorErrorModel(
        num_detectors=split,
        num_observables=ir.n_wires - split,
        errors=tuple(errors),
    )


def _require_every_mechanism_fires(errors: Sequence[DemError]) -> None:
    """Refuse a model carrying a mechanism that never fires.

    A mechanism at probability zero is a record, but it is not a branch: a channel
    whose branches are its ways of firing has no way to state one that never
    happens, and no shot of a realization could fire it. Dropping it silently would
    make the realization stand for a model with fewer mechanisms than the one it was
    read from, so the model is refused instead, naming the mechanism that has no
    branch to be read back from.

    Raises:
        ValueError: A mechanism states probability zero.
    """

    for error in errors:
        if error.probability == 0.0:
            raise ValueError(
                "a realization states one branch per way a fault fires, and this "
                f"model carries a mechanism at probability zero flipping the "
                f"detector(s) {error.detectors} and observable(s) {error.observables}; "
                "no shot of any realization fires it, so no branch could be read "
                "back from and the model has no realization here"
            )


def _mechanisms(instruction: Instruction, index: int, *, split: int) -> list[DemError]:
    """Return one mechanism per non-identity branch of one frame instruction."""

    from .dem import DemError

    layout = tuple(int(target) for target in instruction.wires)
    width = len(layout)
    operators = _operators(instruction, index, width=width)
    declared = _declared_probabilities(instruction, index, count=len(operators))

    mechanisms: list[DemError] = []
    covered: set[int] = set()
    for branch, (operator, probability) in enumerate(
        zip(operators, declared, strict=True)
    ):
        scale = _scale(operator, width)
        if abs(scale * scale - probability) > _OPERATOR_TOLERANCE:
            raise ValueError(
                f"instruction {index} {instruction.name!r} branch {branch} is the "
                f"operator of a frame firing with probability {scale * scale!r} and "
                f"declares {probability!r}; the operators and the declared masses "
                "are two statements of one channel and must agree"
            )
        support = _frame_support(operator / scale, width) if scale > 0.0 else ()
        if support is None:
            raise ValueError(
                f"instruction {index} {instruction.name!r} branch {branch} is not a "
                "product of bit flips up to a phase, so it is not a frame a detector "
                "error model states"
            )
        if not support:
            continue
        flipped = {layout[position] for position in support}
        covered |= flipped
        mechanisms.append(
            DemError(
                probability=probability,
                detectors=tuple(sorted(t for t in flipped if t < split)),
                observables=tuple(sorted(t - split for t in flipped if t >= split)),
                error_id=_error_id(instruction, index),
            )
        )
    unused = tuple(sorted(set(layout) - covered))
    if unused:
        raise ValueError(
            f"instruction {index} {instruction.name!r} names wire(s) {unused} that no "
            "branch flips, so its wire list is wider than the frame it applies"
        )
    return mechanisms


def _operators(
    instruction: Instruction, index: int, *, width: int
) -> tuple[torch.Tensor, ...]:
    """Return the branch operators of a frame instruction as a tuple of matrices."""

    if instruction.matrix is None:
        raise ValueError(
            f"instruction {index} {instruction.name!r} declares no operators, so "
            "there is no frame to read"
        )
    if isinstance(instruction.matrix, torch.Tensor) or not isinstance(
        instruction.matrix, Sequence
    ):
        raise ValueError(
            f"instruction {index} {instruction.name!r} carries its operators as "
            f"{type(instruction.matrix).__name__}; a frame states one matrix per "
            "branch, so the operators must be a sequence of matrices"
        )
    expected = (1 << width, 1 << width)
    operators: list[torch.Tensor] = []
    for branch, operator in enumerate(instruction.matrix):
        if not isinstance(operator, torch.Tensor) or tuple(operator.shape) != expected:
            shape = (
                tuple(operator.shape) if isinstance(operator, torch.Tensor) else None
            )
            raise ValueError(
                f"instruction {index} {instruction.name!r} branch {branch} has shape "
                f"{shape} over {width} wire(s), and a frame on those wires is "
                f"{expected[0]} by {expected[1]}"
            )
        operators.append(operator)
    return tuple(operators)


def _declared_probabilities(
    instruction: Instruction, index: int, *, count: int
) -> tuple[float, ...]:
    """Return the declared branch masses, checked against the branch count."""

    stated = instruction.params.get(_PROBABILITIES_KEY)
    if not isinstance(stated, Sequence) or isinstance(stated, (str, bytes, bytearray)):
        raise ValueError(
            f"instruction {index} {instruction.name!r} does not declare "
            f"{_PROBABILITIES_KEY!r}; a frame states the mass of every branch, "
            "including the branch that fires nothing"
        )
    if len(stated) != count:
        raise ValueError(
            f"instruction {index} {instruction.name!r} declares {len(stated)} "
            f"branch mass(es) for {count} operator(s), and a frame states one mass "
            "per branch"
        )
    probabilities: list[float] = []
    for branch, value in enumerate(stated):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"instruction {index} {instruction.name!r} branch {branch} states "
                f"{value!r} as its mass, and a mass is a real probability"
            )
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"instruction {index} {instruction.name!r} branch {branch} states "
                f"the mass {probability!r}, and a probability is between zero and one"
            )
        probabilities.append(probability)
    total = math.fsum(probabilities)
    if abs(total - 1.0) > _MASS_TOLERANCE:
        raise ValueError(
            f"instruction {index} {instruction.name!r} declares branch masses "
            f"summing to {total!r}; a channel's branches are every way it can fire, "
            "so their masses sum to one"
        )
    return tuple(probabilities)


def _error_id(instruction: Instruction, index: int) -> int | None:
    """Return the fault label an instruction states, if it states one."""

    stated = instruction.metadata.get(_ERROR_ID_KEY)
    if stated is None:
        return None
    if isinstance(stated, bool) or not isinstance(stated, Integral) or stated < 0:
        raise ValueError(
            f"instruction {index} {instruction.name!r} states {stated!r} as its "
            "error id, and an id is a non-negative integer label"
        )
    return int(stated)


def _support(member: DemError, detectors: int) -> frozenset[int]:
    """Return the register positions one mechanism flips."""

    return frozenset(member.detectors) | frozenset(
        detectors + index for index in member.observables
    )


def _frame(layout: tuple[int, ...], support: Iterable[int]) -> torch.Tensor:
    """Return the product of bit flips on ``support`` over the wires ``layout``.

    The operator is built rather than looked up. The stabilizer engine's own
    enumeration of Pauli frames is private to it and covers every frame on a wire
    list, which is not what is needed here: a fault's branch is one specific
    product of bit flips, and it is the same operator the engine identifies again
    when it executes the instruction.
    """

    flipped = frozenset(support)
    operator = torch.ones((1, 1), dtype=torch.complex128)
    for target in layout:
        operator = torch.kron(operator, _BIT_FLIP if target in flipped else _IDENTITY)
    return operator


def _scale(operator: torch.Tensor, width: int) -> float:
    """Return the non-negative real scale of ``operator`` about the identity.

    A branch is written ``scale * unitary``, so the Hilbert-Schmidt norm of the
    branch recovers the scale: ``trace(operator^dagger operator)`` is ``scale**2``
    times the dimension, because the unitary factor is unitary.
    """

    squared = (operator.mH @ operator).trace().real
    return math.sqrt(max(float(squared), 0.0) / (1 << width))


def _frame_support(operator: torch.Tensor, width: int) -> tuple[int, ...] | None:
    """Return the wires a unitary flips, or ``None`` if it is not such a frame.

    A product of bit flips sends the all-zero basis state to the basis state its
    support names, so the operator's first row holds exactly one non-zero entry and
    the column of that entry is the support written as a binary number, most
    significant wire first. The candidate is then rebuilt from that reading and
    compared with the operator, which is what rejects a Pauli word carrying a phase
    flip or a ``Y`` factor: those conjugate a basis state to another one but are
    not products of bit flips.
    """

    columns = torch.nonzero(operator[0].abs() > _OPERATOR_TOLERANCE).flatten().tolist()
    if len(columns) != 1:
        return None
    column = columns[0]
    support = tuple(
        position for position in range(width) if (column >> (width - 1 - position)) & 1
    )
    rebuilt = _frame(tuple(range(width)), support)
    overlap = float((rebuilt.mH @ operator).trace().real) / (1 << width)
    if not torch.allclose(
        operator, overlap * rebuilt, rtol=0.0, atol=_OPERATOR_TOLERANCE
    ):
        return None
    return support


def _count(value: Any, *, name: str) -> int:
    """Normalize a non-negative integer, refusing a flag in its place."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)
