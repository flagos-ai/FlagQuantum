"""Static resource accounting for one program's operation sequence.

:func:`estimate_resources` answers the question CUDA-Q's ``estimate_resources``
answers -- how many operations a program applies, how deep their dependency
chain is, how wide the widest operation is, how many T-family operations it
carries, and how large a register it needs -- by reading a
:class:`~flagquantum.core.ir.CircuitIR` and executing nothing.

Two differences from CUDA-Q are deliberate and stated on the returned record.
CUDA-Q's ``estimate_resources`` executes the kernel and unrolls its loops
through traces and measurement choices, so its numbers depend on the run that
produced them; this one reads the operation list, so a program whose operations
depend on run-time outcomes is refused rather than estimated.  And CUDA-Q
renders an operation's name with its controls prefixed, so a recorded two-control
``x`` and a written ``ccx`` both report ``ccx``; this one reports the canonical
opcode, because ``Instruction`` canonicalises the name at construction and no
field records a control count that the opcode does not already name.

The planner's :class:`~flagquantum.runtime.planner.CircuitAnalysis` describes the
same program for execution planning.  That type is a plan input whose fields are
projected into a plan's identity, so it carries no statement about whether its
numbers were measured; this record exists to carry exactly that statement, and
reuses the compiler's own scheduler rather than a second depth rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.ir import Instruction, ensure_circuit_ir
from ..errors import CapabilityError
from .pipeline import schedule_layers

ESTIMATE_BASIS = "static_instruction_sequence"
# Names what a :class:`ResourceEstimate` was computed from.
#
# Every field of an estimate is a property of the program text. A record carrying this
# value states that no execution, allocation, device query, or timing produced its
# numbers, so it must not be read as measured evidence: it says what a program would
# apply, never what a run did cost.

_T_FAMILY_OPCODES = frozenset({"t", "tdg"})
# The opcodes a T-depth is defined over.
#
# Held here rather than on :class:`~flagquantum.core.operator_schema.OperatorSchema`
# because the family is a dependency-depth notion this module is the only reader of: a
# schema marker would change the operator manifest and the generated capability
# documents for a single caller.


@dataclass(frozen=True)
class ResourceEstimate:
    """A static count over one program's operation sequence.

    The counts describe a program, not a run of it.  Nothing here is measured: no
    wall-clock time, no resident memory, no allocation, no communication, and no
    fidelity.  A program whose operation sequence depends on run-time outcomes is
    refused by :func:`estimate_resources` instead of estimated, because no single
    static count describes it.

    ``depth`` and ``per_wire_depth`` come from the ASAP schedule the compiler
    already builds: ``depth`` is the number of schedule layers, and
    ``per_wire_depth`` holds, per wire, the number of layers up to and including
    the last layer that touches it.  ``t_count`` and ``t_depth`` both cover T and
    T-dagger operations, the pair a T-depth is defined over.

    ``max_operation_width`` is the widest single operation, so a program of
    three-wire operations has width 3 however many wires it declares: that is the
    operation width, not the register size, which is :attr:`n_qubits`.
    ``used_wires`` counts the wires at least one operation touches, so it is
    narrower than :attr:`n_qubits` for a program that does not use its whole
    register, and ``channel_count`` counts the noise channels a lowered noise
    model injected, which keeps a caller from reading noise-injected counts as
    algorithmic work.
    """

    n_wires: int
    used_wires: int
    n_operations: int
    operation_counts: dict[str, int]
    max_operation_width: int
    depth: int
    per_wire_depth: tuple[int, ...]
    t_count: int
    t_depth: int
    channel_count: int

    @property
    def n_qubits(self) -> int:
        """Return the peak qubit count, which is the allocated register size.

        ``CircuitIR`` allocates its whole register before the first operation, so
        no operation can raise a watermark above :attr:`n_wires`: the peak is
        fixed by the program's declared width rather than reached during a run.
        """

        return self.n_wires

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record that states the basis of the count."""

        return {
            "kind": "flagquantum.resource_estimate",
            "estimate_basis": ESTIMATE_BASIS,
            "n_qubits": self.n_qubits,
            "used_wires": self.used_wires,
            "n_operations": self.n_operations,
            "operation_counts": dict(self.operation_counts),
            "max_operation_width": self.max_operation_width,
            "depth": self.depth,
            "per_wire_depth": list(self.per_wire_depth),
            "t_count": self.t_count,
            "t_depth": self.t_depth,
            "channel_count": self.channel_count,
        }


def _data_dependence_reason(instruction: Instruction) -> str | None:
    """Return why no static count describes this instruction, or ``None``."""

    metadata = instruction.metadata
    if metadata.get("is_dynamic"):
        return (
            "the operations the program applies depend on run-time outcomes, so "
            "their count is not a property of the program alone"
        )
    if "condition_clauses" in metadata or "conditions" in metadata:
        return (
            "it is applied only when a classical condition on earlier "
            "measurement results holds, so it may apply zero times"
        )
    return None


def estimate_resources(program: Any) -> ResourceEstimate:
    """Count the static resource requirements of one program.

    Args:
        program: A :class:`~flagquantum.core.ir.CircuitIR` or any object exposing
            ``to_ir()``.  A program carrying unbound
            :class:`~flagquantum.core.parameters.Parameter` angles needs no
            substitution, because the operations a program applies do not depend
            on the values of its angles.

    Returns:
        A :class:`ResourceEstimate` over the program's operation sequence.

    Raises:
        CapabilityError: The program's operation sequence is data-dependent, so
            no single static count describes it.
        TypeError: ``program`` is neither a ``CircuitIR`` nor an object exposing
            ``to_ir()``.

    Examples:
        Count a Bell pair:

        >>> import flagquantum as fq
        >>> from flagquantum.compiler import estimate_resources
        >>> estimate = estimate_resources(fq.Circuit(2).h(0).cx(0, 1))
        >>> estimate.depth, estimate.operation_counts == {"h": 1, "cx": 1}
        (2, True)
    """

    ir = ensure_circuit_ir(program)
    for index, instruction in enumerate(ir):
        reason = _data_dependence_reason(instruction)
        if reason is not None:
            raise CapabilityError(
                f"a static resource estimate is not defined for instruction "
                f"{index} ({instruction.name!r}): {reason}"
            )

    layers = schedule_layers(ir)
    operation_counts: dict[str, int] = {}
    max_operation_width = 0
    channel_count = 0
    t_count = 0
    per_wire_depth = [0] * ir.n_wires
    per_wire_t_depth = [0] * ir.n_wires

    for layer_index, layer in enumerate(layers):
        for instruction in layer:
            name = instruction.name
            operation_counts[name] = operation_counts.get(name, 0) + 1
            max_operation_width = max(max_operation_width, len(instruction.wires))
            if instruction.metadata.get("is_channel"):
                channel_count += 1
            is_t_family = name in _T_FAMILY_OPCODES
            if is_t_family:
                t_count += 1
            # A layer holds operations sharing no wire, so every operation in it
            # reads the wire state left by the layers before it and the order
            # within a layer cannot matter.  A T-family operation opens a new T
            # layer on top of the deepest one among the wires it touches; every
            # other operation carries each wire's T layer forward.
            next_t_depth = max(per_wire_t_depth[wire] for wire in instruction.wires)
            if is_t_family:
                next_t_depth += 1
            for wire in instruction.wires:
                per_wire_depth[wire] = layer_index + 1
                per_wire_t_depth[wire] = next_t_depth

    return ResourceEstimate(
        n_wires=ir.n_wires,
        # A wire belongs to a layer only if an operation touches it, so a wire is
        # touched exactly when it belongs to at least one layer.
        used_wires=sum(1 for depth in per_wire_depth if depth > 0),
        n_operations=len(ir),
        operation_counts=operation_counts,
        max_operation_width=max_operation_width,
        depth=len(layers),
        per_wire_depth=tuple(per_wire_depth),
        t_count=t_count,
        t_depth=max(per_wire_t_depth),
        channel_count=channel_count,
    )


__all__ = ("ESTIMATE_BASIS", "ResourceEstimate", "estimate_resources")
