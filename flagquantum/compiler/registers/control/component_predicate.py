# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CtrlSpecAnd and ControlledViaAnd.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/ctrl_spec_and.py

"""Give the existing predicate emitter directional typed ports and lifetimes."""

from math import prod

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ....core.value_math import ssum
from ....core.value_parameters import is_symbolic_value
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_graph import _namespace_helpers
from ..layout.component_partition import partition_component_program
from .conjunction_program import conjunction_declaration, conjunction_program
from .control_predicate import ControlRegister


def validate_control_ports(
    controls: tuple[Register, ...], patterns: tuple[ControlRegister, ...] | None
) -> int:
    """Validate admitted quantum words and concrete stored-bit activation clauses."""
    if not controls or (patterns is not None and len(controls) != len(patterns)):
        raise ValueError("control ports require one activation pattern per register")
    Signature(controls)
    count = 0
    for index, register in enumerate(controls):
        encoding = {
            "bit": "unsigned",
            "opaque": "unsigned",
            "uint": "unsigned",
            "signed": "twos_complement",
            "ones_complement": "ones_complement",
            "montgomery": "unsigned",
            "fixed_point": "unsigned",
            "finite_field": "unsigned",
            "finite_field_polynomial": "unsigned",
        }.get(register.dtype.encoding)
        if register.dtype.is_symbolic():
            raise ValueError("control predicates require a nonsymbolic source dtype")
        width = executable_register_width(register.dtype)
        if (
            register.dtype.encoding in {"finite_field", "finite_field_polynomial"}
            and dict(register.dtype.parameters).get("characteristic") != 2
        ):
            raise ValueError(
                f"unsupported executable register descriptor {register.dtype!r}"
            )
        if register.dtype.encoding == "fixed_point" and width != 1:
            raise ValueError("Fixed-point controls require single-bit storage")
        count += width * prod(register.shape)
        if (
            register.side != RegisterSide.THRU
            or register.dtype.classical_bits != 0
            or encoding is None
        ):
            raise ValueError("control port dtype, shape and activation must agree")
        if patterns is not None:
            pattern = patterns[index]
            if (
                encoding != pattern.encoding
                or width != pattern.width
                or prod(register.shape) != len(pattern.values)
            ):
                raise ValueError("control port dtype, shape and activation must agree")
            pattern.bits()
    if count != 1 and any(
        register.dtype.encoding in {"finite_field", "finite_field_polynomial"}
        for register in controls
    ):
        raise ValueError("Field storage supports only a single-bit control")
    if not count:
        raise ValueError("structured control requires a nonempty activation")
    return count


def control_predicate_signature(
    controls: tuple[Register, ...], *, uncompute: bool = False
) -> Signature:
    """Declare grouped inputs and owned prefix outputs, including symbolic widths."""
    count = ssum(register.total_qubits() for register in controls)
    symbolic = is_symbolic_value(count)
    if (not symbolic and count < 2) or type(uncompute) is not bool:
        raise ValueError("predicate requires at least two bits and a boolean direction")
    if {r.name for r in controls} & {"junk", "target"}:
        raise ValueError("predicate control names conflict with its logical outputs")
    side = RegisterSide.LEFT if uncompute else RegisterSide.RIGHT
    junk = (
        ()
        if not symbolic and count == 2
        else (Register("junk", RegisterType("opaque", count - 2), side=side),)
    )
    return Signature(
        (*controls, *junk, Register("target", quantum_bit_type(), side=side))
    )


def control_predicate_program(
    controls: tuple[Register, ...],
    patterns: tuple[ControlRegister, ...] | None,
    *,
    name: str = "predicate",
    uncompute: bool = False,
) -> ProgramRecord:
    """Produce or consume target/junk using row-major MSB-first control words.

    Forward owns clean allocation of n-1 prefix outputs. Erasure requires their
    correct computed values, measures in reverse ladder order and releases them
    after clearing. These logical outputs remain live across an intervening
    controlled call. Canonical integer storage retains its descriptor across
    split/join. Montgomery controls use stored residues and bounded unsigned
    controls use their stored bit values. Single-bit field and polynomial controls
    bypass this multi-bit predicate in their caller. The frozen predicate refuses
    wider field storage. Unsupported descriptors raise ValueError.

    With patterns=None binary activation retains a declaration requiring explicit
    resource/numerical bindings. For three or more controls, the locked source
    converts HasLength(count) into positive controls. This graph follows that
    documented constructor rule and emits the existing positive ladder.
    """
    count = validate_control_ports(controls, patterns)
    signature = control_predicate_signature(controls, uncompute=uncompute)
    combine = partition_component_program(controls, "combine", True)
    partition = partition_component_program(controls, "partition", False)
    conjunction = (
        conjunction_declaration(count, uncompute=uncompute)
        if patterns is None and count == 2
        else conjunction_program(
            (
                (1,) * count
                if patterns is None
                else tuple(bit for pattern in patterns for bit in pattern.bits())
            ),
            name="conjunction",
            uncompute=uncompute,
        )
    )
    builder = ComponentProgramBuilder(
        signature,
        (*combine.functions, *partition.functions, *conjunction.functions),
        name=name,
    )
    inputs: dict[str, BindingCollection] = {}
    if uncompute:
        if count > 2:
            junk = builder.inputs["junk"]
            assert isinstance(junk, RegisterBinding)
            inputs["junk"] = builder.split(junk)
        inputs["target"] = builder.inputs["target"]
    packed = builder.call_dict(
        "combine",
        {register.name: builder.inputs[register.name] for register in controls},
    )["x"]
    assert isinstance(packed, RegisterBinding)
    inputs["ctrl"] = builder.split(packed)
    result = builder.call_dict("conjunction", inputs)
    packed = builder.join(result.pop("ctrl"))
    outputs = builder.call_dict("partition", {"x": packed})
    if not uncompute:
        if count > 2:
            outputs["junk"] = builder.join(result["junk"])
        outputs["target"] = result["target"]
    return _namespace_helpers(builder.finalize(outputs), name)
