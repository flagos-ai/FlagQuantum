# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 block_encoding/chebyshev_polynomial.py.

"""Compose source Chebyshev reflection/block calls without owning coefficients."""

from typing import Literal

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder

ChebyshevCall = Literal["reflection", "forward", "inverse"]


def chebyshev_call_sequence(
    order: int, *, adjoint: bool = False
) -> tuple[ChebyshevCall, ...]:
    """Return the literal source stages and their generic exact reverse."""
    stages: list[ChebyshevCall] = []
    for index in range(order):
        stages.extend(("reflection", "inverse" if index % 2 else "forward"))
    if adjoint:
        reverse: dict[ChebyshevCall, ChebyshevCall] = {
            "reflection": "reflection",
            "forward": "inverse",
            "inverse": "forward",
        }
        return tuple(reverse[stage] for stage in reversed(stages))
    return tuple(stages)


def scaled_chebyshev_component_pair(
    property_pair: tuple[ProgramRecord, ProgramRecord], *, controlled: bool = False
) -> tuple[ProgramRecord, ProgramRecord]:
    """Keep the source's single linear-combination call and exact inverse.

    The source property owns coefficients, rounding and normalization. Generic
    positive control exposes a QBit even when the property's specialized child
    exposes an opaque one-bit control word.
    """
    wrappers = []
    for record in property_pair:
        ports = record.functions[0].registers
        if ports is None or not Signature(ports).thru_registers_only:
            raise ValueError("Scaled Chebyshev requires typed THRU property ports")
        if controlled and (
            not ports
            or ports[0].total_qubits() != 1
            or ports[0].dtype.classical_bits != 0
            or ports[0].shape
        ):
            raise ValueError("Scaled Chebyshev requires one positive property control")
        signature = Signature(
            (Register("ctrl", quantum_bit_type()), *ports[1:]) if controlled else ports
        )
        builder = ComponentProgramBuilder(signature)
        values = dict(builder.inputs)
        if controlled:
            values[ports[0].name] = values.pop("ctrl")
        result = builder.call_program(record, values, entry=record.functions[0].name)
        if controlled:
            result["ctrl"] = result.pop(ports[0].name)
        wrappers.append(builder.finalize(result))
    if wrappers[0].functions[0].registers != wrappers[1].functions[0].registers:
        raise ValueError("Scaled Chebyshev requires matching exact property pair ports")
    return wrappers[0], wrappers[1]


def chebyshev_component_pair(
    child: ProgramRecord,
    inverse: ProgramRecord,
    order: int,
    *,
    reflection: ProgramRecord | None = None,
    controlled: bool = False,
) -> tuple[ProgramRecord, ProgramRecord]:
    """Keep reflection/B/reflection/B-adjoint pairs and the odd final pair.

    The caller owns exact child directions and the source Hermitian zero
    reflection (global coefficient -1). Packed child storage remains owned by
    the caller. Reflection and child helpers own their private clean work.
    Source call counts are retained independently of its analytical model.
    In controlled mode the supplied pair and reflection already own their
    source-positive specialization. Their leading control names are mapped to
    the outer source ctrl bit; no generic control is synthesized inside them.
    """
    child_ports = child.functions[0].registers
    if type(controlled) is not bool or child_ports is None:
        raise ValueError("Chebyshev direction requires typed child storage")
    if controlled and (
        not child_ports
        or child_ports[0].total_qubits() != 1
        or child_ports[0].dtype.classical_bits != 0
        or child_ports[0].shape
    ):
        raise ValueError("Controlled Chebyshev requires a leading positive control bit")
    ports = child_ports[1:] if controlled else child_ports
    if (
        not isinstance(order, int)
        or order < 0
        or not Signature(child_ports).thru_registers_only
        or inverse.functions[0].registers != child_ports
        or any(reg.name not in ("system", "ancilla", "resource") for reg in ports)
    ):
        raise ValueError(
            "Chebyshev requires a nonnegative order and packed exact child pair"
        )
    ancilla = next((reg for reg in ports if reg.name == "ancilla"), None)
    if ancilla is not None and type(ancilla.bitsize) is not int:
        raise ValueError("Chebyshev executable ancilla width must be concrete")
    if order > 0 and ancilla is not None and ancilla.bitsize > 0 and reflection is None:
        raise ValueError("Chebyshev requires its exact zero-reflection owner")
    builder = ComponentProgramBuilder(
        Signature(
            (Register("ctrl", quantum_bit_type()), *ports) if controlled else ports
        )
    )
    state = dict(builder.inputs)

    def reflect() -> None:
        if ancilla is not None and ancilla.bitsize > 0:
            assert reflection is not None
            values = {"reg0_": state["ancilla"]}
            reflection_control_name = None
            if controlled:
                reflection_ports = reflection.functions[0].registers
                if (
                    not reflection_ports
                    or reflection_ports[0].total_qubits() != 1
                    or reflection_ports[0].dtype.classical_bits != 0
                    or reflection_ports[0].shape
                ):
                    raise ValueError(
                        "Controlled Chebyshev reflection requires its positive control"
                    )
                reflection_control_name = reflection_ports[0].name
                values[reflection_control_name] = state["ctrl"]
            result = builder.call_program(
                reflection, values, entry=reflection.functions[0].name
            )
            state["ancilla"] = result["reg0_"]
            if controlled:
                assert reflection_control_name is not None
                state["ctrl"] = result[reflection_control_name]

    def apply(record: ProgramRecord) -> None:
        nonlocal state
        values = {reg.name: state[reg.name] for reg in ports}
        if controlled:
            values[child_ports[0].name] = state["ctrl"]
        result = builder.call_program(record, values, entry=record.functions[0].name)
        if controlled:
            state = {"ctrl": result.pop(child_ports[0].name), **result}
        else:
            state = result

    for stage in chebyshev_call_sequence(order):
        if stage == "reflection":
            reflect()
        else:
            apply(child if stage == "forward" else inverse)
    forward = builder.finalize(state)
    pairs = [(child, inverse), (inverse, child)]
    if reflection is not None:
        pairs.append((reflection, reflection))
    return forward, adjoint_component_calls(forward, tuple(pairs))
