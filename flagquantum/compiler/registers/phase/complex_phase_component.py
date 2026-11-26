# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Root order follows locked Qualtran v0.7.0 ComplexPhaseOracle.

"""Compose encoder-owned work, bounded ArcTan and source-selected phase calls."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_control import quantum_control_signature
from ..component_graph import _collections, _elements


def complex_phase_oracle_program(
    encoder: ProgramRecord,
    erase_encoder: ProgramRecord,
    arctan: ProgramRecord,
    erase_arctan: ProgramRecord,
    phases: tuple[ProgramRecord, ...],
    *,
    target_names: tuple[str, ...],
    controlled: bool = False,
) -> ProgramRecord:
    """Preserve exact child calls without reversing measurement primitives.

    Encoded words, phase outputs and each child's private work are distinct
    lifetimes. Encoder targets start zero; the supplied inverse must erase them
    after phase evaluation. All other ports remain caller-owned.
    """
    if type(controlled) is not bool:
        raise ValueError("Complex phase control mode must be boolean")
    modules = (encoder, erase_encoder, arctan, erase_arctan, *phases)
    if any(
        record.physical_context is not None
        or not record.functions
        or any(not function.blocks for function in record.functions)
        for record in modules
    ):
        raise ValueError("Complex phase requires complete logical child Programs")
    ports = encoder.functions[0].registers
    angle_ports = arctan.functions[0].registers
    if (
        ports is None
        or angle_ports is None
        or not Signature(ports).thru_registers_only
        or not Signature(angle_ports).thru_registers_only
        or erase_encoder.functions[0].registers != ports
        or erase_arctan.functions[0].registers != angle_ports
        or len(set(target_names)) != len(target_names)
        or not set(target_names) <= {r.name for r in ports}
    ):
        raise ValueError("Complex phase requires exact THRU pairs and target ownership")
    if controlled and (not ports or not angle_ports):
        raise ValueError("Complex phase positive owners require leading control bits")
    control_port = ports[0] if controlled else None
    angle_control = angle_ports[0] if controlled else None
    if controlled and any(
        port is None
        or port.shape != ()
        or port.dtype != quantum_bit_type()
        or port.dtype.classical_bits
        for port in (control_port, angle_control)
    ):
        raise ValueError("Complex phase positive owners require leading control bits")
    if control_port is not None and control_port.name in target_names:
        raise ValueError("Complex phase control cannot be an encoder target")
    data_ports = ports[1:] if controlled else ports
    angle_data = angle_ports[1:] if controlled else angle_ports
    by_name = {register.name: register for register in data_ports}
    targets = tuple(by_name[name] for name in target_names)
    outer = tuple(r for r in ports if r.name not in target_names)
    angle = {r.name: r for r in angle_data}
    if set(angle) - {"select", "sign", "target"} or "sign" not in angle:
        raise ValueError("Complex phase requires the ArcTan select/sign/target layout")
    width = sum(int(r.total_bits()) for r in targets)
    precision = int(angle["target"].total_bits()) if "target" in angle else 0
    if (
        any(r.dtype.classical_bits for r in targets)
        or width != (int(angle["select"].total_bits()) if "select" in angle else 0)
        or angle["sign"].total_bits() != 1
        or len(phases) != 2 * precision
        or any(
            phase.functions[0].registers is None
            or not Signature(phase.functions[0].registers).thru_registers_only
            or sum(int(r.total_bits()) for r in phase.functions[0].registers)
            != 2 + int(controlled)
            or any(r.bitsize != 1 for r in phase.functions[0].registers)
            for phase in phases
        )
    ):
        raise ValueError("Complex phase requires matching bit layouts and phase stages")
    signature = (
        quantum_control_signature(
            Signature(tuple(r for r in outer if r != control_port)),
            ((quantum_bit_type(), ()),),
        )
        if controlled
        else Signature(outer)
    )
    builder = ComponentProgramBuilder(signature)
    outer_control = tuple(signature)[0] if controlled else None
    control: RegisterBinding | None = None
    if outer_control is not None:
        control_value = builder.inputs[outer_control.name]
        assert isinstance(control_value, RegisterBinding)
        control = control_value
    state = {r.name: builder.inputs[r.name] for r in outer if r != control_port}
    state.update(
        _collections(
            targets,
            (builder.allocate(dtype=r.dtype) for r in targets for _ in r.all_idxs()),
        )
    )
    sign = builder.allocate(dtype=angle["sign"].dtype)
    magnitude = builder.allocate(dtype=angle["target"].dtype) if precision else None
    if control_port is not None:
        assert control is not None
        state[control_port.name] = control
    state = builder.call_program(encoder, state, entry=encoder.functions[0].name)
    if control_port is not None:
        selected_control = state.pop(control_port.name)
        assert isinstance(selected_control, RegisterBinding)
        control = selected_control
    bits = [
        bit
        for word in _elements(state.pop(r.name) for r in targets)
        for bit in builder.split(word)
    ]
    operands = {"sign": sign}
    if width:
        operands["select"] = builder.join(bits, angle["select"].dtype)
    if magnitude is not None:
        operands["target"] = magnitude
    if angle_control is not None:
        assert control is not None
        operands[angle_control.name] = control
    values = builder.call_program(arctan, operands, entry=arctan.functions[0].name)
    if angle_control is not None:
        selected_control = values.pop(angle_control.name)
        assert isinstance(selected_control, RegisterBinding)
        control = selected_control
    returned_sign = values["sign"]
    assert isinstance(returned_sign, RegisterBinding)
    sign = returned_sign
    returned_magnitude = values.pop("target", None)
    if returned_magnitude is not None:
        assert isinstance(returned_magnitude, RegisterBinding)
        digits = builder.split(returned_magnitude)
        for index, phase in enumerate(phases):
            stage_ports = phase.functions[0].registers
            assert stage_ports is not None
            result = builder.call_program(
                phase,
                _collections(
                    stage_ports,
                    (
                        (control, sign, digits[index // 2])
                        if control is not None
                        else (sign, digits[index // 2])
                    ),
                ),
                entry=phase.functions[0].name,
            )
            elements = tuple(_elements(result.values()))
            if controlled:
                control, sign, digits[index // 2] = elements
            else:
                sign, digits[index // 2] = elements
        values["target"] = builder.join(digits, angle["target"].dtype)
    values["sign"] = sign
    if angle_control is not None:
        assert control is not None
        values[angle_control.name] = control
    values = builder.call_program(
        erase_arctan, values, entry=erase_arctan.functions[0].name
    )
    if angle_control is not None:
        selected_control = values.pop(angle_control.name)
        assert isinstance(selected_control, RegisterBinding)
        control = selected_control
    restored = values.pop("select", None)
    if restored is not None:
        assert isinstance(restored, RegisterBinding)
        bits = list(builder.split(restored))
    else:
        bits = []
    words = []
    offset = 0
    for register in targets:
        for _ in register.all_idxs():
            size = int(register.bitsize)
            words.append(builder.join(bits[offset : offset + size], register.dtype))
            offset += size
    state.update(_collections(targets, words))
    if control_port is not None:
        assert control is not None
        state[control_port.name] = control
    state = builder.call_program(
        erase_encoder, state, entry=erase_encoder.functions[0].name
    )
    for binding in _elements(values.values()):
        builder.free(binding)
    if control_port is not None:
        selected_control = state.pop(control_port.name)
        assert isinstance(selected_control, RegisterBinding)
        control = selected_control
    for binding in _elements(state.pop(r.name) for r in targets):
        builder.free(binding)
    if outer_control is not None:
        assert control is not None
        state[outer_control.name] = control
    return builder.finalize(state)
