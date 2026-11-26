# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Root order follows locked Qualtran v0.7.0 ComplexPhaseOracle.

"""Compose encoder-owned work, bounded ArcTan and source-selected phase calls."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_graph import _collections, _elements


def complex_phase_oracle_program(
    encoder: ProgramRecord,
    erase_encoder: ProgramRecord,
    arctan: ProgramRecord,
    erase_arctan: ProgramRecord,
    phases: tuple[ProgramRecord, ...],
    *,
    target_names: tuple[str, ...],
) -> ProgramRecord:
    """Preserve exact child calls without reversing measurement primitives.

    Encoded words, phase outputs and each child's private work are distinct
    lifetimes. Encoder targets start zero; the supplied inverse must erase them
    after phase evaluation. All other ports remain caller-owned.
    """
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
    by_name = {register.name: register for register in ports}
    targets = tuple(by_name[name] for name in target_names)
    outer = tuple(r for r in ports if r.name not in target_names)
    angle = {r.name: r for r in angle_ports}
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
            or sum(int(r.total_bits()) for r in phase.functions[0].registers) != 2
            or any(r.bitsize != 1 for r in phase.functions[0].registers)
            for phase in phases
        )
    ):
        raise ValueError("Complex phase requires matching bit layouts and phase stages")
    builder = ComponentProgramBuilder(Signature(outer))
    state = dict(builder.inputs)
    state.update(
        _collections(
            targets,
            (builder.allocate(dtype=r.dtype) for r in targets for _ in r.all_idxs()),
        )
    )
    sign = builder.allocate(dtype=angle["sign"].dtype)
    magnitude = builder.allocate(dtype=angle["target"].dtype) if precision else None
    state = builder.call_program(encoder, state, entry=encoder.functions[0].name)
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
    values = builder.call_program(arctan, operands, entry=arctan.functions[0].name)
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
                _collections(stage_ports, (sign, digits[index // 2])),
                entry=phase.functions[0].name,
            )
            sign, digits[index // 2] = _elements(result.values())
        values["target"] = builder.join(digits, angle["target"].dtype)
    values["sign"] = sign
    values = builder.call_program(
        erase_arctan, values, entry=erase_arctan.functions[0].name
    )
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
    state = builder.call_program(
        erase_encoder, state, entry=erase_encoder.functions[0].name
    )
    for binding in _elements(values.values()):
        builder.free(binding)
    for binding in _elements(state.pop(r.name) for r in targets):
        builder.free(binding)
    return builder.finalize(state)
