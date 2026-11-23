# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix py/circuit/subs/fuzz_test_util.py.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Assert named physical-register behavior using the existing numerical session."""

import random
from collections.abc import Callable

from ..compiler.artifacts.operand_array import OperandArray
from ..compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from ..compiler.formats.kmx_circuit import KmxCircuit
from ..ecosystem._physical_basis_session import PhysicalBasisSession


def indentation_fix(code: str) -> str:
    saw_indentation = False
    for character in code:
        if character in ("\n", "\r"):
            saw_indentation = False
            continue
        if character in (" ", "\t"):
            saw_indentation = True
            continue
        break
    if saw_indentation:
        code = "if True:\n" + code
    return """
_global_phase = False
def Z(b):
    global _global_phase
    _global_phase ^= b
def neg():
    Z(True)
""" + code


def assert_fuzz_testing_acts_like(
    circuit: KmxCircuit,
    code: str,
    *,
    shots: int,
    context: dict[str, object] | None = None,
    input_sampler: Callable[[], dict[str, int]] | None = None,
    ignore_phase: bool = False,
) -> None:
    """Check each chunk, including later chunks missed by the locked source."""
    if context is None:
        context = {}
    registers = {}
    for index, register in enumerate(circuit.registers):
        if not register.name:
            raise AssertionError(
                f"Unnamed register {OperandIdentity(OperandKind.REGISTER, index)!r}"
            )
        registers[register.name] = register
    used = {
        wire
        for register in registers.values()
        for kind, wire in register.members
        if kind == "q"
    }
    unused = OperandArray(
        OperandIdentity(OperandKind.QUANTUM, wire)
        for wire in range(circuit.qubits)
        if wire not in used
    )
    sim = PhysicalBasisSession(min(shots, 256))
    sim.use_same_registers_as(circuit)
    remaining = shots
    while remaining > 0:
        lanes = range(min(remaining, 256))
        remaining -= len(lanes)
        sim.clear_for_shot()
        inputs = []
        expected = []
        for shot in lanes:
            sample = (
                {
                    name: random.randrange(1 << len(register.members))
                    for name, register in registers.items()
                }
                if input_sampler is None
                else input_sampler()
            )
            assert sample.keys() == registers.keys()
            sample = {name: sample[name] for name in registers}
            for name in registers:
                sim.write_within_shot(name, shot, sample[name])
            local: dict[str, object] = {**sample, **context}
            globals_: dict[str, object] = {}
            exec(indentation_fix(code), globals_, local)
            expected.append(
                {
                    **{name: local[name] for name in registers},
                    "@phase": globals_["_global_phase"],
                }
            )
            inputs.append(dict(sample))
        sim.do(circuit)
        for shot in lanes:
            actual = {
                name: sim.read_within_shot(name, shot, out=int) for name in registers
            }
            actual["@unused"] = sim.read_within_shot(unused, shot, out=int)
            actual["@phase"] = sim.read_shot_phase(shot)
            wanted = expected[shot]
            if (
                any(actual[name] != wanted[name] for name in registers)
                or actual["@unused"] != 0
                or not ignore_phase
                and actual["@phase"] != wanted["@phase"]
            ):
                lines = ["Fuzz testing failed"]
                for name in registers:
                    if actual[name] != wanted[name]:
                        lines.append(
                            f"    FAIL:register {name!r} = {inputs[shot][name]} -> {actual[name]} != {wanted[name]}"
                        )
                    else:
                        lines.append(
                            f"    register {name!r} = {inputs[shot][name]} -> {actual[name]}"
                        )
                if actual["@phase"] != wanted["@phase"]:
                    lines.append(
                        f"    FAIL:@phase={actual['@phase']} != {wanted['@phase']})"
                    )
                else:
                    lines.append("    @phase=" + str(actual["@phase"]))
                lines.append(
                    ("    FAIL:@unused=" if actual["@unused"] else "    @unused=")
                    + str(actual["@unused"])
                )
                lines.extend(
                    f"    var[{name!r}] = {value!r}" for name, value in context.items()
                )
                raise AssertionError("\n".join(lines))
