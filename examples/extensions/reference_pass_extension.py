"""Minimal independently installable compiler-pass extension.

A compiler pass is the smaller of the two Compiler extension kinds: it takes a
FlagQuantum program and returns one, and it makes no claim about a target device.
This example rewrites each ``swap`` into the three ``cx`` gates it equals, which
is a real transformation rather than a placeholder, so a reader can see the
admitted pass change a program.

Copy the five members below, replace ``transform``, and
:func:`flagquantum.ecosystem.extensions.run_compiler_pass_conformance` reports
whether the result meets the contract the crossing enforces.
"""

from __future__ import annotations

from dataclasses import replace

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
)


class ReferenceSwapPass:
    """Rewrite ``swap(low, high)`` as ``cx(low, high); cx(high, low); cx(low, high)``."""

    manifest = ExtensionManifest(
        name="reference_swap_pass",
        version="1.0.0",
        kind="compiler_pass",
        capabilities=frozenset({"circuit_ir"}),
    )

    # The name this pass asks for. It is not the name it is registered under: the
    # crossing builds ``extension.<manifest name>.<this>`` from it, so a pass
    # cannot ask for a built-in name.
    pass_name = "decompose_swap"

    def __init__(self) -> None:
        self.active = False

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
        missing = request.required - self.manifest.capabilities
        return CapabilityResponse(
            not missing,
            self.manifest.capabilities,
            (
                ("missing capabilities: " + ", ".join(sorted(missing)),)
                if missing
                else ()
            ),
        )

    def start(self, config: ExtensionConfig) -> None:
        self.active = True

    def transform(self, program: CircuitIR) -> CircuitIR:
        if not self.active:
            raise RuntimeError("pass is closed")
        instructions: list[Instruction] = []
        for instruction in program.instructions:
            if instruction.name != "swap":
                instructions.append(instruction)
                continue
            low, high = instruction.wires
            instructions.extend(
                (
                    Instruction(name="cx", wires=(low, high)),
                    Instruction(name="cx", wires=(high, low)),
                    Instruction(name="cx", wires=(low, high)),
                )
            )
        # `replace` carries the rest of the program -- dtype, shape, observables,
        # measurements, metadata -- so a pass that touches instructions cannot
        # silently drop a field it did not mean to touch.
        return replace(program, instructions=tuple(instructions))

    def close(self) -> None:
        self.active = False
