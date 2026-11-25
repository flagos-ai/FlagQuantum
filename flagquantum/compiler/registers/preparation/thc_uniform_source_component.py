# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Literal calls follow locked Qualtran v0.7.0 chemistry/thc/prepare.py.

"""Retain the original THC uniform graph without repairing its source defects."""

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..bits.on_each import on_each_word_program
from ..comparison.constant_component import less_than_constant_program
from ..comparison.equality_component import equality_constant_program
from ..comparison.ordering_component import less_equal_program
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..control.multi_control_component import multi_control_x_program
from .identity_preparation_component import identity_preparation_program
from .prepared_reflection_component import prepared_reflection_program


def literal_thc_uniform_pair(
    num_mu: int,
    num_spin_orb: int,
    greater_than_pair: tuple[ProgramRecord, ProgramRecord],
) -> tuple[ProgramRecord, ProgramRecord]:
    """Store all source calls, including the model-only GT and final dirty Free.

    Inputs are MSB-first opaque mu/nu words and three caller-owned QBits.
    Four internal flags preserve the original single-word allocation lifetime.
    This is a logical source graph, not the corrected native mathematical
    preparation. The explicit GreaterThanConstant declarations must remain
    bodyless, so physical binding refuses before reaching the known bad Free.
    """
    if (
        type(num_mu) is not int
        or num_mu < 1
        or type(num_spin_orb) is not int
        or num_spin_orb < 0
    ):
        raise ValueError("Literal THC uniform requires concrete nonnegative dimensions")
    width = num_mu.bit_length()
    bit = quantum_bit_type()
    gt_ports = tuple(
        Signature.build_from_dtypes(x=unsigned_register_type(width), target=bit)
    )
    if any(
        len(record.functions) != 1
        or record.functions[0].registers != gt_ports
        or record.functions[0].blocks
        for record in greater_than_pair
    ):
        raise ValueError("Literal THC uniform retains model-only GT declarations")
    signature = Signature(
        (
            Register("mu", opaque_register_type(width)),
            Register("nu", opaque_register_type(width)),
            *(Register(name, bit) for name in ("nu_eq_mp1", "succ", "rot")),
        )
    )
    h = ProgramRecord(
        (
            bit_gate_function(
                "thc_h", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))
            ),
        )
    )
    x = ProgramRecord(
        (
            bit_gate_function(
                "thc_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
            ),
        )
    )
    toffoli = ProgramRecord(
        (
            bit_gate_function(
                "thc_ccx",
                (Register("ctrl", bit, (2,)), Register("target", bit)),
                CircuitIR(3, (Instruction("ccx", (0, 1, 2)),)),
            ),
        )
    )
    data_size = num_mu * (num_mu + 1) // 2 + num_spin_orb // 2
    # Preserve the original NumPy evaluation order and literal Ry angle.
    angle = float(np.arccos(1 - 2 ** np.floor(np.log2(data_size)) / data_size))
    rotations = tuple(
        ProgramRecord(
            (
                bit_gate_function(
                    f"thc_ry_{index}",
                    (Register("q", bit),),
                    CircuitIR(1, (Instruction("ry", (0,), params={"theta": value}),)),
                ),
            )
        )
        for index, value in enumerate((angle, -angle))
    )
    repeated = on_each_word_program(width, h.functions[0])
    repeated_inverse = adjoint_component_calls(repeated, ((h, h),))
    lt = (
        less_than_constant_program(width, num_mu),
        less_than_constant_program(width, num_mu, adjoint=True),
    )
    equality = (
        equality_constant_program(width, num_mu + 1),
        equality_constant_program(width, num_mu + 1, adjoint=True),
    )
    le = less_equal_program(width)

    def zero_reflection(widths: tuple[int, ...]) -> ProgramRecord:
        ports = tuple(
            Register(f"reg{i}_", opaque_register_type(n)) for i, n in enumerate(widths)
        )
        identity = identity_preparation_program(ports)
        return prepared_reflection_program(
            identity, identity, tuple(reg.name for reg in ports)
        )

    mark, zero = zero_reflection((1, 1, 1, 1)), zero_reflection((width, width, 1))
    mcx = multi_control_x_program((1, 1, 1))
    builder = ComponentProgramBuilder(signature)
    state: dict[str, BindingCollection] = dict(builder.inputs)
    flag_names = ("lte_mu_nu", "lte_nu_mp1", "gt_mu_n", "junk")
    for name, flag in zip(flag_names, builder.split(builder.allocate(4)), strict=True):
        assert isinstance(flag, RegisterBinding)
        state[name] = flag

    def one(program: ProgramRecord, name: str) -> None:
        state[name] = builder.call_program(
            program, {"q": state.pop(name)}, entry=program.functions[0].name
        )["q"]

    def constant(program: ProgramRecord, word: str, flag: str) -> None:
        result = builder.call_program(
            program,
            {"x": state.pop(word), "target": state.pop(flag)},
            entry=program.functions[0].name,
        )
        state[word], state[flag] = result["x"], result["target"]

    def less_equal() -> None:
        result = builder.call_program(
            le,
            {
                "x": state.pop("mu"),
                "y": state.pop("nu"),
                "target": state.pop("lte_mu_nu"),
            },
        )
        state["mu"], state["nu"], state["lte_mu_nu"] = (
            result["x"],
            result["y"],
            result["target"],
        )

    def conjunction() -> None:
        result = builder.call_program(
            toffoli,
            {
                "ctrl": np.asarray(
                    [state.pop("nu_eq_mp1"), state.pop("gt_mu_n")], dtype=object
                ),
                "target": state.pop("junk"),
            },
            entry=toffoli.functions[0].name,
        )
        state["nu_eq_mp1"], state["gt_mu_n"] = np.asarray(result["ctrl"]).flat
        state["junk"] = result["target"]

    def flags(*, equality_flag: bool = True) -> None:
        constant(lt[0], "nu", "lte_nu_mp1")
        less_equal()
        if equality_flag:
            constant(equality[0], "nu", "nu_eq_mp1")
        constant(greater_than_pair[0], "mu", "gt_mu_n")

    one(repeated, "mu")
    one(repeated, "nu")
    one(rotations[0], "rot")
    flags()
    conjunction()
    reflected = builder.call_program(
        mark,
        {
            f"reg{i}_": state.pop(name)
            for i, name in enumerate(("rot", "lte_nu_mp1", "lte_mu_nu", "junk"))
        },
    )
    for i, name in enumerate(("rot", "lte_nu_mp1", "lte_mu_nu", "junk")):
        state[name] = reflected[f"reg{i}_"]
    flags()
    conjunction()
    one(rotations[1], "rot")
    one(repeated, "mu")
    one(repeated, "nu")
    reflected = builder.call_program(
        zero,
        {f"reg{i}_": state.pop(name) for i, name in enumerate(("mu", "nu", "rot"))},
    )
    for i, name in enumerate(("mu", "nu", "rot")):
        state[name] = reflected[f"reg{i}_"]
    one(repeated, "mu")
    one(repeated, "nu")
    flags()
    conjunction()
    recorded = builder.call_program(
        mcx,
        {
            "controls": np.asarray(
                [state.pop(name) for name in ("lte_nu_mp1", "lte_mu_nu", "junk")],
                dtype=object,
            ),
            "target": state.pop("succ"),
        },
    )
    for name, flag in zip(
        ("lte_nu_mp1", "lte_mu_nu", "junk"),
        np.asarray(recorded["controls"]).flat,
        strict=True,
    ):
        state[name] = flag
    state["succ"] = recorded["target"]
    conjunction()
    flags(equality_flag=False)
    one(x, "junk")
    builder.free(builder.join([state.pop(name) for name in flag_names]))
    forward = builder.finalize(state)
    return forward, adjoint_component_calls(
        forward,
        (
            (h, h),
            (x, x),
            (toffoli, toffoli),
            (rotations[0], rotations[1]),
            (rotations[1], rotations[0]),
            (repeated, repeated_inverse),
            lt,
            equality,
            greater_than_pair,
            (le, le),
            (mark, mark),
            (zero, zero),
            (mcx, mcx),
        ),
    )
