# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Call graph follows locked Qualtran v0.7.0 state_preparation_alias_sampling.py.

"""Own the shared dense/sparse five-call alias preparation graph."""

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..bits.on_each import on_each_word_program
from ..comparison.ordering_component import less_equal_program
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..lookup.qrom_component import qrom_program
from ..lookup.qrom_table import QromTable
from ..primitives.controlled_swap_component import controlled_swap_program
from .uniform_preparation_component import uniform_preparation_program


def alias_preparation_component_program(
    signature: Signature,
    lookup_table: QromTable,
    *,
    rows: SymbolicInt,
    selection_name: str = "selection",
    sparse: bool = False,
    adjoint: bool = False,
) -> ProgramRecord:
    """Keep typed logical junk outputs and privately erased child workspace.

    All words are MSB-first THRU ports. The caller supplies the exact source
    selection metadata and lookup table, without an Algorithms object crossing
    into Compiler. Exact inverse reverses component calls, not measured leaves.
    Shape-only child declarations retain the existing physical refusal.
    """
    if type(adjoint) is not bool or type(sparse) is not bool:
        raise ValueError("Alias preparation direction and layout must be boolean")
    mu = signature.get_left("sigma_mu").bitsize
    width = signature.get_left(selection_name).bitsize
    builder = ComponentProgramBuilder(signature)
    words = dict(builder.inputs)
    address = "sparse_index" if sparse else selection_name
    uniform = uniform_preparation_program(rows, adjoint=adjoint)
    bit = quantum_bit_type()
    if type(mu) is int:
        hadamard = bit_gate_function(
            "alias_h", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))
        )
        sigma = on_each_word_program(mu, hadamard)
        if adjoint:
            leaf = ProgramRecord((hadamard,))
            sigma = adjoint_component_calls(sigma, ((leaf, leaf),))
        compare = less_equal_program(mu)
    else:
        sigma = ProgramRecord(
            (FunctionRecord("main", (), registers=tuple(Signature.build(q=mu))),)
        )
        compare = ProgramRecord(
            (
                FunctionRecord(
                    "main",
                    (),
                    registers=tuple(
                        Signature.build_from_dtypes(
                            x=unsigned_register_type(mu),
                            y=unsigned_register_type(mu),
                            target=bit,
                        )
                    ),
                ),
            )
        )
    swap = (
        controlled_swap_program(width)
        if type(width) is int
        else ProgramRecord(
            (
                FunctionRecord(
                    "main",
                    (),
                    registers=tuple(Signature.build(ctrl=1, x=width, y=width)),
                ),
            )
        )
    )
    lookup = qrom_program(lookup_table)
    lookup_words = (selection_name, "alt", "keep") if sparse else ("alt", "keep")
    steps = [
        (uniform, {"target": address}),
        (sigma, {"q": "sigma_mu"}),
        (
            lookup,
            {
                "selection": address,
                **{f"target{i}_": name for i, name in enumerate(lookup_words)},
            },
        ),
        (compare, {"x": "keep", "y": "sigma_mu", "target": "less_than_equal"}),
        (swap, {"ctrl": "less_than_equal", "x": "alt", "y": selection_name}),
    ]
    for child, names in reversed(steps) if adjoint else steps:
        outputs = builder.call_program(
            child, {port: words[name] for port, name in names.items()}
        )
        for port, name in names.items():
            words[name] = outputs[port]
    return builder.finalize(words)
