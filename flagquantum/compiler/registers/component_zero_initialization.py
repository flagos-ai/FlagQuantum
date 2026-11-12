# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _add_classical_kets and initialize_from_zero.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_quimb.py

"""Wrap a typed component with explicit owned zero-initialized inputs."""

from dataclasses import replace

import numpy as np

from ...core.ir.program import ProgramRecord, verify_program_record
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...core.ir.program.signatures import Signature
from .component_builder import BindingCollection, ComponentProgramBuilder
from .layout.component_workspace import allocate_word_function


def initialize_component_from_zero(
    record: ProgramRecord, entry: str = "main", name: str = "from_zero"
) -> ProgramRecord:
    """Add a no-input entry that calls the child with fresh clean words.

    Input elements are prepared in signature then row-major order, matching
    the source IntState(0) loop. Existing clean allocation owns their logical
    zero-state semantics; no gate, reset, tensor or backend is created here.
    Output names, shapes, dtypes and order are retained, with RIGHT direction.
    Original functions and record metadata stay intact; canonical_payload is
    invalidated. Generated allocation names avoid existing definitions.

    Only verified logical typed functions with quantum ports are admitted.
    Unknown entries, wrapper-name collisions and classical inputs raise
    ValueError. Classical outputs are retained without quantum preparation.
    Existing builder and register validation failures propagate.
    Zero-width words retain ownership; empty arrays make no preparation calls.
    Physical binding later requires enough caller-owned clean output wires.
    """
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid zero-initialization graph: {diagnostics}")
    functions = {function.name: function for function in record.functions}
    if entry not in functions:
        raise ValueError(f"unknown zero-initialization entry {entry!r}")
    child = functions[entry]
    if record.physical_context is not None or child.registers is None:
        raise ValueError("zero initialization requires a typed logical entry")
    if name in functions:
        raise ValueError(
            "zero-initialization entry conflicts with an existing definition"
        )
    signature = Signature(child.registers)
    if any(register.dtype.classical_bits for register in signature.lefts()):
        raise ValueError("zero initialization does not admit classical inputs")
    prefix_index = 0
    while any(key.startswith(f"__fq_zero_{prefix_index}_") for key in functions):
        prefix_index += 1
    prefix = f"__fq_zero_{prefix_index}_"
    preparations: dict[RegisterType, str] = {}
    definitions = list(record.functions)
    for register in signature.lefts():
        if any(size == 0 for size in register.shape) or register.dtype in preparations:
            continue
        preparation = prefix + str(len(preparations))
        definitions.append(allocate_word_function(preparation, register.dtype, False))
        preparations[register.dtype] = preparation
    outputs = Signature(
        Register(r.name, r.dtype, r.shape, RegisterSide.RIGHT)
        for r in signature.rights()
    )
    builder = ComponentProgramBuilder(outputs, definitions, name=name)
    inputs: dict[str, BindingCollection] = {}
    for register in signature.lefts():
        values = np.empty(register.shape, dtype=object)
        for index in register.all_idxs():
            values[index] = builder.call_dict(preparations[register.dtype], {})["reg"]
        inputs[register.name] = values if register.shape else values[()]
    module = builder.finalize(builder.call_dict(entry, inputs))
    return replace(record, canonical_payload=None, functions=module.functions)
