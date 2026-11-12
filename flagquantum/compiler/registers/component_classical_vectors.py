# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 xcheck_classical_quimb.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/xcheck_classical_quimb.py

"""Surround a typed component with concrete integer states and effects."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace

import numpy as np

from ...core.ir.program import ProgramRecord, verify_program_record
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from .component_basis_vectors import integer_basis_function
from .component_builder import BindingCollection, ComponentProgramBuilder


def _basis_index(value: object) -> int:
    if isinstance(value, (bool, np.integer)):
        return int(value)
    if type(value) is not int:
        raise ValueError("classical vectors require concrete integer basis indices")
    return value


def _add_classical_kets(
    builder: ComponentProgramBuilder,
    registers: Iterable[Register],
    values: Mapping[str, object],
    prefix: str,
) -> dict[str, BindingCollection]:
    bindings: dict[str, BindingCollection] = {}
    count = 0
    for register in registers:
        data = np.asarray(values[register.name]) if register.shape else None
        words = np.empty(register.shape, dtype=object)
        for index in register.all_idxs():
            value = data[index] if data is not None else values[register.name]
            leaf = prefix + "state_" + str(count)
            builder.define(
                integer_basis_function(leaf, register.dtype, _basis_index(value))
            )
            words[index] = builder.call_dict(leaf, {})["val"]
            count += 1
        bindings[register.name] = words if register.shape else words[()]
    return bindings


def _add_classical_bras(
    builder: ComponentProgramBuilder,
    registers: Iterable[Register],
    values: Mapping[str, object],
    bindings: Mapping[str, BindingCollection],
    prefix: str,
) -> None:
    count = 0
    for register in registers:
        data = np.asarray(values[register.name]) if register.shape else None
        array_bindings = bindings[register.name] if register.shape else None
        if isinstance(array_bindings, RegisterBinding):
            raise ValueError(
                f"bindings {register.name} must be a numpy array: {array_bindings}"
            )
        for index in register.all_idxs():
            value = data[index] if data is not None else values[register.name]
            leaf = prefix + "effect_" + str(count)
            builder.define(
                integer_basis_function(
                    leaf, register.dtype, _basis_index(value), state=False
                )
            )
            binding = (
                array_bindings[index]
                if array_bindings is not None
                else bindings[register.name]
            )
            assert isinstance(binding, RegisterBinding)
            builder.call_dict(leaf, {"val": binding})
            count += 1


def flank_component_with_classical_vectors(
    record: ProgramRecord,
    inputs: Mapping[str, object],
    outputs: Mapping[str, object] | None = None,
    *,
    evaluate: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    entry: str = "main",
    name: str = "classical_vectors",
) -> ProgramRecord:
    """Add input kets, one child call and output effects in source traversal order.

    Values are unsigned basis indices, independently of declared numeric dtype.
    Arrays use asarray and row-major indexing without broadcasting. If outputs
    is None, call the explicit evaluator once after input preparation and the
    child call, passing the original inputs mapping unchanged. Explicit outputs,
    including an empty mapping, bypass evaluation.

    Original functions and record metadata are retained; canonical_payload is
    invalidated. Generated names avoid existing definitions and the wrapper.
    Existing builder and integer-loader admission apply: finite logical quantum
    ports and positive word widths. Integral NumPy scalars and bool become
    built-in integers; floating/symbolic basis indices are explicitly refused.
    No numerical execution, decomposition discovery or backend selection occurs.
    """
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid classical-vector graph: {diagnostics}")
    functions = {function.name: function for function in record.functions}
    if entry not in functions:
        raise ValueError(f"unknown classical-vector entry {entry!r}")
    child = functions[entry]
    if record.physical_context is not None or child.registers is None:
        raise ValueError("classical vectors require a typed logical entry")
    if name in functions:
        raise ValueError("classical-vector entry conflicts with an existing definition")
    signature = Signature(child.registers)
    if any(register.dtype.classical_bits for register in signature):
        raise ValueError("classical vectors do not admit classical ports")
    prefix_index = 0
    while any(
        key.startswith(f"__fq_vector_{prefix_index}_") for key in (*functions, name)
    ):
        prefix_index += 1
    prefix = f"__fq_vector_{prefix_index}_"
    builder = ComponentProgramBuilder(Signature(()), record.functions, name=name)
    bindings = _add_classical_kets(builder, signature.lefts(), inputs, prefix)
    results = builder.call_dict(entry, bindings)
    if outputs is None:
        if evaluate is None:
            raise ValueError("omitted outputs require an explicit classical evaluator")
        outputs = evaluate(inputs)
    _add_classical_bras(builder, signature.rights(), outputs, results, prefix)
    module = builder.finalize({})
    return replace(record, canonical_payload=None, functions=module.functions)
