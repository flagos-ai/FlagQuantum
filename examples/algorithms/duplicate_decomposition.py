# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/lists/has_duplicates.py.

"""Present the literal duplicate-detection graph, including source erasure aliases."""

import numpy as np

from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.control.multi_control_component import (
    multi_control_x_program,
)
from flagquantum.compiler.registers.sorting.sorted_duplicates_component import (
    sorted_duplicates_program,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.basis_symbols import pauli_basis_wire_symbol
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.partition_symbols import partition_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)


def _symbolic_duplicates(ports: Signature, length: int) -> ProgramRecord:
    dtype, bit = ports[0].dtype, quantum_bit_type()
    children = tuple(
        FunctionRecord(
            name,
            (),
            registers=(
                Register("a", dtype),
                Register("b", dtype),
                Register("c", unsigned_register_type(dtype.num_qubits + 1), side=side),
                Register("target", bit, side=side),
            ),
        )
        for name, side in (
            ("half_lt", RegisterSide.RIGHT),
            ("half_lt_inverse", RegisterSide.LEFT),
        )
    )
    x = FunctionRecord("X", (), registers=tuple(Signature.build_from_dtypes(q=bit)))
    builder = ComponentProgramBuilder(ports, (*children, x))
    xs = np.asarray(builder.inputs["xs"]).copy()
    history: list[BindingCollection] = []
    flags: list[BindingCollection] = []
    for i in range(1, length):
        result = builder.call_dict(children[0].name, {"a": xs[i - 1], "b": xs[i]})
        xs[i - 1], xs[i] = result["a"], result["b"]
        history.append(result["c"])
        flags.append(result["target"])
    result = builder.call_program(
        multi_control_x_program((1,) * len(flags)),
        {"controls": np.array(flags, dtype=object), "target": builder.inputs["flag"]},
    )
    flags = list(np.asarray(result["controls"]))
    flag = builder.call_dict(x.name, {"q": result["target"]})["q"]
    for i in reversed(range(1, length)):
        result = builder.call_dict(
            children[1].name,
            {
                "a": xs[i - 1],
                "b": xs[i],
                "c": history.pop(),
                "target": flags.pop(),
            },
        )
        xs[i - 1], xs[i] = result["a"], result["b"]
    return builder.finalize({"xs": xs, "flag": flag})


def duplicate_catalog_program(operation: str, ports: Signature) -> ProgramRecord | None:
    """Reuse the literal concrete owner and declare the symbolic-width source graph.

    The catalog contains uncontrolled four-word instances. Symbolic list length
    remains outside finite graph construction, matching the source refusal.
    """
    if operation != "HasDuplicates":
        return None
    length = ports[0].shape_symbolic[0]
    if type(length) is not int:
        return None
    if type(ports[0].bitsize) is int:
        return sorted_duplicates_program(length, ports[0].dtype)
    return _symbolic_duplicates(ports, length)


def duplicate_symbols(record: ProgramRecord) -> RegisterWireSymbolSupplier:
    """Reuse typed comparison ports, Pauli X and literal MCX glyphs."""
    functions = {f.name: f for f in record.functions}

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        ports = tuple(r.name for r in functions[step.name].registers or ())
        if ports == ("q",):
            return pauli_basis_wire_symbol(step, register, index, gate="x")
        if ports == ("controls", "target"):
            if register is None:
                count = (functions[step.name].registers or ())[0].shape[0]
                return "TextBox", {"text": f"C^{count}X"}
            return (
                ("ModPlus", {})
                if register.name == "target"
                else ("Circle", {"filled": True})
            )
        if register is None:
            return "Text", {"text": "LinearDe..", "fontsize": 10}
        return partition_wire_symbol(step, register, index)

    return supply


def duplicate_score(operation: str, ports: Signature) -> MusicalScoreData | None:
    """Draw all adjacent comparisons, flag update and reverse erasure calls."""
    record = duplicate_catalog_program(operation, ports)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=duplicate_symbols(record),
    )
