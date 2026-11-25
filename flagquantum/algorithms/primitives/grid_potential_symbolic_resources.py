# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Models follow locked Qualtran v0.7.0 grid_ham/potential.py and QubitCount.

"""Query source potential formulas over the complete owned input call graph."""

from collections.abc import Mapping
from dataclasses import fields
from typing import cast

import numpy as np

from ...compiler._component_trace.program_cost_query import query_program_qec_gate_cost
from ...compiler._component_trace.program_logical_width import (
    query_program_logical_width,
)
from ...compiler._component_trace.qec_gate_cost import QECGateCost
from ...compiler.formats.grid_potential_graph import grid_pair_potential_library
from ...compiler.formats.legacy_bookkeeping_factories import bookkeeping_factories
from ...compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
    resolve_component_program,
)
from ...compiler.registers.arithmetic.out_of_place_add_resource_model import (
    source_out_of_place_add_resource_formula,
)
from ...compiler.registers.lookup.qrom_resource_model import source_qrom_cost
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, smax
from .grid_inverse_sqrt import source_grid_inverse_sqrt_resource_formula


def _leaf_formula(
    kind: str,
    registers: tuple[Register, ...],
    arguments: Mapping[str, ResolvedArgument],
    *,
    gate_query: bool,
    legacy_shims: bool,
) -> SymbolicResourceFormula:
    def dimension(key: str) -> SymbolicInt:
        return cast(SymbolicInt, arguments[key])

    width = Signature(registers).n_qubits()
    if kind in {"Allocate", "Free"}:
        return SymbolicResourceFormula(0, 0, 0, smax(0, width))
    if kind == "Cast":
        return SymbolicResourceFormula(0, 0, 0, width)
    if kind == "OutOfPlaceAdder":
        return source_out_of_place_add_resource_formula(
            dimension("bitsize"),
            adjoint=bool(arguments.get("is_adjoint", False)),
            legacy_shims=legacy_shims,
        )
    if kind == "SumOfSquares":
        bits, count = dimension("bitsize"), cast(int, arguments["k"])
        return SymbolicResourceFormula(
            0,
            0,
            0,
            smax(3, count * bits, width),
            toffoli=count * bits**2 - bits - int(count % 3 == 0),
        )
    if kind == "QROM":
        selection = cast(tuple[SymbolicInt, ...], arguments["selection_bitsizes"])[0]
        targets = cast(tuple[SymbolicInt, ...], arguments["target_bitsizes"])
        peak = sum(targets) + 2 * selection - 1
        if not gate_query:
            return SymbolicResourceFormula(0, 0, 0, peak)
        data = tuple(
            tuple(int(value) for value in bank)
            for bank in np.asarray(arguments["data_or_shape"])
        )
        if all(len(set(bank)) == 1 for bank in data) and any(
            type(value) is not int for value in (selection, *targets)
        ):
            # The original early-break path requests an unavailable Cirq
            # decomposition instead of using unary-iteration call counts.
            raise NotImplementedError(
                "Source QROM cannot decompose parameterized uniform data"
            )
        # The source unary traversal tests this bound before visiting data.
        # An undecidable symbolic selection must retain its original TypeError.
        if not 2**selection >= len(data[0]):
            raise AssertionError
        target = max(1, max(value.bit_length() for bank in data for value in bank))
        cost = source_qrom_cost(
            0,
            (int(selection),),
            (len(data[0]),),
            ((1, target),) * len(targets),
            data,
            legacy_shims=legacy_shims,
        ).gates
        return SymbolicResourceFormula(
            cost.and_bloq,
            cost.clifford,
            cost.measurement,
            peak,
            t=cost.t,
            toffoli=cost.toffoli,
            cswap=cost.cswap,
            rotation=cost.rotation,
        )
    if kind in {
        "PolynmomialEvaluationInverseSquareRoot",
        "NewtonRaphsonApproxInverseSquareRoot",
    }:
        newton = kind == "NewtonRaphsonApproxInverseSquareRoot"
        return source_grid_inverse_sqrt_resource_formula(
            dimension("x_sq_bitsize"),
            dimension("poly_bitsize"),
            dimension("target_bitsize" if newton else "out_bitsize"),
            newton=newton,
            legacy_shims=legacy_shims,
        )
    if kind == "QuantumVariableRotation":
        return SymbolicResourceFormula(0, 0, 0, smax(1, width), rotation=width)
    raise NotImplementedError(f"No source potential leaf model for {kind!r}")


def _query(
    arguments: Mapping[str, ResolvedArgument], *, gate_query: bool, legacy_shims: bool
) -> SymbolicResourceFormula:
    library = grid_pair_potential_library(arguments)
    widths: dict[str, SymbolicInt] = {}
    gates: dict[str, QECGateCost] = {}

    def factory(kind: str) -> ComponentFactory:
        def declare(
            name: str,
            registers: tuple[Register, ...] | None,
            values: Mapping[str, ResolvedArgument],
        ) -> FunctionRecord | ProgramRecord:
            assert registers is not None
            formula = _leaf_formula(
                kind,
                registers,
                values,
                gate_query=gate_query,
                legacy_shims=legacy_shims,
            )
            widths[name] = formula.logical_qubits
            gates[name] = QECGateCost(
                **{
                    field.name: getattr(formula, field.name)
                    for field in fields(QECGateCost)
                }
            )
            if kind in {"Allocate", "Free", "Cast"}:
                return bookkeeping_factories()[
                    f"qualtran.bloqs.bookkeeping.{kind.lower()}.{kind}"
                ](name, registers, values)
            return FunctionRecord(name, (), registers=registers)

        return declare

    # Only children are factories: the stored root connections remain authoritative.
    factories = {
        entry.declaration.name: factory(entry.declaration.name.rsplit(".", 1)[-1])
        for entry in library.entries[1:]
    }
    program = resolve_component_program(library, 0, factories)
    peak = query_program_logical_width(program, static_costs=widths)
    total = (
        query_program_qec_gate_cost(program, {}, static_costs=gates)
        if gate_query
        else QECGateCost()
    )
    return SymbolicResourceFormula(
        logical_qubits=peak,
        **{field.name: getattr(total, field.name) for field in fields(QECGateCost)},
    )


def source_grid_pair_potential_resource_formula(
    arguments: Mapping[str, ResolvedArgument], *, legacy_shims: bool = False
) -> SymbolicResourceFormula:
    """Query source gate categories and peak over data-bearing constructor metadata.

    These are source model estimates, not emitted primitive counts. A symbolic
    selection bound retains the original gate-query failure; its independent
    logical-width query remains available below.
    """
    return _query(arguments, gate_query=True, legacy_shims=legacy_shims)


def source_grid_pair_potential_logical_width(
    arguments: Mapping[str, ResolvedArgument],
) -> SymbolicInt:
    """Query the source's successful symbolic peak without querying gate cost."""
    return _query(arguments, gate_query=False, legacy_shims=False).logical_qubits
