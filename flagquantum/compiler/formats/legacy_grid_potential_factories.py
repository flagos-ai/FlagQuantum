# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 chemistry/trotter/grid_ham/potential.py.

"""Reconstruct both literal source potential recipes from owned metadata."""

from collections.abc import Callable, Mapping
from itertools import combinations
from typing import cast

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
)
from ...core.ir.program.registers import Register, RegisterType
from ...errors import SerializationError
from ..registers.arithmetic.grid_kinetic_signature import (
    grid_coordinate_width,
    grid_kinetic_signature,
)
from ..registers.arithmetic.grid_potential_signature import (
    grid_pair_potential_signature,
)
from ..registers.component_builder import ComponentProgramBuilder
from ..registers.lookup.qrom_table import QromTable
from .legacy_arithmetic_factories import _out_of_place_factory
from .legacy_bookkeeping_factories import bookkeeping_factories
from .legacy_component_arguments import ComponentArgument, ComponentArgumentValue
from .legacy_component_export import component_library_from_program
from .legacy_component_library import ComponentDeclaration, ComponentLibrary
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_grid_inverse_sqrt_factories import grid_inverse_sqrt_factories
from .legacy_grid_qvr_factories import grid_qvr_factories
from .legacy_modular_addition_factories import _integer
from .legacy_product_model_factories import product_model_factories
from .legacy_qrom_factories import qrom_factories

CoefficientData = Callable[[int, int], tuple[tuple[int, ...], ...]]
_PAIR = "qualtran.bloqs.chemistry.trotter.grid_ham.potential.PairPotential"
_TOTAL = "qualtran.bloqs.chemistry.trotter.grid_ham.potential.PotentialEnergy"


def grid_potential_factories(
    *,
    coefficient_data: CoefficientData | None = None,
    _pair_graphs: dict[str, ComponentLibrary] | None = None,
) -> dict[str, ComponentFactory]:
    """Keep numerical coefficient conversion outside source recipe construction."""
    children = {
        **bookkeeping_factories(),
        **product_model_factories(),
        **grid_inverse_sqrt_factories(),
        **grid_qvr_factories(),
        **qrom_factories(),
    }

    def pair(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        if not {"bitsize", "qrom_data"} <= arguments.keys() or arguments.keys() - {
            "bitsize",
            "qrom_data",
            "poly_bitsize",
            "inv_sqrt_bitsize",
            "label",
        }:
            raise SerializationError("Invalid pair potential constructor arguments")
        width = _integer(arguments, "bitsize")
        poly = _integer(
            {"poly_bitsize": arguments.get("poly_bitsize", 15)}, "poly_bitsize"
        )
        target = _integer(
            {"inv_sqrt_bitsize": arguments.get("inv_sqrt_bitsize", 24)},
            "inv_sqrt_bitsize",
        )
        if not isinstance(arguments.get("label", "V"), str):
            raise SerializationError("Potential label must be text")
        signature = grid_pair_potential_signature(width)
        if registers is not None and registers != tuple(signature):
            raise SerializationError("Invalid pair potential register signature")
        symbolic_lookup = any(type(value) is not int for value in (width, poly))
        if symbolic_lookup and _pair_graphs is None:
            return FunctionRecord(name, (), registers=tuple(signature))
        raw = arguments["qrom_data"]
        if not isinstance(raw, (tuple, list, np.ndarray)) or len(raw) != 4:
            raise SerializationError(
                "Pair potential requires four concrete coefficient banks"
            )
        data = []
        for bank in raw:
            if (
                not isinstance(bank, (tuple, list, np.ndarray))
                or np.asarray(bank).ndim != 1
            ):
                raise SerializationError("Coefficient banks must be concrete vectors")
            data.append(tuple(bank))
        builder = ComponentProgramBuilder(signature, name=name)
        call_declarations: list[ComponentDeclaration] = []
        system_i, system_j = (
            np.asarray(builder.inputs[key], dtype=object)
            for key in ("system_i", "system_j")
        )

        def call(
            key: str,
            role: str,
            args: Mapping[str, ResolvedArgument],
            bindings: Mapping[str, object],
        ) -> dict[str, object]:
            if key == "qualtran.bloqs.data_loading.qrom.QROM" and symbolic_lookup:
                concrete = QromTable.from_data(*data)
                table = QromTable(
                    concrete.shape,
                    (2 * (width + 1) + 2,),
                    (poly,) * 4,
                    ((),) * 4,
                    datasets=concrete.datasets,
                )
                record: FunctionRecord | ProgramRecord = FunctionRecord(
                    name + "_" + role, (), registers=tuple(table.signature)
                )
            else:
                factory = _out_of_place_factory if key == "adder" else children[key]
                record = factory(name + "_" + role, None, args)
            module = (
                record
                if isinstance(record, ProgramRecord)
                else ProgramRecord((record,))
            )
            if _pair_graphs is not None:
                identifier = (
                    "qualtran.bloqs.arithmetic.addition.OutOfPlaceAdder"
                    if key == "adder"
                    else key
                )
                call_declarations.append(
                    ComponentDeclaration(
                        identifier,
                        tuple(
                            ComponentArgument(
                                field, cast(ComponentArgumentValue, value)
                            )
                            for field, value in args.items()
                        ),
                        module.functions[0].registers,
                    )
                )
            return dict(
                builder.call_program(module, bindings, entry=module.functions[0].name)
            )

        difference_values = []
        for axis in range(3):
            result = call(
                "adder",
                "add",
                {"bitsize": width},
                {"a": system_i[axis], "b": system_j[axis]},
            )
            system_i[axis], system_j[axis] = result["a"], result["b"]
            difference_values.append(result["c"])
        result = call(
            "qualtran.bloqs.arithmetic.multiplication.SumOfSquares",
            "square",
            {"bitsize": width + 1, "k": 3},
            {"input": np.asarray(difference_values, dtype=object)},
        )
        differences, sos = np.asarray(result["input"], dtype=object), result["result"]
        for axis in range(3):
            result = call(
                "adder",
                "add_inverse",
                {"bitsize": width, "is_adjoint": True},
                {"a": system_i[axis], "b": system_j[axis], "c": differences[axis]},
            )
            system_i[axis], system_j[axis] = result["a"], result["b"]

        assert isinstance(sos, RegisterBinding)
        original_dtype = sos.register.dtype
        allocated = [
            call(
                "qualtran.bloqs.bookkeeping.allocate.Allocate",
                "allocate_poly",
                {"dtype": opaque_register_type(poly)},
                {},
            )["reg"]
            for _ in range(4)
        ]
        bound = bounded_unsigned_register_type(2 * (width + 1) + 2, len(data[0]))
        sos = call(
            "qualtran.bloqs.bookkeeping.cast.Cast",
            "selection_view",
            {"inp_dtype": original_dtype, "out_dtype": bound},
            {"reg": sos},
        )["reg"]
        lookup = call(
            "qualtran.bloqs.data_loading.qrom.QROM",
            "lookup",
            {
                "data_or_shape": tuple(np.array(bank) for bank in data),
                "selection_bitsizes": (2 * (width + 1) + 2,),
                "target_bitsizes": (poly,) * 4,
            },
            {
                "selection": sos,
                **{f"target{i}_": value for i, value in enumerate(allocated)},
            },
        )
        sos = call(
            "qualtran.bloqs.bookkeeping.cast.Cast",
            "selection_view_inverse",
            {"inp_dtype": bound, "out_dtype": original_dtype},
            {"reg": lookup["selection"]},
        )["reg"]
        coefficients = [lookup[f"target{i}_"] for i in range(4)]
        poly_out = call(
            "qualtran.bloqs.bookkeeping.allocate.Allocate",
            "allocate_poly",
            {"dtype": opaque_register_type(poly)},
            {},
        )["reg"]
        prefix = "qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt."
        result = call(
            prefix + "PolynmomialEvaluationInverseSquareRoot",
            "polynomial",
            {
                "x_sq_bitsize": 2 * (width + 1) + 2,
                "poly_bitsize": poly,
                "out_bitsize": poly,
            },
            {
                "x_sq": sos,
                "in_coeff": np.asarray(coefficients, dtype=object),
                "out": poly_out,
            },
        )
        sos, coefficients, poly_out = (
            result["x_sq"],
            list(np.asarray(result["in_coeff"], dtype=object)),
            result["out"],
        )
        inverse = call(
            "qualtran.bloqs.bookkeeping.allocate.Allocate",
            "allocate_inverse",
            {"dtype": opaque_register_type(target)},
            {},
        )["reg"]
        result = call(
            prefix + "NewtonRaphsonApproxInverseSquareRoot",
            "newton",
            {
                "x_sq_bitsize": 2 * (width + 1) + 2,
                "poly_bitsize": poly,
                "target_bitsize": target,
            },
            {"x_sq": sos, "poly": poly_out, "target": inverse},
        )
        sos, poly_out = result["x_sq"], result["poly"]
        inverse = call(
            "qualtran.bloqs.chemistry.trotter.grid_ham.qvr.QuantumVariableRotation",
            "phase",
            {"phi_bitsize": target},
            {"phi": result["target"]},
        )["phi"]
        release_roles: dict[RegisterType, str] = {}
        for value in (sos, inverse, *coefficients, poly_out):
            assert isinstance(value, RegisterBinding)
            call(
                "qualtran.bloqs.bookkeeping.free.Free",
                release_roles.setdefault(
                    value.register.dtype, f"free_{len(release_roles)}"
                ),
                {"dtype": value.register.dtype},
                {"reg": value},
            )
        program = builder.finalize({"system_i": system_i, "system_j": system_j})
        if _pair_graphs is not None:
            declarations = {
                name: ComponentDeclaration(
                    _PAIR,
                    tuple(
                        ComponentArgument(field, cast(ComponentArgumentValue, value))
                        for field, value in arguments.items()
                    ),
                    tuple(signature),
                )
            }
            operations = program.functions[0].blocks[0].operations
            for operation, declaration in zip(
                operations, call_declarations, strict=True
            ):
                assert operation.callee is not None
                declarations.setdefault(operation.callee, declaration)
            _pair_graphs[name] = component_library_from_program(
                program, name, declarations, name=name
            )
        return program

    def total(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        if not {"num_elec", "num_grid"} <= arguments.keys() or arguments.keys() - {
            "num_elec",
            "num_grid",
            "poly_bitsize",
            "label",
        }:
            raise SerializationError("Invalid total potential constructor arguments")
        electrons, grid = _integer(arguments, "num_elec"), _integer(
            arguments, "num_grid"
        )
        poly = _integer(
            {"poly_bitsize": arguments.get("poly_bitsize", 15)}, "poly_bitsize"
        )
        label = arguments.get("label", "V")
        if not isinstance(label, str):
            raise SerializationError("Potential label must be text")
        if type(grid) is not int:
            raise SerializationError(
                "Potential bit-length sizing requires a concrete grid"
            )
        assert isinstance(grid, int)
        signature = grid_kinetic_signature(electrons, grid)
        if registers is not None and registers != tuple(signature):
            raise SerializationError("Invalid total potential register signature")
        if type(electrons) is not int or type(poly) is not int:
            return FunctionRecord(name, (), registers=tuple(signature))
        assert isinstance(electrons, int) and isinstance(poly, int)
        if coefficient_data is None:
            raise SerializationError(
                "Total potential requires an owned coefficient-data provider"
            )
        width = grid_coordinate_width(grid)
        data = coefficient_data(2 * width + 2, poly)
        builder = ComponentProgramBuilder(signature, name=name)
        system = np.asarray(builder.inputs["system"], dtype=object)
        if electrons >= 2:
            recipe = pair(
                name + "_pair",
                None,
                {
                    "bitsize": width,
                    "qrom_data": data,
                    "poly_bitsize": poly,
                    "label": label,
                },
            )
            assert isinstance(recipe, ProgramRecord)
            for i, j in combinations(range(electrons), 2):
                result = builder.call_program(
                    recipe,
                    {"system_i": system[i], "system_j": system[j]},
                    entry=recipe.functions[0].name,
                )
                system[i], system[j] = result["system_i"], result["system_j"]
        return builder.finalize({"system": system})

    return {_PAIR: pair, _TOTAL: total}
