# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from locked Qualtran v0.7.0 rotations/phasing_via_cost_function.py.

"""Resolve cost phasing with independently owned evaluation and exact erasure."""

from collections.abc import Callable, Mapping
from dataclasses import replace

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.component_graph import _namespace_helpers
from ..registers.phase.cost_function_component import cost_function_phase_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_product_model_factories import product_model_factories
from .legacy_qvr_factories import qvr_factories

EvaluationAdjoint = Callable[[FunctionRecord], ProgramRecord]
_SQUARE = "qualtran.bloqs.arithmetic.multiplication.Square"
_PHASING = "qualtran.bloqs.rotations.phasing_via_cost_function.PhasingViaCostFunction"


def cost_phasing_factories(
    *,
    exact_adjoint: EvaluationAdjoint | None = None,
    phase_cost_registers: Callable[[FunctionRecord], tuple[str, ...]] | None = None,
) -> dict[str, ComponentFactory]:
    """Supply the complete constructor and its Square/QVR example dependencies.

    Per-compilation recipes retain complete resolved functions, not name guesses.
    Square keeps its original bodyless forward/reverse declarations. An injected
    evaluator owner may supply an executable exact inverse, including helpers and
    measured feedback. Unknown evaluators require that owner explicitly. Custom
    phase owners supply their cost names without guessing from port positions.
    """
    costs: dict[FunctionRecord, tuple[str, ...]] = {}
    squares: dict[FunctionRecord, Mapping[str, ResolvedArgument]] = {}
    square_factory = product_model_factories()[_SQUARE]

    def square(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        result = square_factory(name, registers, arguments)
        assert isinstance(result, FunctionRecord)
        squares[result] = dict(arguments)
        return result

    def track_phase(factory: ComponentFactory) -> ComponentFactory:
        def phase(
            name: str,
            registers: tuple[Register, ...] | None,
            arguments: Mapping[str, ResolvedArgument],
        ) -> FunctionRecord | ProgramRecord:
            result = factory(name, registers, arguments)
            root = result.functions[0] if isinstance(result, ProgramRecord) else result
            cost = arguments["cost_reg"]
            assert isinstance(cost, Register)
            costs[root] = (cost.name,)
            return result

        return phase

    def phasing(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if set(arguments) != {"cost_eval_oracle", "phase_oracle"}:
            raise SerializationError(
                "Cost phasing requires evaluation and phase oracles"
            )
        evaluate, phase = arguments["cost_eval_oracle"], arguments["phase_oracle"]
        if not isinstance(evaluate, FunctionRecord) or not isinstance(
            phase, FunctionRecord
        ):
            raise SerializationError("Cost phasing requires resolved oracle functions")
        if phase in costs:
            cost_names = costs[phase]
        elif phase_cost_registers is not None:
            cost_names = phase_cost_registers(phase)
        else:
            raise SerializationError("Missing cost-register constructor context")
        if exact_adjoint is not None:
            erase = exact_adjoint(evaluate)
        elif evaluate in squares:
            recipe = dict(squares[evaluate])
            recipe["uncompute"] = not bool(recipe.get("uncompute", False))
            reverse = square_factory(evaluate.name + "_adjoint", None, recipe)
            assert isinstance(reverse, FunctionRecord)
            erase = ProgramRecord((reverse,))
        else:
            raise SerializationError("Cost evaluation requires an exact-adjoint owner")
        # Child helper closures already belong to the enclosing library resolver.
        record = cost_function_phase_program(
            ProgramRecord((replace(evaluate, blocks=()),)),
            ProgramRecord((replace(phase, blocks=()),)),
            erase,
            cost_names=cost_names,
        )
        if registers is not None and record.functions[0].registers != registers:
            raise SerializationError("Invalid cost-phasing register signature")
        record = replace(
            record,
            functions=(replace(record.functions[0], name=name), *record.functions[1:]),
        )
        return _namespace_helpers(record, name, external=(evaluate.name, phase.name))

    return {
        **{key: track_phase(factory) for key, factory in qvr_factories().items()},
        _SQUARE: square,
        _PHASING: phasing,
    }
