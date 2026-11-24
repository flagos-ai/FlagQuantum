# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Ordered gate selection adapted from Qualtran v0.7.0 cirq_gate_to_bloq.

"""Select existing native gate owners at the external SDK boundary."""

from collections.abc import Hashable, Iterator, Mapping
from typing import Any

from ...compiler.formats.legacy_component_resolution import ResolvedArgument
from ...compiler.formats.legacy_exchange_factories import exchange_factories
from ...compiler.formats.legacy_fixed_gate_factories import fixed_gate_factories
from ...compiler.formats.legacy_rotation_factories import rotation_factories
from ...compiler.formats.legacy_scalar_factories import scalar_phase_factories
from ...core.ir.program import FunctionRecord, ProgramRecord
from ._named_gate_controls import named_positive_control, named_predicate_control
from ._named_gate_inverse import named_inverse_program
from ._register_definition import _native_definition_module
from .conversion import _cirq, _cirq_parameter_value, _global_phase_angle


class _NamedGateDefinitions(Mapping[Hashable, FunctionRecord]):
    """Resolve encountered named gates and retain only their native helper closure.

    Mapping identity remains local to one import. SDK gates are keys only here;
    all values are owned Program functions. Unknown gates decline selection so
    the existing SDK decomposition and explicit tensor fallback remain available.
    """

    def __init__(self) -> None:
        self._definitions: dict[Hashable, FunctionRecord] = {}
        self._unknown: set[Hashable] = set()
        self._recipes: dict[FunctionRecord, Any] = {}
        self._factories = (
            fixed_gate_factories()
            | exchange_factories()
            | rotation_factories()
            | scalar_phase_factories()
        )

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._definitions)

    def __len__(self) -> int:
        return len(self._definitions)

    def __getitem__(self, gate: Hashable) -> FunctionRecord:
        if gate in self._definitions:
            return self._definitions[gate]
        if gate in self._unknown:
            raise KeyError(gate)
        cirq = _cirq()
        if isinstance(gate, cirq.ops.raw_types._InverseCompositeGate):
            inverse = named_inverse_program(
                gate._original,
                self,
                self._exact_pairs,
                lambda: f"named_gate_{len(self._definitions)}",
            )
            if inverse is not None:
                return self._remember(gate, inverse)
        selected = self._select(gate)
        if selected is None and isinstance(gate, cirq.ControlledGate):
            if len(tuple(gate.control_values.expand())) != 1:
                raise ValueError(
                    "CtrlSpec currently only supports converting SumOfProduct representation with a single AND clause."
                )
            child = self.get(gate.sub_gate)
            positive_gate = cirq.ControlledGate(gate.sub_gate)
            if child is not None and self._select(positive_gate) is not None:
                positive_root = self[positive_gate]
                positive_module = ProgramRecord(
                    tuple(_native_definition_module((positive_root,), self).values())
                )
                resolved = named_predicate_control(
                    f"named_gate_{len(self._definitions)}",
                    child,
                    gate.control_values,
                    lambda current: named_positive_control(current, positive_module),
                )
                return self._remember(gate, resolved)
        if selected is None:
            self._unknown.add(gate)
            raise KeyError(gate)
        source, arguments = selected
        import numpy as np
        import sympy

        normalized: dict[str, ResolvedArgument] = {}
        for key, value in arguments.items():
            if isinstance(value, np.float64):
                value = float(value)
            if isinstance(value, sympy.Expr) and not value.free_symbols:
                value = float(value)
            normalized[key] = value
        arguments = normalized
        name = f"named_gate_{len(self._definitions)}"
        resolved = self._factories["qualtran.bloqs.basic_gates." + source](
            name, None, arguments
        )
        return self._remember(gate, resolved)

    def _remember(
        self, gate: Hashable, resolved: FunctionRecord | ProgramRecord
    ) -> FunctionRecord:
        functions = (
            resolved.functions if isinstance(resolved, ProgramRecord) else (resolved,)
        )
        self._definitions[gate] = functions[0]
        self._recipes[functions[0]] = gate
        for function in functions[1:]:
            previous = self._definitions.setdefault(function.name, function)
            if previous != function:
                raise ValueError(f"Conflicting named helper {function.name!r}")
        return functions[0]

    def _module(self, root: FunctionRecord) -> ProgramRecord:
        return ProgramRecord(tuple(_native_definition_module((root,), self).values()))

    def _exact_pairs(self) -> tuple[tuple[ProgramRecord, ProgramRecord], ...]:
        cirq = _cirq()
        result = []
        for root, gate in tuple(self._recipes.items()):
            inverse = self.get(cirq.inverse(gate))
            if inverse is not None:
                result.append((self._module(root), self._module(inverse)))
        return tuple(result)

    def _select(self, gate: Any) -> tuple[str, dict[str, ResolvedArgument]] | None:
        cirq = _cirq()
        constants: dict[Any, tuple[str, dict[str, ResolvedArgument]]] = {
            cirq.T: ("t_gate.TGate", {}),
            cirq.T**-1: ("t_gate.TGate", {"is_adjoint": True}),
            cirq.S: ("s_gate.SGate", {}),
            cirq.S**-1: ("s_gate.SGate", {"is_adjoint": True}),
            cirq.H: ("hadamard.Hadamard", {}),
            cirq.ControlledGate(cirq.H): ("hadamard.CHadamard", {}),
            cirq.CNOT: ("cnot.CNOT", {}),
            cirq.TOFFOLI: ("toffoli.Toffoli", {}),
            cirq.X: ("x_basis.XGate", {}),
            cirq.Y: ("y_gate.YGate", {}),
            cirq.ControlledGate(cirq.Y): ("y_gate.CYGate", {}),
            cirq.Z: ("z_basis.ZGate", {}),
            cirq.CZ: ("z_basis.CZ", {}),
            cirq.SWAP: ("swap.TwoBitSwap", {}),
            cirq.CSWAP: ("swap.CSwap", {"bitsize": 1}),
            cirq.I: ("identity.Identity", {}),
        }
        if gate in constants:
            return constants[gate]
        if isinstance(gate, cirq.ControlledGate):
            child = self._select(gate.sub_gate)
            if (
                gate.num_controls() == 1
                and tuple(gate.control_values.expand()) == ((1,),)
                and child is not None
            ):
                source, arguments = child
                if source == "rotation.ZPowGate":
                    return "rotation.CZPowGate", arguments
                if source == "rotation.Rz":
                    return "rotation.CRz", arguments
                if source == "global_phase.GlobalPhase":
                    return "rotation.ZPowGate", arguments
            # General conjunctions use the predicate owner in __getitem__;
            # only directly specialized controls are selected here.
            return None
        if isinstance(gate, (cirq.Rx, cirq.Ry, cirq.Rz)):
            return "rotation." + type(gate).__name__, {"angle": gate._rads}
        if (
            isinstance(
                gate, (cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.CZPowGate)
            )
            and gate.global_shift == 0
        ):
            return "rotation." + type(gate).__name__, {"exponent": gate.exponent}
        if isinstance(gate, cirq.GlobalPhaseGate):
            import sympy

            angle = _global_phase_angle(cirq, gate.coefficient)
            if not cirq.is_parameterized(gate):
                from math import pi

                return "global_phase.GlobalPhase", {"exponent": angle / pi}
            symbolic = _cirq_parameter_value(angle, cache={})
            return "global_phase.GlobalPhase", {"exponent": symbolic / sympy.pi}
        return None
