# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ClassicalSimState and PhasedClassicalSimState.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/classical_sim.py

"""Step selected typed SSA calls using explicit numerical value and phase actions."""

from collections.abc import Callable, Iterable, Mapping

import numpy as np

from ..core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from ..core.ir.program.registers import Register, RegisterType
from ..core.ir.program.signatures import Signature
from .binary_field_encoding import resolve_binary_field_encoding
from .classical_action import evaluate_classical_action
from .classical_distributions import ClassicalDistribution, draw_classical_value
from .classical_measurement import MeasurementOutcomePhase, assert_unit_classical_phase
from .classical_typed_values import (
    bind_typed_register_values,
    collect_typed_register_value,
)
from .classical_updates import commit_classical_outputs, commit_phased_classical_outputs
from .finite_field_encoding import FiniteFieldEncoding


class ComponentClassicalState:
    """Numerically evaluate one verified unconditional typed-call entry.

    Compiler supplies operation positions in dependency order. Optional occurrence
    IDs align with block positions and identify random outcomes independently of
    visitation order. Value and phase callbacks run in that order; decomposition
    is an explicit callback, not Python method discovery or graph transformation.

    Integer/bit and resolved binary-field ports use existing native
    validators and storage codecs; opaque ports retain unrestricted values.
    Other encodings, physical graphs, control flow
    and primitive entry operations are refused before actions. Caller values are not
    rewritten. Assignments and last_position remain inspectable after failures;
    outputs are committed before phase rejection, as in the pinned simulator.
    Recursive NotImplementedError failures retain their cause and acquire the
    current function name. Direct action and output-commit errors propagate.
    """

    def __init__(
        self,
        record: ProgramRecord,
        order: tuple[int, ...],
        values: Mapping[str, object],
        value_action: Callable[[FunctionRecord, dict[str, object]], object],
        phase_action: (
            Callable[
                [FunctionRecord, dict[str, object]],
                complex | MeasurementOutcomePhase | None,
            ]
            | None
        ) = None,
        *,
        entry: str = "main",
        field_encodings: Mapping[RegisterType, FiniteFieldEncoding] | None = None,
        phased: bool = False,
        phase: complex = 1.0,
        instance_ids: tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
        fixed_values: Mapping[int, object] | None = None,
        recurse: (
            Callable[
                [FunctionRecord, dict[str, object]],
                tuple[dict[str, object], complex | MeasurementOutcomePhase | None],
            ]
            | None
        ) = None,
    ) -> None:
        diagnostics = verify_program_record(record)
        if diagnostics:
            raise ValueError(f"invalid classical component graph: {diagnostics}")
        functions = {function.name: function for function in record.functions}
        if entry not in functions:
            raise ValueError(f"unknown classical component entry {entry!r}")
        function = functions[entry]
        if (
            record.physical_context is not None
            or function.registers is None
            or len(function.blocks) != 1
        ):
            raise ValueError("classical components require one typed logical block")
        block = function.blocks[0]
        if block.terminator != "return" or block.edges or block.return_values is None:
            raise ValueError("classical components require explicit returned values")
        if (
            len(order) != len(block.operations)
            or any(type(i) is not int for i in order)
            or set(order) != set(range(len(block.operations)))
        ):
            raise ValueError(
                "classical call order must visit every operation exactly once"
            )
        known = set(block.arguments)
        for position in order:
            operation = block.operations[position]
            child = functions.get(operation.callee or "")
            if (
                operation.opcode != "call"
                or child is None
                or child.registers is None
                or operation.parameters
                or operation.condition_clauses
            ):
                raise ValueError(
                    "classical components require unconditional typed calls"
                )
            if not set(operation.operands) <= known:
                raise ValueError("classical call order must respect SSA dependencies")
            known.update(operation.results)
        self._fields = dict(field_encodings or {})
        for registers in (
            function.registers,
            *(functions[op.callee or ""].registers for op in block.operations),
        ):
            assert registers is not None
            for register in registers:
                dtype = register.dtype
                if dtype.encoding == "finite_field":
                    if dtype not in self._fields:
                        self._fields[dtype] = resolve_binary_field_encoding(dtype)
                    if self._fields[dtype].register_type() != dtype:
                        raise ValueError(
                            "finite-field encoding differs from declared metadata"
                        )
                elif dtype.encoding not in {"bit", "opaque", "uint", "signed"}:
                    raise ValueError(
                        "classical component ports require integer or bit encodings "
                        "or a resolved binary-field encoding"
                    )
        ids = (
            tuple(range(len(block.operations)))
            if instance_ids is None
            else instance_ids
        )
        if (
            len(ids) != len(block.operations)
            or any(type(i) is not int or i < 0 for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(
                "instance IDs must be distinct nonnegative integers per operation"
            )
        if type(phased) is not bool:
            raise ValueError("phased evaluation requires a built-in boolean")
        self.assignments: dict[ValueRef, object] = {}
        self.last_position: int | None = None
        self.phase = phase
        self._phased = phased
        self._functions, self._block = functions, block
        self._signature = Signature(function.registers)
        self._positions = iter(order)
        self._ids, self._rng, self._fixed = ids, rng, fixed_values
        self._values, self._phase_action, self._recurse = (
            value_action,
            phase_action,
            recurse,
        )
        self._outcomes: dict[tuple[str, tuple[int, ...]], object] = {}
        self._assign(self._signature.lefts(), block.arguments, values)
        if phased:
            assert_unit_classical_phase(phase)
        elif phase != 1:
            raise ValueError("an initial phase requires phased evaluation")

    def _assign(
        self,
        registers: Iterable[Register],
        identities: tuple[ValueRef, ...],
        values: Mapping[str, object],
    ) -> None:
        refs = iter(identities)
        instance = -1 if self.last_position is None else self._ids[self.last_position]

        def resolve(distribution: ClassicalDistribution) -> object:
            return draw_classical_value(
                distribution,
                instance_id=instance,
                rng=self._rng,
                fixed_values=self._fixed,
            )

        for register in registers:
            bound = bind_typed_register_values(
                register,
                values,
                field_encoding=self._fields.get(register.dtype),
                resolve_distribution=resolve,
            )
            for index, value in bound.items():
                self.assignments[next(refs)] = value
                self._outcomes[register.name, index] = value

    def _collect(
        self, registers: Iterable[Register], identities: tuple[ValueRef, ...]
    ) -> dict[str, object]:
        refs = iter(identities)
        outputs: dict[str, object] = {}
        for register in registers:
            assignments = {
                index: self.assignments[next(refs)] for index in register.all_idxs()
            }
            value = collect_typed_register_value(
                register, assignments, field_encoding=self._fields.get(register.dtype)
            )
            outputs[register.name] = value
        return outputs

    def step(self) -> "ComponentClassicalState":
        """Advance one call; propagate errors with its visible assignment prefix."""
        position = next(self._positions)
        self.last_position = position
        operation = self._block.operations[position]
        function = self._functions[operation.callee or ""]
        assert function.registers is not None
        signature = Signature(function.registers)
        inputs = self._collect(signature.lefts(), operation.operands)

        def decompose(
            values: dict[str, object],
        ) -> tuple[dict[str, object], complex | MeasurementOutcomePhase | None]:
            if self._recurse is None:
                raise NotImplementedError(
                    f"no classical decomposition supplied for {function.name}"
                )
            try:
                return self._recurse(function, values)
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"{function.name} does not support classical simulation: {exc}"
                ) from exc

        outputs, contribution = evaluate_classical_action(
            inputs,
            lambda **values: self._values(function, values),
            lambda **values: (
                self._phase_action(function, values) if self._phase_action else None
            ),
            decompose,
            operation_name=function.name,
        )
        self._outcomes.clear()

        def assign(values: Mapping[str, object]) -> None:
            self._assign(signature.rights(), operation.results, values)
            if (
                self._phased
                and isinstance(contribution, MeasurementOutcomePhase)
                and (contribution.register, contribution.index) not in self._outcomes
            ):
                raise ValueError(
                    "measurement phase requires a current output: "
                    f"{contribution.register}{contribution.index}"
                )

        if self._phased:
            self.phase = commit_phased_classical_outputs(
                outputs, contribution, assign, phase=self.phase, outcomes=self._outcomes
            )
        else:
            commit_classical_outputs(
                outputs, contribution, assign, operation_name=function.name
            )
        return self

    def finalize(self) -> dict[str, object]:
        """Collect returned SSA values; unfinished live outputs raise KeyError."""
        assert self._block.return_values is not None
        return self._collect(self._signature.rights(), self._block.return_values)

    def simulate(self) -> dict[str, object]:
        """Step to exhaustion and collect final values without normalizing phase."""
        try:
            while True:
                self.step()
        except StopIteration:
            return self.finalize()


def simulate_phased_component_calls(
    record: ProgramRecord,
    call_orders: Mapping[str, tuple[int, ...]],
    values: Mapping[str, object],
    value_action: Callable[[FunctionRecord, dict[str, object]], object],
    phase_action: (
        Callable[
            [FunctionRecord, dict[str, object]],
            complex | MeasurementOutcomePhase | None,
        ]
        | None
    ) = None,
    *,
    entry: str = "main",
    instance_ids: Mapping[str, tuple[int, ...]] | None = None,
    field_encodings: Mapping[RegisterType, FiniteFieldEncoding] | None = None,
    rng: np.random.Generator | None = None,
    fixed_values: Mapping[int, object] | None = None,
) -> tuple[dict[str, object], complex]:
    """Evaluate stored typed-call bodies recursively with source phased policy.

    Compiler supplies a call order for each body reached after a numerical
    action returns NotImplemented with no phase. Direct numerical leaves need
    no order. Each nested state starts at phase one, inherits the same RNG or
    fixed-value mapping and returns its accumulated phase to the parent.
    Optional instance IDs align with each function's stored operation positions;
    Compiler's persistent expansion labels may differ from visitation positions.
    Omitted functions use their local positions. IDs are not global call paths.

    This composes the existing state, binder and phase owners; it performs no
    decomposition discovery or graph transformation. Missing body schedules
    raise NotImplementedError with the existing recursive cause context. Other
    admission, action and output failures retain ComponentClassicalState rules.
    Pure source recursion's separate default RNG is outside this phased entry.
    """
    orders = dict(call_orders)
    ids = dict(instance_ids or {})

    def recurse(
        function: FunctionRecord, inputs: dict[str, object]
    ) -> tuple[dict[str, object], complex]:
        return evaluate(function.name, inputs)

    def evaluate(
        name: str, inputs: Mapping[str, object]
    ) -> tuple[dict[str, object], complex]:
        try:
            order = orders[name]
        except KeyError as exc:
            raise NotImplementedError(
                f"no classical call order supplied for {name}"
            ) from exc
        state = ComponentClassicalState(
            record,
            order,
            inputs,
            value_action,
            phase_action,
            entry=name,
            field_encodings=field_encodings,
            phased=True,
            instance_ids=ids.get(name),
            rng=rng,
            fixed_values=fixed_values,
            recurse=recurse,
        )
        return state.simulate(), state.phase

    return evaluate(entry, values)
