# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 optimization/k_xor_sat/kikuchi_guiding_state.py
# at bcb17bb25360ed68ce3bc084645db3e929799389.

"""Kikuchi guiding-state declarations and coherent probabilistic uncomputation."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np

from ..compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ..compiler._component_trace.qec_gate_cost import QECGateCost
from ..compiler._component_trace.resource_model import estimate_component_resources
from ..compiler.registers.bits.on_each import on_each_word_program
from ..compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from ..compiler.registers.component_gate_function import bit_gate_function
from ..compiler.registers.control.multi_control import (
    multi_control_x_ir,
    source_multi_control_x_resource_formula,
)
from ..compiler.registers.control.multi_control_component import multi_control_x_program
from ..compiler.registers.sorting.in_place_sort_declaration import (
    in_place_sort_signature,
    source_in_place_sort_resource_formula,
)
from ..compiler.registers.sorting.sorted_duplicates_resource_model import (
    source_sorted_duplicates_gate_cost,
)
from ..core.ir import CircuitIR, Instruction
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_bindings import RegisterBinding
from ..core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt, SymbolicReal, ceil, log2, pi
from .kxor_instance import Constraint, KXorInstance
from .primitives._block_binding import _BoundBlockOperation, _validate_block_work
from .primitives.prepare_oracle import PrepareOracle
from .primitives.rotation_resource_model import sparse_rotation_preparation_cost
from .primitives.sparse_rotation_preparation import SparseRotationPreparation
from .primitives.symbolic_rotation_resource_model import (
    source_sparse_rotation_preparation_resource_formula,
)

_Node = TypeVar("_Node", bound=Hashable)


@dataclass(frozen=True)
class ScopeGuidingState(PrepareOracle):
    """Declare the weighted scope state for the ell=k source construction."""

    inst: KXorInstance
    eps: SymbolicReal = field(default=1e-6, kw_only=True)

    @property
    def target_bitsize(self) -> SymbolicInt:
        return self.inst.k * self.inst.index_bitsize

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (Register("S", opaque_register_type(self.target_bitsize)),)

    @property
    def phasegrad_bitsize(self) -> SymbolicInt:
        return ceil(log2(2 * pi(self.eps) / self.eps))

    @property
    def state_preparation(self) -> SparseRotationPreparation:
        """Bind the concrete child, whose phase gradient is caller-owned.

        The frozen source root has no decomposition and omits that gradient
        from its signature. This property exposes the existing dependency only.
        """
        width, phase = self.target_bitsize, self.phasegrad_bitsize
        scopes = self.inst.batched_scopes
        if (
            type(width) is not int
            or type(phase) is not int
            or not isinstance(scopes, tuple)
        ):
            raise ValueError(
                "Physical sparse preparation requires concrete scopes and widths"
            )
        return SparseRotationPreparation.from_coefficient_map(
            2**width,
            {self.inst.scope_as_int(scope): complex(value) for scope, value in scopes},
            phase,
        )

    def resource_gates(self) -> QECGateCost:
        if self.inst.is_symbolic():
            model = source_sparse_rotation_preparation_resource_formula(
                2**self.target_bitsize,
                self.inst.num_unique_constraints,
                self.phasegrad_bitsize,
                target_width=self.target_bitsize,
            )
        else:
            width = self.target_bitsize
            if type(width) is not int:
                raise ValueError("Concrete instance requires a concrete target width")
            concrete_model = estimate_component_resources(
                sparse_rotation_preparation_cost(self.state_preparation, width)
            ).gates
            return QECGateCost().plus(concrete_model, 1)
        return QECGateCost().plus(model, 1)

    def source_graph(
        self, *, root: _Node, preparation: _Node
    ) -> SymbolicCallGraph[_Node]:
        return symbolic_call_graph(root, {root: {preparation: 1}})


@dataclass(frozen=True)
class ProbabilisticUncompute:
    """Apply H on a packed word and XOR its zero predicate into a flag.

    The word and flag are logical through outputs. Only the supplied predicate
    workspace is cleaned. No measurement or reset of the logical word occurs.
    """

    bitsize: SymbolicInt

    @property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            q=opaque_register_type(self.bitsize), flag=quantum_bit_type()
        )

    def resource_gates(self) -> QECGateCost:
        pattern = (0,) * self.bitsize if type(self.bitsize) is int else self.bitsize
        return QECGateCost(clifford=self.bitsize).plus(
            source_multi_control_x_resource_formula(pattern), 1
        )

    def program(self) -> ProgramRecord:
        """Compose owned Hadamards and measured zero marking on logical THRU ports."""
        if type(self.bitsize) is not int or self.bitsize < 1:
            raise ValueError("Uncompute requires a positive concrete word width")
        hadamard = bit_gate_function(
            "hadamard",
            tuple(Signature.build(q=1)),
            CircuitIR(1, (Instruction("h", (0,)),)),
        )
        builder = ComponentProgramBuilder(self.signature)
        prepared = builder.call_program(
            on_each_word_program(self.bitsize, hadamard),
            {"q": builder.inputs["q"]},
        )["q"]
        assert isinstance(prepared, RegisterBinding)
        result = builder.call_program(
            multi_control_x_program((0,) * self.bitsize),
            {
                "controls": np.asarray(builder.split(prepared), dtype=object),
                "target": builder.inputs["flag"],
            },
        )
        return builder.finalize(
            {
                "q": builder.join(np.asarray(result["controls"]).flat),
                "flag": result["target"],
            }
        )

    def bind(
        self,
        q: tuple[int, ...],
        flag: int,
        workspace: tuple[int, ...] = (),
        classical_bits: tuple[int, ...] = (),
    ) -> _BoundBlockOperation:
        """Bind big-endian q and clean predicate workspace for circuit.append(op).

        Examples:
            op = ProbabilisticUncompute(1).bind((0,), 1)
            circuit.append(op)
        """
        if type(self.bitsize) is not int or self.bitsize < 1 or len(q) != self.bitsize:
            raise ValueError(
                "Uncompute requires a positive concrete word of the declared width"
            )
        _validate_block_work((*q, flag), workspace, classical_bits)
        predicate = multi_control_x_ir(
            q, (0,) * len(q), flag, workspace, classical_bits
        )
        instructions = tuple(Instruction("h", (wire,)) for wire in q) + predicate
        return _BoundBlockOperation(instructions, max((*q, flag, *workspace)) + 1)


@dataclass(frozen=True, init=False)
class GuidingState(PrepareOracle):
    """Retain the sorted tensor-product guiding-state source declaration."""

    inst: KXorInstance
    ell: SymbolicInt
    amplitude_good_part: SymbolicReal
    eps: SymbolicReal

    def __init__(
        self,
        inst: KXorInstance,
        ell: SymbolicInt,
        *,
        amplitude_good_part: SymbolicReal | None = None,
        eps: SymbolicReal = 1e-6,
    ) -> None:
        object.__setattr__(self, "inst", inst)
        object.__setattr__(self, "ell", ell)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(
            self,
            "amplitude_good_part",
            self.coeff_good if amplitude_good_part is None else amplitude_good_part,
        )

    @property
    def target_bitsize(self) -> SymbolicInt:
        return self.inst.index_bitsize * self.ell

    @property
    def sort_ancilla_bitsize(self) -> SymbolicInt:
        return self.ell * ceil(log2(self.ell))

    @property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.sort_ancilla_bitsize + 3

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (Register("T", opaque_register_type(self.target_bitsize)),)

    @property
    def junk_registers(self) -> tuple[Register, ...]:
        return (Register("ancilla", opaque_register_type(self.ancilla_bitsize)),)

    @property
    def c(self) -> SymbolicInt:
        count = self.ell // self.inst.k
        try:
            return int(count)
        except TypeError:
            return count

    @property
    def scope_guiding_state(self) -> ScopeGuidingState:
        return ScopeGuidingState(self.inst, eps=self.eps / self.c)

    @property
    def coeff_good(self) -> SymbolicReal:
        return 0.99 / 2 ** (self.sort_ancilla_bitsize / 2)

    def source_graph(
        self,
        *,
        root: _Node,
        preparation: _Node,
        sorting: _Node,
        duplicates: _Node,
        uncompute: _Node,
        flag_predicate: _Node,
        flag_x: _Node,
    ) -> SymbolicCallGraph[_Node]:
        return symbolic_call_graph(
            root,
            {
                root: {
                    preparation: self.c,
                    sorting: 1,
                    duplicates: 1,
                    uncompute: 1,
                    flag_predicate: 1,
                    flag_x: 1,
                }
            },
        )

    def resource_gates(self) -> QECGateCost:
        return (
            QECGateCost(clifford=1)
            .plus(self.scope_guiding_state.resource_gates(), self.c)
            .plus(
                source_in_place_sort_resource_formula(
                    self.ell, self.inst.index_bitsize
                ),
                1,
            )
            .plus(
                source_sorted_duplicates_gate_cost(self.ell, self.inst.index_bitsize), 1
            )
            .plus(ProbabilisticUncompute(self.sort_ancilla_bitsize).resource_gates(), 1)
            .plus(source_multi_control_x_resource_formula((0, 0)), 1)
        )

    def program(
        self,
        preparation: FunctionRecord | ProgramRecord,
        sorting: FunctionRecord | ProgramRecord,
        duplicates: FunctionRecord | ProgramRecord,
        uncompute: FunctionRecord | ProgramRecord,
        flag_predicate: FunctionRecord | ProgramRecord,
        flag_x: FunctionRecord | ProgramRecord,
    ) -> ProgramRecord:
        """Bind all six original child roles through shared packed-word rebindings.

        Every child must carry its actual source signature, including the
        bounded pi array. Complete Programs retain their helper definitions and
        private work ownership. FunctionRecord callers preserve their existing
        construction unchanged. Callers supply child bodies and their independent
        physical evidence for source-declaration-only sorting and preparation.
        """
        count, length, width = self.c, self.ell, self.inst.index_bitsize
        scratch_width = self.sort_ancilla_bitsize
        if not all(
            type(value) is int for value in (count, length, width, scratch_width)
        ):
            raise ValueError("Guiding-state program requires concrete partition sizes")
        assert isinstance(count, int) and isinstance(length, int)
        assert isinstance(width, int) and isinstance(scratch_width, int)
        signatures = (
            self.scope_guiding_state.signature,
            in_place_sort_signature(length, unsigned_register_type(width)),
            Signature(
                (
                    Register("xs", unsigned_register_type(width), shape=(length,)),
                    Register("flag", quantum_bit_type()),
                )
            ),
            ProbabilisticUncompute(scratch_width).signature,
            Signature(
                (
                    Register("controls", quantum_bit_type(), shape=(2,)),
                    Register("target", quantum_bit_type()),
                )
            ),
            Signature.build(q=1),
        )
        children = (
            preparation,
            sorting,
            duplicates,
            uncompute,
            flag_predicate,
            flag_x,
        )
        if any(
            isinstance(child, ProgramRecord) and not child.functions
            for child in children
        ):
            raise ValueError("Guiding-state child Programs require a typed entry")
        functions = tuple(
            child.functions[0] if isinstance(child, ProgramRecord) else child
            for child in children
        )
        for function, expected in zip(functions, signatures, strict=True):
            if function.registers != tuple(expected):
                raise ValueError(
                    f"Guiding-state child {function.name!r} has a different source signature"
                )
        builder = ComponentProgramBuilder(
            self.signature,
            (child for child in children if isinstance(child, FunctionRecord)),
        )

        def call(
            child: FunctionRecord | ProgramRecord, bindings: dict[str, object]
        ) -> dict[str, BindingCollection]:
            if isinstance(child, ProgramRecord):
                return builder.call_program(
                    child, bindings, entry=child.functions[0].name
                )
            return builder.call_dict(child.name, bindings)

        target, ancilla = builder.inputs["T"], builder.inputs["ancilla"]
        assert isinstance(target, RegisterBinding) and isinstance(
            ancilla, RegisterBinding
        )
        ancilla_bits = builder.split(ancilla)
        flags = list(ancilla_bits[scratch_width:])
        scratch = builder.join(ancilla_bits[:scratch_width])
        scope_width = self.scope_guiding_state.target_bitsize
        assert isinstance(scope_width, int)
        target_bits = builder.split(target)
        scope_bits: list[RegisterBinding] = []
        for index in range(count):
            scope = builder.join(
                target_bits[index * scope_width : (index + 1) * scope_width]
            )
            result = call(preparation, {"S": scope})["S"]
            assert isinstance(result, RegisterBinding)
            scope_bits.extend(builder.split(result))
        target_bits = np.asarray(scope_bits, dtype=np.object_)
        xs = np.asarray(
            [
                builder.join(
                    target_bits[index * width : (index + 1) * width],
                    unsigned_register_type(width),
                )
                for index in range(length)
            ],
            dtype=object,
        )
        scratch_bits = builder.split(scratch)
        rank_type = signatures[1].get_left("pi").dtype
        rank_width = rank_type.num_qubits
        assert isinstance(rank_width, int)
        pi_words = np.asarray(
            [
                builder.join(
                    scratch_bits[index * rank_width : (index + 1) * rank_width],
                    rank_type,
                )
                for index in range(length)
            ],
            dtype=object,
        )
        sorted_outputs = call(sorting, {"xs": xs, "pi": pi_words})
        checked = call(duplicates, {"xs": sorted_outputs["xs"], "flag": flags[0]})
        flags[0] = checked["flag"]
        scratch = builder.join(
            [
                bit
                for word in np.asarray(sorted_outputs["pi"]).flat
                for bit in builder.split(word)
            ]
        )
        erased = call(uncompute, {"q": scratch, "flag": flags[1]})
        flags[1] = erased["flag"]
        marked = call(
            flag_predicate,
            {"controls": np.asarray(flags[:2], dtype=object), "target": flags[2]},
        )
        flags[:2] = list(np.asarray(marked["controls"]).flat)
        flags[2] = call(flag_x, {"q": marked["target"]})["q"]
        target = builder.join(
            [
                bit
                for word in np.asarray(checked["xs"]).flat
                for bit in builder.split(word)
            ]
        )
        erased_scratch = erased["q"]
        assert isinstance(erased_scratch, RegisterBinding)
        ancilla = builder.join([*builder.split(erased_scratch), *flags])
        return builder.finalize({"T": target, "ancilla": ancilla})


def example_scope_guiding_state() -> ScopeGuidingState:
    instance = KXorInstance(
        n=4,
        k=2,
        constraints=(
            Constraint((0, 1), 1),
            Constraint((2, 3), -1),
            Constraint((1, 2), 1),
        ),
    )
    return ScopeGuidingState(instance)


def symbolic_scope_guiding_state() -> ScopeGuidingState:
    from sympy import symbols

    n, m, k = symbols("n m k", positive=True, integer=True)
    return ScopeGuidingState(KXorInstance.symbolic(n=n, m=m, k=k))


def example_guiding_state() -> GuidingState:
    return GuidingState(example_scope_guiding_state().inst, 4)


def symbolic_guiding_state() -> GuidingState:
    instance = symbolic_scope_guiding_state().inst
    return GuidingState(instance, 2 * instance.k)


def symbolic_count_guiding_state() -> GuidingState:
    from sympy import symbols

    n, m, c = symbols("n m c", positive=True, integer=True)
    k = symbols("k", positive=True, integer=True, even=True)
    return GuidingState(KXorInstance.symbolic(n=n, m=m, k=k), c * k)
