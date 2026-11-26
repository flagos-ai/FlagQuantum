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
from ..compiler.registers.control.multi_control_component import (
    controlled_word_x_program,
    multi_control_x_program,
)
from ..compiler.registers.layout.auto_partition import AutoPartition
from ..compiler.registers.layout.component_auto_partition import (
    auto_partition_component_program,
    auto_partition_signature,
)
from ..compiler.registers.layout.component_partition import (
    partition_component_program,
    partition_signature,
)
from ..compiler.registers.sorting.in_place_sort_declaration import (
    in_place_sort_signature,
    source_in_place_sort_resource_formula,
)
from ..compiler.registers.sorting.sorted_duplicates_component import (
    sorted_duplicates_program,
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
from ..core.value_parameters import is_symbolic_value
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
            controlled_word_x_program((0,) * self.bitsize),
            {"ctrl": prepared, "q": builder.inputs["flag"]},
        )
        return builder.finalize(
            {
                "q": result["ctrl"],
                "flag": result["q"],
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
        preparation: FunctionRecord | ProgramRecord | None = None,
        sorting: FunctionRecord | ProgramRecord | None = None,
        duplicates: FunctionRecord | ProgramRecord | None = None,
        uncompute: FunctionRecord | ProgramRecord | None = None,
        flag_predicate: FunctionRecord | ProgramRecord | None = None,
        flag_x: FunctionRecord | ProgramRecord | None = None,
    ) -> ProgramRecord:
        """Bind all six original child roles through shared packed-word rebindings.

        Every child must carry its actual source signature, including the
        bounded pi array. Complete Programs retain helper definitions and private
        work. The literal Partition/AutoPartition source recipe also admits
        symbolic word widths with a concrete scope count. Unknown child arrays
        remain declarations behind packed interfaces; no finite body is invented.
        Omitted preparation and sorting bodies remain declarations; concrete
        duplicate detection, uncompute and flag defaults reuse their owners.
        Duplicate detection retains the original inverse aliases. Callers supply
        independent missing bodies for physical execution.
        """
        count, length, width = self.c, self.ell, self.inst.index_bitsize
        scratch_width = self.sort_ancilla_bitsize
        if type(count) is not int:
            raise ValueError("Guiding-state program requires a concrete scope count")
        assert isinstance(count, int)
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
        supplied = (preparation, sorting, duplicates, uncompute, flag_predicate, flag_x)
        names = (
            "scope_guiding_state",
            "sort_in_place",
            "has_duplicates",
            "probabilistic_uncompute",
            "zero_flags",
            "flag_x",
        )
        children = tuple(
            (
                FunctionRecord(name, (), registers=tuple(signature))
                if child is None
                else child
            )
            for name, signature, child in zip(names, signatures, supplied, strict=True)
        )
        if (
            duplicates is None
            and type(length) is int
            and length > 1
            and type(width) is int
            and width > 0
        ):
            children = (
                *children[:2],
                sorted_duplicates_program(length, unsigned_register_type(width)),
                *children[3:],
            )
        if uncompute is None and type(scratch_width) is int and scratch_width > 0:
            children = (
                *children[:3],
                ProbabilisticUncompute(scratch_width).program(),
                *children[4:],
            )
        if flag_predicate is None:
            children = (*children[:4], multi_control_x_program((0, 0)), children[5])
        if flag_x is None:
            children = (
                *children[:5],
                bit_gate_function(
                    "flag_x",
                    tuple(signatures[5]),
                    CircuitIR(1, (Instruction("x", (0,)),)),
                ),
            )
        preparation, sorting, duplicates, uncompute, flag_predicate, flag_x = children
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

        def partition(
            parts: tuple[Register, ...],
            bindings: dict[str, object],
            *,
            reverse: bool = False,
        ) -> dict[str, BindingCollection]:
            if all(type(register.bitsize) is int for register in parts):
                module = partition_component_program(parts, "partition", reverse)
            else:
                module = ProgramRecord(
                    (
                        FunctionRecord(
                            "partition",
                            (),
                            registers=tuple(
                                partition_signature(parts, reverse=reverse)
                            ),
                        ),
                    )
                )
            return builder.call_program(module, bindings, entry="partition")

        def packed(
            child: FunctionRecord | ProgramRecord,
            layout: AutoPartition,
            ports: tuple[Register, ...],
            bindings: dict[str, object],
        ) -> dict[str, BindingCollection]:
            module = (
                child if isinstance(child, ProgramRecord) else ProgramRecord((child,))
            )
            root = module.functions[0]
            assert root.registers is not None
            name = root.name + "_packed"
            while any(function.name == name for function in module.functions):
                name += "_view"
            if any(
                is_symbolic_value(*register.shape_symbolic)
                for register in root.registers
            ):
                # The original root exposes only packed words here. A symbolic
                # child array cannot be materialized as a finite call body.
                outer = auto_partition_signature(
                    Signature(root.registers), layout, packed_ports=ports
                )
                module = ProgramRecord(
                    (
                        FunctionRecord(name, (), registers=tuple(outer)),
                        *module.functions,
                    )
                )
            else:
                module = auto_partition_component_program(
                    module,
                    layout,
                    entry=root.name,
                    name=name,
                    packed_ports=ports,
                )
            return builder.call_program(module, bindings, entry=name)

        ancilla_parts = (
            Register("ancilla", opaque_register_type(scratch_width)),
            Register("flags", quantum_bit_type(), shape=(3,)),
        )
        unpacked = partition(ancilla_parts, {"x": builder.inputs["ancilla"]})
        flags = np.asarray(unpacked["flags"], dtype=object).copy()
        scratch = unpacked["ancilla"]
        scope_parts = (
            Register(
                "S",
                opaque_register_type(self.scope_guiding_state.target_bitsize),
                shape=(count,),
            ),
        )
        scopes = np.asarray(
            partition(scope_parts, {"x": builder.inputs["T"]})["S"], dtype=object
        ).copy()
        for index in range(count):
            scopes[index] = call(preparation, {"S": scopes[index]})["S"]
        target = partition(scope_parts, {"S": scopes}, reverse=True)["x"]
        sorted_outputs = packed(
            sorting,
            AutoPartition(
                (
                    ("T", (("xs", self.target_bitsize),)),
                    ("ancilla", (("pi", scratch_width),)),
                )
            ),
            (
                Register("T", opaque_register_type(self.target_bitsize)),
                Register("ancilla", opaque_register_type(scratch_width)),
            ),
            {"T": target, "ancilla": scratch},
        )
        checked = packed(
            duplicates,
            AutoPartition(
                (("T", (("xs", self.target_bitsize),)), ("flag", (("flag", 1),)))
            ),
            (
                Register("T", opaque_register_type(self.target_bitsize)),
                Register("flag", quantum_bit_type()),
            ),
            {"T": sorted_outputs["T"], "flag": flags[0]},
        )
        flags[0] = checked["flag"]
        erased = call(uncompute, {"q": sorted_outputs["ancilla"], "flag": flags[1]})
        flags[1] = erased["flag"]
        marked = call(flag_predicate, {"controls": flags[:2], "target": flags[2]})
        flags[:2] = marked["controls"]
        flags[2] = call(flag_x, {"q": marked["target"]})["q"]
        ancilla = partition(
            ancilla_parts, {"ancilla": erased["q"], "flags": flags}, reverse=True
        )["x"]
        return builder.finalize({"T": checked["T"], "ancilla": ancilla})


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
