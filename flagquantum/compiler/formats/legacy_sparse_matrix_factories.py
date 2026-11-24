# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 block_encoding/sparse_matrix.py.

"""Resolve sparse matrix constructors with native index and entry contexts."""

from collections.abc import Callable, Mapping

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ..registers.arithmetic.sparse_index_component import (
    banded_index_program,
    top_left_index_program,
)
from ..registers.component_control import ControlAdder, _stored_control_adder
from ..registers.lookup.matrix_entry_component import (
    explicit_entry_program,
    uniform_entry_program,
)
from ..registers.lookup.qrom_table import QromTable
from ..registers.preparation.hermitian_sparse_matrix_component import (
    hermitian_sparse_matrix_component_program,
)
from ..registers.preparation.sparse_matrix_component import (
    sparse_matrix_component_program,
)
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_controlled_factories import controlled_via_and_factories
from .legacy_modular_addition_factories import _integer
from .legacy_rotation_factories import _real_argument
from .legacy_word_factories import _rename_program

ExactPair = tuple[ProgramRecord, ProgramRecord]
IndexContext = Callable[[FunctionRecord], tuple[SymbolicInt, SymbolicInt, ExactPair]]
EntryContext = Callable[[FunctionRecord], tuple[SymbolicInt, ExactPair]]
EntryAngle = Callable[[ResolvedArgument], float]
EntryTable = Callable[[SymbolicInt, ResolvedArgument, SymbolicInt], QromTable]
_Directional = Callable[[bool], ProgramRecord]
_PREFIX = "qualtran.bloqs.block_encoding.sparse_matrix."


def sparse_matrix_factories(
    *,
    entry_angle: EntryAngle,
    entry_table: EntryTable,
    sqrt_entry_angle: EntryAngle | None = None,
    index_context: IndexContext | None = None,
    entry_context: EntryContext | None = None,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Retain native roles and exact directions without importing source objects.

    External numerical payload conversion belongs to the supplied boundary.
    Other oracle types require explicit native width/sparsity/inverse owners.
    """
    recipes: dict[FunctionRecord, tuple[_Directional, bool]] = {}
    indices: dict[FunctionRecord, tuple[SymbolicInt, SymbolicInt, ExactPair]] = {}
    entries: dict[FunctionRecord, tuple[SymbolicInt, ExactPair]] = {}
    positive_builders: dict[FunctionRecord, _Directional] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        build: _Directional,
        direction: bool = False,
    ) -> ProgramRecord:
        record = build(direction)
        if registers is not None and record.functions[0].registers != registers:
            raise SerializationError("Invalid sparse component register signature")
        record = _rename_program(record, name)
        recipes[record.functions[0]] = build, direction
        return record

    def oracle(
        kind: str,
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        fields = {
            "TopLeftRowColumnOracle": {"_num_nonzero"},
            "SymmetricBandedRowColumnOracle": {"bandsize"},
            "UniformEntryOracle": {"entry"},
            "ExplicitEntryOracle": {"data", "entry_bitsize"},
            "UniformSqrtEntryOracle": {"entry", "eps"},
        }
        allowed = {"system_bitsize", *fields[kind]}
        required = allowed - {"_num_nonzero", "eps"}
        if not required <= arguments.keys() or arguments.keys() - allowed:
            raise SerializationError("Invalid sparse oracle constructor arguments")
        width = _integer(arguments, "system_bitsize")
        if kind == "TopLeftRowColumnOracle":
            if type(width) is int and width <= 0:
                raise SerializationError("Sparse index width must be positive")
            if "_num_nonzero" in arguments:
                sparsity = _integer(arguments, "_num_nonzero")
            elif registers is not None:
                # The original producer omits the private constructor field,
                # but the bounded ordinal explicitly retains num_nonzero.
                ordinal = Signature(registers).get_left("l")
                raw = dict(ordinal.dtype.parameters).get("bound", 2**width)
                sparsity = _integer({"_num_nonzero": raw}, "_num_nonzero")
            else:
                sparsity = 2**width

            def index_build(direction: bool) -> ProgramRecord:
                return top_left_index_program(width, sparsity, adjoint=direction)

            record = make(name, registers, index_build)
            indices[record.functions[0]] = width, sparsity, (record, index_build(True))
            return record
        if kind == "SymmetricBandedRowColumnOracle":
            bandsize = _integer(arguments, "bandsize")
            if type(width) is int and type(bandsize) is int and 2**width < 2 * bandsize:
                raise SerializationError(
                    "Sparse band size is too large for its index width"
                )

            def band_build(direction: bool) -> ProgramRecord:
                return banded_index_program(width, bandsize, adjoint=direction)

            record = make(name, registers, band_build)
            indices[record.functions[0]] = (
                width,
                2 * bandsize + 1,
                (record, band_build(True)),
            )
            return record
        if kind in ("UniformEntryOracle", "UniformSqrtEntryOracle"):
            convert = entry_angle
            if kind == "UniformSqrtEntryOracle":
                if sqrt_entry_angle is None:
                    raise SerializationError(
                        "Missing square-root entry payload converter"
                    )
                convert = sqrt_entry_angle
                _real_argument({"eps": arguments.get("eps", 1e-11)}, "eps")
            angle = convert(arguments["entry"])

            def entry_build(direction: bool) -> ProgramRecord:
                return uniform_entry_program(width, angle, adjoint=direction)

        else:
            table = entry_table(
                width, arguments["data"], _integer(arguments, "entry_bitsize")
            )

            def entry_build(direction: bool) -> ProgramRecord:
                return explicit_entry_program(width, table, adjoint=direction)

        record = make(name, registers, entry_build)
        entries[record.functions[0]] = width, (record, entry_build(True))
        return record

    def index_owner(
        child: FunctionRecord,
    ) -> tuple[SymbolicInt, SymbolicInt, ExactPair]:
        if child in indices:
            return indices[child]
        if index_context is not None:
            return index_context(child)
        raise SerializationError("Missing exact native sparse index context")

    def entry_owner(child: FunctionRecord) -> tuple[SymbolicInt, ExactPair]:
        if child in entries:
            return entries[child]
        if entry_context is not None:
            return entry_context(child)
        raise SerializationError("Missing exact native sparse entry context")

    def hermitian(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if arguments.keys() != {"col_oracle", "entry_oracle", "eps"}:
            raise SerializationError(
                "Hermitian sparse matrix requires column, entry and eps"
            )
        _real_argument(arguments, "eps")
        column, entry = arguments["col_oracle"], arguments["entry_oracle"]
        if not isinstance(column, FunctionRecord) or not isinstance(
            entry, FunctionRecord
        ):
            raise SerializationError(
                "Hermitian sparse matrix requires native oracle functions"
            )
        width, sparsity, column_pair = index_owner(column)
        entry_width, entry_pair = entry_owner(entry)
        if width != entry_width:
            raise SerializationError("Column and entry oracles must have same bitsize")

        def build(direction: bool) -> ProgramRecord:
            return hermitian_sparse_matrix_component_program(
                width, sparsity, column_pair, entry_pair, adjoint=direction
            )

        record = make(name, registers, build)

        def positive_build(direction: bool) -> ProgramRecord:
            return hermitian_sparse_matrix_component_program(
                width,
                sparsity,
                column_pair,
                entry_pair,
                controlled=True,
                adjoint=direction,
            )

        positive_builders[record.functions[0]] = positive_build
        return record

    def matrix(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if arguments.keys() != {"row_oracle", "col_oracle", "entry_oracle", "eps"}:
            raise SerializationError(
                "Sparse matrix requires row, column, entry and eps"
            )
        _real_argument(arguments, "eps")
        row, column, entry = (
            arguments[key] for key in ("row_oracle", "col_oracle", "entry_oracle")
        )
        if (
            not isinstance(row, FunctionRecord)
            or not isinstance(column, FunctionRecord)
            or not isinstance(entry, FunctionRecord)
        ):
            raise SerializationError(
                "Sparse matrix requires resolved native oracle functions"
            )

        row_width, row_sparse, row_pair = index_owner(row)
        column_width, column_sparse, column_pair = index_owner(column)
        entry_width, entry_pair = entry_owner(entry)
        if row_width != column_width or row_width != entry_width:
            raise SerializationError(
                "Row, column, and entry oracles must have same bitsize"
            )
        if row_sparse != column_sparse:
            raise SerializationError(
                "Unequal row and column sparsities are not supported"
            )

        def build(direction: bool) -> ProgramRecord:
            return sparse_matrix_component_program(
                row_width,
                row_sparse,
                row_pair,
                column_pair,
                entry_pair,
                adjoint=direction,
            )

        return make(name, registers, build)

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in recipes
        ):
            build, direction = recipes[child]
            record = make(name, registers, build, not direction)
            if child in positive_builders:
                original = positive_builders[child]
                positive_builders[record.functions[0]] = lambda reverse: original(
                    not reverse
                )
            return record
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact sparse component adjoint context")

    def bind(kind: str) -> ComponentFactory:
        def factory(
            name: str,
            registers: tuple[Register, ...] | None,
            arguments: Mapping[str, ResolvedArgument],
        ) -> ProgramRecord:
            return oracle(kind, name, registers, arguments)

        return factory

    def positive(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child not in positive_builders:
            raise SerializationError(
                "Missing exact Hermitian sparse positive-control owner"
            )
        record = _rename_program(
            positive_builders[child](False), child.name + "_positive"
        )
        return record, _stored_control_adder(record.functions[0])

    def controlled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child, spec = arguments.get("subbloq"), arguments.get("ctrl_spec")
        if arguments.keys() != {"subbloq", "ctrl_spec"} or not isinstance(
            child, FunctionRecord
        ):
            raise SerializationError(
                "Controlled sparse matrix requires child and control specification"
            )
        if not isinstance(spec, ControlSpecification) or len(spec.clauses) != 1:
            raise SerializationError(
                "Controlled sparse matrix requires one positive scalar QBit"
            )
        dtype, values = spec.clauses[0]
        if dtype != quantum_bit_type() or values.shape or int(values) != 1:
            raise SerializationError(
                "Controlled sparse matrix requires one positive scalar QBit"
            )
        if child not in positive_builders:
            raise SerializationError(
                "Missing exact Hermitian sparse positive-control owner"
            )
        return make(name, registers, positive_builders[child])

    return {
        **controlled_via_and_factories(control_system=positive),
        "qualtran._infra.controlled.Controlled": controlled,
        **{
            _PREFIX + kind: bind(kind)
            for kind in (
                "TopLeftRowColumnOracle",
                "SymmetricBandedRowColumnOracle",
                "UniformEntryOracle",
                "ExplicitEntryOracle",
            )
        },
        _PREFIX + "SparseMatrix": matrix,
        "qualtran.bloqs.block_encoding.sparse_matrix_hermitian.UniformSqrtEntryOracle": bind(
            "UniformSqrtEntryOracle"
        ),
        "qualtran.bloqs.block_encoding.sparse_matrix_hermitian.SparseMatrixHermitian": hermitian,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
