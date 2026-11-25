# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 chemistry/sparse/prepare.py.

"""Bind sparse PREPARE with an explicit native measured lookup inverse."""

from collections.abc import Mapping

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.lookup.qroam_clean_component import qroam_clean_program
from ..registers.lookup.qroam_erasure_component import qroam_measurement_erasure_program
from ..registers.lookup.source_qroam_clean import QroamCleanTable
from ..registers.preparation.sparse_chemistry_prepare_component import (
    sparse_prepare_component_pair,
)
from ..registers.preparation.sparse_chemistry_prepare_metadata import (
    SPARSE_PREPARE_TARGETS,
    sparse_preparation_signature,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_word_factories import _rename_program


def native_sparse_preparation_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Retain constructor fields while selecting the authorized native erasure.

    The source's independent QROAM adjoint factory remains model-only. This
    assembly explicitly owns measured inversion of matching prepared outputs;
    it does not attribute native primitive erasure gates to an upstream body.
    Rotation precision and normalization remain interchange metadata: the
    original executable body does not use those constructor fields.
    """
    pairs: dict[FunctionRecord, tuple[ProgramRecord, ProgramRecord]] = {}

    def store(
        name: str,
        registers: tuple[Register, ...] | None,
        pair: tuple[ProgramRecord, ProgramRecord],
    ) -> ProgramRecord:
        forward, inverse = pair
        if registers is not None and registers != forward.functions[0].registers:
            raise SerializationError("Invalid sparse PREPARE source signature")
        record = _rename_program(forward, name)
        pairs[record.functions[0]] = record, inverse
        return record

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        required = {
            "num_spin_orb",
            "num_non_zero",
            "num_bits_state_prep",
            "alt_pqrs",
            "alt_theta",
            "alt_one_body",
            "ind_pqrs",
            "theta",
            "one_body",
            "keep",
        }
        if not required <= arguments.keys() or arguments.keys() - required - {
            "num_bits_rot_aa",
            "sum_of_l1_coeffs",
            "log_block_size",
        }:
            raise SerializationError("Invalid sparse PREPARE constructor arguments")
        size, rows, mu, log = (
            arguments["num_spin_orb"],
            arguments["num_non_zero"],
            arguments["num_bits_state_prep"],
            arguments.get("log_block_size", 1),
        )
        if (
            type(size) is not int
            or size < 4
            or size % 2
            or type(rows) is not int
            or rows < 2
            or type(mu) is not int
            or mu < 1
            or type(log) is not int
        ):
            raise SerializationError(
                "Sparse PREPARE requires supported concrete physical dimensions"
            )
        assert (
            isinstance(size, int)
            and isinstance(rows, int)
            and isinstance(mu, int)
            and isinstance(log, int)
        )

        def data(key: str, shape: tuple[int, ...]) -> tuple[int, ...]:
            raw = arguments[key]
            if not isinstance(raw, (tuple, list, np.ndarray)):
                raise SerializationError(
                    "Sparse PREPARE datasets require integer vectors"
                )
            array = np.asarray(raw, dtype=object)
            if array.shape != shape or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in array.flat
            ):
                raise SerializationError(
                    "Sparse PREPARE dataset shape or integer type is invalid"
                )
            return tuple(int(value) for value in array.flat)

        indices = data("ind_pqrs", (4, rows))
        alternatives = data("alt_pqrs", (4, rows))
        datasets = (
            *(indices[i * rows : (i + 1) * rows] for i in range(4)),
            data("theta", (rows,)),
            data("one_body", (rows,)),
            *(alternatives[i * rows : (i + 1) * rows] for i in range(4)),
            data("alt_theta", (rows,)),
            data("alt_one_body", (rows,)),
            data("keep", (rows,)),
        )
        table = QroamCleanTable((rows,), (log,), datasets)
        signature = sparse_preparation_signature(size, rows, mu, log)
        widths = tuple(
            int(signature.get_right(target).bitsize)
            for target in SPARSE_PREPARE_TARGETS
        )
        if any(
            not 0 <= value < 1 << width
            for values, width in zip(datasets, widths, strict=True)
            for value in values
        ):
            raise SerializationError("Sparse PREPARE data must fit its target storage")
        lookup = (
            qroam_clean_program(table, widths, reserve_workspace=False),
            qroam_measurement_erasure_program(table, widths),
        )
        return store(
            name, registers, sparse_prepare_component_pair(signature, rows, lookup)
        )

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in pairs
        ):
            forward, inverse = pairs[child]
            return store(name, registers, (inverse, forward))
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing native sparse PREPARE inverse context")

    return {
        "qualtran.bloqs.chemistry.sparse.prepare.PrepareSparse": forward,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
