# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 chemistry/thc/prepare.py.

"""Resolve literal THC preparation graphs with explicit source model leaves."""

from collections.abc import Mapping

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.preparation.thc_preparation_component import thc_prepare_component_pair
from ..registers.preparation.thc_preparation_metadata import (
    THC_PREPARE_TARGETS,
    thc_preparation_signature,
)
from ..registers.preparation.thc_uniform_source_component import (
    literal_thc_uniform_pair,
)
from ..registers.sorting.contiguous_index_component import contiguous_index_program
from .legacy_comparison_factories import comparison_factories
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_qroam_factories import qroam_factories
from .legacy_word_factories import _rename_program


def thc_preparation_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Keep literal source construction separate from mathematical THC preparation.

    The original uniform source has a model-only GT and an invalid clean Free.
    Its native logical graph preserves both, and cannot physically bind. The
    existing mathematical Algorithms operation is not selected by this factory.
    """
    pairs: dict[FunctionRecord, tuple[ProgramRecord, ProgramRecord]] = {}
    greater = comparison_factories()[
        "qualtran.bloqs.arithmetic.comparison.GreaterThanConstant"
    ]
    lookups = qroam_factories(reserve_workspace=False)

    def store(
        name: str,
        registers: tuple[Register, ...] | None,
        pair: tuple[ProgramRecord, ProgramRecord],
    ) -> ProgramRecord:
        forward, inverse = pair
        if registers is not None and registers != forward.functions[0].registers:
            raise SerializationError("Invalid literal THC preparation signature")
        result = _rename_program(forward, name)
        pairs[result.functions[0]] = result, inverse
        return result

    def uniform(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if arguments.keys() != {"num_mu", "num_spin_orb"}:
            raise SerializationError("THC uniform requires num_mu and num_spin_orb")
        m, n = arguments["num_mu"], arguments["num_spin_orb"]
        if type(m) is not int or type(n) is not int or m < 1 or n < 0:
            raise SerializationError("THC uniform requires concrete integer dimensions")
        assert isinstance(m, int) and isinstance(n, int)
        declaration = greater(
            "literal_thc_gt", None, {"bitsize": m.bit_length(), "val": n // 2}
        )
        assert isinstance(declaration, FunctionRecord)
        inverse = FunctionRecord(
            "literal_thc_gt_inverse", (), registers=declaration.registers
        )
        pair = literal_thc_uniform_pair(
            m, n, (ProgramRecord((declaration,)), ProgramRecord((inverse,)))
        )
        return store(name, registers, pair)

    def prepare(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        required = {
            "num_mu",
            "num_spin_orb",
            "alt_mu",
            "alt_nu",
            "alt_theta",
            "theta",
            "keep",
            "keep_bitsize",
            "sum_of_l1_coeffs",
        }
        if not required <= arguments.keys() or arguments.keys() - required - {
            "log_block_size"
        }:
            raise SerializationError("Invalid THC PREPARE constructor arguments")
        m, n, precision, log = (
            arguments["num_mu"],
            arguments["num_spin_orb"],
            arguments["keep_bitsize"],
            arguments.get("log_block_size", 0),
        )
        if (
            type(m) is not int
            or m < 1
            or type(n) is not int
            or n < 0
            or type(precision) is not int
            or precision < 1
            or type(log) is not int
            or log < 0
        ):
            raise SerializationError("THC PREPARE requires concrete integer dimensions")
        assert (
            isinstance(m, int)
            and isinstance(n, int)
            and isinstance(precision, int)
            and isinstance(log, int)
        )
        if log == 0:
            raise SerializationError(
                "Original zero-block THC PREPARE omits junk_target0_"
            )
        rows = m * (m + 1) // 2 + n // 2

        def dataset(key: str) -> tuple[int, ...]:
            raw = arguments[key]
            if not isinstance(raw, (tuple, list, np.ndarray)):
                raise SerializationError("THC PREPARE data require integer vectors")
            values = np.asarray(raw, dtype=object)
            if values.shape != (rows,) or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in values.flat
            ):
                raise SerializationError(
                    "THC PREPARE data shape or integer type is invalid"
                )
            return tuple(int(value) for value in values)

        data = tuple(dataset(key) for key in THC_PREPARE_TARGETS)
        widths = (1, 1, m.bit_length(), m.bit_length(), precision)
        if any(
            not 0 <= value < 1 << width
            for column, width in zip(data, widths, strict=True)
            for value in column
        ):
            raise SerializationError("THC PREPARE data must fit their target words")
        lookup = lookups["qualtran.bloqs.data_loading.qroam_clean.QROAMClean"](
            name + "_lookup",
            None,
            {
                "data_or_shape": data,
                "selection_bitsizes": ((rows - 1).bit_length(),),
                "target_bitsizes": widths,
                "log_block_sizes": (log,),
            },
        )
        assert isinstance(lookup, ProgramRecord)
        inverse_lookup = lookups[
            "qualtran.bloqs.data_loading.qroam_clean.QROAMCleanAdjointWrapper"
        ](
            name + "_lookup_inverse",
            None,
            {"qroam_clean": lookup.functions[0], "log_block_sizes": (log,)},
        )
        assert isinstance(inverse_lookup, ProgramRecord)
        prepared = uniform(name + "_uniform", None, {"num_mu": m, "num_spin_orb": n})
        uniform_pair = pairs[prepared.functions[0]]
        index = contiguous_index_program(m.bit_length(), (rows - 1).bit_length())
        return store(
            name,
            registers,
            thc_prepare_component_pair(
                thc_preparation_signature(m, n, precision, log),
                uniform_pair,
                (index, index),
                (lookup, inverse_lookup),
            ),
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
        raise SerializationError("Missing literal THC preparation inverse context")

    return {
        "qualtran.bloqs.chemistry.thc.prepare.UniformSuperpositionTHC": uniform,
        "qualtran.bloqs.chemistry.thc.prepare.PrepareTHC": prepare,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
