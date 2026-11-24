# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow Qualtran v0.7.0 swap_network source files.

"""Resolve source exchange networks through shared routing and unary owners."""

from collections.abc import Mapping

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.lookup.multiplexed_swap import MultiplexedSwapLayout
from ..registers.lookup.multiplexed_swap_component import multiplexed_swap_program
from ..registers.lookup.swap_with_zero import swap_with_zero_signature
from ..registers.lookup.swap_with_zero_component import swap_with_zero_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def _register_tuple(value: object) -> tuple[Register, ...]:
    ports = (value,) if isinstance(value, Register) else value
    if not isinstance(ports, (tuple, list)) or not all(
        isinstance(reg, Register) for reg in ports
    ):
        raise SerializationError("Swap network requires typed register metadata")
    return tuple(reg for reg in ports if isinstance(reg, Register))


def _dimensions(value: object) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.dtype.kind not in "iu":
            raise SerializationError("Swap network requires integer dimension vectors")
        value = value.tolist()
    if not isinstance(value, (tuple, list)) or any(type(v) is not int for v in value):
        raise SerializationError("Swap network requires literal integer dimensions")
    return tuple(value)


def swap_network_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Retain source network choices and explicitly owned exact inverse recipes.

    SwapWithZero serializes its direction as uncompute. MultiplexedCSwap uses
    the generic Adjoint wrapper, whose measured unary calls require specialized
    exact inverse children. Unknown wrappers delegate or refuse explicitly.
    """
    recipes: dict[FunctionRecord, tuple[MultiplexedSwapLayout, bool]] = {}

    def make_multiplexed(
        name: str,
        registers: tuple[Register, ...] | None,
        layout: MultiplexedSwapLayout,
        inverse: bool,
    ) -> ProgramRecord:
        ports = tuple(layout.signature)
        if registers is not None and registers != ports:
            raise SerializationError("Invalid multiplexed swap register signature")
        if any(
            type(reg.bitsize) is not int
            or any(type(extent) is not int for extent in reg.shape_symbolic)
            for reg in ports
        ):
            record = ProgramRecord((FunctionRecord("main", (), registers=ports),))
        else:
            record = multiplexed_swap_program(layout, adjoint=inverse)
        record = _rename_program(record, name)
        recipes[record.functions[0]] = layout, inverse
        return record

    def multiplexed(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if not {
            "selection_regs",
            "target_bitsize",
        } <= arguments.keys() or arguments.keys() - {
            "selection_regs",
            "target_bitsize",
            "control_regs",
        }:
            raise SerializationError("Invalid multiplexed swap constructor arguments")
        layout = MultiplexedSwapLayout(
            _register_tuple(arguments["selection_regs"]),
            _integer(arguments, "target_bitsize"),
            _register_tuple(arguments.get("control_regs", ())),
        )
        return make_multiplexed(name, registers, layout, False)

    def swap_zero(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if not {
            "selection_bitsizes",
            "target_bitsize",
            "n_target_registers",
        } <= arguments.keys() or arguments.keys() - {
            "selection_bitsizes",
            "target_bitsize",
            "n_target_registers",
            "uncompute",
        }:
            raise SerializationError("Invalid swap-with-zero constructor arguments")
        widths = _dimensions(arguments["selection_bitsizes"])
        shape = _dimensions(arguments["n_target_registers"])
        word = _integer(arguments, "target_bitsize")
        inverse = bool(arguments.get("uncompute", False))
        ports = tuple(swap_with_zero_signature(widths, word, shape))
        if registers is not None and registers != ports:
            raise SerializationError("Invalid swap-with-zero register signature")
        if type(word) is not int:
            return ProgramRecord((FunctionRecord(name, (), registers=ports),))
        return _rename_program(
            swap_with_zero_program(widths, word, shape, adjoint=inverse),
            name,
        )

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            set(arguments) == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in recipes
        ):
            layout, inverse = recipes[child]
            return make_multiplexed(name, registers, layout, not inverse)
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing multiplexed swap adjoint constructor context")

    return {
        "qualtran.bloqs.swap_network.multiplexed_cswap.MultiplexedCSwap": multiplexed,
        "qualtran.bloqs.swap_network.swap_with_zero.SwapWithZero": swap_zero,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
