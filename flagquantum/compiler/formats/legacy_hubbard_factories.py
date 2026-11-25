# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors and call declarations follow Qualtran v0.7.0 Hubbard sources.

"""Bind source Hubbard declarations without inventing ordered execution bodies."""

from collections.abc import Mapping
from functools import partial
from typing import Literal, cast

from ...core.ir.program import FunctionRecord
from ...core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, SymbolicReal
from ...errors import SerializationError
from ..registers.layout.fermionic_signatures import (
    fermionic_pair_signature,
    four_mode_signature,
    spin_word_signature,
)
from .legacy_component_arguments import ComponentArgument, ComponentArgumentValue
from .legacy_component_library import (
    ComponentDeclaration,
    ComponentLibrary,
    ComponentLibraryEntry,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_rotation_factories import _real_argument

HubbardKind = Literal[
    "HoppingPlaquette", "HoppingTile", "HoppingTileHWP", "Interaction", "InteractionHWP"
]
_PREFIX = "qualtran.bloqs.chemistry.trotter.hubbard."
_KINDS: tuple[HubbardKind, ...] = (
    "HoppingPlaquette",
    "HoppingTile",
    "HoppingTileHWP",
    "Interaction",
    "InteractionHWP",
)


def _identity(kind: HubbardKind) -> str:
    module = "hopping" if kind.startswith("Hopping") else "interaction"
    return _PREFIX + module + "." + kind


def _declaration(
    identity: str,
    arguments: Mapping[str, ResolvedArgument],
    signature: Signature,
) -> ComponentDeclaration:
    return ComponentDeclaration(
        identity,
        tuple(
            ComponentArgument(key, cast(ComponentArgumentValue, value))
            for key, value in arguments.items()
        ),
        tuple(signature),
    )


def _root(
    kind: HubbardKind, arguments: Mapping[str, ResolvedArgument]
) -> ComponentDeclaration:
    if kind not in _KINDS:
        raise SerializationError(f"Unknown Hubbard source declaration {kind!r}")
    plaquette = kind == "HoppingPlaquette"
    hopping = kind.startswith("Hopping")
    required = {"kappa"} if plaquette else {"length", "angle"}
    if not hopping:
        required.add("hubb_u")
    allowed = (
        required | {"eps"} | ({"tau", "pink"} if hopping and not plaquette else set())
    )
    if not required <= arguments.keys() or arguments.keys() - allowed:
        raise SerializationError("Invalid Hubbard source constructor arguments")
    values: dict[str, ResolvedArgument] = dict(arguments)
    values.setdefault("eps", 1e-9)
    for key in ("eps", "kappa" if plaquette else "angle"):
        _real_argument(values, key)
    if plaquette:
        signature = four_mode_signature()
    else:
        length = _integer(values, "length")
        if hopping:
            if isinstance(length, int) and length % 2:
                raise ValueError("Only even length lattices are supported")
            values.setdefault("tau", 1.0)
            values.setdefault("pink", True)
            _real_argument(values, "tau")
            pink = values["pink"]
            if type(pink) not in (bool, int) or pink not in (0, 1):
                raise SerializationError("Hubbard color flag must be binary")
        else:
            _real_argument(values, "hubb_u")
        signature = spin_word_signature(length)
    return _declaration(_identity(kind), values, signature)


def _factory(
    kind: HubbardKind,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    declaration = _root(kind, arguments)
    if registers is not None and registers != declaration.registers:
        raise SerializationError("Invalid Hubbard source register signature")
    return FunctionRecord(name, (), registers=declaration.registers)


def hubbard_factories() -> dict[str, ComponentFactory]:
    """Resolve all five bodyless source constructors through shared port owners."""
    return {_identity(kind): partial(_factory, kind) for kind in _KINDS}


def hubbard_component_library(
    kind: HubbardKind,
    arguments: Mapping[str, ResolvedArgument],
    *,
    name: str | None = None,
) -> ComponentLibrary:
    """Retain source constructor data and unordered direct-call multiplicities.

    No connections are emitted: the source provides counts, not execution order.
    Precision remains metadata and does not select a synthesis implementation.
    """
    root = _root(kind, arguments)
    values = {argument.name: argument.value for argument in root.arguments}
    if name is None:
        name = (
            f"H_h^{'p' if values['pink'] else 'g'}(HWP)"
            if kind == "HoppingTileHWP"
            else kind
        )
    eps = values["eps"]
    children: list[tuple[ComponentDeclaration, SymbolicInt]] = []

    def fourier(count: SymbolicInt) -> None:
        children.append(
            (
                _declaration(
                    "qualtran.bloqs.qft.two_bit_ffft.TwoBitFFFT",
                    {"k": 0, "n": 1, "eps": eps, "is_adjoint": False},
                    fermionic_pair_signature(),
                ),
                count,
            )
        )

    def rotation(angle: SymbolicReal, count: SymbolicInt) -> None:
        children.append(
            (
                _declaration(
                    "qualtran.bloqs.basic_gates.rotation.Rz",
                    {"angle": angle, "eps": eps},
                    Signature((Register("q", quantum_bit_type()),)),
                ),
                count,
            )
        )

    def phasing(width: SymbolicInt, exponent: SymbolicReal) -> None:
        ports = (Register("x", unsigned_register_type(width)),) if width != 0 else ()
        children.append(
            (
                _declaration(
                    "qualtran.bloqs.rotations.hamming_weight_phasing.HammingWeightPhasing",
                    {"bitsize": width, "exponent": exponent, "eps": eps},
                    Signature(ports),
                ),
                2,
            )
        )

    if kind == "HoppingPlaquette":
        fourier(4)
        rotation(cast(SymbolicReal, values["kappa"]), 2)
    else:
        length = cast(SymbolicInt, values["length"])
        angle = cast(SymbolicReal, values["angle"])
        if kind.startswith("Hopping"):
            angle *= cast(SymbolicReal, values["tau"])
            if kind == "HoppingTileHWP":
                fourier(4 * length**2 // 2)
                phasing(2 * length**2 // 4, angle)
            else:
                children.append(
                    (
                        _root("HoppingPlaquette", {"kappa": angle, "eps": eps}),
                        length**2 // 2,
                    )
                )
        else:
            angle *= cast(SymbolicReal, values["hubb_u"])
            if kind == "InteractionHWP":
                phasing(length**2 // 2, angle)
            else:
                rotation(angle, length**2)
    return ComponentLibrary(
        name,
        (
            ComponentLibraryEntry(
                0,
                root,
                counts=tuple(
                    (index, count) for index, (_, count) in enumerate(children, start=1)
                ),
            ),
            *(
                ComponentLibraryEntry(index, child)
                for index, (child, _) in enumerate(children, start=1)
            ),
        ),
    )
