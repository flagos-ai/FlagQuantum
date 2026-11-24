# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 block_encoding/chebyshev_polynomial.py.

"""Bind Chebyshev constructors to exact native block and coefficient owners."""

from collections.abc import Callable, Mapping
from math import pi

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ..registers.component_control import ControlAdder, _stored_control_adder
from ..registers.preparation.chebyshev_component import (
    chebyshev_component_pair,
    scaled_chebyshev_component_pair,
)
from ..registers.preparation.identity_preparation_component import (
    identity_preparation_program,
)
from ..registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from ..registers.primitives.global_phase_component import scalar_phase_program
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_controlled_factories import controlled_via_and_factories
from .legacy_modular_addition_factories import _integer
from .legacy_unitary_block_factories import unitary_block_factories
from .legacy_word_factories import _rename_program

ExactPair = tuple[ProgramRecord, ProgramRecord]
PairOwner = Callable[[FunctionRecord], ExactPair]
ScaledOwner = Callable[[FunctionRecord, SymbolicInt, SymbolicInt], ExactPair]


def chebyshev_factories(
    *,
    child_owner: PairOwner | None = None,
    positive_child_owner: PairOwner | None = None,
    scaled_owner: ScaledOwner | None = None,
    positive_scaled_owner: ScaledOwner | None = None,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Keep constructor pairs scoped; coefficient normalization stays external.

    Default Unitary constructors report their exact pairs through the existing
    owner. Other children need an explicit pair. Scaled polynomial coefficients
    and source normalization cannot be inferred from a native unitary body;
    scaled_owner returns the actual linear_combination property's native pair.
    Positive owners supply the original specialized child/property pair rather
    than controlling an inferred body or changing coefficient ownership.
    """
    pairs: dict[FunctionRecord, ExactPair] = {}
    roles: dict[FunctionRecord, tuple[FunctionRecord, int, bool]] = {}
    scaled_roles: dict[
        FunctionRecord, tuple[FunctionRecord, SymbolicInt, SymbolicInt, bool]
    ] = {}
    controls: dict[FunctionRecord, tuple[FunctionRecord, ControlSpecification]] = {}
    units = unitary_block_factories(
        on_pair=lambda child, pair: pairs.__setitem__(child, pair)
    )

    def store(
        name: str, registers: tuple[Register, ...] | None, pair: ExactPair
    ) -> ProgramRecord:
        forward, inverse = pair
        if registers is not None and registers != forward.functions[0].registers:
            raise SerializationError("Invalid Chebyshev constructor register signature")
        if forward.functions[0].registers != inverse.functions[0].registers:
            raise SerializationError("Chebyshev requires matching exact pair ports")
        result = _rename_program(forward, name)
        pairs[result.functions[0]] = result, inverse
        return result

    def child_pair(child: FunctionRecord) -> ExactPair:
        if child not in pairs:
            if child_owner is None:
                raise SerializationError("Missing exact Chebyshev child owner")
            pairs[child] = child_owner(child)
        return pairs[child]

    def reflection(width: int, *, controlled: bool = False) -> ProgramRecord:
        selections = (Register("reg0_", opaque_register_type(width)),)
        preparation = identity_preparation_program(selections)
        return prepared_reflection_program(
            preparation,
            preparation,
            ("reg0_",),
            control_value=1 if controlled else None,
            global_phase=-1,
            scalar_phase=None if controlled else scalar_phase_program(pi),
        )

    def ordinary(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child = arguments.get("block_encoding")
        if arguments.keys() != {"block_encoding", "order"} or not isinstance(
            child, FunctionRecord
        ):
            raise SerializationError("Chebyshev requires its block and order")
        order = _integer(arguments, "order")
        forward, inverse = child_pair(child)
        ports = forward.functions[0].registers
        if ports is None:
            raise SerializationError("Chebyshev requires typed child ports")
        if type(order) is not int or any(type(reg.bitsize) is not int for reg in ports):
            declaration = FunctionRecord("main", (), registers=ports)
            record = ProgramRecord((declaration,))
            return store(name, registers, (record, record))
        ancilla = next((int(reg.bitsize) for reg in ports if reg.name == "ancilla"), 0)
        result = store(
            name,
            registers,
            chebyshev_component_pair(
                forward,
                inverse,
                order,
                reflection=reflection(ancilla) if order and ancilla else None,
            ),
        )
        roles[result.functions[0]] = child, order, False
        return result

    def positive_pair(child: FunctionRecord) -> ExactPair:
        if child in scaled_roles:
            if positive_scaled_owner is None:
                raise SerializationError(
                    "Missing exact positive scaled coefficient owner"
                )
            original, order, precision, inverse = scaled_roles[child]
            pair = scaled_chebyshev_component_pair(
                positive_scaled_owner(original, order, precision), controlled=True
            )
            return (pair[1], pair[0]) if inverse else pair
        if child not in roles or positive_child_owner is None:
            raise SerializationError("Missing exact positive Chebyshev child owner")
        original, order, inverse = roles[child]
        forward, backward = positive_child_owner(original)
        ports = forward.functions[0].registers
        if ports is None:
            raise SerializationError("Positive Chebyshev requires typed child ports")
        ancilla = next((reg.bitsize for reg in ports if reg.name == "ancilla"), 0)
        if type(ancilla) is not int:
            raise SerializationError("Positive Chebyshev requires concrete ancilla")
        pair = chebyshev_component_pair(
            forward,
            backward,
            order,
            reflection=(
                reflection(ancilla, controlled=True) if order and ancilla else None
            ),
            controlled=True,
        )
        return (pair[1], pair[0]) if inverse else pair

    def positive(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        record = _rename_program(positive_pair(child)[0], child.name + "_positive")
        return record, _stored_control_adder(record.functions[0])

    via_and = controlled_via_and_factories(control_system=positive)[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ]

    def controlled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child, spec = arguments.get("subbloq"), arguments.get("ctrl_spec")
        if (
            arguments.keys() != {"subbloq", "ctrl_spec"}
            or not isinstance(child, FunctionRecord)
            or not isinstance(spec, ControlSpecification)
            or len(spec.clauses) != 1
        ):
            raise SerializationError("Chebyshev requires one positive scalar QBit")
        dtype, values = spec.clauses[0]
        if dtype != quantum_bit_type() or values.shape or int(values) != 1:
            raise SerializationError("Chebyshev requires one positive scalar QBit")
        return store(name, registers, positive_pair(child))

    def predicate(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        result = via_and(name, registers, arguments)
        child, spec = arguments.get("subbloq"), arguments.get("ctrl_spec")
        assert isinstance(child, FunctionRecord) and isinstance(
            spec, ControlSpecification
        )
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        controls[root] = child, spec
        return result

    def scaled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child = arguments.get("block_encoding")
        if (
            not {"block_encoding", "order"} <= arguments.keys()
            or arguments.keys() - {"block_encoding", "order", "lambd_bits"}
            or not isinstance(child, FunctionRecord)
        ):
            raise SerializationError(
                "Scaled Chebyshev requires block, order and precision"
            )
        if scaled_owner is None:
            raise SerializationError("Missing exact scaled Chebyshev coefficient owner")
        order = _integer(arguments, "order")
        precision = (
            _integer(arguments, "lambd_bits") if "lambd_bits" in arguments else 5
        )
        result = store(
            name,
            registers,
            scaled_chebyshev_component_pair(scaled_owner(child, order, precision)),
        )
        scaled_roles[result.functions[0]] = child, order, precision, False
        return result

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
            result = store(name, registers, (inverse, forward))
            if child in roles:
                original, order, direction = roles[child]
                roles[result.functions[0]] = original, order, not direction
            if child in scaled_roles:
                original, scaled_order, precision, direction = scaled_roles[child]
                scaled_roles[result.functions[0]] = (
                    original,
                    scaled_order,
                    precision,
                    not direction,
                )
            return result
        if arguments.keys() == {"subbloq"} and isinstance(child, FunctionRecord):
            if child in controls:
                original, spec = controls[child]
                reverse = adjoint(name + "_inverse", None, {"subbloq": original})
                function = (
                    reverse.functions[0]
                    if isinstance(reverse, ProgramRecord)
                    else reverse
                )
                return predicate(
                    name, registers, {"subbloq": function, "ctrl_spec": spec}
                )
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact Chebyshev adjoint context")

    return {
        **units,
        "qualtran.bloqs.block_encoding.chebyshev_polynomial.ChebyshevPolynomial": ordinary,
        "qualtran.bloqs.block_encoding.chebyshev_polynomial.ScaledChebyshevPolynomial": scaled,
        "qualtran._infra.adjoint.Adjoint": adjoint,
        "qualtran._infra.controlled.Controlled": controlled,
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd": predicate,
    }
