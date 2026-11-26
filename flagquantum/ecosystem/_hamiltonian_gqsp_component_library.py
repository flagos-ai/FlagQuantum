"""Bind existing Hamiltonian numerics and exact native dependency ownership."""

from collections.abc import Callable
from dataclasses import dataclass

from ..algorithms.primitives.hamiltonian_gqsp import HamiltonianGQSP
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_hamiltonian_gqsp_factories import (
    HamiltonianGQSPContext,
    hamiltonian_gqsp_factories,
)
from ..compiler.formats.legacy_unitary_block_factories import UnitaryPairOwner
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..errors import SerializationError
from ._generalized_qsp_component_library import generalized_qsp_library_factories


@dataclass(frozen=True)
class HamiltonianGQSPBinding:
    """Associate a native numerical model with its exact typed child definitions."""

    model: HamiltonianGQSP
    walk_pair: tuple[ProgramRecord, ProgramRecord]
    positive_walk_pair: tuple[ProgramRecord, ProgramRecord]
    prepare_pair: tuple[ProgramRecord, ProgramRecord]
    junk_names: tuple[str, ...] = ()


def hamiltonian_gqsp_library_factories(
    binding_owner: Callable[
        [FunctionRecord, ResolvedArgument, ResolvedArgument], HamiltonianGQSPBinding
    ],
) -> dict[str, ComponentFactory]:
    """Reuse native coefficient/phase numerics without importing original SDKs."""

    def context(
        walk: FunctionRecord, time: ResolvedArgument, precision: ResolvedArgument
    ) -> HamiltonianGQSPContext:
        binding = binding_owner(walk, time, precision)
        if binding.model.time != time or binding.model.precision != precision:
            raise SerializationError("Hamiltonian numerical model context mismatch")
        if binding.walk_pair[0].functions[0].registers != walk.registers:
            raise SerializationError("Hamiltonian numerical walk context mismatch")

        def pair_owner(
            pair: tuple[ProgramRecord, ProgramRecord],
        ) -> UnitaryPairOwner:
            def own(child: FunctionRecord) -> tuple[ProgramRecord, ProgramRecord]:
                if child != walk:
                    raise SerializationError("Unknown Hamiltonian walk dependency")
                return pair

            return own

        factory = next(
            iter(
                generalized_qsp_library_factories(
                    child_owner=pair_owner(binding.walk_pair),
                    positive_child_owner=pair_owner(binding.positive_walk_pair),
                ).values()
            )
        )
        model = binding.model.gqsp
        processor = factory(
            "hamiltonian_processor",
            None,
            {
                "U": walk,
                "P": model.p,
                "Q": model.q,
                "negative_power": model.negative_power,
                "precision": model.precision,
            },
        )
        assert isinstance(processor, ProgramRecord)
        return HamiltonianGQSPContext(
            processor, binding.prepare_pair, binding.junk_names
        )

    return hamiltonian_gqsp_factories(context)
