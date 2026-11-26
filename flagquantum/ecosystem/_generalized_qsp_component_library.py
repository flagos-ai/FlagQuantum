"""Bind the existing Simulation phase recursion to native GQSP construction."""

from ..compiler.formats.legacy_component_resolution import ComponentFactory
from ..compiler.formats.legacy_gqsp_factories import generalized_qsp_factories
from ..compiler.formats.legacy_unitary_block_factories import UnitaryPairOwner
from ..simulation.numerics.qsp_phases import qsp_phase_factors


def generalized_qsp_library_factories(
    *,
    child_owner: UnitaryPairOwner,
    positive_child_owner: UnitaryPairOwner,
) -> dict[str, ComponentFactory]:
    """Retain explicit native child ownership and keep numerical work in Simulation."""

    def phases(
        left: tuple[complex, ...], right: tuple[complex, ...]
    ) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        theta, phi, lambd = qsp_phase_factors(left, right)
        return tuple(map(float, theta)), tuple(map(float, phi)), lambd

    return generalized_qsp_factories(
        phase_factors=phases,
        child_owner=child_owner,
        positive_child_owner=positive_child_owner,
    )
