"""Neutral-atom register provider: geometry, connectivity, and preflight.

A neutral-atom machine is not a fixed lattice. The vendor publishes a register
of optically trapped atoms, each at a physical coordinate, and which pairs
interact follows from how far apart they are. The connectivity a compiler must
route against is therefore derived from the declared geometry rather than read
from a coupling list, and the radius that decides "interacting" is a property of
the atom species and the excitation scheme, not of this package. That radius is
a required argument here: a default would silently invent a device.

This adapter implements discovery and preflight. It does not submit, because the
program a neutral-atom machine executes is an analog schedule over the register,
and this package emits neither that schedule nor the vendor's serialization of
it. A submission attempt is refused before a job exists and names that gap,
rather than being translated into a digital circuit the hardware would not run
as declared.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ...compiler import CouplingMap
from ...deployment.cloud import CloudBackendProfile, DeploymentPackage
from .contracts import (
    ProviderSubmissionPreview,
    ProviderTaskHandle,
    QuantumProvider,
    preflight_submission,
)
from .declaration import (
    declared_gate_names,
    declared_text,
    declared_value,
    declared_width,
    edge_pair,
)

NEUTRAL_ATOM_PROVIDER = "neutral-atom"

#: The names a vendor register uses for its atom list.
REGISTER_NAMES = ("sites", "atoms", "register", "positions", "qubits")

#: A register is declared in the vendor's own length unit. Coordinates are
#: compared in that unit, so the interaction radius must be given in it too.
REGISTER_UNITS = ("um", "micrometers", "nm", "m")

_ANALOG_BLOCKER = "neutral_atom_submission_requires_an_analog_program_schedule"


def _coordinate(value: Any, *, owner: str, index: int) -> tuple[float, float]:
    if isinstance(value, Mapping):
        x = declared_value(value, "x", "X", default=None)
        y = declared_value(value, "y", "Y", default=None)
        if x is None or y is None:
            position = declared_value(
                value, "position", "coord", "coordinate", default=None
            )
            if position is None:
                raise ValueError(
                    f"{owner} site {index} must declare x and y coordinates"
                )
            value = position
        else:
            value = (x, y)
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{owner} site {index} must declare x and y coordinates")
    if len(value) < 2:
        raise ValueError(f"{owner} site {index} must declare x and y coordinates")
    coordinates: list[float] = []
    for component in value[:2]:
        if isinstance(component, bool) or not isinstance(component, int | float):
            raise ValueError(f"{owner} site {index} coordinates must be numbers")
        number = float(component)
        if not math.isfinite(number):
            raise ValueError(f"{owner} site {index} coordinates must be finite")
        coordinates.append(number)
    return (coordinates[0], coordinates[1])


def neutral_atom_sites(
    declaration: Any, *, owner: str = "Neutral atom"
) -> tuple[tuple[float, float], ...]:
    """Return the register's atom coordinates, refusing an unusable register."""

    raw = declared_value(declaration, *REGISTER_NAMES, default=None)
    if raw is None:
        raise ValueError(f"{owner} declaration does not expose a register")
    if isinstance(raw, Mapping):
        entries: Sequence[Any] = [
            (
                {**dict(coordinate), "name": name}
                if isinstance(coordinate, Mapping)
                else {"position": coordinate, "name": name}
            )
            for name, coordinate in raw.items()
        ]
    elif isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise ValueError(f"{owner} register must be a sequence of sites")
    else:
        entries = raw
    if not entries:
        raise ValueError(f"{owner} register does not declare any sites")
    sites: list[tuple[float, float]] = []
    for index, entry in enumerate(entries):
        sites.append(_coordinate(entry, owner=owner, index=index))
    if len(set(sites)) != len(sites):
        raise ValueError(f"{owner} register places two atoms at the same coordinate")
    return tuple(sites)


def interaction_edges(
    sites: Sequence[tuple[float, float]],
    *,
    interaction_radius: float,
    owner: str = "Neutral atom",
) -> tuple[tuple[int, int], ...]:
    """Return every pair of sites closer than the declared interaction radius.

    The radius is inclusive: a pair exactly at the radius interacts, which is
    the boundary convention a distance threshold is normally stated with. A
    non-positive or non-finite radius is refused rather than treated as
    "no interactions", because that would report a connected register as
    isolated.
    """

    if isinstance(interaction_radius, bool) or not isinstance(
        interaction_radius, int | float
    ):
        raise ValueError(f"{owner} interaction radius must be a number")
    radius = float(interaction_radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"{owner} interaction radius must be positive and finite")
    edges: set[tuple[int, int]] = set()
    for left in range(len(sites)):
        for right in range(left + 1, len(sites)):
            distance = math.dist(sites[left], sites[right])
            if distance <= radius:
                edges.add(edge_pair((left, right), owner=owner))
    return tuple(sorted(edges))


def neutral_atom_backend_profile(
    declaration: Any,
    *,
    interaction_radius: float,
    basis_gates: Sequence[str] = (),
    register_units: str = "um",
) -> CloudBackendProfile:
    """Build a fail-closed backend profile from one neutral-atom register.

    ``interaction_radius`` must be given in ``register_units``, the same unit
    the register's coordinates use. The declared atom count, when the vendor
    states one, must agree with the number of coordinate pairs.
    """

    owner = "Neutral atom"
    if register_units not in REGISTER_UNITS:
        raise ValueError(
            f"register_units must be one of {', '.join(REGISTER_UNITS)}, "
            f"got {register_units!r}"
        )
    name = declared_text(
        declaration, "name", "device", "id", owner=owner, fact="device name"
    )
    sites = neutral_atom_sites(declaration, owner=owner)
    declared_atoms = declared_value(
        declaration, "n_qubits", "n_atoms", "atom_count", default=None
    )
    if declared_atoms is not None:
        stated, _source = declared_width(
            {"n_atoms": declared_atoms}, "n_atoms", owner=owner, fact="atom count"
        )
        if stated != len(sites):
            raise ValueError(
                f"{owner} declaration states {declared_atoms!r} atoms but places "
                f"{len(sites)} sites"
            )
    edges = interaction_edges(sites, interaction_radius=interaction_radius, owner=owner)
    coupling_map = CouplingMap(n_wires=len(sites), edges=edges)
    declared_gates = declared_gate_names(
        declared_value(declaration, "gates", "basis_gates", default=None), owner=owner
    )
    return CloudBackendProfile(
        provider=NEUTRAL_ATOM_PROVIDER,
        name=name,
        n_qubits=len(sites),
        basis_gates=(
            declared_gate_names(basis_gates, owner=owner)
            if basis_gates
            else declared_gates
        ),
        coupling_map=coupling_map,
        supports_openqasm=False,
        supports_dynamic_circuits=False,
        is_simulator=False,
        metadata={
            "capability_source": "neutral-atom register declaration",
            "register_units": register_units,
            "interaction_radius": float(interaction_radius),
            "declared_site_count": len(sites),
            "derived_coupling_count": len(edges),
            "connectivity_source": "derived_from_declared_geometry",
        },
    )


class NeutralAtomProvider(QuantumProvider):
    """Describe and preflight one neutral-atom register without submitting.

    Parameters:
        register: The declared register: a mapping with ``name`` and one of
            :data:`REGISTER_NAMES`, or an object exposing the same.
        interaction_radius: Distance within which two atoms interact, in
            ``register_units``.
        register_units: The unit the register's coordinates use.
        basis_gates: The gate set the register's digital mode declares, if any.

    Raises:
        ValueError: The register, radius, or unit declaration is unusable.
        RuntimeError: A submission was attempted on this adapter.

    **What is not here.** This adapter submits nothing. A neutral-atom machine
    executes an analog schedule over the register, and this package emits
    neither that schedule nor a vendor's serialization of it, so
    :meth:`submit` refuses and :meth:`preflight` reports the refusal as its
    blocker instead of translating the program into a digital circuit the
    hardware would not run as declared.
    """

    provider = NEUTRAL_ATOM_PROVIDER

    def __init__(
        self,
        *,
        register: Any,
        interaction_radius: float,
        register_units: str = "um",
        basis_gates: Sequence[str] = (),
    ) -> None:
        self.register = register
        self.interaction_radius = interaction_radius
        self.register_units = register_units
        self.basis_gates = tuple(basis_gates)

    def list_devices(
        self, n_qubits: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        profile = neutral_atom_backend_profile(
            self.register,
            interaction_radius=self.interaction_radius,
            basis_gates=self.basis_gates,
            register_units=self.register_units,
        )
        if n_qubits is not None and profile.n_qubits < n_qubits:
            return ()
        return (profile,)

    def dry_run(self, package: DeploymentPackage) -> ProviderSubmissionPreview:
        """Validate locally and name what this adapter cannot emit."""

        preview = preflight_submission(
            package,
            provider=self.provider,
            qasm_versions=None,
            dynamic_circuits_supported=False,
        )
        blockers = (*preview.blockers, _ANALOG_BLOCKER)
        return ProviderSubmissionPreview(
            provider=preview.provider,
            backend=preview.backend,
            shots=preview.shots,
            program_format="analog_register_schedule",
            program=preview.program,
            compatible=False,
            blockers=blockers,
        )

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        raise RuntimeError(
            "neutral-atom submission requires an analog program schedule, which "
            "this package does not emit; declare and preflight the register "
            "instead, and submit through a vendor program builder"
        )


__all__ = (
    "NEUTRAL_ATOM_PROVIDER",
    "REGISTER_NAMES",
    "REGISTER_UNITS",
    "NeutralAtomProvider",
    "interaction_edges",
    "neutral_atom_backend_profile",
    "neutral_atom_sites",
)
