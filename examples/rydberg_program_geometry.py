"""Construct native atom controls and inspect their verified imposed geometry.

Core construction is implemented here. Native numerical consumption and sampled
events remain separate deliveries; this example does not execute on hardware.
"""

from __future__ import annotations

import flagquantum as fq
from flagquantum.errors import CapabilityError
from flagquantum.experimental.simulation import RydbergProgram


def main() -> None:
    program = RydbergProgram(
        atoms=(("a", "left"), ("b", "near")),
        sites={"left": (0.0, 0.0), "near": (2.0, 0.0), "far": (4.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=64.0,
        loss_rate=0.2,
        minimum_separation=1.0,
        transport_channels=("aod",),
        drive_channels=("laser",),
    )
    program.move(
        "b", source="near", destination="far", start=0.0, duration=0.5, channel="aod"
    )
    program.pulse(
        "b", start=0.0, duration=0.5, amplitude=fq.Parameter("omega"), channel="laser"
    )
    program.to_ir()
    print("Atom order:", program.atom_ids)
    print("Product dimensions:", program.dimensions)
    print("Commanded b position at t=0.25:", program.commanded_position("b", 0.25))
    print("Survival effect shape:", tuple(program.survival_operator().shape))
    try:
        fq.run(program)
    except CapabilityError:
        print("Circuit route: refused; native evolution awaits Simulation integration")


if __name__ == "__main__":
    main()
