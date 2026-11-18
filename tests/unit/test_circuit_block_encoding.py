"""Measured block snapshots preserve logical auxiliaries and every branch phase."""

from itertools import product
from math import pi

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.circuit_block_encoding import (
    CircuitBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler.registers.primitives.and_compute import (
    and_compute_ir,
    and_uncompute_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


def measured_block() -> CircuitBlockEncoding:
    snapshots = []
    models = []
    for controlled, adjoint in product((False, True), repeat=2):
        s, a, r = (1, 2, 3) if controlled else (0, 1, 2)
        target = 4 if controlled else 3
        compute = and_compute_ir(s, a, target)
        erase = and_uncompute_ir(s, a, target, 0)
        phase = (
            (Instruction("cphase", (0, s), {"theta": -pi / 2 if adjoint else pi / 2}),)
            if controlled
            else (Instruction("sdg" if adjoint else "s", (s,)),)
        )
        if controlled:
            body = (
                *compute,
                *and_compute_ir(0, target, 5),
                Instruction("cz", (5, r)),
                *and_uncompute_ir(0, target, 5, 1),
                *erase,
            )
        else:
            body = (*compute, Instruction("cz", (target, r)), *erase)
        gates = (*phase, *body) if adjoint else (*body, *phase)
        n = 6 if controlled else 4
        snapshots.append(CircuitIR(n, gates))
        models.append(ComponentCost("Independent test declaration", n))
    return CircuitBlockEncoding(1, 1, 1, 2.0, 0.01, tuple(snapshots), tuple(models))


@pytest.mark.parametrize("controlled,adjoint", tuple(product((False, True), repeat=2)))
def test_all_branches_on_arbitrary_logical_auxiliaries_and_spectator(
    controlled: bool, adjoint: bool
) -> None:
    block = measured_block()
    work, bits = block.requirements(controlled=controlled)
    assert (work, bits) == ((2, 2) if controlled else (1, 1))
    workspace = (1, 3) if controlled else (1,)
    operation = block.make_on(
        system=(2,),
        ancilla=(0,),
        resource=(4,),
        control=(5,) if controlled else (),
        workspace=workspace,
        classical_bits=tuple(range(8, 8 + bits)),
        adjoint=adjoint,
    )
    gates = Circuit(7).append(operation).to_ir().instructions
    initial = torch.zeros(128, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for index in range(128):
        if any((index >> (6 - wire)) & 1 for wire in workspace):
            continue
        s, a, r, c = (((index >> (6 - wire)) & 1) for wire in (2, 0, 4, 5))
        amplitude = complex(index + 1, index % 7 - 3)
        phase = ((1j if not adjoint else -1j) ** s) * ((-1) ** (s * a * r))
        initial[index] = amplitude
        expected[index] = amplitude * (phase if not controlled or c else 1)
    expected /= initial.norm()
    initial /= initial.norm()
    events = sum(g.name == "measure" for g in gates)
    for outcomes in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, gates, outcomes),
            expected / (2 ** (events / 2)),
            atol=2e-13,
            rtol=0,
        )
    assert (
        block.resource_model(controlled=controlled, adjoint=adjoint)
        is block.models[2 * int(controlled) + int(adjoint)]
    )


def test_lcu_child_factory_reuses_the_same_binding_path() -> None:
    block = measured_block()
    ports = {"system": (2,), "ancilla": (0,), "resource": (4,)}
    child = block.bind_child(5, ports, (1, 3), (8, 9), False)
    direct = block.make_on(
        **ports, control=(5,), workspace=(1, 3), classical_bits=(8, 9)
    )
    assert child == direct
    with pytest.raises(ValueError, match="declared work"):
        block.bind_child(5, ports, (), (), False)
    with pytest.raises(ValueError, match="declared ports"):
        block.make_on(system=(0,), control=(1, 2))


def test_uncaptured_control_and_unknown_cost_do_not_disable_known_plain_direction() -> (
    None
):
    from dataclasses import replace

    full = measured_block()
    assert full.models is not None
    plain = replace(
        full,
        snapshots=(*full.snapshots[:2], None, None),
        models=(full.models[0], None, None, None),
    )
    assert plain.requirements() == full.requirements()
    assert plain.resource_model() == full.models[0]
    with pytest.raises(ValueError, match="No independent resource model"):
        plain.resource_model(adjoint=True)
    with pytest.raises(ValueError, match="not captured"):
        plain.requirements(controlled=True)
    with pytest.raises(ValueError, match="directional pair"):
        replace(full, snapshots=(*full.snapshots[:3], None))
