"""Pinned phase-gradient gates, complex states, and measured cleanup branches."""

from collections import Counter
from itertools import product
from math import pi
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.phase_gradient import (
    phase_gradient_state_ir,
    phase_gradient_unitary_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def source_gradient_gates(
    source: Any,
    ports: SourcePorts,
    outputs: SourcePorts,
    *,
    work: int | None = None,
    bit: int = 0,
) -> tuple[tuple[Instruction, ...], tuple[Any, ...]]:
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.basic_gates.rotation import ZPowGate
    from qualtran.bloqs.bookkeeping import Allocate
    from qualtran.bloqs.mcmt import And

    phases: list[Any] = []

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, Allocate):
            assert not bloq.dirty and not inputs
            return (), {"reg": outputs["phase_grad"]}
        if isinstance(bloq, Hadamard):
            return (Instruction("h", inputs["q"]),), inputs
        if isinstance(bloq, ZPowGate):
            phases.append(bloq)
            return (
                Instruction(
                    "phase", inputs["q"], params={"theta": pi * float(bloq.exponent)}
                ),
            ), inputs
        if isinstance(bloq, And):
            assert work is not None
            out = {"ctrl": inputs["ctrl"]}
            if not bloq.uncompute:
                out["target"] = (work,)
            return (
                reference_arithmetic_gates(
                    bloq,
                    {**inputs, "target": (work,)},
                    (),
                    {work: bit},
                    cirq_style=True,
                ),
                out,
            )
        return None

    gates, returned = walk_source_graph(source, ports, outputs, leaf)
    assert returned == outputs
    return gates, tuple(phases)


@pytest.mark.parametrize("n,exponent", [(1, 1.0), (3, -0.75), (4, 1.0), (8, 0.625)])
@pytest.mark.parametrize("controlled", [False, True])
def test_pinned_gradient_gates_and_resource_model(
    n: int, exponent: float, controlled: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    gradient = tuple(reversed(range(n)))
    ports = {"phase_grad": gradient}
    control, work = (n, n + 1) if controlled else (None, None)
    if control is not None:
        ports["ctrl"] = (control,)
    bit = 29 if controlled else None
    source = PhaseGradientUnitary(
        n, exponent=exponent, is_controlled=controlled, eps=2e-8
    )
    reference, phases = source_gradient_gates(source, ports, ports, work=work, bit=29)
    gates = phase_gradient_unitary_ir(
        gradient, exponent=exponent, control=control, work=work, classical_bit=bit
    )
    assert_source(gates, reference)
    assert tuple(
        (g.wires, g.params["theta"]) for g in gates if g.name == "phase"
    ) == tuple((g.wires, g.params["theta"]) for g in reference if g.name == "phase")
    assert sorted(float(leaf.exponent) for leaf in phases) == sorted(
        exponent / 2**i for i in range(n)
    )
    assert all(leaf.eps == pytest.approx(source.eps / n) for leaf in phases)
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert counts["phase"] == n
    assert counts["t"] + counts["tdg"] == 4 * model.and_bloq == 4 * n * controlled
    assert counts["measure"] == counts["reset"] == model.measurement == n * controlled
    assert counts["cx"] == 6 * n * controlled
    assert get_cost_value(source, QubitCount()) == n + 2 * controlled
    # Literal phases stay as radians. The source's cost categorization is separate.
    if (n, exponent) == (4, 1.0):
        assert (model.t, model.rotation) == (1, 1)
    if (n, exponent) == (8, 0.625):
        assert (model.t, model.rotation) == (0, 8)


@pytest.mark.parametrize("n,exponent", [(1, 0.625), (3, -0.75), (4, 1.0), (8, -1.0)])
@pytest.mark.parametrize("controlled", [False, True])
def test_gradient_complex_phases_and_measured_branches(
    n: int, exponent: float, controlled: bool
) -> None:
    gradient = tuple(range(n))
    control, work, bit = (n, n + 1, 17) if controlled else (None, None, None)
    width = n + 1 + 2 * controlled
    addresses = torch.arange(1 << width, dtype=torch.int64)
    enabled = ((addresses >> 2) & 1) if controlled else torch.ones_like(addresses)
    word = addresses >> (1 + 2 * controlled)
    angles = (2 * pi * exponent / 2**n) * word.to(torch.float64) * enabled
    weights = addresses.to(torch.float64) + 1
    state = torch.complex(weights, weights * (weights % 5 - 2))
    if controlled:
        state[(addresses & 2) != 0] = 0
    state /= state.norm()
    expected = state * torch.exp(1j * angles)
    kwargs = dict(control=control, work=work, classical_bit=bit)
    gates = phase_gradient_unitary_ir(gradient, exponent=exponent, **kwargs)
    inverse = phase_gradient_unitary_ir(gradient, exponent=-exponent, **kwargs)
    events = n * controlled
    histories = (
        product((0, 1), repeat=events)
        if events <= 4
        else ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events)))
    )
    scale = 2 ** (-events / 2)
    for history in histories:
        torch.testing.assert_close(
            forced_branch(state, gates, history), expected * scale, atol=2e-12, rtol=0
        )
        torch.testing.assert_close(
            forced_branch(expected, inverse, history), state * scale, atol=2e-12, rtol=0
        )


@pytest.mark.parametrize("exponent", [-1.0, 1.0, -0.625])
def test_source_state_preparation_and_unpreparation(exponent: float) -> None:
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    n = 5
    wires = tuple(range(n))
    source = PhaseGradientState(n, exponent=exponent)
    reference, _ = source_gradient_gates(source, {}, {"phase_grad": wires})
    gates = phase_gradient_state_ir(wires, exponent=exponent)
    assert_source(gates, reference)
    init = torch.zeros(1 << n, dtype=torch.complex128)
    init[0] = 1
    expected = torch.as_tensor(np.asarray(source.tensor_contract()).copy()).reshape(-1)
    torch.testing.assert_close(
        forced_branch(init, gates, ()), expected, atol=2e-12, rtol=0
    )
    inverse = phase_gradient_state_ir(wires, exponent=exponent, inverse=True)
    torch.testing.assert_close(
        forced_branch(expected, inverse, ()), init, atol=2e-12, rtol=0
    )
    assert get_cost_value(source, QubitCount()) == n
    assert get_cost_value(source, QECGatesCost()).measurement == 0


def test_thc_legacy_positive_gradient_stream_and_validation() -> None:
    wires = (4, 1, 7)
    hadamards = tuple(Instruction("h", (wire,)) for wire in wires)
    for inverse in (False, True):
        phases = tuple(
            Instruction(
                "phase",
                (wire,),
                params={"theta": (-1 if inverse else 1) * pi / (1 << i)},
            )
            for i, wire in enumerate(wires)
        )
        assert phase_gradient_state_ir(wires, inverse=inverse) == (
            (*phases, *hadamards) if inverse else (*hadamards, *phases)
        )
    with pytest.raises(ValueError, match="finite"):
        phase_gradient_unitary_ir(wires, exponent=float("nan"))
    with pytest.raises(ValueError, match="no workspace"):
        phase_gradient_unitary_ir(wires, work=0)
    with pytest.raises(ValueError, match="requires work"):
        phase_gradient_unitary_ir(wires, control=0)
    with pytest.raises(ValueError, match="distinct"):
        phase_gradient_unitary_ir(wires, control=0, work=wires[0], classical_bit=0)


def test_append_controlled_phase_gradient_example() -> None:
    from examples.algorithms.register_phase_gradient import main

    main()
