"""Preserve repeated source control roots and isolate their builder failure."""

from cmath import exp
from collections import Counter
from itertools import product
from math import pi

import pytest
import torch

from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.indexed_component_control import (
    conjoined_indexed_component_ir,
)
from flagquantum.compiler.registers.indexed_component_control_resource_model import (
    conjoined_indexed_component_cost,
)
from flagquantum.compiler.registers.phase_family import phase_family_ir
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_indexed_component_cost_profiles import _assert_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = pytest.mark.unit
PATTERNS = (
    tuple(product((0, 1), repeat=2))
    + tuple(product((0, 1), repeat=3))
    + ((0, 1, 0, 1), (1, 0, 1, 0))
)


def _native(values, phase, adjoint):
    n = len(values)
    work = tuple(range(n + 2, 2 * n + 2))
    bits = tuple(range(40, 40 + n))

    def child(index, predicate):
        if phase:
            return phase_family_ir(
                "s" if index == 0 else "t",
                n + 1,
                controls=(predicate,),
                values=(1,),
                adjoint=adjoint,
            )
        return (Instruction("cx" if index == 0 else "cz", (predicate, n + 1)),)

    gates = conjoined_indexed_component_ir(
        ((n,),),
        (2,),
        child,
        tuple(range(n)),
        values,
        workspace=work,
        classical_bits=bits,
        adjoint=adjoint,
    )
    return gates, work, bits


def _source(values, phase):
    import numpy as np
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import SGate, TGate, XGate, ZGate
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq

    source = ApplyLthBloq(np.array([SGate(), TGate()] if phase else [XGate(), ZGate()]))
    for value in reversed(values):
        source = source.controlled(CtrlSpec(cvs=value))
    assert source.cvs == values
    return source


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("values", PATTERNS)
@pytest.mark.parametrize("phase,adjoint", tuple(product((False, True), repeat=2)))
def test_original_repeated_root_gate_dependencies_and_profiles(values, phase, adjoint):
    source = _source(values, phase)
    if adjoint:
        source = source.adjoint()
    n = len(values)
    gates, work, bits = _native(values, phase, adjoint)
    original = reference_arithmetic_gates(
        source.decompose_bloq().flatten(),
        {"control": tuple(range(n)), "selection": (n,), "q": (n + 1,)},
        work,
        dict(zip(work, bits, strict=True)),
        cirq_style=True,
        preserve_components=True,
        keep_pauli_leaves=True,
        keep_exchange_leaves=True,
    )
    assert_source_dependencies(gates, original)
    counts = Counter(g.name for g in gates)
    assert counts["t"] + counts["tdg"] == 4 * n
    assert counts["measure"] == n
    children = (
        (
            ComponentCost("CS", 2, GateCost(rotation=1)),
            ComponentCost("CT", 2, GateCost(rotation=1)),
        )
        if phase
        else (
            ComponentCost("CX", 2, GateCost(clifford=1)),
            ComponentCost("CZ", 2, GateCost(clifford=1)),
        )
    )
    for legacy in (False, True):
        model = conjoined_indexed_component_cost(
            (1,),
            (2,),
            children,
            values,
            target_width=1,
            adjoint=adjoint,
            legacy_shims=legacy,
        )
        _assert_model(model, source, legacy)


@pytest.mark.parametrize("values", PATTERNS[:12])
@pytest.mark.parametrize("phase,adjoint", tuple(product((False, True), repeat=2)))
def test_complete_complex_branches_with_both_control_polarities(values, phase, adjoint):
    n = len(values)
    data = n + 2
    gates, work, _ = _native(values, phase, adjoint)
    initial = torch.zeros(1 << (data + len(work)), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    pattern = sum(bit << (n - i - 1) for i, bit in enumerate(values))
    for word in range(1 << data):
        amp = complex(word + 1, 3 - word) / (1 << data)
        initial[word << len(work)] = amp
        active = word >> 2 == pattern
        address = (word >> 1) & 1
        if phase:
            theta = (pi / 2 if address == 0 else pi / 4) * (-1 if adjoint else 1)
            factor = exp(1j * theta) if active and word & 1 else 1
            output = word
        else:
            output = word ^ int(active and address == 0)
            factor = -1 if active and address == 1 and word & 1 else 1
        expected[output << len(work)] = amp * factor
    for branch in product((0, 1), repeat=n):
        torch.testing.assert_close(
            forced_branch(initial, gates, branch),
            expected * 2 ** (-n / 2),
            atol=4e-12,
            rtol=0,
        )


@pytest.mark.parametrize("values", [(0, 0), (0, 1), (1, 0), (1, 1), (0, 1, 0)])
def test_all_phase_round_trip_measurement_pairs(values):
    n = len(values)
    forward, work, _ = _native(values, True, False)
    reverse, _, _ = _native(values, True, True)
    state = torch.zeros(1 << (2 * n + 2), dtype=torch.complex128)
    state[:: 1 << len(work)] = torch.tensor(
        [complex(i + 1, 2 - i) for i in range(1 << (n + 2))], dtype=torch.complex128
    )
    state /= state.norm()
    for branch in product((0, 1), repeat=2 * n):
        torch.testing.assert_close(
            forced_branch(state, (*forward, *reverse), branch),
            state * 2 ** (-n),
            atol=4e-12,
            rtol=0,
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("cv", [0, 1])
def test_original_control_adder_failure_is_not_repaired(cv):
    from qualtran import BloqBuilder, CtrlSpec

    source = _source((1, 0), False)
    root, adder = source.get_ctrl_system(CtrlSpec(cvs=cv))
    assert root.cvs == (cv, 1, 0)
    assert len(root.decompose_bloq().bloq_instances) == 3
    bb, soqs = BloqBuilder.from_signature(source.signature, add_registers_allowed=True)
    ctrl = bb.add_register("extra", 1)
    with pytest.raises(TypeError, match="integer scalar arrays"):
        adder(bb, [ctrl], soqs)


@pytest.mark.parametrize(
    "change",
    [
        {"controls": (0,), "values": (1,)},
        {"values": (1, 2)},
        {"controls": (0, 2)},
        {"workspace": (4,)},
        {"classical_bits": (0, 0)},
        {"adjoint": 1},
    ],
)
def test_invalid_stack_refused_before_child(change):
    def refuse(index, predicate):
        pytest.fail("Invalid stack reached child emitter")

    kwargs = dict(
        axes=((2,),),
        extents=(2,),
        emit_controlled=refuse,
        controls=(0, 1),
        values=(1, 0),
        workspace=(4, 5),
        classical_bits=(0, 1),
    )
    kwargs.update(change)
    with pytest.raises(ValueError):
        conjoined_indexed_component_ir(**kwargs)


def test_stacked_append_example(capsys):
    from examples.algorithms.indexed_component_controls import main

    main(stacked=True)
    output = capsys.readouterr().out
    assert "Modern model Clifford count 14" in output
    assert "Legacy model Clifford count 22" in output
    assert "Emitted T/T-dagger instructions 12" in output
    assert "Model logical peak 8" in output
