"""Stored noncommuting U calls retain source control, adjoint and GQSP leaves."""

from collections import Counter
from collections.abc import Mapping
from itertools import product
from math import pi
from typing import Any

import cirq
import numpy as np
import pytest
import torch

from flagquantum.algorithms.primitives.generalized_qsp import (
    GeneralizedQSP,
    UnitaryEmitter,
)
from flagquantum.compiler.registers.component_adjoint import adjoint_component_hierarchy
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_control import (
    control_component_via_predicate,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.phase.su2_rotation import su2_rotation_ir
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    controlled_z_power_ir,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_component_control import add_h, add_phase, body, fixture_program
from tests.unit.test_register_constant_addition import reference_arithmetic_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def stored_unitary(
    repeats: int,
) -> tuple[dict[tuple[int | None, bool], ProgramRecord], Any]:
    """Build H then phase; its inverse must phase backwards before applying H."""
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import Hadamard, ZPowGate

    bit = quantum_bit_type()
    base, _ = fixture_program(1)
    definitions = (
        *base.functions[1:],
        *body(
            "pi",
            (Register("q", bit),),
            CircuitIR(1, (Instruction("phase", (0,), {"theta": -pi / 4}),)),
        ),
        *body(
            "cpi",
            (Register("q", bit, (2,)),),
            CircuitIR(3, controlled_z_power_ir(0, 1, -0.25, 2, 0)),
        ),
    )
    signature = Signature([Register("q", bit)])
    layer = ComponentProgramBuilder(signature, definitions, name="layer")
    q = layer.call_dict("h", layer.inputs)["q"]
    q = layer.call_dict("p", {"q": q})["q"]
    native = ComponentProgramBuilder(signature, layer.finalize({"q": q}).functions)
    bb, values = BloqBuilder.from_signature(Hadamard().signature)
    q = native.inputs["q"]
    for _ in range(repeats):
        q = native.call_dict("layer", {"q": q})["q"]
        values["q"] = bb.add(Hadamard(), q=values["q"])
        values["q"] = bb.add(ZPowGate(0.25), q=values["q"])
    forward = native.finalize({"q": q})
    inverse = adjoint_component_hierarchy(forward, {"h": "h", "p": "pi"})
    result: dict[tuple[int | None, bool], ProgramRecord] = {
        (None, False): forward,
        (None, True): inverse,
    }
    for adjoint, record in ((False, forward), (True, inverse)):
        for value in (0, 1):
            result[value, adjoint] = control_component_via_predicate(
                record,
                {
                    "h": add_h,
                    "pi" if adjoint else "p": (
                        _add_inverse_phase if adjoint else add_phase
                    ),
                },
                (Register("ctrl", bit),),
                (ControlRegister(1, (value,)),),
            )
    return result, bb.finalize(**values)


def _add_inverse_phase(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    output = builder.call_dict(
        "cpi", {"q": np.array([control, values["q"]], dtype=object)}
    )["q"]
    assert isinstance(output, np.ndarray)
    return output[0], (output[1],)


def source_gates(
    source: Any, ports: SourcePorts, slots: int
) -> tuple[Instruction, ...]:
    """Read original connections with the same declared atomic normalization."""
    from qualtran.bloqs.basic_gates import CHadamard, Hadamard, XGate, ZPowGate
    from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
    from qualtran.bloqs.mcmt import And

    erased = 0

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        nonlocal erased
        if isinstance(bloq, SU2RotationGate):
            return (
                su2_rotation_ir(
                    inputs["q"][0],
                    float(bloq.theta),
                    float(bloq.phi),
                    float(bloq.lambd),
                ),
                inputs,
            )
        if isinstance(bloq, (Hadamard, XGate)):
            return (
                Instruction("h" if isinstance(bloq, Hadamard) else "x", inputs["q"]),
            ), inputs
        if isinstance(bloq, CHadamard):
            return (Instruction("ch", (*inputs["ctrl"], *inputs["target"])),), inputs
        if isinstance(bloq, ZPowGate):
            return (
                Instruction("phase", inputs["q"], {"theta": pi * float(bloq.exponent)}),
            ), inputs
        if isinstance(bloq, And):
            outputs = {"ctrl": inputs["ctrl"]}
            if not bloq.uncompute:
                outputs["target"] = (0,)
            gates = reference_arithmetic_gates(
                bloq,
                {**inputs, "target": (0,)},
                (),
                {0: erased % slots},
                cirq_style=True,
            )
            erased += int(bloq.uncompute)
            return gates, outputs
        return None

    gates, returned = walk_source_graph(source, ports, ports, leaf)
    assert returned == ports
    return gates


def emitter(programs: dict[tuple[int | None, bool], ProgramRecord]) -> UnitaryEmitter:
    def emit(
        signal: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        value: int | None,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        inputs = (ports["q"],) if value is None else ((signal,), ports["q"])
        return bind_register_program(
            programs[value, adjoint], inputs, clean=work
        ).circuit.instructions

    return emit


@pytest.mark.parametrize("repeats", [1, 2])
@pytest.mark.parametrize("value", [None, 0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_stored_control_and_exact_adjoint_match_original_leaves(
    repeats: int, value: int | None, adjoint: bool
) -> None:
    from qualtran import CtrlSpec

    programs, source = stored_unitary(repeats)
    if adjoint:
        source = source.adjoint()
    ports: SourcePorts = {"q": (3,)}
    if value is not None:
        source = source.controlled(CtrlSpec(cvs=value))
        ports["ctrl"] = (1,)
    gates = emitter(programs)(1, {"q": (3,)}, (0,), value, adjoint)
    assert gates == source_gates(source, ports, repeats)


@pytest.mark.parametrize("repeats", [1, 2])
@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("negative_power", [0, 1, 6])
def test_complete_stored_u_gqsp_all_measurement_histories(
    repeats: int, length: int, negative_power: int
) -> None:
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source

    from flagquantum.simulation.numerics.qsp_sampling import random_qsp_polynomial

    programs, u = stored_unitary(repeats)
    p = tuple(random_qsp_polynomial(length, random_state=np.random.RandomState(17)))
    source = Source.from_qsp_polynomial(u, p, negative_power=negative_power)
    native = GeneralizedQSP.from_qsp_polynomial(
        (("q", 1),), emitter(programs), p, negative_power=negative_power
    )
    gates = native.emit(1, (3,), (0,))
    assert gates == source_gates(source, {"signal": (1,), "q": (3,)}, repeats)
    measurements = repeats * (length - 1)
    assert Counter(g.name for g in gates)["measure"] == measurements
    assert_costs(source, length, negative_power, repeats)
    reference = cirq.unitary(source)
    order = (1, 3, 0, 2)
    coherent = np.array([1 + 2j, -3 + 1j, 2 - 4j, 1 - 1j])
    coherent /= np.linalg.norm(coherent)
    for initial in (*np.eye(4, dtype=complex), coherent):
        logical = np.kron(initial, [1, 0, 0, 0]).reshape((2,) * 4)
        state = torch.tensor(logical.transpose(tuple(np.argsort(order))).flatten())
        for history in product((0, 1), repeat=measurements):
            actual = (
                forced_branch(state, gates, history)
                .numpy()
                .reshape((2,) * 4)
                .transpose(order)
                .reshape(4, 4)
            )
            np.testing.assert_allclose(
                actual[:, 0],
                (reference @ initial) * 2 ** (-measurements / 2),
                atol=8e-14,
                rtol=0,
            )
            np.testing.assert_array_equal(actual[:, 1:], 0)
        circuit = DynamicCircuit(4, inputs=state, dtype=torch.complex128)
        circuit.append(native.make_on(signal=1, target=(3,), workspace=(0,)))
        assert circuit.to_ir().instructions == gates
        actual = (
            run_dynamic(circuit, shots=1, seed=23)
            .final_states[0]
            .detach()
            .numpy()
            .reshape((2,) * 4)
            .transpose(order)
            .reshape(4, 4)
        )
        np.testing.assert_allclose(
            actual[:, 0], reference @ initial, atol=1e-13, rtol=0
        )
        np.testing.assert_array_equal(actual[:, 1:], 0)


def assert_costs(source: Any, length: int, negative_power: int, repeats: int) -> None:
    """Price actual child identities separately from physical primitive counts."""
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.algorithms.primitives.generalized_qsp_model import (
        source_generalized_qsp_cost,
    )
    from flagquantum.compiler._component_trace.compound_control_resource import (
        CONTROLLED_HADAMARD_COST,
    )
    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
    from flagquantum.compiler._component_trace.resource_generalization import (
        ControlledZPowerCall,
        RotationCall,
    )
    from flagquantum.compiler._component_trace.resource_model import (
        ComponentCost,
        GateCost,
        estimate_component_resources,
    )

    fields = (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    )

    def priced(token: Any) -> GateCost:
        cost = query_qec_gate_cost(token)
        return GateCost(**{field: int(getattr(cost, field)) for field in fields})

    controlled = (
        GateCost()
        .plus(CONTROLLED_HADAMARD_COST.gates, repeats)
        .plus(priced(ControlledZPowerCall(0.25)), repeats)
    )
    backward = (
        GateCost()
        .plus(CONTROLLED_HADAMARD_COST.gates, repeats)
        .plus(priced(ControlledZPowerCall(-0.25)), repeats)
    )
    bare = GateCost(clifford=repeats).plus(priced(RotationCall("zpow", -0.25)), repeats)
    model = source_generalized_qsp_cost(
        length,
        1,
        negative_power=negative_power,
        signal_rotation=ComponentCost("SU2", 1, GateCost(rotation=3)),
        zero_control_unitary=ComponentCost(
            "zero-U", 3, GateCost(clifford=2).plus(controlled, 1)
        ),
        unitary_adjoint=ComponentCost("U-adjoint", 1, bare),
        controlled_adjoint=ComponentCost("controlled-adjoint", 3, backward),
    )
    actual = estimate_component_resources(model)
    expected = get_cost_value(source, QECGatesCost())
    for field in fields:
        assert getattr(actual.gates, field) == getattr(expected, field)
    assert actual.peak_logical_qubits == 3


@pytest.mark.parametrize("repeats", [1, 2])
def test_control_then_exact_adjoint_equals_exact_adjoint_then_control_and_expansion(
    repeats: int,
) -> None:
    from flagquantum.compiler.registers.component_expansion import (
        expand_component_calls,
    )

    programs, _ = stored_unitary(repeats)
    reversed_control = adjoint_component_hierarchy(
        programs[1, False], {"ch": "ch", "cp": "cpi"}
    )

    def bound(record: ProgramRecord) -> tuple[Instruction, ...]:
        return bind_register_program(
            record, ((1,), (3,)), clean=(0,)
        ).circuit.instructions

    assert bound(reversed_control) == bound(programs[1, True])
    # Replace the stored hierarchy by its expanded Program behind the same emitter.
    expanded = {
        key: expand_component_calls(record)[0] for key, record in programs.items()
    }
    for key in programs:
        assert emitter(programs)(1, {"q": (3,)}, (0,), *key) == emitter(expanded)(
            1, {"q": (3,)}, (0,), *key
        )
    with pytest.raises(ValueError, match="workspace"):
        emitter(programs)(1, {"q": (3,)}, (), 1, True)
