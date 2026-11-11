"""Actual ZPow control leaves and every GQSP measurement history match source."""

from collections import Counter
from collections.abc import Mapping
from itertools import product
from math import pi
from typing import Any

import cirq
import numpy as np
import pytest
import torch

from flagquantum.algorithms.primitives.generalized_qsp import GeneralizedQSP
from flagquantum.compiler.registers.phase.su2_rotation import su2_rotation_ir
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    z_power_with_control_ir,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _source_gates(source: Any, ports: SourcePorts) -> tuple[Instruction, ...]:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.basic_gates.rotation import ZPowGate
    from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
    from qualtran.bloqs.mcmt import And

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
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
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        if isinstance(bloq, ZPowGate):
            return (
                Instruction(
                    "phase", inputs["q"], params={"theta": pi * float(bloq.exponent)}
                ),
            ), inputs
        if isinstance(bloq, And):
            outputs = {"ctrl": inputs["ctrl"]}
            if not bloq.uncompute:
                outputs["target"] = (0,)
            return (
                reference_arithmetic_gates(
                    bloq, {**inputs, "target": (0,)}, (), {0: 7}, cirq_style=True
                ),
                outputs,
            )
        return None

    gates, returned = walk_source_graph(source, ports, ports, leaf)
    assert returned == ports
    return gates


def _unitary(
    signal: int,
    ports: Mapping[str, tuple[int, ...]],
    work: tuple[int, ...],
    value: int | None,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    (target,) = ports["q"]
    if value is None:
        return z_power_with_control_ir(target, 0.29, adjoint=adjoint)
    (clean,) = work
    return z_power_with_control_ir(
        target,
        0.29,
        control=signal,
        value=value,
        work=clean,
        classical_bit=7,
        adjoint=adjoint,
    )


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_literal_z_power_control_leaf_stream(value: int, adjoint: bool) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates.rotation import ZPowGate

    source = ZPowGate(-0.29 if adjoint else 0.29).controlled(
        ctrl_spec=CtrlSpec(cvs=value)
    )
    ports: SourcePorts = (
        {source.ctrl_reg_names[0]: (1,), "q": (3,)}
        if hasattr(source, "ctrl_reg_names")
        else {"q": (1, 3)}
    )
    gates = z_power_with_control_ir(
        3, 0.29, control=1, value=value, work=0, classical_bit=7, adjoint=adjoint
    )
    expected = _source_gates(source, ports)
    assert gates == expected
    assert Counter(g.name for g in gates) == Counter(g.name for g in expected)
    assert len(gates) == 18 + 2 * (value == 0)


@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("negative_power", [0, 1, 6])
def test_full_gqsp_source_u_leaf_stream_all_histories_complex_amplitudes_and_work_zero(
    length: int, negative_power: int
) -> None:
    from qualtran.bloqs.basic_gates.rotation import ZPowGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.simulation.numerics.qsp_sampling import random_qsp_polynomial

    p = tuple(random_qsp_polynomial(length, random_state=np.random.RandomState(17)))
    source = Source.from_qsp_polynomial(
        ZPowGate(0.29), p, negative_power=negative_power
    )
    native = GeneralizedQSP.from_qsp_polynomial(
        (("q", 1),), _unitary, p, negative_power=negative_power
    )
    gates = native.emit(1, (3,), (0,))
    # The original Bloq decomposition imports its Cirq control nodes back
    # through the source converter, reaching ControlledViaAnd/CZPow/And.
    # Dense Cirq unitary below is a separate semantic oracle, not a leaf trace.
    expected = _source_gates(source, {"signal": (1,), "q": (3,)})
    assert gates == expected
    assert Counter(g.name for g in gates) == Counter(g.name for g in expected)
    measurements = length - 1
    counts = Counter(g.name for g in gates)
    assert counts["measure"] == counts["reset"] == measurements
    assert counts["t"] + counts["tdg"] == 4 * measurements
    from flagquantum.algorithms.primitives.generalized_qsp_model import (
        source_generalized_qsp_cost,
    )
    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
    from flagquantum.compiler._component_trace.resource_generalization import (
        ControlledZPowerCall,
        PredicateControlledCall,
        RotationCall,
    )
    from flagquantum.compiler._component_trace.resource_model import (
        ComponentCost,
        GateCost,
        estimate_component_resources,
    )

    def priced(name: str, width: int, token: Any) -> ComponentCost:
        cost = query_qec_gate_cost(token)
        return ComponentCost(
            name,
            width,
            GateCost(
                **{
                    field: int(getattr(cost, field))
                    for field in (
                        "t",
                        "toffoli",
                        "cswap",
                        "and_bloq",
                        "clifford",
                        "rotation",
                        "measurement",
                    )
                }
            ),
        )

    model = source_generalized_qsp_cost(
        length,
        1,
        negative_power=negative_power,
        signal_rotation=ComponentCost("SU2", 1, GateCost(rotation=3)),
        zero_control_unitary=priced(
            "zero-U", 3, PredicateControlledCall(ControlledZPowerCall(0.29), (0,))
        ),
        unitary_adjoint=priced("U-adjoint", 1, RotationCall("zpow", -0.29)),
        controlled_adjoint=priced("controlled-adjoint", 3, ControlledZPowerCall(-0.29)),
    )
    estimated = estimate_component_resources(model)
    source_cost = get_cost_value(source, QECGatesCost())
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(estimated.gates, category) == getattr(source_cost, category)
    assert estimated.peak_logical_qubits == 3
    assert estimated.gates.and_bloq == estimated.gates.measurement == measurements
    reference = cirq.unitary(source)
    order = (1, 3, 0, 2)  # signal,target,work,idle -> scrambled physical wires
    rng = np.random.RandomState(3)
    coherent = rng.normal(size=4) + 1j * rng.normal(size=4)
    coherent /= np.linalg.norm(coherent)
    for initial in (*np.eye(4, dtype=complex), coherent):
        logical = np.kron(initial, [1, 0, 0, 0]).reshape((2,) * 4)
        state = torch.tensor(logical.transpose(tuple(np.argsort(order))).flatten())
        # Both outcomes per And erase, including absolute branch scale and phase.
        for history in product((0, 1), repeat=measurements):
            actual = (
                forced_branch(state, gates, history)
                .numpy()
                .reshape((2,) * 4)
                .transpose(order)
                .reshape(4, 4)
            )
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(
                actual[:, 0],
                (reference @ initial) * 2 ** (-measurements / 2),
                atol=4e-14,
                rtol=0,
            )
            np.testing.assert_array_equal(actual[:, 1:], 0)
        circuit = DynamicCircuit(4, inputs=state, dtype=torch.complex128)
        circuit.append(native.make_on(signal=1, target=(3,), workspace=(0,)))
        assert circuit.to_ir().instructions == gates
        result = run_dynamic(circuit, shots=1, seed=23)
        actual = (
            result.final_states[0]
            .detach()
            .numpy()
            .reshape((2,) * 4)
            .transpose(order)
            .reshape(4, 4)
        )
        np.testing.assert_allclose(
            actual[:, 0], reference @ initial, atol=5e-14, rtol=0
        )
        np.testing.assert_array_equal(actual[:, 1:], 0)


def test_original_matrix_gate_has_no_leaf_decomposition() -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate

    with pytest.raises(DecomposeNotImplementedError):
        MatrixGate(bitsize=1, matrix=np.eye(2)).decompose_bloq()


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"adjoint": 1}, "boolean"),
        ({"value": True}, "zero or one"),
        ({"value": 0}, "Bare"),
        ({"work": 0}, "Bare"),
        ({"control": 1}, "clean work"),
        ({"control": 1, "work": 0, "classical_bit": -1}, "nonnegative"),
        ({"control": 3, "work": 0, "classical_bit": 7}, "distinct"),
    ],
)
def test_control_boundary_validation(kwargs: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        z_power_with_control_ir(3, 0.29, **kwargs)
