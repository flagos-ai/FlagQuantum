"""Entangling two-register U follows original controlled GQSP connections."""

from collections import Counter
from collections.abc import Mapping
from itertools import product
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
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.primitives.controlled_not import (
    cnot_with_controls_ir,
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
from tests.unit.test_component_control import add_h, add_phase, body
from tests.unit.test_generalized_qsp_stored_unitary import (
    _add_inverse_phase,
    source_gates,
    stored_unitary,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _add_cnot(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    outputs = builder.call_dict(
        "ccn",
        {
            "ctrl": np.array([control, values["ctrl"]], dtype=object),
            "target": values["target"],
        },
    )
    controls = outputs["ctrl"]
    assert isinstance(controls, np.ndarray)
    return controls[0], (controls[1], outputs["target"])


def entangling_unitary(
    repeats: int,
) -> tuple[dict[tuple[int | None, bool], ProgramRecord], Any]:
    """Store a shared H(a), CNOT(a,b), ZPow(b) layer on named QBit ports."""
    from qualtran import BloqBuilder
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, ZPowGate

    bit = quantum_bit_type()
    existing, _ = stored_unitary(1)
    definitions = (
        *(
            f
            for f in existing[None, False].functions
            if f.name not in {"main", "layer"}
        ),
        *body(
            "cn",
            (Register("ctrl", bit), Register("target", bit)),
            CircuitIR(2, cnot_with_controls_ir(0, 1)),
        ),
        *body(
            "ccn",
            (Register("ctrl", bit, (2,)), Register("target", bit)),
            CircuitIR(3, cnot_with_controls_ir(1, 2, controls=(0,), values=(1,))),
        ),
    )
    signature = Signature([Register("a", bit), Register("b", bit)])
    layer = ComponentProgramBuilder(signature, definitions, name="entangle")
    a = layer.call_dict("h", {"q": layer.inputs["a"]})["q"]
    a, b = layer.call_tuple("cn", {"ctrl": a, "target": layer.inputs["b"]})
    b = layer.call_dict("p", {"q": b})["q"]
    native = ComponentProgramBuilder(
        signature, layer.finalize({"a": a, "b": b}).functions
    )
    bb, source = BloqBuilder.from_signature(SourceSignature.build(a=1, b=1))
    ports = native.inputs
    for _ in range(repeats):
        ports = native.call_dict("entangle", ports)
        source["a"] = bb.add(Hadamard(), q=source["a"])
        source["a"], source["b"] = bb.add(CNOT(), ctrl=source["a"], target=source["b"])
        source["b"] = bb.add(ZPowGate(0.25), q=source["b"])
    forward = native.finalize(ports)
    inverse = adjoint_component_hierarchy(forward, {"h": "h", "cn": "cn", "p": "pi"})
    programs: dict[tuple[int | None, bool], ProgramRecord] = {
        (None, False): forward,
        (None, True): inverse,
    }
    for adjoint, record in ((False, forward), (True, inverse)):
        for value in (0, 1):
            programs[value, adjoint] = control_component_via_predicate(
                record,
                {
                    "h": add_h,
                    "cn": _add_cnot,
                    "pi" if adjoint else "p": (
                        _add_inverse_phase if adjoint else add_phase
                    ),
                },
                (Register("ctrl", bit),),
                (ControlRegister(1, (value,)),),
            )
    return programs, bb.finalize(**source)


def emitter(programs: dict[tuple[int | None, bool], ProgramRecord]) -> UnitaryEmitter:
    def emit(
        signal: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        value: int | None,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        data = (ports["a"], ports["b"])
        inputs = data if value is None else ((signal,), *data)
        return bind_register_program(
            programs[value, adjoint], inputs, clean=work
        ).circuit.instructions

    return emit


@pytest.mark.parametrize("repeats", [1, 2])
@pytest.mark.parametrize("value", [None, 0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_entangling_control_and_adjoint_full_source_leaves(
    repeats: int, value: int | None, adjoint: bool
) -> None:
    from qualtran import CtrlSpec

    from tests.unit.register_source_graph import SourcePorts

    programs, source = entangling_unitary(repeats)
    if adjoint:
        source = source.adjoint()
    ports: SourcePorts = {"a": (4,), "b": (1,)}
    if value is not None:
        source = source.controlled(CtrlSpec(cvs=value))
        ports["ctrl"] = (3,)
    gates = emitter(programs)(3, {"a": (4,), "b": (1,)}, (0,), value, adjoint)
    assert gates == source_gates(source, ports, repeats)
    counts = Counter(g.name for g in gates)
    assert counts["ccx"] == (repeats if value is not None else 0)
    assert counts["measure"] == counts["reset"] == (repeats if value is not None else 0)


@pytest.mark.parametrize("repeats", [1, 2])
@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("negative_power", [0, 1, 6])
def test_entangling_gqsp_full_leaf_order_amplitudes_all_branches_and_work_zero(
    repeats: int, length: int, negative_power: int
) -> None:
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source

    from flagquantum.simulation.numerics.qsp_sampling import random_qsp_polynomial

    programs, u = entangling_unitary(repeats)
    p = tuple(random_qsp_polynomial(length, random_state=np.random.RandomState(17)))
    source = Source.from_qsp_polynomial(u, p, negative_power=negative_power)
    native = GeneralizedQSP.from_qsp_polynomial(
        (("a", 1), ("b", 1)), emitter(programs), p, negative_power=negative_power
    )
    gates = native.emit(3, (4, 1), (0,))
    assert gates == source_gates(
        source, {"signal": (3,), "a": (4,), "b": (1,)}, repeats
    )
    measurements = repeats * (length - 1)
    counts = Counter(g.name for g in gates)
    assert counts["measure"] == counts["reset"] == counts["ccx"] == measurements
    assert_costs(source, length, negative_power, repeats)
    reference = cirq.unitary(source)
    order = (3, 4, 1, 0, 2)  # signal,a,b,work,idle; target layout is a then b
    coherent = np.array([complex(i - 2, (i * i) % 7 - 3) for i in range(8)])
    coherent /= np.linalg.norm(coherent)
    for initial in (*np.eye(8, dtype=complex), coherent):
        logical = np.kron(initial, [1, 0, 0, 0]).reshape((2,) * 5)
        state = torch.tensor(logical.transpose(tuple(np.argsort(order))).flatten())
        for history in product((0, 1), repeat=measurements):
            actual = (
                forced_branch(state, gates, history)
                .numpy()
                .reshape((2,) * 5)
                .transpose(order)
                .reshape(8, 4)
            )
            np.testing.assert_allclose(
                actual[:, 0],
                (reference @ initial) * 2 ** (-measurements / 2),
                atol=1e-13,
                rtol=0,
            )
            np.testing.assert_array_equal(actual[:, 1:], 0)
        circuit = DynamicCircuit(5, inputs=state, dtype=torch.complex128)
        circuit.append(native.make_on(signal=3, target=(4, 1), workspace=(0,)))
        assert circuit.to_ir().instructions == gates
        actual = (
            run_dynamic(circuit, shots=1, seed=23)
            .final_states[0]
            .detach()
            .numpy()
            .reshape((2,) * 5)
            .transpose(order)
            .reshape(8, 4)
        )
        np.testing.assert_allclose(
            actual[:, 0], reference @ initial, atol=2e-13, rtol=0
        )
        np.testing.assert_array_equal(actual[:, 1:], 0)


def assert_costs(
    source: Any,
    length: int,
    negative_power: int,
    repeats: int,
    *,
    system_width: int = 2,
) -> None:
    """Use existing primitive query identities; never infer model from emitted IR."""
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
        FixedTCall,
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

    def price(token: Any) -> GateCost:
        cost = query_qec_gate_cost(token)
        return GateCost(**{field: int(getattr(cost, field)) for field in fields})

    controlled = (
        GateCost()
        .plus(CONTROLLED_HADAMARD_COST.gates, repeats)
        .plus(price(FixedTCall("toffoli")), repeats)
    )
    positive = controlled.plus(price(ControlledZPowerCall(0.25)), repeats)
    backward = controlled.plus(price(ControlledZPowerCall(-0.25)), repeats)
    bare = GateCost(clifford=2 * repeats).plus(
        price(RotationCall("zpow", -0.25)), repeats
    )
    model = source_generalized_qsp_cost(
        length,
        system_width,
        negative_power=negative_power,
        signal_rotation=ComponentCost("SU2", 1, GateCost(rotation=3)),
        zero_control_unitary=ComponentCost(
            "zero-U", system_width + 2, GateCost(clifford=2).plus(positive, 1)
        ),
        unitary_adjoint=ComponentCost("U-adjoint", system_width, bare),
        controlled_adjoint=ComponentCost(
            "controlled-adjoint", system_width + 2, backward
        ),
    )
    actual = estimate_component_resources(model)
    expected = get_cost_value(source, QECGatesCost())
    for field in fields:
        assert getattr(actual.gates, field) == getattr(expected, field)
    assert actual.peak_logical_qubits == system_width + 2


@pytest.mark.parametrize("repeats", [1, 2])
def test_entangling_control_adjoint_commute_and_expansion_replaces_stored_graph(
    repeats: int,
) -> None:
    programs, u = entangling_unitary(repeats)
    reversed_control = adjoint_component_hierarchy(
        programs[1, False], {"ch": "ch", "cp": "cpi", "ccn": "ccn"}
    )
    assert (
        bind_register_program(reversed_control, ((3,), (4,), (1,)), clean=(0,)).circuit
        == bind_register_program(
            programs[1, True], ((3,), (4,), (1,)), clean=(0,)
        ).circuit
    )
    expanded = {
        key: expand_component_calls(record)[0] for key, record in programs.items()
    }
    for key in programs:
        assert emitter(programs)(3, {"a": (4,), "b": (1,)}, (0,), *key) == emitter(
            expanded
        )(3, {"a": (4,), "b": (1,)}, (0,), *key)
    # One actual layer creates an entangled state from |00>, so the fixture
    # cannot reduce to independent single-register phase rotations.
    if repeats == 1:
        qubits = cirq.LineQubit.range(2)
        circuit = u.to_cirq_circuit(
            cirq_quregs={"a": np.array([qubits[0]]), "b": np.array([qubits[1]])}
        )
        state = circuit.unitary(qubit_order=qubits)[:, 0].reshape(2, 2)
        assert abs(np.linalg.det(state)) > 0.49
    with pytest.raises(ValueError, match="workspace"):
        emitter(programs)(3, {"a": (4,), "b": (1,)}, (), 1, True)
