"""Verify the locked programmed ancilla fixtures and adaptive branches."""

from cmath import exp
from dataclasses import fields
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import pi, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sympy import Symbol, symbols
from sympy import pi as symbolic_pi

from examples.algorithms.programmed_phase_catalog import (
    ProgrammedAncillaExample,
    ProgrammedRotationExample,
)
from flagquantum.compiler._component_trace.call_graph import SymbolAllocator
from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    MeasurementCall,
    ResourceCall,
    RotationCall,
)
from flagquantum.compiler._component_trace.symbolic_math import (
    SymbolicInt,
    SymbolicReal,
)
from flagquantum.compiler.registers.phase.programmed_phase_resource_model import (
    ProgrammedAncillaCall,
    ProgrammedRotationModel,
)
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.rotations import programmable_ancilla_rotation

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(programmable_ancilla_rotation.__file__).parent
    for name, digest in (
        ("__init__.py", "e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8"),
        (
            "programmable_ancilla_rotation.py",
            "89561f7a56e1567210a14ee6af90ee65505c3172",
        ),
        (
            "programmable_ancilla_rotation_test.py",
            "7882ccc1522dc4abe2568e1e2d636d31f837faea",
        ),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return programmable_ancilla_rotation


def source_call_key(child: Any) -> ResourceCall | ProgrammedAncillaCall:
    name = type(child).__name__
    if name == "ZPowProgrammedAncilla":
        return ProgrammedAncillaCall(child.exponent, child.eps)
    if name == "ZPowGate":
        return RotationCall("zpow", child.exponent, child.eps)
    if name == "Allocate":
        assert not child.dirty
        return BookkeepingCall("allocate", quantum_bit_type())
    if name == "Hadamard":
        return CliffordCall("h", 1)
    if name == "CNOT":
        return CliffordCall("cx", 2)
    if name == "XGate":
        return CliffordCall("x", 1)
    assert name == "MeasureZ"
    return MeasurementCall("z")


@pytest.mark.parametrize("symbolic", [False, True])
def test_pinned_ancilla_factory(symbolic: bool, source_module: Any) -> None:
    factory = (
        "_zpow_programmed_ancilla_symb" if symbolic else "_zpow_programmed_ancilla"
    )
    source = getattr(source_module, factory).make()
    op = ProgrammedAncillaExample(
        ProgrammedAncillaCall(Symbol("t") if symbolic else pi / 4)
    )
    assert op.model.direct_calls() == {
        source_call_key(k): v for k, v in source.bloq_counts().items()
    }
    (register,) = op.signature
    (pinned,) = source.signature
    assert (register.name, register.dtype.num_qubits, register.side.name) == (
        pinned.name,
        pinned.bitsize,
        pinned.side.name,
    )
    if symbolic:
        with pytest.raises(ValueError, match="Bind the programmed state"):
            DynamicCircuit(1).append(op)


@pytest.mark.parametrize("index", range(5))
def test_original_ancilla_tensor_grid(index: int, source_module: Any) -> None:
    exponent = float(np.random.default_rng(42).random(size=5)[index])
    source = source_module.ZPowProgrammedAncilla(exponent)
    op = ProgrammedAncillaExample(ProgrammedAncillaCall(exponent))
    circuit = DynamicCircuit(1)
    circuit.append(op)
    first, phase = circuit.to_ir().instructions
    assert (first.name, first.wires, phase.name, phase.wires) == (
        "h",
        (0,),
        "phase",
        (0,),
    )
    assert phase.params["theta"] == pi * exponent
    prepared = forced_branch(
        torch.tensor([1, 0], dtype=torch.complex128), (first, phase), ()
    )
    expected = torch.as_tensor(np.asarray(source.tensor_contract()).copy())
    torch.testing.assert_close(prepared, expected, rtol=0, atol=2e-12)


def assert_model_matches(model: ProgrammedRotationModel, reference: Any) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma

    assert model.direct_calls() == {
        source_call_key(k): v for k, v in reference.bloq_counts().items()
    }
    formula = model.resource_formula()
    expected = get_cost_value(reference, QECGatesCost())
    assert formula.logical_qubits == get_cost_value(reference, QubitCount())
    for f in fields(QECGateCost):
        assert getattr(formula, f.name) == getattr(expected, f.name)
    assert model.synthesis_t_count() == t_counts_from_sigma(reference.call_graph()[1])


@pytest.mark.parametrize("kind", ["concrete", "symbolic_angle", "symbolic_rounds"])
def test_pinned_rotation_factory(kind: str, source_module: Any) -> None:
    phi, epsilon, n = symbols(r"\phi \epsilon n")
    if kind == "concrete":
        factory, model = "_zpow_using_programmed_ancilla", ProgrammedRotationModel(
            pi / 4
        )
    elif kind == "symbolic_angle":
        factory = "_zpow_using_programmed_ancilla_symb"
        model = ProgrammedRotationModel(phi / symbolic_pi, epsilon, 3)
    else:
        factory = "_zpow_using_programmed_ancilla_symb_rounds"
        model = ProgrammedRotationModel(phi / symbolic_pi, rounds=n)
    source = getattr(source_module, factory).make()
    assert_model_matches(model, source)
    op = ProgrammedRotationExample(model)
    (register,) = op.signature
    (pinned,) = source.signature
    assert (register.name, register.dtype.num_qubits, register.side.name) == (
        pinned.name,
        pinned.bitsize,
        pinned.side.name,
    )
    if kind != "concrete":
        with pytest.raises(ValueError, match="Bind the programmed rotation"):
            DynamicCircuit(2).append(op)


def test_symbolic_failure_probability_estimate(source_module: Any) -> None:
    theta, epsilon, p = symbols(r"theta \epsilon p")
    model = ProgrammedRotationModel.from_failure_probability(
        theta, max_fail_probability=p, precision=epsilon
    )
    reference = source_module.ZPowUsingProgrammedAncilla.from_failure_probability(
        theta, max_fail_probability=p, eps=epsilon
    )
    assert model.rounds == reference.n_rounds
    assert_model_matches(model, reference)


@pytest.mark.parametrize("correct", [False, True])
def test_composed_symbolic_rounds_preserve_fresh_source_angles(
    source_module: Any, correct: bool
) -> None:
    from qualtran.resource_counting import SympySymbolAllocator

    theta, epsilon, n = symbols("theta epsilon n")
    model = ProgrammedRotationModel(theta, epsilon, n, correct)
    source = source_module.ZPowUsingProgrammedAncilla(
        theta, eps=epsilon, n_rounds=n, apply_final_correction=correct
    )
    allocator, original_allocator = SymbolAllocator(), SympySymbolAllocator()
    # The containing graph may already have allocated an angle for another node.
    assert allocator.new_symbol(r"\phi") == original_allocator.new_symbol(r"\phi")
    angles: list[SymbolicReal] = []
    for _ in range(3):
        calls = model.direct_calls(allocator=allocator)
        expected = {
            source_call_key(k): v
            for k, v in source.build_call_graph(original_allocator).items()
        }
        assert calls == expected
        angles.extend(k.exponent for k in calls if isinstance(k, ProgrammedAncillaCall))
    assert len(set(angles)) == 3
    assert model.direct_calls() == model.direct_calls()


def test_symbol_allocator_reaches_terminal_expansion(source_module: Any) -> None:
    from qualtran.bloqs.rotations.programmable_ancilla_rotation import (
        ZPowProgrammedAncilla,
    )
    from qualtran.resource_counting import SympySymbolAllocator

    n = Symbol("n")
    model = ProgrammedRotationModel(0.375, rounds=n)
    source = source_module.ZPowUsingProgrammedAncilla(0.375, n_rounds=n)
    allocator, original_allocator = SymbolAllocator(), SympySymbolAllocator()
    for _ in range(2):
        expected: dict[ResourceCall, SymbolicInt] = {}
        for call, count in source.build_call_graph(original_allocator).items():
            children = (
                call.bloq_counts()
                if isinstance(call, ZPowProgrammedAncilla)
                else {call: 1}
            )
            for child, repetitions in children.items():
                key = source_call_key(child)
                assert not isinstance(key, ProgrammedAncillaCall)
                expected[key] = expected.get(key, 0) + repetitions * count
        assert model.terminal_calls(allocator=allocator) == expected


@pytest.mark.parametrize("rounds", [0, 1, 2, 3])
@pytest.mark.parametrize("correct", [False, True])
def test_concrete_rounds_and_all_raw_branches(
    rounds: int, correct: bool, source_module: Any
) -> None:
    theta, epsilon = pi / 4, 5e-4
    model = ProgrammedRotationModel(theta, epsilon, rounds, correct)
    reference = source_module.ZPowUsingProgrammedAncilla(
        theta, eps=epsilon, n_rounds=rounds, apply_final_correction=correct
    )
    assert_model_matches(model, reference)
    initial = torch.tensor([1, 2j, 0, 0, -0.5j, 1 + 2j, 0, 0], dtype=torch.complex128)
    initial /= initial.norm()
    circuit = DynamicCircuit(3)
    circuit.append(ProgrammedRotationExample(model))
    all_gates = circuit.to_ir().instructions
    gates = tuple(g for g in all_gates if g.name != "classical_xor")
    assert len(gates) == 5 * rounds + correct
    assert len(all_gates) == len(gates) + max(0, rounds - 1)
    for i in range(rounds):
        h, rotation, cx, measure, x = gates[i * 5 : i * 5 + 5]
        active = tuple((j, 1) for j in range(i))
        assert (h.name, rotation.name, cx.name, measure.name, x.name) == (
            "h",
            "phase",
            "cx",
            "measure",
            "x",
        )
        assert (h.wires, rotation.wires, cx.wires, measure.wires, x.wires) == (
            (1,),
            (1,),
            (0, 1),
            (1,),
            (1,),
        )
        assert all(
            g.metadata.get("conditions", ()) == active for g in (h, rotation, cx)
        )
        assert measure.metadata.get("classical_bit") == i
        assert measure.metadata.get("conditions", ()) == active
        assert measure.metadata.get("guarded_effect", False) == bool(i)
        assert x.metadata.get("conditions") == (*active, (i, 1))
        assert rotation.params["theta"] == pi * 2**i * theta
    if correct:
        final = gates[-1]
        assert final.wires == (0,)
        assert final.params["theta"] == pi * 2**rounds * theta
        assert final.metadata.get("conditions", ()) == tuple(
            (j, 1) for j in range(rounds)
        )
    for history in product((0, 1), repeat=rounds):
        failures = next((i for i, bit in enumerate(history) if not bit), rounds)
        if any(history[failures:]):
            expected = torch.zeros_like(initial)
        else:
            global_phase = exp(1j * pi * (2**failures - 1) * theta)
            amplitude = global_phase / sqrt(2 ** min(failures + 1, rounds))
            expected = initial * amplitude
            net = theta if failures < rounds or correct else -(2**failures - 1) * theta
            expected[4:] *= exp(1j * pi * net)
        torch.testing.assert_close(
            forced_branch(initial, all_gates, history), expected, rtol=0, atol=2e-12
        )


@pytest.mark.parametrize("invalid", [0, -0.25, 2, float("nan")])
def test_invalid_failure_probability(invalid: float) -> None:
    with pytest.raises(ValueError, match="failure bound"):
        ProgrammedRotationModel.from_failure_probability(
            0.25, max_fail_probability=invalid
        )


def test_probability_one_appends_identity(source_module: Any) -> None:
    model = ProgrammedRotationModel.from_failure_probability(
        0.25, max_fail_probability=1
    )
    assert model.rounds == 0
    reference = source_module.ZPowUsingProgrammedAncilla.from_failure_probability(
        0.25, max_fail_probability=1
    )
    assert_model_matches(model, reference)
    circuit = DynamicCircuit(1)
    circuit.append(ProgrammedRotationExample(model))
    assert circuit.to_ir().instructions == ()
