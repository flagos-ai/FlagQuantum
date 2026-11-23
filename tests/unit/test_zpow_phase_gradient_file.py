"""Frozen gradient ZPow fixtures and complete, phase-sensitive small cases."""

from cmath import exp
from collections import Counter
from dataclasses import fields
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import pi, sqrt, tau
from pathlib import Path
from typing import Any

import pytest
import torch
from sympy import Symbol

from examples.algorithms.zpow_phase_gradient_catalog import GradientZPowExample
from flagquantum.compiler._component_trace.resource_model import GateCost
from flagquantum.compiler.registers.phase.phase_gradient import phase_gradient_state_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.rotations import zpow_via_phase_gradient

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(zpow_via_phase_gradient.__file__).parent
    for name, digest in (
        ("__init__.py", "e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8"),
        ("zpow_via_phase_gradient.ipynb", "52d26d103692d41efd47ca719ad24102cf984b79"),
        ("zpow_via_phase_gradient.py", "b9fe0ca61754f68aef838ef23abf2e6ab14e7690"),
        ("zpow_via_phase_gradient_test.py", "17fa1e59be1b4d5b0b845531897485f688f41724"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return zpow_via_phase_gradient


@pytest.mark.parametrize("kind", ["concrete", "symbolic_precision", "symbolic_angle"])
def test_original_factory_signature_and_resources(
    kind: str, source_module: Any
) -> None:
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    if kind == "concrete":
        op = GradientZPowExample.from_precision(3 / 8, epsilon=1e-11)
        source = source_module._zpow_const_via_phase_grad.make()
    elif kind == "symbolic_precision":
        op = GradientZPowExample.from_precision(3 / 8, epsilon=Symbol("eps"))
        source = source_module._zpow_const_via_phase_grad_symb_prec.make()
    else:
        op = GradientZPowExample.from_precision(Symbol("t"), epsilon=1e-11)
        source = source_module._zpow_const_via_phase_grad_symb_angle.make()
    assert [(r.name, r.dtype.num_qubits, r.side.name) for r in op.signature] == [
        (r.name, r.bitsize, r.side.name) for r in source.signature
    ]
    (phase_port,) = [r for r in op.signature if r.name == "phase_grad"]
    dtype = source.phase_grad_dtype
    assert dict(phase_port.dtype.parameters) == {
        "fractional_width": dtype.num_frac,
        "signed": dtype.signed,
    }
    labels = {}
    for child, count in source.bloq_counts().items():
        if isinstance(child, AddIntoPhaseGrad):
            labels[
                f"AddIntoPhaseGrad({child.x_bitsize}, {child.phase_bitsize}, sign={child.sign})"
            ] = count
        else:
            assert child.subbloq == source._load_bloq
            labels[
                f"ControlledXorK({child.subbloq.dtype.num_qubits}, {child.subbloq.k})"
            ] = count
    assert op.call_graph().leaf_counts == labels
    actual = op.resource_model()
    expected = get_cost_value(source, QECGatesCost())
    assert actual.logical_qubits == get_cost_value(source, QubitCount())
    for field in fields(GateCost):
        assert getattr(actual, field.name) == getattr(expected, field.name)
    if kind != "concrete":
        with pytest.raises(ValueError, match="Bind the Z-power"):
            DynamicCircuit(120).append(op)


def assert_bound_stream(
    op: GradientZPowExample,
    source: Any,
) -> tuple[Instruction, ...]:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

    n = op.width
    assert isinstance(n, int)
    gradient, angle, carries = (
        tuple(range(1, n + 1)),
        tuple(range(n + 1, 2 * n + 1)),
        tuple(range(2 * n + 1, 3 * n)),
    )
    ports = {"q": (0,), "phase_grad": gradient}
    stages = []

    def leaf(
        bloq: Any, inputs: SourcePorts, _desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, Allocate):
            assert not bloq.dirty
            stages.append("allocate")
            return (), {"reg": angle}
        if isinstance(bloq, Free):
            assert inputs["reg"] == angle
            stages.append("free")
            return (), {}
        if bloq == source._load_bloq.controlled():
            assert inputs == {"ctrl": (0,), "x": angle}
            stages.append("xor")
            return reference_arithmetic_gates(bloq, inputs, (), {}), inputs
        if isinstance(bloq, AddIntoPhaseGrad):
            assert inputs == {"x": angle, "phase_grad": gradient}
            assert bloq.sign == (-1 if op.adjoint else 1)
            stages.append("add")
            add = Add(QUInt(n), QUInt(n))
            return (
                reference_arithmetic_gates(
                    add.adjoint() if op.adjoint else add,
                    {"a": angle, "b": gradient},
                    carries,
                    dict(zip(carries, range(n - 1), strict=True)),
                ),
                inputs,
            )
        return None

    expected, outputs = walk_source_graph(
        source.adjoint() if op.adjoint else source, ports, ports, leaf
    )
    assert outputs == ports
    assert stages == ["allocate", "xor", "add", "xor", "free"]
    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    gates = circuit.to_ir().instructions
    assert_source_dependencies(gates, expected)
    return gates


@pytest.mark.parametrize("adjoint", [False, True])
def test_original_forty_bit_body(adjoint: bool, source_module: Any) -> None:
    source = source_module._zpow_const_via_phase_grad.make()
    op = GradientZPowExample(3 / 8, 40, adjoint)
    gates = assert_bound_stream(op, source)
    counts = Counter(g.name for g in gates)
    assert op.qubit_count == 120
    assert (counts["t"] + counts["tdg"], counts["cx"], counts["measure"]) == (
        156,
        469,
        39,
    )
    graph = (source.adjoint() if adjoint else source).decompose_bloq()
    pairs = tuple(product((0, 1), (0, 1, 1 << 39, (1 << 40) - 1)))
    start = tuple(q << 119 | y << 79 for q, y in pairs)
    end = tuple(
        q << 119 | int(graph.on_classical_vals(q=q, phase_grad=y)["phase_grad"]) << 79
        for q, y in pairs
    )

    def bits(word: int) -> tuple[int, ...]:
        return tuple(word >> i & 1 for i in reversed(range(120)))

    assert certified_and_channel_map(gates, tuple(map(bits, start))) == (
        tuple(map(bits, end)),
        39,
    )
    state = {j: exp(0.17j * i) / sqrt(len(start)) for i, j in enumerate(start)}
    expected = dict(zip(end, state.values(), strict=True))
    for seed in (4, 19):
        actual = execute_sparse_register(CircuitIR(120, gates), dict(state), seed=seed)
        assert all(
            abs(actual.get(j, 0) - expected.get(j, 0)) < 2e-12
            for j in actual.keys() | expected.keys()
        )


@pytest.mark.parametrize("exponent,width", [(0.75, 3), (0.75, 4), (0.175, 7)])
def test_original_tensor_cases(exponent: float, width: int, source_module: Any) -> None:
    source = source_module.ZPowConstViaPhaseGradient(exponent, width)
    op = GradientZPowExample(exponent, width)
    body = assert_bound_stream(op, source)
    n = op.qubit_count
    target = 1 << (n - 1)
    gradient = tuple(range(1, 1 + width))
    stream = (
        *phase_gradient_state_ir(gradient, exponent=-1),
        *body,
        *phase_gradient_state_ir(gradient, inverse=True, exponent=-1),
    )
    constant = int(source._load_bloq.k)
    if width > 4:
        modulus = 1 << width
        pairs = tuple(product((0, 1), range(modulus)))

        def word(q: int, y: int) -> tuple[int, ...]:
            value = q << (n - 1) | y << (2 * width - 1)
            return tuple(value >> i & 1 for i in reversed(range(n)))

        assert certified_and_channel_map(body, tuple(word(q, y) for q, y in pairs)) == (
            tuple(word(q, (y + q * constant) % modulus) for q, y in pairs),
            width - 1,
        )
    precise = tau * constant / 2**width
    residual = pi * exponent - precise
    assert abs(exp(0.5j * residual) - 1) <= 2**-width
    for vector in ((1, 1j), (1j, 1)):
        state = {0: vector[0] / sqrt(2), target: vector[1] / sqrt(2)}
        expected = {0: state[0], target: state[target] * exp(1j * precise)}
        if width <= 4:
            dense = torch.zeros(1 << n, dtype=torch.complex128)
            out = torch.zeros_like(dense)
            dense[[0, target]] = torch.tensor(
                list(state.values()), dtype=torch.complex128
            )
            out[[0, target]] = torch.tensor(
                list(expected.values()), dtype=torch.complex128
            )
            for history in product((0, 1), repeat=width - 1):
                torch.testing.assert_close(
                    forced_branch(dense, stream, history),
                    out * 2 ** (-(width - 1) / 2),
                    atol=2e-12,
                    rtol=0,
                )
        else:
            for seed in (4, 19):
                actual = execute_sparse_register(
                    CircuitIR(n, stream), dict(state), seed=seed
                )
                assert all(
                    abs(actual.get(j, 0) - expected.get(j, 0)) < 2e-12
                    for j in actual.keys() | expected.keys()
                )


def test_symbolic_precision_retains_negative_unencoded_load() -> None:
    from flagquantum.compiler.registers.phase.zpow_phase_gradient_component import (
        zpow_via_phase_gradient_program,
    )
    from flagquantum.core.ir.program import verify_program_record

    record = zpow_via_phase_gradient_program(
        -0.5, Symbol("n", positive=True, integer=True)
    )
    assert verify_program_record(record) == ()
    assert len(record.functions[0].blocks[0].operations) == 5
    with pytest.raises(ValueError, match="nonnegative"):
        zpow_via_phase_gradient_program(-0.5, 3)
