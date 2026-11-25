"""Verify all layers of the pinned routing circuit against a quadratic reference."""

import ast
import hashlib
import itertools
import json
import operator
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.services.synthesis import _synthesize

pytestmark = pytest.mark.integration
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
_WIDTH = 16
_PARAMS = (
    tuple(0.013 * (i + 1) for i in range(24)),
    tuple((-1) ** i * 0.019 * (i + 1) for i in range(24)),
)
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Pow: operator.pow,
}


def _original() -> Path:
    manifest = json.loads((_FIXTURES / "vehicle_routing_sources.json").read_text())
    [row] = manifest["fixtures"]
    path = _FIXTURES / row["file"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    license_bytes = (_FIXTURES / manifest["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == manifest["license_sha256"]
    return path


def _cost(node, bits):
    if isinstance(node, ast.Constant):
        assert type(node.value) in {int, float}
        return node.value
    if isinstance(node, ast.Subscript):
        assert isinstance(node.value, ast.Name) and node.value.id == "x"
        assert isinstance(node.slice, ast.Constant) and type(node.slice.value) is int
        return bits[node.slice.value]
    if isinstance(node, ast.BinOp):
        return _OPERATORS[type(node.op)](
            _cost(node.left, bits), _cost(node.right, bits)
        )
    raise AssertionError(ast.dump(node))


def _reference(source: str, params: tuple[float, ...]) -> fq.Circuit:
    layers = re.findall(
        r"^  phase \((.*), params\[(\d+)\]\);\n  mixer_layer\(params\[(\d+)\], x\);$",
        source,
        re.MULTILINE,
    )
    assert len(layers) == 12
    expressions = {expression for expression, _, _ in layers}
    [expression] = expressions
    node = ast.parse(expression, mode="eval").body
    words = torch.arange(1 << _WIDTH)
    bits = [((words >> i) & 1).to(torch.float64) for i in range(_WIDTH)]
    costs = _cost(node, bits)
    constant = float(costs[0])
    linear = [float(costs[1 << i]) - constant for i in range(_WIDTH)]
    pairs = [
        (i, j, float(costs[(1 << i) | (1 << j)]) - constant - linear[i] - linear[j])
        for i, j in itertools.combinations(range(_WIDTH), 2)
    ]
    reconstructed = constant + sum(a * bit for a, bit in zip(linear, bits, strict=True))
    for i, j, coefficient in pairs:
        reconstructed += coefficient * bits[i] * bits[j]
    torch.testing.assert_close(reconstructed, costs, atol=2e-12, rtol=0)

    circuit = fq.Circuit(_WIDTH, dtype=torch.complex128)
    for wire in range(_WIDTH):
        circuit.h(wire)
    # Qmod array index zero maps to the least-significant statevector bit.
    wires = tuple(reversed(range(_WIDTH)))
    for layer, (_, cost_index, mixer_index) in enumerate(layers):
        assert (int(cost_index), int(mixer_index)) == (2 * layer, 2 * layer + 1)
        gamma, beta = params[2 * layer : 2 * layer + 2]
        # P(t) X P(t) X is exp(it) I, preserving the constant cost phase.
        angle = gamma * constant
        circuit.phase(0, theta=angle).x(0).phase(0, theta=angle).x(0)
        for i, coefficient in enumerate(linear):
            if abs(coefficient) > 1e-13:
                circuit.phase(wires[i], theta=gamma * coefficient)
        for i, j, coefficient in pairs:
            if abs(coefficient) > 1e-13:
                circuit.cphase(wires[i], wires[j], theta=gamma * coefficient)
        for wire in range(_WIDTH):
            circuit.rx(wire, theta=0.1 * beta)
    return circuit


@pytest.mark.parametrize("params", _PARAMS)
def test_vehicle_routing_all_amplitudes(params: tuple[float, ...]) -> None:
    path = _original()
    program = _synthesize(
        path, classical_inputs={"params": params}, max_qubits=16, max_workspace=0
    )
    assert program.report["workspace_qubits"] == 0
    actual = fq.run(program).statevector().flatten()
    expected = fq.run(_reference(path.read_text(), params)).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)


def test_vehicle_routing_respects_operation_budget() -> None:
    with pytest.raises(ValueError, match="operation|budget"):
        _synthesize(
            _original(), classical_inputs={"params": _PARAMS[0]}, max_operations=100
        )
