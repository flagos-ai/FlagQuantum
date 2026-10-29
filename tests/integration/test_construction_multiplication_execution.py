"""Real CPU polynomial, phase-oracle and nonzero-gradient acceptance."""

import cmath
import math
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler._construction.lowering import lower_arithmetic
from flagquantum.compiler._construction.profile import uint_type
from flagquantum.compiler._hybrid.model import (
    INDEX,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
    ValueId,
)
from flagquantum.compiler.resource_estimation import estimate_resources
from flagquantum.core.ir import CircuitIR
from flagquantum.core.parameters import Parameter

pytestmark = pytest.mark.integration


def _value(scope: str, index: int, width: int) -> Value:
    return Value(ValueId(scope, index), uint_type(width))


def _function(
    name: str, args: tuple[Value, ...], ops: tuple[Operation, ...]
) -> HybridProgram:
    return HybridProgram(name, Region((Block(args, ops),)))


def _build(
    kind: Literal["polynomial", "product", "constant"] = "polynomial",
    a: int = 2,
    b: int = 2,
    constant: int = 3,
) -> tuple[CircuitIR, tuple[int, ...]]:
    x, y = _value("main", 0, a), _value("main", 1, b)
    maximum = (
        ((1 << a) - 1) ** 2
        if kind == "polynomial"
        else (((1 << a) - 1) * (constant if kind == "constant" else (1 << b) - 1))
    )
    product_width = max(1, maximum.bit_length())
    product = _value("main", 2, product_width)
    functions = []
    if kind == "polynomial":
        arg, square = _value("square", 0, a), _value("square", 1, product_width)
        functions.append(
            _function(
                "square",
                (arg,),
                (
                    Operation("qint.multiply", (arg, arg), (square,)),
                    Operation("qint.return", (square,)),
                ),
            )
        )
        operations = [Operation("qint.call", (x,), (product,), {"callee": "square"})]
    elif kind == "constant":
        number = Value(ValueId("main", 3), INDEX)
        operations = [
            Operation(
                "arith.constant", results=(number,), attributes={"value": constant}
            ),
            Operation("qint.multiply", (x, number), (product,)),
        ]
    else:
        operations = [Operation("qint.multiply", (x, y), (product,))]
    result, work = product, 2 * product_width
    if kind == "polynomial":
        width = max(1, (maximum + (1 << b) - 1).bit_length())
        result = _value("main", 4, width)
        operations.append(Operation("qint.add", (product, y), (result,)))
        work = max(work, 2 * max(product_width, b) + 2)
    else:
        width = product_width
    operations.append(Operation("qint.return", (result,)))
    functions.append(_function("main", (x, y), tuple(operations)))
    begin = a + b + width
    output = tuple(range(a + b, begin))
    program = lower_arithmetic(
        functions,
        {x.id: tuple(range(a)), y.id: tuple(range(a, a + b))},
        output,
        entry="main",
        n_qubits=begin + work,
        workspace=tuple(range(begin, begin + work)),
        workspace_state="fresh_zero",
    )
    return program, output


@pytest.fixture
def options() -> fq.ExecutionOptions:
    return fq.ExecutionOptions(
        mode="statevector",
        backend="torch",
        device="cpu",
        precision="complex128",
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )


def _basis_run(
    program: CircuitIR, initial: int, expected: int, options: fq.ExecutionOptions
) -> None:
    circuit = fq.Circuit(program.n_wires, dtype=torch.complex128, device="cpu")
    for q in range(program.n_wires):
        if initial & (1 << (program.n_wires - 1 - q)):
            circuit.x(q)
    state = fq.run(circuit.compose(program), options=options).statevector().reshape(-1)
    reference = torch.zeros_like(state)
    reference[expected] = 1
    torch.testing.assert_close(state, reference, atol=1e-12, rtol=0)


def test_all_16_polynomial_inputs_and_nonzero_xor_targets_on_real_cpu(
    options: fq.ExecutionOptions,
) -> None:
    program, _ = _build()
    for x in range(4):
        for y in range(4):
            for target in (0, ((3 * x + 5 * y) % 15) + 1):
                initial = (((x << 2) | y) << 4 | target) << 10
                expected = (((x << 2) | y) << 4 | (target ^ (x * x + y))) << 10
                _basis_run(program, initial, expected, options)
    for source in (program, compile_program(program)):
        resources = estimate_resources(source)
        assert resources.n_qubits == 18
        assert resources.n_operations == 162
        assert resources.operation_counts == {"cx": 106, "ccx": 56}


@pytest.mark.parametrize(
    "kind,a,b,constant",
    [
        ("product", 2, 2, 3),
        ("product", 1, 3, 3),
        ("product", 3, 1, 3),
        ("constant", 2, 1, 0),
        ("constant", 2, 1, 5),
    ],
)
def test_products_constants_unequal_widths_and_boundaries_on_real_cpu(
    options: fq.ExecutionOptions,
    kind: Literal["product", "constant"],
    a: int,
    b: int,
    constant: int,
) -> None:
    program, output = _build(kind, a, b, constant)
    width = len(output)
    work = program.n_wires - a - b - width
    for x in range(1 << a):
        for y in range(1 << b):
            value = x * constant if kind == "constant" else x * y
            for target in (0, (1 << width) - 1):
                prefix = (x << b) | y
                _basis_run(
                    program,
                    (prefix << width | target) << work,
                    (prefix << width | (target ^ value)) << work,
                    options,
                )


def test_complex_entangled_inputs_outputs_and_clean_workspace() -> None:
    block, _ = _build()
    # Existing Circuit.state owns arbitrary amplitudes; fq.run basis inputs
    # above are prepared with gates. The last qubit is an external reference.
    program = CircuitIR(n_wires=19, instructions=block.instructions, dtype="complex128")
    amplitudes = torch.randn(
        512, dtype=torch.complex128, generator=torch.Generator().manual_seed(303)
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << 19, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    row = 0
    for x in range(4):
        for y in range(4):
            for target in range(16):
                for reference in range(2):
                    prefix = (x << 2) | y
                    initial[((prefix << 4 | target) << 11) | reference] = amplitudes[
                        row
                    ]
                    expected[
                        ((prefix << 4 | (target ^ (x * x + y))) << 11) | reference
                    ] = amplitudes[row]
                    row += 1
    circuit = fq.Circuit.from_ir(program, inputs=initial, device="cpu")
    state = circuit.state().reshape(-1)
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


def test_compute_mark_uncompute_oracle_has_exact_relative_phase(
    options: fq.ExecutionOptions,
) -> None:
    program, output = _build()
    block = fq.Circuit.from_ir(program, dtype=torch.complex128)
    oracle = fq.Circuit(18, dtype=torch.complex128)
    for q in range(4):
        oracle.h(q)
    oracle.s(1).t(2).compose(block).z(output[1]).compose(block.adjoint())
    expected = torch.zeros(1 << 18, dtype=torch.complex128)
    for x in range(4):
        for y in range(4):
            phase = (1j if x & 1 else 1) * (cmath.exp(1j * math.pi / 4) if y & 2 else 1)
            sign = -1 if (x * x + y) & 4 else 1
            expected[((x << 2) | y) << 14] = 0.25 * phase * sign
    state = fq.run(oracle, options=options).statevector().reshape(-1)
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("width", [1, 2])
def test_parameter_rebinding_preserves_nonzero_analytic_gradients(
    compiled: bool,
    width: int,
) -> None:
    program, output = _build(a=width, b=width)
    circuit = fq.Circuit(program.n_wires, dtype=torch.complex128, device="cpu")
    circuit.ry(width - 1, Parameter("theta")).compose(program)
    if compiled:
        circuit = fq.Circuit.from_ir(compile_program(circuit), device="cpu")
    assert circuit.parameter_names == ("theta",)
    values = []
    identities = set()
    for theta in (0.37, 1.1, 2.0):
        parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
        bound = circuit.bind_parameters({"theta": parameter})
        assert bound.dtype == torch.complex128
        assert str(bound.device) == "cpu"
        assert parameter.requires_grad
        identities.add(bound.to_ir().content_hash)
        value = bound.expectation_ps(z=[output[-1]])
        gradient = torch.autograd.grad(value.sum(), parameter)[0]
        torch.testing.assert_close(
            value, torch.cos(parameter).reshape_as(value), atol=1e-10, rtol=0
        )
        torch.testing.assert_close(gradient, -torch.sin(parameter), atol=1e-8, rtol=0)
        assert torch.isfinite(gradient) and gradient.abs() > 0.1
        values.append(float(value.item()))
    assert len(set(values)) == 3
    assert len(identities) == 3
