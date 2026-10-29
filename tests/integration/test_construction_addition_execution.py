"""Real CPU statevectors prove XOR targets, phase coherence and clean workspace."""

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.lowering import lower_addition
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

pytestmark = pytest.mark.integration


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


def _addition(a: int, b: int, out: int) -> tuple[HybridProgram, Value, Value]:
    x, y, result = (
        Value(ValueId("main", index), uint_type(width))
        for index, width in enumerate((a, b, out))
    )
    program = HybridProgram(
        "main",
        Region(
            (
                Block(
                    (x, y),
                    (
                        Operation("qint.add", (x, y), (result,)),
                        Operation("qint.return", (result,)),
                    ),
                ),
            )
        ),
    )
    return program, x, y


def _encode(value: int, wires: tuple[int, ...], count: int) -> int:
    return sum(
        ((value >> bit) & 1) << (count - 1 - wire)
        for bit, wire in enumerate(reversed(wires))
    )


def test_all_16_pairs_all_8_targets_and_zero_workspace_on_real_cpu(
    options: fq.ExecutionOptions,
) -> None:
    main, x, y = _addition(2, 2, 3)
    program = lower_addition(
        (main,),
        {x.id: (0, 1), y.id: (2, 3)},
        (4, 5, 6),
        entry="main",
        n_qubits=13,
        workspace=tuple(range(7, 13)),
        workspace_state="fresh_zero",
    )
    # Execute every target as a gate-prepared Circuit: fq.run does not consume
    # Circuit's private arbitrary-state buffer through its gate-level IR.
    for a in range(4):
        for b in range(4):
            for z in range(8):
                initial = (((a << 2) | b) << 3 | z) << 6
                expected_index = (((a << 2) | b) << 3 | (z ^ (a + b))) << 6
                circuit = fq.Circuit(13, dtype=torch.complex128)
                for wire in range(13):
                    if initial & (1 << (12 - wire)):
                        circuit.x(wire)
                circuit.compose(program)
                state = fq.run(circuit, options=options).statevector().reshape(-1)
                expected = torch.zeros_like(state)
                expected[expected_index] = 1
                torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    resources = estimate_resources(program)
    assert resources.n_qubits == 13
    assert resources.n_operations == 45
    assert resources.operation_counts == {"cx": 37, "ccx": 8}


def test_arbitrary_complex_entangled_inputs_and_permuted_maps_keep_exact_phase(
    options: fq.ExecutionOptions,
) -> None:
    main, x, y = _addition(2, 2, 3)
    x_wires, y_wires, out = (0, 2), (1, 3), (6, 4, 5)
    program = lower_addition(
        (main,),
        {x.id: x_wires, y.id: y_wires},
        out,
        entry="main",
        n_qubits=14,
        workspace=(13, 12, 11, 10, 9, 8),
        workspace_state="caller_asserted_zero",
    )
    generator = torch.Generator().manual_seed(20261029)
    amplitudes = torch.randn(256, generator=generator, dtype=torch.complex128)
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << 14, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    row = 0
    for a in range(4):
        for b in range(4):
            for z in range(8):
                for reference in range(2):
                    prefix = (
                        _encode(a, x_wires, 14)
                        | _encode(b, y_wires, 14)
                        | _encode(reference, (7,), 14)
                    )
                    initial[prefix | _encode(z, out, 14)] = amplitudes[row]
                    expected[prefix | _encode(z ^ (a + b), out, 14)] = amplitudes[row]
                    row += 1
    circuit = fq.Circuit.from_ir(
        program, dtype=torch.complex128, device="cpu", inputs=initial
    )
    # Existing Circuit.state owns arbitrary initial amplitudes; gate-level run
    # integration is covered by the exhaustive basis cases above.
    state = circuit.state()
    assert state.device.type == "cpu"
    torch.testing.assert_close(state.reshape(-1), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("a,b", [(1, 2), (2, 1)])
def test_unequal_widths_full_carry_and_inverse_on_real_cpu(
    options: fq.ExecutionOptions,
    a: int,
    b: int,
) -> None:
    main, x, y = _addition(a, b, 3)
    start = a + b + 3
    program = lower_addition(
        (main,),
        {x.id: tuple(range(a)), y.id: tuple(range(a, a + b))},
        tuple(range(a + b, start)),
        entry="main",
        n_qubits=start + 6,
        workspace=tuple(range(start, start + 6)),
        workspace_state="fresh_zero",
    )
    circuit = fq.Circuit(program.n_wires, dtype=torch.complex128)
    for wire in range(a + b):
        circuit.x(wire)
    block = fq.Circuit.from_ir(program, dtype=torch.complex128)
    expected_index = (((1 << (a + b)) - 1) << 3 | ((1 << a) + (1 << b) - 2)) << 6
    state = fq.run(circuit.compose(block), options=options).statevector().reshape(-1)
    expected = torch.zeros_like(state)
    expected[expected_index] = 1
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    restored = (
        fq.run(circuit.compose(block.adjoint()), options=options)
        .statevector()
        .reshape(-1)
    )
    reference = torch.zeros_like(restored)
    reference[((1 << (a + b)) - 1) << 9] = 1
    torch.testing.assert_close(restored, reference, atol=1e-12, rtol=0)


def test_constant_zero_and_repeated_read_composition_on_real_cpu(
    options: fq.ExecutionOptions,
) -> None:
    x = Value(ValueId("main", 0), uint_type(1))
    zero = Value(ValueId("main", 1), INDEX)
    preserved = Value(ValueId("main", 2), uint_type(1))
    doubled = Value(ValueId("main", 3), uint_type(2))
    main = HybridProgram(
        "main",
        Region(
            (
                Block(
                    (x,),
                    (
                        Operation(
                            "arith.constant", results=(zero,), attributes={"value": 0}
                        ),
                        Operation("qint.add", (x, zero), (preserved,)),
                        Operation("qint.add", (preserved, preserved), (doubled,)),
                        Operation("qint.return", (doubled,)),
                    ),
                ),
            )
        ),
    )
    program = lower_addition(
        (main,),
        {x.id: (0,)},
        (1, 2),
        entry="main",
        n_qubits=11,
        workspace=tuple(range(3, 11)),
        workspace_state="fresh_zero",
    )
    circuit = (
        fq.Circuit(11, dtype=torch.complex128).h(0).compose(fq.Circuit.from_ir(program))
    )
    state = fq.run(circuit, options=options).statevector().reshape(-1)
    expected = torch.zeros_like(state)
    expected[0] = expected[6 << 8] = 2**-0.5
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


def test_composable_call_and_constant_addition_on_real_cpu(
    options: fq.ExecutionOptions,
) -> None:
    argument = Value(ValueId("double", 0), uint_type(1))
    double_result = Value(ValueId("double", 1), uint_type(2))
    double = HybridProgram(
        "double",
        Region(
            (
                Block(
                    (argument,),
                    (
                        Operation("qint.add", (argument, argument), (double_result,)),
                        Operation("qint.return", (double_result,)),
                    ),
                ),
            )
        ),
    )
    x = Value(ValueId("main", 0), uint_type(1))
    doubled = Value(ValueId("main", 1), uint_type(2))
    one = Value(ValueId("main", 2), INDEX)
    answer = Value(ValueId("main", 3), uint_type(2))
    main = HybridProgram(
        "main",
        Region(
            (
                Block(
                    (x,),
                    (
                        Operation("qint.call", (x,), (doubled,), {"callee": "double"}),
                        Operation(
                            "arith.constant", results=(one,), attributes={"value": 1}
                        ),
                        Operation("qint.add", (doubled, one), (answer,)),
                        Operation("qint.return", (answer,)),
                    ),
                ),
            )
        ),
    )
    program = lower_addition(
        (double, main),
        {x.id: (0,)},
        (1, 2),
        entry="main",
        n_qubits=13,
        workspace=tuple(range(3, 13)),
        workspace_state="fresh_zero",
    )
    circuit = fq.Circuit(13, dtype=torch.complex128).h(0).s(0).compose(program)
    state = fq.run(circuit, options=options).statevector().reshape(-1)
    expected = torch.zeros_like(state)
    expected[1 << 10] = 2**-0.5
    expected[7 << 10] = 1j * 2**-0.5
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
