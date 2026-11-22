"""Check the complete GF Python wrapper composition against frozen records."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

import pytest

from flagquantum.compiler._component_trace.trace import ComponentCall
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx_block_marks import (
    decode_kmx_block_marks,
    encode_kmx_block_marks,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.finite_field.division_binding import (
    field_division_binding,
)
from flagquantum.compiler.registers.finite_field.field_binding import (
    field_addition_binding,
    field_constant_binding,
    field_product_phase_binding,
)
from flagquantum.compiler.registers.finite_field.inverse_binding import (
    field_inverse_binding,
)
from flagquantum.compiler.registers.finite_field.product_binding import (
    field_product_binding,
)
from flagquantum.core._binary_field import BinaryField
from tests.unit.fixtures.generate_field_physical_binding import Case, work_size

pytestmark = pytest.mark.unit


def native_trace(
    case: Case,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    width, route = case["width"], case["route"]
    left, right, output = (
        tuple(reversed(range(start, start + width)))
        for start in (0, width, 2 * width + 1)
    )
    owned = BinaryField(width, case["modulus"]) if case["modulus"] is not None else None
    declared = max(
        case["classical_width"],
        width if route in ("phase", "add_bits") else 1 if "classical" in route else 0,
    )
    count = work_size(case)
    leases = tuple(range(3 * width + 1, 3 * width + 1 + count))
    if case["reuse_pool"]:
        leases = leases[::-1]
    work = tuple(
        q
        for start in range(0, count, width)
        for q in reversed(leases[start : start + width])
    )
    control = (
        2 * width
        if "quantum" in route
        else (
            Bit(declared - 1)
            if "classical" in route
            else False if "false" in route else True
        )
    )
    if route.startswith("add"):
        offset: object = (
            3
            if "constant" in route
            else (
                tuple(Bit(b) for b in reversed(range(width)))
                if "bits" in route
                else left
            )
        )
        return field_addition_binding(offset, output, field=owned, control=control)
    if route.startswith("constant"):
        return field_constant_binding(
            1 if width == 1 else 3, output, field=owned, divide=route.endswith("div")
        )
    if route == "phase":
        return field_product_phase_binding(
            left,
            right,
            tuple(Bit(b) for b in reversed(range(width))),
            field=owned,
            classical_width=declared,
        )
    if "inverse" in route:
        return field_inverse_binding(
            left,
            output,
            field=owned,
            workspace=work,
            retain_chain="scaffold" in route,
            erase=route.startswith("erase"),
            classical_width=declared,
        )
    mode: Literal["init", "xor", "erase"] = (
        "init"
        if route.startswith("init")
        else "erase" if route.startswith("erase") else "xor"
    )
    if "product" in route:
        return field_product_binding(
            left,
            right,
            output,
            field=owned,
            workspace=work,
            control=control,
            mode=mode,
            classical_width=declared,
        )
    return field_division_binding(
        left,
        left if route.endswith("self") else right,
        output,
        field=owned,
        workspace=work,
        retain_chain="scaffold" in route,
        mode=mode,
        classical_width=declared,
    )


def test_complete_original_binding_streams_and_kmx_roundtrip() -> None:
    fixture = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("field_physical_binding.json")
        .read_text()
    )
    assert fixture["source_blob"] == "5f111727a1e383394ef856e7104ff4817ae12f31"
    for row in fixture["cases"]:
        result = native_trace(row["input"])
        text = encode_kmx_block_marks(*result)
        body = "\n".join(text.splitlines()[1:])
        assert len(body.splitlines()) == row["records"], row["input"]
        assert (
            dict(Counter(line.split()[0] for line in body.splitlines()))
            == row["counts"]
        ), row["input"]
        assert hashlib.sha256(body.encode()).hexdigest() == row["sha256"], row["input"]
        assert decode_kmx_block_marks(text) == result


def _basis_index(width: int, words: tuple[tuple[tuple[int, ...], int], ...]) -> int:
    result = 0
    for word, value in words:
        for position, wire in enumerate(word):
            result |= ((value >> (len(word) - position - 1)) & 1) << (width - wire - 1)
    return result


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("mode", ["init", "xor", "erase"])
@pytest.mark.parametrize(
    "control_kind", ["true", "quantum", "classical_on", "classical_off"]
)
def test_product_wrapper_all_complex_measurement_branches(
    degree: int,
    mode: Literal["init", "xor", "erase"],
    control_kind: str,
) -> None:
    from itertools import product

    import torch

    from flagquantum.compiler.lowering.reversible_phase_execution import (
        bind_reversible_phase_execution,
    )
    from flagquantum.core.classical_dataflow import ClassicalXor
    from tests.unit.measurement_checks import expand_reset_branches, forced_branch

    field = BinaryField(degree)
    left, right, target = (
        tuple(range(start, start + degree)) for start in (0, degree, 2 * degree)
    )
    quantum_control = 3 * degree
    work = tuple(
        range(
            quantum_control + 1,
            quantum_control + 1 + (degree if control_kind == "quantum" else 0),
        )
    )
    width = quantum_control + len(work) + 2
    spectator = width - 1
    control = (
        quantum_control
        if control_kind == "quantum"
        else Bit(3) if control_kind.startswith("classical") else True
    )
    program, _ = field_product_binding(
        left,
        right,
        target,
        field=field,
        mode=mode,
        control=control,
        workspace=work,
        classical_width=5,
    )
    setup = tuple(
        ClassicalXor((), bit, bit == 3 and control_kind == "classical_on").instruction()
        for bit in program.input_bits
    )
    gates = expand_reset_branches(
        (*setup, *bind_reversible_phase_execution(program)), first_slot=100
    )
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for a, b, enabled, tail in product(
        range(1 << degree), range(1 << degree), (0, 1), (0, 1)
    ):
        active = (
            bool(enabled)
            if control_kind == "quantum"
            else control_kind != "classical_off"
        )
        value = field.multiply(a, b)
        old = (
            value
            if mode == "erase" and control_kind != "quantum"
            else (1 << degree) - 1
        )
        new = (
            (value if active else 0)
            if mode == "init"
            else old ^ (value if active else 0)
        )
        ports = (
            (left, a),
            (right, b),
            ((quantum_control,), enabled),
            ((spectator,), tail),
        )
        index = _basis_index(width, (*ports, (target, old)))
        out = _basis_index(width, (*ports, (target, new)))
        amplitude = torch.tensor(
            complex(1 + 3 * a + 7 * b + enabled, 5 - 2 * b + 3 * tail - a),
            dtype=torch.complex128,
        )
        initial[index] = amplitude
        expected[out] = amplitude
    norm = initial.norm()
    initial /= norm
    expected /= norm
    events = sum(gate.name == "measure" for gate in gates)
    target_resets = degree if mode == "init" else 0
    total = torch.zeros_like(initial.real)
    for history in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, history)
        active_events = target_resets if control_kind == "classical_off" else events
        sign = -1 if sum(history[:target_resets]) % 2 else 1
        branch = expected * sign / 2 ** (active_events / 2)
        if control_kind == "classical_off" and any(history[target_resets:]):
            branch = torch.zeros_like(expected)
        torch.testing.assert_close(actual, branch, atol=8e-14, rtol=8e-14)
        total += actual.abs().square()
    torch.testing.assert_close(total, expected.abs().square(), atol=8e-14, rtol=8e-14)
    if work:
        work_mask = sum(1 << (width - wire - 1) for wire in work)
        states = torch.arange(1 << width)
        assert torch.count_nonzero(total[(states & work_mask) != 0]) == 0


def test_native_wrapper_admission_and_immutable_caller_ports() -> None:
    target, left, right = (4, 5), (0, 1), (2, 3)
    field = BinaryField(2)
    before = (target, left, right, field)
    for constant in (True, False, left, 1.5):
        with pytest.raises(NotImplementedError):
            field_constant_binding(constant, target, field=field)
    for constant in (-1, 0, 4, 7):
        with pytest.raises(ValueError):
            field_constant_binding(constant, target, field=field)
    with pytest.raises(ValueError):
        field_product_binding(left, right, (4,), field=field, mode="init")
    with pytest.raises(ValueError):
        field_inverse_binding(left, (4,), field=field)
    with pytest.raises(ValueError):
        field_division_binding(
            left, right, (4,), workspace=(6, 7), field=field, mode="init"
        )
    with pytest.raises(ValueError):
        field_product_binding(left, right, target, control=6, workspace=(7,))
    with pytest.raises(ValueError):
        field_inverse_binding(left, target, retain_chain=True, workspace=(6,))
    with pytest.raises(ValueError):
        field_product_phase_binding(left, right, (True, False), field=field)
    with pytest.raises(ValueError):
        field_addition_binding(-1, target)
    with pytest.raises(ValueError):
        field_addition_binding(4, target)
    assert (target, left, right, field) == before


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("kind", ["inverse", "division"])
@pytest.mark.parametrize("retain", [False, True])
def test_inverse_and_division_wrapper_joined_all_branches(
    degree: int,
    kind: str,
    retain: bool,
) -> None:
    from itertools import product

    import torch

    from flagquantum.compiler.lowering.reversible_phase_execution import (
        bind_reversible_phase_execution,
    )
    from flagquantum.compiler.registers.finite_field.division_chain import (
        field_division_workspace_size,
    )
    from flagquantum.compiler.registers.finite_field.inverse_chain import (
        field_inverse_chain_size,
        field_inverse_workspace_size,
    )
    from flagquantum.core.classical_dataflow import ClassicalXor
    from tests.unit.measurement_checks import expand_reset_branches, forced_branch

    source = tuple(range(degree))
    right = tuple(range(degree, 2 * degree)) if kind == "division" else ()
    offset = degree + len(right)
    target = tuple(range(offset, offset + degree))
    count = (
        field_division_workspace_size(degree)
        if right
        else (
            field_inverse_chain_size(degree)
            if retain
            else field_inverse_workspace_size(degree)
        )
    )
    work = tuple(range(offset + degree, offset + degree + count))
    width = offset + degree + count + 1
    if right:
        forward, _ = field_division_binding(
            source, right, target, workspace=work, retain_chain=retain, mode="init"
        )
        erase, _ = field_division_binding(
            source, right, target, workspace=work, retain_chain=retain, mode="erase"
        )
    else:
        forward, _ = field_inverse_binding(
            source, target, workspace=work, retain_chain=retain
        )
        erase, _ = field_inverse_binding(
            source, target, workspace=work, retain_chain=retain, erase=True
        )
    instructions = tuple(
        gate
        for program in (forward, erase)
        for gate in (
            *tuple(ClassicalXor((), bit).instruction() for bit in program.input_bits),
            *bind_reversible_phase_execution(program),
        )
    )
    gates = expand_reset_branches(instructions, first_slot=100)
    events = sum(gate.name == "measure" for gate in gates)
    assert events <= 12
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for a, b, tail in product(range(1 << degree), range(1 << len(right)), (0, 1)):
        ports = ((source, a), (right, b), ((width - 1,), tail))
        amplitude = torch.tensor(
            complex(1 + 3 * a + b, 7 - 2 * b - a + tail), dtype=torch.complex128
        )
        initial[_basis_index(width, (*ports, (target, (1 << degree) - 1)))] = amplitude
        expected[_basis_index(width, ports)] = amplitude
    norm = initial.norm()
    initial /= norm
    expected /= norm
    total = torch.zeros_like(initial.real)
    for history in product((0, 1), repeat=events):
        sign = -1 if sum(history[:degree]) % 2 else 1
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            actual, expected * sign / 2 ** (events / 2), atol=9e-14, rtol=9e-14
        )
        total += actual.abs().square()
    torch.testing.assert_close(total, expected.abs().square(), atol=9e-14, rtol=9e-14)
    clear_mask = sum(1 << (width - wire - 1) for wire in (*target, *work))
    assert torch.count_nonzero(total[(torch.arange(1 << width) & clear_mask) != 0]) == 0


def test_wrapper_normal_append_and_counts() -> None:
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.lowering.reversible_phase_execution import (
        bind_reversible_phase_execution,
    )
    from flagquantum.core.classical_dataflow import ClassicalXor
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    field = BinaryField(2)
    left, right, target, work = (0, 1), (2, 3), (4, 5), (7, 8)
    result = field_product_binding(
        left, right, target, field=field, control=6, workspace=work, mode="init"
    )
    program, _ = result
    counts = Counter(operation.opcode for operation in program.operations)
    assert counts["r"] == 4 and counts["hmr"] == 2
    gates = (
        *tuple(ClassicalXor((), bit).instruction() for bit in program.input_bits),
        *bind_reversible_phase_execution(program),
    )
    initial = torch.zeros(1 << program.n_qubits, dtype=torch.complex128)
    index = _basis_index(program.n_qubits, ((left, 3), (right, 2), ((6,), 1)))
    initial[index] = 1
    circuit = DynamicCircuit(program.n_qubits, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(program.n_qubits, gates))
    trajectories = run_dynamic(circuit, shots=8, seed=71, strategy="batched")
    expected = torch.zeros_like(initial)
    expected[
        _basis_index(
            program.n_qubits,
            ((left, 3), (right, 2), ((6,), 1), (target, field.multiply(3, 2))),
        )
    ] = 1
    torch.testing.assert_close(
        trajectories.final_states.reshape(8, -1),
        expected.expand(8, -1),
        atol=1e-12,
        rtol=0,
    )
