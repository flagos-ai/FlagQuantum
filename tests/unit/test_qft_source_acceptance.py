# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source assertion scope: locked Qualtran v0.7.0 qft/*_test.py.

"""Complete QFT protocols and full measured-channel operator proofs."""

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha1
from importlib.metadata import version
from math import sqrt, tau
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.algorithms.primitives.approximate_qft import (
    approximate_qft_ir,
    approximate_qft_workspace_size,
)
from flagquantum.algorithms.primitives.qft_phase_gradient import (
    qft_phase_gradient_ir,
    qft_phase_gradient_requirements,
)
from flagquantum.compiler.registers.arithmetic.integer_product import (
    integer_product_requirements,
)
from flagquantum.compiler.registers.arithmetic.product_accumulation import (
    product_accumulation_ir,
)
from flagquantum.compiler.registers.phase.phase_addition import (
    phase_add_ir,
    phase_add_workspace_size,
)
from flagquantum.compiler.registers.phase.phase_gradient import phase_gradient_state_ir
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.source_example_protocols import source_example_protocols

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_sources() -> None:
    import kickmix
    import qualtran.bloqs.qft

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and callable(kickmix.Simulator)
    root = Path(qualtran.bloqs.qft.__file__).parent
    for name, blob in (
        ("qft_text_book.py", "c560deab207982dff006944015f99e59eb262943"),
        ("qft_text_book_test.py", "031b499182d27681ed3add569a4d0125c548b704"),
        ("qft_phase_gradient.py", "6597eba42bf90995e7e857ea452405a657fc87df"),
        ("qft_phase_gradient_test.py", "d6fee8c159f98ceb8d5bbac7fcd04faf39d7c4aa"),
        ("approximate_qft.py", "e6dfcee8614cf80463c97adfbe282daa239c3dc4"),
        ("approximate_qft_test.py", "7dd0b10f8da35fc476e0d49dda2158c8ada93535"),
        ("two_bit_ffft.py", "a707f9c8025443e291933c0d43d6f45f7a4f9e58"),
        ("two_bit_ffft_test.py", "1fd5edf1be77b20570c8494cdca673137f20fa94"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize(
    "module,name,symbolic",
    (
        ("qft_text_book", "_qft_text_book", False),
        ("qft_text_book", "_symbolic_qft", True),
        ("qft_phase_gradient", "_qft_phase_gradient_small", False),
        ("approximate_qft", "_approximate_qft_small", False),
        ("approximate_qft", "_approximate_qft_from_epsilon", False),
        ("two_bit_ffft", "_two_bit_ffft", False),
    ),
)
def test_all_actual_factory_raw_protocols(
    module: str, name: str, symbolic: bool
) -> None:
    from importlib import import_module

    example = getattr(import_module("qualtran.bloqs.qft." + module), name)
    observed = source_example_protocols(example)
    assert {mode: status for mode, (status, _) in observed.items()} == {
        "make": "PASS",
        "decompose": "NA" if symbolic else "PASS",
        "counts": "UNVERIFIED" if symbolic else "FAIL",
        "serialize": "PASS",
        "qtyping": "NA" if symbolic else "PASS",
    }


def certified_child_permutation(
    gates: tuple[Instruction, ...],
    wires: tuple[int, ...],
    workspace: tuple[int, ...],
    source: Any,
) -> tuple[torch.Tensor, int]:
    """Extract the exact channel from emitted gates on every raw child input."""
    local = {wire: i for i, wire in enumerate((*wires, *workspace))}
    canonical = tuple(
        replace(gate, wires=tuple(local[w] for w in gate.wires)) for gate in gates
    )
    logical = len(wires)
    basis = tuple(
        tuple((value >> (logical - i - 1)) & 1 for i in range(logical))
        + (0,) * len(workspace)
        for value in range(1 << logical)
    )
    outputs, events = certified_and_channel_map(canonical, basis)
    ports = tuple((r.name, int(r.total_bits())) for r in source.signature)
    assert sum(width for _, width in ports) == logical
    result = []
    for value, bits in enumerate(outputs):
        assert not any(bits[logical:])
        actual = sum(bit << (logical - i - 1) for i, bit in enumerate(bits[:logical]))
        arguments, shift = {}, logical
        for name, width in ports:
            shift -= width
            arguments[name] = (value >> shift) & ((1 << width) - 1)
        expected = source.on_classical_vals(**arguments)
        packed = 0
        for name, width in ports:
            packed = (packed << width) | int(expected[name])
        assert actual == packed
        result.append(actual)
    assert len(set(result)) == len(result)
    return torch.tensor(result, dtype=torch.int64), events


def fourier_operator(
    width: int,
    precision: int,
    with_reverse: bool,
    adjoint: bool,
    *,
    recursive: bool = False,
) -> tuple[torch.Tensor, Counter[str], int, float]:
    """Compose actual certified channels on every complex input column.

    This is a test-only channel proof. Workspace is eliminated only after each
    emitted arithmetic block proves its full basis map, positive branch scalar
    and zero work. No ideal adder or sampled trajectory substitutes for it.
    """
    pytest.importorskip("cirq")
    import attrs
    import cirq
    from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct
    from qualtran.bloqs.qft.approximate_qft import ApproximateQFT
    from qualtran.bloqs.qft.qft_phase_gradient import QFTPhaseGradient
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

    logical = width + precision
    if recursive:
        assert precision == width
        required, slot_count = qft_phase_gradient_requirements(width)
    else:
        required = slot_count = approximate_qft_workspace_size(width, precision)
    data, gradient = tuple(range(width)), tuple(range(width, logical))
    workspace = tuple(range(logical, logical + required))
    slots = tuple(range(slot_count))
    emit = qft_phase_gradient_ir if recursive else approximate_qft_ir
    gates = emit(
        data, gradient, workspace, slots, with_reverse=with_reverse, adjoint=adjoint
    )
    envelope = (
        *phase_gradient_state_ir(gradient, exponent=-1),
        *gates,
        *phase_gradient_state_ir(gradient, exponent=-1, inverse=True),
    )
    appended = DynamicCircuit(logical + required).append(
        FieldOperation(logical + required, envelope)
    )
    assert tuple(appended.to_ir().instructions) == envelope
    source = (
        QFTPhaseGradient(width, with_reverse)
        if recursive
        else ApproximateQFT(width, precision, with_reverse)
    )
    child_type = PlusEqualProduct if recursive else AddIntoPhaseGrad
    layers = tuple(
        cirq.decompose(
            source.on(*cirq.LineQubit.range(logical)),
            keep=lambda op: op.gate in (cirq.H, cirq.SWAP)
            or isinstance(getattr(op.gate, "bloq", op.gate), child_type),
        )
    )
    phase = torch.exp(
        -1j * tau * torch.arange(1 << precision, dtype=torch.float64) / (1 << precision)
    ) / sqrt(1 << precision)
    small = Circuit(precision, dtype=torch.complex128)
    local_gradient = tuple(range(precision))
    small.append(
        FieldOperation(precision, phase_gradient_state_ir(local_gradient, exponent=-1))
    )
    torch.testing.assert_close(small.state().flatten(), phase, rtol=0, atol=2e-15)
    small.append(
        FieldOperation(
            precision,
            phase_gradient_state_ir(local_gradient, exponent=-1, inverse=True),
        )
    )
    clean = torch.zeros_like(phase)
    clean[0] = 1
    torch.testing.assert_close(small.state().flatten(), clean, rtol=0, atol=2e-15)
    columns = (
        torch.eye(1 << width, dtype=torch.complex128)[:, None, :] * phase[None, :, None]
    ).reshape(1 << logical, 1 << width)
    labels = torch.arange(1 << logical)
    cursor, events = 0, 0
    for operation in reversed(layers) if adjoint else layers:
        physical = []
        for qubit in operation.qubits:
            assert isinstance(qubit, cirq.LineQubit)
            physical.append(qubit.x)
        wires = tuple(physical)
        if operation.gate in (cirq.H, cirq.SWAP):
            name = "h" if operation.gate == cirq.H else "swap"
            assert gates[cursor] == Instruction(name, wires)
            cursor += 1
            masks = tuple(1 << (logical - w - 1) for w in wires)
            if name == "h":
                (mask,) = masks
                low = labels[(labels & mask) == 0]
                zero, one = columns[low].clone(), columns[low ^ mask].clone()
                columns[low], columns[low ^ mask] = (zero + one) / sqrt(2), (
                    zero - one
                ) / sqrt(2)
            else:
                left, right = masks
                changed = ((labels & left) != 0) != ((labels & right) != 0)
                columns = columns[labels ^ (changed.to(torch.int64) * (left | right))]
            continue
        child = getattr(operation.gate, "bloq", operation.gate)
        if recursive:
            assert isinstance(child, PlusEqualProduct) and not child.is_adjoint
            child = child.adjoint() if adjoint else child
            a, b, c = (
                int(child.a_bitsize),
                int(child.b_bitsize),
                int(child.result_bitsize),
            )
            _, product = integer_product_requirements(a, b)
            used = product + c - 1
            block = product_accumulation_ir(
                wires[:a],
                wires[a : a + b],
                wires[a + b :],
                workspace[:product],
                workspace[product:used],
                slots[: c - 1],
                adjoint=adjoint,
            )
        else:
            assert isinstance(child, AddIntoPhaseGrad)
            assert child.controlled_by == child.right_shift == child.sign == 1
            if adjoint:
                child = attrs.evolve(child, sign=-1)
            active = int(child.x_bitsize)
            used = phase_add_workspace_size(
                active, active + 1, right_shift=1, controlled=True
            )
            block = phase_add_ir(
                wires[1 : active + 1],
                wires[active + 1 :],
                workspace[:used],
                slots[:used],
                right_shift=1,
                sign=int(child.sign),
                control=wires[0],
            )
        actual = gates[cursor : cursor + len(block)]
        assert actual == block
        cursor += len(block)
        permutation, count = certified_child_permutation(
            actual, wires, workspace[:used], child
        )
        events += count
        local_values = sum(
            ((labels >> (logical - wire - 1)) & 1) << (len(wires) - i - 1)
            for i, wire in enumerate(wires)
        )
        new_values, targets = permutation[local_values], labels.clone()
        for i, wire in enumerate(wires):
            shift = logical - wire - 1
            targets = (targets & ~(1 << shift)) | (
                ((new_values >> (len(wires) - i - 1)) & 1) << shift
            )
        moved = torch.empty_like(columns)
        moved[targets] = columns
        columns = moved
    assert cursor == len(gates)
    counts = Counter(g.name for g in gates)
    assert counts["measure"] == counts["reset"] == events
    joint = columns.reshape(1 << width, 1 << precision, 1 << width)
    matrix = torch.einsum("g,dgc->dc", phase.conj(), joint)
    leak = float((joint - matrix[:, None, :] * phase[None, :, None]).abs().max())
    assert leak < 3e-14
    return matrix, counts, events, leak


@pytest.mark.parametrize("with_reverse", (False, True))
def test_seven_bit_full_complex_operator_error_and_all_histories(
    with_reverse: bool, record_property: Callable[[str, object], None]
) -> None:
    pytest.importorskip("cirq")
    import cirq
    from qualtran.bloqs.qft.approximate_qft import ApproximateQFT
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.algorithms.primitives.approximate_qft_resource_model import (
        source_approximate_qft_formula,
    )

    forward, counts, events, leak = fourier_operator(7, 6, with_reverse, False)
    inverse, inverse_counts, inverse_events, inverse_leak = fourier_operator(
        7, 6, with_reverse, True
    )
    expected = torch.as_tensor(
        cirq.unitary(
            cirq.QuantumFourierTransformGate(7, without_reverse=not with_reverse)
        ),
        dtype=torch.complex128,
    )
    torch.testing.assert_close(forward, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(inverse, expected.mH, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        inverse @ forward, torch.eye(128, dtype=torch.complex128), rtol=0, atol=3e-13
    )
    record_property("max_operator_error", float((forward - expected).abs().max()))
    record_property("gradient_leak", max(leak, inverse_leak))
    record_property("forward_primitives", dict(counts))
    record_property("inverse_primitives", dict(inverse_counts))
    record_property("forward_raw_histories", 2**events)
    record_property("inverse_raw_histories", 2**inverse_events)
    original = ApproximateQFT(7, 6, with_reverse)
    source_cost = get_cost_value(original, QECGatesCost())
    declared = source_approximate_qft_formula(7, 6, with_reverse=with_reverse)
    for field in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(declared, field) == getattr(source_cost, field)
    record_property("source_qec_model", source_cost.asdict())
    record_property("source_model_peak", get_cost_value(original, QubitCount()))
    record_property("physical_root_wires", 13 + approximate_qft_workspace_size(7, 6))
    preparation = phase_gradient_state_ir(tuple(range(7, 13)), exponent=-1)
    erase_preparation = phase_gradient_state_ir(
        tuple(range(7, 13)), exponent=-1, inverse=True
    )
    record_property(
        "envelope_additional_primitives",
        dict(Counter(g.name for g in (*preparation, *erase_preparation))),
    )


@pytest.mark.slow
def test_seven_bit_native_append_complex_spectator_and_zero_work() -> None:
    width, precision = 7, 6
    required = approximate_qft_workspace_size(width, precision)
    logical = width + precision
    total = logical + required + 1
    data, gradient = tuple(range(width)), tuple(range(width, logical))
    workspace = tuple(range(logical, logical + required))
    gates = (
        *phase_gradient_state_ir(gradient, exponent=-1),
        *approximate_qft_ir(data, gradient, workspace, tuple(range(required))),
        *phase_gradient_state_ir(gradient, exponent=-1, inverse=True),
    )
    values = torch.arange(1 << (width + 1), dtype=torch.float64).reshape(1 << width, 2)
    inputs = torch.complex(values + 1, values.square().remainder(23) - 11)
    inputs /= inputs.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    addresses = (
        torch.arange(1 << width)[:, None] << (precision + required + 1)
    ) | torch.arange(2)
    initial[addresses] = inputs
    circuit = DynamicCircuit(total, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(total, gates))
    actual = run_dynamic(circuit, shots=1, seed=20261125, strategy="trajectory")
    matrix, _, _, _ = fourier_operator(7, 6, True, False)
    expected = torch.zeros_like(initial)
    expected[addresses] = matrix @ inputs
    torch.testing.assert_close(
        actual.final_states[0].flatten(), expected, rtol=0, atol=3e-12
    )


@pytest.mark.parametrize("width", (2, 3, 4, 5))
@pytest.mark.parametrize("with_reverse", (False, True))
@pytest.mark.parametrize("recursive", (False, True))
def test_full_original_exact_operator_grids(
    width: int,
    with_reverse: bool,
    recursive: bool,
    record_property: Callable[[str, object], None],
) -> None:
    pytest.importorskip("cirq")
    import cirq

    expected = torch.as_tensor(
        cirq.unitary(
            cirq.QuantumFourierTransformGate(width, without_reverse=not with_reverse)
        ),
        dtype=torch.complex128,
    )
    forward, _, events, leak = fourier_operator(
        width, width, with_reverse, False, recursive=recursive
    )
    inverse, _, inverse_events, inverse_leak = fourier_operator(
        width, width, with_reverse, True, recursive=recursive
    )
    torch.testing.assert_close(forward, expected, rtol=0, atol=2e-13)
    torch.testing.assert_close(inverse, expected.mH, rtol=0, atol=2e-13)
    torch.testing.assert_close(
        inverse @ forward,
        torch.eye(1 << width, dtype=forward.dtype),
        rtol=0,
        atol=2e-13,
    )
    record_property("max_operator_error", float((forward - expected).abs().max()))
    record_property("gradient_leak", max(leak, inverse_leak))
    record_property("forward_raw_histories", 2**events)
    record_property("inverse_raw_histories", 2**inverse_events)
