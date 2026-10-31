"""Controlled predicate phase compares to a general evaluator and exact states."""

import ast
import itertools
import math
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.lowering import Schedule
from tests.integration.test_synthesis_register_workflow import check_state
from tests.team.compiler.test_construction_controlled_phase import (
    alias,
    build,
    controlled,
    general_phase,
    mask,
    products,
)

pytestmark = pytest.mark.integration


def always(enable: int, x: int) -> int:
    return enable and (x * 0 + 1)


def never(enable: int, x: int) -> int:
    return enable and (x * 0)


def shared(enable: int, x: int) -> int:
    return enable and (enable or x)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize(
    "function,widths",
    [
        (mask, {"enable": 1, "flag": 1}),
        (alias, {"x": 1}),
        (controlled, {"enable": 1, "x": 2, "y": 2}),
        (products, {"x": 2, "y": 2}),
        (always, {"enable": 1, "x": 1}),
        (never, {"enable": 1, "x": 1}),
        (shared, {"enable": 1, "x": 1}),
    ],
)
def test_direct_and_general_prefix_preserve_entangled_reference_and_absolute_sign(
    strategy: Schedule, function: Callable[..., object], widths: Mapping[str, int]
) -> None:
    candidate = build(function, widths, strategy)
    baseline = general_phase(function, widths, strategy)
    n_input = sum(widths.values())
    amplitudes = torch.randn(
        1 << (n_input + 1),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(976),
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    truths = [
        int(bool(function(*values)))
        for values in itertools.product(
            *(range(1 << width) for width in widths.values())
        )
    ]
    for program in (candidate.circuit, baseline):
        n = program.n_wires
        assert n + 1 <= 18
        work = n - n_input
        initial = torch.zeros(1 << (n + 1), dtype=torch.complex128)
        expected = torch.zeros_like(initial)
        for values, truth in enumerate(truths):
            for reference in range(2):
                index = (values << (work + 1)) | reference
                amplitude = amplitudes[(values << 1) | reference]
                initial[index] = amplitude
                expected[index] = amplitude * (-1) ** truth
        embedded = fq.Circuit(n + 1, dtype=torch.complex128).compose(
            program, qubits=range(n)
        )
        observed = (
            fq.Circuit.from_ir(embedded.to_ir(), inputs=initial, device="cpu")
            .state()
            .reshape(-1)
        )
        torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
        assert (
            float(
                observed.reshape(1 << n_input, 1 << work, 2)[:, 1:, :]
                .abs()
                .square()
                .sum()
            )
            <= 1e-12
        )
    assert len(candidate.circuit.instructions) < len(baseline.instructions)


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_public_all_32_control_data_inputs_and_actual_plan(
    objective: Literal["width", "depth"],
) -> None:
    for enable in range(2):
        for x in range(4):
            for y in range(4):
                program = synth.synthesize(
                    controlled,
                    inputs={
                        "enable": synth.UInt(1),
                        "x": synth.UInt(2),
                        "y": synth.UInt(2),
                    },
                    input_state=synth.InputState.basis(enable=enable, x=x, y=y),
                    output_mode="phase",
                    objective=objective,
                    max_qubits=16,
                    memory_limit_bytes=16 * 1024**2,
                )
                result = fq.run(program)
                check_state(
                    program, result, {(enable, x, y): (-1) ** int(enable and x < y)}
                )
                assert program.report["resources"] == program._checked.verify_result(
                    result
                )
                assert tuple(program.registers) == ("enable", "x", "y")
                assert "arithmetic_resources" not in program.report


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_two_qubit_public_mask_fits_zero_workspace_and_exact_budget(
    objective: Literal["width", "depth"],
) -> None:
    program = synth.synthesize(
        mask,
        inputs={"enable": synth.UInt(1), "flag": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        output_mode="phase",
        objective=objective,
        max_qubits=2,
        max_workspace=0,
        max_operations=3,
    )
    result = fq.run(program)
    check_state(
        program,
        result,
        {(c, x): (-1) ** (c * x) / 2 for c in range(2) for x in range(2)},
    )
    assert program.report["workspace_qubits"] == 0
    assert program.report["resources"].n_qubits == 2
    assert program.report["resources"].n_operations == 3
    assert program.execution.state_bytes == 64
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            mask,
            inputs={"enable": synth.UInt(1), "flag": synth.UInt(1)},
            input_state=synth.InputState.uniform(),
            output_mode="phase",
            max_operations=2,
        )


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize(
    "function,widths",
    [
        (always, {"enable": 1, "x": 1}),
        (never, {"enable": 1, "x": 1}),
        (controlled, {"enable": 1, "x": 2, "y": 2}),
    ],
)
def test_control_interference_reads_the_predicate_and_double_application_is_identity(
    objective: Literal["width", "depth"],
    function: Callable[..., object],
    widths: Mapping[str, int],
) -> None:
    program = synth.synthesize(
        function,
        inputs={name: synth.UInt(w) for name, w in widths.items()},
        output_mode="phase",
        objective=objective,
        max_qubits=16,
    )
    n = program.circuit.n_wires
    n_input = sum(widths.values())
    preparation = fq.Circuit(n, dtype=torch.complex128)
    for q in range(n_input):
        preparation.h(q)
    preparation.s(n_input - 1)
    circuit = preparation.compose(program.circuit).h(0)
    observed = (
        fq.run(circuit, options=fq.ExecutionOptions(precision="complex128"))
        .statevector()
        .reshape(-1)
    )
    expected = torch.zeros_like(observed)
    data_count = 1 << (n_input - 1)
    for values in itertools.product(
        *(range(1 << w) for w in tuple(widths.values())[1:])
    ):
        data = 0
        for value, width in zip(values, tuple(widths.values())[1:], strict=True):
            data = (data << width) | value
        truth = int(bool(function(1, *values)))
        index = ((truth << (n_input - 1)) | data) << (n - n_input)
        expected[index] = (1j if data & 1 else 1) / math.sqrt(data_count)
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    twice = fq.Circuit(n, dtype=torch.complex128)
    for q in range(n_input):
        twice.h(q)
    twice.s(n_input - 1).compose(program.circuit).compose(program.circuit)
    observed = (
        fq.run(twice, options=fq.ExecutionOptions(precision="complex128"))
        .statevector()
        .reshape(-1)
    )
    expected.zero_()
    for bits in range(1 << n_input):
        expected[bits << (n - n_input)] = (1j if bits & 1 else 1) / math.sqrt(
            1 << n_input
        )
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_public_complex_preparation_keeps_control_zero_identity(
    objective: Literal["width", "depth"],
) -> None:
    preparation = (
        fq.Circuit(5, dtype=torch.complex128).h(0).h(1).h(2).h(3).h(4).s(0).t(2)
    )
    program = synth.synthesize(
        controlled,
        inputs={"enable": synth.UInt(1), "x": synth.UInt(2), "y": synth.UInt(2)},
        output_mode="phase",
        objective=objective,
        preparation=preparation.to_ir(),
        preparation_map={"enable": (0,), "x": (1, 2), "y": (3, 4)},
        max_qubits=16,
    )
    expected = {}
    for c in range(2):
        for x in range(4):
            for y in range(4):
                angle = (x & 1) * math.pi / 4
                phase = (1j if c else 1) * complex(math.cos(angle), math.sin(angle))
                expected[(c, x, y)] = phase * (-1) ** int(c and x < y) / math.sqrt(32)
    result = fq.run(program)
    check_state(program, result, expected)
    assert dict(
        program.probabilities(result, registers=("enable", "x", "y"))
    ) == pytest.approx(dict.fromkeys(expected, 1 / 32), abs=1e-12)


def test_value_mask_remains_a_clean_xor_result() -> None:
    program = synth.synthesize(
        mask,
        inputs={"enable": synth.UInt(1), "flag": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
    )
    check_state(
        program,
        fq.run(program),
        {(c, x, c * x): 0.5 for c in range(2) for x in range(2)},
    )
    assert program.report["workspace_qubits"] == 2
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            mask,
            inputs={"enable": synth.UInt(1), "flag": synth.UInt(1)},
            max_workspace=0,
        )


def test_public_controlled_example_executes_without_private_imports_or_gates() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "examples.synthesis_controlled_predicate"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    lines = dict(
        line.split(": ", 1) for line in result.stdout.splitlines() if ": " in line
    )
    assert ast.literal_eval(lines["control/data probabilities"]) == pytest.approx(
        {(c, x, y): 1 / 32 for c in range(2) for x in range(4) for y in range(4)},
        abs=1e-12,
    )
    assert lines["two-bit workspace"] == "0"
    assert "Uniform probabilities do not verify" in result.stdout
    source = (root / "examples/synthesis_controlled_predicate.py").read_text()
    assert (
        "._" not in source
        and "fq.Circuit" not in source
        and "physical_of" not in source
    )
