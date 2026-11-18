"""Review-only block-stage replacement, not a production v4 reader or compiler.

The recorded prototype replaces a retained native baseline behind one callable.
The same consumer checks actual operators, schedules and emits either result.
Core parsing, routed lineage and artifact handoff remain implementation gates.
"""

from __future__ import annotations

import cmath
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import pytest
import torch

from flagquantum.compiler.formats.openqasm import emit_openqasm
from flagquantum.compiler.lowering.schedule_legalization import (
    schedule_circuit_dependencies,
)
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[2]
_TARGET = hashlib.sha256(b"review-only-full-cz-target").hexdigest()
_REJECTIONS = (
    "not_lower_cz_cost",
    "synthesis_unavailable",
    "residual_rejected",
    "instruction_growth_rejected",
)


@dataclass(frozen=True)
class _Rewrite:
    input_instruction_indices: tuple[int, ...]
    output_instruction_indices: tuple[int, ...]
    physical_qubits: tuple[int, int]
    baseline_cz_count: int
    replacement_cz_count: int
    equivalence: Literal["unitary_up_to_global_phase"]
    phase_real: float
    phase_imag: float
    max_abs_residual: float


@dataclass(frozen=True)
class _Stage:
    source_program: CircuitIR
    program: CircuitIR
    source_content_hash: str
    target_snapshot_id: str
    rewrites: tuple[_Rewrite, ...]
    rejection_counts: tuple[tuple[str, int], ...]
    optimization_identity: str


def _identity(stage: _Stage) -> str:
    # A test-domain identity; this does not assign a production v4 encoding.
    payload = {
        "schema": "review_only_block_stage",
        "input": stage.source_content_hash,
        "output": stage.program.content_hash,
        "target": stage.target_snapshot_id,
        "rewrites": [asdict(item) for item in stage.rewrites],
        "rejection_counts": stage.rejection_counts,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _stage(
    source: CircuitIR, output: CircuitIR, rewrites: tuple[_Rewrite, ...]
) -> _Stage:
    value = _Stage(
        source,
        output,
        source.content_hash,
        _TARGET,
        rewrites,
        tuple((key, 0) for key in _REJECTIONS),
        "",
    )
    return replace(value, optimization_identity=_identity(value))


def _fixture() -> tuple[CircuitIR, CircuitIR, _Rewrite]:
    payload = json.loads(
        (
            _ROOT / "docs/api-changes/examples/compiler-cz-block-lineage-v4.json"
        ).read_text()
    )

    def program(key: str) -> CircuitIR:
        return CircuitIR(
            2,
            tuple(
                Instruction(
                    item["opcode"],
                    tuple(item["physical_qubits"]),
                    params=item["parameters"],
                )
                for item in payload[key]
            ),
            dtype="complex128",
            measurements=(MeasurementNode("samples", (0, 1)),),
        )

    record = dict(payload["proposed_block_rewrite"])
    for key in (
        "input_instruction_indices",
        "output_instruction_indices",
        "physical_qubits",
    ):
        record[key] = tuple(record[key])
    return (
        program("native_input_instructions"),
        program("proposed_final_instructions"),
        _Rewrite(**record),
    )


def _operator(program: CircuitIR, indices: tuple[int, ...]) -> torch.Tensor:
    """Embed explicit RX/RZ/CZ definitions, independently of Compiler synthesis."""
    size = 1 << program.n_wires
    result = torch.eye(size, dtype=torch.complex128)
    for index in indices:
        instruction = program.instructions[index]
        gate = torch.zeros((size, size), dtype=torch.complex128)
        for column in range(size):
            if instruction.name == "cz":
                occupied = all(
                    column & (1 << (program.n_wires - 1 - wire))
                    for wire in instruction.wires
                )
                gate[column, column] = -1 if occupied else 1
                continue
            mask = 1 << (program.n_wires - 1 - instruction.wires[0])
            angle = float(instruction.params["theta"])
            if instruction.name == "rz":
                gate[column, column] = cmath.exp(
                    (1j if column & mask else -1j) * angle / 2
                )
            elif instruction.name == "rx":
                gate[column, column] = math.cos(angle / 2)
                gate[column ^ mask, column] = -1j * math.sin(angle / 2)
            else:
                raise ValueError("review oracle supports only RX/RZ/CZ")
        result = gate @ result
    return result


def _check(stage: _Stage, baseline: CircuitIR, *, budget: int) -> None:
    if (
        stage.source_program is not baseline
        or stage.source_content_hash != baseline.content_hash
        or stage.target_snapshot_id != _TARGET
    ):
        raise ValueError("stale baseline instance, content or target")
    if stage.optimization_identity != _identity(stage):
        raise ValueError("stage identity mismatch")
    if tuple(key for key, _ in stage.rejection_counts) != _REJECTIONS or any(
        type(value) is not int or value < 0 for _, value in stage.rejection_counts
    ):
        raise ValueError("invalid closed rejection counters")
    output = stage.program
    if replace(baseline, instructions=output.instructions).content_hash != (
        output.content_hash
    ):
        raise ValueError("non-instruction circuit semantics changed")
    if len(output.instructions) - len(baseline.instructions) > budget:
        raise ValueError("combined native growth budget exceeded")
    inputs: Counter[int] = Counter()
    outputs: Counter[int] = Counter()
    for rewrite in stage.rewrites:
        before = rewrite.input_instruction_indices
        after = rewrite.output_instruction_indices
        if (
            not before
            or len(before) > 64
            or tuple(sorted(set(before))) != before
            or (after and after != tuple(range(after[0], after[0] + len(after))))
            or len(after) > len(before) + 32
            or any(
                type(i) is not int or not 0 <= i < len(baseline.instructions)
                for i in before
            )
            or any(
                type(i) is not int or not 0 <= i < len(output.instructions)
                for i in after
            )
        ):
            raise ValueError("invalid bounded block membership")
        inputs.update(before)
        outputs.update(after)
        pair = rewrite.physical_qubits
        if len(set(pair)) != 2 or any(
            not set(program.instructions[index].wires) <= set(pair)
            for program, members in ((baseline, before), (output, after))
            for index in members
        ):
            raise ValueError("block physical support mismatch")
        for program, members in ((baseline, before), (output, after)):
            for index in range(members[0], members[-1]) if members else ():
                if index not in members and set(
                    program.instructions[index].wires
                ) & set(pair):
                    raise ValueError("block crosses a dependent instruction")
        actual_before = sum(baseline.instructions[i].name == "cz" for i in before)
        actual_after = sum(output.instructions[i].name == "cz" for i in after)
        if (
            rewrite.baseline_cz_count != actual_before
            or rewrite.replacement_cz_count != actual_after
            or actual_after >= actual_before
        ):
            raise ValueError("strict actual CZ savings required")
        phase = complex(rewrite.phase_real, rewrite.phase_imag)
        residual = float(
            (_operator(baseline, before) - phase * _operator(output, after)).abs().max()
        )
        if (
            rewrite.equivalence != "unitary_up_to_global_phase"
            or not math.isfinite(abs(phase))
            or abs(abs(phase) - 1) > 1e-11
            or not math.isfinite(rewrite.max_abs_residual)
            or not 0 <= rewrite.max_abs_residual <= 1e-11
            or residual > 1e-11
        ):
            raise ValueError("actual operator equivalence rejected")
    if any(count != 1 for count in (*inputs.values(), *outputs.values())):
        raise ValueError("overlapping block memberships")
    unchanged_input = [i for i in range(len(baseline.instructions)) if i not in inputs]
    unchanged_output = [i for i in range(len(output.instructions)) if i not in outputs]
    if len(unchanged_input) != len(unchanged_output) or any(
        baseline.instructions[left] != output.instructions[right]
        for left, right in zip(unchanged_input, unchanged_output, strict=True)
    ):
        raise ValueError("uncovered or changed native instruction")
    phase_product = math.prod(
        complex(item.phase_real, item.phase_imag) for item in stage.rewrites
    )
    complete_residual = float(
        (
            _operator(baseline, tuple(range(len(baseline.instructions))))
            - phase_product * _operator(output, tuple(range(len(output.instructions))))
        )
        .abs()
        .max()
    )
    if complete_residual > 1e-11:
        raise ValueError("complete operator equivalence rejected")


def _consume(
    baseline: CircuitIR,
    implementation: Callable[[CircuitIR], _Stage],
) -> tuple[_Stage, str]:
    result = implementation(baseline)
    _check(result, baseline, budget=0)
    schedule = schedule_circuit_dependencies(result.program, target_snapshot_id=_TARGET)
    assert schedule.program is result.program
    return result, emit_openqasm(result.program, version=3.0)


def test_same_schedule_and_emission_consumer_accepts_replaced_implementation() -> None:
    baseline, optimized, rewrite = _fixture()

    def identity(source: CircuitIR) -> _Stage:
        return _stage(source, source, ())

    def recorded_prototype(source: CircuitIR) -> _Stage:
        assert source.content_hash == baseline.content_hash
        return _stage(source, optimized, (rewrite,))

    original, original_text = _consume(baseline, identity)
    replacement, replacement_text = _consume(baseline, recorded_prototype)
    assert original.program is baseline
    assert replacement.source_program is baseline
    assert original_text.count("cz ") == 4
    assert replacement_text.count("cz ") == 2
    assert original.optimization_identity != replacement.optimization_identity
    assert replacement.program.measurements == baseline.measurements


def test_fake_records_match_approved_stage_and_rewrite_contract() -> None:
    contract = json.loads(
        (_ROOT / "contracts/compiler-cz-block-lineage-v4-candidate.json").read_text()
    )
    baseline, output, rewrite = _fixture()
    assert contract["implementation_authorized"] is True
    assert contract["approved_on"] == "2026-11-18"
    assert tuple(asdict(rewrite)) == tuple(contract["block_rewrite_fields"])
    assert tuple(_Stage.__dataclass_fields__) == tuple(
        contract["block_stage_result_fields"]
    )
    assert tuple(dict(_stage(baseline, output, (rewrite,)).rejection_counts)) == tuple(
        contract["rejection_counter_keys"]
    )


@pytest.mark.parametrize(
    "change",
    ["source", "target", "identity", "membership", "cost", "phase", "operator"],
)
def test_rehashed_tampering_cannot_replace_semantic_verification(change: str) -> None:
    baseline, output, rewrite = _fixture()
    candidate = _stage(baseline, output, (rewrite,))
    if change == "source":
        candidate = replace(candidate, source_program=replace(baseline))
    elif change == "target":
        candidate = replace(candidate, target_snapshot_id="f" * 64)
    elif change == "identity":
        candidate = replace(candidate, optimization_identity="f" * 64)
    elif change == "membership":
        candidate = replace(
            candidate,
            rewrites=(
                replace(
                    rewrite,
                    input_instruction_indices=rewrite.input_instruction_indices[1:],
                ),
            ),
        )
    elif change == "cost":
        candidate = replace(
            candidate, rewrites=(replace(rewrite, replacement_cz_count=0),)
        )
    elif change == "phase":
        candidate = replace(candidate, rewrites=(replace(rewrite, phase_real=1.0),))
    else:
        instructions = list(output.instructions)
        instructions[0] = replace(instructions[0], params={"theta": 0.0})
        candidate = replace(
            candidate, program=replace(output, instructions=tuple(instructions))
        )
    if change != "identity":
        candidate = replace(candidate, optimization_identity=_identity(candidate))
    with pytest.raises(ValueError):
        _check(candidate, baseline, budget=0)


def test_identity_cancellation_keeps_a_rewrite_with_empty_output_membership() -> None:
    baseline = CircuitIR(2, (Instruction("cz", (0, 1)), Instruction("cz", (1, 0))))
    output = replace(baseline, instructions=())
    rewrite = _Rewrite(
        (0, 1), (), (0, 1), 2, 0, "unitary_up_to_global_phase", 1.0, 0.0, 0.0
    )
    candidate = _stage(baseline, output, (rewrite,))
    _check(candidate, baseline, budget=0)
    with pytest.raises(ValueError, match="uncovered"):
        _check(_stage(baseline, output, ()), baseline, budget=0)


def test_fixture_baseline_matches_source_swap_then_reversed_cx() -> None:
    baseline, _, _ = _fixture()
    expected = torch.eye(4, dtype=torch.complex128)[[0, 3, 1, 2]]
    actual = _operator(baseline, tuple(range(len(baseline.instructions))))
    phase = actual[0, 0] / expected[0, 0]
    torch.testing.assert_close(actual, phase * expected, atol=1e-11, rtol=0)
    assert abs(abs(complex(phase)) - 1) <= 1e-11


def test_disjoint_spectator_keeps_its_unchanged_native_ancestry() -> None:
    spectator = Instruction("rz", (2,), params={"theta": 0.37})
    baseline = CircuitIR(
        3, (Instruction("cz", (0, 1)), spectator, Instruction("cz", (1, 0)))
    )
    output = replace(baseline, instructions=(spectator,))
    rewrite = _Rewrite(
        (0, 2), (), (0, 1), 2, 0, "unitary_up_to_global_phase", 1.0, 0.0, 0.0
    )
    _check(_stage(baseline, output, (rewrite,)), baseline, budget=0)
    dependent = replace(spectator, wires=(0,))
    baseline = replace(
        baseline,
        instructions=(baseline.instructions[0], dependent, baseline.instructions[2]),
    )
    output = replace(output, instructions=(dependent,))
    with pytest.raises(ValueError, match="dependent instruction"):
        _check(_stage(baseline, output, (rewrite,)), baseline, budget=0)


def test_combined_growth_budget_rejects_before_operator_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, output, rewrite = _fixture()
    output = replace(
        output,
        instructions=output.instructions
        + (Instruction("rz", (0,), params={"theta": 0.0}),) * 6,
    )
    rewrite = replace(
        rewrite, output_instruction_indices=tuple(range(len(output.instructions)))
    )
    candidate = _stage(baseline, output, (rewrite,))

    def forbidden(*args: object) -> torch.Tensor:
        raise AssertionError("operator work occurred before budget admission")

    monkeypatch.setitem(_check.__globals__, "_operator", forbidden)
    with pytest.raises(ValueError, match="growth budget"):
        _check(candidate, baseline, budget=0)


@pytest.mark.parametrize(
    "change", ["duplicate", "overlap", "nan", "counter", "block_limit", "growth_limit"]
)
def test_rehashed_structural_and_resource_violations_fail_closed(change: str) -> None:
    baseline, output, rewrite = _fixture()
    rewrites = (rewrite,)
    if change == "duplicate":
        rewrites = (
            replace(
                rewrite,
                input_instruction_indices=(0,) + rewrite.input_instruction_indices,
            ),
        )
    elif change == "overlap":
        rewrites = (rewrite, rewrite)
    elif change == "nan":
        rewrites = (replace(rewrite, max_abs_residual=float("nan")),)
    elif change == "block_limit":
        baseline = replace(baseline, instructions=baseline.instructions * 3)
        rewrites = (
            replace(
                rewrite,
                input_instruction_indices=tuple(range(len(baseline.instructions))),
            ),
        )
    elif change == "growth_limit":
        output = replace(output, instructions=output.instructions * 3)
        rewrites = (
            replace(
                rewrite,
                output_instruction_indices=tuple(range(len(output.instructions))),
            ),
        )
    candidate = _stage(baseline, output, rewrites) if change != "nan" else None
    if change == "nan":
        # Strict canonical JSON rejects non-finite witnesses before identity creation.
        with pytest.raises(ValueError, match="JSON compliant"):
            _stage(baseline, output, rewrites)
        return
    assert candidate is not None
    if change == "counter":
        candidate = replace(candidate, rejection_counts=(("unknown", 0),))
        candidate = replace(candidate, optimization_identity=_identity(candidate))
    with pytest.raises(ValueError):
        _check(candidate, baseline, budget=100)
