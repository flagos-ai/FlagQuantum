"""The public fifty-control native path executes a proved rank-one MPS."""

import json
from pathlib import Path
from typing import NoReturn, TypedDict

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.mcx.main import build_program
from flagquantum import synth
from flagquantum.compiler._construction.selection import select_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem._qmod.source import load_qmod
from flagquantum.runtime.execution_plan_contract import plan_from_dict, plan_to_dict
from flagquantum.runtime.mps_result import native_mps
from flagquantum.simulation.mps.state import MPSState
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


class _Limits(TypedDict, total=False):
    max_qubits: int
    max_workspace: int
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


def definition(enabled: int, target: int = 0) -> synth.Program:
    prep = "\n".join(f"    X(c[{bit}])" for bit in range(enabled))
    prep += "\n    X(t)" if target else ""
    entry = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef main(c:Output[QArray[QBit]],t:Output[QBit]):\n    allocate(50,c)\n    allocate(t)\n"
        + prep
        + "\n    control(c,lambda:X(t))\n"
    )
    return synth.synthesize(entry, memory_limit_bytes=2048, max_workspace=13)


def forbid_dense(*args: object, **kwargs: object) -> NoReturn:
    pytest.fail("basis-product admission materialized a dense state")


def test_original_width_and_public_native_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MPSState, "to_statevector", forbid_dense)
    case = next(
        c
        for c in json.loads(
            (ROOT / "tests/fixtures/classiq_qmod/corpus_catalog.json").read_text()
        )["cases"]
        if c["id"] == "tutorials/basic_tutorials/mcx/mcx.qmod"
    )
    imported = load_qmod(ROOT / case["original"]["path"])
    selected = select_register_program(
        imported.program, imported.functions, include_boolean=True
    ).selected
    assert selected and selected.target
    program = build_program()
    assert program.circuit.n_wires == selected.target.submitted.n_wires == 64
    assert (
        len(program.circuit.instructions)
        == len(selected.target.submitted.instructions)
        == 169
    )
    assert program.report["execution_profile"] == "exact_computational_basis_mps"
    assert program.report["intermediate_mps_bond_bound"] == 1
    assert program.report["workspace_qubits"] == 13
    assert len(program.registers["cntrl"]) == 50
    result = fq.run(program)
    native = native_mps(result)
    original = native_mps(
        fq.run(
            selected.target.submitted,
            options=fq.ExecutionOptions(
                mode="mps",
                precision="complex128",
                allow_approximate=False,
                memory_limit_bytes=2048,
            ),
        )
    )
    assert original and native
    for left, right in zip(native.tensors, original.tensors, strict=True):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=0)
    assert native and native.n_qubits == 64 and native.max_bond == 1
    assert not native.truncation_records
    assert (
        result.plan
        and result.plan.state_mode == "mps"
        and result.plan.state_bytes == 2048
    )
    assert dict(program.probabilities(result, registers=("target",))) == {(0,): 1.0}
    with pytest.raises(ValueError, match="readout budget"):
        program.probabilities(result, registers=("cntrl",))
    restored = plan_from_dict(plan_to_dict(program.execution))
    restored_result = fq.run(restored)
    assert dict(program.probabilities(restored_result, registers=("target",))) == {
        (0,): 1.0
    }


@pytest.mark.parametrize("enabled", (0, 49, 50))
@pytest.mark.parametrize("target", (0, 1))
def test_complete_product_state_and_all_scratch_zero(
    enabled: int, target: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MPSState, "to_statevector", forbid_dense)
    program = definition(enabled, target)
    actual = native_mps(fq.run(program))
    reference = fq.Circuit(64, dtype=torch.complex128)
    for bit in range(enabled):
        reference.x(49 - bit)
    if target ^ (enabled == 50):
        reference.x(50)
    expected = native_mps(
        fq.run(
            reference.to_ir(),
            options=fq.ExecutionOptions(
                mode="mps",
                backend="torch",
                device="cpu",
                precision="complex128",
                allow_approximate=False,
                allow_backend_fallback=False,
                memory_limit_bytes=2048,
            ),
        )
    )
    assert actual and expected
    assert actual.max_bond == expected.max_bond == 1
    assert not actual.truncation_records and not expected.truncation_records
    for left, right in zip(actual.tensors, expected.tensors, strict=True):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=0)


def test_absolute_negative_sign_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MPSState, "to_statevector", forbid_dense)
    circuit = CircuitIR(
        n_wires=64,
        instructions=(
            Instruction("x", (0,)),
            Instruction("x", (1,)),
            Instruction("z", (0,)),
            Instruction("ccx", (0, 1, 63)),
        ),
    )
    state = native_mps(
        fq.run(
            circuit,
            options=fq.ExecutionOptions(
                mode="mps",
                precision="complex128",
                allow_approximate=False,
                memory_limit_bytes=2048,
            ),
        )
    )
    assert state and state.max_bond == 1 and not state.truncation_records
    phase = 1 + 0j
    for index, site in enumerate(state.tensors):
        amplitudes = site.reshape(2)
        bit = int(index in {0, 1, 63})
        assert amplitudes[1 - bit] == 0
        phase *= complex(amplitudes[bit])
    assert phase == pytest.approx(-1 + 0j, abs=1e-12)


@pytest.mark.parametrize(
    "budget",
    [
        {"memory_limit_bytes": 2047},
        {"max_qubits": 63},
        {"max_workspace": 12},
        {"max_operations": 168},
        {"max_depth": 166},
    ],
)
def test_actual_budgets_reject(budget: _Limits) -> None:
    from examples.synthesis.tutorials.basics.mcx.main import main

    with pytest.raises(ValueError, match="budget|memory|limit|candidate|workspace"):
        synth.synthesize(main, **budget)


def test_native_coherent_program_keeps_dense_budget_rejection() -> None:
    entry = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef main(c:Output[QArray[QBit]],t:Output[QBit]):\n    allocate(50,c)\n    allocate(t)\n    H(c[0])\n    control(c,lambda:X(t))\n"
    )
    with pytest.raises(ValueError, match="memory|budget|limit"):
        synth.synthesize(entry, memory_limit_bytes=2048)


@pytest.mark.parametrize("coherent", (False, True))
def test_basis_proof_does_not_certify_custom_initial_state_or_coherent_sequence(
    coherent: bool,
) -> None:
    circuit = fq.Circuit(64, dtype=torch.complex128)
    if coherent:
        circuit.h(0)
    circuit.ccx(0, 49, 63)
    plan = fq.plan(
        circuit.to_ir() if coherent else circuit,
        options=fq.ExecutionOptions(
            mode="mps", allow_approximate=False, memory_limit_bytes=2048
        ),
    )
    assert plan.state_bytes > 2048
    # The raw Runtime plan remains inspectable; Services refuses admission
    # against the state-buffer ceiling before executing a synthesis Program.
