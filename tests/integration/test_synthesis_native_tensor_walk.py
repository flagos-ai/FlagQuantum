"""Independent zero-entry walk proof and complete occupied-support MPS states."""

import ast
import math
import re
from pathlib import Path
from typing import NoReturn, TypedDict

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.tensor_hypercontraction import (
    main as native,
)
from examples.synthesis.applications.chemistry.tensor_hypercontraction.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.integration
ORIGINAL = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/applications/chemistry/tensor_hypercontraction/tensor_hypercontraction.qmod"
)


class Limits(TypedDict, total=False):
    max_qubits: int
    max_workspace: int
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


def forbid_dense(*args: object, **kwargs: object) -> NoReturn:
    pytest.fail("the 83-bit tensor walk must not allocate a dense full state")


def occupied_state(program: synth.Program) -> torch.Tensor:
    actual = fq.run(program).native()
    assert isinstance(actual, MPSState)
    assert not actual.truncation_records and actual.max_bond <= 4
    if "ell" in program.registers:
        active = (*program.registers["t"], *program.registers["psi"][:2])
    else:
        active = (
            *program.registers["t"],
            *program.registers["psi.plus2"],
            *program.registers["psi.plus1"],
        )
    words = []
    for index in range(64):
        bits = ["0"] * 83
        for wire, bit in zip(active, f"{index:06b}", strict=True):
            bits[wire] = bit
        words.append("".join(bits))
    observed = actual.amplitudes(words).flatten()
    # All omitted ELL, inactive spin and workspace words have zero mass.
    assert float(observed.abs().square().sum()) == pytest.approx(1, abs=1e-12)
    assert program.report["workspace_qubits"] == 0
    return observed


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_original_native_independent_complete_support(
    variant: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MPSState, "to_statevector", forbid_dense)
    weights = (
        native.CLOCK_AMPLITUDES
        if variant == 0
        else tuple(
            ((-1) ** i)
            * (i + variant)
            / math.sqrt(sum((j + variant) ** 2 for j in range(16)))
            for i in range(16)
        )
    )
    clock_phase = 0.0
    spin_phase = 0.0
    p = native.build_program(weights, clock_phase, spin_phase)
    assert p.execution.state_mode == "mps" and p.execution.state_bytes == 42496
    assert p.report["intermediate_mps_bond_bound"] == 4
    assert p.circuit.n_wires == 83
    expected = (
        fq.run(reference_circuit(weights, clock_phase, spin_phase))
        .statevector()
        .flatten()
    )
    torch.testing.assert_close(occupied_state(p), expected, atol=1e-12, rtol=0)
    text = ORIGINAL.read_text()
    match = re.search(r"inplace_prepare_amplitudes\((\[.*?\]), 0, x\)", text, re.S)
    assert match and tuple(ast.literal_eval(match[1])) == native.CLOCK_AMPLITUDES
    if variant:
        text = text[: match.start(1)] + repr(list(weights)) + text[match.end(1) :]
    source = (
        synth.synthesize(ORIGINAL, max_qubits=83, max_workspace=0)
        if variant == 0
        else synth.synthesize(
            text, source_format="qmod", max_qubits=83, max_workspace=0
        )
    )
    torch.testing.assert_close(occupied_state(source), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("clock_phase,spin_phase", [(0.37, -0.61), (-0.74, 1.22)])
def test_complex_clock_and_spin_coherence(
    clock_phase: float, spin_phase: float
) -> None:
    weights = tuple(
        ((-1) ** i) * (i + 1) / math.sqrt(sum(j * j for j in range(1, 17)))
        for i in range(16)
    )
    p = native.build_program(weights, clock_phase, spin_phase)
    expected = (
        fq.run(reference_circuit(weights, clock_phase, spin_phase))
        .statevector()
        .flatten()
    )
    torch.testing.assert_close(occupied_state(p), expected, atol=1e-12, rtol=0)


def body(text: str, name: str) -> str:
    start = re.search(r"\bq(?:func|perm) " + name + r"\([^)]*\)\s*\{", text)
    assert start
    depth = 1
    end = start.end()
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start.end() : end - 1]


def test_independent_preparation_orthogonality_and_select_guard() -> None:
    text = ORIGINAL.read_text()
    assign = body(text, "assign_aux")
    assert "aux1 = nu <= 31;" in assign and "aux2 = mu <= nu;" in assign
    assert "aux3 = mu > 2.5;" in assign and "aux4 = 1;" in assign
    # On zero addresses, m1=(nu==32)=0. The final CCX leaves aux4=1,
    # so all three final marker controls are true and inverse preparation
    # first flips succ from zero to one. All preceding gates preserve succ.
    nu = mu = 0
    m1 = int(nu == 32)
    flags = (int(nu <= 31), int(mu <= nu), 1 ^ (m1 & int(mu > 2.5)))
    assert flags == (1, 1, 1)
    early = body(text, "create_equal_superposition")
    assert early.count("X(ell.succ)") == 1
    assert early.endswith("  }\n")
    assert "ell.succ" not in early[: early.rfind("  within {")]
    # Inverting the later preparation preserves zero addresses and succ:
    # its last swap exchanges two zeros, alias comparison is 0<0, QROM
    # writes only other fields, and inverse address arithmetic sees 0,0.
    later = body(text, "prepare_after_equal_superposition")
    assert "aux = ell.keep < ell.sigma;" in later
    assert "ell.succ" not in later
    writes = set(
        re.findall(r"single_qrom_access\([^,]+, ell\.(\w+),", body(text, "in_s"))
    )
    assert writes and writes.isdisjoint({"mu", "nu", "succ", "m1", "sigma", "aux", "s"})
    selection = body(text, "select")
    assert "_ciz(ell.succ, psi.down[0]);" in selection
    assert "_cciz(ell.succ, ell.m1, psi.down[0]);" in selection
    assert selection.count("ell.succ") == 2
    for helper in ("in_mu", "in_nu", "controlled_basis_change"):
        assert "ell.succ" not in body(text, helper)
    # Explicit inverse-early simulation on the twelve-bit active space.
    # succ is bit11; mu/nu span bits0..9 and r_br is bit10.
    state = torch.zeros(4096, dtype=torch.complex128)
    state[1 << 11] = 1
    labels = torch.arange(4096)

    def single(bit: int, matrix: torch.Tensor) -> None:
        nonlocal state
        tensor = state.reshape(2 ** (11 - bit), 2, 2**bit)
        state = torch.einsum("ab,xby->xay", matrix, tensor).reshape(-1)

    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    for bit in range(10):
        single(bit, h)
    state[(labels & 1023) == 1023] *= torch.where(
        ((labels[(labels & 1023) == 1023] >> 10) & 1) == 1, -1, 1
    )
    for bit in range(10):
        single(bit, h)
    angle = 1.955751954173603
    ry = torch.tensor(
        [
            [math.cos(angle / 2), -math.sin(angle / 2)],
            [math.sin(angle / 2), math.cos(angle / 2)],
        ],
        dtype=torch.complex128,
    )
    single(10, ry)
    mark = ((labels & 31) <= ((labels >> 5) & 31)) & (((labels >> 10) & 1) == 1)
    state[mark] *= -1
    single(10, ry.mH)
    for bit in range(10):
        single(bit, h)
    assert float(state[:2048].abs().square().sum()) == 0
    assert float(state.abs().square().sum()) == pytest.approx(1, abs=1e-12)
    assert abs(complex(state[0])) == 0


def trace_clock(text: str, word: int) -> None:
    walk = body(text, "in_t")
    walk = re.sub(r"CCX\(\{([^}]+)\}, ([^)]+)\)", r"CCX(\1, \2)", walk)
    tokens = [
        token.strip()
        for token in re.split(r"([{};])", walk)
        if token.strip() and token != ";"
    ]
    values = {f"t[{bit}]": (word >> bit) & 1 for bit in range(4)}

    def block(
        index: int, stop: bool = False, execute: bool = True
    ) -> tuple[int, list[str]]:
        inverse: list[str] = []
        while index < len(tokens):
            token = tokens[index]
            index += 1
            if token == "}":
                assert stop
                return index, inverse
            if token == "within":
                assert tokens[index] == "{"
                index, undo = block(index + 1, True, execute)
                assert tokens[index : index + 2] == ["apply", "{"]
                index, _ = block(index + 2, True, execute)
                if execute:
                    for target in reversed(undo):
                        values[target] ^= 1
            elif token.startswith("X("):
                target = token[2:-1]
                if execute:
                    values[target] ^= 1
                inverse.append(target)
            elif token.endswith(": qbit"):
                continue
            elif token.startswith("allocate("):
                target = token[9:-1]
                assert target not in values
                if execute:
                    values[target] = 0
            elif token.startswith("free("):
                target = token[5:-1]
                if execute:
                    assert values.pop(target) == 0
            elif token.startswith(("CX(", "CCX(")):
                args = [
                    arg.strip() for arg in token[token.index("(") + 1 : -1].split(",")
                ]
                if execute:
                    values[args[-1]] ^= math.prod(values[arg] for arg in args[:-1])
            elif token.startswith(("select(", "controlled_reflect(")):
                # Independent guard/orthogonality proof above establishes both
                # calls preserve ell=0 for arbitrary spin/clock amplitudes.
                continue
            else:
                raise AssertionError(token)
        assert not stop
        return index, inverse

    block(0)
    assert values == {f"t[{bit}]": (word >> bit) & 1 for bit in range(4)}


@pytest.mark.parametrize("word", range(16))
def test_complete_clock_control_trace_and_cleanup(word: int) -> None:
    trace_clock(ORIGINAL.read_text(), word)


@pytest.mark.parametrize(
    "weights",
    [(1.0,), tuple([0.0] * 16), (float("nan"),) + native.CLOCK_AMPLITUDES[1:]],
)
def test_bad_clock_refuses(weights: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="amplitudes"):
        native.build_program(weights)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_qubits": 82},
        {"max_operations": 10},
        {"max_depth": 10},
        {"memory_limit_bytes": 1024},
    ],
)
def test_budgets_refuse(limits: Limits) -> None:
    with pytest.raises(
        (PlanningError, HybridVerificationError),
        match="memory|budget|limit|exceed|requires|infeasible",
    ):
        synth.synthesize(
            native.main,
            classical_inputs={
                "weights": list(native.CLOCK_AMPLITUDES),
                "clock_phase": 0.0,
                "spin_phase": 0.0,
            },
            **limits,
        )
