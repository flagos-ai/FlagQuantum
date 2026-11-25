"""Compilation-local analysis reuse preserves mutation and ownership boundaries."""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import compile
from flagquantum.compiler.optimization import commutation, pipeline
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter
from tests.unit.test_native_block_optimization import _operator

pytestmark = pytest.mark.unit


def _program(theta: object = 0.2) -> CircuitIR:
    return CircuitIR(
        2,
        (
            Instruction("rz", (0,), {"theta": theta}),
            Instruction("cz", (0, 1)),
            Instruction("rz", (0,), {"theta": 0.3}),
        ),
    )


def test_identical_inputs_share_only_inside_one_optimization_call() -> None:
    program = _program()
    with patch.object(
        commutation, "_sweep_commutation", wraps=commutation._sweep_commutation
    ) as sweep:
        first = commutation.analyze_commutation(program)
        assert commutation.analyze_commutation(program) == first
        assert sweep.call_count == 2
        with commutation._reuse_commutation_analysis():
            scoped = commutation.analyze_commutation(program)
            assert commutation.analyze_commutation(program) is scoped
            assert sweep.call_count == 3
        commutation.analyze_commutation(program)
        assert sweep.call_count == 4
        with commutation._reuse_commutation_analysis():
            assert commutation.analyze_commutation(program) == scoped
            assert sweep.call_count == 5


@pytest.mark.parametrize(
    "theta", [Parameter("theta"), torch.tensor([0.2, 0.5], requires_grad=True)]
)
def test_parameter_values_are_not_read_or_retained(theta: object) -> None:
    with commutation._reuse_commutation_analysis():
        first = commutation.analyze_commutation(_program(theta))
        assert commutation.analyze_commutation(_program(0.7)) is first
    if isinstance(theta, torch.Tensor):
        assert theta.grad is None and theta.requires_grad


def test_mutating_parameter_presence_invalidates_the_structural_key() -> None:
    gate = Instruction("x", (0,))
    program = CircuitIR(1, (gate, Instruction("x", (0,))))
    with commutation._reuse_commutation_analysis():
        initial = commutation.analyze_commutation(program)
        assert isinstance(gate.params, dict)
        gate.params["not_a_declared_parameter"] = 0.5
        changed = commutation.analyze_commutation(program)
        assert changed != initial
        assert changed.blocks_by_qubit[0] == ((0,), (1,))
    assert changed == commutation.analyze_commutation(program)


@pytest.mark.parametrize("mutation", ["order", "qubits", "matrix", "opcode"])
def test_structural_changes_recompute_the_analysis(mutation: str) -> None:
    original = _program()
    gates = list(original.instructions)
    if mutation == "order":
        gates.insert(1, Instruction("h", (0,)))
    elif mutation == "qubits":
        gates[1] = Instruction("cx", (1, 0))
    elif mutation == "matrix":
        gates[0] = replace(gates[0], matrix=torch.eye(2))
    else:
        gates[0] = Instruction("rx", (0,), {"theta": 0.2})
    changed = replace(original, instructions=tuple(gates))
    with commutation._reuse_commutation_analysis():
        previous = commutation.analyze_commutation(original)
        actual = commutation.analyze_commutation(changed)
        assert actual is not previous
    assert actual == commutation.analyze_commutation(changed)


def test_changed_checker_does_not_inherit_another_checker_result() -> None:
    program = _program()
    with commutation._reuse_commutation_analysis():
        normal = commutation.analyze_commutation(program)
        with patch.object(commutation, "commute", return_value=False):
            refused = commutation.analyze_commutation(program)
        assert refused != normal
        assert commutation.analyze_commutation(program) == normal


def test_nested_and_expired_copied_contexts_do_not_retain_results() -> None:
    program = _program()
    with commutation._reuse_commutation_analysis():
        outer = commutation.analyze_commutation(program)
        copied = copy_context()
        with commutation._reuse_commutation_analysis():
            assert commutation.analyze_commutation(program) is not outer
        assert commutation.analyze_commutation(program) is outer
    assert copied.run(commutation.analyze_commutation, program) is not outer
    assert copied.run(commutation.analyze_commutation, program) is not copied.run(
        commutation.analyze_commutation, program
    )


def test_exception_closes_the_optimization_scope() -> None:
    program = _program()
    with (
        pytest.raises(RuntimeError, match="deliberate pass failure"),
        commutation._reuse_commutation_analysis(),
    ):
        scoped = commutation.analyze_commutation(program)
        raise RuntimeError("deliberate pass failure")
    assert commutation.analyze_commutation(program) is not scoped


def test_threaded_compilations_keep_independent_scopes() -> None:
    program = _program()

    def analyze() -> tuple[object, object]:
        with commutation._reuse_commutation_analysis():
            first = commutation.analyze_commutation(program)
            return first, commutation.analyze_commutation(program)

    with ThreadPoolExecutor(max_workers=2) as pool:
        pairs = list(pool.map(lambda _: analyze(), range(2)))
    assert all(a is b for a, b in pairs)
    assert pairs[0][0] is not pairs[1][0]


def test_default_workflow_matches_an_uncached_replacement_and_operator() -> None:
    source = (
        fq.Circuit(3).h(0).rz(0, theta=0.2).cx(0, 1).h(0).rz(0, theta=0.3).cx(0, 1).h(2)
    )
    with patch.object(
        commutation, "_sweep_commutation", wraps=commutation._sweep_commutation
    ) as sweep:
        current = compile(source)
        reused_sweeps = sweep.call_count
    with (
        patch.object(pipeline, "_reuse_commutation_analysis", nullcontext),
        patch.object(
            commutation, "_sweep_commutation", wraps=commutation._sweep_commutation
        ) as sweep,
    ):
        uncached = compile(source)
        uncached_sweeps = sweep.call_count
    assert reused_sweeps < uncached_sweeps
    assert current.to_dict() == uncached.to_dict()
    torch.testing.assert_close(
        _operator(current), _operator(source.to_ir()), atol=1e-12, rtol=0
    )
    assert compile(source).to_dict() == current.to_dict()


def test_memo_does_not_keep_input_tensors_alive() -> None:
    with commutation._reuse_commutation_analysis():
        theta = torch.tensor(0.2, requires_grad=True)
        observed = weakref.ref(theta)
        program = _program(theta)
        commutation.analyze_commutation(program)
        del program, theta
        gc.collect()
        assert observed() is None
