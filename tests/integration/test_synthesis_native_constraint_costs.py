"""Complete native dominating/set-cover programs retain nonbinary slack weights."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.minimum_dominating_set import (
    main as dominating,
)
from examples.synthesis.applications.optimization.minimum_dominating_set.comparisons import (
    circuit as dominating_ref,
)
from examples.synthesis.applications.optimization.set_cover import main as covering
from examples.synthesis.applications.optimization.set_cover.comparisons import (
    circuit as covering_ref,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.factor_reference import factored_cost_circuit
from examples.synthesis.quadratic_reference import (
    affine_square_factors,
    order_register_factors,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._construction.frontend_python import _QuantumFunction

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Case:
    name: str
    native: _QuantumFunction
    bindings: tuple[tuple[float, ...], ...]
    widths: tuple[int, ...]
    residuals: tuple[tuple[tuple[int, ...], tuple[float, ...]], ...]
    slack_weights: tuple[tuple[float, ...], ...]
    reference: Callable[[tuple[float, ...]], fq.Circuit]
    qubits: int
    source_fields: tuple[str, ...]


CASES = (
    Case(
        "minimum_dominating_set",
        dominating.main,
        dominating_ref.BINDINGS,
        dominating_ref.WIDTHS,
        tuple(dominating_ref.residuals()),
        dominating_ref.SLACK_WEIGHTS,
        dominating_ref.reference_circuit,
        20,
        ("v.x",) + tuple(f"v.dominating_rule_{i}_slack_var" for i in range(6)),
    ),
    Case(
        "set_cover",
        covering.main,
        covering_ref.BINDINGS,
        covering_ref.WIDTHS,
        tuple(covering_ref.residuals()),
        covering_ref.SLACK_WEIGHTS,
        covering_ref.reference_circuit,
        23,
        ("v.x",) + tuple(f"v.independent_rule_{i}_slack_var" for i in range(1, 11)),
    ),
)


@pytest.mark.parametrize("width", range(1, 9))
def test_affine_square_reference_matches_every_local_classical_input(
    width: int,
) -> None:
    coefficients = tuple(
        1.0 if i % 3 == 0 else -2.0 if i % 3 == 1 else 3.0 for i in range(width)
    )
    factors = affine_square_factors(tuple(range(width)), coefficients, 1.25, -0.5)
    assert max(len(bits) for bits, _ in factors) <= 2
    for basis in range(1 << width):
        bits = tuple((basis >> i) & 1 for i in range(width))
        actual = sum(
            table[sum(bits[bit] << i for i, bit in enumerate(support))]
            for support, table in factors
        )
        expected = (
            -0.5
            * (1.25 + sum(c * b for c, b in zip(coefficients, bits, strict=True))) ** 2
        )
        assert actual == expected


def test_two_bit_factor_reference_preserves_dense_phase_and_mixer_order() -> None:
    coefficients = (1.0, -2.0, 0.5, 1.5)
    factors = affine_square_factors((0, 1, 2, 3), coefficients, -0.75, -0.6)
    params = (0.37, -0.41, 0.29, 0.63)
    reference = factored_cost_circuit(4, factors, params)
    costs = torch.tensor(
        [
            -0.6
            * (-0.75 + sum(c * ((basis >> i) & 1) for i, c in enumerate(coefficients)))
            ** 2
            for basis in range(16)
        ],
        dtype=torch.float64,
    )
    dense = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        dense.h(wire)
    for gamma, beta in zip(params[::2], params[1::2], strict=True):
        dense.any(*range(4), matrix=torch.diag(torch.exp(1j * gamma * costs)))
        for wire in range(4):
            dense.rx(wire, theta=beta)
    torch.testing.assert_close(
        fq.run(reference).statevector(), fq.run(dense).statevector(), atol=1e-12, rtol=0
    )


def test_asymmetric_field_layout_matches_direct_dense_cost() -> None:
    coefficients = (1.0, -2.0, 0.5, 1.5)
    factors = affine_square_factors((0, 1, 2, 3), coefficients, -0.75, -0.6)
    ordered = order_register_factors(factors, (2, 1, 1))
    reference = factored_cost_circuit(4, ordered, (0.37, -0.41))
    costs = []
    for basis in range(16):
        fields = ((basis >> 2) & 1, (basis >> 3) & 1, (basis >> 1) & 1, basis & 1)
        costs.append(
            -0.6
            * (-0.75 + sum(c * b for c, b in zip(coefficients, fields, strict=True)))
            ** 2
        )
    phases = torch.exp(0.37j * torch.tensor(costs, dtype=torch.float64))
    dense = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        dense.h(wire)
    dense.any(*range(4), matrix=torch.diag(phases))
    for wire in range(4):
        dense.rx(wire, theta=-0.41)
    torch.testing.assert_close(
        fq.run(reference).statevector(), fq.run(dense).statevector(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_each_residual_preserves_original_slack_weights(case: Case) -> None:
    for support, coefficients in case.residuals:
        factors = affine_square_factors(support, coefficients, 1.0, -10.0)
        for basis in range(1 << len(support)):
            values = {bit: (basis >> i) & 1 for i, bit in enumerate(support)}
            actual = sum(
                table[sum(values[bit] << i for i, bit in enumerate(subset))]
                for subset, table in factors
            )
            assert (
                actual
                == -10
                * (
                    1
                    + sum(
                        c * values[bit]
                        for bit, c in zip(support, coefficients, strict=True)
                    )
                )
                ** 2
            )
    if case.name == "minimum_dominating_set":
        assert case.slack_weights[2] == case.slack_weights[4] == (1.0, 2.0, 1.0)
    else:
        assert all(weights == (1.0,) * len(weights) for weights in case.slack_weights)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dataset", [0, 1])
@pytest.mark.slow
def test_full_source_and_native_programs_match_independent_circuit(
    case: Case, dataset: int
) -> None:
    params = case.bindings[dataset]
    native = synth.synthesize(
        case.native, classical_inputs={"params": params}, max_workspace=0
    )
    source = synth.synthesize(
        ROOT
        / f"tests/fixtures/classiq_qmod/corpus/original/applications/optimization/{case.name}/{case.name}.qmod",
        classical_inputs={"params": params},
        max_workspace=0,
    )
    reference = case.reference(params)
    registers = ("x",) + tuple(f"s{i}" for i in range(len(case.widths)))
    compare_states(native, reference, registers)
    compare_states(source, reference, case.source_fields)
    for program in (native, source):
        resource = program.report["resources"]
        assert isinstance(resource, ResourceEstimate)
        assert resource.n_qubits == case.qubits
        assert resource.max_operation_width <= 2
        assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dataset", [0, 1])
def test_objectives_preserve_complete_instruction_sequences(
    case: Case, dataset: int
) -> None:
    params = case.bindings[dataset]
    width = synth.synthesize(
        case.native,
        classical_inputs={"params": params},
        objective="width",
        max_workspace=0,
    )
    depth = synth.synthesize(
        case.native,
        classical_inputs={"params": params},
        objective="depth",
        max_workspace=0,
    )
    assert width.circuit.instructions == depth.circuit.instructions
    assert width.report["resources"] == depth.report["resources"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_actual_full_width_and_operation_budgets_are_enforced(case: Case) -> None:
    params = case.bindings[0]
    with pytest.raises(ValueError):
        synth.synthesize(
            case.native,
            classical_inputs={"params": params},
            max_qubits=case.qubits - 1,
        )
    with pytest.raises(ValueError):
        synth.synthesize(
            case.native, classical_inputs={"params": params}, max_operations=1
        )
    with pytest.raises(ValueError):
        synth.synthesize(
            case.native, classical_inputs={"params": params}, memory_limit_bytes=1
        )
