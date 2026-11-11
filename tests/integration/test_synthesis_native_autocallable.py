"""Complete native/source/independent-Circuit autocallable acceptance."""

import hashlib
import math
from itertools import product
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.autocallable_options.quantum_autocallable_option_pricing import (
    main as entry,
)
from examples.synthesis.applications.finance.autocallable_options.quantum_autocallable_option_pricing.comparisons.circuit import (
    prepared_state,
    reference_state,
)
from examples.synthesis.applications.finance.autocallable_options.quantum_autocallable_option_pricing.physics import (
    DISTRIBUTION,
    EXPONENTIAL,
    predicate_bounds,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.errors import PlanningError
from flagquantum.services.synthesis import _synthesize
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
ORIGINAL = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/applications/finance/autocallable_options/quantum_autocallable_option_pricing.qmod"
)


def test_source_identity() -> None:
    assert (
        hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
        == "eeb66f5e3446a3498ece2269b97599cc6d0ef2d8895265276ecf605872bce386"
    )


@pytest.mark.parametrize("shift", [0.1274075, 0.22])
def test_all_grid_branches_and_strict_boundaries(shift: float) -> None:
    first, second, final, barrier0, barrier1, barrier2, bias = predicate_bounds(shift)
    offset = math.trunc((shift - 3 * 0.238268134860707) * 4)
    for x0, x1, x2 in product(range(4), repeat=3):
        a, b, c = x0 + offset, x0 + x1 + 2 * offset, x0 + x1 + x2 + 3 * offset
        assert (x0 > first) == (a > 0)
        assert (x0 + x1 > second) == (b > 0)
        assert (x0 + x1 + x2 < final) == (c < 0)
        assert (x0 < barrier0) == (a < -1)
        assert (x0 + x1 < barrier1) == (b < -1)
        assert (x0 + x1 + x2 < barrier2) == (c < -1)
        if c < 0:
            for reference in range(16):
                assert (x0 + x1 + x2 + bias >= reference) == (c + 8 >= reference)


@pytest.mark.parametrize("shift", [0.1274075, 0.22])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_complete_native_source_and_circuit(k: int, shift: float) -> None:
    program = entry.build_program(k, shift=shift)
    result = fq.run(program)
    actual = result.statevector().reshape(2048, -1)
    expected = reference_state(k, shift=shift)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-20
    if shift == 0.1274075:
        original = _synthesize(ORIGINAL, classical_inputs={"k": k})
    else:
        source = ORIGINAL.read_text().replace("0.1274075 +", "0.22 +")
        original = _synthesize(
            source,
            source_format="qmod",
            qmod_machine_precision=2,
            classical_inputs={"k": k},
        )
    source_state = fq.run(original).statevector().reshape(2, 16, 4, 4, 4)
    canonical = source_state.permute(0, 4, 3, 2, 1).flatten()
    torch.testing.assert_close(canonical, expected, atol=1e-10, rtol=0)
    density = actual[:, 0].reshape(2, 1024)
    independent = expected.reshape(2, 1024)
    torch.testing.assert_close(
        density @ density.mH, independent @ independent.mH, atol=1e-10, rtol=0
    )
    with pytest.raises(ValueError, match="readout budget exceeded"):
        program.probabilities(result, registers=("indicator",))
    probabilities = dict(
        program.probabilities(result, registers=("indicator",), max_amplitudes=2**24)
    )
    assert probabilities[(1,)] == pytest.approx(
        float(independent[1].abs().square().sum()), abs=1e-10
    )
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 24
    assert resources.n_operations <= 5000


@pytest.mark.parametrize("k", [-1, 5, True, 1.5])
def test_unbounded_or_noninteger_iterations_refused(k: int) -> None:
    with pytest.raises(ValueError, match="integer"):
        entry.build_program(k)


@pytest.mark.parametrize("shift", [float("nan"), float("inf"), 0.0, True])
def test_unvalidated_model_parameters_refused(shift: float) -> None:
    with pytest.raises(ValueError, match="Shift|Validated"):
        predicate_bounds(shift)


PREPARE_IMPORT = (
    "from examples.synthesis.applications.finance.autocallable_options."
    "quantum_autocallable_option_pricing.operations import prepare\n"
)
PREPARE_BODY = """x0: QNum = QNum()
x1: QNum = QNum()
x2: QNum = QNum()
aux: QNum = QNum()
allocate(2, x0)
allocate(2, x1)
allocate(2, x2)
allocate(4, aux)
allocate(indicator)
if marker == 1:
    X(indicator)
else:
    R(0.73, 1.5707963267948966, indicator)
    RZ(0.41, indicator)
prepare(probabilities, exponential, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5], bounds[6], x0, x1, x2, aux, indicator)
drop(x0)
drop(x1)
drop(x2)
drop(aux)
"""


@pytest.mark.parametrize("shift", [0.1274075, 0.22])
@pytest.mark.parametrize("marker", [1, 2])
def test_nonzero_and_coherent_marker_xor(marker: int, shift: float) -> None:
    function = definition(
        "probabilities:list[float], exponential:list[float], bounds:list[int], "
        "marker:int, indicator:Output[QBit]",
        PREPARE_BODY,
        PREPARE_IMPORT,
    )
    program = synth.synthesize(
        function,
        classical_inputs={
            "probabilities": list(DISTRIBUTION),
            "exponential": list(EXPONENTIAL),
            "bounds": predicate_bounds(shift),
            "marker": marker,
        },
        max_qubits=24,
        memory_limit_bytes=1024**3,
    )
    actual = fq.run(program).statevector().reshape(2048, -1)
    expected = prepared_state(shift=shift, marker=marker)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-20


def test_prepare_rejects_mutable_argument_aliases() -> None:
    body = PREPARE_BODY.replace(
        "x0, x1, x2, aux, indicator)", "x0, x0, x2, aux, indicator)"
    )
    function = definition(
        "probabilities:list[float], exponential:list[float], bounds:list[int], "
        "marker:int, indicator:Output[QBit]",
        body,
        PREPARE_IMPORT,
    )
    with pytest.raises(HybridVerificationError, match="disjoint writable locations"):
        synth.synthesize(
            function,
            classical_inputs={
                "probabilities": list(DISTRIBUTION),
                "exponential": list(EXPONENTIAL),
                "bounds": predicate_bounds(0.1274075),
                "marker": 1,
            },
        )


def test_conditional_output_cannot_escape_a_helper() -> None:
    function = definition(
        "indicator:Output[QBit]",
        "q:QNum=QNum()\nallocate(1,q)\n"
        "control(q==0,lambda: produce(indicator))\ndrop(q)",
        "@qfunc\ndef produce(target:Output[QBit]):\n    allocate(target)\n",
    )
    with pytest.raises(HybridVerificationError, match="Const-only ports"):
        synth.synthesize(function)


@pytest.mark.parametrize(
    ("limits", "error_type"),
    [
        ({"max_qubits": 10}, synth.SynthesisError),
        ({"max_workspace": 0}, synth.SynthesisError),
        ({"max_operations": 1}, synth.SynthesisError),
        ({"memory_limit_bytes": 1024}, PlanningError),
    ],
)
def test_full_model_resource_budgets(
    limits: dict[str, int], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type, match="qubits|workspace|operations|memory"):
        synth.synthesize(
            entry.main,
            classical_inputs={
                "k": 0,
                "probabilities": list(DISTRIBUTION),
                "exponential": list(EXPONENTIAL),
                "bounds": predicate_bounds(0.1274075),
            },
            max_qubits=limits.get("max_qubits", 24),
            max_workspace=limits.get("max_workspace", 256),
            max_operations=limits.get("max_operations", 100_000),
            memory_limit_bytes=limits.get("memory_limit_bytes", 1024**3),
        )
