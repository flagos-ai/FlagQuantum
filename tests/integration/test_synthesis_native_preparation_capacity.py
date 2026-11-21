"""Fresh native vectors expand without broadening inplace or helper admission."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend.program import capture_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("size", [256, 512, 4096, 16384])
@pytest.mark.parametrize("probabilities", [False, True])
def test_fresh_vectors_preserve_every_complex_amplitude(
    size: int, probabilities: bool
) -> None:
    values = torch.arange(1, size + 1, dtype=torch.float64)
    if probabilities:
        values /= values.sum()
        expected = values.sqrt()
    else:
        values[::3] *= -1
        values /= values.norm()
        expected = values
    function = "prepare_state" if probabilities else "prepare_amplitudes"
    source = definition(
        "values:list[float], q:Output[QArray[QBit]]",
        f"{function}(values,0,q)",
    )
    bindings = values.tolist()
    program = synth.synthesize(
        source,
        classical_inputs={"values": bindings},
        max_qubits=int(math.log2(size)),
        max_workspace=0,
        max_operations=35000,
        memory_limit_bytes=64 * 1024**2,
    )
    bindings[0] = 12.0
    state = fq.run(program).statevector().flatten()
    assert state.numel() == size
    assert program.report["workspace_qubits"] == 0
    torch.testing.assert_close(state, expected.to(torch.complex128), atol=1e-12, rtol=0)


@pytest.mark.parametrize("size", [0, 1, 3, 32768])
def test_invalid_fresh_lengths_fail_before_lowering(size: int) -> None:
    source = definition(
        "values:list[float],q:Output[QNum]", "prepare_amplitudes(values,0,q)"
    )
    with pytest.raises(HybridVerificationError, match="budget|program.prepare"):
        capture_program(source, {"values": [0.0] * size})


@pytest.mark.parametrize("bad", [True, float("inf"), float("nan")])
def test_bound_vectors_validate_before_coercion(bad: float) -> None:
    source = definition(
        "values:list[float],q:Output[QNum]", "prepare_amplitudes(values,0,q)"
    )
    with pytest.raises(HybridVerificationError, match="program.binding"):
        capture_program(source, {"values": [bad, 0.0]})


def test_aggregate_bindings_include_scalars_and_snapshot_arrays() -> None:
    source = definition(
        "values:list[float], angle:float,q:Output[QBit]",
        "allocate(q)\nRX(angle,q)",
    )
    capture_program(source, {"values": [0.0] * 16383, "angle": 0.3})
    with pytest.raises(HybridVerificationError, match="budget"):
        capture_program(source, {"values": [0.0] * 16384, "angle": 0.3})


@pytest.mark.parametrize(
    "function", ["inplace_prepare_amplitudes", "inplace_prepare_state"]
)
def test_inplace_native_profile_stays_at_256(function: str) -> None:
    source = definition(
        "values:list[float],q:Output[QNum[9]]", f"allocate(q)\n{function}(values,0,q)"
    )
    with pytest.raises(HybridVerificationError, match="2..256"):
        capture_program(source, {"values": [1.0] + [0.0] * 511})


def test_helper_array_binding_stays_at_4096() -> None:
    helper = "@qfunc\ndef helper(values:list[float],q:QBit):\n    RX(values[0],q)\n"
    source = definition(
        "values:list[float],q:Output[QBit]", "allocate(q)\nhelper(values,q)", helper
    )
    capture_program(source, {"values": [0.0] * 4096})
    with pytest.raises(HybridVerificationError, match="4096"):
        capture_program(source, {"values": [0.0] * 4097})


@pytest.mark.parametrize("values", [[0.0, 0.0], [1.0, 1.0]])
def test_norm_validation_is_preserved(values: list[float]) -> None:
    source = definition(
        "values:list[float],q:Output[QNum]", "prepare_amplitudes(values,0,q)"
    )
    with pytest.raises(HybridVerificationError, match="normalized"):
        capture_program(source, {"values": values})


@pytest.mark.parametrize(
    "budgets",
    [{"max_qubits": 8}, {"max_operations": 1}, {"memory_limit_bytes": 1}],
)
def test_fresh_capacity_does_not_override_requested_budgets(
    budgets: dict[str, int],
) -> None:
    source = definition(
        "values:list[float],q:Output[QNum]", "prepare_amplitudes(values,0,q)"
    )
    with pytest.raises(
        (PlanningError, ValueError), match="budget|qubit|memory|resource"
    ):
        synth.synthesize(
            source, classical_inputs={"values": [1 / math.sqrt(512)] * 512}, **budgets
        )


def test_qmod_vector_profile_stays_at_256() -> None:
    vector = ",".join(["1"] + ["0"] * 511)
    with pytest.raises(ValueError, match="256"):
        synth.synthesize(
            f"qfunc main(output q:qnum) {{prepare_amplitudes([{vector}],0,q);}}",
            source_format="qmod",
        )
