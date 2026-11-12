"""Native DQI full states, retained model data and nonzero coherent decoder XOR."""

import math
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat import (
    main as native,
)
from examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat.comparisons.circuit import (
    decoder_circuit,
    reference_circuit,
)
from examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat.data import (
    PARITY_COLUMNS,
    SIGNS,
    SYNDROME_TABLE,
    WEIGHTS,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    H,
    Output,
    QArray,
    QBit,
    R,
    adjoint,
    allocate,
    phase,
    qfunc,
)

pytestmark = pytest.mark.integration
ORIGINAL = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/algorithms/search_and_optimization/dqi/dqi_max_xorsat.qmod"
)


def table_inputs() -> dict[str, list[int]]:
    return {
        "syndrome_bits": [
            (word >> bit) & 1 for word, _ in SYNDROME_TABLE for bit in range(6)
        ],
        "mask_bits": [
            (mask >> bit) & 1 for _, mask in SYNDROME_TABLE for bit in range(6)
        ],
    }


def main_inputs() -> dict[str, list[int] | list[float]]:
    return table_inputs() | {
        "amplitudes": native.symmetric_amplitudes(WEIGHTS),
        "matrix": [
            (column >> row) & 1 for row in range(6) for column in PARITY_COLUMNS
        ],
        "signs": list(SIGNS),
    }


def verify(program: synth.Program, expected: torch.Tensor) -> torch.Tensor:
    raw = fq.run(program).statevector()
    assert isinstance(raw, torch.Tensor)
    state = raw.reshape(4096, -1)
    torch.testing.assert_close(state[:, 0], expected.flatten(), atol=1e-10, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-22
    return state[:, 0]


@pytest.mark.parametrize(
    "weights",
    [WEIGHTS, (0.3, -0.4, math.sqrt(0.75), 0.0), (-0.6, 0.3, math.sqrt(0.55), 0.0)],
)
def test_full_native_source_and_independent_states(weights: tuple[float, ...]) -> None:
    expected = fq.run(reference_circuit(weights)).statevector()
    program = native.build_program(weights)
    state = verify(program, expected)
    assert len(program.registers["y"]) == len(program.registers["solution"]) == 6
    assert float(state.reshape(64, 64)[1:].abs().square().sum()) < 1e-22
    if weights == WEIGHTS:
        source_program = synth.synthesize(ORIGINAL)
    else:
        source = ORIGINAL.read_text()
        start = source.index("[", source.index("  prepare_amplitudes("))
        end = source.index("]", start) + 1
        source = source[:start] + repr(list(weights)) + source[end:]
        source_program = synth.synthesize(source, source_format="qmod")
    verify(source_program, expected)


def test_native_model_retains_original_columns_signs_weights_and_lookup() -> None:
    source = ORIGINAL.read_text()
    columns = [0] * 6
    for row, parity in re.findall(r"out\[(\d+)\] \^= (.*?);", source):
        for column in re.findall(r"y\[(\d+)\]", parity):
            columns[int(column)] |= 1 << int(row)
    assert tuple(columns) == PARITY_COLUMNS
    observed = tuple(
        (int(word), int(mask))
        for word, mask in re.findall(
            r"control \(syndrome == (\d+)\) \{\s*error \^= (\d+);", source
        )
    )
    assert observed == SYNDROME_TABLE and len(observed) == 22
    literal = re.findall(
        r"prepare_amplitudes\((\[[^\]]+\]), 0, k_num_errors\);", source
    )[0]
    assert (
        tuple(
            float(x) for x in re.findall(r"[-+]?\d+(?:\.\d*)?(?:e[-+]?\d+)?", literal)
        )
        == WEIGHTS
    )
    literal = re.findall(
        r"vector_product_phase_expanded___0\((\[[0-9, ]+\]), y\);", source
    )[0]
    assert tuple(int(x) for x in re.findall(r"\d+", literal)) == SIGNS


@qfunc
def prepare_complex_error(phi: float, y: QArray[QBit], solution: QArray[QBit]) -> None:
    for bit in range(6):
        R(0.23 * (bit + 1), math.pi / 2, y[bit])
        phase(y[bit], phi * (bit + 1))
        H(solution[bit])


@qfunc
def decoder_main(
    phi: float,
    syndrome_bits: list[int],
    mask_bits: list[int],
    y: Output[QArray[QBit]],
    solution: Output[QArray[QBit]],
) -> None:
    allocate(6, y)
    allocate(6, solution)
    prepare_complex_error(phi, y, solution)
    native.decode(syndrome_bits, mask_bits, y, solution)


@qfunc
def decoder_roundtrip(
    phi: float,
    syndrome_bits: list[int],
    mask_bits: list[int],
    y: Output[QArray[QBit]],
    solution: Output[QArray[QBit]],
) -> None:
    allocate(6, y)
    allocate(6, solution)
    prepare_complex_error(phi, y, solution)
    native.decode(syndrome_bits, mask_bits, y, solution)
    with adjoint():
        native.decode(syndrome_bits, mask_bits, y, solution)


def initial_reference(phi: float) -> fq.Circuit:
    circuit = fq.Circuit(12, dtype=torch.complex128)
    for bit in range(6):
        circuit.ry(5 - bit, theta=0.23 * (bit + 1)).phase(
            5 - bit, theta=phi * (bit + 1)
        ).h(11 - bit)
    return circuit


@pytest.mark.parametrize("phi", [0.37, -0.61])
def test_decoder_xor_on_nonzero_complex_error_and_coherent_syndromes(
    phi: float,
) -> None:
    bindings = table_inputs() | {"phi": phi}
    program = synth.synthesize(
        decoder_main, classical_inputs=bindings, memory_limit_bytes=64 * 1024**2
    )
    initial = fq.run(initial_reference(phi)).statevector().reshape(64, 64)
    expected = torch.zeros_like(initial)
    lookup = dict(SYNDROME_TABLE)
    for word in range(64):
        for syndrome in range(64):
            expected[word ^ lookup.get(syndrome, 0), syndrome] = initial[word, syndrome]
    reference = initial_reference(phi)
    decoder_circuit(reference)
    torch.testing.assert_close(
        fq.run(reference).statevector().flatten(),
        expected.flatten(),
        atol=1e-12,
        rtol=0,
    )
    verify(program, expected)


@pytest.mark.parametrize("phi", [0.37, -0.61])
def test_checked_decoder_inverse_restores_all_complex_amplitudes(phi: float) -> None:
    program = synth.synthesize(
        decoder_roundtrip,
        classical_inputs=table_inputs() | {"phi": phi},
        memory_limit_bytes=64 * 1024**2,
    )
    verify(program, fq.run(initial_reference(phi)).statevector())


@qfunc
def aliased_decoder(
    syndrome_bits: list[int], mask_bits: list[int], q: Output[QArray[QBit]]
) -> None:
    allocate(6, q)
    native.decode(syndrome_bits, mask_bits, q, q)


def test_decoder_alias_refuses_before_lowering() -> None:
    with pytest.raises(HybridVerificationError, match="disjoint writable"):
        synth.synthesize(aliased_decoder, classical_inputs=table_inputs())


@pytest.mark.parametrize(
    "weights",
    [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (math.nan, 0.0, 0.0, 0.0),
        (math.inf, 0.0, 0.0, 0.0),
        (True, 0.0, 0.0, 0.0),
        (1.0, 0.0),
    ],
)
def test_invalid_weights_refuse(weights: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="weights|weight|amplitudes"):
        native.build_program(weights)


@pytest.mark.parametrize("kind", ["qubits", "operations", "depth", "memory"])
def test_budget_refusals(kind: str) -> None:
    bindings = main_inputs()
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if kind == "qubits":
            synth.synthesize(native.main, classical_inputs=bindings, max_qubits=11)
        elif kind == "operations":
            synth.synthesize(native.main, classical_inputs=bindings, max_operations=1)
        elif kind == "depth":
            synth.synthesize(native.main, classical_inputs=bindings, max_depth=1)
        else:
            synth.synthesize(
                native.main, classical_inputs=bindings, memory_limit_bytes=16
            )


def test_scratch_free_dqi_preserves_the_independent_state() -> None:
    program = synth.synthesize(
        native.main, classical_inputs=main_inputs(), max_workspace=0, max_qubits=12
    )
    assert program.report["workspace_qubits"] == 0
    verify(program, fq.run(reference_circuit(WEIGHTS)).statevector())
