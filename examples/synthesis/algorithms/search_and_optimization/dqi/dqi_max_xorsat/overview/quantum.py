"""Reuse the original native DQI kernel with freshly generated classical data."""

from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    X,
    allocate,
    inplace_prepare_amplitudes,
    qfunc,
)

from ..main import main, symmetric_amplitudes
from .model import SIGNS, DQIModel, dicke_amplitudes


@qfunc
def unary_state(count: int, result: Output[QArray[QBit]]) -> None:
    allocate(15, result)
    for bit in range(count):
        X(result[bit])


@qfunc
def dicke_state(
    unary_weight: int, values: list[float], result: Output[QArray[QBit]]
) -> None:
    allocate(6, result)
    for bit in range(unary_weight):
        X(result[bit])
    apply_dicke_to_unary(unary_weight, values, result)


@qfunc
def apply_dicke_to_unary(
    unary_weight: int, values: list[float], target: QArray[QBit]
) -> None:
    for bit in range(unary_weight):
        X(target[bit])
    inplace_prepare_amplitudes(values, 0, target)


def build_unary(count: int = 8) -> synth.Program:
    """Prepare the source's 15-bit unary word for an admitted classical count."""
    if type(count) is not int or not 0 <= count <= 15:
        raise ValueError("The unary count must be an integer in 0..15")
    return synth.synthesize(
        unary_state, classical_inputs={"count": count}, max_qubits=15, max_workspace=0
    )


def build_dicke(weight: int, *, use_unary: bool) -> synth.Program:
    """Prepare either the direct or known-unary six-bit source example."""
    return synth.synthesize(
        dicke_state,
        classical_inputs={
            "unary_weight": weight if use_unary else 0,
            "values": dicke_amplitudes(weight),
        },
        max_qubits=6,
        max_workspace=0,
    )


def build_dqi(model: DQIModel) -> synth.Program:
    """Evaluate the coherent encoder and decoder for the fresh source incidence table."""
    return synth.synthesize(
        main,
        classical_inputs={
            "amplitudes": symmetric_amplitudes(model.weights),
            "matrix": model.matrix,
            "signs": SIGNS,
            "syndrome_bits": tuple(
                word >> bit & 1 for word, _ in model.syndrome_table for bit in range(6)
            ),
            "mask_bits": tuple(
                mask >> bit & 1 for _, mask in model.syndrome_table for bit in range(6)
            ),
        },
        max_qubits=15,
        max_workspace=3,
        memory_limit_bytes=64 * 1024**2,
    )
