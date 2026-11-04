"""Check the pinned Classiq preparations against independent amplitudes."""

import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.algorithms.primitives.state_preparation import append_arbitrary_state
from flagquantum.synth import (
    Output,
    QNum,
    allocate,
    prepare_uniform_interval_state,
    qfunc,
)

pytestmark = pytest.mark.integration
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


@qfunc
def interval(q: Output[QNum]) -> None:
    allocate(5, q)
    prepare_uniform_interval_state(2, 8, q)


@qfunc
def trimmed(q: Output[QNum]) -> None:
    allocate(4, q)
    prepare_uniform_interval_state(0, 3, q)


@pytest.mark.parametrize(
    ("name", "width", "start", "stop", "native"),
    [("interval", 5, 2, 8, interval), ("trimmed", 4, 0, 3, trimmed)],
)
def test_uniform_original(name, width, start, stop, native):
    source = CORPUS / f"prepare_uniform_{name}_state.qmod"
    expected = torch.zeros(1 << width, dtype=torch.complex128)
    expected[start:stop] = 1 / math.sqrt(stop - start)
    for program in (
        synth.synthesize(source, source_format="qmod"),
        synth.synthesize(native),
    ):
        torch.testing.assert_close(
            fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
        )

    circuit = fq.Circuit(width, dtype=torch.complex128)
    append_arbitrary_state(circuit, expected, range(width))
    torch.testing.assert_close(
        fq.run(circuit).statevector().flatten(), expected, atol=1e-7, rtol=0
    )


@pytest.mark.parametrize(
    "body",
    [
        "allocate(4, q); prepare_uniform_interval_state(3, 3, q);",
        "allocate(4, q); prepare_uniform_interval_state(2, 17, q);",
        "allocate(9, q); prepare_uniform_trimmed_state(3, q);",
        "allocate(4, q); hadamard_transform(q); prepare_uniform_trimmed_state(3, q);",
    ],
)
def test_uniform_preparation_refuses_bad_bounds_or_nonfresh_target(body):
    with pytest.raises(ValueError, match="register.prepare"):
        synth.synthesize(f"qfunc main(output q:qnum) {{{body}}}", source_format="qmod")
