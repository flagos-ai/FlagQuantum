"""Native feature encoders preserve original shapes, bit order and phase."""

from pathlib import Path
from typing import Literal

import pytest

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.data_encoding.encode_in_angle.comparisons.circuit import (
    reference_circuit as angle_reference,
)
from examples.synthesis.functions.data_encoding.encode_in_angle.main import (
    build_program as build_angle,
)
from examples.synthesis.functions.data_encoding.encode_on_bloch.comparisons.circuit import (
    reference_circuit as bloch_reference,
)
from examples.synthesis.functions.data_encoding.encode_on_bloch.main import (
    build_program as build_bloch,
)
from flagquantum import synth

pytestmark = pytest.mark.integration
SOURCES = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/functions/qmod_library_reference/classiq_open_library/variational_data_encoding"
)


@pytest.mark.parametrize(
    ("kind", "data"),
    [
        ("angle", [0.2, -0.3, 0.4, 0.5]),
        ("angle", [0.17, 2.63, -2.49, 0.91]),
        ("bloch", [0.2, -0.3, 0.4, 0.5, 0.6, -0.7, 0.8]),
        ("bloch", [0.17, 2.63, -2.49, 0.91, -0.25, 0.37, 1.58]),
    ],
)
def test_two_nonzero_bindings_match_original_and_independent_circuit(
    kind: Literal["angle", "bloch"], data: list[float]
) -> None:
    build = build_angle if kind == "angle" else build_bloch
    reference = angle_reference if kind == "angle" else bloch_reference
    name = "encode_in_angle" if kind == "angle" else "encode_on_bloch"
    native = build(data)
    original = synth.synthesize(
        SOURCES / f"{name}.qmod", classical_inputs={"data": data}
    )
    for program in (native, original):
        compare_states(program, reference(data), ("x",))
        assert tuple(program.registers) == ("x",)
        assert len(program.registers["x"]) == 4
        assert program.report["workspace_qubits"] == 0
    assert dict(
        native.probabilities(fq.run(native), registers=("x",))
    ) == pytest.approx(
        dict(original.probabilities(fq.run(original), registers=("x",))), abs=1e-12
    )


@pytest.mark.parametrize("size", [0, 1, 3, 5, 6, 8])
def test_original_feature_shapes_reject_missing_or_extra_values(size: int) -> None:
    for build in (build_angle, build_bloch):
        with pytest.raises(ValueError, match="exactly"):
            build([0.1] * size)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_encoders_reuse_finite_plain_real_validation(value: float) -> None:
    for build, size in ((build_angle, 4), (build_bloch, 7)):
        with pytest.raises(ValueError, match="finite reals"):
            build([value] + [0.1] * (size - 1))
