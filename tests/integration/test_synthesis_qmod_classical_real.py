"""Verify bound Qmod angles and borrowed array gates against a Circuit."""

import hashlib
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.ecosystem import InteropConversionError
from flagquantum.services.synthesis import _synthesize

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/qml_with_classiq_guide.qmod"
)


def test_original_qml_angles_match_combined_rx():
    raw = SOURCE.read_bytes()
    blob = f"blob {len(raw)}\0".encode() + raw
    assert hashlib.sha1(blob).hexdigest() == "d0879a1decbdc9d1ef8a9045d24901b97dd090f7"
    input_0, weight_0 = 0.37, -0.81
    program = _synthesize(
        SOURCE, classical_inputs={"input_0": input_0, "weight_0": weight_0}
    )
    reference = fq.Circuit(1, dtype=torch.complex128).rx(0, input_0 + weight_0)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    "bindings",
    [
        None,
        {"input_0": 0.37},
        {"input_0": 0.37, "weight_0": math.inf},
        {"input_0": True, "weight_0": 0.2},
        {"input_0": 0.37, "weight_0": 0.2, "extra": 1.0},
    ],
)
def test_classical_bindings_require_exact_names_and_finite_reals(bindings):
    with pytest.raises(InteropConversionError, match=r"qmod\.binding"):
        _synthesize(SOURCE, classical_inputs=bindings)


def test_bound_helper_checks_caller_array_bounds():
    source = SOURCE.read_text().replace("q[0]", "q[1]", 1)
    with pytest.raises(InteropConversionError, match="out of bounds"):
        _synthesize(
            source,
            source_format="qmod",
            classical_inputs={"input_0": 0.37, "weight_0": -0.81},
        )
