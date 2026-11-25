"""Unmodified tutorial sources execute at their original quantum dimensions."""

import hashlib
import json
import math
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum import synth
from tests.synthesis_workflow_helpers import check_state

pytestmark = pytest.mark.integration
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


@pytest.mark.parametrize(
    "filename", ["classiq_overview_tutorial.qmod", "execution_tutorial.qmod"]
)
@pytest.mark.parametrize("input_kind", ["file", "text"])
@pytest.mark.usefixtures("legacy_register_selection")
def test_original_tutorial_full_state_and_clean_workspace(
    filename: str, input_kind: str
) -> None:
    manifest = json.loads((CORPUS / "tutorial_sources.json").read_text())
    assert manifest["commit"] == "2bfd835e73b6882fcc515a725098b04766c238e7"
    assert manifest["repository"] == "https://github.com/Classiq/classiq-library"
    assert (
        hashlib.sha256((CORPUS / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    entry = next(item for item in manifest["fixtures"] if item["file"] == filename)
    raw = (CORPUS / filename).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == entry["git_blob_sha1"]
    )
    if input_kind == "file":
        p = synth.synthesize(
            CORPUS / filename, max_qubits=24, memory_limit_bytes=512 * 1024**2
        )
    else:
        p = synth.synthesize(
            raw.decode("utf-8"),
            source_format="qmod",
            source_name=entry["upstream_path"],
            max_qubits=24,
            memory_limit_bytes=512 * 1024**2,
        )
    assert dict(p.report["register_widths"]) == {"x": 3, "y": 6}
    assert p.report["resources"].n_qubits == 21
    assert p.execution.state_bytes == 32 * 1024**2
    result = fq.run(p)
    expected = {(x, x * x + 1): 1 / math.sqrt(8) for x in range(8)}
    check_state(p, result, expected)
    assert dict(
        p.probabilities(result, registers=("x", "y"), max_amplitudes=1 << 24)
    ) == pytest.approx(dict.fromkeys(expected, 1 / 8), abs=1e-12)
